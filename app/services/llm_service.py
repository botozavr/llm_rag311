import asyncio
import logging

from app.config import settings
from app.exceptions import (
    LLMServiceError,
    ModelNotLoadedError,
    GenerationError,
    AppValidationError,
)

logger = logging.getLogger("llm_api.llm_service")


class LLMService:
    """Загрузка и инференс LLM (локально или через API)"""

    def __init__(self):
        self.model      = None
        self.tokenizer  = None
        self._loaded    = False
        self._device    = None
        self._semaphore = asyncio.Semaphore(settings.max_concurrent_requests)

    # ================================================================
    # Загрузка
    # ================================================================

    async def load(self) -> None:
        if settings.inference_mode == "local":
            await self._load_local()
        else:
            logger.info("Режим API — локальная модель не загружается")
            self._loaded = True

    async def _load_local(self) -> None:
        logger.info("Загрузка модели: %s", settings.local_model_name)

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._device = "cuda" if torch.cuda.is_available() else "cpu"

            if self._device == "cuda":
                logger.info(
                    "GPU: %s | VRAM: %.1f GB",
                    torch.cuda.get_device_name(0),
                    torch.cuda.get_device_properties(0).total_memory / 1e9,
                )
            else:
                logger.warning("GPU не найден — используется CPU (медленно)")

            self._hf_login()

            load_kwargs = dict(
                pretrained_model_name_or_path=settings.local_model_name,
                trust_remote_code=True,
                **({"token": settings.hf_token} if settings.hf_token else {}),
            )

            # Шаг 1: токенизатор
            logger.info("Загрузка токенизатора...")
            self.tokenizer = AutoTokenizer.from_pretrained(**load_kwargs)

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Шаг 2: chat template (не меняет словарь — можно до модели)
            self._setup_chat_template()

            # Шаг 3: модель
            logger.info("Загрузка весов модели (может занять несколько минут)...")
            model_kwargs = self._build_model_kwargs(load_kwargs)
            self.model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
            self.model.eval()

            # Шаг 4: специальные токены ПОСЛЕ модели + resize embeddings
            self._setup_special_tokens()

            self._loaded = True
            self._log_memory()

        except LLMServiceError:
            raise
        except Exception as e:
            logger.exception("Ошибка загрузки модели")
            raise LLMServiceError(
                f"Не удалось загрузить '{settings.local_model_name}': {e}", 500
            )

    def _hf_login(self) -> None:
        """Авторизация в HuggingFace если задан токен"""
        if not settings.hf_token:
            return
        try:
            from huggingface_hub import login
            login(token=settings.hf_token, add_to_git_credential=False)
            logger.info("✅ HuggingFace авторизация успешна")
        except Exception as e:
            logger.warning("HuggingFace авторизация не удалась: %s", e)

    def _build_model_kwargs(self, base_kwargs: dict) -> dict:
        """Параметры загрузки: float16 или 4-bit квантизация"""
        import torch

        if settings.use_4bit and self._device == "cuda":
            logger.info("Режим загрузки: 4-bit квантизация")
            from transformers import BitsAndBytesConfig
            return {
                **base_kwargs,
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                ),
                "device_map": "auto",
            }

        dtype = torch.float16 if self._device == "cuda" else torch.float32
        logger.info(
            "Режим загрузки: %s",
            "float16" if dtype == torch.float16 else "float32",
        )
        return {
            **base_kwargs,
            "torch_dtype": dtype,
            "device_map":  "auto",
        }

    def _log_memory(self) -> None:
        """Логирование использования VRAM после загрузки"""
        try:
            import torch
            if torch.cuda.is_available():
                alloc = torch.cuda.memory_allocated() / 1e9
                rsrvd = torch.cuda.memory_reserved()  / 1e9
                logger.info(
                    "✅ Модель загружена | VRAM: %.1f GB alloc / %.1f GB reserved",
                    alloc, rsrvd,
                )
        except Exception:
            logger.info("✅ Модель загружена")

    # ================================================================
    # Настройка токенизатора
    # ================================================================

    def _setup_special_tokens(self) -> None:
        """
        Синхронизирует токенизатор с уже загруженной моделью.

        Для Vikhr токены <|im_start|> и <|im_end|> уже встроены
        в веса модели (vocab_size=40002), но могут быть не зарегистрированы
        в токенизаторе как special_tokens.

        Правило: НИКОГДА не делаем resize если модель уже знает эти токены.
        """
        model_name = settings.local_model_name.lower()

        if "vikhr" not in model_name:
            logger.info("Специальные токены не требуются для этой модели")
            return

        tokenizer_vocab_size = len(self.tokenizer)
        model_vocab_size = self.model.config.vocab_size

        logger.info(
            "Словарь токенизатора: %d | Словарь модели: %d",
            tokenizer_vocab_size, model_vocab_size,
        )

        # ✅ Словари уже совпадают — всё хорошо
        if tokenizer_vocab_size == model_vocab_size:
            logger.info(
                "✅ Словари синхронизированы (%d) — ничего делать не нужно",
                tokenizer_vocab_size,
            )
            return

        # Токенизатор меньше модели — нужно добавить токены в токенизатор
        # БЕЗ resize модели (она уже правильного размера)
        if tokenizer_vocab_size < model_vocab_size:
            logger.info(
                "Токенизатор (%d) < модель (%d) — добавляем токены в токенизатор",
                tokenizer_vocab_size, model_vocab_size,
            )

            existing = set(self.tokenizer.all_special_tokens)
            tokens_to_add = [
                t for t in ["<|im_start|>", "<|im_end|>"]
                if t not in existing
            ]

            if tokens_to_add:
                self.tokenizer.add_special_tokens(
                    {"additional_special_tokens": tokens_to_add}
                )
                logger.info(
                    "✅ Добавлены токены в токенизатор: %s (vocab: %d → %d)",
                    tokens_to_add,
                    tokenizer_vocab_size,
                    len(self.tokenizer),
                )

            # Проверяем результат
            final_tokenizer_size = len(self.tokenizer)
            if final_tokenizer_size != model_vocab_size:
                logger.warning(
                    "⚠️  После добавления токенов словари всё ещё не совпадают: "
                    "tokenizer=%d, model=%d",
                    final_tokenizer_size, model_vocab_size,
                )
            else:
                logger.info(
                    "✅ Словари синхронизированы: %d токенов",
                    final_tokenizer_size,
                )
            return

        # Токенизатор больше модели — нештатная ситуация
        if tokenizer_vocab_size > model_vocab_size:
            logger.warning(
                "⚠️  Токенизатор (%d) > модель (%d) — нестандартная ситуация. "
                "Попробуем resize модели.",
                tokenizer_vocab_size, model_vocab_size,
            )
            try:
                self.model.resize_token_embeddings(
                    tokenizer_vocab_size,
                    mean_resizing=False,
                )
                logger.info(
                    "✅ Модель расширена до %d токенов",
                    tokenizer_vocab_size,
                )
            except Exception as e:
                logger.error("❌ resize_token_embeddings не удался: %s", e)

    def _setup_chat_template(self) -> None:
        """
        Устанавливает chat template если его нет.
        Определяет нужный формат автоматически по имени модели.
        """
        if self.tokenizer.chat_template is not None:
            logger.info("✅ Chat template уже есть в токенизаторе")
            self._verify_chat_template()
            return

        model_name = settings.local_model_name.lower()
        logger.info("Chat template не найден — определяем по имени модели...")

        if "vikhr" in model_name:
            self._set_chatml_template()
        elif "saiga" in model_name and "llama3" in model_name:
            self._set_saiga_llama3_template()
        elif "saiga" in model_name:
            self._set_saiga_mistral_template()
        elif "mistral" in model_name or "mixtral" in model_name:
            self._set_chatml_template()
        else:
            logger.warning(
                "Неизвестная модель '%s' — используем ChatML как дефолт",
                settings.local_model_name,
            )
            self._set_chatml_template()

    def _set_chatml_template(self) -> None:
        """
        ChatML формат — для Vikhr, Mistral-Instruct и других.

        Пример:
            <|im_start|>system
            Ты помощник.<|im_end|>
            <|im_start|>user
            Привет!<|im_end|>
            <|im_start|>assistant
        """
        logger.info("Устанавливаем ChatML template...")
        self.tokenizer.chat_template = (
            "{% for message in messages %}"
                "<|im_start|>{{ message['role'] }}\n"
                "{{ message['content'] }}<|im_end|>\n"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
                "<|im_start|>assistant\n"
            "{% endif %}"
        )
        self._verify_chat_template()

    def _set_saiga_llama3_template(self) -> None:
        """
        Saiga Llama3 формат.

        Пример:
            <|begin_of_text|><|start_header_id|>system<|end_header_id|>
            Ты помощник.<|eot_id|>
            <|start_header_id|>user<|end_header_id|>
            Привет!<|eot_id|>
            <|start_header_id|>assistant<|end_header_id|>
        """
        logger.info("Устанавливаем Saiga Llama3 template...")
        self.tokenizer.chat_template = (
            "{% if messages[0]['role'] == 'system' %}"
                "{{ '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n' }}"
                "{{ messages[0]['content'] }}"
                "{{ '<|eot_id|>' }}"
                "{% set messages = messages[1:] %}"
            "{% else %}"
                "{{ '<|begin_of_text|>' }}"
            "{% endif %}"
            "{% for message in messages %}"
                "{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' }}"
                "{{ message['content'] }}"
                "{{ '<|eot_id|>' }}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
                "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
            "{% endif %}"
        )
        self._verify_chat_template()

    def _set_saiga_mistral_template(self) -> None:
        """
        Saiga Mistral формат.

        Пример:
            <s>system
            Ты помощник.</s>
            <s>user
            Привет!</s>
            <s>bot
        """
        logger.info("Устанавливаем Saiga Mistral template...")
        self.tokenizer.chat_template = (
            "{% for message in messages %}"
                "{% if message['role'] == 'system' %}"
                    "<s>system\n{{ message['content'] }}</s>\n"
                "{% elif message['role'] == 'user' %}"
                    "<s>user\n{{ message['content'] }}</s>\n"
                "{% elif message['role'] == 'assistant' %}"
                    "<s>bot\n{{ message['content'] }}</s>\n"
                "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}<s>bot\n{% endif %}"
        )
        self._verify_chat_template()

    def _verify_chat_template(self) -> None:
        """Проверка что установленный template корректно работает"""
        try:
            test = self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": "Ты помощник."},
                    {"role": "user",   "content": "Привет!"},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            logger.info("✅ Chat template работает")
            logger.debug("Пример промпта:\n%s", test)
        except Exception as e:
            logger.warning("❌ Тест chat template не прошёл: %s", e)

    def _format_fallback(self, messages: list[dict]) -> str:
        """
        Ручное форматирование если apply_chat_template недоступен.
        Использует ChatML как универсальный формат.
        """
        parts = []
        for m in messages:
            role    = m["role"]
            content = m["content"]
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
        parts.append("<|im_start|>assistant\n")
        return "".join(parts)

    # ================================================================
    # Публичный интерфейс генерации
    # ================================================================

    async def predict(
        self,
        messages:    list[dict],
        max_tokens:  int   = 200,
        temperature: float = 0.7,
    ) -> tuple[str, int]:
        """
        Генерация для чата.
        Параметры настроены для живого диалога:
        top_p=0.9, top_k=50 — более свободная генерация.
        """
        if not self._loaded:
            raise ModelNotLoadedError()

        async with self._semaphore:
            try:
                if settings.inference_mode == "local":
                    return await self._predict_local(messages, max_tokens, temperature)
                else:
                    return await self._predict_api(messages, max_tokens, temperature)
            except (LLMServiceError, ModelNotLoadedError):
                raise
            except Exception as e:
                logger.exception("Ошибка генерации")
                raise GenerationError(str(e))

    async def predict_rag(
        self,
        messages:    list[dict],
        max_tokens:  int   = 300,
        temperature: float = 0.1,
    ) -> tuple[str, int]:
        """
        Генерация для RAG.
        Параметры настроены для точного следования контексту:
        top_p=0.8, top_k=30 — строже, меньше галлюцинаций.
        """
        if not self._loaded:
            raise ModelNotLoadedError()

        async with self._semaphore:
            try:
                if settings.inference_mode == "local":
                    return await self._predict_local_rag(messages, max_tokens, temperature)
                else:
                    return await self._predict_api(messages, max_tokens, temperature)
            except (LLMServiceError, ModelNotLoadedError):
                raise
            except Exception as e:
                logger.exception("Ошибка RAG-генерации")
                raise GenerationError(str(e))

    # ================================================================
    # Локальная генерация
    # ================================================================

    async def _predict_local(
        self,
        messages:    list[dict],
        max_tokens:  int,
        temperature: float,
    ) -> tuple[str, int]:
        """Генерация для чата — свободные параметры"""
        do_sample = temperature > 0.05

        gen_kwargs = dict(
            max_new_tokens=max_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            repetition_penalty=1.1,
            no_repeat_ngram_size=3,
            **(
                dict(
                    do_sample=True,
                    temperature=temperature,
                    top_p=0.9,
                    top_k=50,
                )
                if do_sample else
                dict(do_sample=False)
            ),
        )

        prompt       = self._build_prompt(messages)
        inputs, ilen = self._tokenize(prompt)

        return await asyncio.to_thread(
            self._generate, inputs, ilen, gen_kwargs, "Chat"
        )

    async def _predict_local_rag(
            self,
            messages: list[dict],
            max_tokens: int,
            temperature: float,
    ) -> tuple[str, int]:
        """Генерация для RAG — строгие параметры"""

        # Жёстко ограничиваем токены для RAG
        # Длинные ответы = галлюцинации + долго
        max_tokens = min(max_tokens, 200)

        do_sample = temperature > 0.05

        gen_kwargs = dict(
            max_new_tokens=max_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            repetition_penalty=1.15,
            no_repeat_ngram_size=4,
            **(
                dict(
                    do_sample=True,
                    temperature=temperature,
                    top_p=0.8,
                    top_k=30,
                )
                if do_sample else
                dict(do_sample=False)
            ),
        )

        prompt = self._build_prompt(messages)
        inputs, ilen = self._tokenize(prompt)

        return await asyncio.to_thread(
            self._generate, inputs, ilen, gen_kwargs, "RAG"
        )

    # ================================================================
    # Вспомогательные методы
    # ================================================================

    def _build_prompt(self, messages: list[dict]) -> str:
        """Форматирование промпта через chat template"""
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception as e:
            logger.warning("apply_chat_template не сработал: %s — используем fallback", e)
            prompt = self._format_fallback(messages)

        logger.debug("Промпт (%d симв.): %s...", len(prompt), prompt[:150])
        return prompt

    def _tokenize(self, prompt: str) -> tuple[dict, int]:
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        )

        input_length = inputs["input_ids"].shape[1]
        vocab_size = self.model.config.vocab_size
        max_token_id = inputs["input_ids"].max().item()

        logger.debug(
            "Токенов: %d | max_token_id: %d | vocab_size: %d",
            input_length, max_token_id, vocab_size,
        )

        if max_token_id >= vocab_size:
            logger.error(
                "❌ Токен ID %d >= vocab_size %d!",
                max_token_id, vocab_size,
            )
            raise LLMServiceError(
                f"Токен ID {max_token_id} выходит за пределы словаря ({vocab_size}). "
                f"Перезапустите сервер.",
                500,
            )

        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        return inputs, input_length

    def _generate(
        self,
        inputs:       dict,
        input_length: int,
        gen_kwargs:   dict,
        label:        str = "Gen",
    ) -> tuple[str, int]:
        """
        Синхронная генерация.
        Запускается через asyncio.to_thread чтобы не блокировать event loop.
        Возвращает (текст_ответа, количество_новых_токенов).
        """
        import time
        import torch

        t0 = time.monotonic()

        with torch.no_grad():
            output = self.model.generate(**inputs, **gen_kwargs)

        elapsed    = time.monotonic() - t0
        new_tokens = output[0][input_length:]
        n          = len(new_tokens)

        logger.info(
            "[%s] Сгенерировано %d токенов за %.1f сек (%.1f tok/s)",
            label, n, elapsed, n / elapsed if elapsed > 0 else 0,
        )

        text = self.tokenizer.decode(
            new_tokens,
            skip_special_tokens=True,
        ).strip()

        # Освобождаем память GPU
        del output
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return text or "Модель не сгенерировала ответ", n

    # ================================================================
    # API режим
    # ================================================================

    async def _predict_api(
        self,
        messages:    list[dict],
        max_tokens:  int,
        temperature: float,
    ) -> tuple[str, int]:
        """Инференс через Hugging Face Inference API"""
        import aiohttp

        if not settings.hf_api_token:
            raise AppValidationError(
                "HF_API_TOKEN не задан. "
                "Добавьте в .env или используйте INFERENCE_MODE=local"
            )

        prompt  = messages[-1]["content"] if messages else ""
        headers = {
            "Authorization": f"Bearer {settings.hf_api_token}",
            "Content-Type":  "application/json",
        }
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens":  max_tokens,
                "temperature":     max(temperature, 0.01),
                "do_sample":       temperature > 0,
                "return_full_text": False,
            },
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    settings.hf_api_url,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        text = (
                            result[0].get("generated_text", "")
                            if isinstance(result, list) else
                            result.get("generated_text", "")
                        )
                        return text, len(text.split())

                    elif resp.status == 503:
                        raise LLMServiceError(
                            "API: модель загружается на HF серверах. Повторите через минуту.",
                            503,
                        )
                    else:
                        raise LLMServiceError(
                            f"API вернул ошибку {resp.status}",
                            resp.status,
                        )

        except aiohttp.ClientError as e:
            raise LLMServiceError(f"Ошибка подключения к API: {e}", 502)
        except asyncio.TimeoutError:
            raise LLMServiceError("Таймаут запроса к API (60 сек)", 504)

    # ================================================================
    # Статус
    # ================================================================

    def is_loaded(self) -> bool:
        return self._loaded

    def get_info(self) -> dict:
        info = {
            "mode":                    settings.inference_mode,
            "loaded":                  self._loaded,
            "model_name":              settings.local_model_name,
            "max_concurrent_requests": settings.max_concurrent_requests,
        }

        if settings.inference_mode == "local" and self._loaded:
            info["device"] = str(self._device)
            try:
                import torch
                if torch.cuda.is_available():
                    info["gpu_name"] = torch.cuda.get_device_name(0)
                    info["gpu_memory_allocated_gb"] = round(
                        torch.cuda.memory_allocated() / 1e9, 2
                    )
                    info["gpu_memory_reserved_gb"] = round(
                        torch.cuda.memory_reserved() / 1e9, 2
                    )
            except Exception:
                pass

        return info