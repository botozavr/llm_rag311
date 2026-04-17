import re
import asyncio
import logging
from pathlib import Path
from typing import Optional

from app.config import settings
from app.exceptions import RAGNotAvailableError

logger = logging.getLogger("llm_api.rag_service")


class RAGService:
    """Векторная база + поиск + формирование промпта"""

    def __init__(self):
        self.vectorstore = None
        self.embeddings  = None
        self._loaded     = False

    # ── Загрузка ────────────────────────────────────────────────

    async def load(self) -> None:
        logger.info("Инициализация RAG сервиса...")

        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError:
            from langchain_community.embeddings import HuggingFaceEmbeddings

        from langchain_community.vectorstores import Chroma

        def _load_embeddings():
            return HuggingFaceEmbeddings(
                model_name=settings.embedding_model,
                model_kwargs={"device": "cuda"},
                encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
            )

        logger.info("Загрузка модели эмбеддингов: %s", settings.embedding_model)
        self.embeddings = await asyncio.to_thread(_load_embeddings)

        chroma_path = Path(settings.chroma_db_path)

        if chroma_path.exists() and any(chroma_path.iterdir()):
            logger.info("Загрузка базы: %s", settings.chroma_db_path)

            def _load_vs():
                return Chroma(
                    persist_directory=settings.chroma_db_path,
                    embedding_function=self.embeddings,
                    collection_name=settings.collection_name,
                )

            self.vectorstore = await asyncio.to_thread(_load_vs)
            count = self.vectorstore._collection.count()
            logger.info("Документов в базе: %d", count)
        else:
            logger.warning(
                "База не найдена в %s. Запустите: python scripts/index_documents.py",
                settings.chroma_db_path,
            )
            chroma_path.mkdir(parents=True, exist_ok=True)

            def _create_empty():
                return Chroma(
                    persist_directory=settings.chroma_db_path,
                    embedding_function=self.embeddings,
                    collection_name=settings.collection_name,
                )

            self.vectorstore = await asyncio.to_thread(_create_empty)

        self._loaded = True
        logger.info("✅ RAG сервис готов")

    # ── Поиск ───────────────────────────────────────────────────

    def search(
        self,
        query:           str,
        top_k:           int            = 5,
        score_threshold: Optional[float] = None,
    ) -> list[dict]:
        if not self._loaded or self.vectorstore is None:
            raise RAGNotAvailableError("RAG сервис не инициализирован")

        results = self.vectorstore.similarity_search_with_relevance_scores(
            query, k=top_k,
        )

        return [
            {
                "content":  doc.page_content,
                "metadata": doc.metadata,
                "score":    round(score, 4),
            }
            for doc, score in results
            if score_threshold is None or score >= score_threshold
        ]

    # ── Форматирование ──────────────────────────────────────────

    @staticmethod
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Глубокая очистка текста от артефактов PDF/DOCX.
        """


        # Убираем PDF-мусор типа /g2B1 /g19E /g29C
        text = re.sub(r'/g[0-9A-Fa-f]{2,4}\b', '', text)

        # Убираем разбитые буквы из PDF: "а к к р е д и т а ц и я"
        # Паттерн: одна буква + пробел повторяется 3+ раза подряд
        text = re.sub(r'\b([А-ЯЁа-яёA-Za-z] ){3,}[А-ЯЁа-яёA-Za-z]\b', '', text)

        # Убираем слипшиеся слова с заглавной буквы посередине
        # типа "акцииции" — нет простого способа, но можно убрать повторы букв
        text = re.sub(r'([а-яёА-ЯЁ])\1{2,}', r'\1', text)  # ааа → а

        # Убираем строки только из спецсимволов (✓ ✅ 🎯 и т.п.)
        text = re.sub(r'^[\s\U0001F300-\U0001FFFF✓✅🎯📋✨]+$', '', text, flags=re.MULTILINE)

        # Нормализуем пробелы и переносы
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ \t]{2,}', ' ', text)

        # Убираем пустые строки в начале/конце
        lines = [line.strip() for line in text.splitlines()]
        lines = [line for line in lines if line]  # убираем пустые
        text = '\n'.join(lines)

        return text.strip()

    def format_context(
        self,
        documents:  list[dict],
        max_length: int = None,
    ) -> str:
        max_length = max_length or settings.max_context_length

        if not documents:
            return ""

        parts        = []
        total_length = 0

        for i, doc in enumerate(documents, 1):
            source  = Path(doc["metadata"].get("source", "?")).name
            content = self.clean_text(doc["content"])
            score   = doc["score"]

            part = (
                f"[Источник {i} | {source} | релевантность: {score:.2f}]\n"
                f"{content}\n"
            )

            if total_length + len(part) > max_length:
                remaining = max_length - total_length
                if remaining > 200:
                    truncated = part[:remaining]
                    last_dot  = truncated.rfind(".")
                    part = (truncated[:last_dot + 1] if last_dot > 100 else truncated)
                    part += "\n[...обрезано...]"
                    parts.append(part)
                break

            parts.append(part)
            total_length += len(part)

        return "\n---\n".join(parts)

    def build_rag_prompt(
            self,
            query: str,
            context: str,
            system_prompt: Optional[str] = None,
    ) -> list[dict]:
        """
        RAG промпт с жёсткими ограничениями формата.
        Запрещаем markdown, заголовки, длинные списки.
        """

        system_prompt = (
            "Ты помощник. Отвечай КРАТКО и ТОЛЬКО по тексту документов. "
            "Максимум 3-5 предложений. "
            "Не используй заголовки и markdown. "
            "Не придумывай URLs и детали которых нет в тексте. "
            "Если информации нет — скажи об этом."
        )

        # Формируем пронумерованные цитаты
        citations = self._build_citations(context)

        user_message = (
            f"Документы:\n"
            f"{citations}\n\n"
            f"Вопрос: {query}\n\n"
            f"Краткий ответ (2-4 предложения) строго по документам:"
        )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

    def _build_citations(self, context: str) -> str:
        """
        Преобразует контекст в короткие пронумерованные цитаты.
        Каждая цитата — не более 300 символов.
        """
        parts = context.split("\n---\n")

        citations = []
        for i, part in enumerate(parts, 1):
            lines = part.strip().split("\n")
            # Пропускаем первую строку (заголовок источника)
            text_lines = [l.strip() for l in lines[1:] if l.strip()]
            text = " ".join(text_lines)

            # Обрезаем до 300 символов по последней точке
            if len(text) > 300:
                truncated = text[:300]
                last_dot = truncated.rfind(".")
                text = truncated[:last_dot + 1] if last_dot > 50 else truncated

            if text:
                citations.append(f"[{i}] {text}")

        return "\n".join(citations)

    # ── Статус ──────────────────────────────────────────────────

    def is_loaded(self) -> bool:
        return self._loaded

    def get_stats(self) -> dict:
        if not self._loaded or self.vectorstore is None:
            return {"status": "not_loaded"}
        try:
            return {
                "status":             "loaded",
                "collection_name":    settings.collection_name,
                "document_count":     self.vectorstore._collection.count(),
                "embedding_model":    settings.embedding_model,
                "chroma_path":        settings.chroma_db_path,
                "max_context_length": settings.max_context_length,
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}