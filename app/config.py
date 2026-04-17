from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    """
    Все настройки приложения.
    Читаются из .env файла или переменных окружения.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM ─────────────────────────────────────────────────────
    inference_mode: str = Field(
        default="local",
        description="local или api",
    )
    local_model_name: str = Field(
        default="IlyaGusev/saiga_mistral_7b_merged",
        description="Название модели на HuggingFace",
    )
    max_concurrent_requests: int = Field(
        default=1,
        description="Макс. одновременных запросов к LLM",
    )
    use_4bit: bool = Field(
        default=False,
        description="Загрузить модель в 4-bit (экономия VRAM)",
    )

    # ── HuggingFace ──────────────────────────────────────────────
    hf_token: str = Field(
        default="",
        description="HuggingFace токен для приватных моделей",
    )
    hf_api_token: str = Field(
        default="",
        description="HF Inference API токен",
    )
    hf_api_url: str = Field(
        default="https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium",
    )

    # ── RAG ──────────────────────────────────────────────────────
    rag_enabled: bool = Field(
        default=True,
        description="Включить RAG сервис",
    )
    chroma_db_path: str = Field(
        default="./chroma_db",
        description="Путь к векторной базе",
    )
    documents_path: str = Field(
        default="./documents",
        description="Путь к документам для индексации",
    )
    embedding_model: str = Field(
        default="intfloat/multilingual-e5-base",
        description="Модель для эмбеддингов",
    )
    collection_name: str = Field(
        default="documents",
    )
    chunk_size: int     = Field(default=1000)
    chunk_overlap: int  = Field(default=200)
    max_context_length: int = Field(
        default=2000,
        description="Макс. символов контекста для LLM",
    )

    # ── Логирование ──────────────────────────────────────────────
    log_dir: str   = Field(default="./logs")
    log_level: str = Field(default="INFO")

    # ── Приложение ───────────────────────────────────────────────
    debug: bool = Field(default=False)
    host: str   = Field(default="0.0.0.0")
    port: int   = Field(default=8000)


# Глобальный объект настроек — импортировать отовсюду
settings = Settings()