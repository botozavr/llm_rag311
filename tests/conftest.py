"""
Фикстуры для тестов.
Подменяем реальные сервисы моками через app.state ДО старта lifespan.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient


def make_llm_mock() -> MagicMock:
    """Заглушка LLM сервиса"""
    mock = MagicMock()

    # Синхронные методы
    mock.is_loaded.return_value = True
    mock.get_info.return_value  = {
        "mode":       "local",
        "loaded":     True,
        "model_name": "test-model",
        "device":     "cpu",
    }

    # Асинхронные методы
    mock.predict     = AsyncMock(return_value=("Тестовый ответ от LLM", 10))
    mock.predict_rag = AsyncMock(return_value=("Тестовый RAG ответ", 15))

    return mock


def make_rag_mock() -> MagicMock:
    """Заглушка RAG сервиса"""
    mock = MagicMock()

    # Синхронные методы
    mock.is_loaded.return_value = True
    mock.get_stats.return_value = {
        "status":          "loaded",
        "document_count":  100,
        "collection_name": "documents",
        "embedding_model": "test-embeddings",
        "chroma_path":     "./chroma_db",
    }
    mock.search.return_value = [
        {
            "content":  "Для получения аккредитации необходимо подать заявку через Госуслуги.",
            "metadata": {"source": "documents/test.pdf", "page": 1},
            "score":    0.92,
        },
        {
            "content":  "Срок рассмотрения заявки — 15 рабочих дней.",
            "metadata": {"source": "documents/test.pdf", "page": 2},
            "score":    0.85,
        },
    ]
    mock.format_context.return_value = (
        "[Источник 1 | test.pdf | релевантность: 0.92]\n"
        "Для получения аккредитации необходимо подать заявку через Госуслуги."
    )
    mock.build_rag_prompt.return_value = [
        {"role": "system", "content": "Ты помощник."},
        {"role": "user",   "content": "Как получить аккредитацию?"},
    ]

    return mock


@pytest.fixture(scope="session")
def client():
    """
    HTTP клиент для тестов.

    Патчим lifespan чтобы он не грузил реальные модели,
    а сразу устанавливал моки в app.state.
    """
    from contextlib import asynccontextmanager
    from fastapi import FastAPI
    import main as main_module

    llm = make_llm_mock()
    rag = make_rag_mock()

    # Подменяем lifespan — он не будет грузить модели
    @asynccontextmanager
    async def mock_lifespan(app: FastAPI):
        app.state.llm_service = llm
        app.state.rag_service = rag
        yield

    # Патчим lifespan в модуле main
    with patch.object(main_module.app.router, "lifespan_context", mock_lifespan):
        with TestClient(main_module.app, raise_server_exceptions=False) as c:
            # Устанавливаем state напрямую (на случай если patch не сработал)
            c.app.state.llm_service = llm
            c.app.state.rag_service = rag
            yield c


@pytest.fixture
def llm_mock(client) -> MagicMock:
    """Прямой доступ к моку LLM"""
    return client.app.state.llm_service


@pytest.fixture
def rag_mock(client) -> MagicMock:
    """Прямой доступ к моку RAG"""
    return client.app.state.rag_service