import uuid
import time
import asyncio
import logging
from fastapi import APIRouter, Request
from app.schemas import RagRequest, RagResponse, RagSource
from app.exceptions import RAGNotAvailableError, LLMServiceError, AppValidationError
from app.services.llm_service import LLMService
from app.services.rag_service import RAGService

logger = logging.getLogger("llm_api.routers.rag")
router = APIRouter(prefix="/rag", tags=["RAG"])


def _get_rag(req: Request) -> RAGService:
    rag: RAGService = req.app.state.rag_service
    if rag is None:
        raise RAGNotAvailableError("RAG отключён (RAG_ENABLED=false)")
    if not rag.is_loaded():
        raise RAGNotAvailableError("RAG не инициализирован — запустите index_documents.py")
    return rag


@router.post("", response_model=RagResponse)
async def rag_endpoint(request: RagRequest, req: Request):
    """Поиск по документам + генерация ответа"""
    start_time = time.monotonic()
    rid        = getattr(req.state, "request_id", str(uuid.uuid4())[:8])
    client_ip  = getattr(req.state, "client_ip",  "unknown")

    logger.info(
        "[%s] 🔍 RAG от %s | top_k=%d | '%s...'",
        rid, client_ip, request.top_k, request.query[:60],
    )

    rag: RAGService = _get_rag(req)
    llm: LLMService = req.app.state.llm_service

    try:
        # 1. Поиск
        docs = await asyncio.to_thread(rag.search, request.query, request.top_k)
        logger.info("[%s] Найдено чанков: %d", rid, len(docs))

        if not docs:
            return RagResponse(
                request_id=rid,
                answer="Не найдено релевантных документов. Попробуйте переформулировать запрос.",
                sources=[],
                sources_used=0,
                latency_ms=round((time.monotonic() - start_time) * 1000, 2),
            )

        # 2. Контекст + промпт
        context  = rag.format_context(docs)
        messages = rag.build_rag_prompt(request.query, context)

        # 3. Генерация
        answer, tokens_used = await llm.predict_rag(
            messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )

        # 4. Источники
        sources = [
            RagSource(
                text=d["content"][:500],
                source=d["metadata"].get("source", "unknown"),
                score=d["score"],
                page=d["metadata"].get("page"),
            )
            for d in docs
        ] if request.include_sources else []

        latency_ms = (time.monotonic() - start_time) * 1000
        logger.info(
            "[%s] ✅ RAG | %d источников | %d токенов | %.0f ms",
            rid, len(sources), tokens_used, latency_ms,
        )

        return RagResponse(
            request_id=rid,
            answer=answer,
            sources=sources,
            sources_used=len(docs),
            latency_ms=round(latency_ms, 2),
        )

    except (LLMServiceError, RAGNotAvailableError):
        raise
    except Exception as e:
        logger.exception("[%s] Ошибка RAG", rid)
        raise LLMServiceError(f"Ошибка RAG: {e}", 500)


@router.get("/search", summary="Только поиск (без генерации)")
async def rag_search(query: str, top_k: int = 5, req: Request = None):
    """Поиск релевантных фрагментов без генерации — для отладки"""
    rag = _get_rag(req)

    # ✅ Валидация
    query = query.strip()
    if not query:
        raise AppValidationError("Параметр 'query' не может быть пустым")
    if len(query) < 2:
        raise AppValidationError("Запрос слишком короткий (минимум 2 символа)")

    results = await asyncio.to_thread(rag.search, query, top_k)
    rid     = getattr(req.state, "request_id", "?") if req else "?"
    logger.info("[%s] /rag/search '%s...' → %d результатов", rid, query[:30], len(results))

    return {"query": query, "results": results, "count": len(results)}


@router.get("/stats", summary="Статистика RAG базы")
async def rag_stats(req: Request):
    rag: RAGService = req.app.state.rag_service
    if rag is None:
        return {"status": "disabled"}
    return rag.get_stats()