import uuid
import time
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from app.logger import setup_logging
from app.config import settings
from app.schemas import ErrorResponse
from app.exceptions import LLMServiceError
from app.services.llm_service import LLMService
from app.services.rag_service import RAGService
from app.routers import chat, rag, system

logger = setup_logging()


# ==================== Lifespan ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 60)
    logger.info("🚀 Запуск LLM Inference API v3.0")
    logger.info("   Модель : %s", settings.local_model_name)
    logger.info("   Режим  : %s", settings.inference_mode.upper())
    logger.info("   RAG    : %s", "включён" if settings.rag_enabled else "выключен")
    logger.info("=" * 60)

    try:
        # LLM
        llm = LLMService()
        await llm.load()
        app.state.llm_service = llm

        # RAG
        if settings.rag_enabled:
            try:
                rag_svc = RAGService()
                await rag_svc.load()
                app.state.rag_service = rag_svc
            except Exception as e:
                logger.error("❌ RAG не загружен: %s", e)
                app.state.rag_service = None
        else:
            app.state.rag_service = None

        logger.info("=" * 60)
        logger.info("✅ Сервер готов → http://%s:%d", settings.host, settings.port)
        logger.info("   Документация → http://%s:%d/docs", settings.host, settings.port)
        logger.info("=" * 60)

        yield

    finally:
        logger.info("👋 Завершение работы...")
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("🧹 GPU память очищена")
        except Exception:
            pass


# ==================== Middleware ====================

class RequestLoggingMiddleware:

    def __init__(self, app):
        self.app     = app
        self._logger = logging.getLogger("llm_api.requests")

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request    = Request(scope, receive, send)
        request_id = str(uuid.uuid4())[:8]

        client_ip  = request.client.host if request.client else "unknown"
        forwarded  = request.headers.get("X-Forwarded-For")
        if forwarded:
            client_ip = forwarded.split(",")[0].strip()

        scope.setdefault("state", {})
        scope["state"]["request_id"] = request_id
        scope["state"]["client_ip"]  = client_ip

        self._logger.info(
            "[%s] ➜  %s %s  (от %s)",
            request_id, request.method, request.url.path, client_ip,
        )

        start_time      = time.monotonic()
        response_status = [200]

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                response_status[0] = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            ms   = (time.monotonic() - start_time) * 1000
            code = response_status[0]
            fn   = (
                self._logger.error   if code >= 500 else
                self._logger.warning if code >= 400 else
                self._logger.info
            )
            fn("[%s] ←  %s %s  %d  (%.0f ms)", request_id, request.method, request.url.path, code, ms)


# ==================== App ====================

app = FastAPI(
    title="LLM Inference API with RAG",
    description="RAG + LLM на русском языке",
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(RequestLoggingMiddleware)
app.include_router(system.router)
app.include_router(chat.router)
app.include_router(rag.router)


# ==================== Обработчики ошибок ====================

@app.exception_handler(LLMServiceError)
async def llm_error_handler(request: Request, exc: LLMServiceError):
    rid = getattr(request.state, "request_id", "?")
    logger.error("[%s] LLMServiceError %d: %s", rid, exc.status_code, exc.message)
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.message, request_id=rid, status_code=exc.status_code,
        ).model_dump(),
    )

@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    rid    = getattr(request.state, "request_id", "?")
    errors = [f"{'→'.join(str(loc) for loc in e['loc'])}: {e['msg']}" for e in exc.errors()]
    detail = "; ".join(errors)
    logger.warning("[%s] Ошибка валидации: %s", rid, detail)
    return JSONResponse(
        status_code=400,
        content=ErrorResponse(
            error="Ошибка валидации", detail=detail, request_id=rid, status_code=400,
        ).model_dump(),
    )

@app.exception_handler(Exception)
async def general_error_handler(request: Request, exc: Exception):
    rid = getattr(request.state, "request_id", "?")
    logger.exception("[%s] Необработанное исключение", rid)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Внутренняя ошибка сервера",
            detail=str(exc) if settings.debug else None,
            request_id=rid,
            status_code=500,
        ).model_dump(),
    )


# ==================== Запуск ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        workers=1,
        log_level="warning",
        access_log=False,
    )