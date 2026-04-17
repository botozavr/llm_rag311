import uuid
import time
import logging
from fastapi import APIRouter, Request
from app.schemas import ChatRequest, ChatResponse
from app.services.llm_service import LLMService

logger = logging.getLogger("llm_api.routers.chat")
router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post("", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, req: Request):
    """Чат с LLM"""
    start_time = time.monotonic()
    rid        = getattr(req.state, "request_id", str(uuid.uuid4())[:8])
    client_ip  = getattr(req.state, "client_ip",  "unknown")

    logger.info(
        "[%s] 💬 Chat от %s | %d сообщ. | temp=%.2f | max_tokens=%d",
        rid, client_ip, len(request.messages), request.temperature, request.max_tokens,
    )
    logger.debug("[%s] Вопрос: %s", rid, request.messages[-1].content[:100])

    llm: LLMService = req.app.state.llm_service
    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    response_text, tokens_used = await llm.predict(
        messages,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
    )

    finish_reason = "length" if tokens_used >= request.max_tokens else "stop"
    latency_ms    = (time.monotonic() - start_time) * 1000

    logger.info(
        "[%s] ✅ Chat | %d токенов | %.0f ms | %s",
        rid, tokens_used, latency_ms, finish_reason,
    )

    return ChatResponse(
        request_id=rid,
        text=response_text,
        tokens_used=tokens_used,
        finish_reason=finish_reason,
        latency_ms=round(latency_ms, 2),
    )