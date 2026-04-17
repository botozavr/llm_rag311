from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator


# ==================== Chat ====================

class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"] = Field(
        ..., description="Роль отправителя"
    )
    content: str = Field(
        ..., min_length=1, max_length=10000, description="Текст сообщения"
    )

    @field_validator("content")
    def not_empty(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("Сообщение не может быть пустым")
        return stripped


class ChatRequest(BaseModel):
    messages:    list[ChatMessage] = Field(..., min_length=1, max_length=50)
    temperature: float             = Field(default=0.7,  ge=0.0, le=2.0)
    max_tokens:  int               = Field(default=200,  ge=1,   le=1024)

    @field_validator("messages")
    def last_message_from_user(cls, v: list[ChatMessage]) -> list[ChatMessage]:
        if v and v[-1].role != "user":
            raise ValueError("Последнее сообщение должно быть от пользователя")
        return v


class ChatResponse(BaseModel):
    request_id:   str
    text:         str
    tokens_used:  Optional[int]                         = None
    finish_reason: Literal["stop", "length", "error"]  = "stop"
    latency_ms:   float


# ==================== RAG ====================

class RagRequest(BaseModel):
    query:           str   = Field(..., min_length=1, max_length=5000)
    top_k:           int   = Field(default=3,   ge=1,  le=20)    # было 5
    temperature:     float = Field(default=0.1, ge=0.0, le=2.0)  # было 0.3
    max_tokens:      int   = Field(default=150, ge=1,  le=512)   # было 512
    include_sources: bool  = Field(default=True)

    @field_validator("query")
    def validate_query(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("Запрос не может быть пустым")
        if len(stripped) < 2:
            raise ValueError("Запрос слишком короткий (минимум 2 символа)")
        return stripped


class RagSource(BaseModel):
    text:   str
    source: str
    score:  float
    page:   Optional[int] = None


class RagResponse(BaseModel):
    request_id:   str
    answer:       str
    sources:      list[RagSource] = Field(default_factory=list)
    sources_used: int
    latency_ms:   float


# ==================== Общие ====================

class ErrorResponse(BaseModel):
    error:      str
    detail:     Optional[str] = None
    request_id: Optional[str] = None
    status_code: int