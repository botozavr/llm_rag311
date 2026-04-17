import time
import logging
from fastapi import APIRouter, HTTPException, Request
from app.config import settings
from app.services.llm_service import LLMService
from app.services.rag_service import RAGService

logger = logging.getLogger("llm_api.routers.system")
router = APIRouter(tags=["System"])


@router.get("/")
async def root():
    return {
        "service":     "LLM Inference API with RAG",
        "version":     "3.0.0",
        "model":       settings.local_model_name,
        "mode":        settings.inference_mode,
        "rag_enabled": settings.rag_enabled,
        "docs":        "/docs",
    }


@router.get("/health")
async def health():
    return {"status": "ok", "timestamp": time.time()}


@router.get("/ready")
async def ready(req: Request):
    llm: LLMService = req.app.state.llm_service
    rag: RAGService = req.app.state.rag_service

    if not llm.is_loaded():
        raise HTTPException(status_code=503, detail="LLM не загружена")

    return {
        "status": "ready",
        "llm":    llm.get_info(),
        "rag":    rag.get_stats() if rag else {"status": "disabled"},
    }


@router.get("/stats")
async def stats():
    """Статистика GPU"""
    try:
        import torch
        if not torch.cuda.is_available():
            return {"gpu_available": False}
        props = torch.cuda.get_device_properties(0)
        return {
            "gpu_available": True,
            "gpu_name":      torch.cuda.get_device_name(0),
            "total_gb":      round(props.total_memory / 1e9, 2),
            "allocated_gb":  round(torch.cuda.memory_allocated() / 1e9, 2),
            "reserved_gb":   round(torch.cuda.memory_reserved()  / 1e9, 2),
            "free_gb":       round((props.total_memory - torch.cuda.memory_reserved()) / 1e9, 2),
        }
    except Exception as e:
        return {"error": str(e)}