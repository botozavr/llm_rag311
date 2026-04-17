class LLMServiceError(Exception):
    """Базовая ошибка LLM сервиса"""
    def __init__(self, message: str, status_code: int = 500):
        self.message     = message
        self.status_code = status_code
        super().__init__(message)


class ModelNotLoadedError(LLMServiceError):
    def __init__(self):
        super().__init__("Модель не загружена. Сервис не готов.", 503)


class RAGNotAvailableError(LLMServiceError):
    def __init__(self, reason: str = "RAG сервис недоступен"):
        super().__init__(reason, 503)


class GenerationError(LLMServiceError):
    def __init__(self, message: str):
        super().__init__(f"Ошибка генерации: {message}", 500)


class AppValidationError(LLMServiceError):
    def __init__(self, message: str):
        super().__init__(message, 400)