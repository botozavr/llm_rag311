import logging
import os
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler


def setup_logging() -> logging.Logger:
    """
    Настройка логирования:
    - консоль с цветами
    - файл с ротацией (10MB x 5 файлов)
    """

    LOG_DIR   = Path(os.getenv("LOG_DIR",   "./logs"))
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
    LOG_FILE  = LOG_DIR / "app.log"

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    class ColoredFormatter(logging.Formatter):
        COLORS = {
            "DEBUG":    "\033[36m",
            "INFO":     "\033[32m",
            "WARNING":  "\033[33m",
            "ERROR":    "\033[31m",
            "CRITICAL": "\033[41m",
        }
        RESET = "\033[0m"

        def format(self, record: logging.LogRecord) -> str:
            copy = logging.makeLogRecord(record.__dict__)
            color = self.COLORS.get(copy.levelname, self.RESET)
            copy.levelname = f"{color}{copy.levelname}{self.RESET}"
            return super().format(copy)

    PLAIN_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    DATE_FORMAT  = "%Y-%m-%d %H:%M:%S"

    plain_formatter   = logging.Formatter(fmt=PLAIN_FORMAT, datefmt=DATE_FORMAT)
    colored_formatter = ColoredFormatter(fmt=PLAIN_FORMAT,  datefmt=DATE_FORMAT)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(LOG_LEVEL)
    console_handler.setFormatter(colored_formatter)

    file_handler = RotatingFileHandler(
        filename=LOG_FILE,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(LOG_LEVEL)
    file_handler.setFormatter(plain_formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(LOG_LEVEL)
    root.addHandler(console_handler)
    root.addHandler(file_handler)

    for noisy in [
        "httpx", "httpcore", "urllib3",
        "chromadb", "sentence_transformers",
        "huggingface_hub.utils._http",
        "huggingface_hub.file_download",
        "filelock",
    ]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logger = logging.getLogger("llm_api")
    logger.info("Логи: консоль + файл %s (уровень: %s)", LOG_FILE.resolve(), LOG_LEVEL)
    return logger