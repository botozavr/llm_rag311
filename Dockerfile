# ================================================================
# Базовый образ: PyTorch с CUDA 12.8 (для RTX 5060 Ti / Blackwell)
# nvidia/cuda — официальный образ с драйверами CUDA
# cudnn9 — нужен для работы трансформеров
# python3.11 — стабильная версия для всех ML библиотек
# ================================================================
# FROM nvidia/cuda:12.8.0-cudnn9-runtime-ubuntu22.04
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

# ================================================================
# Системные переменные
# ================================================================
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    # Куда pip ставит пакеты
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Кэш HuggingFace моделей внутри контейнера
    HF_HOME=/app/.cache/huggingface \
    # Кэш torch
    TORCH_HOME=/app/.cache/torch

# ================================================================
# Системные зависимости с повторными попытками при сетевых сбоях
# ================================================================
RUN for i in 1 2 3; do \
        apt-get update && \
        apt-get install -y --no-install-recommends \
            python3.11 \
            python3.11-dev \
            python3-pip \
            python3.11-venv \
            build-essential \
            curl \
            wget \
            git \
            poppler-utils \
        && apt-get clean \
        && rm -rf /var/lib/apt/lists/* \
        && break || sleep 20; \
    done

# Делаем python3.11 дефолтным
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Обновляем pip
RUN python -m pip install --upgrade pip setuptools wheel

# ================================================================
# Рабочая директория
# ================================================================
WORKDIR /app

# ================================================================
# Зависимости — копируем ОТДЕЛЬНО от кода
# Это позволяет Docker кэшировать слой с зависимостями.
# Если код изменился, но requirements.txt нет — pip не запускается.
# ================================================================
COPY requirements.txt .

# Устанавливаем torch отдельно (специальный индекс для CUDA 12.8)
RUN pip install torch --index-url https://download.pytorch.org/whl/cu128

# Устанавливаем остальные зависимости
RUN pip install -r requirements.txt

# ================================================================
# Копируем код приложения
# ================================================================
COPY app/         ./app/
COPY scripts/     ./scripts/
COPY main.py      .

# ================================================================
# Создаём директории которые нужны при работе
# ================================================================
RUN mkdir -p \
    /app/logs \
    /app/documents \
    /app/chroma_db \
    /app/.cache/huggingface \
    /app/.cache/torch

# ================================================================
# Порт
# ================================================================
EXPOSE 8000

# ================================================================
# Healthcheck — Docker будет проверять что сервис живой
# ================================================================
HEALTHCHECK \
    --interval=30s \
    --timeout=10s \
    --start-period=120s \
    --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ================================================================
# Команда запуска
# ================================================================
CMD ["python", "-m", "uvicorn", "main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "warning", \
     "--no-access-log"]