FROM python:3.13-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    REDIS_URL=redis://redis:6379/0 \
    YOLO_CONFIG_DIR=/app/.ultralytics

WORKDIR /app

RUN mkdir -p /app/.ultralytics

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libgomp1 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install -r requirements.txt

# Bundled weights in models/ — host must download first; no GitHub access during Docker build.
COPY scripts/ensure_ultralytics_weights.py scripts/
COPY models/ models/
ENV BUILD_IN_DOCKER=1
RUN python scripts/ensure_ultralytics_weights.py

COPY . .

RUN mkdir -p storage/uploads storage/previews storage/outputs storage/alerts \
             face_recognition/models

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
