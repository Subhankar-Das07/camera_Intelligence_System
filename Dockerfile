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

# ── Step 1: Install CPU-only PyTorch BEFORE requirements.txt ──────────────────
# PyPI defaults to serving the CUDA-enabled wheel of torch (~2.5 GB) because
# it bundles every nvidia-* math library. On Intel hardware none of that CUDA
# code ever runs -- it is pure wasted space.
#
# By installing torch + torchvision from the official PyTorch *cpu* index first,
# pip will see that torch is already satisfied when it later processes
# requirements.txt (via ultralytics → torch) and will NOT download the heavy
# CUDA wheel. This saves ~2.5–3 GB from the final image.
#
# The --index-url flag tells pip to look at the CPU-only wheel index for these
# two packages only; everything else still comes from PyPI.
RUN pip install --upgrade pip && \
    pip install \
        torch \
        torchvision \
        --index-url https://download.pytorch.org/whl/cpu

# ── Step 2: Install all other project dependencies ────────────────────────────
# torch is already in the environment, so ultralytics will NOT re-install it.
# All nvidia-* packages are therefore never downloaded.
RUN pip install -r requirements.txt

# Pre-download YOLO weights so first Monitor / pipeline run is not blocked on GitHub
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt'); YOLO('yolov8n-pose.pt')"

COPY . .

RUN mkdir -p storage/uploads storage/previews storage/outputs storage/alerts \
             face_recognition/models

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
