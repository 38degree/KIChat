# TTS Service — XTTS-v2 (Coqui TTS)
# Multilingual Text-to-Speech mit Deutsch-Support
FROM nvidia/cuda:12.8.0-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    ffmpeg \
    libsndfile1 \
    libsndfile1-dev \
    espeak-ng \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY server.py .

# Standard-Referenz-Audio für Stimme (kann überschrieben werden via Volume Mount)
RUN mkdir -p /models/tts/reference

EXPOSE 8001

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]
