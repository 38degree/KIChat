# OCR Service — Surya + marker-pdf
# Gerichtsakten PDF → Markdown
FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-deu \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY server.py .

EXPOSE 8003

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8003", "--workers", "1"]
