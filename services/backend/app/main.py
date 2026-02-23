"""
Psychiatrische KI-Plattform — FastAPI Backend
==============================================

Orchestriert alle Services:
  - LLM (vLLM) → Chat, Reasoning
  - RAG (Qdrant + Embeddings) → Wissensbasierte Antworten
  - STT (Whisper) → Spracheingabe + Transkription
  - TTS (XTTS-v2) → Sprachausgabe
  - Denoiser (Resemble Enhance) → Audio-Bereinigung
  - OCR (Surya + marker-pdf) → PDF-Verarbeitung

Exponiert OpenAI-kompatible API für Open WebUI.
"""

import uuid
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.config import get_settings
from app.routes import chat, audio, documents, rag, health
from app.rag.embedding import EmbeddingService
from app.rag.vectorstore import VectorStoreService
from app.audio.stt import STTService


# ---------------------------------------------------------------------------
# Lifespan — Initialisierung und Shutdown
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(application: FastAPI):
    """Start/Stop aller Services."""
    settings = get_settings()
    logger.info("=== Psychiatrische KI-Plattform startet ===")
    logger.info(f"LLM: {settings.llm_model} via {settings.llm_base_url}")
    logger.info(f"Embedding: {settings.embedding_model} auf {settings.embedding_device}")
    logger.info(f"Qdrant: {settings.qdrant_host}:{settings.qdrant_port}")

    # Embedding-Modell laden (CPU — blockiert GPU nicht)
    logger.info("Lade Embedding-Modell...")
    embedding_svc = EmbeddingService()
    await embedding_svc.initialize()
    application.state.embedding = embedding_svc
    logger.info("Embedding-Modell geladen.")

    # Qdrant-Verbindung + Collection sicherstellen
    logger.info("Verbinde mit Qdrant...")
    vector_svc = VectorStoreService(embedding_svc)
    await vector_svc.initialize()
    application.state.vectorstore = vector_svc
    logger.info("Qdrant bereit.")

    # STT (Whisper) laden
    logger.info("Lade Whisper STT-Modell...")
    stt_svc = STTService()
    await stt_svc.initialize()
    application.state.stt = stt_svc
    logger.info("Whisper STT bereit.")

    logger.info("=== Alle Services bereit ===")

    yield

    # Shutdown
    logger.info("=== Shutdown ===")
    if hasattr(application.state, "stt"):
        await application.state.stt.shutdown()
    logger.info("Shutdown abgeschlossen.")


# ---------------------------------------------------------------------------
# App erstellen
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Psychiatrische KI-Plattform",
    description="Lokale KI-Plattform für psychiatrische Diagnostik und Dokumentation",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS für Open WebUI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routen einbinden
# ---------------------------------------------------------------------------
app.include_router(health.router, tags=["Health"])
app.include_router(chat.router, prefix="/v1", tags=["Chat (OpenAI-kompatibel)"])
app.include_router(audio.router, prefix="/v1", tags=["Audio (STT/TTS)"])
app.include_router(documents.router, prefix="/api", tags=["Dokumente (OCR/RAG)"])
app.include_router(rag.router, prefix="/api", tags=["RAG Verwaltung"])
