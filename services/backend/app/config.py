"""
Psychiatrische KI-Plattform — Zentrale Konfiguration
Alle Einstellungen werden aus Umgebungsvariablen geladen.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache


class Settings(BaseSettings):
    """Zentrale Konfiguration — wird aus .env / Umgebungsvariablen geladen."""

    # --- LLM ---
    llm_base_url: str = Field("http://vllm:8000/v1", alias="LLM_BASE_URL")
    llm_model: str = Field("Qwen/Qwen2.5-72B-Instruct", alias="LLM_MODEL")
    llm_max_tokens: int = Field(4096, alias="LLM_MAX_TOKENS")
    llm_temperature: float = Field(0.3, alias="LLM_TEMPERATURE")

    # --- STT (Whisper) ---
    stt_model: str = Field("openai/whisper-large-v3", alias="STT_MODEL")
    stt_language: str = Field("de", alias="STT_LANGUAGE")
    stt_device: str = Field("cuda", alias="STT_DEVICE")

    # --- TTS ---
    tts_service_url: str = Field("http://tts:8001", alias="TTS_SERVICE_URL")

    # --- Denoiser ---
    denoiser_service_url: str = Field("http://denoiser:8002", alias="DENOISER_SERVICE_URL")

    # --- OCR ---
    ocr_service_url: str = Field("http://ocr:8003", alias="OCR_SERVICE_URL")

    # --- Embedding ---
    embedding_model: str = Field(
        "intfloat/multilingual-e5-large", alias="EMBEDDING_MODEL"
    )
    embedding_device: str = Field("cpu", alias="EMBEDDING_DEVICE")
    embedding_batch_size: int = Field(32, alias="EMBEDDING_BATCH_SIZE")

    # --- Qdrant ---
    qdrant_host: str = Field("qdrant", alias="QDRANT_HOST")
    qdrant_port: int = Field(6333, alias="QDRANT_PORT")
    qdrant_collection: str = Field(
        "psychiatric_knowledge", alias="QDRANT_COLLECTION"
    )

    # --- RAG ---
    rag_chunk_size: int = Field(512, alias="RAG_CHUNK_SIZE")
    rag_chunk_overlap: int = Field(64, alias="RAG_CHUNK_OVERLAP")
    rag_top_k: int = Field(5, alias="RAG_TOP_K")
    rag_similarity_threshold: float = Field(0.7, alias="RAG_SIMILARITY_THRESHOLD")

    # --- Server ---
    backend_host: str = Field("0.0.0.0", alias="BACKEND_HOST")
    backend_port: int = Field(8080, alias="BACKEND_PORT")

    # --- HuggingFace ---
    hf_token: str = Field("", alias="HF_TOKEN")

    model_config = {"env_file": ".env", "extra": "ignore", "populate_by_name": True}


@lru_cache()
def get_settings() -> Settings:
    """Singleton für Settings — cached nach erstem Aufruf."""
    return Settings()
