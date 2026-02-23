"""
Embedding-Service — multilingual-e5-large
==========================================

Läuft auf CPU (Grace ARM), um GPU-Speicher für das LLM freizuhalten.
Unterstützt Batch-Embedding für effiziente Dokumenten-Indexierung.
"""

import asyncio
from typing import Union

import torch
from loguru import logger
from sentence_transformers import SentenceTransformer

from app.config import get_settings


class EmbeddingService:
    """Verwaltet das Embedding-Modell (multilingual-e5-large)."""

    def __init__(self):
        self._model: SentenceTransformer | None = None
        self._settings = get_settings()
        self._dimension: int = 1024  # multilingual-e5-large Output-Dimension

    async def initialize(self):
        """Modell laden (blockierend → in Thread ausführen)."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_model)

    def _load_model(self):
        """Lädt das Embedding-Modell."""
        logger.info(
            f"Lade Embedding-Modell: {self._settings.embedding_model} "
            f"auf {self._settings.embedding_device}"
        )
        self._model = SentenceTransformer(
            self._settings.embedding_model,
            device=self._settings.embedding_device,
            trust_remote_code=True,
        )
        # Dimension aus dem Modell auslesen
        self._dimension = self._model.get_sentence_embedding_dimension()
        logger.info(
            f"Embedding-Modell geladen. Dimension: {self._dimension}, "
            f"Device: {self._settings.embedding_device}"
        )

    def is_ready(self) -> bool:
        return self._model is not None

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed(self, texts: Union[str, list[str]]) -> list[list[float]]:
        """
        Erzeugt Embeddings für einen oder mehrere Texte.
        
        multilingual-e5-large erwartet Präfixe:
          - "query: " für Suchanfragen
          - "passage: " für Dokumente/Chunks
        
        Diese Methode wird intern aufgerufen — Präfixe müssen vorher gesetzt sein.
        """
        if self._model is None:
            raise RuntimeError("Embedding-Modell nicht initialisiert.")

        if isinstance(texts, str):
            texts = [texts]

        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None,
            lambda: self._model.encode(
                texts,
                batch_size=self._settings.embedding_batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            ),
        )

        return embeddings.tolist()

    async def embed_query(self, query: str) -> list[float]:
        """Embedding für eine Suchanfrage (mit 'query: '-Präfix)."""
        prefixed = f"query: {query}"
        results = await self.embed(prefixed)
        return results[0]

    async def embed_documents(self, documents: list[str]) -> list[list[float]]:
        """Embeddings für Dokument-Chunks (mit 'passage: '-Präfix)."""
        prefixed = [f"passage: {doc}" for doc in documents]
        return await self.embed(prefixed)
