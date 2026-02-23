"""
Vectorstore-Service — Qdrant Integration
==========================================

Verwaltet die Vektordatenbank für RAG:
  - Collection erstellen/verwalten
  - Dokumente chunken, embedden und speichern
  - Ähnlichkeitssuche mit Metadaten-Filter
  - Dokumentenverwaltung (Liste, Löschen)
"""

import uuid
from typing import Optional

from loguru import logger
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    ScalarQuantizationConfig,
    ScalarType,
    VectorParams,
)

from app.config import get_settings
from app.rag.embedding import EmbeddingService
from app.rag.chunking import TextChunker


class VectorStoreService:
    """Qdrant-basierter Vectorstore mit Embedding-Integration."""

    def __init__(self, embedding_service: EmbeddingService):
        self._client: AsyncQdrantClient | None = None
        self._embedding = embedding_service
        self._chunker = TextChunker()
        self._settings = get_settings()

    async def initialize(self):
        """Verbindung herstellen und Collection sicherstellen."""
        self._client = AsyncQdrantClient(
            host=self._settings.qdrant_host,
            port=self._settings.qdrant_port,
            timeout=30,
        )
        await self._ensure_collection()

    async def is_ready(self) -> bool:
        """Prüft ob Qdrant erreichbar ist."""
        try:
            if self._client is None:
                return False
            await self._client.get_collections()
            return True
        except Exception:
            return False

    async def _ensure_collection(self):
        """Erstellt die Collection falls nicht vorhanden."""
        collection_name = self._settings.qdrant_collection
        collections = await self._client.get_collections()
        existing = [c.name for c in collections.collections]

        if collection_name not in existing:
            logger.info(f"Erstelle Collection: {collection_name}")
            await self._client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=self._embedding.dimension,
                    distance=Distance.COSINE,
                ),
                quantization_config=ScalarQuantizationConfig(
                    type=ScalarType.INT8,
                    always_ram=True,
                ),
            )
            logger.info(f"Collection '{collection_name}' erstellt (INT8 Quantisierung)")
        else:
            logger.info(f"Collection '{collection_name}' existiert bereits.")

    async def recreate_collection(self):
        """Löscht und erstellt die Collection neu."""
        collection_name = self._settings.qdrant_collection
        try:
            await self._client.delete_collection(collection_name)
            logger.info(f"Collection '{collection_name}' gelöscht.")
        except Exception:
            pass
        await self._ensure_collection()

    # ------------------------------------------------------------------
    # Indexierung
    # ------------------------------------------------------------------
    async def add_text(self, text: str, metadata: dict) -> int:
        """
        Chunked einen Text, erzeugt Embeddings und speichert in Qdrant.
        Gibt die Anzahl indizierter Chunks zurück.
        """
        # Text in Chunks aufteilen
        chunks = self._chunker.chunk(text)
        if not chunks:
            return 0

        # Embeddings erzeugen
        embeddings = await self._embedding.embed_documents(chunks)

        # In Qdrant speichern
        points = []
        for chunk_text, embedding in zip(chunks, embeddings):
            point_id = str(uuid.uuid4())
            points.append(
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "text": chunk_text,
                        **metadata,
                    },
                )
            )

        await self._client.upsert(
            collection_name=self._settings.qdrant_collection,
            points=points,
        )

        return len(points)

    async def add_transcript(self, transcript: dict, patient_id: str) -> int:
        """Indexiert ein Transkript mit Patienten-Metadaten."""
        text = transcript.get("text", "")
        metadata = {
            "source": transcript.get("filename", "audio_transcript"),
            "document_type": "transcript",
            "patient_id": patient_id,
            "duration": transcript.get("duration", 0),
            "document_id": str(uuid.uuid4()),
            "page": 0,
        }
        return await self.add_text(text, metadata)

    # ------------------------------------------------------------------
    # Suche
    # ------------------------------------------------------------------
    async def search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[dict] = None,
    ) -> list[dict]:
        """
        Ähnlichkeitssuche in Qdrant.
        Gibt Liste von {text, score, metadata} zurück.
        """
        # Query embedden
        query_embedding = await self._embedding.embed_query(query)

        # Filter aufbauen
        qdrant_filter = None
        if filters:
            conditions = []
            for key, value in filters.items():
                conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )
            if conditions:
                qdrant_filter = Filter(must=conditions)

        # Suche ausführen
        results = await self._client.query_points(
            collection_name=self._settings.qdrant_collection,
            query=query_embedding,
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True,
        )

        # Ergebnisse formatieren
        formatted = []
        for point in results.points:
            payload = point.payload or {}
            formatted.append(
                {
                    "text": payload.get("text", ""),
                    "score": point.score,
                    "metadata": {
                        k: v for k, v in payload.items() if k != "text"
                    },
                }
            )

        return formatted

    # ------------------------------------------------------------------
    # Verwaltung
    # ------------------------------------------------------------------
    async def list_documents(self) -> list[dict]:
        """Listet unterschiedliche Dokumente (gruppiert nach document_id)."""
        # Scroll durch alle Punkte, nur Metadaten
        documents = {}
        offset = None
        batch_size = 100

        while True:
            results = await self._client.scroll(
                collection_name=self._settings.qdrant_collection,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            points, next_offset = results

            for point in points:
                doc_id = point.payload.get("document_id", "unknown")
                if doc_id not in documents:
                    documents[doc_id] = {
                        "document_id": doc_id,
                        "source": point.payload.get("source", ""),
                        "document_type": point.payload.get("document_type", ""),
                        "patient_id": point.payload.get("patient_id", ""),
                        "case_number": point.payload.get("case_number", ""),
                        "total_pages": point.payload.get("total_pages", 0),
                        "chunks": 0,
                    }
                documents[doc_id]["chunks"] += 1

            if next_offset is None:
                break
            offset = next_offset

        return list(documents.values())

    async def delete_document(self, document_id: str) -> int:
        """Löscht alle Chunks eines Dokuments."""
        # Zähle zuerst die Punkte
        results = await self._client.scroll(
            collection_name=self._settings.qdrant_collection,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="document_id", match=MatchValue(value=document_id)
                    )
                ]
            ),
            limit=10000,
            with_payload=False,
            with_vectors=False,
        )
        count = len(results[0])

        # Löschen
        await self._client.delete(
            collection_name=self._settings.qdrant_collection,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="document_id", match=MatchValue(value=document_id)
                    )
                ]
            ),
        )

        return count

    async def get_stats(self) -> dict:
        """Statistiken der Collection."""
        try:
            info = await self._client.get_collection(
                self._settings.qdrant_collection
            )
            return {
                "collection": self._settings.qdrant_collection,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "segments_count": len(info.segments) if info.segments else 0,
                "status": info.status.value if info.status else "unknown",
            }
        except Exception as e:
            return {"error": str(e)}
