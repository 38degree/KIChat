"""
RAG-Verwaltungs-Endpunkte
=========================

Verwaltung der Vektordatenbank: Collections, Statistiken, Suche.
"""

from typing import Optional

from fastapi import APIRouter, Request, HTTPException, Query
from loguru import logger

from app.config import get_settings

router = APIRouter()


# ---------------------------------------------------------------------------
# GET /api/rag/stats — Statistiken der Vektordatenbank
# ---------------------------------------------------------------------------
@router.get("/rag/stats")
async def rag_stats(request: Request):
    """Statistiken: Anzahl Vektoren, Collections, Speicher."""
    try:
        vectorstore = request.app.state.vectorstore
        stats = await vectorstore.get_stats()
        return stats
    except Exception as e:
        logger.error(f"RAG-Stats fehlgeschlagen: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# POST /api/rag/search — Manuelle Suche in der Wissensdatenbank
# ---------------------------------------------------------------------------
@router.post("/rag/search")
async def rag_search(
    request: Request,
    query: str = Query(..., description="Suchanfrage"),
    top_k: int = Query(5, ge=1, le=20),
    document_type: Optional[str] = Query(None),
    patient_id: Optional[str] = Query(None),
):
    """Direkte Suche in Qdrant mit optionalen Filtern."""
    settings = get_settings()

    try:
        vectorstore = request.app.state.vectorstore

        filters = {}
        if document_type:
            filters["document_type"] = document_type
        if patient_id:
            filters["patient_id"] = patient_id

        results = await vectorstore.search(
            query=query,
            top_k=top_k,
            filters=filters if filters else None,
        )

        return {
            "query": query,
            "results": results,
            "total": len(results),
            "filters": filters,
        }

    except Exception as e:
        logger.error(f"RAG-Suche fehlgeschlagen: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# POST /api/rag/reindex — Collection neu aufbauen
# ---------------------------------------------------------------------------
@router.post("/rag/reindex")
async def rag_reindex(request: Request):
    """Löscht die Collection und baut sie neu auf (Vorsicht!)."""
    try:
        vectorstore = request.app.state.vectorstore
        await vectorstore.recreate_collection()
        return {"status": "success", "message": "Collection wurde neu erstellt."}
    except Exception as e:
        logger.error(f"Reindex fehlgeschlagen: {e}")
        raise HTTPException(status_code=500, detail=str(e))
