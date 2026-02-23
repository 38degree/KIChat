"""
Dokument-Endpunkte — PDF-Upload, OCR, Indexierung
==================================================

Pipeline: PDF-Upload → OCR (Surya/marker-pdf) → Chunking → Embedding → Qdrant
"""

import uuid
from typing import Optional

import httpx
from fastapi import APIRouter, File, Form, Request, UploadFile, HTTPException
from loguru import logger

from app.config import get_settings

router = APIRouter()


# ---------------------------------------------------------------------------
# POST /api/documents/upload — PDF hochladen und verarbeiten
# ---------------------------------------------------------------------------
@router.post("/documents/upload")
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    document_type: str = Form("gerichtsakte"),
    patient_id: Optional[str] = Form(None),
    case_number: Optional[str] = Form(None),
):
    """
    Verarbeitet PDF-Dokumente:
    1. PDF an OCR-Service senden (Surya + marker-pdf)
    2. Markdown-Ergebnis chunken
    3. Chunks embedden und in Qdrant speichern
    """
    settings = get_settings()
    vectorstore = request.app.state.vectorstore

    pdf_bytes = await file.read()
    if len(pdf_bytes) == 0:
        raise HTTPException(status_code=400, detail="Leere Datei.")

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Nur PDF-Dateien werden unterstützt.")

    logger.info(
        f"Dokument-Upload: {file.filename}, {len(pdf_bytes) / 1024 / 1024:.1f} MB, "
        f"Typ: {document_type}, Patient: {patient_id or 'k.A.'}, "
        f"Aktenzeichen: {case_number or 'k.A.'}"
    )

    # --- OCR-Verarbeitung ---
    try:
        logger.info("Sende an OCR-Service...")
        async with httpx.AsyncClient(timeout=600) as client:
            resp = await client.post(
                f"{settings.ocr_service_url}/process",
                files={"file": (file.filename, pdf_bytes, "application/pdf")},
                data={"output_format": "markdown"},
            )
            resp.raise_for_status()
            ocr_result = resp.json()

        markdown_text = ocr_result.get("markdown", "")
        pages = ocr_result.get("pages", [])
        total_pages = ocr_result.get("total_pages", 0)

        logger.info(
            f"OCR abgeschlossen: {total_pages} Seiten, "
            f"{len(markdown_text)} Zeichen extrahiert"
        )

    except httpx.HTTPError as e:
        logger.error(f"OCR-Service Fehler: {e}")
        raise HTTPException(status_code=502, detail=f"OCR-Verarbeitung fehlgeschlagen: {e}")

    if not markdown_text.strip():
        raise HTTPException(
            status_code=422, detail="OCR konnte keinen Text aus dem PDF extrahieren."
        )

    # --- Chunking und Indexierung ---
    try:
        doc_id = str(uuid.uuid4())
        metadata = {
            "source": file.filename,
            "document_id": doc_id,
            "document_type": document_type,
            "patient_id": patient_id or "",
            "case_number": case_number or "",
            "total_pages": total_pages,
        }

        # Seitenweise Chunks erstellen (jede Seite hat Seitennummer)
        chunks_indexed = 0
        if pages:
            for page_info in pages:
                page_text = page_info.get("text", "")
                page_num = page_info.get("page", 0)
                if page_text.strip():
                    page_metadata = {**metadata, "page": page_num}
                    count = await vectorstore.add_text(
                        text=page_text,
                        metadata=page_metadata,
                    )
                    chunks_indexed += count
        else:
            # Fallback: Gesamttext chunken
            count = await vectorstore.add_text(
                text=markdown_text,
                metadata={**metadata, "page": 0},
            )
            chunks_indexed = count

        logger.info(f"Indexiert: {chunks_indexed} Chunks aus {file.filename}")

        return {
            "status": "success",
            "document_id": doc_id,
            "filename": file.filename,
            "total_pages": total_pages,
            "chunks_indexed": chunks_indexed,
            "characters_extracted": len(markdown_text),
            "metadata": metadata,
        }

    except Exception as e:
        logger.error(f"Indexierung fehlgeschlagen: {e}")
        raise HTTPException(
            status_code=500, detail=f"Indexierung fehlgeschlagen: {e}"
        )


# ---------------------------------------------------------------------------
# GET /api/documents — Alle indexierten Dokumente auflisten
# ---------------------------------------------------------------------------
@router.get("/documents")
async def list_documents(request: Request):
    """Listet alle indexierten Dokumente mit Metadaten."""
    try:
        vectorstore = request.app.state.vectorstore
        docs = await vectorstore.list_documents()
        return {"documents": docs, "total": len(docs)}
    except Exception as e:
        logger.error(f"Dokument-Liste fehlgeschlagen: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# DELETE /api/documents/{document_id} — Dokument aus Index entfernen
# ---------------------------------------------------------------------------
@router.delete("/documents/{document_id}")
async def delete_document(request: Request, document_id: str):
    """Entfernt ein Dokument und alle zugehörigen Chunks aus Qdrant."""
    try:
        vectorstore = request.app.state.vectorstore
        deleted = await vectorstore.delete_document(document_id)
        return {
            "status": "success",
            "document_id": document_id,
            "chunks_deleted": deleted,
        }
    except Exception as e:
        logger.error(f"Löschung fehlgeschlagen: {e}")
        raise HTTPException(status_code=500, detail=str(e))
