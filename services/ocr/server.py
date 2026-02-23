"""
OCR Service — Surya + marker-pdf
==================================

Verarbeitet PDF-Dokumente (Gerichtsakten) zu strukturiertem Markdown.

Endpunkte:
  POST /process  — PDF → Markdown/JSON
  GET  /health   — Health-Check
"""

import io
import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from loguru import logger

app = FastAPI(title="OCR Service", version="1.0.0")

# Globale Referenzen (lazy loaded)
_marker_converter = None


def _get_converter():
    """Lazy-Load des marker-pdf Converters (spart GPU-Memory wenn nicht genutzt)."""
    global _marker_converter
    if _marker_converter is None:
        logger.info("Lade marker-pdf Converter...")
        try:
            from marker.converters.pdf import PdfConverter
            from marker.config.parser import ConfigParser
            from marker.models import create_model_dict

            config_parser = ConfigParser(
                {
                    "output_format": "markdown",
                    "languages": ["de", "en"],
                    "disable_multiprocessing": True,
                }
            )
            models = create_model_dict()
            _marker_converter = PdfConverter(
                config=config_parser.generate_config_dict(),
                artifact_dict=models,
            )
            logger.info("marker-pdf Converter geladen.")
        except Exception as e:
            logger.error(f"marker-pdf Laden fehlgeschlagen: {e}")
            raise RuntimeError(f"marker-pdf nicht verfügbar: {e}")
    return _marker_converter


@app.get("/health")
async def health():
    return {"status": "ok", "service": "ocr"}


@app.post("/process")
async def process_pdf(
    file: UploadFile = File(...),
    output_format: str = Form("markdown"),
):
    """
    Verarbeitet ein PDF mit marker-pdf (Surya-basiert).
    
    Returns:
        - markdown: Gesamter extrahierter Text als Markdown
        - pages: Liste von {page, text} für seitenweise Verarbeitung
        - total_pages: Seitenanzahl
        - metadata: Extrahierte Dokumentmetadaten
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Nur PDF-Dateien unterstützt.")

    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Leere Datei.")

    logger.info(f"OCR-Verarbeitung: {file.filename} ({len(pdf_bytes) / 1024 / 1024:.1f} MB)")

    try:
        # PDF in temporäre Datei schreiben
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name

        try:
            converter = _get_converter()

            # marker-pdf Konvertierung
            rendered = converter(tmp_path)
            markdown_text = rendered.markdown

            # Seitenweise Ergebnisse extrahieren
            pages = []
            if hasattr(rendered, "children") and rendered.children:
                for i, page_block in enumerate(rendered.children):
                    page_text = ""
                    if hasattr(page_block, "rendered"):
                        page_text = page_block.rendered.markdown
                    elif hasattr(page_block, "children"):
                        parts = []
                        for child in page_block.children:
                            if hasattr(child, "rendered"):
                                parts.append(child.rendered.markdown)
                        page_text = "\n".join(parts)
                    pages.append({"page": i + 1, "text": page_text})

            # Wenn keine Seiten extrahiert, Gesamttext als eine Seite
            if not pages and markdown_text:
                pages = [{"page": 1, "text": markdown_text}]

            # Metadaten extrahieren
            metadata = {}
            if hasattr(rendered, "metadata"):
                metadata = rendered.metadata or {}

            total_pages = len(pages)

            logger.info(
                f"OCR abgeschlossen: {file.filename} → "
                f"{total_pages} Seiten, {len(markdown_text)} Zeichen"
            )

            return {
                "markdown": markdown_text,
                "pages": pages,
                "total_pages": total_pages,
                "metadata": metadata,
                "filename": file.filename,
            }

        finally:
            # Temporäre Datei aufräumen
            os.unlink(tmp_path)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"OCR-Verarbeitung fehlgeschlagen: {e}")
        raise HTTPException(
            status_code=500, detail=f"OCR-Verarbeitung fehlgeschlagen: {e}"
        )


@app.post("/ocr-raw")
async def ocr_raw(
    file: UploadFile = File(...),
):
    """
    Direkte Surya-OCR ohne marker-pdf (für einzelne Bilder/Scans).
    Fallback wenn marker-pdf nicht funktioniert.
    """
    try:
        from surya.recognition import RecognitionPredictor
        from surya.detection import DetectionPredictor
        from PIL import Image

        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))

        det_predictor = DetectionPredictor()
        rec_predictor = RecognitionPredictor()

        # Text erkennen
        from surya.recognition import run_recognition
        from surya.detection import run_detection

        det_results = run_detection([image], det_predictor)
        rec_results = run_recognition(
            [image], det_results, rec_predictor, languages=["de", "en"]
        )

        text_lines = []
        if rec_results:
            for line in rec_results[0].text_lines:
                text_lines.append(line.text)

        return {
            "text": "\n".join(text_lines),
            "lines": len(text_lines),
        }

    except Exception as e:
        logger.error(f"Surya-OCR fehlgeschlagen: {e}")
        raise HTTPException(status_code=500, detail=f"OCR fehlgeschlagen: {e}")
