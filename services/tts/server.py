"""
TTS Service — XTTS-v2 (Coqui TTS)
====================================

Multilingual Text-to-Speech mit Deutsch-Support und Voice-Cloning.

Endpunkte:
  POST /synthesize       — Text → Audio (WAV)
  POST /synthesize-stream — Text → Audio (Streaming)
  POST /clone-voice       — Referenz-Audio hochladen für Voice-Cloning
  GET  /voices           — Verfügbare Stimmen auflisten
  GET  /health           — Health-Check
"""

import io
import os
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="TTS Service", version="1.0.0")

# Globale Referenz
_tts_model = None
_reference_dir = Path("/models/tts/reference")

DEVICE = os.getenv("TTS_DEVICE", "cuda")


class SynthesizeRequest(BaseModel):
    text: str
    language: str = "de"
    voice: str = "default"
    format: str = "wav"


def _get_model():
    """Lazy-Load XTTS-v2 Modell."""
    global _tts_model
    if _tts_model is None:
        logger.info("Lade XTTS-v2 Modell...")
        try:
            from TTS.api import TTS

            model_name = os.getenv(
                "TTS_MODEL", "tts_models/multilingual/multi-dataset/xtts_v2"
            )
            _tts_model = TTS(model_name, gpu=(DEVICE == "cuda"))
            logger.info(f"XTTS-v2 geladen auf {DEVICE}")
        except Exception as e:
            logger.error(f"TTS-Modell Laden fehlgeschlagen: {e}")
            raise RuntimeError(f"TTS nicht verfügbar: {e}")
    return _tts_model


def _get_reference_audio(voice: str = "default") -> Optional[str]:
    """Findet Referenz-Audio für Voice-Cloning."""
    _reference_dir.mkdir(parents=True, exist_ok=True)

    # Suche nach benannter Stimme
    for ext in [".wav", ".mp3", ".flac", ".ogg"]:
        path = _reference_dir / f"{voice}{ext}"
        if path.exists():
            return str(path)

    # Fallback: Default-Referenz
    for ext in [".wav", ".mp3", ".flac", ".ogg"]:
        path = _reference_dir / f"default{ext}"
        if path.exists():
            return str(path)

    return None


@app.get("/health")
async def health():
    return {"status": "ok", "service": "tts"}


@app.get("/voices")
async def list_voices():
    """Listet alle verfügbaren Stimmen (Referenz-Audios)."""
    _reference_dir.mkdir(parents=True, exist_ok=True)
    voices = []
    for f in _reference_dir.iterdir():
        if f.suffix in [".wav", ".mp3", ".flac", ".ogg"]:
            voices.append({"name": f.stem, "file": f.name})

    return {"voices": voices, "default_available": any(v["name"] == "default" for v in voices)}


@app.post("/synthesize")
async def synthesize(request: SynthesizeRequest):
    """
    Konvertiert Text zu Sprache.
    
    Verwendet XTTS-v2 mit optionalem Voice-Cloning.
    Ohne Referenz-Audio wird eine Standard-Stimme verwendet.
    """
    if not request.text:
        raise HTTPException(status_code=400, detail="Kein Text angegeben.")

    if len(request.text) > 10000:
        raise HTTPException(
            status_code=400,
            detail="Text zu lang (max. 10.000 Zeichen). Teile den Text auf."
        )

    logger.info(
        f"TTS-Synthese: {len(request.text)} Zeichen, "
        f"Sprache: {request.language}, Stimme: {request.voice}"
    )

    try:
        model = _get_model()
        reference_audio = _get_reference_audio(request.voice)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            if reference_audio:
                # Voice-Cloning mit Referenz-Audio
                model.tts_to_file(
                    text=request.text,
                    file_path=tmp_path,
                    speaker_wav=reference_audio,
                    language=request.language,
                )
            else:
                # Standard-Stimme (erste verfügbare)
                model.tts_to_file(
                    text=request.text,
                    file_path=tmp_path,
                    language=request.language,
                )

            # Audio lesen und zurückgeben
            audio_data, sample_rate = sf.read(tmp_path)

            # Format konvertieren
            buffer = io.BytesIO()
            sf.write(buffer, audio_data, sample_rate, format="WAV")
            buffer.seek(0)

            content_type = {
                "wav": "audio/wav",
                "mp3": "audio/mpeg",
            }.get(request.format, "audio/wav")

            return StreamingResponse(
                buffer,
                media_type=content_type,
                headers={
                    "Content-Disposition": f"attachment; filename=speech.{request.format}"
                },
            )

        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"TTS-Synthese fehlgeschlagen: {e}")
        raise HTTPException(status_code=500, detail=f"TTS fehlgeschlagen: {e}")


@app.post("/clone-voice")
async def clone_voice(
    file: UploadFile = File(...),
    name: str = Form("default"),
):
    """
    Lädt Referenz-Audio für Voice-Cloning hoch.
    Mindestens 6 Sekunden klares Sprach-Audio empfohlen.
    """
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Leere Datei.")

    _reference_dir.mkdir(parents=True, exist_ok=True)

    # Speichere als WAV
    ext = Path(file.filename).suffix or ".wav"
    save_path = _reference_dir / f"{name}{ext}"

    with open(save_path, "wb") as f:
        f.write(audio_bytes)

    logger.info(f"Referenz-Audio gespeichert: {save_path} ({len(audio_bytes)} bytes)")

    return {
        "status": "success",
        "voice_name": name,
        "file": str(save_path),
        "size_bytes": len(audio_bytes),
    }
