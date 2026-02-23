"""
Audio-Endpunkte — STT (Spracheingabe + Transkription) und TTS (Sprachausgabe)
=============================================================================

OpenAI-kompatible Audio-API für Open WebUI Integration:
  - POST /v1/audio/transcriptions  → Whisper STT
  - POST /v1/audio/speech          → XTTS-v2 TTS

Zusätzlich:
  - POST /v1/audio/transcribe-long → Lange Aufnahmen (Batch)
  - POST /v1/audio/denoise         → Audio-Bereinigung
"""

import io
import tempfile
import time
from pathlib import Path
from typing import Optional

import httpx
from fastapi import APIRouter, File, Form, Request, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from loguru import logger

from app.config import get_settings

router = APIRouter()


# ---------------------------------------------------------------------------
# POST /v1/audio/transcriptions — OpenAI-kompatible STT
# ---------------------------------------------------------------------------
@router.post("/audio/transcriptions")
async def transcribe_audio(
    request: Request,
    file: UploadFile = File(...),
    model: str = Form("whisper-large-v3"),
    language: Optional[str] = Form(None),
    response_format: Optional[str] = Form("json"),
    prompt: Optional[str] = Form(None),
):
    """
    Transkribiert Audio zu Text (Whisper large-v3).
    OpenAI-kompatibles Format für Open WebUI.
    """
    settings = get_settings()
    stt = request.app.state.stt

    if not stt.is_ready():
        raise HTTPException(status_code=503, detail="STT-Service nicht bereit.")

    # Audio lesen
    audio_bytes = await file.read()
    if len(audio_bytes) == 0:
        raise HTTPException(status_code=400, detail="Leere Audio-Datei.")

    logger.info(
        f"STT-Anfrage: {file.filename}, {len(audio_bytes)} bytes, Sprache: {language or settings.stt_language}"
    )

    try:
        result = await stt.transcribe(
            audio_bytes=audio_bytes,
            language=language or settings.stt_language,
        )

        if response_format == "verbose_json":
            return {
                "text": result["text"],
                "language": result.get("language", settings.stt_language),
                "duration": result.get("duration", 0),
                "segments": result.get("segments", []),
            }
        else:
            return {"text": result["text"]}

    except Exception as e:
        logger.error(f"STT fehlgeschlagen: {e}")
        raise HTTPException(status_code=500, detail=f"Transkription fehlgeschlagen: {e}")


# ---------------------------------------------------------------------------
# POST /v1/audio/speech — OpenAI-kompatible TTS
# ---------------------------------------------------------------------------
@router.post("/audio/speech")
async def text_to_speech(request: Request):
    """
    Konvertiert Text zu Sprache (XTTS-v2).
    OpenAI-kompatibles Format für Open WebUI.
    """
    settings = get_settings()

    body = await request.json()
    text = body.get("input", "")
    voice = body.get("voice", "default")
    response_format = body.get("response_format", "wav")

    if not text:
        raise HTTPException(status_code=400, detail="Kein Text angegeben.")

    logger.info(f"TTS-Anfrage: {len(text)} Zeichen, Stimme: {voice}")

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{settings.tts_service_url}/synthesize",
                json={
                    "text": text,
                    "language": "de",
                    "voice": voice,
                    "format": response_format,
                },
            )
            resp.raise_for_status()

        content_type = {
            "wav": "audio/wav",
            "mp3": "audio/mpeg",
            "opus": "audio/opus",
        }.get(response_format, "audio/wav")

        return StreamingResponse(
            io.BytesIO(resp.content),
            media_type=content_type,
            headers={
                "Content-Disposition": f"attachment; filename=speech.{response_format}"
            },
        )

    except httpx.HTTPError as e:
        logger.error(f"TTS-Service nicht erreichbar: {e}")
        raise HTTPException(status_code=502, detail=f"TTS-Service Fehler: {e}")


# ---------------------------------------------------------------------------
# POST /v1/audio/transcribe-long — Lange Patientenaufnahmen
# ---------------------------------------------------------------------------
@router.post("/audio/transcribe-long")
async def transcribe_long_audio(
    request: Request,
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
    denoise: bool = Form(True),
    patient_id: Optional[str] = Form(None),
):
    """
    Transkribiert lange Audio-Aufnahmen mit optionaler Vorbereinigung.
    Pipeline: Upload → Denoise (optional) → Whisper → Strukturiertes Transkript
    """
    settings = get_settings()
    stt = request.app.state.stt

    if not stt.is_ready():
        raise HTTPException(status_code=503, detail="STT-Service nicht bereit.")

    audio_bytes = await file.read()
    if len(audio_bytes) == 0:
        raise HTTPException(status_code=400, detail="Leere Audio-Datei.")

    logger.info(
        f"Langform-Transkription: {file.filename}, {len(audio_bytes) / 1024 / 1024:.1f} MB, "
        f"Denoise: {denoise}, Patient: {patient_id or 'anonym'}"
    )

    # --- Optionale Audio-Bereinigung ---
    if denoise:
        try:
            logger.info("Starte Audio-Bereinigung...")
            async with httpx.AsyncClient(timeout=600) as client:
                resp = await client.post(
                    f"{settings.denoiser_service_url}/denoise",
                    files={"file": (file.filename, audio_bytes, file.content_type)},
                    data={"enhance": "true"},
                )
                resp.raise_for_status()
                audio_bytes = resp.content
                logger.info(
                    f"Audio bereinigt: {len(audio_bytes) / 1024 / 1024:.1f} MB"
                )
        except Exception as e:
            logger.warning(f"Audio-Bereinigung fehlgeschlagen, nutze Original: {e}")

    # --- Transkription ---
    try:
        result = await stt.transcribe(
            audio_bytes=audio_bytes,
            language=language or settings.stt_language,
            word_timestamps=True,
        )

        # Strukturiertes Ergebnis
        transcript = {
            "text": result["text"],
            "language": result.get("language", settings.stt_language),
            "duration": result.get("duration", 0),
            "filename": file.filename,
            "patient_id": patient_id,
            "denoised": denoise,
            "segments": result.get("segments", []),
        }

        # Optional in Vektordatenbank speichern
        if patient_id:
            try:
                vectorstore = request.app.state.vectorstore
                await vectorstore.add_transcript(transcript, patient_id)
                transcript["indexed"] = True
                logger.info(f"Transkript in Vektordatenbank indexiert (Patient: {patient_id})")
            except Exception as e:
                logger.warning(f"Indexierung fehlgeschlagen: {e}")
                transcript["indexed"] = False

        return transcript

    except Exception as e:
        logger.error(f"Langform-Transkription fehlgeschlagen: {e}")
        raise HTTPException(
            status_code=500, detail=f"Transkription fehlgeschlagen: {e}"
        )


# ---------------------------------------------------------------------------
# POST /v1/audio/denoise — Standalone Audio-Bereinigung
# ---------------------------------------------------------------------------
@router.post("/audio/denoise")
async def denoise_audio(
    file: UploadFile = File(...),
    enhance: bool = Form(True),
):
    """
    Bereinigt Audio-Dateien (Hintergrundgeräusche entfernen + optional Enhancen).
    Nutzt Resemble Enhance Service.
    """
    settings = get_settings()
    audio_bytes = await file.read()

    if len(audio_bytes) == 0:
        raise HTTPException(status_code=400, detail="Leere Audio-Datei.")

    logger.info(
        f"Denoising: {file.filename}, {len(audio_bytes) / 1024 / 1024:.1f} MB, Enhance: {enhance}"
    )

    try:
        async with httpx.AsyncClient(timeout=600) as client:
            resp = await client.post(
                f"{settings.denoiser_service_url}/denoise",
                files={"file": (file.filename, audio_bytes, file.content_type)},
                data={"enhance": str(enhance).lower()},
            )
            resp.raise_for_status()

        return StreamingResponse(
            io.BytesIO(resp.content),
            media_type="audio/wav",
            headers={
                "Content-Disposition": f"attachment; filename=denoised_{file.filename}"
            },
        )

    except httpx.HTTPError as e:
        logger.error(f"Denoiser-Service Fehler: {e}")
        raise HTTPException(status_code=502, detail=f"Denoiser-Service Fehler: {e}")
