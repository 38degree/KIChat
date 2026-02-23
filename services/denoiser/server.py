"""
Denoiser Service — Resemble Enhance
======================================

Audio-Bereinigung für klinische Aufnahmen.

Zwei Modi:
  1. Denoise-only: Entfernt Hintergrundgeräusche
  2. Denoise + Enhance: Zusätzlich Bandbreiten-Erweiterung auf 44.1kHz

Endpunkte:
  POST /denoise  — Audio-Datei bereinigen
  GET  /health   — Health-Check
"""

import io
import os
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from loguru import logger

app = FastAPI(title="Denoiser Service", version="1.0.0")

DEVICE = os.getenv("DENOISER_DEVICE", "cuda")

# Globale Referenzen (lazy loaded)
_denoiser = None
_enhancer = None


def _load_models():
    """Lazy-Load Resemble Enhance Modelle."""
    global _denoiser, _enhancer

    if _denoiser is not None:
        return

    logger.info("Lade Resemble Enhance Modelle...")
    try:
        from resemble_enhance.enhancer.inference import denoise, enhance

        # Warmup — lädt die Modelle implizit
        _denoiser = denoise
        _enhancer = enhance
        logger.info(f"Resemble Enhance geladen auf {DEVICE}")

    except ImportError:
        logger.warning(
            "resemble-enhance nicht verfügbar. "
            "Fallback auf torchaudio spectral gating."
        )
        _denoiser = _fallback_denoise
        _enhancer = None


def _fallback_denoise(audio: torch.Tensor, sr: int, device: str) -> tuple:
    """
    Einfaches Spectral-Gating Denoising als Fallback
    wenn resemble-enhance nicht installiert ist.
    """
    # Simplifizierter Noise-Gate basierend auf RMS
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)

    # RMS berechnen
    rms = torch.sqrt(torch.mean(audio ** 2))
    threshold = rms * 0.1

    # Unter Threshold = Noise → dämpfen
    mask = (torch.abs(audio) > threshold).float()
    # Smoothing
    kernel_size = 1024
    if mask.shape[-1] > kernel_size:
        kernel = torch.ones(1, 1, kernel_size, device=mask.device) / kernel_size
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        mask = torch.nn.functional.conv1d(
            mask, kernel, padding=kernel_size // 2
        )
        mask = mask.squeeze(0)

    mask = torch.clamp(mask, 0, 1)
    denoised = audio * mask

    return denoised.squeeze(0), sr


@app.get("/health")
async def health():
    return {"status": "ok", "service": "denoiser"}


@app.post("/denoise")
async def denoise_audio(
    file: UploadFile = File(...),
    enhance: str = Form("true"),
):
    """
    Bereinigt Audio-Dateien.
    
    Args:
        file: Audio-Datei (WAV, MP3, FLAC, OGG)
        enhance: "true" für Denoise + Enhance, "false" nur Denoise
    
    Returns:
        Bereinigte Audio-Datei (WAV, 44.1kHz)
    """
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Leere Datei.")

    do_enhance = enhance.lower() == "true"

    logger.info(
        f"Denoising: {file.filename}, {len(audio_bytes) / 1024 / 1024:.1f} MB, "
        f"Enhance: {do_enhance}"
    )

    try:
        _load_models()

        # Audio in temporäre Datei schreiben und laden
        with tempfile.NamedTemporaryFile(
            suffix=Path(file.filename).suffix or ".wav", delete=False
        ) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            # Audio laden
            audio, sr = torchaudio.load(tmp_path)

            # Mono konvertieren
            if audio.shape[0] > 1:
                audio = audio.mean(dim=0, keepdim=False)
            else:
                audio = audio.squeeze(0)

            # Denoising
            if _denoiser is not None:
                logger.info("Starte Denoising...")
                denoised, new_sr = _denoiser(audio, sr, DEVICE)
                logger.info("Denoising abgeschlossen.")
            else:
                denoised, new_sr = audio, sr

            # Optional: Enhancement (Bandbreiten-Erweiterung)
            if do_enhance and _enhancer is not None:
                logger.info("Starte Enhancement...")
                enhanced, new_sr = _enhancer(
                    denoised, new_sr, DEVICE,
                    nfe=32,  # Number of function evaluations für CFM
                )
                output_audio = enhanced
                output_sr = new_sr
                logger.info("Enhancement abgeschlossen.")
            else:
                output_audio = denoised
                output_sr = new_sr

            # Zu numpy konvertieren
            if isinstance(output_audio, torch.Tensor):
                output_audio = output_audio.cpu().numpy()

            # Normalisieren (-1 bis 1)
            max_val = np.abs(output_audio).max()
            if max_val > 0:
                output_audio = output_audio / max_val * 0.95

            # Als WAV in Buffer schreiben
            buffer = io.BytesIO()
            sf.write(buffer, output_audio, output_sr, format="WAV")
            buffer.seek(0)

            logger.info(
                f"Denoising abgeschlossen: {file.filename} → "
                f"{output_sr}Hz, {len(output_audio) / output_sr:.1f}s"
            )

            return StreamingResponse(
                buffer,
                media_type="audio/wav",
                headers={
                    "Content-Disposition": f"attachment; filename=denoised_{file.filename}"
                },
            )

        finally:
            os.unlink(tmp_path)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Denoising fehlgeschlagen: {e}")
        raise HTTPException(status_code=500, detail=f"Denoising fehlgeschlagen: {e}")
