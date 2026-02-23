"""
Speech-to-Text Service — Whisper large-v3
==========================================

Unterstützt zwei Modi:
  1. Echtzeit-STT: Kurze Audio-Segmente (<30s) für Chat-Spracheingabe
  2. Batch-Transkription: Lange Aufnahmen mit Zeitstempeln

Verwendet faster-whisper (CTranslate2) wenn auf ARM verfügbar,
fällt auf openai-whisper zurück.
"""

import asyncio
import io
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from loguru import logger

from app.config import get_settings


class STTService:
    """Whisper-basiertes Speech-to-Text."""

    def __init__(self):
        self._model = None
        self._settings = get_settings()
        self._backend = "unknown"

    async def initialize(self):
        """Whisper-Modell laden (in Thread, da blockierend)."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_model)

    def _load_model(self):
        """Versucht faster-whisper, fällt auf openai-whisper zurück."""
        model_name = self._settings.stt_model
        device = self._settings.stt_device

        # --- Versuch 1: faster-whisper (optimierte CTranslate2-Engine) ---
        try:
            from faster_whisper import WhisperModel

            # faster-whisper erwartet Modellnamen wie "large-v3"
            fw_model_name = model_name.replace("openai/whisper-", "")
            compute_type = "float16" if device == "cuda" else "int8"

            logger.info(
                f"Lade faster-whisper: {fw_model_name} "
                f"({device}, {compute_type})"
            )
            self._model = WhisperModel(
                fw_model_name,
                device=device,
                compute_type=compute_type,
                cpu_threads=4,
            )
            self._backend = "faster-whisper"
            logger.info("faster-whisper erfolgreich geladen.")
            return

        except Exception as e:
            logger.warning(f"faster-whisper nicht verfügbar: {e}")

        # --- Versuch 2: openai-whisper (PyTorch-basiert) ---
        try:
            import whisper

            # openai-whisper erwartet "large-v3"
            ow_model_name = model_name.replace("openai/whisper-", "")
            logger.info(f"Lade openai-whisper: {ow_model_name} ({device})")
            self._model = whisper.load_model(ow_model_name, device=device)
            self._backend = "openai-whisper"
            logger.info("openai-whisper erfolgreich geladen.")
            return

        except Exception as e:
            logger.error(f"openai-whisper fehlgeschlagen: {e}")
            raise RuntimeError(
                f"Kein Whisper-Backend verfügbar. "
                f"Installiere 'faster-whisper' oder 'openai-whisper'."
            )

    def is_ready(self) -> bool:
        return self._model is not None

    async def shutdown(self):
        """Modell freigeben."""
        self._model = None
        logger.info("STT-Service heruntergefahren.")

    async def transcribe(
        self,
        audio_bytes: bytes,
        language: str = "de",
        word_timestamps: bool = False,
    ) -> dict:
        """
        Transkribiert Audio-Bytes zu Text.

        Returns:
            dict mit 'text', 'language', 'duration', 'segments'
        """
        if self._model is None:
            raise RuntimeError("STT-Modell nicht initialisiert.")

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self._transcribe_sync(audio_bytes, language, word_timestamps),
        )
        return result

    def _transcribe_sync(
        self,
        audio_bytes: bytes,
        language: str,
        word_timestamps: bool,
    ) -> dict:
        """Synchrone Transkription (wird in Thread ausgeführt)."""

        # Audio-Bytes in temporäre Datei schreiben
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            tmp.write(audio_bytes)
            tmp.flush()
            tmp_path = tmp.name

            if self._backend == "faster-whisper":
                return self._transcribe_faster_whisper(
                    tmp_path, language, word_timestamps
                )
            else:
                return self._transcribe_openai_whisper(
                    tmp_path, language, word_timestamps
                )

    def _transcribe_faster_whisper(
        self, audio_path: str, language: str, word_timestamps: bool
    ) -> dict:
        """Transkription mit faster-whisper."""
        segments_iter, info = self._model.transcribe(
            audio_path,
            language=language,
            beam_size=5,
            word_timestamps=word_timestamps,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=500,
                speech_pad_ms=200,
            ),
        )

        segments = []
        full_text_parts = []
        for segment in segments_iter:
            seg_data = {
                "start": round(segment.start, 2),
                "end": round(segment.end, 2),
                "text": segment.text.strip(),
            }
            if word_timestamps and segment.words:
                seg_data["words"] = [
                    {
                        "word": w.word,
                        "start": round(w.start, 2),
                        "end": round(w.end, 2),
                        "probability": round(w.probability, 3),
                    }
                    for w in segment.words
                ]
            segments.append(seg_data)
            full_text_parts.append(segment.text.strip())

        return {
            "text": " ".join(full_text_parts),
            "language": info.language,
            "duration": round(info.duration, 2),
            "segments": segments,
        }

    def _transcribe_openai_whisper(
        self, audio_path: str, language: str, word_timestamps: bool
    ) -> dict:
        """Transkription mit openai-whisper."""
        result = self._model.transcribe(
            audio_path,
            language=language,
            word_timestamps=word_timestamps,
            verbose=False,
        )

        segments = []
        for seg in result.get("segments", []):
            seg_data = {
                "start": round(seg["start"], 2),
                "end": round(seg["end"], 2),
                "text": seg["text"].strip(),
            }
            if word_timestamps and "words" in seg:
                seg_data["words"] = [
                    {
                        "word": w["word"],
                        "start": round(w["start"], 2),
                        "end": round(w["end"], 2),
                        "probability": round(w.get("probability", 0), 3),
                    }
                    for w in seg["words"]
                ]
            segments.append(seg_data)

        duration = segments[-1]["end"] if segments else 0

        return {
            "text": result["text"].strip(),
            "language": result.get("language", language),
            "duration": round(duration, 2),
            "segments": segments,
        }
