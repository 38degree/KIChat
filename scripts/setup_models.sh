#!/bin/bash
# ============================================================
# Modell-Download und Setup
# ============================================================
#
# Lädt alle benötigten Modelle für die Plattform herunter.
#
# Verwendung:
#   export HF_TOKEN=your_token
#   ./scripts/setup_models.sh
# ============================================================

set -euo pipefail

HF_TOKEN="${HF_TOKEN:?Bitte HF_TOKEN setzen (export HF_TOKEN=your_token)}"

echo "============================================="
echo "  Modell-Download & Setup"
echo "============================================="

# --- Verzeichnisse erstellen ---
echo "[1/6] Erstelle Verzeichnisstruktur..."
mkdir -p models/{llm,whisper,tts,embedding,ocr,denoiser}
mkdir -p data/{vectordb,documents,audio,transcripts,webui,calibration}
mkdir -p models/tts/reference

# --- Embedding-Modell ---
echo ""
echo "[2/6] Lade Embedding-Modell (multilingual-e5-large)..."
python3 -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('intfloat/multilingual-e5-large')
model.save('models/embedding/multilingual-e5-large')
print('Embedding-Modell gespeichert.')
" 2>/dev/null || echo "WARNUNG: Embedding-Download wird beim ersten Start automatisch durchgeführt."

# --- Whisper ---
echo ""
echo "[3/6] Whisper large-v3 wird beim ersten Start automatisch geladen."
echo "       (ca. 3GB, gespeichert in HuggingFace Cache)"

# --- TTS ---
echo ""
echo "[4/6] TTS (XTTS-v2) wird beim ersten Start automatisch geladen."
echo "       (ca. 2GB, gespeichert in HuggingFace Cache)"
echo ""
echo "TIPP: Für Voice-Cloning ein 6-10s WAV mit klarer Stimme bereitstellen:"
echo "  cp meine_stimme.wav models/tts/reference/default.wav"

# --- OCR ---
echo ""
echo "[5/6] OCR (Surya + marker-pdf) Modelle werden beim ersten Start geladen."
echo "       (ca. 2-3GB)"

# --- Denoiser ---
echo ""
echo "[6/6] Denoiser (Resemble Enhance) Modelle werden beim ersten Start geladen."
echo "       (ca. 500MB)"

echo ""
echo "============================================="
echo "  Setup abgeschlossen!"
echo "============================================="
echo ""
echo "Wichtige nächste Schritte:"
echo ""
echo "1. HuggingFace Token in .env eintragen:"
echo "   HF_TOKEN=${HF_TOKEN:0:10}..."
echo ""
echo "2. Optional: LLM quantisieren (empfohlen für 72B Modell):"
echo "   ./scripts/quantize_model.sh"
echo ""
echo "3. Plattform starten:"
echo "   docker compose up -d"
echo ""
echo "4. Open WebUI öffnen:"
echo "   http://localhost:3000"
echo ""
echo "5. Voice-Clone Referenz hochladen (optional):"
echo "   curl -X POST http://localhost:8001/clone-voice \\"
echo "     -F 'file=@meine_stimme.wav' -F 'name=default'"
