#!/usr/bin/env bash
# ============================================================
# Schnellstart-Guide und nützliche Commands
# ============================================================
#
# Dieser Guide enthält alle wichtigsten Commands für
# den täglichen Betrieb der psychiatrischen KI-Plattform
# auf Ubuntu/DGX Spark.

# ==========================================================
# 0. INITIAL SETUP (einmalig)
# ==========================================================

# 1a. Alle Voraussetzungen installieren
chmod +x scripts/install_ubuntu.sh
./scripts/install_ubuntu.sh

# 1b. Oder manuell: HuggingFace Token in .env setzen
cp .env.example .env
nano .env  # HF_TOKEN=hf_xxxxx eintragen

# 1c. Optional: 72B Modell quantisieren (45-90 Minuten)
export HF_TOKEN=hf_xxxxx
./scripts/quantize_model.sh

# 1d. Quantisierung evaluieren
./scripts/evaluate_quantization.sh

# ==========================================================
# 1. CONTAINER STARTEN / STOPPEN
# ==========================================================

# Alle Services starten
docker compose up -d

# Im Hintergrund starten mit Output
docker compose up

# Services stoppen
docker compose down

# Services stoppen und Volumes löschen (ACHTUNG: Daten weg!)
docker compose down -v

# Neu bauen und starten (nach Code-Änderungen)
docker compose up -d --build

# Nur einen Service neu starten
docker compose restart backend
docker compose restart vllm
docker compose restart tts

# ==========================================================
# 2. LOGS UND TROUBLESHOOTING
# ==========================================================

# Logs aller Services anschauen
docker compose logs

# Live-Logs für einen Service
docker compose logs -f backend
docker compose logs -f vllm
docker compose logs -f qdrant

# Letzte 100 Zeilen eines Services
docker compose logs --tail=100 backend

# Status aller Container
docker compose ps

# Detaillierte Info zu einem Container
docker compose ps backend

# In einen Container reinschauen (Bash)
docker compose exec backend bash
docker compose exec vllm bash

# GPU-Speicher anschauen (von Host)
nvidia-smi
nvidia-smi -l 1  # Jede Sekunde aktualisieren
nvidia-smi dmon  # Verbesserte Echtzeit-Anzeige

# Memory-Auslastung allgemein
free -h
df -h

# ==========================================================
# 3. HEALTH CHECKS
# ==========================================================

# Alle Services prüfen
curl http://localhost:8080/health/detail | jq '.'

# Einzelne Services
curl http://localhost:8000/health                    # vLLM
curl http://localhost:8080/health                   # Backend
curl http://localhost:6333/health                   # Qdrant
curl http://localhost:8001/health                   # TTS
curl http://localhost:8002/health                   # Denoiser
curl http://localhost:8003/health                   # OCR

# vLLM Modell-Info
curl http://localhost:8000/v1/models | jq '.'

# Qdrant Collection-Info
curl http://localhost:6333/api/rag/stats | jq '.'

# ==========================================================
# 4. API BEISPIELE (ChatGPT-ähnliche Nutzung)
# ==========================================================

# 4a. Einfacher Chat ohne RAG
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen2.5-72B-Instruct",
    "messages": [
      {
        "role": "user",
        "content": "Erkläre die Diagnosekriterien für Depression nach ICD-11"
      }
    ],
    "temperature": 0.3,
    "max_tokens": 1024
  }' | jq '.choices[0].message.content'

# 4b. Multi-turn Conversation (mehrere Nachrichten)
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen2.5-72B-Instruct",
    "messages": [
      {"role": "user", "content": "Erkläre Lithium-Therapie"},
      {"role": "assistant", "content": "Lithium ist ein Stimmungsstabilisator..."},
      {"role": "user", "content": "Welche Nebenwirkungen?"}
    ],
    "temperature": 0.3
  }' | jq '.'

# 4c. Streaming-Response (Live-Antwort während Generation)
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen2.5-72B-Instruct",
    "messages": [{"role": "user", "content": "Was ist Schizophrenie?"}],
    "stream": true
  }'

# ==========================================================
# 5. AUDIO (STT, TTS, DENOISING)
# ==========================================================

# 5a. Audio transkribieren (kurz, <30s)
curl -X POST http://localhost:8080/v1/audio/transcriptions \
  -F "file=@aufnahme.wav" \
  -F "language=de" | jq '.text'

# 5b. Lange Aufnahmen transkribieren (mit Denoise + Patient ID)
curl -X POST http://localhost:8080/v1/audio/transcribe-long \
  -F "file=@patientengespraech.wav" \
  -F "language=de" \
  -F "denoise=true" \
  -F "patient_id=P12345" | jq '.text'

# 5c. Audio bereinigen (Renoising)
curl -X POST http://localhost:8080/v1/audio/denoise \
  -F "file=@noisy_audio.wav" \
  -F "enhance=true" \
  --output denoised_audio.wav

# 5d. Text to Speech (Sprachausgabe)
curl -X POST http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Guten Tag, wie kann ich Ihnen heute helfen?",
    "voice": "default",
    "response_format": "wav"
  }' --output antwort.wav

# Audio mit ffplay abspielen
ffplay antwort.wav

# ==========================================================
# 6. DOKUMENTE & RAG (PDFs, Gerichtsakten)
# ==========================================================

# 6a. Gerichtsakte hochladen
curl -X POST http://localhost:8080/api/documents/upload \
  -F "file=@gerichtsakte.pdf" \
  -F "document_type=gerichtsakte" \
  -F "patient_id=P12345" \
  -F "case_number=Az. 123/2025" | jq '.'

# 6b. Alle indexierten Dokumente auflisten
curl http://localhost:8080/api/documents | jq '.documents'

# 6c. Direkte Suche in der Wissensdatenbank
curl -X POST "http://localhost:8080/api/rag/search?query=Lithium+Dosierung&top_k=5" | jq '.results'

# 6d. RAG-Statistiken
curl http://localhost:8080/api/rag/stats | jq '.'

# 6e. Dokument löschen
curl -X DELETE http://localhost:8080/api/documents/DOCUMENT_ID | jq '.'

# ==========================================================
# 7. VOICE CLONING (TTS persistente Stimme)
# ==========================================================

# 7a. Referenz-Audio hochladen (6-10s klare Sprache)
curl -X POST http://localhost:8001/clone-voice \
  -F "file=@meine_stimme.wav" \
  -F "name=default" | jq '.'

# 7b. Verfügbare Stimmen anzeigen
curl http://localhost:8001/voices | jq '.'

# ==========================================================
# 8. ENTWICKLUNG & DEBUGGING
# ==========================================================

# Interaktive Shell in Backend-Container
docker compose exec backend bash

# Python-Packages installieren (im Container)
docker compose exec backend pip install paket_name

# Code editieren (local, wird automatisch gemountet)
nano services/backend/app/routes/chat.py

# Backend dann neu starten damit Änderungen wirksam werden
docker compose restart backend

# Logs nur nach Neustart anschauen
docker compose logs -f --tail=50 backend

# ==========================================================
# 9. PERFORMANCE & MONITORING
# ============================================================

# Real-time GPU-Monitoring
watch -n 0.5 'nvidia-smi'

# Alle Prozesse auf GPU
nvidia-smi pmon

# CPU + Memory auf DGX Spark (ARM)
top
htop  # wenn installiert

# Docker-Container CPU/Memory
docker stats

# Container-Details
docker inspect psych-vllm

# ==========================================================
# 10. WARTUNG & UPDATES
# ==========================================================

# Alle Images herunterladen (um sicherzustellen)
docker compose pull

# Alte/ungenutzte Images löschen
docker image prune -a

# Volumes bereinigen (ACHTUNG!)
docker volume prune

# Komplette Neuinstallation
docker compose down -v
docker image prune -a
docker compose up -d --build

# ==========================================================
# 11. BACKUPS ERSTELLEN
# ==========================================================

# Qdrant Vektordatenbank sichern
tar -czf backup_qdrant_$(date +%Y%m%d).tar.gz data/vectordb/

# Alle Dokumente sichern
tar -czf backup_documents_$(date +%Y%m%d).tar.gz data/documents/

# Konfiguration sichern
cp .env backup_config_$(date +%Y%m%d).env

# Komplettes Backup (alle Docker Volumes)
docker compose down
tar -czf backup_complete_$(date +%Y%m%d).tar.gz models/ data/
docker compose up -d

# ==========================================================
# 12. OPENAI-KOMPATIBLE CLIENTS
# ==========================================================

# Python (gegen lokale API)
python3 << 'PYTHON_EOF'
from openai import OpenAI

client = OpenAI(
    api_key="sk-placeholder",
    base_url="http://localhost:8080/v1"
)

response = client.chat.completions.create(
    model="Qwen2.5-72B-Instruct",
    messages=[
        {"role": "user", "content": "Was ist ICD-11?"}
    ]
)

print(response.choices[0].message.content)
PYTHON_EOF

# Node.js
npm install openai
# In JavaScript:
# const OpenAI = require('openai');
# const client = new OpenAI({
#   apiKey: 'sk-placeholder',
#   baseURL: 'http://localhost:8080/v1'
# });

# ==========================================================
# 13. TIPPS & TRICKS
# ==========================================================

# Ambiente speichern für schnellen Zugriff
source ~/.bashrc
alias psych='cd /path/to/NvidiaKI && docker compose'

# Seiteneffekt freigeben nach langen Inferenzen
docker exec psych-vllm nvidia-smi  # GPU-Status
docker compose restart vllm         # Service zurücksetzen

# vLLM mit verbessertem Logging starten
docker compose exec vllm \
  python3 -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-72B-Instruct \
  --verbose

# ==========================================================
