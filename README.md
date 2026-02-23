# Psychiatrische KI-Plattform

Lokale, DSGVO-konforme KI-Plattform für psychiatrische Kliniken auf **NVIDIA DGX Spark** (GB10 Blackwell, 128GB Unified Memory).

## Architektur

```
┌─────────────────────────────────────────────────────┐
│            NVIDIA DGX Spark (128GB)                 │
│                                                     │
│  ┌──────────┐   ┌────────────┐   ┌──────────────┐  │
│  │ Open     │──▶│  Backend   │──▶│ vLLM         │  │
│  │ WebUI    │   │  (FastAPI) │   │ Qwen 72B FP4 │  │
│  │ :3000    │   │  :8080     │   │ :8000        │  │
│  └──────────┘   └─────┬──────┘   └──────────────┘  │
│                       │                              │
│         ┌─────────────┼─────────────┐               │
│         ▼             ▼             ▼               │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐          │
│  │ Qdrant   │  │ TTS      │  │ Denoiser │          │
│  │ VectorDB │  │ XTTS-v2  │  │ Resemble │          │
│  │ :6333    │  │ :8001    │  │ :8002    │          │
│  └──────────┘  └──────────┘  └──────────┘          │
│                                                     │
│  ┌──────────┐  ┌────────────────────────┐           │
│  │ OCR      │  │  Whisper large-v3      │           │
│  │ Surya    │  │  (im Backend integriert)│          │
│  │ :8003    │  └────────────────────────┘           │
│  └──────────┘                                       │
└─────────────────────────────────────────────────────┘
```

## Komponenten

| Service | Modell | Funktion | GPU-RAM |
|---------|--------|----------|---------|
| **LLM** | Qwen 2.5 72B-Instruct (NVFP4) | Reasoning, Chat, Diagnostik | ~40-44 GB |
| **STT** | Whisper large-v3 | Spracheingabe, Transkription | ~3-4 GB |
| **TTS** | XTTS-v2 | Sprachausgabe (Deutsch) | ~2-3 GB |
| **Embedding** | multilingual-e5-large | Vektorisierung für RAG | ~1.5 GB (CPU) |
| **VectorDB** | Qdrant | RAG Wissensdatenbank | ~1-2 GB |
| **Denoiser** | Resemble Enhance | Audio-Bereinigung | ~1-2 GB |
| **OCR** | Surya + marker-pdf | Gerichtsakten-Verarbeitung | ~4-6 GB |
| **Frontend** | Open WebUI | Chat-Oberfläche | CPU only |
| **Gesamt** | | | **~55-65 GB / 128 GB** |

## Schnellstart

### 1. Voraussetzungen

- NVIDIA DGX Spark mit Ubuntu
- Docker + NVIDIA Container Toolkit
- HuggingFace Account & Token

### 2. Setup

```bash
# Repository klonen
cd /pfad/zu/NvidiaKI

# HuggingFace Token setzen
cp .env.example .env  # Oder .env direkt editieren
# HF_TOKEN=hf_xxxxx eintragen

# Verzeichnisse + optionale Modelle vorbereiten
chmod +x scripts/*.sh
./scripts/setup_models.sh
```

### 3. (Optional) LLM quantisieren

```bash
# NVFP4 Quantisierung des 72B Modells (~60-90 Min.)
./scripts/quantize_model.sh

# Quantisierung evaluieren
./scripts/evaluate_quantization.sh
```

### 4. Starten

```bash
# Alle Services starten
docker compose up -d

# Logs verfolgen
docker compose logs -f

# Status prüfen
docker compose ps
```

### 5. Verwenden

- **Chat-Oberfläche**: http://localhost:3000
- **API-Dokumentation**: http://localhost:8080/docs
- **Qdrant Dashboard**: http://localhost:6333/dashboard

## API-Endpunkte

### Chat (OpenAI-kompatibel)
```bash
# Chat mit RAG
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen2.5-72B-Instruct",
    "messages": [{"role": "user", "content": "Diagnosekriterien für Schizophrenie nach ICD-11?"}]
  }'
```

### Spracheingabe (STT)
```bash
# Audio transkribieren
curl -X POST http://localhost:8080/v1/audio/transcriptions \
  -F "file=@aufnahme.wav" \
  -F "language=de"
```

### Lange Aufnahmen transkribieren
```bash
# Mit Audio-Bereinigung + Patienten-ID
curl -X POST http://localhost:8080/v1/audio/transcribe-long \
  -F "file=@patientengespraech.wav" \
  -F "language=de" \
  -F "denoise=true" \
  -F "patient_id=P12345"
```

### Sprachausgabe (TTS)
```bash
# Text zu Sprache
curl -X POST http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Guten Tag, wie kann ich Ihnen helfen?", "voice": "default"}' \
  --output antwort.wav
```

### PDF/OCR Verarbeitung
```bash
# Gerichtsakte hochladen und indexieren
curl -X POST http://localhost:8080/api/documents/upload \
  -F "file=@gerichtsakte.pdf" \
  -F "document_type=gerichtsakte" \
  -F "patient_id=P12345" \
  -F "case_number=Az. 123/2025"
```

### RAG-Verwaltung
```bash
# Dokument-Liste
curl http://localhost:8080/api/documents

# Direkte Suche in Wissensdatenbank
curl -X POST "http://localhost:8080/api/rag/search?query=Lithium+Dosierung&top_k=5"

# Statistiken
curl http://localhost:8080/api/rag/stats
```

## NVFP4-Quantisierung

### Vorteile
- **~3.5x weniger Speicher** vs. FP16 → 72B Modell passt in ~40GB
- **Native Blackwell Tensor Core** Unterstützung (kein Software-Overhead)
- **<1% Genauigkeitsverlust** in Standard-Benchmarks
- Höherer Throughput durch geringere Memory-Bandbreite

### Nachteile / Risiken
- Psychiatrische Fachterminologie könnte stärker unter Quantisierung leiden
- Kalibrierung mit domänenspezifischen Daten empfohlen
- Nicht auf ältere GPU-Generationen portabel
- Attention-Layer sind empfindlicher → Mixed FP4/FP8 für kritische Layer möglich

### Evaluierung
Das Evaluations-Script (`scripts/evaluate_quantization.sh`) testet:
- 10 psychiatrische Fachfragen (Diagnostik, Forensik, Pharmakologie, etc.)
- Fachbegriff-Abdeckung (automatisch)
- Halluzinations-Erkennung (manuell zu prüfen)
- Latenz-Messung

## Inferenz-Optimierungen

| Optimierung | Technik | Effekt |
|-------------|---------|--------|
| NVFP4 | TensorRT Model Optimizer | 3.5x weniger Memory |
| PagedAttention | vLLM built-in | Effizientes KV-Cache |
| Continuous Batching | vLLM built-in | Parallele Requests |
| Prefix Caching | `--enable-prefix-caching` | RAG-Prompt gecacht |
| Embedding auf CPU | Grace ARM CPU | GPU entlastet |
| Qdrant INT8 | Scalar Quantization | 4x weniger RAM für Vektoren |
| On-Demand Loading | OCR/Denoiser lazy | GPU-Memory freigeben |

## Verzeichnisstruktur

```
NvidiaKI/
├── docker-compose.yml          # Service-Orchestrierung
├── .env                        # Konfiguration (Tokens, Ports, Modelle)
├── services/
│   ├── backend/                # FastAPI Backend (Chat, RAG, STT-Routing)
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   └── app/
│   │       ├── main.py         # App-Einstieg, Lifespan
│   │       ├── config.py       # Zentrale Settings
│   │       ├── routes/         # API-Endpunkte
│   │       │   ├── chat.py     # /v1/chat/completions (OpenAI-API)
│   │       │   ├── audio.py    # /v1/audio/* (STT, TTS)
│   │       │   ├── documents.py # /api/documents/* (PDF/OCR)
│   │       │   ├── rag.py      # /api/rag/* (Suche, Stats)
│   │       │   └── health.py   # /health
│   │       ├── rag/            # RAG-Pipeline
│   │       │   ├── embedding.py    # multilingual-e5-large
│   │       │   ├── vectorstore.py  # Qdrant-Integration
│   │       │   └── chunking.py     # Text-Chunking
│   │       └── audio/          # Audio-Verarbeitung
│   │           └── stt.py      # Whisper STT
│   ├── tts/                    # XTTS-v2 Sprachausgabe
│   ├── denoiser/               # Resemble Enhance Audio-Bereinigung
│   ├── ocr/                    # Surya + marker-pdf OCR
│   └── qdrant/                 # Qdrant-Konfiguration
├── scripts/
│   ├── setup_models.sh         # Modell-Download
│   ├── quantize_model.sh       # NVFP4-Quantisierung
│   └── evaluate_quantization.sh # Evaluierung
├── models/                     # Modell-Dateien (git-ignoriert)
└── data/                       # Persistente Daten (git-ignoriert)
```

## Datenschutz

- **Alle Daten bleiben lokal** — kein Cloud-Kontakt
- **DSGVO-konform** — keine Datenübertragung an Dritte
- Audio-Dateien werden nach Verarbeitung temporär gelöscht
- Modelle laufen auf der lokalen Hardware

## Troubleshooting

### vLLM startet nicht / Out of Memory
```bash
# GPU-Speicher prüfen
nvidia-smi  # oder tegrastats auf DGX Spark

# GPU-Memory-Utilization reduzieren (in .env)
LLM_GPU_MEMORY_UTILIZATION=0.40

# Oder kleineres Modell verwenden
LLM_MODEL=Qwen/Qwen2.5-32B-Instruct
```

### OCR-Ergebnisse schlecht
```bash
# Batch-Size reduzieren für bessere Qualität
OCR_BATCH_SIZE=32
```

### Audio-Bereinigung langsam
```bash
# Enhance deaktivieren (nur Denoise)
# In der API: enhance=false
```

## Lizenz

Dieses Projekt nutzt Open-Source-Komponenten mit verschiedenen Lizenzen:
- Qwen 2.5: Apache 2.0
- Whisper: MIT
- XTTS-v2: MPL-2.0
- Surya OCR: GPL-3.0
- Qdrant: Apache 2.0
- Resemble Enhance: MIT
- Open WebUI: MIT
