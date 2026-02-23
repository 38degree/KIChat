# Deployment auf Ubuntu/DGX Spark — Häufige Fragen

## 1. Installation

### Schritt 1: Voraussetzungen
```bash
# Auf eigenem Laptop/Desktop:
# Ubuntu 20.04 LTS oder neuer, mit NVIDIA GPU

# Auf DGX Spark:
# Ubuntu ist bereits vorinstalliert
# NVIDIA GPU-Treiber ist installiert
```

### Schritt 2: Automatisches Setup
```bash
cd /path/to/NvidiaKI
chmod +x scripts/install_ubuntu.sh
./scripts/install_ubuntu.sh

# Der Script installiert alles:
# - Docker
# - NVIDIA Container Toolkit
# - Berechtigungen
# - Projekt-Struktur
```

### Schritt 3: HuggingFace Token
```bash
# Token besorgen unter https://huggingface.co/settings/tokens
1. Besuche: https://huggingface.co/settings/tokens
2. Klick: "New token"
3. Permissions: "read"
4. Copy token zu .env:

nano .env
# Ändere:
# HF_TOKEN=hf_YOUR_TOKEN_HERE
# zu:
# HF_TOKEN=hf_aBcDefGhIjKlMnOpQrStUvWxYz...
```

---

## 2. Container starten

### Quickstart
```bash
cd /path/to/NvidiaKI

# Starten
docker compose up -d

# Öffnen
# http://localhost:3000
```

### Erster Start (dauert länger)
- vLLM lädt Modell → ~2 Minuten
- TTS lädt XTTS-v2 → ~1 Minute
- Backend initialisiert → ~30s
- **Gesamt: ~5 Minuten**

### Logs verfolgen
```bash
# Alle Services
docker compose logs -f

# Nur vLLM (LLM-Modell)
docker compose logs -f vllm

# Nur Backend (Chat-API)
docker compose logs -f backend

# Nur TTS (Sprachausgabe)
docker compose logs -f tts
```

---

## 3. Häufige Fehler & Lösungen

### ❌ "docker: command not found"
```bash
# Docker ist nicht installiert
./scripts/install_ubuntu.sh
```

### ❌ "Error response from daemon: could not select device driver"
```bash
# NVIDIA Container Toolkit nicht installiert
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Testen
docker run --rm --gpus all nvidia/cuda:12.8.0-runtime-ubuntu24.04 nvidia-smi
```

### ❌ "vLLM: Out of Memory"
```bash
# GPU hat nicht genug Speicher
# Optionen:
# 1. Memory-Utilization senken:
#    In .env: LLM_GPU_MEMORY_UTILIZATION=0.40

# 2. Kleineres Modell verwenden:
#    LLM_MODEL=Qwen/Qwen2.5-32B-Instruct

# 3. vLLM nur neu starten
docker compose restart vllm

# Memory-Status prüfen
nvidia-smi
```

### ❌ "CUDA out of memory" während OCR
```bash
# OCR-Batch-Size reduzieren (in .env):
OCR_BATCH_SIZE=32
# Statt 64

docker compose restart ocr
```

### ❌ "HF_TOKEN not found" / "Authentication required"
```bash
# Token in .env nicht gesetzt
nano .env
# Sicherstellen dass HF_TOKEN=hf_xxxxx korrekt ist

docker compose restart backend
```

### ❌ "Connection refused: localhost:8000"
```bash
# vLLM startet noch oder ist gecrasht
docker compose logs -f vllm

# vLLM neu starten
docker compose restart vllm

# Warten auf Start (bis zu 5 Min.)
watch -n 1 curl http://localhost:8000/health
```

### ❌ "No space left on device"
```bash
# Kein Speatheraplatz für Modelle
df -h

# Modelle befinden sich hier:
ls -lh data/vectordb/
ls -lh models/

# Option 1: Speicher freigeben
docker system prune
rm -rf data/vectordb/*  # ⚠️  Wissensdatenbank wird gelöscht

# Option 2: Externe Festplatte nutzen
mkdir /mnt/external/nvidiaki
cd /path/to/NvidiaKI
ln -s /mnt/external/nvidiaki/data ./data_external
```

### ❌ "Torch version mismatch" / "CUDA version incompatible"
```bash
# Kann nach NVIDIA Driver-Update passieren
# Lösung: Container neu bauen
docker compose down -v
docker compose up -d --build
```

---

## 4. Performance-Tuning

### GPU-Memory optimieren
```bash
# .env anpassen:

# Aktuell (Standard für 72B):
LLM_GPU_MEMORY_UTILIZATION=0.45
LLM_TENSOR_PARALLEL_SIZE=1

# Für noch bessere Performance:
LLM_GPU_MEMORY_UTILIZATION=0.50
VLLM_ENABLE_PREFIX_CACHING=true  # RAG-Cache

# Für kleinere GPUs:
LLM_GPU_MEMORY_UTILIZATION=0.35
LLM_MAX_MODEL_LEN=4096
```

### Schnellere Antworten (Time-to-first-token)
```bash
# Temperature senken (niedrig = deterministisch)
# Im API-Call:
"temperature": 0.1  # 0.0-0.3 für medizinische Texte

# Max-Tokens begrenzen
"max_tokens": 2048  # Statt unbegrenzt
```

### Whisper schneller
```bash
# .env:
STT_DEVICE=cuda  # Sicherstellen dass GPU genutzt wird

# Oder speichern Sie cached Whisper in NVMe
mkdir -p /mnt/nvme/whisper_cache
export TRANSFORMERS_CACHE=/mnt/nvme/whisper_cache
```

### TTS schneller
```bash
# XTTS-v2 Voice-Cloning deaktivieren (wenn nicht nötig)
# Vereinfachter Code in server.py (kommt auf Anfrage)
```

---

## 5. Backup & Daten-Sicherung

### Vektordatenbank sichern
```bash
# Qdrant backing up
tar -czf backup_qdrant_$(date +%Y%m%d).tar.gz data/vectordb/

# Wiederherstellen
tar -xzf backup_qdrant_20250223.tar.gz
docker compose restart qdrant
```

### Alle Dokumente sichern
```bash
tar -czf backup_documents_$(date +%Y%m%d).tar.gz data/documents/
```

### Konfiguration sichern
```bash
cp .env backup_env_$(date +%Y%m%d).txt  # Sicher lagern!
```

### Automatisiches tägliches Backup
```bash
# backup.sh erstellen:
#!/bin/bash
BACKUP_DIR="/backups/nvidiaki"
mkdir -p "$BACKUP_DIR"
tar -czf "$BACKUP_DIR/backup_$(date +%Y%m%d).tar.gz" \
  data/vectordb/ data/documents/
# Alte Backups löschen (älter als 30 Tage)
find "$BACKUP_DIR" -name "*.tar.gz" -mtime +30 -delete

# Cronjob: crontab -e
# 0 2 * * * /home/user/backup.sh  # täglich 02:00 Uhr
```

---

## 6. Skalierung & Load-Testing

### Mit mehreren Requests gleichzeitig testen
```bash
# Parallel 5 Anfragen (concurrent)
for i in {1..5}; do
  curl -X POST http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"default","messages":[{"role":"user","content":"Frage '$i'"}]}' &
done
wait

# Monitoring dabei
docker stats --no-stream
```

### vLLM Quantisierung testen
```bash
# Mit FP4 vs. FP8 vs. FP16 vergleichen
./scripts/evaluate_quantization.sh
```

---

## 7. Sicherheit

### Firewall/Netzwerk
```bash
# Ports nur lokal accessible machen (sicher für Klinik-Netzwerk)
# In docker-compose.yml Ports ändern:
# Statt:    ports: ["3000:8080"]
# Nutze:    ports: ["127.0.0.1:3000:8080"]

# Nur localhost kann zugreifen:
curl http://localhost:3000  # ✓ funktioniert
curl http://192.168.1.100:3000  # ✗ funktioniert nicht
```

### HTTPS mit Reverse Proxy (Nginx)
```bash
# Nginx installieren
sudo apt-get install nginx

# Config (vereinfacht):
# /etc/nginx/sites-available/nvidiaki
server {
    listen 443 ssl;
    server_name medial.clinic.local;
    
    ssl_certificate /etc/ssl/certs/cert.pem;
    ssl_certificate_key /etc/ssl/private/key.pem;
    
    location / {
        proxy_pass http://localhost:3000;
    }
}

sudo systemctl restart nginx
# Zugriff: https://medial.clinic.local
```

### Authentifizierung
```bash
# Open WebUI hat keine Built-in-Auth
# Option: Nginx Basic Auth
# oder: Reverse Proxy mit OAuth2 (external)
```

---

## 8. Monitoring & Logging

### Container-Status
```bash
# Live-Übersicht
docker compose ps
# oder mit Monitoring-Tool
docker stats

# Nur ein Service
docker compose ps backend
```

### Performance-Metriken
```bash
# GPU-Auslastung
nvidia-smi -l 1

# vLLM Inferenz-Speed
docker compose logs -f vllm | grep "throughput"

# Backend API Response-Zeit
docker compose logs -f backend | grep "duration"
```

### Zentrales Logging (optional)
```bash
# Mit ELK-Stack oder Loki (komplexer Setup)
# Für jetzt: einfache Docker-Logs ausreichend
```

---

## 9. Wartung

### Regelmäßige Aufgaben

**Täglich:**
- Logs prüfen auf Fehler: `docker compose logs --tail=200 backend`
- GPU-Speicher check: `nvidia-smi`

**Wöchentlich:**
- Updates prüfen: `docker compose pull`
- Backups prüfen: `ls -lh backup_*`

**Monatlich:**
- Alte Logs aufräumen
- Performance-Evaluation durchführen
- Speicherverbrauch prüfen

### Container Update
```bash
# Neue Images herunterladen
docker compose pull

# Mit neuesten Images starten
docker compose up -d

# Oder nur einen Service:
docker compose pull vllm
docker compose up -d vllm
```

---

## 10. Support & Debugging

### Logs mit mehr Details
```bash
# Alle Logs speichern
docker compose logs > logs_$(date +%Y%m%d_%H%M%S).txt

# Für Debugging dem Entwickler schicken (ohne Tokens!)
cat logs_*.txt | grep -v "hf_" > logs_sanitized.txt
```

### Stack-Trace
```bash
# Wenn ein Service crasht:
docker compose logs <service> | head -100
docker compose logs <service> | tail -50
```

### Container in den Debug-Modus
```bash
docker compose exec backend bash
# Jetzt kann man im Container commands testen
python3 -c "import torch; print(torch.cuda.is_available())"
```

---

## Noch Fragen?

- **Dokumentation**: Siehe [README.md](README.md)
- **API-Docs**: http://localhost:8080/docs (Swagger UI)
- **Logs**: `docker compose logs -f`
