# Deployment-Checkliste für Ubuntu/DGX Spark

## Phase 1: Hardware-Vorbereitung ☐

- [ ] Ubuntu 20.04 LTS oder neuer installiert
- [ ] NVIDIA GPU Driver installiert (`nvidia-smi` funktioniert)
- [ ] Internet-Verbindung verfügbar
- [ ] ~200GB freier Speicher (Modelle + Daten)
- [ ] RAM: mind. 16GB, besser 32GB+

## Phase 2: Docker Setup ☐

```bash
chmod +x scripts/install_ubuntu.sh
./scripts/install_ubuntu.sh
```

Nach Script:
- [ ] Docker installiert (`docker --version`)
- [ ] NVIDIA Container Toolkit installiert
- [ ] GPU-Zugang funktioniert (`docker run --rm --gpus all ...`)
- [ ] Benutzer zur docker-Gruppe hinzugefügt (muss ab/anmelden)

## Phase 3: HuggingFace Token ☐

- [ ] Token generiert auf https://huggingface.co/settings/tokens
- [ ] `.env` Datei mit Token gefüllt
- [ ] Token im Format `hf_xxxxx` (nicht `your_token_here`)

```bash
# Prüfen:
grep "HF_TOKEN=hf_" .env
```

## Phase 4: Projekt-Setup ☐

```bash
# Im Projekt-Verzeichnis:
cd /path/to/NvidiaKI

# (Wird automatisch durch install_ubuntu.sh gemacht:)
mkdir -p models/{llm,whisper,tts,embedding,ocr,denoiser}
mkdir -p data/{vectordb,documents,audio,transcripts,webui,calibration}
chmod +x scripts/*.sh
```

- [ ] Verzeichnisse vorhanden
- [ ] Scripts ausführbar

## Phase 5: Optional — LLM Quantisierung  ☐

_(Skip wenn keine Quantisierung nötig; kostet ~1-2h)_

```bash
./scripts/quantize_model.sh
./scripts/evaluate_quantization.sh
```

- [ ] Quantisierung abgeschlossen
- [ ] Evaluation durchgeführt
- [ ] Qualität akzeptabel

## Phase 6: Container starten ☐

```bash
docker compose up -d
```

Monitoring beim Start:
```bash
# In anderem Terminal:
docker compose logs -f
```

**Wartezeiten (ca.):**
- vLLM startet → 2-3 Min
- Backend initialisiert → 1 Min
- TTS lädt → 1 Min
- Total: ~5 Min

Zeichen dass bereit:
```
vllm        | INFO:     Application startup complete
backend     | INFO: Application startup complete
tts         | INFO: Application startup complete
```

- [ ] Alle Services starten ohne Fehler
- [ ] Keine "Out of Memory" Fehler
- [ ] Keine "Connection refused" Fehler

## Phase 7: Health Checks ☐

```bash
# Status aller Services
docker compose ps

# Health Details
curl http://localhost:8080/health/detail | jq '.'

# Einzeln:
curl http://localhost:3000/    # Open WebUI
curl http://localhost:8000/health  # vLLM
curl http://localhost:6333/health  # Qdrant
curl http://localhost:8001/health  # TTS
```

Alle sollten Status "ok" zurückgeben.

- [ ] Open WebUI lädt (http://localhost:3000)
- [ ] vLLM antwortet
- [ ] Backend antwortet
- [ ] Qdrant antwortet
- [ ] TTS antwortet

## Phase 8: Funktions-Test ☐

### Test 1: Chat
```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "Hallo"}]
  }'
```
- [ ] Antwort erhalten (ca. 5-15s)

### Test 2: RAG (ohne Dokumente)
```bash
curl -X POST "http://localhost:8080/api/rag/search?query=test&top_k=1"
```
- [ ] Keine Fehler (auch wenn keine Dokumente indiziert)

### Test 3: Whisper STT
```bash
# Einfache Testdatei generieren
echo "test audio" | ffmpeg -f lavfi -i anullsrc=r=16000:cl=mono -t 1 -q:a 9 test.wav

curl -X POST http://localhost:8080/v1/audio/transcriptions \
  -F "file=@test.wav" \
  -F "language=de"
```
- [ ] Transkription liefert (evtl. "test audio" oder ähnlich)

### Test 4: TTS
```bash
curl -X POST http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hallo", "voice": "default"}' \
  --output test.wav

# Audio abspielen
aplay test.wav  # oder ffplay
```
- [ ] Audio-Datei generiert
- [ ] Audio abspielen funktioniert

## Phase 9: Produktivbetrieb ✓

### Backups aktivieren
- [ ] Tägliches Backup der Vektordatenbank einrichten
- [ ] `.env` und Konfiguration sichern

### Monitoring
- [ ] Cron-Job für Logs-Rotation
- [ ] Regelmäßige GPU-Speicher Prüfung

### Sicherheit (optional)
- [ ] Nginx-Reverse-Proxy für HTTPS konfigurieren
- [ ] Firewall auf nur localhost beschränken

### Dokumentation
- [ ] Notizen über Customization
- [ ] Kontakt-Info für Support

## Phase 10: Laufender Betrieb ✓

**Täglich:**
```bash
# Status prüfen
docker compose ps

# Logs auf Fehler prüfen
docker compose logs --tail=100 backend
```

**Wöchentlich:**
```bash
# Backups prüfen
ls -lh data/vectordb/

# GPU-Speicher
nvidia-smi
```

**Nach Updates:**
```bash
docker compose pull
docker compose up -d --build
docker compose ps  # alles ok?
```

---

## Troubleshooting-Reference

| Problem | Lösung |
|---------|--------|
| vLLM startet nicht | `docker compose logs -f vllm` ansehen, 5 Min warten |
| Out of Memory | LLM_GPU_MEMORY_UTILIZATION senken oder kleineres Modell |
| HF_TOKEN Fehler | Token in `.env` prüfen, Typo? |
| Port bereits in Verwendung | `lsof -i :3000` prüfen, andere App stoppen |
| "Connection refused" | Service startet noch, `docker compose ps` prüfen |
| Slow Responses | GPU-speicher prüfen (`nvidia-smi`), andere Processes beenden |

---

## Erfolg-Kriterien ✅

- [ ] Web-Interface lädt unter http://localhost:3000
- [ ] Chat funktioniert (Antwort in <20s)
- [ ] Audio-Upload funktioniert
- [ ] Keine Fehler in Logs nach 5 Minuten Start
- [ ] GPU wird genutzt (nvidia-smi zeigt >1% Nutzung)

**Wenn alle Haken drin: Deployment erfolgreich! 🎉**

---

## Nächste Schritte

1. **Daten-Ingestion**: Gerichtsakten als PDFs hochladen
   ```bash
   curl -X POST http://localhost:8080/api/documents/upload \
     -F "file=@gerichtsakte.pdf" \
     -F "document_type=gerichtsakte"
   ```

2. **Voice Cloning** (optional): Referenz-Audio hochladen
   ```bash
   curl -X POST http://localhost:8001/clone-voice \
     -F "file=@stimme.wav" -F "name=default"
   ```

3. **Integration**: Open WebUI in Klinik-Browser bookmarken
   - Standard-URL: http://192.168.X.X:3000 (vom Host/IP)

4. **Monitoring**: Regelmäßige Wartung planen
