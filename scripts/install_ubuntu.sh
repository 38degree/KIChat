#!/usr/bin/env bash
# ============================================================
# Installation und Setup auf Ubuntu/DGX Spark
# ============================================================
#
# Dieser Script installiert alle Voraussetzungen und startet
# die psychiatrische KI-Plattform auf einem Ubuntu-System.
#
# Ausgeglichene Schritte:
#   1. System-Update
#   2. Docker Installation
#   3. NVIDIA Container Toolkit
#   4. Berechtigungen
#   5. Konfiguration
#   6. Start
#
# Verwendung:
#   chmod +x scripts/install_ubuntu.sh
#   ./scripts/install_ubuntu.sh
# ============================================================

set -euo pipefail

# Farben für Output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Funktionen
print_header() {
    echo -e "\n${GREEN}========================================${NC}"
    echo -e "${GREEN}  $1${NC}"
    echo -e "${GREEN}========================================${NC}\n"
}

print_step() {
    echo -e "${YELLOW}[$(date +%H:%M:%S)]${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

check_command() {
    if command -v "$1" &> /dev/null; then
        print_success "$1 bereits installiert"
        return 0
    else
        return 1
    fi
}

# ============================================================
# [ 1 ] System-Update
# ============================================================
print_header "1. System-Update"

print_step "Aktualisiere Package-Listen..."
sudo apt-get update

print_step "Installiere Basis-Tools..."
sudo apt-get install -y --no-install-recommends \
    curl \
    wget \
    ca-certificates \
    gnupg \
    lsb-release \
    apt-transport-https \
    software-properties-common

print_success "System aktualisiert"

# ============================================================
# [ 2 ] Docker Installation
# ============================================================
print_header "2. Docker Installation"

if check_command docker; then
    DOCKER_VERSION=$(docker --version)
    print_step "Docker-Version: $DOCKER_VERSION"
else
    print_step "Installiere Docker..."
    
    # Docker Repository hinzufügen
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
        sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
    
    echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] \
        https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | \
        sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    
    # Docker installieren
    sudo apt-get update
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
    
    print_success "Docker installiert: $(docker --version)"
fi

# ============================================================
# [ 3 ] NVIDIA Container Toolkit
# ============================================================
print_header "3. NVIDIA Container Toolkit"

if check_command nvidia-smi; then
    print_step "NVIDIA GPU gefunden:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
else
    print_error "nvidia-smi nicht gefunden - GPU Treiber installieren?"
    exit 1
fi

if check_command nvidia-container-runtime; then
    print_step "NVIDIA Container Toolkit bereits installiert"
else
    print_step "Installiere NVIDIA Container Toolkit..."
    
    # Repository hinzufügen
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | \
        sudo apt-key add -
    
    curl -s -L https://nvidia.github.io/nvidia-docker/ubuntu20.04/nvidia-docker.list | \
        sudo tee /etc/apt/sources.list.d/nvidia-docker.list
    
    # Aktualisieren und installieren
    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
    
    print_success "NVIDIA Container Toolkit installiert"
fi

# GPU-Zugang testen
print_step "Teste GPU-Zugang via Docker..."
docker run --rm --gpus all nvidia/cuda:12.8.0-runtime-ubuntu24.04 nvidia-smi || {
    print_error "GPU-Zugang fehlgeschlagen!"
    exit 1
}
print_success "GPU-Zugang funktioniert"

# ============================================================
# [ 4 ] Berechtigungen
# ============================================================
print_header "4. Berechtigungen konfigurieren"

print_step "Füge Benutzer zur docker-Gruppe hinzu..."
sudo usermod -aG docker "$USER"
print_step "WICHTIG: Melde dich ab und wieder an, damit Änderungen wirksam werden!"
print_step "Oder nutze: newgrp docker"

# ============================================================
# [ 5 ] Projekt-Setup
# ============================================================
print_header "5. Projekt-Konfiguration"

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
print_step "Arbeitsverzeichnis: $PROJECT_DIR"

# .env vorbereiten
if [ ! -f "$PROJECT_DIR/.env" ]; then
    print_step "Erstelle .env aus .env.example..."
    cp "$PROJECT_DIR/.env.example" "$PROJECT_DIR/.env"
    print_step "WICHTIG: Bearbeite .env und setze HF_TOKEN"
    print_step "Datei: $PROJECT_DIR/.env"
fi

# Verzeichnisse erstellen
print_step "Erstelle Daten-Verzeichnisse..."
mkdir -p "$PROJECT_DIR/models"/{llm,whisper,tts,embedding,ocr,denoiser}
mkdir -p "$PROJECT_DIR/data"/{vectordb,documents,audio,transcripts,webui,calibration}
mkdir -p "$PROJECT_DIR/models/tts/reference"

# Scripts ausführbar machen
chmod +x "$PROJECT_DIR/scripts"/*.sh

print_success "Projekt-Verzeichnisse erstellt"

# ============================================================
# [ 6 ] HuggingFace Token
# ============================================================
print_header "6. HuggingFace Authentication"

if grep -q "HF_TOKEN=hf_" "$PROJECT_DIR/.env" 2>/dev/null || \
   grep -q "HF_TOKEN=your_" "$PROJECT_DIR/.env" 2>/dev/null; then
    print_error "HuggingFace Token nicht gesetzt!"
    echo ""
    echo "  1. Besuche: https://huggingface.co/settings/tokens"
    echo "  2. Erstelle neuen Token (mit 'read' Berechtigung)"
    echo "  3. Öffne: $PROJECT_DIR/.env"
    echo "  4. Ersetze: HF_TOKEN=your_token_here"
    echo "               mit: HF_TOKEN=hf_xxxxx"
    echo ""
    print_step "Warte auf Eingabe..."
    read -p "Drücke Enter wenn Token gesetzt wurde..."
fi

# Token validieren
if grep -q "export HF_TOKEN" "$PROJECT_DIR/.env"; then
    HF_TOKEN=$(grep "HF_TOKEN=" "$PROJECT_DIR/.env" | cut -d= -f2 | tr -d ' ')
    if [ "${HF_TOKEN:0:3}" = "hf_" ]; then
        print_success "HuggingFace Token gefunden"
    else
        print_error "Token-Format ungültig (sollte mit 'hf_' beginnen)"
    fi
fi

# ============================================================
# [ 7 ] Optional: Modelle vorbereiten
# ============================================================
print_header "7. Modelle vorbereiten (optional)"

read -p "Modelle jetzt herunterladen? (empfohlen, >10GB; y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_step "Starte Modell-Setup..."
    if [ -f "$PROJECT_DIR/scripts/setup_models.sh" ]; then
        bash "$PROJECT_DIR/scripts/setup_models.sh"
    else
        print_error "setup_models.sh nicht gefunden"
    fi
else
    print_step "Modelle werden beim ersten Start automatisch heruntergeladen"
fi

# ============================================================
# [ 8 ] Docker Compose starten
# ============================================================
print_header "8. Container-Stack starten"

cd "$PROJECT_DIR"

print_step "Baue Custom Images..."
docker compose build

print_step "Starte Services..."
docker compose up -d

# ============================================================
# [ 9 ] Health Checks
# ============================================================
print_header "9. Starten und Health Checks"

print_step "Warte auf Service-Start (60s)..."
sleep 30

print_step "Prüfe Services..."

CHECKS=(
    "backend:8080:/health"
    "vllm:8000:/health"
    "qdrant:6333:/health"
    "tts:8001:/health"
    "denoiser:8002:/health"
    "ocr:8003:/health"
)

for check in "${CHECKS[@]}"; do
    IFS=':' read -r service port endpoint <<< "$check"
    if curl -sf "http://localhost:$port$endpoint" > /dev/null 2>&1; then
        print_success "$service ist bereit ($port$endpoint)"
    else
        print_step "$service wird noch gestartet... (dies kann 2-5 Min. dauern)"
    fi
done

# ============================================================
# [ 10 ] Zusammenfassung
# ============================================================
print_header "10. Installation abgeschlossen!"

echo "Services:"
echo "  • Open WebUI:     http://localhost:3000"
echo "  • Backend API:    http://localhost:8080"
echo "  • vLLM LLM:       http://localhost:8000"
echo "  • Qdrant VectorDB: http://localhost:6333"
echo "  • TTS:            http://localhost:8001"
echo "  • Denoiser:       http://localhost:8002"
echo "  • OCR:            http://localhost:8003"
echo ""
echo "Logs anschauen:"
echo "  docker compose logs -f            # Alle Services"
echo "  docker compose logs -f backend    # Nur Backend"
echo "  docker compose logs -f vllm       # Nur LLM"
echo ""
echo "Stoppen:"
echo "  docker compose down"
echo ""
echo "Neu starten:"
echo "  docker compose up -d"
echo ""
echo "Status:"
docker compose ps

print_success "Alles bereit!"
