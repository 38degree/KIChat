#!/bin/bash
# ============================================================
# NVFP4 Quantisierung — Qwen 2.5 72B-Instruct
# ============================================================
#
# Quantisiert das LLM mit TensorRT Model Optimizer für FP4-Inferenz
# auf dem NVIDIA DGX Spark (Blackwell GB10).
#
# Voraussetzungen:
#   - Docker mit GPU-Support
#   - HuggingFace Token mit Zugang zum Modell
#   - ~150GB freier Speicherplatz (Original + quantisiert)
#
# Verwendung:
#   ./scripts/quantize_model.sh
#
# Dauer: 45-90 Minuten je nach Modellgröße
# ============================================================

set -euo pipefail

# --- Konfiguration ---
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-72B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-./models/llm/qwen2.5-72b-fp4}"
CALIBRATION_DATA="${CALIBRATION_DATA:-./data/calibration}"
HF_TOKEN="${HF_TOKEN:?Bitte HF_TOKEN setzen (export HF_TOKEN=your_token)}"

# TensorRT-LLM Container für DGX Spark
TRTLLM_IMAGE="nvcr.io/nvidia/tensorrt-llm/release:spark-single-gpu-dev"

echo "============================================="
echo "  NVFP4 Quantisierung"
echo "============================================="
echo "Modell:    ${MODEL_NAME}"
echo "Output:    ${OUTPUT_DIR}"
echo "Container: ${TRTLLM_IMAGE}"
echo "============================================="

# --- Verzeichnisse erstellen ---
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${CALIBRATION_DATA}"

# --- Prüfe GPU-Zugang ---
echo ""
echo "[1/5] Prüfe GPU-Zugang..."
docker run --rm --gpus all "${TRTLLM_IMAGE}" nvidia-smi || {
    echo "FEHLER: Docker GPU-Zugang nicht verfügbar."
    echo "Stelle sicher, dass NVIDIA Container Toolkit installiert ist."
    exit 1
}

# --- Prüfe Speicherplatz ---
echo ""
echo "[2/5] Prüfe Speicherplatz..."
AVAILABLE_GB=$(df -BG . | tail -1 | awk '{print $4}' | tr -d 'G')
if [ "${AVAILABLE_GB}" -lt 150 ]; then
    echo "WARNUNG: Nur ${AVAILABLE_GB}GB verfügbar, empfohlen sind 150GB+"
fi

# --- Kalibrierungsdaten erstellen (falls nicht vorhanden) ---
if [ ! -f "${CALIBRATION_DATA}/calibration.jsonl" ]; then
    echo ""
    echo "[3/5] Erstelle Kalibrierungsdaten (psychiatrischer Fachtext)..."
    cat > "${CALIBRATION_DATA}/calibration.jsonl" << 'CALIBRATION_EOF'
{"text": "Die Diagnose einer schweren depressiven Episode (ICD-11: 6A70.2) erfordert eine sorgfältige Differentialdiagnostik unter Berücksichtigung komorbider Angststörungen und Persönlichkeitsstörungen."}
{"text": "Bei der Behandlung der Schizophrenie mit atypischen Antipsychotika wie Clozapin ist ein regelmäßiges Blutbild-Monitoring zur Erkennung einer Agranulozytose zwingend erforderlich."}
{"text": "Die gutachterliche Beurteilung der Schuldfähigkeit nach §§ 20, 21 StGB setzt eine Exploration des Beschuldigten voraus, die mindestens die Erhebung einer ausführlichen biographischen Anamnese umfasst."}
{"text": "Gemäß dem Betreuungsrecht (§ 1896 BGB) ist die Einrichtung einer rechtlichen Betreuung nur zulässig, wenn der Betroffene seine Angelegenheiten ganz oder teilweise nicht besorgen kann."}
{"text": "Die kognitive Verhaltenstherapie (KVT) zeigt in Meta-Analysen eine Effektstärke von d=0.71 für die Behandlung generalisierter Angststörungen, wobei die Kombination mit SSRI die Remissionsrate signifikant erhöht."}
{"text": "Im Rahmen der forensisch-psychiatrischen Begutachtung ist die Unterscheidung zwischen einer vorübergehenden Störung der Geistestätigkeit und einer krankhaften seelischen Störung für die Beurteilung der Schuldfähigkeit maßgeblich."}
{"text": "Die Elektrokrampftherapie (EKT) gilt als Goldstandard bei therapieresistenter Depression und katatoner Schizophrenie. Die Indikation sollte interdisziplinär gestellt werden."}
{"text": "Posttraumatische Belastungsstörungen (PTBS, ICD-11: 6B40) können durch EMDR und traumafokussierte kognitive Verhaltenstherapie behandelt werden. Die Behandlungsdauer beträgt typischerweise 12-20 Sitzungen."}
{"text": "Die Unterbringung nach PsychKG erfordert eine unmittelbare Gefahr für Leib und Leben des Betroffenen oder Dritter. Die richterliche Genehmigung muss innerhalb von 24 Stunden eingeholt werden."}
{"text": "Lithium bleibt der Goldstandard in der Phasenprophylaxe bipolarer Störungen. Therapeutische Serumkonzentrationen liegen bei 0.6-0.8 mmol/l für die Langzeitbehandlung, mit regelmäßiger Kontrolle der Schilddrüsen- und Nierenfunktion."}
CALIBRATION_EOF
    echo "Kalibrierungsdaten erstellt."
else
    echo "[3/5] Kalibrierungsdaten vorhanden."
fi

# --- Modell herunterladen und quantisieren ---
echo ""
echo "[4/5] Starte NVFP4 Quantisierung..."
echo "Dies kann 45-90 Minuten dauern..."
echo ""

docker run --rm --gpus all \
    -v "$(pwd)/models:/models" \
    -v "$(pwd)/data/calibration:/calibration" \
    -e "HF_TOKEN=${HF_TOKEN}" \
    -e "HF_HOME=/models/.cache" \
    "${TRTLLM_IMAGE}" \
    bash -c "
        set -e

        echo '--- Installiere Model Optimizer ---'
        pip install nvidia-modelopt[all] -q

        echo '--- Starte Quantisierung ---'
        python3 -c \"
import modelopt.torch.quantization as mtq
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json

model_name = '${MODEL_NAME}'
output_dir = '/models/llm/qwen2.5-72b-fp4'

print(f'Lade Modell: {model_name}')
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map='auto',
    trust_remote_code=True,
)

# Kalibrierungsdaten laden
calib_texts = []
with open('/calibration/calibration.jsonl', 'r') as f:
    for line in f:
        calib_texts.append(json.loads(line)['text'])

print(f'Kalibrierungsdaten: {len(calib_texts)} Texte')

# Kalibrierungs-Funktion
def calibrate_loop(model):
    for text in calib_texts:
        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            model(**inputs)

# NVFP4 Quantisierung
print('Starte FP4 Quantisierung...')
quant_config = mtq.FP8_DEFAULT_CFG.copy()
# FP4 für Gewichte, FP8 für Aktivierungen
quant_config['quant_cfg']['*weight_quantizer'] = {
    'num_bits': 4,
    'axis': None,
}

model = mtq.quantize(model, quant_config, forward_loop=calibrate_loop)

print(f'Speichere quantisiertes Modell nach {output_dir}')
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print('Quantisierung abgeschlossen!')
\"
    "

echo ""
echo "[5/5] Quantisierung abgeschlossen!"
echo ""
echo "============================================="
echo "  Quantisiertes Modell: ${OUTPUT_DIR}"
echo "============================================="
echo ""
echo "Nächste Schritte:"
echo "  1. Evaluation:  ./scripts/evaluate_quantization.sh"
echo "  2. LLM_MODEL in .env auf lokalen Pfad setzen"
echo "  3. docker compose up -d"
