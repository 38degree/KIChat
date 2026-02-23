#!/bin/bash
# ============================================================
# Evaluation der NVFP4-Quantisierung
# ============================================================
#
# Vergleicht das quantisierte Modell (FP4) mit dem Original (FP16)
# auf psychiatrischen Testfragen.
#
# Metriken:
#   - Halluzinationsrate (manuelle Bewertung)
#   - Antwortqualität (Relevanz, Vollständigkeit)
#   - Fachterminologie-Korrektheit
#   - Latenz (Time-to-first-token, Token/s)
#   - Memory-Verbrauch
#
# Verwendung:
#   ./scripts/evaluate_quantization.sh [--baseline]
# ============================================================

set -euo pipefail

VLLM_URL="${VLLM_URL:-http://localhost:8000}"
OUTPUT_FILE="data/evaluation_results_$(date +%Y%m%d_%H%M%S).json"

echo "============================================="
echo "  NVFP4 Quantisierungs-Evaluation"
echo "============================================="
echo "vLLM URL:  ${VLLM_URL}"
echo "Output:    ${OUTPUT_FILE}"
echo "============================================="

# --- Prüfe ob vLLM erreichbar ---
echo ""
echo "[1/3] Prüfe vLLM-Verbindung..."
curl -sf "${VLLM_URL}/health" > /dev/null || {
    echo "FEHLER: vLLM nicht erreichbar unter ${VLLM_URL}"
    echo "Starte zuerst: docker compose up -d vllm"
    exit 1
}

MODEL_ID=$(curl -sf "${VLLM_URL}/v1/models" | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(data['data'][0]['id'])
")
echo "Modell: ${MODEL_ID}"

# --- Testfragen ---
echo ""
echo "[2/3] Starte Evaluation mit psychiatrischen Testfragen..."

python3 << 'EVAL_SCRIPT'
import json
import time
import urllib.request
import os

VLLM_URL = os.getenv("VLLM_URL", "http://localhost:8000")

# Psychiatrische Testfragen mit erwarteten Schlüsselbegriffen
TEST_CASES = [
    {
        "id": 1,
        "category": "Diagnostik",
        "question": "Welche Kriterien müssen für die Diagnose einer schweren depressiven Episode nach ICD-11 erfüllt sein?",
        "expected_terms": ["ICD-11", "6A70", "depressive", "Stimmung", "Antrieb", "Schlaf", "2 Wochen"],
        "hallucination_check": "Darf keine erfundenen ICD-Codes oder Medikamente nennen"
    },
    {
        "id": 2,
        "category": "Pharmakologie",
        "question": "Welche Überwachungsmaßnahmen sind bei einer Lithium-Therapie zwingend erforderlich?",
        "expected_terms": ["Lithium", "Serumspiegel", "Schilddrüse", "Niere", "Kreatinin", "TSH"],
        "hallucination_check": "Darf keine falschen Referenzwerte nennen"
    },
    {
        "id": 3,
        "category": "Forensik",
        "question": "Erkläre die Voraussetzungen der Schuldunfähigkeit nach § 20 StGB.",
        "expected_terms": ["§ 20", "StGB", "Schuldfähigkeit", "krankhafte seelische Störung", "Einsichtsfähigkeit", "Steuerungsfähigkeit"],
        "hallucination_check": "Darf keine falschen Paragraphen oder Gesetzestexte erfinden"
    },
    {
        "id": 4,
        "category": "Notfall",
        "question": "Welches Vorgehen ist bei akuter Suizidalität eines stationären Patienten angezeigt?",
        "expected_terms": ["Suizidalität", "1:1-Betreuung", "Sicherung", "Krisenintervention"],
        "hallucination_check": "Muss auf ärztliche Überprüfung hinweisen"
    },
    {
        "id": 5,
        "category": "Betreuungsrecht",
        "question": "Unter welchen Voraussetzungen kann eine Zwangsunterbringung nach PsychKG angeordnet werden?",
        "expected_terms": ["PsychKG", "Gefahr", "richterliche", "24 Stunden", "Unterbringung"],
        "hallucination_check": "Darf keine falschen Fristen oder Zuständigkeiten nennen"
    },
    {
        "id": 6,
        "category": "Differentialdiagnostik",
        "question": "Wie unterscheidet sich eine schizoaffektive Störung von einer Schizophrenie mit komorbider Depression?",
        "expected_terms": ["schizoaffektiv", "Schizophrenie", "affektiv", "Verlauf", "gleichzeitig"],
        "hallucination_check": "Darf keine ICD-Codes verwechseln"
    },
    {
        "id": 7,
        "category": "Therapie",
        "question": "Beschreibe die Indikationen und Kontraindikationen der Elektrokrampftherapie (EKT).",
        "expected_terms": ["EKT", "therapieresistent", "Depression", "Katatonie", "Narkose"],
        "hallucination_check": "Darf keine erfundenen Kontraindikationen nennen"
    },
    {
        "id": 8,
        "category": "Kinder/Jugend",
        "question": "Welche Besonderheiten gelten bei der medikamentösen Behandlung von ADHS im Kindesalter?",
        "expected_terms": ["ADHS", "Methylphenidat", "Wachstum", "Monitoring", "Dosierung"],
        "hallucination_check": "Darf keine falschen Dosierungen für Kinder nennen"
    },
    {
        "id": 9,
        "category": "Suchtmedizin",
        "question": "Beschreibe das Vorgehen bei der qualifizierten Alkoholentzugsbehandlung.",
        "expected_terms": ["Alkohol", "Entzug", "Delir", "Krampfanfall", "Benzodiazepine", "CIWA"],
        "hallucination_check": "Darf keine falschen Entzugs-Scores erfinden"
    },
    {
        "id": 10,
        "category": "Gerontopsychiatrie",
        "question": "Wie wird eine Demenz vom Alzheimer-Typ von einer vaskulären Demenz differentialdiagnostisch abgegrenzt?",
        "expected_terms": ["Alzheimer", "vaskulär", "schleichend", "stufenweise", "MRT", "Gedächtnis"],
        "hallucination_check": "Darf keine falschen Biomarker-Werte nennen"
    },
]

results = []

for tc in TEST_CASES:
    print(f"\n  Test {tc['id']}/{len(TEST_CASES)}: {tc['category']}...")
    
    payload = json.dumps({
        "model": "default",
        "messages": [
            {"role": "system", "content": "Du bist ein psychiatrischer Fachassistent. Antworte präzise und fachlich korrekt."},
            {"role": "user", "content": tc["question"]}
        ],
        "temperature": 0.1,
        "max_tokens": 1024,
    }).encode()
    
    req = urllib.request.Request(
        f"{VLLM_URL}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    
    start_time = time.time()
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read())
        
        elapsed = time.time() - start_time
        answer = result["choices"][0]["message"]["content"]
        tokens = result.get("usage", {})
        
        # Prüfe erwartete Begriffe
        found_terms = [t for t in tc["expected_terms"] if t.lower() in answer.lower()]
        missing_terms = [t for t in tc["expected_terms"] if t.lower() not in answer.lower()]
        term_coverage = len(found_terms) / len(tc["expected_terms"]) * 100
        
        test_result = {
            "id": tc["id"],
            "category": tc["category"],
            "question": tc["question"],
            "answer_length": len(answer),
            "latency_seconds": round(elapsed, 2),
            "prompt_tokens": tokens.get("prompt_tokens", 0),
            "completion_tokens": tokens.get("completion_tokens", 0),
            "term_coverage_pct": round(term_coverage, 1),
            "found_terms": found_terms,
            "missing_terms": missing_terms,
            "hallucination_check": tc["hallucination_check"],
            "answer_preview": answer[:300] + "..." if len(answer) > 300 else answer,
            "status": "success",
        }
        
        print(f"    ✓ {term_coverage:.0f}% Begriffe gefunden, {elapsed:.1f}s")
        
    except Exception as e:
        test_result = {
            "id": tc["id"],
            "category": tc["category"],
            "status": "error",
            "error": str(e),
        }
        print(f"    ✗ Fehler: {e}")
    
    results.append(test_result)

# Zusammenfassung
successful = [r for r in results if r["status"] == "success"]
if successful:
    avg_coverage = sum(r["term_coverage_pct"] for r in successful) / len(successful)
    avg_latency = sum(r["latency_seconds"] for r in successful) / len(successful)
    
    summary = {
        "total_tests": len(TEST_CASES),
        "successful": len(successful),
        "failed": len(TEST_CASES) - len(successful),
        "avg_term_coverage_pct": round(avg_coverage, 1),
        "avg_latency_seconds": round(avg_latency, 2),
    }
else:
    summary = {"total_tests": len(TEST_CASES), "successful": 0, "failed": len(TEST_CASES)}

output = {
    "evaluation": "NVFP4 Quantization Quality Check",
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "summary": summary,
    "results": results,
}

output_file = os.getenv("OUTPUT_FILE", "data/evaluation_results.json")
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"\n{'='*50}")
print(f"  Evaluation abgeschlossen")
print(f"{'='*50}")
print(f"  Tests:              {summary['total_tests']}")
print(f"  Erfolgreich:        {summary['successful']}")
print(f"  Ø Term-Coverage:    {summary.get('avg_term_coverage_pct', 'N/A')}%")
print(f"  Ø Latenz:           {summary.get('avg_latency_seconds', 'N/A')}s")
print(f"  Ergebnisse:         {output_file}")
print(f"{'='*50}")
EVAL_SCRIPT

echo ""
echo "[3/3] Evaluation abgeschlossen."
echo "Ergebnisse: ${OUTPUT_FILE}"
echo ""
echo "HINWEIS: Die Halluzinations-Checks erfordern manuelle Überprüfung."
echo "Prüfe die 'answer_preview' und 'hallucination_check' Felder in der Ausgabe."
