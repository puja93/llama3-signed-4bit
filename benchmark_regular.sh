#!/bin/bash
# ==============================================================================
# Benchmark Script: Original LLaMA-3.2-1B Unquantized Baseline (BF16, 2.3 GB)
# Runs the full EleutherAI / Hugging Face LM Evaluation Harness and redirects
# all output logs and structured JSON metrics to tests/results/
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="/opt/anaconda3/bin/python3"
if [ ! -f "$PYTHON_BIN" ]; then
    PYTHON_BIN="$(which python3)"
fi

RESULTS_DIR="tests/results"
mkdir -p "$RESULTS_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${RESULTS_DIR}/benchmark_regular_${TIMESTAMP}.log"
JSON_FILE="${RESULTS_DIR}/benchmark_regular_${TIMESTAMP}.json"
LATEST_LOG="${RESULTS_DIR}/benchmark_regular_latest.log"
LATEST_JSON="${RESULTS_DIR}/benchmark_regular_latest.json"

DEFAULT_TASKS="arc_challenge,hellaswag,mmlu,truthfulqa_mc2,winogrande,gsm8k"

echo "================================================================================"
echo "  BENCHMARK: ORIGINAL 1B REGULAR MODEL (BF16 model.safetensors, ~2.3 GB)"
echo "  Timestamp:   ${TIMESTAMP}"
echo "  Log File:    ${LOG_FILE}"
echo "  JSON File:   ${JSON_FILE}"
echo "  Interpreter: ${PYTHON_BIN}"
echo "================================================================================"

# Execute evaluation and tee output to both console and log file
"$PYTHON_BIN" tests/eval.py \
    --model_type hf \
    --tasks "$DEFAULT_TASKS" \
    --output_json "$JSON_FILE" \
    "$@" 2>&1 | tee "$LOG_FILE"

# Create/update latest symlinks
cp "$LOG_FILE" "$LATEST_LOG"
cp "$JSON_FILE" "$LATEST_JSON"

echo ""
echo "================================================================================"
echo "  REGULAR MODEL BENCHMARK COMPLETE"
echo "  Log saved to:  ${LOG_FILE} (and ${LATEST_LOG})"
echo "  JSON saved to: ${JSON_FILE} (and ${LATEST_JSON})"
echo "================================================================================"
