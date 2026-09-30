#!/bin/bash
# ==============================================================================
# Benchmark Script: LLaMA-3.2 Signed 4-Bit Native Engine (626 MB)
# Runs the full EleutherAI LM Evaluation Harness and redirects
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
LOG_FILE="${RESULTS_DIR}/benchmark_compressed_${TIMESTAMP}.log"
JSON_FILE="${RESULTS_DIR}/benchmark_compressed_${TIMESTAMP}.json"
LATEST_LOG="${RESULTS_DIR}/benchmark_compressed_latest.log"
LATEST_JSON="${RESULTS_DIR}/benchmark_compressed_latest.json"

DEFAULT_TASKS="arc_challenge,hellaswag,mmlu,truthfulqa_mc2,winogrande,gsm8k"

echo "================================================================================"
echo "  BENCHMARK: SIGNED 4-BIT NATIVE ENGINE (626 MB)"
echo "  Timestamp:   ${TIMESTAMP}"
echo "  Log File:    ${LOG_FILE}"
echo "  JSON File:   ${JSON_FILE}"
echo "  Interpreter: ${PYTHON_BIN}"
echo "================================================================================"

# Execute evaluation and tee output to both console and log file
"$PYTHON_BIN" tests/eval.py \
    --model_type signed_4bit \
    --tasks "$DEFAULT_TASKS" \
    --output_json "$JSON_FILE" \
    "$@" 2>&1 | tee "$LOG_FILE"

# Create/update latest symlinks
cp "$LOG_FILE" "$LATEST_LOG"
cp "$JSON_FILE" "$LATEST_JSON"

echo ""
echo "================================================================================"
echo "  COMPRESSED BENCHMARK COMPLETE"
echo "  Log saved to:  ${LOG_FILE} (and ${LATEST_LOG})"
echo "  JSON saved to: ${JSON_FILE} (and ${LATEST_JSON})"
echo "================================================================================"
