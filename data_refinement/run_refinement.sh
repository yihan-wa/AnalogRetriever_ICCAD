#!/bin/bash
# ============================================================================
# Two-Stage LLM-Based Data Refinement Pipeline
# ============================================================================
# Usage:
#   export OPENROUTER_API_KEY="your-key"
#   bash data_refinement/run_refinement.sh [START [END]]  # optional subset
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${DATA_DIR:-${SCRIPT_DIR}/raw_masala_chai}"   # original Masala-CHAI release with data_mapping.json
MAPPING="${DATA_DIR}/data_mapping.json"
RUN_DIR="${REFINEMENT_RUN_DIR:-${SCRIPT_DIR}/output}"
STAGE1_DIR="${RUN_DIR}/stage1"
STAGE2_DIR="${RUN_DIR}/stage2"
RESULTS_DIR="${RUN_DIR}/results"

# Parse optional arguments
START=${1:-0}
END=${2:-""}

SUBSET_ARGS=(--start "$START")
if [ -n "$END" ]; then
    SUBSET_ARGS+=(--end "$END")
fi

echo "============================================"
echo "  Two-Stage LLM-Based Data Refinement"
echo "============================================"
echo "Data dir:    ${DATA_DIR}"
echo "Stage1 out:  ${STAGE1_DIR}"
echo "Stage2 out:  ${STAGE2_DIR}"
echo "Results:     ${RESULTS_DIR}"
echo ""

# Check original input identities before simulator work or any paid API request.
# The paper uses 6,069 paired originals and 302 image-only inputs. Broken paths
# inside the original mapping are errors, not newly discovered missing samples.
python3 "${SCRIPT_DIR}/stage1_refine.py" \
    --data-dir "${DATA_DIR}" --mapping "${MAPPING}" --validate-only \
    --expected-original 6069 --expected-generated 302

if ! command -v ngspice >/dev/null 2>&1; then
    echo "ERROR: Ngspice is required before running the refinement pipeline"
    exit 1
fi

# Check API key
if [ -z "${OPENROUTER_API_KEY:-}" ]; then
    echo "ERROR: OPENROUTER_API_KEY not set!"
    echo "Run: export OPENROUTER_API_KEY='your-key'"
    exit 1
fi

mkdir -p "${RESULTS_DIR}"

# ── Step 0: Evaluate original dataset ────────────────────────────────────────
echo ">>> Step 0: Evaluating original Masala-CHAI dataset..."
python3 -u "${SCRIPT_DIR}/spice_checker.py" \
    --data-dir "${DATA_DIR}" \
    --mapping "${MAPPING}" \
    --output "${RESULTS_DIR}/original_check.json" \
    --workers 16
echo ""

# ── Step 1: Stage 1 — Repair originals; generate missing pairs ──────────────
echo ">>> Step 1: Stage 1 — Repairing original netlists and completing missing pairs..."
python3 -u "${SCRIPT_DIR}/stage1_refine.py" \
    --data-dir "${DATA_DIR}" \
    --mapping "${MAPPING}" \
    --output-dir "${STAGE1_DIR}" \
    "${SUBSET_ARGS[@]}" \
    --expected-original 6069 --expected-generated 302 \
    --workers 16 \
    --delay 0.2
echo ""

# ── Step 1b: Check Stage 1 results ──────────────────────────────────────────
echo ">>> Step 1b: Evaluating Stage 1 dataset..."
python3 -u "${SCRIPT_DIR}/spice_checker.py" \
    --data-dir "${STAGE1_DIR}" \
    --mapping "${STAGE1_DIR}/data_mapping.json" \
    --output "${RESULTS_DIR}/stage1_check.json" \
    --workers 16
echo ""

# ── Step 2: Stage 2 — Netlist Repair + Description Refinement ───────────────
echo ">>> Step 2: Stage 2 — Repairing netlists and refining descriptions..."
python3 -u "${SCRIPT_DIR}/stage2_refine.py" \
    --data-dir "${DATA_DIR}" \
    --stage1-dir "${STAGE1_DIR}" \
    --output-dir "${STAGE2_DIR}" \
    --max-iters 5 \
    --workers 16 \
    --delay 0.2
echo ""

# ── Step 2b: Check Stage 2 results ──────────────────────────────────────────
echo ">>> Step 2b: Evaluating Stage 2 dataset..."
python3 -u "${SCRIPT_DIR}/spice_checker.py" \
    --data-dir "${STAGE2_DIR}" \
    --mapping "${STAGE2_DIR}/data_mapping.json" \
    --output "${RESULTS_DIR}/stage2_check.json" \
    --workers 16
echo ""

# ── Step 3: Generate summary table ──────────────────────────────────────────
echo ">>> Step 3: Generating Table 1..."
python3 -u "${SCRIPT_DIR}/generate_table1.py" \
    --original "${RESULTS_DIR}/original_check.json" \
    --stage1 "${RESULTS_DIR}/stage1_check.json" \
    --stage2 "${RESULTS_DIR}/stage2_check.json" \
    --output "${RESULTS_DIR}/table1.json"

echo ""
echo "============================================"
echo "  Pipeline Complete!"
echo "============================================"
echo "Original dataset:  ${DATA_DIR}"
echo "Stage 1 output:    ${STAGE1_DIR}"
echo "Stage 2 output:    ${STAGE2_DIR}"
echo "Results & table:   ${RESULTS_DIR}"
echo "Prompts:           ${SCRIPT_DIR}/prompts/"
