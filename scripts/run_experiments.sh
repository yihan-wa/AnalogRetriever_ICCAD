#!/usr/bin/env bash
# Reproduce paper Table 2: data preparation, functional clusters, the four
# zero-shot baselines, the TI / TIC (GCN) / TIC (RGCN) ablations and
# AnalogRetriever, followed by evaluation on the held-out split.
#
# Usage (from a checkout with Masala-Chai/ downloaded, see README):
#   bash scripts/run_experiments.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS="$ROOT/results"
LOGS="$RESULTS/logs"
mkdir -p "$LOGS" "$ROOT/checkpoints"
cd "$ROOT"

run() {  # run <log name> <command...>
    local name=$1; shift
    echo ">>> $name"
    "$@" 2>&1 | tee "$LOGS/$name.log"
}

{
    echo "commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
    python3 -c 'import platform, torch; print("python:", platform.python_version()); print("torch:", torch.__version__, "cuda:", torch.version.cuda)'
    if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi; fi
} > "$LOGS/environment.txt"
python3 -m pip freeze > "$LOGS/pip_freeze.txt"

run validate_data python3 scripts/validate_release.py --data-dir Masala-Chai
run prepare_dataset python3 scripts/prepare_dataset.py
run build_clusters python3 scripts/build_clusters.py --k 30

run eval_clip python3 -m analogretriever.evaluation.evaluate_clip --save_json "$RESULTS/clip.json"
run eval_crop python3 -m analogretriever.evaluation.evaluate_crop --save_json "$RESULTS/crop.json"
run eval_chatls python3 -m analogretriever.evaluation.evaluate_chatls --save_json "$RESULTS/chatls.json"
run eval_nettag python3 -m analogretriever.evaluation.evaluate_nettag --save_json "$RESULTS/nettag.json"

run train_ti python3 -m analogretriever.training.train_bimodal --output checkpoints/ti.pth
run train_tic_gcn python3 -m analogretriever.training.train_trimodal --conv gcn --output checkpoints/tic_gcn.pth
run train_tic_rgcn python3 -m analogretriever.training.train_trimodal --conv rgcn --output checkpoints/tic_rgcn.pth
run train_analogretriever python3 -m analogretriever.training.train_curriculum --output checkpoints/analogretriever.pth

for model in ti tic_gcn tic_rgcn analogretriever; do
    run "eval_$model" python3 -m analogretriever.evaluation.evaluate_retrieval \
        --checkpoint "checkpoints/$model.pth" --save_json "$RESULTS/$model.json"
done

run table2 python3 scripts/generate_table2.py --results-dir "$RESULTS"
