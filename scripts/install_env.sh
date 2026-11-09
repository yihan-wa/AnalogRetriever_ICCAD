#!/usr/bin/env bash
# Install into the active environment. Install the desired PyTorch/CUDA build first.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -c 'import torch; print("PyTorch", torch.__version__, "CUDA available:", torch.cuda.is_available())'
python3 -m pip install -r requirements.txt
python3 -m pip install -e . --no-deps
python3 -c 'import torch, torch_geometric, clip, transformers, pandas, requests; print("Core imports OK")'
