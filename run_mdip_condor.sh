#!/usr/bin/env bash
set -euo pipefail

# HTCondor batch entry point for radial M-DIP runs.
# Usage:
#   ./run_mdip_condor.sh [input_npz] [n_iter]

REPO_DIR="${REPO_DIR:-$(pwd)}"
ENV_NAME="${ENV_NAME:-m-dip}"
RAW_FOLDER="${RAW_FOLDER:-./data/AA/traindata}"
OUT_FOLDER="${OUT_FOLDER:-./results/AA}"
INPUT_NPZ="${1:-slice_1_8_nbins30.npz}"
N_ITER="${2:-1000}"

export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_cache_${USER:-condor}}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl_cache_${USER:-condor}}"
mkdir -p "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" "$OUT_FOLDER"

cd "$REPO_DIR"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "$ENV_NAME"
fi

python - <<'PY'
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
PY

python M-DIP-AA.py \
  --raw-folder "$RAW_FOLDER" \
  --filename "$INPUT_NPZ" \
  --out-folder "$OUT_FOLDER" \
  --n-iter "$N_ITER"
