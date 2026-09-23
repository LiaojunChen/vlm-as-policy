#!/usr/bin/env bash
set -euo pipefail
EVAL_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="${EVAL_ROOT}/runtime/bin:${PATH}"
export PYTHONPATH="${EVAL_ROOT}:${EVAL_ROOT}/RoboTwin/envs/curobo/src:${EVAL_ROOT}/RoboTwin/scripts${PYTHONPATH:+:${PYTHONPATH}}"
export VK_ICD_FILENAMES="${EVAL_ROOT}/runtime/nvidia_icd.json"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export VLM_MODEL=zdtaichu
export VLM_ENDPOINT="${VLM_ENDPOINT:-http://127.0.0.1:18050/v1}"
export VLM_API_KEY=EMPTY
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export XDG_RUNTIME_DIR="/tmp/zdtaichu-robotwin-xdg"
mkdir -p "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR"
