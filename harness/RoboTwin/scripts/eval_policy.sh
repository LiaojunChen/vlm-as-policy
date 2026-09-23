#!/bin/bash
set -euo pipefail

CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROBOTWIN_ROOT="$(cd "${CURRENT_DIR}/.." && pwd)"

cd "${ROBOTWIN_ROOT}"

# Headless SAPIEN runs need an explicit NVIDIA ICD in this container. Keep the
# defaults overridable for hosts with a different driver/runtime layout.
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}"
export VK_DRIVER_FILES="${VK_DRIVER_FILES:-${VK_ICD_FILENAMES}}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/robodawn-xdg-runtime}"
mkdir -p "${XDG_RUNTIME_DIR}"
chmod 700 "${XDG_RUNTIME_DIR}"
export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1}"
export no_proxy="${no_proxy:-localhost,127.0.0.1}"

if [[ "${1:-}" == "serve" ]]; then
    shift
    exec python "${ROBOTWIN_ROOT}/scripts/eval_policy_server.py" "$@"
fi

RUN_SCHEDULER=false
if [[ "${1:-}" == "multitask" ]]; then
    shift
    RUN_SCHEDULER=true
else
    for argument in "$@"; do
        if [[ "${argument}" == "--config" || "${argument}" == --config=* ]]; then
            RUN_SCHEDULER=true
            break
        fi
    done
fi

if [[ "${RUN_SCHEDULER}" == "true" ]]; then
    exec python "${ROBOTWIN_ROOT}/scripts/eval_policy_multitask.py" "$@"
fi

ROBOTWIN_EVAL_ARGS=()
if [[ -n "${ROBOTWIN_EVAL_ARGS_FILE:-}" ]]; then
    if [[ ! -f "${ROBOTWIN_EVAL_ARGS_FILE}" ]]; then
        echo "[RoboTwin][ERROR] Eval args file does not exist: ${ROBOTWIN_EVAL_ARGS_FILE}" >&2
        exit 1
    fi
    mapfile -t ROBOTWIN_EVAL_ARGS < "${ROBOTWIN_EVAL_ARGS_FILE}"
fi

PYTHONWARNINGS=ignore::UserWarning \
exec python "${ROBOTWIN_ROOT}/scripts/eval_policy_xpolicylab.py" "$@" "${ROBOTWIN_EVAL_ARGS[@]}"
