#!/usr/bin/env bash
# Launch/reuse the local ZDTaichu model on two GPUs, then evaluate every
# registered RoboTwin task exactly once with the frozen v72 RoboDawn policy.
set -euo pipefail

HARNESS_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd -- "$HARNESS_ROOT/.." && pwd)"
MODEL_ROOT="$WORKSPACE_ROOT/ZDTaichu5.0-9B"
MODEL_PYTHON="$MODEL_ROOT/.venv/bin/python"
ROBOTWIN_PYTHON="$WORKSPACE_ROOT/robodawn_robotwin/.venv-robotwin310/bin/python"

GPU_A=1
GPU_B=0
PORT_A=18052
PORT_B=18053
MAX_STEPS=30
TIMEOUT_S=1200
KEEP_MODELS=0
OUTPUT=""

usage() {
  cat <<'EOF'
Usage:
  ./run_full50_once.sh [options]

Options:
  --output PATH       Result directory. Default: results/full50_once_<UTC timestamp>
  --keep-models       Leave model services started by this script running afterwards.
  --max-steps N       Policy decision limit per task. Default: 30.
  --timeout N         Wall-time limit per task in seconds. Default: 1200.
  -h, --help          Show this help.

Protocol:
  - 50 registered RoboTwin tasks
  - project: robodawn (v72)
  - one expert-validated seed and one episode per task
  - two workers, GPU 1/0, model endpoints 18052/18053
  - 320x240 head camera, 37-degree FOV, JSON-schema repair mode

An explicit existing --output can be supplied to resume an interrupted run.
Completed task results are never overwritten.
EOF
}

while (($#)); do
  case "$1" in
    --output)
      [[ $# -ge 2 ]] || { echo "--output requires a value" >&2; exit 2; }
      OUTPUT="$2"; shift 2 ;;
    --keep-models) KEEP_MODELS=1; shift ;;
    --max-steps)
      [[ $# -ge 2 ]] || { echo "--max-steps requires a value" >&2; exit 2; }
      MAX_STEPS="$2"; shift 2 ;;
    --timeout)
      [[ $# -ge 2 ]] || { echo "--timeout requires a value" >&2; exit 2; }
      TIMEOUT_S="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ "$MAX_STEPS" =~ ^[1-9][0-9]*$ ]] || { echo "--max-steps must be positive" >&2; exit 2; }
[[ "$TIMEOUT_S" =~ ^[1-9][0-9]*$ ]] || { echo "--timeout must be positive" >&2; exit 2; }
[[ -x "$MODEL_PYTHON" ]] || { echo "Missing model Python: $MODEL_PYTHON" >&2; exit 1; }
[[ -x "$ROBOTWIN_PYTHON" ]] || { echo "Missing RoboTwin Python: $ROBOTWIN_PYTHON" >&2; exit 1; }
[[ -f "$MODEL_ROOT/model/config.json" ]] || { echo "Missing model checkpoint: $MODEL_ROOT/model" >&2; exit 1; }

if [[ -z "$OUTPUT" ]]; then
  OUTPUT="$HARNESS_ROOT/results/full50_once_$(date -u +%Y%m%dT%H%M%SZ)"
elif [[ "$OUTPUT" != /* ]]; then
  OUTPUT="$HARNESS_ROOT/$OUTPUT"
fi
case "$OUTPUT/" in
  "$HARNESS_ROOT"/*) ;;
  *) echo "Output must be inside $HARNESS_ROOT" >&2; exit 1 ;;
esac

SEED_COUNT=$(find "$HARNESS_ROOT/seeds" -mindepth 2 -maxdepth 2 -name result.json -type f | wc -l)
[[ "$SEED_COUNT" -eq 50 ]] || { echo "Expected 50 task seed manifests, found $SEED_COUNT" >&2; exit 1; }

mkdir -p "$OUTPUT/launcher"
LAUNCH_LOG="$OUTPUT/launcher/launcher.log"
exec > >(tee -a "$LAUNCH_LOG") 2>&1

source "$HARNESS_ROOT/runtime/env.sh"
export VLM_JSON_SCHEMA=1

declare -a OWNED_PIDS=()

cleanup() {
  local code=$?
  if [[ "$KEEP_MODELS" -eq 0 && ${#OWNED_PIDS[@]} -gt 0 ]]; then
    echo "Stopping model services started by this script: ${OWNED_PIDS[*]}"
    for pid in "${OWNED_PIDS[@]}"; do
      kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    done
  fi
  exit "$code"
}
trap cleanup EXIT INT TERM

health_json() {
  local port=$1
  curl --silent --show-error --fail --max-time 5 "http://127.0.0.1:${port}/health" 2>/dev/null
}

health_matches() {
  local port=$1 gpu=$2 payload
  payload=$(health_json "$port") || return 1
  HEALTH_PAYLOAD="$payload" "$ROBOTWIN_PYTHON" - "$gpu" <<'PY'
import json, os, sys
value = json.loads(os.environ["HEALTH_PAYLOAD"])
expected_gpu = sys.argv[1]
ok = (
    value.get("status") == "ok"
    and value.get("model") == "zdtaichu"
    and value.get("enable_thinking") is False
    and value.get("json_schema_enforced") is True
    and str(value.get("visible_gpu")) == expected_gpu
)
raise SystemExit(0 if ok else 1)
PY
}

port_listens() {
  "$ROBOTWIN_PYTHON" - "$1" <<'PY'
import socket, sys
with socket.socket() as sock:
    raise SystemExit(0 if sock.connect_ex(("127.0.0.1", int(sys.argv[1]))) == 0 else 1)
PY
}

start_or_reuse_model() {
  local gpu=$1 port=$2
  if health_matches "$port" "$gpu"; then
    echo "Reusing healthy model endpoint: GPU=$gpu port=$port"
    return
  fi
  if port_listens "$port"; then
    echo "Port $port is occupied by a service that does not match the required model/GPU." >&2
    exit 1
  fi

  local log="$OUTPUT/launcher/model_${port}.log"
  echo "Starting ZDTaichu on GPU=$gpu port=$port; log=$log"
  CUDA_VISIBLE_DEVICES="$gpu" \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    setsid "$MODEL_PYTHON" -u "$HARNESS_ROOT/runtime/taichu_server.py" \
      --port "$port" >> "$log" 2>&1 &
  local pid=$!
  OWNED_PIDS+=("$pid")
  echo "$pid" > "$OUTPUT/launcher/model_${port}.pid"
}

wait_for_model() {
  local gpu=$1 port=$2 deadline=$((SECONDS + 900))
  until health_matches "$port" "$gpu"; do
    if ((SECONDS >= deadline)); then
      echo "Timed out waiting for GPU=$gpu model endpoint on port $port" >&2
      tail -100 "$OUTPUT/launcher/model_${port}.log" 2>/dev/null || true
      exit 1
    fi
    sleep 5
  done
  echo "Model ready: GPU=$gpu endpoint=http://127.0.0.1:${port}/v1"
}

start_or_reuse_model "$GPU_A" "$PORT_A"
start_or_reuse_model "$GPU_B" "$PORT_B"
wait_for_model "$GPU_A" "$PORT_A"
wait_for_model "$GPU_B" "$PORT_B"

cat > "$OUTPUT/launcher/config.json" <<EOF
{
  "harness": "$HARNESS_ROOT",
  "output": "$OUTPUT",
  "tasks": 50,
  "episodes_per_task": 1,
  "max_policy_decisions": $MAX_STEPS,
  "timeout_s": $TIMEOUT_S,
  "gpu_ids": [$GPU_A, $GPU_B],
  "model_endpoints": ["http://127.0.0.1:$PORT_A/v1", "http://127.0.0.1:$PORT_B/v1"],
  "json_schema": true
}
EOF

echo "Starting RoboTwin full50 evaluation"
echo "Results: $OUTPUT"
"$ROBOTWIN_PYTHON" -u "$HARNESS_ROOT/run_cohort_v3.py" \
  --output "$OUTPUT" \
  --projects robodawn \
  --workers 2 \
  --gpu-ids "$GPU_A" "$GPU_B" \
  --endpoints "http://127.0.0.1:$PORT_A/v1" "http://127.0.0.1:$PORT_B/v1" \
  --model zdtaichu \
  --model-checkpoint phyRSI/ZDTaichu5.0-9B \
  --max-steps "$MAX_STEPS" \
  --timeout "$TIMEOUT_S" \
  --head-resolution 320

"$ROBOTWIN_PYTHON" - "$OUTPUT/summary.json" <<'PY'
import json, sys
summary = json.load(open(sys.argv[1], encoding="utf-8"))
print("Evaluation complete:")
print(json.dumps({key: summary.get(key) for key in ("requested", "completed", "successes", "errors")}, indent=2))
if summary.get("requested") != 50 or summary.get("completed", 0) + summary.get("errors", 0) != 50:
    raise SystemExit("Evaluation did not terminate all 50 task episodes")
PY

echo "Report: $OUTPUT/REPORT.md"
echo "CSV:    $OUTPUT/results.csv"
echo "JSON:   $OUTPUT/summary.json"
