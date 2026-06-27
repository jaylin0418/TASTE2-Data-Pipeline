#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

unset PYTHONPATH || true
unset PYTHONHOME || true
unset SSL_CERT_FILE || true

PYTHON_BIN="${SYN_PYTHON:-auto}"
if [[ "$PYTHON_BIN" == "auto" ]]; then
  if [[ -x "$HOME/miniconda3/envs/breezyvoice_py310/bin/python" ]]; then
    PYTHON_BIN="$HOME/miniconda3/envs/breezyvoice_py310/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi

CFG_DEFAULT="$SCRIPT_DIR/../conf/base_para_breezy.yaml"
CFG="${CFG:-$CFG_DEFAULT}"

# Required: RUN_ROOT
RUN_ROOT="${RUN_ROOT:-${1:-}}"
if [[ -z "$RUN_ROOT" ]]; then
  echo "ERROR: RUN_ROOT is required" >&2
  echo "Usage: RUN_ROOT=/work/.../syn_para_TC ./generate_para_tts.sh" >&2
  echo "   or: ./generate_para_tts.sh /work/.../syn_para_TC" >&2
  exit 2
fi
if [[ "${1:-}" == "$RUN_ROOT" ]]; then
  shift || true
fi

GPUS="${GPUS:-}"
TOPICS="${TOPICS:-}"
TOPICS_PER_GPU="${TOPICS_PER_GPU:-1}"
CONCURRENT_PER_GPU="${CONCURRENT_PER_GPU:-1}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --topics)            TOPICS="$2";            shift 2 ;;
    --gpus)              GPUS="$2";              shift 2 ;;
    --topics-per-gpu)    TOPICS_PER_GPU="$2";    shift 2 ;;
    --concurrent-per-gpu) CONCURRENT_PER_GPU="$2"; shift 2 ;;
    -h|--help)
      cat <<'EOF'
Usage: ./generate_para_tts.sh RUN_ROOT [OPTIONS]

Options:
  --topics T1,T2,...        Topics to process (default: auto-detect from RUN_ROOT/para/)
  --gpus 0,1,2,3            GPU IDs (default: auto-detect)
  --topics-per-gpu N        Topics assigned to each GPU (default: 1)
  --concurrent-per-gpu N    Topics running in parallel on same GPU (default: 1)

Example (4 GPUs, 1 topic per GPU at a time):
  ./generate_para_tts.sh /work/user/syn_para_TC --gpus 0,1,2,3 --topics-per-gpu 11

Postprocessing is applied automatically after each dialogue.

Environment overrides:
  RUN_ROOT, GPUS, TOPICS, TOPICS_PER_GPU, CONCURRENT_PER_GPU, CFG, SYN_PYTHON
EOF
      exit 0 ;;
    *) echo "[ERROR] Unknown argument: $1" >&2; exit 1 ;;
  esac
done

# Auto-detect topics from RUN_ROOT/para/ if not specified
if [[ -z "$TOPICS" ]]; then
  TOPICS="$(ls "$RUN_ROOT/para/" 2>/dev/null | tr '\n' ',' | sed 's/,$//')"
  if [[ -z "$TOPICS" ]]; then
    echo "ERROR: No topics found under $RUN_ROOT/para/ and --topics not specified" >&2
    exit 2
  fi
fi

echo "=============================="
echo "CFG               = $CFG"
echo "RUN_ROOT          = $RUN_ROOT"
echo "TOPICS            = $TOPICS"
echo "GPUS              = ${GPUS:-<auto-detect>}"
echo "TOPICS_PER_GPU    = $TOPICS_PER_GPU"
echo "CONCURRENT_PER_GPU= $CONCURRENT_PER_GPU"
echo "=============================="

args=(
  --config "$CFG"
  --run-root "$RUN_ROOT"
  --topics "$TOPICS"
  --topics-per-gpu "$TOPICS_PER_GPU"
  --concurrent-per-gpu "$CONCURRENT_PER_GPU"
  --python-bin "$PYTHON_BIN"
  --worker-script dialogue_v2_para/run_topic_para_tts.py
  --continue-on-error
)

if [[ -n "$GPUS" ]]; then
  args+=(--gpus "$GPUS")
fi

"$PYTHON_BIN" ../run_multi_topic_tts_workers.py "${args[@]}"
