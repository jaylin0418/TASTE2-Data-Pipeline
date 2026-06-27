#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

unset PYTHONPATH || true
unset PYTHONHOME || true

if [[ "${KEEP_SSL_CERT_FILE:-0}" != "1" ]]; then
  if [[ -n "${SSL_CERT_FILE:-}" && ! -f "${SSL_CERT_FILE}" ]]; then
    unset SSL_CERT_FILE || true
  fi
fi

export OPENROUTER_TIMEOUT_SECONDS="${OPENROUTER_TIMEOUT_SECONDS:-120}"
export OPENROUTER_REQUEST_MAX_RETRIES="${OPENROUTER_REQUEST_MAX_RETRIES:-20}"
export OPENROUTER_RETRY_BACKOFF_BASE="${OPENROUTER_RETRY_BACKOFF_BASE:-2}"
export OPENROUTER_RETRY_BACKOFF_CAP="${OPENROUTER_RETRY_BACKOFF_CAP:-300}"

PYTHON_BIN="${SYN_PYTHON:-auto}"
if [[ "$PYTHON_BIN" == "auto" ]]; then
  if [[ -x "$HOME/miniconda3/envs/breezyvoice_py310/bin/python" ]]; then
    PYTHON_BIN="$HOME/miniconda3/envs/breezyvoice_py310/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi

if [[ "${KEEP_SSL_CERT_FILE:-0}" != "1" ]]; then
  if [[ -z "${SSL_CERT_FILE:-}" ]]; then
    CERTIFI_CA="$($PYTHON_BIN -c 'import certifi; print(certifi.where())' 2>/dev/null || true)"
    if [[ -n "$CERTIFI_CA" && -f "$CERTIFI_CA" ]]; then
      export SSL_CERT_FILE="$CERTIFI_CA"
    fi
  fi
fi

CFG_DEFAULT="$SCRIPT_DIR/../conf/base_variant_breezy.yaml"
CFG="${CFG:-$CFG_DEFAULT}"

PER_TOPIC_COUNT="${PER_TOPIC_COUNT:-18}"
OUTPUT_ROOT_BASE="${OUTPUT_ROOT_BASE:-/work/${USER:-$(whoami)}}"
DATA_ROOT="${DATA_ROOT:-}"
WORKERS="${WORKERS:-1}"
TOPICS="${TOPICS:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --per-topic-count)  PER_TOPIC_COUNT="$2";  shift 2 ;;
    --output-root-base) OUTPUT_ROOT_BASE="$2"; shift 2 ;;
    --output-root)      DATA_ROOT="$2";        shift 2 ;;
    --workers)          WORKERS="$2";          shift 2 ;;
    --topics)           TOPICS="$2";           shift 2 ;;
    -h|--help)
      cat <<'EOF'
Usage: ./generate_para_txt.sh [OPTIONS]

Options:
  --per-topic-count N      Scenarios per topic (default: 100)
  --output-root PATH       Full path for data root (e.g. /work/user/syn_para_TC)
  --output-root-base PATH  Base dir; data root = BASE/syn_para (default: /work/$USER)
  --workers N              Parallel topics (default: 1)
  --topics T1,T2,...       Comma-separated topics (default: all from yaml)

Environment overrides:
  PER_TOPIC_COUNT, DATA_ROOT, OUTPUT_ROOT_BASE, WORKERS, TOPICS, CFG, SYN_PYTHON
EOF
      exit 0 ;;
    *) echo "[ERROR] Unknown argument: $1" >&2; exit 1 ;;
  esac
done

# Read default topics from yaml if not specified
if [[ -z "$TOPICS" ]]; then
  TOPICS="$($PYTHON_BIN -c "
from omegaconf import OmegaConf
cfg = OmegaConf.load('$CFG')
print(','.join(cfg.multi_topic_run.topics))
")"
fi

if [[ -z "$DATA_ROOT" ]]; then
  DATA_ROOT="$OUTPUT_ROOT_BASE/syn_variant"
fi

echo "=============================="
echo "CFG             = $CFG"
echo "DATA_ROOT       = $DATA_ROOT"
echo "PER_TOPIC_COUNT = $PER_TOPIC_COUNT"
echo "WORKERS         = $WORKERS"
echo "SSL_CERT_FILE   = ${SSL_CERT_FILE:-<unset>}"
echo "=============================="

IFS=',' read -ra TOPIC_LIST <<< "$TOPICS"
PIDS=()
FAILED=()

run_topic() {
  local topic="$1"
  echo "[START] $topic"
  "$PYTHON_BIN" ../pipeline_v2_para/syn_para_breezy.py \
    --config-name base_variant_breezy \
    data_root="$DATA_ROOT" \
    scenario.topic="$topic" \
    scenario.n="$PER_TOPIC_COUNT" \
    "stages=[scenario,system_prompt,dialogue]" \
    huggingface.push_to_hub=false \
    && echo "[DONE] $topic" \
    || echo "[FAIL] $topic"
}

export -f run_topic
export PYTHON_BIN DATA_ROOT PER_TOPIC_COUNT CFG

for topic in "${TOPIC_LIST[@]}"; do
  topic="$(echo "$topic" | xargs)"  # trim whitespace
  [[ -z "$topic" ]] && continue

  run_topic "$topic" &
  PIDS+=($!)

  # throttle to WORKERS parallel
  while [[ ${#PIDS[@]} -ge $WORKERS ]]; do
    NEW_PIDS=()
    for pid in "${PIDS[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then
        NEW_PIDS+=("$pid")
      fi
    done
    PIDS=("${NEW_PIDS[@]}")
    [[ ${#PIDS[@]} -ge $WORKERS ]] && sleep 2
  done
done

# Wait for remaining
for pid in "${PIDS[@]}"; do
  wait "$pid" || true
done

echo "=============================="
echo "ALL TOPICS DONE"
echo "=============================="
