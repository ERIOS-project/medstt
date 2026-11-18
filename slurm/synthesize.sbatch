#!/bin/bash
#SBATCH --job-name=medstt-tts
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --array=0-3
#SBATCH --output=slurm-logs/%x-%A_%a.out
#
# Phase 2: synthesize train/test audio through vLLM-Omni. Each array task handles one
# shard of the sentences; resubmitting skips what is already done.
#
# Usage (from the repository root):
#   jid=$(sbatch --parsable -A <account> -p <partition> --array=0-7 \
#           slurm/synthesize.sbatch <sentences-file> <dataset-name> [medstt options])
#   sbatch -A <account> -p <partition> --dependency=afterok:${jid} \
#           slurm/finalize_dataset.sbatch <dataset-name>

source "${SLURM_SUBMIT_DIR}/slurm/env.sh"

SENTENCES=${1:?usage: synthesize.sbatch <sentences-file> <dataset-name> [options]}
OUTPUT=${2:?usage: synthesize.sbatch <sentences-file> <dataset-name> [options]}
shift 2

# Use an existing server when explicitly supplied; otherwise start one per shard.
MODEL=$(python - <<'PYMODEL'
from medstt.workspace import Workspace
from medstt.tts.backends import MODEL_ID
print(Workspace.open().section("tts").get("model", MODEL_ID))
PYMODEL
)
if [[ -z "${MEDSTT_TTS_BASE_URL:-}" ]]; then
  SERVER_BIN="${MEDSTT_TTS_VLLM_BIN:-${REPO_DIR}/.venv-vllm-omni/bin/vllm}"
  [[ -x "$SERVER_BIN" ]] || { echo "Missing vLLM-Omni executable: $SERVER_BIN" >&2; exit 1; }
  PORT="${MEDSTT_TTS_PORT:-$(python - <<'PYPORT'
import socket
with socket.socket() as sock:
    sock.bind(("127.0.0.1", 0))
    print(sock.getsockname()[1])
PYPORT
)}"
  SERVER_LOG="${SLURM_SUBMIT_DIR}/slurm-logs/vllm-tts-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID:-0}.log"
  read -r -a SERVER_ARGS <<< "${MEDSTT_TTS_SERVER_ARGS:-}"
  setsid "$SERVER_BIN" serve "$MODEL" --omni --host 127.0.0.1 --port "$PORT" \
    "${SERVER_ARGS[@]}" > "$SERVER_LOG" 2>&1 &
  SERVER_PID=$!
  cleanup() {
    kill -TERM -- "-$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  }
  trap cleanup EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM
  STARTUP_TIMEOUT="${MEDSTT_TTS_STARTUP_TIMEOUT:-900}"
  [[ "$STARTUP_TIMEOUT" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid startup timeout" >&2; exit 1; }
  START=$SECONDS
  echo "[medstt] waiting for vLLM-Omni; server log: $SERVER_LOG"
  until curl --silent --fail --max-time 2 "http://127.0.0.1:$PORT/health" > /dev/null; do
    kill -0 "$SERVER_PID" 2>/dev/null || { echo "vLLM-Omni exited; see $SERVER_LOG" >&2; exit 1; }
    if (( SECONDS - START >= STARTUP_TIMEOUT )); then
      echo "vLLM-Omni startup timed out; see $SERVER_LOG" >&2
      exit 1
    fi
    sleep 2
  done
  export MEDSTT_TTS_BASE_URL="http://127.0.0.1:$PORT/v1"
fi

medstt tts synthesize "$SENTENCES" --output "$OUTPUT" \
  --base-url "$MEDSTT_TTS_BASE_URL" \
  --shard-index "${SLURM_ARRAY_TASK_ID:-0}" \
  --num-shards "${SLURM_ARRAY_TASK_COUNT:-1}" \
  "$@"
