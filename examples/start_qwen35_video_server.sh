#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3.5-4B}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-DEBUG}"
LOG_DIR="${LOG_DIR:-.hotpath/video-server}"
PID_FILE="${PID_FILE:-${LOG_DIR}/vllm.pid}"
STDOUT_LOG="${STDOUT_LOG:-${LOG_DIR}/vllm.stdout.log}"
STDERR_LOG="${STDERR_LOG:-${LOG_DIR}/vllm.stderr.log}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"

mkdir -p "${LOG_DIR}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "error: missing python at ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ -f "${PID_FILE}" ]]; then
  old_pid="$(cat "${PID_FILE}")"
  if kill -0 "${old_pid}" 2>/dev/null; then
    echo "server already running with pid ${old_pid}"
    exit 0
  fi
  rm -f "${PID_FILE}"
fi

nohup env VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL}" "${PYTHON_BIN}" -c "from vllm.entrypoints.cli.main import main; main()" \
  serve "${MODEL}" \
  --port "${PORT}" \
  --tensor-parallel-size 1 \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --enforce-eager \
  --language-model-only \
  > "${STDOUT_LOG}" \
  2> "${STDERR_LOG}" \
  < /dev/null &

server_pid=$!
echo "${server_pid}" > "${PID_FILE}"
echo "started server pid ${server_pid}"

for _ in $(seq 1 180); do
  if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    echo "server ready on http://127.0.0.1:${PORT}"
    exit 0
  fi
  if ! kill -0 "${server_pid}" 2>/dev/null; then
    echo "server exited during startup" >&2
    tail -n 80 "${STDERR_LOG}" >&2 || true
    exit 1
  fi
  sleep 1
done

echo "server did not become ready in time" >&2
echo "stderr tail:" >&2
tail -n 80 "${STDERR_LOG}" >&2 || true
exit 1
