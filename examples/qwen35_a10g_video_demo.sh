#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
PORT="${PORT:-8000}"
DURATION="${DURATION:-60}"
CONCURRENCY="${CONCURRENCY:-6}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-12288}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
OUT_DIR="${OUT_DIR:-.hotpath/qwen35-video}"
ARTIFACT_DIR="${ARTIFACT_DIR:-.hotpath/qwen35-video-artifacts}"
TRAFFIC_FILE="${TRAFFIC_FILE:-examples/qwen35_a10g_video_traffic.jsonl}"

mkdir -p "${OUT_DIR}" "${ARTIFACT_DIR}"

echo
echo "[1/5] Locking GPU clocks for reproducibility"
hotpath lock-clocks || true

echo
echo "[2/5] Starting vLLM for ${MODEL}"
LOG_DIR="${ARTIFACT_DIR}" MODEL="${MODEL}" PORT="${PORT}" \
MAX_MODEL_LEN="${MAX_MODEL_LEN}" GPU_MEM_UTIL="${GPU_MEM_UTIL}" \
./examples/start_qwen35_video_server.sh

cleanup() {
  LOG_DIR="${ARTIFACT_DIR}" ./examples/stop_qwen35_video_server.sh || true
}
trap cleanup EXIT

echo
echo "[3/5] Profiling live traffic with hotpath"
hotpath serve-profile \
  --endpoint "http://127.0.0.1:${PORT}" \
  --engine vllm \
  --traffic "${TRAFFIC_FILE}" \
  --concurrency "${CONCURRENCY}" \
  --duration "${DURATION}" \
  --server-log "${ARTIFACT_DIR}/vllm.stdout.log" \
  --output "${OUT_DIR}"

echo
echo "[4/5] Rendering the human-readable report"
hotpath serve-report "${OUT_DIR}/serve_profile.db"

echo
echo "[5/5] Rendering the deployment recommendation"
hotpath disagg-config "${OUT_DIR}/serve_profile.db" --format vllm
