#!/usr/bin/env bash
set -euo pipefail

PID_FILE="${PID_FILE:-.hotpath/video-server/vllm.pid}"

if [[ ! -f "${PID_FILE}" ]]; then
  echo "no pid file at ${PID_FILE}"
  exit 0
fi

pid="$(cat "${PID_FILE}")"
if kill -0 "${pid}" 2>/dev/null; then
  kill "${pid}" 2>/dev/null || true
  echo "stopped server pid ${pid}"
else
  echo "pid ${pid} is not running"
fi

rm -f "${PID_FILE}"
