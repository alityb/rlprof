# hotpath

Profiler for LLM inference.

hotpath profiles live vLLM and SGLang servers, analyzes request and GPU behavior, and recommends when to split prefill and decode.

## What it does

- Profile a live endpoint with real traffic
- Analyze queueing, prefill, decode, cache, and batching
- Recommend disaggregation and generate deployment configs

## Install

```bash
uv tool install hotpath
```

## Quick start

Profile a live vLLM server:

```bash
hotpath serve-profile \
  --endpoint http://localhost:8000 \
  --traffic prompts.jsonl \
  --concurrency 4 \
  --duration 60 \
  --output .hotpath/run
```

View the report:

```bash
hotpath serve-report .hotpath/run/serve_profile.db
```

Generate deployment configs:

```bash
hotpath disagg-config .hotpath/run/serve_profile.db --format all
```

If you want server-side request timing, start vLLM with debug logs and pass the log file:

```bash
VLLM_LOGGING_LEVEL=DEBUG vllm serve <model> 2>vllm.log &

hotpath serve-profile \
  --endpoint http://localhost:8000 \
  --traffic prompts.jsonl \
  --server-log vllm.log \
  --concurrency 4 \
  --duration 60
```

If you want kernel-level GPU traces, add `--nsys`:

```bash
hotpath serve-profile \
  --endpoint http://localhost:8000 \
  --traffic prompts.jsonl \
  --nsys
```

## Traffic format

JSONL, one request per line:

```json
{"prompt": "Explain KV cache eviction policy.", "max_tokens": 256}
{"prompt": "Write a Python retry decorator with exponential backoff.", "max_tokens": 400}
```

ShareGPT format is also supported.

## Commands

| Command | Description |
|---------|-------------|
| `serve-profile` | Profile a live vLLM or SGLang server |
| `serve-report` | Print a serving analysis report |
| `disagg-config` | Generate deployment configs for disaggregated serving |
| `profile` | Run GPU kernel profiling under RL-style traffic |
| `report` | View a saved kernel profile |
| `diff` | Compare two kernel profiles |
| `bench` | Benchmark individual GPU kernel implementations |
| `export` | Export profile data to JSON, CSV, or OTLP |
| `doctor` | Check local profiling environment |
| `lock-clocks` | Lock GPU clocks for reproducible measurements |

## System requirements

- Linux
- NVIDIA GPU with CUDA driver
- `nsys` for kernel profiling
- vLLM or SGLang for serving analysis

## Build from source

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

Install from source:

```bash
uv tool install .
```

Requirements: CMake 3.28+, C++20 compiler, SQLite3.

## How it works

hotpath stores results in SQLite and combines three data sources:

1. Kernel traces from `nsys`
2. Server metrics from `/metrics`
3. Request lifecycle timing from client traces and vLLM debug logs

The report turns those signals into latency breakdowns, cache analysis, prefix-sharing analysis, and a disaggregation recommendation.

## Release notes

See [CHANGELOG.md](CHANGELOG.md).
