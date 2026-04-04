# hotpath

hotpath — the profiler for LLM inference. Kernel timing, request lifecycle tracing, and disaggregation analysis for vLLM and SGLang.

## What it does

- **Profile** — capture CUDA kernel timing, request lifecycle events, and server metrics from a live vLLM or SGLang endpoint.
- **Analyze** — decompose GPU time into prefill vs decode phases, measure KV cache efficiency, detect prefix sharing patterns, and characterize workload behavior.
- **Advise** — run an analytical disaggregation model to determine whether splitting prefill and decode onto separate GPU pools improves throughput and latency, and generate ready-to-use deployment configs.

## Quick Start

Profile a live vLLM server:

```bash
hotpath serve-profile \
  --endpoint http://localhost:8000 \
  --duration 60 \
  --traffic prompts.jsonl \
  --output .hotpath/serve_run
```

View the results:

```bash
hotpath serve-report .hotpath/serve_run/serve_profile.db
```

Generate disaggregation deployment configs:

```bash
hotpath disagg-config .hotpath/serve_run/serve_profile.db --format all
```

## Commands

| Command | Description |
|---------|-------------|
| `serve-profile` | Profile a live vLLM/SGLang server with traffic replay |
| `serve-report` | Generate a human-readable serving analysis report |
| `disagg-config` | Output deployment configs for disaggregated serving |
| `profile` | Run GPU kernel profiling under RL-style rollout workloads |
| `report` | View a saved kernel profile |
| `diff` | Compare two kernel profiles |
| `bench` | Benchmark individual GPU kernel implementations |
| `export` | Export profile data to JSON, CSV, or OTLP format |
| `doctor` | Check local profiling environment |

## Install

```bash
pip install hotpath
```

## System Requirements

- Linux
- NVIDIA GPU with CUDA driver
- `nsys` (for kernel profiling)
- `vllm` or `sglang` (for serving analysis)
- CMake 3.28+, C++20 compiler, SQLite3 (for building from source)

## Build From Source

```bash
cmake -S . -B build
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

Install from source:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install .
```

## How it Works

hotpath is a single C++ binary with no runtime dependencies beyond SQLite3. It collects data from three sources:

1. **Kernel traces** — nsys captures GPU kernel execution. hotpath parses the output, categorizes kernels (gemm, attention, MoE, etc.), and classifies them as prefill or decode phase.
2. **Server metrics** — Prometheus metrics from vLLM or SGLang `/metrics` endpoints are polled at 1Hz. Batch size, queue depth, KV cache pressure, and preemption counts are tracked over time.
3. **Request lifecycle** — vLLM debug logs are parsed to extract per-request timestamps (arrival, queue, prefill, decode, completion). These are stored as structured traces and can be exported as OpenTelemetry spans.

The disaggregation advisor uses a simplified M/G/1 queueing model to estimate whether splitting prefill and decode onto separate GPU pools would improve throughput. It searches over P:D ratios and accounts for KV transfer overhead to produce a concrete recommendation with deployment configs for vLLM, llm-d, and Dynamo.

All data is stored in SQLite databases for offline analysis and comparison.

## Release Notes

See [CHANGELOG.md](CHANGELOG.md).
