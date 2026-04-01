# rlprof

`rlprof` is a measurement tool for profiling vLLM inference under RL-style rollout workloads.

It records:
- CUDA kernel timing from `nsys`
- vLLM `/metrics` samples
- rollout traffic shape statistics
- optional benchmark results for selected GPU kernels

`rlprof` stores collected data in SQLite and reports raw numbers. It is not a training framework, dashboard, or optimization layer.

## Install

Install the published package from PyPI:

```bash
pip install rlprof
```

## System Requirements

- Linux
- NVIDIA GPU
- CUDA driver
- `nsys`
- `vllm`

## Build From Source

For local development or source builds, install the required system packages first:

- CMake 3.28 or newer
- C++20 compiler
- SQLite development headers

Then build the project:

```bash
cmake -S . -B build
cmake --build build --parallel
```

Run the test suite:

```bash
ctest --test-dir build --output-on-failure
```

Install from the local source tree:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install .
```

## Release Notes

See [CHANGELOG.md](CHANGELOG.md).
