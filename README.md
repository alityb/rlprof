# rlprof

`rlprof` profiles vLLM inference under RL-style rollout traffic.

It collects:
- CUDA kernel timing from `nsys`
- vLLM `/metrics` samples
- traffic shape statistics
- optional GPU kernel benchmark results

It stores the results in SQLite and prints raw numbers.

## Install

PyPI:

```bash
pip install rlprof
```

Bench extras:

```bash
pip install 'rlprof[bench]'
```

From source:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install .
```

## System Requirements

- Linux
- NVIDIA GPU
- CUDA driver
- `nsys`
- `vllm`

For source builds:

- `cmake >= 3.28`
- C++20 compiler
- SQLite development headers

## Build From Source

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

## Quick Start

Profile a local server lifecycle:

```bash
./build/rlprof profile \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --prompts 1 \
  --rollouts 1 \
  --min-tokens 8 \
  --max-tokens 8 \
  --input-len 16 \
  --output .rlprof/local_smoke
```

Report:

```bash
./build/rlprof report .rlprof/local_smoke.db
```

Validate:

```bash
./build/rlprof validate .rlprof/local_smoke.db
```

## Remote Workflow

Save a target:

```bash
./build/rlprof target add a10g \
  --host a10g-box \
  --workdir /srv/rlprof \
  --python /srv/rlprof/.venv/bin/python \
  --vllm /srv/rlprof/.venv/bin/vllm
```

Check and bootstrap it:

```bash
./build/rlprof doctor --target a10g
./build/rlprof target bootstrap a10g
```

Run a remote profile:

```bash
./build/rlprof profile \
  --target a10g \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --prompts 1 \
  --rollouts 1 \
  --min-tokens 8 \
  --max-tokens 8 \
  --input-len 16 \
  --output .rlprof/remote_smoke
```

Recover a finished remote run:

```bash
./build/rlprof recover \
  --target a10g \
  --remote-db /srv/rlprof/.rlprof/remote_smoke.db \
  --output .rlprof/remote_smoke_recovered.db
```

`rlprof` uses plain `ssh` and `scp`. Use an SSH alias or any other non-interactive SSH setup that those commands can use directly.

Example:

```sshconfig
Host a10g-box
  HostName gpu-host.example.com
  User ubuntu
  IdentityFile ~/.ssh/a10g.pem
  IdentitiesOnly yes
```

If remote `python3` and `vllm` are already on `PATH`, `--python` and `--vllm` are optional.

## Attach Modes

Metrics-only attach:

```bash
./build/rlprof profile \
  --attach http://127.0.0.1:8000 \
  --prompts 8 \
  --rollouts 4 \
  --min-tokens 128 \
  --max-tokens 256 \
  --input-len 128 \
  --output .rlprof/attach_metrics
```

Known-process attach:

```bash
./build/rlprof profile \
  --attach http://127.0.0.1:8000 \
  --attach-pid 215839 \
  --prompts 1 \
  --rollouts 1 \
  --min-tokens 8 \
  --max-tokens 8 \
  --input-len 16 \
  --fetch-nsys-rep \
  --output .rlprof/attach_process
```

When native `nsys` PID attach is unavailable, `rlprof` uses the strongest same-host fallback it can:
- direct PID attach
- traced clone on a free local port
- replace-and-restore on the original port

The same command works through a saved remote target:

```bash
./build/rlprof profile \
  --target a10g \
  --attach http://127.0.0.1:8000 \
  --attach-pid 215839 \
  --prompts 1 \
  --rollouts 1 \
  --min-tokens 8 \
  --max-tokens 8 \
  --input-len 16 \
  --output .rlprof/attach_remote
```

## Faster Repeated Profiling

### `profile --repeat`

Launch-mode `profile --repeat N` reuses one `vllm serve` process and one `nsys` session across all measured runs.

```bash
./build/rlprof profile \
  --model Qwen/Qwen3-8B \
  --prompts 8 \
  --rollouts 2 \
  --min-tokens 128 \
  --max-tokens 256 \
  --input-len 128 \
  --repeat 2 \
  --discard-first-run \
  --output .rlprof/qwen_repeat
```

Attach-by-process repeats use the same fast path. Metrics-only attach does not.

### Managed warm servers

For repeated local runs against the same model, keep one warm server alive:

```bash
./build/rlprof server start \
  --name qwen-warm \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --port 8030 \
  --max-model-len 2048

./build/rlprof profile \
  --server qwen-warm \
  --prompts 1 \
  --rollouts 1 \
  --min-tokens 8 \
  --max-tokens 8 \
  --input-len 16 \
  --output .rlprof/qwen_warm_run

./build/rlprof server stop qwen-warm
```

On the validated A10G path, the steady-state managed-server profile completed in `20.23s`; the equivalent cold single-run profile completed in `48.46s`.

### `soak-profile`

`soak-profile` repeats the same profile configuration and writes one profile per iteration:

```bash
./build/rlprof soak-profile \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --prompts 1 \
  --rollouts 1 \
  --min-tokens 8 \
  --max-tokens 8 \
  --input-len 16 \
  --iterations 10 \
  --pause-sec 5 \
  --validate-each \
  --output .rlprof/qwen_soak
```

Launch-mode and attach-by-process soak runs reuse one traced lifecycle when possible.

## Fetch Policy

Remote `profile` skips the heavy local `.nsys-rep` copy by default.

Default remote fetch:
- `.db`
- `.sqlite`
- `nvidia-smi` XML
- server log

Fetch the report too:

```bash
./build/rlprof profile \
  --target a10g \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --fetch-nsys-rep \
  --output .rlprof/remote_full
```

If you skipped the local `.nsys-rep`, use `recover` later or rerun with `--fetch-nsys-rep`.

## Bench

```bash
./build/rlprof bench \
  --kernel silu_and_mul \
  --shapes 64x4096,256x4096 \
  --warmup 20 \
  --n-iter 200 \
  --repeats 5 \
  --output .rlprof/bench_silu.json
```

The CUDA bench helper uses adaptive batched `torch.cuda.Event` timing by default. It targets a longer measured window per timed sample and reduces per-launch jitter on very small kernels.

Optional graph replay:

```bash
./build/rlprof bench \
  --kernel silu_and_mul \
  --shapes 64x4096,256x4096 \
  --warmup 20 \
  --n-iter 200 \
  --repeats 5 \
  --batch-ms-target 10 \
  --cuda-graph-replay on \
  --output .rlprof/bench_silu_graph.json
```

Remote bench:

```bash
./build/rlprof bench \
  --target a10g \
  --kernel silu_and_mul \
  --shapes 64x4096,256x4096 \
  --warmup 20 \
  --n-iter 200 \
  --repeats 5 \
  --output .rlprof/bench_remote_silu.json
```

Compare archived bench runs:

```bash
./build/rlprof bench-compare A.json B.json
```

## Core Commands

Profile and inspection:
- `rlprof profile`
- `rlprof report`
- `rlprof diff`
- `rlprof export`
- `rlprof validate`
- `rlprof artifacts`
- `rlprof trace`
- `rlprof manifest`
- `rlprof cleanup`

Bench and environment:
- `rlprof bench`
- `rlprof bench-compare`
- `rlprof doctor`
- `rlprof lock-clocks`
- `rlprof unlock-clocks`

Remote and orchestration:
- `rlprof target add|list|show|remove|bootstrap`
- `rlprof recover`
- `rlprof aggregate`
- `rlprof cluster-profile`
- `rlprof soak-profile`
- `rlprof server start|list|show|stop|prune`

Shell integration:
- `rlprof completion bash`
- `rlprof completion zsh`
- `rlprof completion fish`

## Interactive Mode

These commands fall back to sequential prompts when required arguments are missing:
- `rlprof`
- `rlprof profile`
- `rlprof bench`
- `rlprof report`
- `rlprof diff`
- `rlprof export`

Passing explicit flags skips prompts.

## Stored Data

Each profile stores:
- `meta`
- `kernels`
- `vllm_metrics`
- `vllm_metrics_summary`
- `traffic_stats`

Default artifact set:
- `.db`
- `.sqlite`
- `nvidia-smi` XML
- server log

Optional or derived artifacts:
- `.nsys-rep`
- CSV export files
- JSON export file
- manifest

## Validation Status

Validated on this codebase:
- local single-host profile
- remote single-host profile
- remote bootstrap
- remote recover
- remote bench
- `--fetch-nsys-rep`
- `profile --repeat`
- `soak-profile`
- managed warm local servers
- attach-by-process fallback on the tested single-host path
- packaging with `pip install .`
- raw kernel totals vs stored totals
- raw metric samples vs stored summaries

## Boundaries

- `profile --attach URL` is metrics-only.
- `profile --attach URL --attach-pid PID` needs host-local visibility into the source process and a clonable `vllm serve` command line when native PID attach is unavailable.
- `profile --target TARGET --attach URL --attach-pid PID` runs the same attach logic on the target host. The attach URL must refer to the same host-local service on that GPU machine.
- `cluster-profile` still needs real validation on distinct GPU hosts.
- `aggregate` merges completed profiles after the fact. It does not create synchronized traces by itself.
- kernel categories are conservative buckets; raw kernel names are authoritative.

## Test Commands

Standard suite:

```bash
ctest --test-dir build --output-on-failure
```

Heavier non-GPU stress pass:

```bash
ctest --test-dir build --output-on-failure --schedule-random --repeat until-fail:25
```

## Release Notes

See [CHANGELOG.md](CHANGELOG.md).
