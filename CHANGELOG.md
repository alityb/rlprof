# Changelog

## v0.1.1 - 2026-04-01

Packaging and release automation update.

Highlights:

- PyPI source distribution cleanup so local virtualenvs, build outputs, and macOS metadata files do not leak into published artifacts
- Linux wheel build configuration via `cibuildwheel` for CPython 3.10, 3.11, and 3.12 on `manylinux_2_28`
- GitHub Actions release workflow for building `sdist` plus Linux wheels and publishing to PyPI via trusted publishing on GitHub Releases

## v0.1.0 - 2026-03-31

Initial public release.

Highlights:

- local and remote `profile`, `report`, `diff`, `export`, `validate`, `artifacts`, `trace`, and `doctor`
- interactive prompt flows alongside the existing flag-based CLI
- real GPU `bench` backed by `torch.cuda.Event`, archived JSON output, and `bench-compare`
- SSH target registry, bootstrap, recover, and remote single-host profiling
- profile stability mode, clock helpers, manifest generation, cleanup, soak runs, and cluster rollup
- non-GPU CI and self-hosted GPU smoke workflow
- controller-verified remote workflow using saved SSH targets, `target bootstrap`, `bench --target`, `profile --target`, and `recover`
- `profile --attach ... --attach-pid ...` support that now uses the best available host-local tracing path: native PID attach when `nsys` exposes it, clone-under-trace when a second local copy fits, and replace-and-restore when a single GPU cannot hold both copies
- attach-by-process fast reuse for `profile --repeat N` and `soak-profile`, so repeated attach runs reuse one traced replacement lifecycle instead of relaunching the fallback per iteration
- stricter `cluster-profile` host handling with distinct-host enforcement by default and explicit `--allow-duplicate-hosts` loopback testing
- fast reuse path for `profile --repeat N` and `soak-profile` when `rlprof` launches the server, with one server startup and one `nsys` session reused across iterations
- local managed warm-server workflow: `server start/list/show/stop` and `profile --server NAME`, reusing one loaded `vllm serve` and one `nsys` session across separate commands
- managed-server hardening: explicit `--max-model-len`, stale-state pruning, stale-lock recovery, and listener PID fallback from `lsof` to `ss`
- publish-ready package metadata for the `pip install rlprof` release surface, plus a rewritten README aligned to the controller-verified local/remote workflows
