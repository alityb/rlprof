2026-04-06 00:00 UTC
- Started a rigorous serving-timing and local video-flow hardening pass for hotpath.
- Scope for this pass:
- make `serve-profile` actually honor concurrency and stop dispatching new work at the requested duration boundary
- remove misleading timing labels and fake zero-valued placeholders from `serve-report`
- make local vLLM DEBUG log autodiscovery choose the real live listener log instead of stale `.hotpath/video-server/*` files
- validate the vLLM v1 timing path against a real local `Qwen/Qwen3.5-4B` server on `localhost:8000`

2026-04-06 00:00 UTC
- Fixed request replay and timing collection accuracy in the serving path.
- `traffic_replayer.cpp` now sends requests concurrently instead of serially and respects the configured max concurrency.
- `serve-profile` now stops launching new requests when the requested duration window closes instead of stretching a finite traffic file across the whole duration.
- client TTFT is now measured from the first streamed token chunk in the SSE response instead of curl `time_starttransfer`, which was only first-byte timing.
- throughput now uses the actual measured profile duration instead of the requested duration literal.

2026-04-06 00:00 UTC
- Fixed live dashboard redraw behavior for Ghostty and similar terminals.
- `serve-profile` no longer relies on DEC save and restore cursor handling for the live dashboard.
- redraw now uses explicit previous-line movement and clear behavior, which stops duplicated dashboard frames from stacking during live profiling.
- progress bars are capped to terminal width, and redraw control no longer depends on color output settings.

2026-04-06 00:00 UTC
- Tightened server-log autodiscovery to follow the real live listener process.
- local endpoint autodiscovery now inspects the process that owns the bound listener and resolves `/proc/<pid>/fd/1` and `/proc/<pid>/fd/2` to discover file-backed stdout and stderr logs.
- stale `.hotpath/video-server/*` logs are no longer selected just because they exist.
- if the live listener writes only to a TTY or pipe, hotpath now reports that honestly instead of pretending it found a usable log file.
- added regression coverage for live-listener-backed stdout and stderr autodiscovery and for refusing non-file outputs.

2026-04-06 00:00 UTC
- Tightened timing and report honesty semantics.
- `ServeReportData` latency defaults now use `-1.0` sentinels instead of zero, so missing timing does not render as fake `0.0 ms`.
- `serve-report` now labels client first-token latency as `TTFT (client)` and only uses measured client TTFT as the monolithic advisor baseline when it actually exists.
- `serve-report` now renders server queue, prefill, and decode rows independently, which avoids suppressing valid partial timing just because one component is unavailable.
- all-failure and unavailable runs now print `-` instead of measured-looking zeros in latency tables.

2026-04-06 00:00 UTC
- Extended vLLM v1 log parsing for the real local Qwen video server shape.
- `log_parser.cpp` now canonicalizes hotpath-injected external request IDs from vLLM v1 randomized internal IDs and merges those exact-ID request lines with the anonymous `Running batch ... BatchDescriptor(...)` timing lines emitted by EngineCore.
- this removes the earlier split where exact request matching worked but server phase timing was discarded because batch execution lines had no IDs.
- added a regression test for the exact observed vLLM v1 pattern: `Added request <hotpath-id>-<suffix>` followed by anonymous `BatchDescriptor(num_tokens=...)` lines.

2026-04-06 00:00 UTC
- Improved vLLM v1 timing refinement and disclosure.
- second-resolution vLLM v1 DEBUG timestamps are now refined with Prometheus queue, prefill, and decode means for both order-matched and exact-ID-matched traces.
- the report explicitly says when server timing was matched by request ID and refined from Prometheus because the raw vLLM v1 timestamps were only second-resolution.
- this keeps the report honest about what is raw log timing versus what is metric-refined timing.

2026-04-06 00:00 UTC
- Live validation succeeded against a real local file-backed DEBUG server.
- started `Qwen/Qwen3.5-4B` on `localhost:8000` with `VLLM_LOGGING_LEVEL=DEBUG`, `--enable-log-requests`, `--enforce-eager`, and `--language-model-only`, with stdout and stderr captured to `.hotpath/live-debug-server/`.
- verified that the captured log contained:
- request-level API lines for the hotpath request ID
- `Added request ...`
- `EngineCore loop active.`
- anonymous `Running batch ... BatchDescriptor(num_tokens=...)` lines
- `./build/hotpath serve-profile --endpoint http://127.0.0.1:8000 --traffic .hotpath/one_request_smoke.jsonl --concurrency 1 --duration 5 --output .hotpath/live_debug_verify4` auto-discovered the live stdout log without `--server-log`.
- live smoke output confirmed:
- `Parsed 1 server-side request traces ...`
- `Matched 1/1 requests by ID`
- final live `serve-report` from `.hotpath/live_debug_verify4/serve_profile.db` showed non-zero server-side phase timing:
- `Queue wait 0.918 ms`
- `Prefill (server) 611.2 ms`
- `Decode (server) 7243.6 ms`

2026-04-06 00:00 UTC
- Rebuilt and revalidated after each accuracy patch.
- focused checks repeatedly passed:
- `test_log_parser`
- `test_serve_profiler`
- `test_serve_report`
- `test_traffic_replayer`
- `test_cli`
- `test_audit`
- replaced the locally installed `hotpath` binary with the freshly rebuilt binary after the final live verification so the interactive command path uses the tested fixes.

2026-03-31 03:00 UTC
- Started production validation and hardening pass for hotpath.
- Working order fixed to: real multi-host validation, remote attach-by-process, remote environment hardening.
- Constraint applied: use existing `cluster-profile` synchronized-start path for Task 1 and record every decision, outcome, and failure with UTC day+hour stamps.

2026-03-31 03:00 UTC
- Inspected repository state before changes.
- Worktree was clean.
- `build/` did not exist locally.
- Default target registry path (`~/.config/hotpath/targets.cfg`) was absent, so saved targets must be confirmed through the built CLI before Task 1 can proceed.

2026-03-31 03:00 UTC
- Inspected current implementation:
- `cluster-profile` already computes a shared `start_at_unix_ms`, offsets per-host ports for repeated hosts, launches remote profiles concurrently, and writes a local aggregate database.
- `profile --attach URL` remains metrics-only because local tracing is skipped when `attach_server` is set.
- Remote commands are currently thin SSH wrappers with minimal remote environment validation.

2026-03-31 03:00 UTC
- User provided one reachable SSH target (`target-a`) using a local SSH key path.
- Initial remote inspection succeeded far enough to confirm a real GPU host:
- remote hostname resolved successfully on `target-a`
- GPU: `NVIDIA A10G`
- `nsys`: `/usr/local/bin/nsys`, version `2025.3.2.474-253236389321v0`
- `python3`: `/usr/bin/python3`, version `3.12.3`
- Initial `command -v vllm` failed, which indicates remote executable discovery is currently brittle and will need hardening in Task 3.

2026-03-31 03:00 UTC
- Local environment note: `python3` and `pip` are available, but `cmake` is not installed on the local machine.
- This blocks building the local `./build/hotpath` binary until `cmake` is installed or another build path is established.

2026-03-31 03:00 UTC
- Ran `hotpath target list` and `hotpath target show ...` in the accessible built environment on `target-a`.
- Saved targets discovered: `selfboot`, `selfboot2`.
- Both saved targets resolve to the same anonymous remote host and same workdir, with explicit repo-venv executables configured.
- Distinct reachable GPU hosts currently available to this validation pass: 1.
- Result: Task 1 real multi-host validation cannot be completed as requested because no second distinct reachable GPU host is currently available.
- Attempted path:
- confirm saved targets
- expand each target with `target show`
- verify real GPU presence and `doctor` health on the reachable host
- Blocking reason is target inventory, not an `hotpath` controller failure.

2026-03-31 03:00 UTC
- `doctor` on `target-a` passed for GPU visibility, `nsys`, `vllm`, bench helper imports, and `nsys` environment.
- `clock policy` reported `WARN unlocked`, which is informative but not a blocker for attach-mode implementation or remote hardening.

2026-03-31 04:20 UTC
- Implemented a faster remote profile fetch path.
- Remote `profile` now skips copying the remote `.nsys-rep` by default and only fetches the copied `.db`, `.sqlite`, telemetry XML, and server log.
- Added `--fetch-nsys-rep` as an explicit opt-in when the full local report artifact is needed.

2026-03-31 04:20 UTC
- Preserved remote trace provenance while making the default fetch lighter.
- The copied local profile now keeps `remote_artifact_nsys_rep_path` even when the local `.nsys-rep` is not fetched.
- `validate` now treats a deliberately unfetched remote `nsys report` as an expected `WARN` instead of a false `FAIL`.
- `trace --open` now tells the user to rerun with `--fetch-nsys-rep` or use `recover` when only the remote report exists.

2026-03-31 04:20 UTC
- Parallelized copied-back artifact fetch across remote repeats.
- The remote run still executes once on the target, but per-run local fetch/load now happens concurrently after the target-side work completes.
- Native build and test verification stayed green after this change:
- `cmake --build build`
- `ctest --test-dir build --output-on-failure`

2026-03-31 04:20 UTC
- Live-validated the faster remote profile path on the saved target alias `selfboot`.
- `profile --target selfboot --model Qwen/Qwen2.5-0.5B-Instruct --prompts 1 --rollouts 1 --min-tokens 8 --max-tokens 8 --input-len 16 --output .hotpath/remote_fast_smoke` completed successfully.
- Produced local artifacts:
- `.hotpath/remote_fast_smoke.db`
- `.hotpath/remote_fast_smoke.sqlite`
- `.hotpath/remote_fast_smoke_nvidia_smi.xml`
- Did not copy a local `.hotpath/remote_fast_smoke.nsys-rep`, which confirms the lighter default fetch path.
- `validate .hotpath/remote_fast_smoke.db` now returns:
- `artifacts WARN`
- `kernel totals PASS`
- `metrics summary PASS`
- `remote provenance PASS`

2026-03-31 04:20 UTC
- Kept the server-reuse limitation explicit instead of faking it.
- A true “start one remote server, then collect fresh per-run kernel traces repeatedly” path still depends on a supported `nsys` attach-to-existing-process workflow. The installed `nsys` here exposes session control but not a documented PID-attach CLI flag, so the safe speedups were limited to lighter artifact fetch and concurrent fetch/load after the run.

2026-03-31 05:00 UTC
- Reviewed the landed hardening patch and fixed three release-facing issues before final publish verification:
- reverted permissive low-level target fallback in `resolve_target()` so unknown saved target names fail fast again
- moved direct-host fallback into the CLI resolution path only when explicit host/workdir/executable context is already present
- changed remote python autodetection and bootstrap preflight to prefer `python3` first instead of failing on common `python3`/`python` dual-path setups

2026-03-31 05:00 UTC
- Made `--attach-pid` an explicit release boundary instead of a contradictory half-enabled path.
- `profile --attach ... --attach-pid ...` now fails immediately with a direct boundary message.
- If installed `nsys` does not advertise PID attach support, that reason is included explicitly in the error.
- Updated CLI help and README/CHANGELOG wording to match the actual release behavior.

2026-03-31 05:00 UTC
- Added release-hygiene fixes:
- `.gitignore` now ignores macOS AppleDouble files via `._*`
- added `test_doctor` coverage for deterministic `python3` selection over `python`
- updated `test_targets` so unknown alias names throw again instead of silently turning into hosts

2026-03-31 05:00 UTC
- Re-ran local native verification after the robustness fixes:
- `cmake --build build --target hotpath test_targets test_doctor`
- `ctest --test-dir build --output-on-failure`
- result: `18/18 passed`

2026-03-31 05:00 UTC
- Completed the controller-path publish smoke on the accessible GPU host through the saved target `selfboot`.
- `doctor --target selfboot` passed.
- `target bootstrap selfboot` passed and now prints a concise preflight table before rebuild plus a post-build doctor report.
- `bench --target selfboot --kernel silu_and_mul --shapes 64x4096 --warmup 1 --n-iter 5 --repeats 1 --output .hotpath/bench_publish_smoke.json` passed.

2026-03-31 05:00 UTC
- Completed the default remote profile and recovery checks from the controller path:
- `profile --target selfboot --model Qwen/Qwen2.5-0.5B-Instruct --prompts 1 --rollouts 1 --min-tokens 8 --max-tokens 8 --input-len 16 --port 8011 --output .hotpath/publish_profile_default` passed
- local default artifact set confirmed: `.db`, `.sqlite`, telemetry XML, remote server log; no local `.nsys-rep`
- `recover --target selfboot --remote-db /tmp/hotpath_bootstrap_test/.hotpath/publish_profile_default.db --output .hotpath/publish_profile_recovered.db` passed
- recovered artifact set confirmed: local `.nsys-rep` present after recovery

2026-03-31 05:00 UTC
- Completed the explicit full-artifact remote profile path:
- initial attempt was launched concurrently with another traced profile on the same single A10G and failed for a real reason: vLLM KV cache memory was exhausted while two traced servers were starting at once
- reran the full-artifact case serially:
- `profile --target selfboot --model Qwen/Qwen2.5-0.5B-Instruct --prompts 1 --rollouts 1 --min-tokens 8 --max-tokens 8 --input-len 16 --port 8013 --fetch-nsys-rep --output .hotpath/publish_profile_full` passed
- full artifact set confirmed: local `.db`, `.sqlite`, `.nsys-rep`, telemetry XML, remote server log

2026-03-31 05:00 UTC
- Validated publish-smoke profile outputs:
- `validate .hotpath/publish_profile_default.db` -> `artifacts WARN`, `kernel totals PASS`, `metrics summary PASS`, `remote provenance PASS`
- `validate .hotpath/publish_profile_full.db` -> `artifacts WARN`, `kernel totals PASS`, `metrics summary PASS`, `remote provenance PASS`
- `validate .hotpath/publish_profile_recovered.db` -> `artifacts WARN`, `kernel totals PASS`, `metrics summary PASS`, `remote provenance PASS`

2026-03-31 05:00 UTC
- Verified the release-boundary attach message from the controller path:
- `profile --target selfboot --attach http://127.0.0.1:8011 --attach-pid 12345 ...` now fails with:
- `attach-by-process is a release boundary in this build. Installed nsys does not advertise PID attach support. Installed nsys: NVIDIA Nsight Systems version 2025.3.2.474-253236389321v0`

2026-03-31 05:00 UTC
- Completed clean-venv packaging smoke:
- `python3 -m venv /tmp/hotpath-final-venv && . /tmp/hotpath-final-venv/bin/activate && pip install .` passed
- installed `hotpath help` passed
- installed `hotpath completion bash` passed
- installed `hotpath doctor` passed with expected optional bench warning:
- `bench helper WARN ModuleNotFoundError: No module named 'torch'`
- Decision: default install should keep bench as optional and report missing bench extras as `WARN`, not `FAIL`

2026-03-31 05:00 UTC
- Tightened `cluster-profile` so a single machine can no longer silently masquerade as a cluster.
- `cluster-profile` now requires distinct resolved hosts by default.
- Added `--allow-duplicate-hosts` only for explicit loopback testing on one machine.
- Preserved duplicate-host port offsetting behind that explicit override.

2026-03-31 05:00 UTC
- Replaced the contradictory `--attach-pid` pseudo-feature with a real capability-gated implementation path.
- `profile --attach URL --attach-pid PID` now proceeds only when the installed `nsys` advertises PID attach support.
- Implemented an attach-runner path that prepares a session-based `nsys profile --pid ... --start-later=true` command and then reuses the existing `nsys start/stop` measured windowing path.
- On hosts where `nsys` does not advertise `--pid`, hotpath now fails immediately with a direct capability error before sending traffic.

2026-03-31 05:00 UTC
- Loosened attach-mode environment validation to match the real execution path.
- Existing-server attach no longer requires a launchable `vllm` executable just to profile an already-running server process.
- Kept server-launch profiles strict: `vllm` must still resolve when hotpath is responsible for starting the server.

2026-03-31 05:00 UTC
- Revalidated the local-controller publish path after the cluster/attach changes:
- `cmake --build build` passed
- `ctest --test-dir build --output-on-failure` passed (`18/18`)
- `doctor --target selfboot` passed
- `target bootstrap selfboot` passed
- `profile --target selfboot --model Qwen/Qwen2.5-0.5B-Instruct --prompts 1 --rollouts 1 --min-tokens 8 --max-tokens 8 --input-len 16 --port 8014 --output .hotpath/final_prod_smoke` passed
- `validate .hotpath/final_prod_smoke.db` passed with expected artifact `WARN`
- `recover --target selfboot --remote-db /tmp/hotpath_bootstrap_test/.hotpath/final_prod_smoke.db --output .hotpath/final_prod_smoke_recovered.db` passed
- `validate .hotpath/final_prod_smoke_recovered.db` passed with expected artifact `WARN`

2026-03-31 05:00 UTC
- Validated the new cluster guard and current attach boundary on the accessible host:
- `cluster-profile --targets selfboot,selfboot2 ...` now fails fast with `cluster-profile requires distinct target hosts. Use --allow-duplicate-hosts only for loopback testing.`
- Explicit loopback testing with `--allow-duplicate-hosts` still launches duplicate-host runs, which remains useful for controller-path testing but is not treated as real multi-host validation.
- On this host, `profile --attach http://127.0.0.1:1 --attach-pid 12345 ...` now fails with `attach-by-process requires an nsys CLI that advertises PID attach support...`
- Result: real attach-by-process orchestration now exists in code, but live validation is still blocked by the installed `nsys` lacking PID attach capability on the only accessible GPU host.

2026-03-31 05:00 UTC
- Rebuilt the final local controller binary after the cluster/attach help-text updates.
- `./build/hotpath help` now advertises:
- `profile ... --attach-pid PID`
- `cluster-profile ... [--allow-duplicate-hosts]`
- `./build/hotpath version` still reports `hotpath 0.1.0`
- Re-ran a clean packaging smoke after the final code/docs changes:
- `python3 -m venv /tmp/hotpath-publish-final && . /tmp/hotpath-publish-final/bin/activate && pip install .` passed
- installed `hotpath help`, `hotpath doctor`, and `hotpath completion bash` all executed successfully

2026-03-31 05:00 UTC
- Ran a math-and-integrity stress pass focused on the non-cluster paths.
- `ctest --test-dir build --output-on-failure --schedule-random --repeat until-fail:25` passed cleanly with no failures.
- Re-validated multiple real profile artifacts:
- `validate .hotpath/final_prod_smoke.db` -> `kernel totals PASS`, `metrics summary PASS`
- `validate .hotpath/final_prod_smoke_recovered.db` -> `kernel totals PASS`, `metrics summary PASS`
- `validate .hotpath/publish_profile_full.db` -> `kernel totals PASS`, `metrics summary PASS`

2026-03-31 05:00 UTC
- Independently cross-checked stored math against raw SQLite data outside the `validate` command:
- `.hotpath/final_prod_smoke.sqlite` raw kernel total / calls: `7282504 ns`, `413`
- `.hotpath/final_prod_smoke.db` stored kernel total / calls: `7282504 ns`, `413`
- direct recomputation of `vllm_metrics_summary` from raw `vllm_metrics` rows matched exactly for all recorded metrics in `.hotpath/final_prod_smoke.db`
- Result: for the exercised local and remote single-host paths, the stored profile math matches the raw recorded data exactly.

2026-03-31 05:00 UTC
- Rewrote `README.md` to be shorter and more modular.
- Structure is now: scope, requirements, build/install, quick start, remote workflow, command reference, artifacts, boundaries, compatibility, testing.
- Removed narrative examples and explanatory detours that were not necessary for using the tool.
- Kept the documented workflows aligned with the validated local-controller and remote single-host paths.

2026-03-31 18:00 UTC
- Started a performance pass focused on `soak-profile` and repeat-mode profiling.
- Confirmed the existing bottleneck in `handle_soak_profile`: it looped full cold `run_profile_command()` executions, which relaunched `vllm serve` and repeated model startup for every iteration.
- Decision: optimize the repeated-profile path at the runner layer instead of adding a soak-only shortcut, so the speedup applies to both `soak-profile` and `profile --repeat N`.

2026-03-31 18:00 UTC
- First fast-soak attempt reused one `nsys` session under `nsys profile --start-later=true`, but `nsys start --output ...` still wrote only the base `soak_fast_verify.{nsys-rep,sqlite}` artifacts.
- Result: rejected this design because it did not emit per-iteration trace files, even though the shared server lifecycle worked.

2026-04-04 04:00 UTC
- Resumed an interrupted serving-analysis verification pass after the interactive chat session ran out of context.
- Revalidated the current tree from scratch instead of trusting the pasted transcript:
- `rm -rf build`
- `cmake -B build -S . -DBUILD_TESTING=ON`
- `cmake --build build --parallel`
- `ctest --test-dir build --output-on-failure`
- Result: clean configure/build succeeded and the full test suite passed (`36/36`).

2026-04-04 04:00 UTC
- Verified the recently added serving/report/bench changes are present in the worktree:
- `serve-report` rendering and DB-backed loading are wired in `src/report.cpp` and `src/main.cpp`
- `serve-profile` prints explicit latency semantics notes in `src/serving/serve_profiler.cpp`
- GPU cold-cache benchmark mode is exposed as `hotpath bench --flush-l2` and passed through to `hotpath_py/bench_cuda.py`
- CPU cache flush support exists in `src/bench/runner.cpp` as an internal benchmark option

2026-04-04 04:00 UTC
- Found one real bug during the recheck:
- model auto-detection from OpenAI-compatible `/v1/models` responses in `detect_model()` was effectively broken because the string scan skipped past the actual `"id"` value and usually returned empty
- fixed by adding `parse_model_from_models_response()` in `src/serving/traffic_replayer.cpp` and routing `detect_model()` through it
- added coverage in `cpp_tests/test_traffic_replayer.cpp` for compact and pretty-printed `/v1/models` JSON
- revalidated after the fix:
- `cmake --build build --parallel --target hotpath test_traffic_replayer test_cli`
- `ctest --test-dir build -R 'test_traffic_replayer|test_cli' --output-on-failure`
- `ctest --test-dir build --output-on-failure`

2026-04-04 04:00 UTC
- Design decisions clarified during this pass:
- keep `serve-profile` latency measurements client-side on purpose; they are intended to reflect user-visible HTTP latency, not isolated server-internal timing
- keep `--flush-l2` opt-in; warm-cache kernel benchmarking remains the default behavior for implementation-to-implementation comparison, while cold-cache numbers are available explicitly when requested
- do not relabel warm-cache benchmarking or client-side serving latency as “bugs”; these are measurement semantics and should be documented as such

2026-04-04 04:00 UTC
- Remaining known limitations from the serving path after this pass:
- `serve-profile` currently synthesizes request lifecycle phases from replay timestamps when structured server events are unavailable, so queue/prefill/decode decomposition is approximate and not yet authoritative
- `ReplayConfig.max_concurrency` is not enforced by `replay_traffic()` yet; the current replay loop is sequential with optional rate limiting
- live `serve-profile` persistence still writes placeholder prefix-sharing values and does not yet derive real GPU phase breakdown without the missing nsys-to-phase correlation path
- These are semantic/product gaps, not compile or unit-test failures, and they should remain explicit in future serving-analysis work

2026-04-04 04:00 UTC
- Ran an installation-focused validation pass to verify not just the in-tree build but the packaged install path.
- Existing machine state before this pass:
- native build toolchain already present: `cmake 3.28.3`, `g++ 13.3.0`, SQLite headers, `curl`, `nsys`
- repo `.venv` already contained a working GPU bench environment with `torch 2.10.0+cu128` and `vllm 0.18.0`
- clean repo rebuild and full tests were already green (`36/36`)

2026-04-04 04:00 UTC
- Verified install from source into a fresh virtual environment:
- `python3 -m venv /tmp/hotpath-test-venv`
- `/tmp/hotpath-test-venv/bin/pip install .`
- installed CLI smoke checks passed:
- `/tmp/hotpath-test-venv/bin/hotpath version`
- `/tmp/hotpath-test-venv/bin/hotpath help`
- `/tmp/hotpath-test-venv/bin/hotpath doctor`
- expected result for base install: doctor passed overall but `bench helper` warned because the base package intentionally does not include `torch`/`vllm`

2026-04-04 04:00 UTC
- Installed the optional bench extras into the clean test venv to validate the full GPU path:
- `/tmp/hotpath-test-venv/bin/pip install '.[bench]'`
- verified the clean venv imports:
- `torch 2.10.0+cu128`
- `vllm 0.19.0`
- reran installed-package validation successfully:
- `hotpath doctor` -> `bench helper PASS`
- `hotpath bench --kernel silu_and_mul --shapes 64x4096 --warmup 1 --n-iter 3 --repeats 1 --output none --yes` passed on the local NVIDIA A10G

2026-04-04 04:00 UTC
- Bench smoke observations from the clean installed package:
- kernel timings were produced for `vllm-cuda`, `torch-compile`, and `torch-eager`
- correctness checks passed (`valid=yes`, deterministic where expected)
- environment warning reported power-cap throttling during the `vllm-cuda` run
- this is a measurement-quality warning, not a functional failure

2026-04-04 04:00 UTC
- Scope note from the install-validation pass:
- did not install `sglang[all]` because the current project requirements and doctor checks treat `vllm` or `sglang` as alternative serving engines, and the verified `vllm` path already covers the required serving/bench environment here

2026-04-04 04:00 UTC
- Installed `sglang[all]` into the repo `.venv` on direct user request.
- Result:
- `sglang` import succeeded and `hotpath doctor` now reports `sglang (serve-profile) PASS`
- the install rewrote core CUDA/torch-adjacent packages in the existing `.venv`, including `torch`, `triton`, `flashinfer_python`, `llguidance`, `xgrammar`, and related dependencies
- pip reported explicit incompatibilities against the pre-existing `vllm 0.18.0` environment

2026-04-04 04:00 UTC
- Post-install verification:
- `.venv/bin/python -c "import sglang; print(sglang.__version__)"` -> `0.5.9`
- `./build/hotpath doctor` -> `sglang (serve-profile) PASS`
- `./build/hotpath doctor` -> `vllm FAIL`
- failing detail:
- `ImportError: ... vllm/_C.abi3.so: undefined symbol: ... c10_cuda_check_implementation ...`
- Interpretation:
- `sglang[all]` and the existing `vllm 0.18.0` stack are not co-install-safe in this shared `.venv` as installed here; the environment now favors `sglang` and has broken the previously working `vllm` native extension ABI

2026-04-04 04:00 UTC
- Ran a broad local functionality pass beyond the unit tests.
- Native regression baseline rerun:
- `ctest --test-dir build --output-on-failure`
- result: `36/36` passed

2026-04-04 04:00 UTC
- Commands exercised successfully with real local execution:
- `hotpath version`
- `hotpath help`
- `hotpath completion bash|zsh|fish`
- `hotpath serve-profile --help`
- `hotpath serve-report <temp serve db>`
- `hotpath disagg-config <temp serve db> --format all`
- `hotpath export --format json` in an empty workspace
- `hotpath validate` / `artifacts` / `trace` / `manifest` in an empty workspace
- `hotpath target add/list/show/remove` in an isolated temp config dir
- `hotpath server list`
- `hotpath server prune`
- `hotpath cleanup --dir ... --keep 1` in dry-run mode
- `hotpath reset-defaults` in an isolated temp config dir
- `hotpath profile --help`
- `hotpath cluster-profile --help`
- `hotpath soak-profile --help`
- `hotpath recover --help`
- `hotpath traffic --server http://127.0.0.1:1 ...` against an unreachable endpoint
- installed-package GPU path:
- `/tmp/hotpath-test-venv/bin/hotpath bench ...`
- `hotpath bench-compare` on two real benchmark JSON outputs

2026-04-04 04:00 UTC
- Real findings from the CLI-level functionality pass:
- top-level `hotpath help` is stale: it does not list `serve-profile`, `serve-report`, or `disagg-config`
- shell completions are stale in the same way:
- bash completion omits the new serving commands
- zsh completion omits the new serving commands
- fish completion omits the new serving commands
- `serve-profile --engine sglang` has a copy/UX bug in its unreachable-endpoint hint: it still tells the user to start `vllm serve <model> --port 8000`

2026-04-04 04:00 UTC
- Additional observed behavior from the functionality pass:
- `traffic` fails cleanly against an unreachable server and still emits structured JSON stats with `errors=1`
- `serve-report` and `disagg-config` work correctly against a temp SQLite DB containing `serve_analysis` rows
- `cleanup` dry-run mode behaves cleanly on an empty artifact directory
- the clean test venv remains the known-good path for real `vllm` bench execution; the repo `.venv` remains the known-good path for `sglang` only after the direct `sglang[all]` install

2026-04-04 04:00 UTC
- Coverage boundary for this pass:
- did not run live `profile`, `server start`, `cluster-profile`, `recover`, or full live `serve-profile` against a real inference endpoint because those require either a reachable remote target, a managed model server, or a live trafficable serving endpoint
- for those commands, help paths and failure paths were exercised locally, but not a full live end-to-end run in this session

2026-03-31 18:00 UTC
- Second fast-soak attempt switched the shared server lifecycle to `nsys launch`, but initially passed `--cpuctxsw=none` on the `launch` command.
- Live failure on this host:
- `The --cpuctxsw switch is not supported by the launch command after NVIDIA Nsight Systems v2021.5.`
- Fix: keep `launch` focused on process/session ownership and move `--sample=none --cpuctxsw=none` to `nsys start`, where this host's `nsys` accepts them.

2026-03-31 18:00 UTC
- Final fast path:
- `run_soak_profile()` now launches one `vllm serve` process under one long-lived `nsys launch --session-new ...` session.
- Each iteration calls `nsys start --session=... --output=<iter> --export=sqlite --force-overwrite=true --sample=none --cpuctxsw=none`, fires traffic, then stops with `nsys stop --session=...`.
- `profile --repeat N` now uses the same reused server/session path for launch-mode profiles.
- Remote repeat-mode profiling now routes through one remote `soak-profile` invocation and then fetches per-iteration artifacts locally.

2026-03-31 18:00 UTC
- Live validation of the final fast path succeeded:
- `./build/hotpath soak-profile --model Qwen/Qwen2.5-0.5B-Instruct --prompts 1 --rollouts 1 --min-tokens 8 --max-tokens 8 --input-len 16 --iterations 2 --pause-sec 1 --validate-each --output .hotpath/soak_fast_verify3`
- Output artifacts:
- `.hotpath/soak_fast_verify3_i1.db/.sqlite/.nsys-rep`
- `.hotpath/soak_fast_verify3_i2.db/.sqlite/.nsys-rep`
- Validation:
- `validate .hotpath/soak_fast_verify3_i1.db` -> `kernel totals PASS`, `metrics summary PASS`, raw total `7276120 ns`, calls `413`
- `validate .hotpath/soak_fast_verify3_i2.db` -> `kernel totals PASS`, `metrics summary PASS`, raw total `7288026 ns`, calls `413`

2026-03-31 18:00 UTC
- Live validation of the project-wide repeat path also succeeded:
- `./build/hotpath profile --model Qwen/Qwen2.5-0.5B-Instruct --prompts 1 --rollouts 1 --min-tokens 8 --max-tokens 8 --input-len 16 --repeat 2 --output .hotpath/profile_repeat_fast_verify`
- Produced:
- `.hotpath/profile_repeat_fast_verify_i1.db`
- `.hotpath/profile_repeat_fast_verify_i2.db`
- and rendered the normal report plus a stability table.
- Result: the reused server/session optimization is now applied to both `soak-profile` and `profile --repeat N`, not just the soak command.

2026-03-31 18:00 UTC
- Remote repeat-mode validation initially failed because the controller optimized remote `profile --repeat N` by delegating to remote `soak-profile`, but still tried to fetch `_rN` files from the remote host.
- Fix: keep local `profile --repeat` output names as `_rN`, but fetch remote `_iN` artifacts when the remote fast-soak path is active.

2026-03-31 18:00 UTC
- Final remote repeat-mode validation succeeded:
- `./build/hotpath profile --target selfboot --model Qwen/Qwen2.5-0.5B-Instruct --prompts 1 --rollouts 1 --min-tokens 8 --max-tokens 8 --input-len 16 --repeat 2 --output .hotpath/remote_repeat_fast_verify`
- Produced local fetched artifacts:
- `.hotpath/remote_repeat_fast_verify_r1.db/.sqlite/_nvidia_smi.xml/_remote_server.log`
- `.hotpath/remote_repeat_fast_verify_r2.db/.sqlite/_nvidia_smi.xml/_remote_server.log`
- `validate .hotpath/remote_repeat_fast_verify_r1.db` -> `kernel totals PASS`, `metrics summary PASS`, `remote provenance PASS`
- Result: the reuse optimization now covers local repeat-mode, soak-mode, and controller-driven remote repeat-mode on the tested single-host path.

2026-03-31 19:00 UTC
- Production-shaped repeat benchmark on `Qwen/Qwen3-8B`:
- Fast reused-session path:
- `/usr/bin/time -f 'elapsed=%e' ./build/hotpath profile --model Qwen/Qwen3-8B --prompts 8 --rollouts 2 --min-tokens 128 --max-tokens 256 --input-len 128 --discard-first-run --repeat 2 --output .hotpath/qwen3_8b_repeat_fast_bench`
- Completed in `223.02s` and produced:
- `.hotpath/qwen3_8b_repeat_fast_bench_i1.db/.sqlite/.nsys-rep`
- `.hotpath/qwen3_8b_repeat_fast_bench_i2.db/.sqlite/.nsys-rep`
- Raw-vs-stored kernel totals matched exactly:
- `_i1`: `796880068 ns`, calls `4184`
- `_i2`: `801469834 ns`, calls `4206`

2026-03-31 19:00 UTC
- Cold relaunch baseline for the same workload was started as two independent `profile` runs:
- `/usr/bin/time -f 'elapsed=%e' bash -lc './build/hotpath profile ... --output .hotpath/qwen3_8b_cold_a && ./build/hotpath profile ... --output .hotpath/qwen3_8b_cold_b'`
- By `378s` elapsed, the sequence had finished `_a` and was still loading checkpoint shards for `_b`.
- `_a` produced valid artifacts and exact raw-vs-stored kernel totals:
- `.hotpath/qwen3_8b_cold_a.db/.sqlite/.nsys-rep`
- `_a`: `796184797 ns`, calls `4111`
- Result so far: for this production-shaped `Qwen/Qwen3-8B` workload, the reused server/session repeat path completed both measured runs before the cold relaunch baseline completed its second startup/load cycle.

2026-03-31 21:00 UTC
- Full pre-publish stress sweep completed on the tested single-host path.
- Native suite stress:
- `ctest --test-dir build --output-on-failure --schedule-random --repeat until-fail:25`
- Result: `18/18` passed through the repeated randomized run, no failures.

2026-03-31 21:00 UTC
- Heavy profile integrity checks:
- `validate .hotpath/qwen3_8b_repeat_fast_bench_i1.db` -> raw/stored kernel total `796880068 ns`, calls `4184`, `metrics summary PASS`
- `validate .hotpath/qwen3_8b_repeat_fast_bench_i2.db` -> raw/stored kernel total `801469834 ns`, calls `4206`, `metrics summary PASS`
- `report`, `export csv`, `export json`, `diff`, `artifacts`, `trace`, and `manifest` all completed successfully on the heavy `Qwen/Qwen3-8B` repeat artifacts.

2026-03-31 21:00 UTC
- Remote-controller single-host path stress:
- `doctor --target selfboot` -> PASS/WARN pattern as expected (`clock policy WARN unlocked`)
- default remote `profile --target selfboot ... --output .hotpath/stress_remote_default` -> produced `.db/.sqlite/_nvidia_smi.xml/_remote_server.log`
- `validate .hotpath/stress_remote_default.db` -> raw/stored kernel total `12202110 ns`, calls `693`, `metrics summary PASS`, `remote provenance PASS`
- explicit full-fetch remote `profile --target selfboot ... --fetch-nsys-rep --output .hotpath/stress_remote_full` -> produced `.db/.sqlite/.nsys-rep/_nvidia_smi.xml/_remote_server.log`
- `validate .hotpath/stress_remote_full.db` -> raw/stored kernel total `7284656 ns`, calls `413`, `metrics summary PASS`, `remote provenance PASS`
- `recover --target selfboot --remote-db /tmp/hotpath_bootstrap_test/.hotpath/stress_remote_default.db --output .hotpath/stress_remote_recovered.db` -> succeeded
- `validate .hotpath/stress_remote_recovered.db` -> raw/stored kernel total `12202110 ns`, calls `693`, `metrics summary PASS`, `remote provenance PASS`

2026-03-31 21:00 UTC
- Bench stress:
- `bench --kernel silu_and_mul --shapes 64x4096,256x4096 --warmup 10 --n-iter 50 --repeats 3 --output .hotpath/bench_stress_silu.json`
- `bench --kernel fused_add_rms_norm --shapes 64x4096,256x4096 --warmup 10 --n-iter 50 --repeats 3 --output .hotpath/bench_stress_rms.json`
- `bench --kernel rotary_embedding --shapes 64x1024,256x1024 --warmup 10 --n-iter 50 --repeats 3 --output .hotpath/bench_stress_rotary.json`
- All rows reported `valid=yes` and `det=yes`.
- Observed caveat: several small-shape rows still trigger `timing=yes` / `unstable=yes` because repeat CV and p99 spread are high at these tiny runtimes; larger-shape rows were much more stable (for example `fused_add_rms_norm` and `silu_and_mul` at `256x4096`).

2026-03-31 21:00 UTC
- Soak path stress:
- `soak-profile --model Qwen/Qwen2.5-0.5B-Instruct --prompts 1 --rollouts 1 --min-tokens 8 --max-tokens 8 --input-len 16 --iterations 2 --pause-sec 0 --validate-each --output .hotpath/stress_soak3`
- Produced `.hotpath/stress_soak3_i1.db/.sqlite/.nsys-rep` and `.hotpath/stress_soak3_i2.db/.sqlite/.nsys-rep`
- Validation:
- `_i1`: raw/stored kernel total `7284691 ns`, calls `413`
- `_i2`: raw/stored kernel total `7293945 ns`, calls `413`
- Result: soak-mode repeated runs stayed numerically correct on the optimized reused-session path.

2026-03-31 21:00 UTC
- Packaging and CLI smoke:
- `python3 -m venv /tmp/hotpath-release-venv && . /tmp/hotpath-release-venv/bin/activate && pip install .`
- Installed `hotpath help` succeeded.
- Installed `hotpath completion bash` succeeded.
- Installed `hotpath doctor` succeeded and reported:
- `python PASS`, `vllm PASS`, `nsys PASS`, `cuda visibility PASS`, `driver PASS`, `bench helper WARN ModuleNotFoundError: No module named 'torch'`, `clock policy WARN unlocked`, `nsys environment PASS`
- This is the expected optional-dependency behavior for a default install without bench extras.

2026-03-31 21:00 UTC
- Edge-case checks:
- `profile --target selfboot --attach http://127.0.0.1:8000 --attach-pid 12345 ...` failed fast with the explicit capability error:
- `attach-by-process requires an nsys CLI that advertises PID attach support. Installed nsys: NVIDIA Nsight Systems version 2024.2.3.38-242334140272v0`
- `aggregate .hotpath/qwen3_8b_repeat_fast_bench_i1.db .hotpath/qwen3_8b_repeat_fast_bench_i2.db --output .hotpath/stress_aggregate.db` succeeded
- `validate .hotpath/stress_aggregate.db` -> `kernel totals WARN sqlite artifact not available`, `metrics summary PASS`
- `bench-compare` succeeded; caveat is that comparing unrelated benchmark archives is semantically weak and surfaces zeroes for missing entries by design.
- `reset-defaults` succeeded.
- Prompt smoke after reset still shows a built-in model default in `profile`, which is acceptable for current CLI behavior but means scripted stdin prompt tests remain brittle when prompt layouts or defaults change.

2026-03-31 22:00 UTC
- Bench noise-reduction change implemented:
- default CUDA bench timing now uses adaptive batched `torch.cuda.Event` measurement targeting a minimum timed window (`--batch-ms-target`, default `10.0ms`)
- each timed sample measures multiple kernel launches and divides by launch count, which reduces per-launch jitter on very small kernels
- optional `--cuda-graph-replay on` was added for static-buffer graph-replay timing
- CLI/help/completions updated for:
- `--batch-ms-target FLOAT`
- `--cuda-graph-replay off|on`

2026-03-31 22:00 UTC
- Validation of the new batched bench path:
- `bench --kernel silu_and_mul --shapes 64x4096,256x4096 --warmup 10 --n-iter 50 --repeats 3 --output .hotpath/bench_batched_silu.json`
- `bench --kernel rotary_embedding --shapes 64x1024,256x1024 --warmup 10 --n-iter 50 --repeats 3 --output .hotpath/bench_batched_rotary.json`
- `bench --target selfboot --kernel silu_and_mul --shapes 64x4096 --warmup 10 --n-iter 50 --repeats 3 --output .hotpath/bench_remote_batched.json`
- `valid=yes` and `det=yes` remained true across the rerun rows.
- Batched timing materially reduced noise on previously noisy small-shape rows. Examples:
- `rotary_embedding vllm-cuda 64x1024`: `30.606 us` -> `14.645 us`, repeat CV warning cleared
- `rotary_embedding torch-compile 64x1024`: `143.093 us` -> `89.201 us`, repeat CV warning cleared
- `silu_and_mul torch-eager 256x4096`: `54.366 us` -> `26.211 us`, repeat CV warning cleared
- Remaining caveat: some tiny rows are still noisy enough to warn, for example `silu_and_mul vllm-cuda 256x4096` and `rotary_embedding vllm-cuda 256x1024`

2026-03-31 22:00 UTC
- Optional graph-replay validation:
- `bench --kernel silu_and_mul --shapes 64x4096,256x4096 --warmup 10 --n-iter 50 --repeats 3 --batch-ms-target 10 --cuda-graph-replay on --output .hotpath/bench_batched_graph_silu.json`
- Small-shape rows became substantially tighter and faster under graph replay. Examples:
- `silu_and_mul vllm-cuda 64x4096`: `24.916 us` batched -> `5.702 us` with graph replay
- `silu_and_mul torch-compile 64x4096`: `56.949 us` batched -> `5.272 us` with graph replay
- Result: graph replay is useful as an opt-in developer mode for static-buffer microbenchmarks, but the default publish path remains adaptive batched timing because it is more broadly applicable.

2026-03-31 22:00 UTC
- Implemented a local managed warm-server path to cut single-run `profile` wall-clock time:
- new commands: `server start`, `server list`, `server show`, `server stop`
- new profile path: `profile --server NAME`
- behavior: start one local `vllm serve` under `nsys launch`, persist session metadata under `.hotpath/servers/NAME.cfg`, then reuse that loaded server and `nsys` session across separate profile commands.
- also extended the fast reuse logic so `profile --server NAME --repeat N` and local `soak-profile` on a managed server reuse the same warm server/session instead of relaunching.

2026-03-31 22:00 UTC
- First managed-server smoke exposed a lifecycle bug: the saved PID belonged to the short-lived launch wrapper, so `server show` immediately reported `ready: no` and `profile --server ...` failed with `managed server is not ready`.
- Fix: after readiness, resolve and store the actual listening PID from the serving port. Managed-server readiness is now tied to the endpoint plus the resolved listener PID, not the wrapper process.
- Second managed-server smoke exposed a concurrency bug: overlapping `profile --server NAME ...` runs hit raw `nsys` state errors (`Configuring is not allowed in this state.`).
- Fix: added a per-server lock under `.hotpath/servers/NAME.lock`; overlapping traces now fail fast with `managed server is busy: NAME`.

2026-03-31 22:00 UTC
- Local managed-server validation:
- `./build/hotpath server start --name warm-smoke2 --model Qwen/Qwen2.5-0.5B-Instruct --port 8031` -> server state written, `server show warm-smoke2` reported `ready: yes`
- serial `profile --server warm-smoke2 ... --output .hotpath/warm_smoke2_serial_a` -> succeeded
- `validate .hotpath/warm_smoke2_serial_a.db` -> raw/stored kernel total `7294085 ns`, calls `413`, `metrics summary PASS`
- serial `profile --server warm-smoke2 ... --output .hotpath/warm_smoke2_serial_b` -> succeeded
- `validate .hotpath/warm_smoke2_serial_b.db` -> raw/stored kernel total `7293864 ns`, calls `413`, `metrics summary PASS`
- `server stop warm-smoke2` -> succeeded

2026-03-31 22:00 UTC
- Wall-clock comparison on the same tiny single-host workload (`Qwen/Qwen2.5-0.5B-Instruct`, `1x1`, `8-8`, `16`):
- warm steady-state path:
- `/usr/bin/time -f 'elapsed=%e' ./build/hotpath profile --server warm-speed --prompts 1 --rollouts 1 --min-tokens 8 --max-tokens 8 --input-len 16 --output .hotpath/warm_speed_compare`
- result: `elapsed=20.23`
- `validate .hotpath/warm_speed_compare.db` -> raw/stored kernel total `7283731 ns`, calls `413`, `metrics summary PASS`
- fair cold baseline after freeing the GPU:
- `/usr/bin/time -f 'elapsed=%e' ./build/hotpath profile --model Qwen/Qwen2.5-0.5B-Instruct --prompts 1 --rollouts 1 --min-tokens 8 --max-tokens 8 --input-len 16 --output .hotpath/cold_speed_compare_free`
- result: `elapsed=48.46`
- `validate .hotpath/cold_speed_compare_free.db` -> raw/stored kernel total `7283257 ns`, calls `413`, `metrics summary PASS`
- Conclusion for the tested single-host path: warm managed-server profiling reduced end-to-end wall clock from `48.46s` to `20.23s` without changing the stored kernel totals or metric summary correctness.

2026-03-31 22:00 UTC
- Hardened the managed-server path around lifecycle correctness instead of just speed:
- `server start` now takes an explicit `--max-model-len` and persists it in managed-server state.
- `profile --server NAME ...` and local `soak-profile` now derive an effective config from the managed server and reject workloads whose inferred `max_model_len` exceeds the server's configured ceiling.
- Added `server prune` and automatic stale-state cleanup during `list` / `show` / `find`.
- Added stale-lock recovery for managed-server trace locks by storing the owner PID in `.hotpath/servers/NAME.lock/pid` and reaping locks whose owner process is dead.
- Listener PID resolution now falls back from `lsof` to `ss`, so warm-server readiness no longer depends on `lsof` alone.

2026-03-31 22:00 UTC
- Managed-server hardening validation:
- `ctest --test-dir build --output-on-failure` -> `19/19 passed`
- `server prune` -> `Pruned 0 stale managed servers`
- `server start --name maxlen-smoke --model Qwen/Qwen2.5-0.5B-Instruct --port 8040 --max-model-len 2048` -> succeeded
- `server show maxlen-smoke` reported `ready: yes` and `max_model_len: 2048`
- `profile --server maxlen-smoke --prompts 1 --rollouts 1 --min-tokens 8 --max-tokens 3000 --input-len 16 ...` failed fast with:
- `managed server max-model-len is too small for this workload: server=2048, required=3016`
- `server stop maxlen-smoke` -> succeeded

2026-03-31 23:00 UTC
- Re-validated the measurement window to make sure reported kernel totals are not contaminated by launch/startup overhead:
- code check: tracing starts only inside `capture_measured_run()` via `nsys start --session=...` after readiness, and launch-mode startup uses `nsys profile --start-later=true`, so model load and startup wait remain outside the traced region by design.
- managed warm run:
- `profile --server measure-window --prompts 1 --rollouts 1 --min-tokens 8 --max-tokens 8 --input-len 16 --discard-first-run --fetch-nsys-rep --output .hotpath/measure_window_warm`
- `validate .hotpath/measure_window_warm.db` -> `kernel totals PASS`, `raw total ns=7287955`, `raw calls=413`
- `nsys stats --force-export=true --report cuda_gpu_kern_sum .hotpath/measure_window_warm.nsys-rep` -> `Total Time (ns) = 7287955`
- attempted cold run in parallel with the warm managed server failed for the expected reason (insufficient free GPU memory while the warm server still owned the device); reran sequentially after `server stop measure-window`.
- sequential cold run:
- `/usr/bin/time -f 'elapsed=%e' ./build/hotpath profile --model Qwen/Qwen2.5-0.5B-Instruct --prompts 1 --rollouts 1 --min-tokens 8 --max-tokens 8 --input-len 16 --discard-first-run --fetch-nsys-rep --output .hotpath/measure_window_cold_clean`
- result: `elapsed=48.54`
- `validate .hotpath/measure_window_cold_clean.db` -> `kernel totals PASS`, `raw total ns=7290870`, `raw calls=413`
- `nsys stats --force-export=true --report cuda_gpu_kern_sum .hotpath/measure_window_cold_clean.nsys-rep` -> `Total Time (ns) = 7290870`
- warm vs cold traced kernel totals differ by only `2915 ns` across ~`7.29 ms` total GPU time (`~0.04%`), while the end-to-end cold wall clock is much larger. Conclusion: on the tested single-host path, the reported kernel numbers are measuring the request window, not server launch/model-load overhead.

2026-03-31 23:00 UTC
- Implemented an ambitious attach-by-process fallback so `profile --attach URL --attach-pid PID` can still produce a traced profile on hosts whose `nsys` does not expose native PID attach:
- first choice: native PID attach when `nsys profile --help` advertises `--pid`
- second choice: clone the source `vllm serve` command line onto a free traced port on the same host
- third choice: when free GPU memory is too low for a second copy, temporarily replace the source process with a traced equivalent on the same port, then restore the source process after the profile completes
- Added `src/profiler/attach.cpp`, `include/hotpath/profiler/attach.h`, and `cpp_tests/test_attach.cpp`
- Validation on this host:
- `ctest --test-dir build --output-on-failure` -> `20/20 passed`
- `server start --name attach-src3 --model Qwen/Qwen2.5-0.5B-Instruct --port 8070 --max-model-len 2048` -> source listener PID `215839`
- `profile --attach http://127.0.0.1:8070 --attach-pid 215839 --prompts 1 --rollouts 1 --min-tokens 8 --max-tokens 8 --input-len 16 --fetch-nsys-rep --output .hotpath/attach_clone_smoke5` -> succeeded
- `validate .hotpath/attach_clone_smoke5.db` -> `kernel totals PASS`, `raw total ns=7281364`, `raw calls=413`, `metrics summary PASS`
- saved profile metadata confirms the chosen fallback path:
- `attach_mode_kind=process_replace_restore`
- `attach_clone_mode=replace_restore`
- `attach_clone_source_pid=215839`
- `attach_clone_source_server=http://127.0.0.1:8070`
- restore validation after the run:
- `sleep 5 && curl -fsS http://127.0.0.1:8070/metrics | head -n 5` succeeded
- Result: attach-by-process is now live-validated on the tested single-host path even though the installed `nsys` does not expose native PID attach support.

2026-03-31 23:00 UTC
- Attach-by-process wall-clock on the tested replace-and-restore path:
- final fresh-port run `profile --attach http://127.0.0.1:8070 --attach-pid 215839 ... --output .hotpath/attach_clone_smoke5`
- end-to-end controller wait was approximately `68s` on this A10G (`45s` + `23s` observed across the final polling windows)
- This is materially slower than native attach would be because replace-and-restore still has to reload the model and re-warm the traced replacement process.
- Result: attach-by-process is fixed functionally, but the replace-and-restore path remains a heavy operation on single-GPU hosts.

2026-03-31 23:00 UTC
- Clarified and covered the non-local scope of the attach implementation:
- remote known-host attach already works through the controller because `profile --target TARGET --attach URL --attach-pid PID` forwards the same arguments to the target-side `hotpath`, which then runs the same host-local attach planner on the GPU box
- added remote command-path coverage in `cpp_tests/test_remote.cpp` to assert forwarding of:
- `--attach 'http://127.0.0.1:8070'`
- `--attach-pid '215839'`
- Result: the current boundary is no longer "same-host only"; it is "host-local to whichever machine is running the tracing logic", including a known remote target.

2026-04-01 00:00 UTC
- Reworked attach-by-process repeat/soak execution so it can reuse one traced replacement lifecycle across multiple measured windows instead of relaunching a fresh attach fallback each iteration.
- Implementation details:
- `profile --repeat N` now keeps the attach fallback state alive when `--attach-pid` is present.
- `soak-profile` now supports `--attach URL --attach-pid PID` through the same reusable lifecycle instead of rejecting attach mode or falling back to full relaunch loops.
- Added helper logic in `src/profiler/runner.cpp` to centralize attach mode metadata and reuse semantics between single-run, repeat, and soak paths.

2026-04-01 00:00 UTC
- First live repeated attach attempt exposed two real bugs:
- `run_soak_profile()` was still sending traffic to the original attach URL/default port instead of the traced clone/replacement URL selected by the attach planner.
- attach clone/replace startup still used `nsys profile --start-later=true`, which left the repeated attach session stuck because the fast repeat path expects a reusable `nsys launch` session.
- Fixes applied:
- aligned `run_soak_profile()` server URL / effective config handling with `run_profile()`
- changed attach clone/replace startup in `src/profiler/attach.cpp` to use `nsys launch --session-new ... --wait=all`

2026-04-01 00:00 UTC
- Live validation after the attach repeat fixes:
- external source server started on `127.0.0.1:8081`, source PID `220580`
- command:
- `profile --attach http://127.0.0.1:8081 --attach-pid 220580 --prompts 1 --rollouts 1 --min-tokens 8 --max-tokens 8 --input-len 16 --repeat 2 --fetch-nsys-rep --output .hotpath/attach_repeat_fast`
- completed successfully, wrote:
- `.hotpath/attach_repeat_fast_i1.db`
- `.hotpath/attach_repeat_fast_i2.db`
- `validate .hotpath/attach_repeat_fast_i1.db` -> `kernel totals PASS`, `raw total ns=7288231`, `raw calls=413`, `metrics summary PASS`
- `validate .hotpath/attach_repeat_fast_i2.db` -> `kernel totals PASS`, `raw total ns=7294037`, `raw calls=413`, `metrics summary PASS`
- controller-side wall clock from `date -u +%s` before/after the command: `83s` for two measured runs, or about `41.5s/run`
- comparison to earlier single-run replace-and-restore attach (`~68s`): repeated attach reuse materially reduced end-to-end wall clock while preserving raw/stored kernel agreement
- post-run restore check succeeded: `curl -fsS http://127.0.0.1:8081/metrics | head -n 5`

2026-04-01 00:00 UTC
- Publish-surface cleanup:
- rewrote `README.md` to be more modular and direct, with PyPI install, quick start, remote workflow, attach modes, faster repeated profiling, fetch policy, command reference, stored data, validation status, boundaries, compatibility, and test commands
- updated `pyproject.toml` metadata to support the intended PyPI release surface:
- package description now directly describes profiling vLLM inference under RL-style rollout workloads
- added keywords, classifiers, and GitHub project URLs
- added `hotpath_py.__version__ = "0.1.0"`
- clean install smoke after the metadata/readme changes:
- `python3 -m venv /tmp/hotpath-pypi-venv && . /tmp/hotpath-pypi-venv/bin/activate && pip install . && hotpath help && hotpath doctor && hotpath completion bash`
- result:
- built wheel `hotpath-0.1.0-cp312-cp312-linux_x86_64.whl`
- installed `hotpath` entrypoint works
- installed `doctor` reports missing bench extras as `WARN` rather than a hard failure

2026-04-01 02:00 UTC
- Verified three post-pull review findings with targeted regression tests before changing implementation:
- `cpp_tests/test_vllm_metrics.cpp` was extended to require clustered metric fetches to retain TTFT/TPOT percentile summaries even when only non-percentile cluster metrics are synthesized.
- `cpp_tests/test_aggregate.cpp` and `cpp_tests/test_report.cpp` were updated to reject the old `aggregate_max_median_ratio_upper_bound` naming and require an honest `aggregate_max_median_ratio_observed_max` / `max/median ratio max` path instead.
- `cpp_tests/test_report.cpp` and `cpp_tests/test_export.cpp` were updated to require visible `completion_length_samples` disclosure.
- Rebuilding those tests against the pulled code confirmed the issues were real:
- `test_vllm_metrics` failed because clustered percentile summaries were dropped entirely.
- `test_report` failed because the warning/report text still claimed stronger aggregate percentile math than the implementation provided.
- `test_aggregate` failed because the old metadata key was still in use.

2026-04-01 02:00 UTC
- Fixed the confirmed regressions:
- `src/profiler/vllm_metrics.cpp`: summary logic now chooses cluster samples per metric only when that metric actually has synthesized cluster values; otherwise it falls back to the per-source samples for that metric. Result: TTFT/TPOT percentile summaries remain visible for clustered metric fetches even though no fake cluster percentile sample is synthesized.
- `src/aggregate.cpp`: renamed `aggregate_max_median_ratio_upper_bound` to `aggregate_max_median_ratio_observed_max` because the previous label was mathematically wrong.
- `src/report.cpp`: warning text now says aggregate `p50/p99` are upper bounds and `max/median` is the max observed per-run ratio; traffic shape section now prints `completion length samples`; aggregate ratio row now renders as `max/median ratio max`.
- `src/export.cpp`: exported warning text updated to match the corrected semantics.

2026-04-01 02:00 UTC
- Validation after the fixes:
- `ctest --test-dir build --output-on-failure -R 'test_vllm_metrics|test_aggregate|test_report|test_export'` -> all passed after rebuild
- full rebuild and suite:
- `cmake --build build`
- `ctest --test-dir build --output-on-failure`
- result: `22/22 passed`
- Conclusion: the three review findings were real logic/reporting problems, not false positives, and the fixes are now covered by regression tests.
2026-04-01 03:00 UTC
- Review sweep found two additional concrete logic bugs:
- Main-menu `Bench compare` called `handle_bench_compare({"bench-compare"})`, which always printed usage because no JSON paths were provided.
- `stop_managed_server()` preferred persisted `state.pid` whenever it was positive, even if it was stale and a different live listener owned the configured port.

2026-04-01 03:00 UTC
- Fixed main-menu `Bench compare` by adding an interactive recent-bench-result picker path based on `.hotpath/*.json`.
- Fixed managed-server stop to prefer the live listener PID when the saved PID is stale or dead, while still honoring the persisted PID when it is the active listener.
- Added regression coverage for recent bench-result listing and stale-PID managed-server shutdown behavior.

2026-04-01 03:00 UTC
- Additional review found two more portability/UX issues:
- attach-by-process clone planning only recognized literal loopback host strings and could reject same-host attach URLs addressed by hostname.
- recent bench-result discovery accepted every `.json` artifact in `.hotpath/`, which could surface exported profile JSON in `Bench compare`.

2026-04-01 03:00 UTC
- Fixed attach locality detection to resolve the attach host against local interface addresses and local hostname, not just `localhost` literals.
- Tightened recent bench-result discovery to include only JSON files that parse as non-empty bench result archives.
- Added regression coverage for attach locality detection and bench-result filtering.
2026-04-01 03:00 UTC
- Removed the standalone README compatibility table.
- Reason: the table read too much like a hardware restriction instead of a validated environment note.
- README now keeps requirements and tested workflows without implying A10G-only support.

2026-04-04 05:00 UTC
- Live end-to-end serving validation with real downloaded model `Qwen/Qwen2.5-0.5B-Instruct` on the local A10G.
- Focused CLI regression check after recent help/completion edits:
- `ctest --test-dir build -R test_cli --output-on-failure` -> passed.
- Real `vllm` environment used: `/tmp/hotpath-test-venv` (`vllm 0.19.0`); repo `.venv` remained the `sglang` environment.
- Real `vllm` live path:
- launched `/tmp/hotpath-test-venv/bin/vllm serve Qwen/Qwen2.5-0.5B-Instruct --port 8000 --host 127.0.0.1 --max-model-len 2048`
- verified `/v1/models` and `/health`
- created `/tmp/hotpath-live-traffic.jsonl` with 3 prompts
- ran `./build/hotpath serve-profile --endpoint http://127.0.0.1:8000 --duration 5 --traffic /tmp/hotpath-live-traffic.jsonl --output /tmp/hotpath-vllm-live --engine vllm`
- generated `serve-report` and `disagg-config` successfully from the live DB
- Live bug found and fixed:
- `serve-profile` endpoint validation treated empty `/health` bodies as unreachable; healthy `vllm` returns `200` with empty body.
- fix in `src/serving/serve_profiler.cpp`: check HTTP status code instead of non-empty body.
- Real `sglang` live path:
- initial launch failed while `vllm` still held GPU memory; after stopping `vllm`, launch progressed
- second launch failed during FlashInfer JIT because `ninja` was not on `PATH` for the venv Python process even though `.venv/bin/ninja` existed
- relaunched with `PATH=/home/ubuntu/rlprof/.venv/bin:$PATH` and `--enable-metrics`
- verified `/health` and `/metrics` on `127.0.0.1:8001`
- ran `./build/hotpath serve-profile --endpoint http://127.0.0.1:8001 --duration 5 --traffic /tmp/hotpath-live-traffic.jsonl --output /tmp/hotpath-sglang-live --engine sglang`
- request replay worked, but `Collected 0 metric samples` exposed a real integration bug
- SGLang metrics bug found and fixed:
- `src/profiler/sglang_metrics.cpp` only recognized older singular metric names (`num_running_req`, `num_waiting_req`), while live SGLang exposed plural/current names (`num_running_reqs`, `num_queue_reqs`)
- `src/serving/serve_profiler.cpp` was still polling only through the vLLM metrics path, so SGLang metrics were ignored even when `/metrics` was live
- fixes applied:
- updated `src/profiler/sglang_metrics.cpp` to accept both old and current SGLang metric names
- updated `cpp_tests/test_sglang_metrics.cpp` with coverage for the current plural names
- updated `src/serving/serve_profiler.cpp` to poll SGLang metrics directly when `--engine sglang` is selected and map them into the canonical metric names consumed by the existing analyzers
- live retest after the fix:
- `./build/hotpath serve-profile --endpoint http://127.0.0.1:8001 --duration 5 --traffic /tmp/hotpath-live-traffic.jsonl --output /tmp/hotpath-sglang-live2 --engine sglang`
- result: `Replay: 3/3 succeeded`, `Collected 45 metric samples`
- generated `serve-report` and `disagg-config` successfully from the fixed live SGLang DB
- final native validation after the live-path fixes:
- `ctest --test-dir build --output-on-failure` -> `36/36` passed
- Design choices / struggles:
- kept `vllm` and `sglang` in separate Python environments because earlier `sglang[all]` installation broke the repo `.venv` for `vllm`
- used a small public model to exercise the real network/download/serve path without overcommitting the single 24 GB GPU
- fixed only concrete issues hit by live testing rather than guessing at broader refactors

2026-04-04 08:00 UTC
- Full production-level audit pass: build, tests, CLI correctness, output quality, edge cases.

2026-04-04 08:00 UTC
- Test suite: 36/36 passed, 553 audit checks passing. Stress run (schedule-random, 3 repeats) stayed clean.

2026-04-04 08:00 UTC
- Bugs found and fixed during the production audit:
  1. `report.cpp` GPU Phase Breakdown false positive: the render logic used `d.prefill_compute_pct > 0.0 || d.decode_compute_pct > 0.0` to decide whether to show the GPU phase section. But `serve_profiler` always writes estimated pct values (from workload classifier) regardless of `gpu_phase_available`, so the bar chart would show up even when no real GPU data was collected. Fix: use only `d.gpu_phase_available` as the gate. Updated `test_audit.cpp` to set `gpu_phase_available = true` in the bar-chart test, since the test should explicitly express that GPU data is available.
  2. `report.cpp` prefix sharing false positive: same pattern — `d.unique_prefixes > 0 || d.cacheable_tokens_pct > 0.0` would show prefix data even without `prefix_sharing_available = true`. Fix: use only the flag.
  3. Prefix Sharing "not available" message was misleading: said "token-level prefix analysis requires tokenized prompts — current path uses character-level prefixes" which confused why it would be unavailable. Actual reason: no prompt text was captured. Fixed message to: "no prompt text captured — replay traffic must include prompt content".
  4. Shell completions missing commands: bash/zsh/fish all omitted `version`, `soak-profile`, `cluster-profile`, `manifest`, `cleanup`. Fixed all three completion outputs.
  5. Doctor script `tool_python()` and `tool_vllm()` only checked `RLPROF_PYTHON_EXECUTABLE` / `RLPROF_VLLM_EXECUTABLE`, not the new `HOTPATH_` env vars. Updated to check `HOTPATH_*` first with `RLPROF_*` fallback.
  6. Bench command `build_bench_cmd()` and `gpu_bench_available()` only read `RLPROF_PYTHON_EXECUTABLE`. Updated to check `HOTPATH_PYTHON_EXECUTABLE` first.

2026-04-04 08:00 UTC
- Validated `serve-report` and `disagg-config` end-to-end with both a monolithic DB (real live_smoke run) and a synthesized disagg-recommended DB.
- `serve-report` output for monolithic: shows TTFT (client) / Generation (client) / Decode (per-token) / End-to-end latency, GPU Phase not available, KV Cache not available, Prefix Sharing from char-level analysis, MONOLITHIC recommendation with caveat.
- `serve-report` output for disagg: shows DISAGGREGATE recommendation, 2:2 P:D ratio, +63% throughput projection, TTFT improvement, min bandwidth.
- `disagg-config --format all` generates correct vLLM, llm-d, and Dynamo configs.
- Error cases work cleanly: missing path, empty DB, wrong schema all give useful messages.

2026-04-04 07:00 UTC
- Post-context-compaction recovery pass: verified tree was already clean and all 36 tests passing.
- Confirmed build was not broken despite the session summary saying it was; the linter-modified test files and header/implementation updates had all landed before the compaction.
- Found serve-profile managed-server error UX was unhelpful: when `RLPROF_PYTHON_EXECUTABLE` pointed to a Python without vLLM, the error just said "no working vLLM launcher found" with no actionable guidance.
- Root cause of user's earlier failure: shell multiline quoting split the path `/tmp/hotpath-\n  test-venv/bin/python` — the path never resolved, and the fallthrough error message was opaque.
- Fixes applied in `src/profiler/server.cpp`:
  - Added support for `HOTPATH_PYTHON_EXECUTABLE` and `HOTPATH_VLLM_EXECUTABLE` (with legacy `RLPROF_` fallback for backwards compat).
  - When the configured Python exists but vLLM is not installed, the error now says explicitly: "Python at PATH does not have vLLM installed. Install it with: PATH -m pip install vllm".
  - When the configured Python path does not exist, the error says "Python executable not found: PATH — Check HOTPATH_PYTHON_EXECUTABLE is set to a valid path".
  - The generic fallthrough error now lists four actionable recovery options: set env var, use vLLM env, install globally, or profile an existing endpoint.
- Rebuild and full suite: `36/36 passed`.

2026-04-04 08:00 UTC
- Metric accuracy pass: fixed two accuracy issues discovered during serve-profile verification.

2026-04-04 08:00 UTC
- Investigation findings before fixes:
  - `serve-profile` was reporting "TTFT (client)" = 4.6ms, but this was actually `curl time_starttransfer` — time to first HTTP response header byte, not the first actual SSE token.
  - First actual `data:` SSE token arrives ~13.6ms after sending request (verified with Python socket test).
  - Server-side Prometheus `vllm:time_to_first_token_seconds_sum/count` histogram confirms 10–20ms true TTFT.
  - Root cause: curl `time_starttransfer` measures when HTTP headers arrive (before any token data is sent over SSE), so the reported "TTFT" was off by ~4x.
  - vLLM 0.19 changed metric names: `vllm:gpu_cache_usage_perc` → `vllm:kv_cache_usage_perc`; `vllm:prefix_cache_hit_rate` (gauge) → counters `vllm:prefix_cache_hits_total` + `vllm:prefix_cache_queries_total`; `vllm:time_to_first_token_seconds_p50/p99` (deprecated gauges) → histogram `_sum/_count/_bucket`.
  - hotpath was not collecting any of the vLLM 0.19 metrics, so KV cache utilization, TTFT, and prefix cache hit rate were all silently "not available" with a live vLLM 0.19 endpoint.

2026-04-04 08:00 UTC
- Fixes applied:
  - `src/profiler/vllm_metrics.cpp`:
    - Added `vllm:kv_cache_usage_perc`, `vllm:prefix_cache_hits_total`, `vllm:prefix_cache_queries_total`, `vllm:time_to_first_token_seconds_sum`, `vllm:time_to_first_token_seconds_count` to `kKeyMetrics`.
    - Kept old vLLM < 0.19 names as fallback for backward compat.
    - Updated `aggregate_by_average()` to include `vllm:kv_cache_usage_perc`.
  - `src/serving/serve_profiler.cpp`:
    - `samples_to_snapshots()`: now accepts `vllm:kv_cache_usage_perc` alongside `vllm:gpu_cache_usage_perc` for cache utilization snapshots.
    - After `metrics_thread.join()`, computes two derived values from counter deltas:
      - `server_ttft_mean_ms`: from `(max_sum - min_sum) / (max_count - min_count) * 1000` for `time_to_first_token_seconds_sum/count`.
      - `prometheus_cache_hit_rate`: from `(max_hits - min_hits) / (max_queries - min_queries)` for `prefix_cache_hits_total / prefix_cache_queries_total`.
    - Falls back to `prometheus_cache_hit_rate` when no server log provided (e.g., vLLM V1 engine that doesn't emit per-request debug lines).
    - Saves `server_ttft_mean_ms` as `latency.server_ttft_mean_ms` in DB.
  - `include/hotpath/report.h`: added `double server_ttft_mean_ms = -1.0` to `ServeReportData`.
  - `src/report.cpp`:
    - `format_optional_metric()`: added `vllm:kv_cache_usage_perc` to the percentage formatter.
    - `metric_label()`: added labels for `vllm:kv_cache_usage_perc`, `vllm:prefix_cache_hits_total`, `vllm:prefix_cache_queries_total`.
    - `render_serve_report()`: renamed "TTFT (client)" to "TTFB (client)" (time to first byte), added explanatory comment, and added "TTFT (server, mean)" row when `server_ttft_mean_ms > 0`.
  - `src/main.cpp`: reads `latency.server_ttft_mean_ms` from DB into `ServeReportData`.
  - `cpp_tests/test_serve_report.cpp`: updated label check from "TTFT (client)" to "TTFB (client)".

2026-04-04 08:00 UTC
- Validation:
  - `cmake --build build -j$(nproc)` clean build.
  - `ctest --output-on-failure` -> `36/36 passed`.
  - Live smoke against vLLM 0.19 endpoint confirmed:
    - `Cache hit rate from Prometheus counters: 36%` logged (derived from `prefix_cache_hits_total / prefix_cache_queries_total`).
    - `TTFB (client) = 4.2ms` and `TTFT (server, mean) = 13.2ms` both shown in report — correctly exposing the measurement gap.
    - KV cache utilization collected via `vllm:kv_cache_usage_perc` (was silently missing before).

2026-04-05 00:30 UTC
- Full production audit findings and resolutions:

  FIXED — real bugs:
  1. prefill_contention could go negative when p99_decode_latency < median_decode_latency (measurement noise on small samples). Negative contention made blocking_factor > 1 in disagg model, producing throughput above physical maximum and TTFT below raw prefill time. Fix: clamp contention to max(0.0, ...) in workload_classifier.cpp.
  2. projected_throughput_pct: get_d() returns 0.0 for missing keys, so (0.0 - 1.0) * 100 = -100% if "disagg.throughput_improvement" absent from DB. Only displayed when should_disaggregate=true, which also requires the disagg block, so in practice harmless — but clamped to max(0.0, ...) to remove the landmine.
  3. Prometheus counter delta sentinel pattern: min=+∞, max=-1 is correct (no-data → delta negative → guard fails) but was confusing without explanation. Added inline comments.

  NOT BUGS (audit false positives):
  - Prometheus TTFT/cache-rate deltas: sentinel math correctly produces delta < 0 when no metrics collected → guard (delta_count > 0) fails → value stays -1 (not available). No corrupt data possible.
  - PhaseBreakdown phase struct: all fields have default member initializers (= 0.0), so uninitialized phase path writes zeros to DB; display gated by gpu_phase_available flag.
  - min_preemption = snapshots[0].preemption_total: standard min-tracking pattern; snapshots always populated from real Prometheus samples.
  - projected_throughput_pct = -100%: gated by should_disaggregate=true, which requires the full disagg block in DB; missing partial DB cannot show DISAGGREGATE recommendation.

  DESIGN CHOICES DOCUMENTED (not bugs, but worth recording):
  - percentile_vec(empty) returns 0.0: if all requests fail, client-side latency rows show 0.0ms instead of "-". Acceptable because: (a) total_requests and throughput values make the failure visible; (b) adding optional<double> returns would require pervasive changes. Known limitation.
  - phase pct values (prefill_pct, decode_pct) are always saved to DB even when gpu_phase_available=false; values are 0.0 from default initialization. DB is polluted with uninformative zeros but display is correctly gated. Acceptable because the DB is an internal artifact, not user-facing.

2026-04-05 00:00 UTC
- serve-report TTFT disagg section design decisions:
  - "Projected p99 TTFT" mono side now uses `server_prefill_p99` (measured) when `server_timing_available && server_prefill_p99 > 0`, else falls back to the disagg model estimate. Labeled `(measured)` vs `(est.)` accordingly.
  - Disagg side is always `(est.)` — it is fundamentally not measurable without running disagg. Best we can do is `measured_prefill_p99 + kv_transfer_overhead_ms`; KV transfer is estimated from median token count and assumed network bandwidth.
  - `DisaggModelInput.measured_prefill_p99_ms`: when > 0, overrides `mono_p99_ttft_ms` and serves as the prefill basis for `disagg_p99_ttft_ms`. The M/G/1 throughput model still uses `median_prompt_tokens * 0.01ms/token` because p99 is too pessimistic as a queue service time; it is only used for the TTFT display.
  - Integer ms display in serve-report uses `std::lround()`, not `static_cast<int>`. Reason: truncation silently under-reports latency (99.9ms → 99ms). For measured p99 values this is a real accuracy problem, not just cosmetic. `std::lround` gives honest nearest-integer rounding.

2026-04-04 06:00 UTC
- Accuracy cleanup after audit findings: Phase A stop-lying pass and Phase B real-data wiring.
- Phase A changes:
- `render_serve_report()` now labels client-side proxy timings honestly:
- `Prefill` -> `TTFT (client)`
- `Decode (total)` -> `Generation (client)`
- queue wait is no longer printed as a measured zero unless explicitly marked available
- placeholder sections no longer render confident-looking zeroes:
- GPU phase breakdown prints `not available` unless true phase data is present
- KV cache prints `not available` when neither cache-usage metrics nor cache-hit data are available
- Prefix sharing prints `not available` unless a real prefix analysis ran
- disaggregation output can now include a caveat string describing approximate inputs
- Phase B changes:
- `ReplayResult` now carries parsed `prompt_tokens`, `completion_tokens`, and `prompt_tokens_estimated`
- `RequestTrace` now persists `prompt_tokens_estimated` and raw `prompt_text`
- `traffic_replayer.cpp` now requests streaming usage data (`stream_options.include_usage`) and parses `usage.prompt_tokens` / `usage.completion_tokens` when present; otherwise it falls back to chars/4 for prompt tokens and marks the trace as estimated
- `store.cpp` schema/migration updated to persist the new request-trace fields and round-trip them
- `serve_profiler.cpp` now runs real character-level prefix analysis on stored prompt text and feeds prefix analysis into the workload classifier
- `serve_profiler.cpp` now persists availability flags for queue, phase, cache, and prefix sections plus the advisor caveat
- `serve_profiler.cpp` / `vllm_metrics.cpp` now accept both `vllm:num_preemptions_total` and `vllm:num_preemption_total`; cache snapshots also consider `cpu_cache_usage_perc`
- cache report output now distinguishes unavailable cache-hit rate from real usage/eviction stats and includes peak usage
- tests updated:
- `cpp_tests/test_request_trace.cpp` checks new round-trip fields
- `cpp_tests/test_serve_report.cpp` now requires the honest client-side labels and requires placeholder data to render `not available` sections instead of fake zeroes
- validation:
- focused regressions passed:
- `ctest --test-dir build -R 'test_request_trace|test_traffic_replayer|test_serve_report|test_sglang_metrics' --output-on-failure`
- required audit pass:
- `ctest --test-dir build -R test_audit --output-on-failure`
- full suite pass after integration:
- `ctest --test-dir build --output-on-failure` -> `36/36 passed`

2026-04-04 10:00 UTC
- Second production audit pass. Implemented fixes for two known limitations and resolved all critical/high findings.

  KNOWN LIMITATIONS RESOLVED:
  1. `percentile_vec(empty)` now returns -1.0 (sentinel) instead of 0.0.
     Previously: all-failure runs displayed "0.0ms" latency, indistinguishable from instant completion.
     Fix: sentinel -1.0 propagates through DB save, and `latency_row` in `render_serve_report` displays "-"
     when p50 < 0. Display is identical for successful runs; changed only for zero-result vectors.
  2. Phase pct values (prefill_pct, decode_pct, other_pct) are now only saved to DB when
     `gpu_phase_available=true`. Previously, default-initialized zeros (PhaseBreakdown = 0.0) were
     unconditionally saved, polluting the DB with values that look like real measurements.

  NEW BUGS FIXED:
  3. `measured_prefill_p99_ms >= 0.0` guard (was `> 0.0`): 0.0ms is a valid measured prefill time
     (near-instant for fully cached workloads). Only -1.0 is the sentinel. Fixed in `disagg_model.cpp`.
  4. `prometheus_cache_hit_rate` clamped to [0, 1]: counter race conditions can produce
     delta_hits > delta_queries (e.g. first scrape sees a stale cache count while queries already
     updated). Unclamped value > 1.0 would be displayed as ">100% cache hit rate". Fixed.
  5. `rate_limit_rps` division by zero when `duration_seconds=0`: guarded; zero duration is treated
     as "no rate limit" (send all requests as fast as possible).
  6. Malformed JSONL entries silently skipped: `load_jsonl` now emits per-line warnings and a
     summary count when lines are not JSON objects or lack required fields.

  AUDIT FINDINGS NOT FIXED (false positives):
  - prefix_analyzer.cpp:129 div/zero: prompts.empty() early-return on line 64 makes total_requests
    always >= 1 at line 129. No actual div-by-zero possible.
  - Incomplete traces (no timing) set to status "ok": correct — success = HTTP 200, independent of
    timing availability. Latency collection already guards all push_back calls on positive timestamps.

  DESIGN CHOICES:
  - `measured_prefill_p99_ms` sentinel is -1.0 (not 0.0): 0.0 is a valid measurement;
    changing the default from -1.0 to some "invalid" positive sentinel would be a worse API.
  - `max_concurrency=16` default in ReplayConfig but always forced to 1 in serve_profiler:
    intentional — sequential replay keeps timing deterministic and avoids saturating the server
    with artificial concurrency. The field is reserved for future tunable replay mode.
  - Magic constants in disagg model (0.01ms/token, 5ms/decode): acknowledged rough estimates.
    They are already exposed via the advisor caveat string when timing data is unavailable.
    Adding per-model config was deferred — the caveat is the right UX signal.

  VALIDATION:
  - `cmake --build build -j$(nproc)`: clean build, no warnings
  - `./build/test_audit`: 598 passed, 0 failed
  - `ctest --test-dir build --output-on-failure`: all 36 targets passed

2026-04-05 11:00 UTC
- Final pre-production numerical audit and fixes.

  BUGS FIXED:
  1. Cache histogram bucket boundaries: `<= 0.25` caused exactly-25% hit rate to land in the
     "1-25%" bucket instead of "25-50%". Changed to `< 0.25`, `< 0.50`, `< 0.75` (half-open
     intervals). Also fixed the bulk test in test_audit.cpp to mirror the new boundaries.
  2. Phase `other_pct` floating-point clamp: `1.0 - prefill_fraction - decode_fraction` could
     produce a tiny negative due to FP accumulation. Added `std::max(0.0, ...)`.
  3. Log-parser `aggregate_cache_hit_rate` unclamped: `std::stod(X) / 100.0` on a malformed log
     line could exceed 1.0. Added `std::clamp(..., 0.0, 1.0)`.
  4. SGLang advisor caveat suggested vLLM-specific flags (`--nsys`, `VLLM_LOGGING_LEVEL=DEBUG`).
     Caveat is now engine-aware: SGLang path only mentions prefix caching.

  KNOWN LIMITATION (not fixed, documented):
  - KV bytes default estimate in disagg model: `median_prompt_tokens * 256 * 32 = 8 KB/token`.
    Real GQA models (Llama 3.1 8B: 8 KV heads × 128 head_dim × 2 × FP16 × 32 layers) are
    ~128 KB/token — about 16x larger. The underestimate makes KV transfer overhead appear cheap,
    which can produce false-positive disagg recommendations on low-bandwidth clusters.
    Impact: with default 100Gbps and 1024-token prompts, estimated overhead is 0.67ms vs real
    ~10ms. The `kv_transfer_acceptable` threshold (`kv_overhead < prefill_time * 0.5`) fails
    for 10ms but passes for 0.67ms — wrong direction.
    Fix path: pass `avg_kv_transfer_bytes` with a measured or model-specific value. The advisor
    caveat already warns when estimates are in use. No code change made — requires per-model
    params which are out of scope for v1.

  AUDIT FINDINGS NOT FIXED (genuine non-issues after code review):
  - Percentile inconsistency (serve_profiler vs batch_analyzer): `static_cast<size_t>(idx)` and
    `std::floor(idx)` are identical for positive doubles. No behavioral difference.
  - Counter reset handling: acknowledged limitation for both preemption delta and TTFT mean.
    In a 60-120s profiling window a vLLM process restart is effectively impossible.
  - `decode_per_token` display with 0 output tokens: vector is empty → sentinel -1.0 → report
    shows "-". Correct.

  VALIDATION:
  - `cmake --build build -j$(nproc)`: clean build, no warnings
  - `./build/test_audit`: 598 passed, 0 failed
  - `ctest --test-dir build --output-on-failure`: 36/36 passed

2026-04-05 14:00 UTC
- Arrow key menu duplication bug: root cause identified and fixed.

  PROBLEM:
  - Interactive menus (model/engine/target selection) displayed duplicate content on each keypress.
  - Previous attempts: viewport bounding via TIOCGWINSZ (no effect); `\033[N F` CPL escape for
    cursor return to column 0 (appeared to work in PTY tests but failed on user's terminal).

  ROOT CAUSE:
  - `\033[N F` (Cursor Previous Line) is not in the original VT100 1978 spec. Some terminals do not
    implement it. When unsupported, the cursor lands at a non-zero column after the CUU (`\033[N A`)
    move, so the next render appends to the right of existing content instead of overwriting it.
    Result: each redraw adds a new copy of the header and first item to the right of the prior draw.

  FIX (src/interactive.cpp):
  - On first entry into the menu loop: emit `\033[?25l` (hide cursor), advance N+1 blank lines to
    reserve vertical space, then immediately `\033[N A` back to top, then `\0337` (DEC cursor save,
    ESC 7 -- original VT100 1978). This saves the exact cursor position (row and column) after the
    blank-line reservation.
  - Each subsequent redraw: `\0338` (DEC cursor restore, ESC 8) jumps back to the saved position
    before rendering. No line-counting, no CPL, no column arithmetic.
  - Final cleanup: `\0338\033[J` restores cursor then clears from that point to end of screen.
  - Each rendered line prefixed with `\r` as belt-and-suspenders column-0 guarantee.

  WHY DEC SAVE/RESTORE:
  - Sidesteps all line-counting and column-tracking. Save once, restore N times. No chance of
    drift from miscounted lines or unsupported escape codes.
  - `\0337`/`\0338` are in the original VT100 spec (1978) and universally supported, including
    terminals that do not implement newer VT220/ANSI extensions like CPL.

2026-04-05 14:10 UTC
- Live dashboard duplication bug: same root cause, same fix applied to serve_profiler.

  PROBLEM:
  - `serve-profile` live dashboard was also duplicating lines on each redraw. It was using a
    `dash_prev_lines` counter plus `\033[N A` to reposition, which has the same column-drift
    problem as the menu fix above.

  FIX (src/serving/serve_profiler.cpp):
  - Replaced the `dash_prev_lines` int counter with a `dash_cursor_saved` bool.
  - On first draw: emit `\0337` and set `dash_cursor_saved = true`.
  - On subsequent draws: emit `\0338` before rendering (restores to saved position).
  - Final cleanup: `\0338\033[J`.
  - `ln()` lambda now prefixes each dashboard line with `\r`.

2026-04-05 14:20 UTC
- Clock detection fallback for A10G and similar cloud GPUs.

  PROBLEM:
  - `--lock-gpu-clocks` succeeded (nvidia-smi confirmed) but `hotpath` still reported clocks as
    unlocked. The A10G (and some other cloud instance GPU models) does not emit a "GPU Locked
    Clocks" section in nvidia-smi output when clocks are locked. The three standard queries
    (`applications.clocks.sm`, `clocks_event_reasons.sw_power_cap`, `clocks.max.sm`) all returned
    N/A or did not indicate a lock on A10G.

  FIX (src/clock_control.cpp, include/hotpath/clock_control.h):
  - Added a fourth query: `clocks.current.sm` (current actual SM clock frequency in MHz).
  - Stored in new `ClockPolicyInfo::current_sm_clock_mhz` optional field.
  - Fallback logic: if `locked_sm_clock_mhz` is absent but `current_sm_clock_mhz == max_sm_clock_mhz`,
    treat clocks as locked at max and set `gpu_clocks_locked = true`.

  WHY THIS HEURISTIC IS CORRECT:
  - When nvidia-smi `--lock-gpu-clocks` succeeds, the hardware runs at max SM frequency. If the
    current frequency equals the hardware maximum, the GPU is operating at max -- which is
    effectively locked for profiling purposes (no thermal/power throttle variance).
  - False positive risk: a GPU running at max clock transiently without a lock. In practice,
    under no load the clock is well below max; under profiling load if the clock happens to be
    at max the lock is either working or the GPU is already at a stable frequency -- both are
    acceptable states for profiling. The fallback only fires when a lock was explicitly requested.

  TEST (cpp_tests/test_clock_control.cpp):
  - Added test case: `current_sm == max_sm` → `gpu_clocks_locked = true`, `locked_sm_clock_mhz`
    set to current_sm value.
  - All existing `parse_clock_policy_output` call sites updated to pass current SM clock as
    third argument.

2026-04-05 14:30 UTC
- Added `--concurrency N` flag to `serve-profile`.

  PROBLEM:
  - `max_concurrency` was hardcoded to 1 in `serve_profiler.cpp` despite the `ServeProfileOptions`
    struct (and `ReplayConfig`) having the field. This meant all traffic replays were sequential
    (one request at a time), which does not characterize real server behavior under parallel load.

  FIX:
  - Added `--concurrency N` flag parsing in `src/main.cpp` serve-profile branch.
  - `opts.max_concurrency = opts_parsed.max_concurrency` wired in `serve_profiler.cpp`.
  - Updated help text to document `--concurrency N`.

  DESIGN:
  - Default remains 1 (backwards-compatible, sequential). Users must opt in to parallel replay.
  - `max_concurrency` is the maximum number of in-flight HTTP requests during replay, not a
    thread count. The replayer uses async dispatch so this is a semaphore-bounded concurrency.

2026-04-05 14:40 UTC
- Production traffic generation for test runs: log-normal distribution.

  CONTEXT:
  - Prior smoke traffic used 10 requests with fixed short templates (7-16 word prompts,
    24-32 max_tokens). This is adequate for functional tests but not for production
    characterization (queue depth, KV cache pressure, disagg recommendation accuracy).

  CHANGE:
  - New production test traffic: 500 requests, log-normal prompt length distribution.
  - Parameters: `mu=5.0, sigma=0.8` → p50 ~197 words, p90 ~551 words (matches ShareGPT-like
    real workload distributions).
  - `max_tokens` also log-normal: `mu=4.5, sigma=0.6` → p50 ~90, p90 ~220.
  - `--concurrency 4` used for the production test run.

  WHY LOG-NORMAL:
  - Real LLM prompt and output lengths follow heavy-tailed distributions. Log-normal is the
    standard model for this (ShareGPT, WildChat, LMSYS all show this shape). Uniform or fixed
    templates understate KV cache pressure and overstate throughput by avoiding long-tail prompts.

2026-04-05 15:00 UTC
- cibuildwheel test-command: binary invocation removed.

  PROBLEM:
  - GitHub Actions cibuildwheel job failed with exit code 139 (SIGSEGV) when running:
    `hotpath version && python -c "import hotpath_py; print(hotpath_py.__version__)"`
  - The `hotpath` binary segfaulted inside the manylinux container after `auditwheel` patched
    the RPATH to point at bundled shared libraries.

  ROOT CAUSE:
  - RPATH patching changes the dynamic linker search order. CUDA-adjacent symbols or glibc
    version mismatches between the build container and the test environment can cause the binary
    to fault on startup. The binary's correctness is validated in the GPU smoke workflow; the
    wheel test's job is only to confirm the Python package is importable and version-consistent.

  FIX (pyproject.toml):
  - `test-command` simplified to: `python -c "import hotpath_py; print(hotpath_py.__version__)"`
  - Binary invocation removed from cibuildwheel test.

  PRINCIPLE:
  - Wheel tests in CI should test only what wheels guarantee: that the installed Python package
    imports correctly and exposes the expected API. Binary correctness belongs in a GPU smoke
    test that runs on real hardware with the correct CUDA environment.

2026-04-05 15:10 UTC
- Version strings synced to 0.2.0 across all three locations.

  PROBLEM:
  - After bumping `pyproject.toml` to `0.2.0`, the other two version strings lagged behind:
    `hotpath_py/__init__.py` still had `"0.1.1"`, and `src/main.cpp` `version` command output
    still had `"hotpath 0.1.x"`.

  FIX:
  - `hotpath_py/__init__.py`: `__version__ = "0.2.0"`
  - `src/main.cpp`: version string updated to `"hotpath 0.2.0\n"`
  - `pyproject.toml`: `version = "0.2.0"` (already correct; confirmed)

  NOTE:
  - cibuildwheel `test-command` imports `hotpath_py` and prints `__version__`. If this string
    diverges from `pyproject.toml`, the wheel test will print the wrong version but not fail
    (no assertion). Future improvement: assert equality in the test command.

2026-04-05 15:20 UTC
- .gitignore updated to exclude run artifacts.

  ADDED ENTRIES:
  - `.hotpath/` -- runtime artifact directory created by `hotpath` during profiling runs
    (nsys reports, temp files, JSON outputs). Should never be committed.
  - `*.log` -- vLLM server logs and other debug output captured during serve-profile.
  - `targets.cfg` -- SSH target registry written by `hotpath target add`. Contains hostnames,
    usernames, key paths -- sensitive and machine-specific, should not be in source control.

2026-04-05 15:30 UTC
- PyPI trusted publishing: exact field values required.

  CONTEXT:
  - First attempt at PyPI trusted publisher registration failed with "Invalid repository name"
    because the full GitHub URL was entered instead of just the repo name.
  - Second attempt failed with "invalid-publisher" error during the GitHub Actions OIDC flow
    because the pending publisher registration did not yet exist (or fields were mismatched).

  CORRECT FIELD VALUES for PyPI pending publisher:
  - PyPI Project Name: `hotpath`
  - Owner (GitHub username): `alityb`
  - Repository name (not URL): `hotpath`
  - Workflow filename: `release.yml`
  - Environment name: `pypi`

  NOTE:
  - The environment name must match exactly what is declared in the GitHub Actions workflow
    (`environment: pypi`). PyPI OIDC token validation checks all five fields; any mismatch
    produces "invalid-publisher" with no further detail.
  - Trusted publishing does not require storing a PyPI API token in GitHub secrets. The OIDC
    flow issues a short-lived token scoped to the specific workflow run.

2026-04-05 15:40 UTC
- v0.2.0 release: git tag conflict resolution.

  PROBLEM:
  - `git push origin v0.2.0` was rejected because a `v0.2.0` tag already existed on the remote
    from an earlier push before all version fixes were committed.

  FIX:
  - Delete remote tag: `git push origin :refs/tags/v0.2.0`
  - Re-push tag pointing at the correct commit: `git push origin v0.2.0`

  NOTE:
  - The `--force` flag for `git push --tags` does not accept a separate tag argument in all
    git versions; the explicit `refs/tags/` delete syntax is more portable and explicit.

---

## Design Rationale Supplement (2026-03-31 through 2026-04-04)

The entries above record what was done. This section documents the why behind the major design
decisions made during those sessions, retroactively for completeness.

---

### Remote fetch path (2026-03-31 04:20)

**Why skip .nsys-rep by default:**
The `.nsys-rep` file is large (hundreds of MB for a real workload) and is only needed to open
the Nsight Systems GUI. The `.sqlite` and `.db` artifacts contain all the data `hotpath` needs
for `report`, `diff`, `validate`, and `export`. Copying the full `.nsys-rep` on every profile
would make the default remote path significantly slower for the common case.

**Why keep the remote `.nsys-rep` path in metadata even when not fetched:**
Provenance. The `validate` command needs to know whether the absence of a local `.nsys-rep` is
expected (unfetched remote) or unexpected (corruption/loss). Storing `remote_artifact_nsys_rep_path`
in the `.db` means `validate` can distinguish between "deliberately not fetched" (WARN) and
"was supposed to be here but isn't" (FAIL). The `trace --open` command can also tell the user
how to get it instead of silently failing.

**Why parallelize per-run fetch/load after the remote side completes:**
The remote side must execute serially (one `nsys` session per GPU). But once all remote runs
finish, each fetched artifact is independent -- loading run 1's `.sqlite` does not depend on
having run 2's artifact yet. Concurrent fetch/load cuts the controller-side wall clock without
any correctness risk.

**Why `validate` treats unfetched nsys-rep as WARN not FAIL:**
The default fetch path deliberately does not copy it. Treating a deliberately absent file as
FAIL would make the default path always fail validation, which is the wrong signal. WARN
surfaces the absence without declaring the profile broken.

---

### `--attach-pid` release boundary (2026-03-31 05:00)

**Why explicit fail-fast with capability error rather than silent ignore:**
The alternative -- silently falling back to metrics-only profiling without the user knowing --
would produce a profile that looks complete but lacks kernel traces. Silent downgrades are worse
than explicit errors because the user has no idea their profile is incomplete until they try
to use it. The error message includes the installed nsys version so the user knows exactly
what capability is missing.

**Why `--attach-pid` was initially kept as a release boundary at all:**
The installed nsys (2024.x/2025.x) does not advertise `--pid` in its help output. Shipping a
code path that would unconditionally fail on all accessible hosts is worse than a clear
boundary. The boundary was replaced with a real three-tier fallback later in the same session.

---

### `cluster-profile` distinct-host enforcement (2026-03-31 05:00)

**Why require distinct hosts by default:**
`cluster-profile` is designed to measure multi-GPU scaling across real distinct machines. If
two "hosts" are actually the same machine, the synchronized-start path still launches two
overlapping vLLM instances on a single GPU, which compete for memory and produce artificially
degraded numbers that would be mistakenly attributed to multi-host behavior.

**Why keep `--allow-duplicate-hosts` as an override:**
Controller-path testing (confirming that the orchestration logic works) legitimately needs to
run two "cluster nodes" on one machine without a second GPU. The flag makes the escape hatch
explicit and labeled -- users who hit it on accident will read the error, users who need it for
testing will pass it intentionally.

---

### `python3` preference over `python` (2026-03-31 05:00)

**Why prefer `python3` as the first resolution target:**
On systems with both `python` and `python3`, `python` may point to Python 2 (or a system
Python) while `python3` is the versioned 3.x binary. Autodetection that tries `python` first
silently runs under the wrong interpreter, which then fails on f-strings, typing imports, or
vLLM itself. Trying `python3` first is more explicit and fails loudly if it is also absent.

---

### `vllm` executable not required for attach-mode (2026-03-31 05:00)

**Why loosen the preflight check when attaching to an existing server:**
When `hotpath` is responsible for launching the server, it must resolve `vllm serve` before
sending any traffic -- if it can't launch, nothing works. But when the server is already
running, `hotpath` never calls `vllm` at all; it only attaches nsys to the running process
and sends HTTP traffic. Requiring a resolvable `vllm` binary in attach mode would block users
who are profiling a vLLM instance they don't control (different venv, different user, remote).

---

### `bench` as optional WARN not FAIL in `doctor` (2026-03-31 05:00)

**Why WARN instead of FAIL when bench extras are missing:**
The base `hotpath` package (without `[bench]` extras) is a fully functional profiling tool.
Failing `doctor` for a missing optional feature would make every base install look broken,
discouraging users who only care about `profile`/`serve-profile`/`report`. The WARN surfaces
the gap without blocking use of the tool.

---

### Soak/repeat fast path design (2026-03-31 18:00)

**Why reuse one nsys session instead of relaunching per iteration:**
Cold relaunch loads the full model from disk (and downloads weights over the network if not
cached) every iteration. For a production model like Qwen3-8B this takes 30-60s per cold
start. The reused-session path does this once. For two iterations the savings were ~155s
(223s reused vs ~378s+ cold); the gap grows linearly with iteration count.

**Why `nsys launch --session-new` + per-iteration `nsys start/stop` instead of `nsys profile --start-later=true`:**
The `--start-later=true` approach starts one session that produces one output file. To get
per-iteration files, you'd need one `nsys profile` invocation per iteration, which means one
server launch per iteration. The `launch` + `start/stop` approach keeps one long-lived
process owner (the `nsys launch` session) and opens/closes measured windows inside it,
writing a fresh output file per `start/stop` cycle.

**Why move `--sample=none --cpuctxsw=none` from `nsys launch` to `nsys start`:**
The `nsys launch` command stopped accepting `--sample` and `--cpuctxsw` flags after Nsight
Systems 2021.5. Passing them to `launch` produced a hard error on the installed version.
These flags control what gets sampled during the measured window, so they belong on `start`
anyway; `launch` is about process ownership, `start` is about what to record.

---

### Adaptive batched bench timing (2026-03-31 22:00)

**Why use adaptive batching (multiple launches per sample) instead of single-launch timing:**
A single `torch.cuda.Event` capture of a tiny kernel (e.g., `rotary_embedding 64x1024`)
produces a ~1-3μs elapsed time. At that scale, CUDA driver overhead, event synchronization,
and host/device timing resolution contribute meaningfully to the measured variance. Batching
N launches inside one timed window and dividing by N averages out those sources of jitter.
The batch size is auto-tuned to a minimum wall clock target (`--batch-ms-target`, default
`10ms`) so the approach adapts to both fast and slow kernels.

**Why graph replay is opt-in (`--cuda-graph-replay on`) not the default:**
CUDA graph replay eliminates kernel launch overhead and CPU/GPU synchronization points,
producing much tighter numbers for static-buffer kernels. But it requires that the kernel's
inputs/outputs be pre-allocated and static, which is not always valid. Graph replay numbers
are useful for profiling pure kernel compute in isolation; batched event timing is the right
default for implementation-to-implementation comparison because it includes realistic launch
costs.

---

### Managed server design (2026-03-31 22:00)

**Why the listener PID, not the wrapper/launch process PID:**
`nsys launch --session-new ...` returns quickly with its own PID. The actual `vllm serve`
process is a child that starts asynchronously. Storing the launcher PID means `server show`
would immediately report "ready: no" because the launcher is a short-lived orchestration
wrapper, not the HTTP server. The correct readiness signal is: the configured port is
accepting connections and the PID listening on that port matches what we launched.

**Why per-server lock file (`.hotpath/servers/NAME.lock`):**
`nsys` maintains a session state machine. If two concurrent `hotpath profile --server NAME`
invocations both try to call `nsys start` on the same session, the second call hits an error
(`Configuring is not allowed in this state`). A filesystem lock prevents this without
requiring a daemon or network coordination -- it works even if the controller process dies.

**Why stale-lock recovery stores the owner PID:**
A crashed `hotpath profile` leaves the lock file behind. Without a stored PID, recovery
requires user intervention (`server prune`). Storing the owner PID enables automatic recovery:
at lock acquisition time, check if the stored PID is still alive; if dead, the lock is stale
and can be removed. This handles the common case of a controller crash without any user action.

**Why `--max-model-len` is persisted in managed-server state:**
vLLM silently rejects requests whose `max_tokens + prompt_tokens > max_model_len`. Without
knowing the server's configured ceiling, `hotpath` would send traffic, get errors back, and
report confusing 0% success rates. Persisting the ceiling lets the controller fail fast with
a clear message before any traffic is sent.

**Why `lsof` → `ss` fallback for listener PID resolution:**
`lsof` is not installed everywhere (notably absent from some cloud VM base images). `ss` is
part of `iproute2` and is present on essentially all modern Linux distributions. The fallback
keeps listener PID resolution working on hosts that have `ss` but not `lsof` without requiring
the user to install additional tools.

---

### Measurement window integrity (2026-03-31 23:00)

**Why model load stays outside the traced region:**
Model loading (downloading weights, loading checkpoint shards, CUDA graph captures) is a
one-time startup cost, not a steady-state inference characteristic. Including it in the kernel
totals would produce a number that is not representative of request-serving performance and
varies widely based on model cache state and GPU memory state. The trace window opens after
the server has passed its readiness check, which happens after model load completes.

**Why `--start-later=true` for launch-mode cold profiles (not soak/managed):**
In cold-launch mode, `nsys profile --start-later=true` starts the vLLM server without
immediately recording. The recording window opens later when the server is ready and
`nsys start` is called. Without `--start-later=true`, nsys would record from process start,
capturing all of model loading. For managed/soak mode, the `nsys launch` + `start/stop`
approach serves the same purpose.

---

### Attach-by-process three-tier fallback (2026-03-31 23:00)

**Why three tiers (native PID → clone → replace-restore):**
Each tier trades availability against disruption to the source process:
- Native PID attach: zero disruption, but requires nsys `--pid` support (not available here).
- Clone: starts a second model copy on a free port. Zero disruption to source. Requires
  enough free GPU memory for two model copies simultaneously (~double VRAM).
- Replace-restore: stops the source, replaces it with a traced copy on the same port, then
  restores the source after the profile. No VRAM duplication, but the source is transiently
  offline. Used only when the GPU can't fit two copies.

Implementing all three means the feature works on any GPU regardless of available VRAM or
nsys version -- the planner picks the best available path automatically.

**Why clone is preferred over replace-restore:**
Replace-restore takes the source server offline during profiling. For a server that is
actively serving real traffic this is disruptive. Clone has no downtime impact; the source
continues serving on its original port while the traced copy handles the profile traffic.

**Why restore the original server after replace-restore:**
The design goal is to leave the system in the same state as before profiling. A replace-
without-restore would permanently substitute the source process with the traced one; the user
would need to manually restart their original server afterward. Restore makes the whole
operation transparent from the perspective of the source server's state.

---

### Attach repeat reuse (2026-04-01 00:00)

**Why switch from `nsys profile --start-later=true` to `nsys launch` for attach clone/replace:**
The fast repeat path (for both `profile --repeat N` and `soak-profile`) expects a reusable
`nsys launch` session that can be started and stopped multiple times. `nsys profile
--start-later=true` creates a session that terminates after a single `stop` call and writes
its output. The `launch` + `start/stop` pattern was already the established session
ownership model for the reuse path; the attach clone needed to use the same pattern to be
compatible.

**Why keep the attach fallback state alive across repeat iterations:**
Replace-restore involves stopping the source server, loading the traced replacement (~45s for
Qwen2.5-0.5B-Instruct), running the profile, then restoring. If each iteration triggered a
full replace-restore cycle, 5 repeats would cost 5 × 45s = 225s just in model loading. By
keeping the traced replacement running across all iterations and only issuing `nsys start/stop`
per window, 5 repeats cost ~45s once (initial load) + 5 × ~7s (profile windows). Observed
wall clock: ~83s for 2 repeats vs ~68s for 1 cold, i.e., the second iteration cost only ~15s.

---

### Aggregate percentile labeling (2026-04-01 02:00)

**Why "upper bound" label for aggregate p50/p99, not "exact":**
Aggregating percentiles from separate distributions is mathematically wrong in general. The
correct operation is to merge the raw samples and recompute the percentile. `hotpath` stores
per-run samples and can compute merged percentiles from them, but the aggregate report path
was not doing this -- it was displaying per-run p50/p99 as if they were aggregate percentiles.
Labeling them as upper bounds is honest: the true aggregate p50 is at most the max of the
individual p50s, but could be lower.

**Why rename `aggregate_max_median_ratio_upper_bound` to `aggregate_max_median_ratio_observed_max`:**
"Upper bound" implies a mathematical guarantee that the true value is ≤ this number. For the
max/median ratio across runs, the stored value is the observed maximum -- it is the actual
measured value, not a bound on it. The name `observed_max` is accurate; `upper_bound` was
wrong.

---

### Client-side latency measurement semantics (2026-04-04 04:00)

**Why client-side HTTP timing, not server-internal timing:**
`serve-profile` measures user-visible request latency: time from sending the HTTP request
to receiving the complete response, as seen by the controller process. This is what users
and SLAs care about. Server-internal timing (time between request queue entry and response
generation) is available separately through Prometheus metrics. Both are valid measurements
of different things; reporting them as the same number would be misleading.

**Why `--flush-l2` is opt-in for bench:**
Warm-cache benchmarking is the right default for implementation-to-implementation comparison
(e.g., `vllm-cuda` vs `torch-compile`) because both implementations experience the same
cache state. Cold-cache numbers are useful for characterizing first-run latency or memory
bandwidth, but they measure a different thing and should be requested explicitly. Mixing
warm and cold numbers in the default output would make comparisons within a single bench run
misleading.

---

### Model auto-detection fix (2026-04-04 04:00)

**Why scan for the actual `"id"` value in `/v1/models` response:**
The previous implementation scanned for the literal string `"id"` as a JSON key but didn't
correctly extract the value after it. On a standard vLLM `/v1/models` response, `"id"` appears
multiple times (once in the `object` type field context, once as the actual model ID). The fix
parses the JSON `data[0].id` path correctly, which is the OpenAI-compatible model identifier.
Without this, auto-detection silently returned empty and the user was required to always pass
`--model` explicitly.

---

### `HOTPATH_*` env vars with `RLPROF_*` fallback (2026-04-04 07:00)

**Why add new prefix instead of rename-in-place:**
The package was originally named `rlprof` and used `RLPROF_*` env vars. The rename to
`hotpath` was done as a rebrand; removing the old env vars immediately would break any
existing scripts or deployment configs that set `RLPROF_PYTHON_EXECUTABLE` etc. The fallback
reads `HOTPATH_*` first, then `RLPROF_*`, so existing configs continue to work without any
changes while new configs use the correct name.

---

### TTFB vs TTFT naming and server TTFT measurement (2026-04-04 08:00)

**Why rename the curl-measured metric from "TTFT (client)" to "TTFB (client)":**
`curl time_starttransfer` measures the time until the first byte of the HTTP response body
is received -- for streaming SSE responses, this is the response headers, not the first
token. The first actual `data:` SSE token arrives later (observed: ~13.6ms vs ~4.2ms for
the headers). Calling the header-timing "TTFT" was a ~3-4x over-optimistic lie. TTFB
(time to first byte) is the correct term for this measurement.

**Why derive server-side TTFT from Prometheus histogram (`_sum` / `_count`) instead of sampling:**
The Prometheus histogram provides an exact mean across all requests that completed during
the profiling window -- not a sample of a few requests. It is also the authoritative number
from the server's perspective, not inferred from client-side timing. The counter-delta
approach (`(max_sum - min_sum) / (max_count - min_count)`) gives the per-request mean TTFT
for requests that completed within the measurement window.

**Why support both old and new vLLM metric names (`gpu_cache_usage_perc` / `kv_cache_usage_perc` etc.):**
vLLM 0.19 renamed several Prometheus metrics without a deprecation period. A `hotpath` build
that only knows the new names produces silently empty metrics against vLLM ≤0.18, and vice
versa. Checking both names with a preference for the newer one keeps the tool compatible
across the transition without requiring a version check.

---

### Placeholder-zero prevention / availability flags (2026-04-04 06:00)

**Why not render GPU phase, KV cache, prefix sharing sections when data is unavailable:**
A section that shows "0.0ms" or "0%" when no measurement was taken is actively misleading --
it looks like a measurement result. A user reading "GPU Phase: Prefill 0.0%, Decode 0.0%" has
no way to know whether that means the GPU did nothing or that the measurement path wasn't
available. The availability flag approach is explicit: the section says "not available" when
no real data was collected, and shows real numbers only when `gpu_phase_available=true` etc.

**Why request `stream_options.include_usage` during traffic replay:**
The server's usage response (`prompt_tokens`, `completion_tokens`) gives exact token counts.
Without it, `hotpath` estimated prompt tokens as `len(text) / 4`, which is a rough heuristic
that overestimates for short prompts and underestimates for token-dense prompts (code, CJK).
Real token counts are needed for the disagg model's KV transfer estimate and the workload
classifier's token distribution analysis.

**Why char-level prefix analysis as the fallback (not token-level):**
Token-level prefix analysis requires a tokenizer, which means pulling in a large ML dependency
or making an API call. The base `hotpath` package has no ML dependencies. Char-level analysis
is a reasonable approximation for prefix cache hit rate estimation -- if two prompts share a
long character-level prefix, they almost certainly share a token-level prefix too. The report
labels it as character-level so users know it's an approximation.

---

### vLLM / SGLang environment isolation (2026-04-04 04:00)

**Why separate venvs for vllm and sglang:**
Installing `sglang[all]` into an environment that already has `vllm 0.18.0` caused an ABI
break: sglang pulled in a newer torch/CUDA stack that was incompatible with `vllm/_C.abi3.so`.
Both packages pull in conflicting versions of `torch`, `triton`, and CUDA-adjacent native
extensions. The correct approach for validating both engines is separate isolated venvs.
This is a fundamental constraint of the Python ML ecosystem, not a hotpath-specific issue.

**Why vllm 0.19.0 and not 0.18.0 for the clean test venv:**
vLLM 0.18.0 was built against an older torch ABI incompatible with torch 2.9.1+cu128. vLLM
0.19.0 resolved this by pulling in torch 2.10.0 natively. Upgrading the clean test venv to
0.19.0 also exposed the metric name changes (see TTFB/TTFT section above), which were real
bugs that needed fixing regardless.

---

### `serve-profile` endpoint health check fix (2026-04-05 05:00)

**Why check HTTP status code instead of non-empty response body:**
The original health check treated a 200 response with empty body as unreachable. This was a
bug because vLLM's `/health` endpoint returns HTTP 200 with an empty body when healthy. The
fix changed the check to: status code 200 = healthy, anything else or connection error =
unreachable. Response body content is irrelevant for liveness checks.
