#!/usr/bin/env python3

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch

from bench_cuda_kernels import build_registry


@dataclass
class BenchResult:
    kernel: str
    implementation: str
    shape: tuple[int, int]
    dtype: str
    avg_ms: float
    min_ms: float
    p50_ms: float
    p99_ms: float
    bandwidth_gb_s: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--shapes", default="1x4096,64x4096,256x4096")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--n-iter", type=int, default=200)
    return parser.parse_args()


def parse_shapes(spec: str) -> list[tuple[int, int]]:
    shapes: list[tuple[int, int]] = []
    for token in spec.split(","):
        if not token:
            continue
        left, right = token.split("x", 1)
        batch = int(left)
        hidden = int(right)
        if batch <= 0 or hidden <= 0:
            raise ValueError(f"invalid shape: {token}")
        shapes.append((batch, hidden))
    if not shapes:
        raise ValueError("at least one shape is required")
    return shapes


def percentile(values: list[float], quantile: float) -> float:
    values = sorted(values)
    index = int(((len(values) - 1) * quantile) + 0.999999999)
    return values[index]


def dtype_from_name(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def dtype_size(dtype: torch.dtype) -> int:
    if dtype in (torch.bfloat16, torch.float16):
        return 2
    if dtype == torch.float32:
        return 4
    raise ValueError(f"unsupported dtype: {dtype}")


def measure(fn, warmup: int, n_iter: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times: list[float] = []
    for _ in range(n_iter):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return times


def benchmark(
    kernel: str,
    implementation: str,
    setup,
    fn,
    shape: tuple[int, int],
    dtype_name: str,
    warmup: int,
    n_iter: int,
    bytes_processed: int,
) -> BenchResult:
    dtype = dtype_from_name(dtype_name)
    state = setup(shape, dtype)
    times = measure(lambda: fn(state), warmup, n_iter)
    avg_ms = sum(times) / len(times)
    return BenchResult(
        kernel=kernel,
        implementation=implementation,
        shape=shape,
        dtype=dtype_name,
        avg_ms=avg_ms,
        min_ms=min(times),
        p50_ms=percentile(times, 0.50),
        p99_ms=percentile(times, 0.99),
        bandwidth_gb_s=bytes_processed / (avg_ms / 1000.0) / 1e9,
    )


def render(results: list[BenchResult]) -> str:
    lines = [
        f"{'kernel':<18}  {'implementation':<18}  {'shape':<12}  "
        f"{'avg ms':>8}  {'min ms':>8}  {'p50 ms':>8}  {'p99 ms':>8}  {'GB/s':>11}",
        "-" * 105,
    ]
    for result in results:
        shape = f"{result.shape[0]}x{result.shape[1]}"
        lines.append(
            f"{result.kernel:<18}  {result.implementation:<18}  {shape:<12}  "
            f"{result.avg_ms:>8.3f}  {result.min_ms:>8.3f}  {result.p50_ms:>8.3f}  "
            f"{result.p99_ms:>8.3f}  {result.bandwidth_gb_s:>11.3f}"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for tools/bench_cuda.py")

    args = parse_args()
    shapes = parse_shapes(args.shapes)
    dtype = dtype_from_name(args.dtype)
    registry = build_registry(dtype_size(dtype))

    if args.kernel not in registry:
        raise SystemExit(f"unsupported kernel: {args.kernel}")

    setup, implementations, bytes_fn = registry[args.kernel]
    results: list[BenchResult] = []
    torch.set_grad_enabled(False)

    with torch.inference_mode():
        for implementation_name, fn in implementations:
            for shape in shapes:
                results.append(
                    benchmark(
                        kernel=args.kernel,
                        implementation=implementation_name,
                        setup=setup,
                        fn=fn,
                        shape=shape,
                        dtype_name=args.dtype,
                        warmup=args.warmup,
                        n_iter=args.n_iter,
                        bytes_processed=bytes_fn(shape),
                    )
                )

    print(render(results), end="")


if __name__ == "__main__":
    main()
