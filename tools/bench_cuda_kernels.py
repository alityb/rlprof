from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from vllm import _custom_ops as vllm_ops
import vllm.model_executor.layers.activation


State = dict[str, object]
SetupFn = Callable[[tuple[int, int], torch.dtype], State]
RunFn = Callable[[State], None]


@triton.jit
def silu_mul_kernel(
    x_ptr,
    out_ptr,
    stride_x0,
    stride_x1,
    stride_out0,
    stride_out1,
    hidden_size,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    block = tl.program_id(1)
    cols = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = cols < hidden_size

    gate = tl.load(x_ptr + row * stride_x0 + cols * stride_x1, mask=mask, other=0.0)
    up = tl.load(
        x_ptr + row * stride_x0 + (cols + hidden_size) * stride_x1,
        mask=mask,
        other=0.0,
    )
    gate_fp32 = gate.to(tl.float32)
    out = (gate_fp32 * tl.sigmoid(gate_fp32) * up.to(tl.float32)).to(gate.dtype)
    tl.store(out_ptr + row * stride_out0 + cols * stride_out1, out, mask=mask)


@triton.jit
def fused_add_rms_norm_kernel(
    x_ptr,
    residual_ptr,
    weight_ptr,
    stride_x0,
    stride_x1,
    stride_r0,
    stride_r1,
    hidden_size,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    acc = tl.zeros((), dtype=tl.float32)
    for start in range(0, hidden_size, BLOCK_SIZE):
        cols = start + tl.arange(0, BLOCK_SIZE)
        mask = cols < hidden_size
        x = tl.load(x_ptr + row * stride_x0 + cols * stride_x1, mask=mask, other=0.0)
        residual = tl.load(
            residual_ptr + row * stride_r0 + cols * stride_r1,
            mask=mask,
            other=0.0,
        )
        combined = x.to(tl.float32) + residual.to(tl.float32)
        acc += tl.sum(combined * combined, axis=0)

    inv_rms = tl.rsqrt(acc / hidden_size + eps)
    for start in range(0, hidden_size, BLOCK_SIZE):
        cols = start + tl.arange(0, BLOCK_SIZE)
        mask = cols < hidden_size
        x = tl.load(x_ptr + row * stride_x0 + cols * stride_x1, mask=mask, other=0.0)
        residual = tl.load(
            residual_ptr + row * stride_r0 + cols * stride_r1,
            mask=mask,
            other=0.0,
        )
        weight = tl.load(weight_ptr + cols, mask=mask, other=0.0)
        combined = x.to(tl.float32) + residual.to(tl.float32)
        normalized = (combined * inv_rms) * weight.to(tl.float32)
        tl.store(
            residual_ptr + row * stride_r0 + cols * stride_r1,
            combined.to(residual.dtype),
            mask=mask,
        )
        tl.store(
            x_ptr + row * stride_x0 + cols * stride_x1,
            normalized.to(x.dtype),
            mask=mask,
        )


@triton.jit
def rotary_embedding_kernel(
    q_ptr,
    k_ptr,
    pos_ptr,
    cache_ptr,
    stride_q0,
    stride_q1,
    stride_k0,
    stride_k1,
    stride_cache0,
    stride_cache1,
    num_heads,
    half_head_size,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < half_head_size
    token_idx = row // num_heads
    position = tl.load(pos_ptr + token_idx)

    q_base = q_ptr + row * stride_q0
    k_base = k_ptr + row * stride_k0
    cache_base = cache_ptr + position * stride_cache0

    cos = tl.load(cache_base + cols * stride_cache1, mask=mask, other=0.0).to(tl.float32)
    sin = tl.load(
        cache_base + (cols + half_head_size) * stride_cache1,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    q0_in = tl.load(q_base + cols * stride_q1, mask=mask, other=0.0)
    q1_in = tl.load(
        q_base + (cols + half_head_size) * stride_q1,
        mask=mask,
        other=0.0,
    )
    k0_in = tl.load(k_base + cols * stride_k1, mask=mask, other=0.0)
    k1_in = tl.load(
        k_base + (cols + half_head_size) * stride_k1,
        mask=mask,
        other=0.0,
    )

    q0 = q0_in.to(tl.float32)
    q1 = q1_in.to(tl.float32)
    k0 = k0_in.to(tl.float32)
    k1 = k1_in.to(tl.float32)

    q_out0 = q0 * cos - q1 * sin
    q_out1 = q1 * cos + q0 * sin
    k_out0 = k0 * cos - k1 * sin
    k_out1 = k1 * cos + k0 * sin

    tl.store(q_base + cols * stride_q1, q_out0.to(q0_in.dtype), mask=mask)
    tl.store(q_base + (cols + half_head_size) * stride_q1, q_out1.to(q1_in.dtype), mask=mask)
    tl.store(k_base + cols * stride_k1, k_out0.to(k0_in.dtype), mask=mask)
    tl.store(k_base + (cols + half_head_size) * stride_k1, k_out1.to(k1_in.dtype), mask=mask)


def silu_setup(shape: tuple[int, int], dtype: torch.dtype) -> State:
    batch, hidden = shape
    return {
        "x": torch.randn((batch, hidden * 2), device="cuda", dtype=dtype),
        "hidden": hidden,
    }


def silu_vllm(state: State) -> None:
    x = state["x"]
    hidden = state["hidden"]
    out = torch.empty((x.shape[0], hidden), device=x.device, dtype=x.dtype)
    torch.ops._C.silu_and_mul(out, x)


def build_silu_torch_compile() -> RunFn:
    @torch.compile(dynamic=False)
    def compiled(x: torch.Tensor) -> torch.Tensor:
        hidden = x.shape[-1] // 2
        return F.silu(x[..., :hidden]) * x[..., hidden:]

    return lambda state: compiled(state["x"])


def silu_triton(state: State) -> None:
    x = state["x"]
    hidden = int(state["hidden"])
    out = torch.empty((x.shape[0], hidden), device=x.device, dtype=x.dtype)
    block = min(max(triton.next_power_of_2(hidden), 64), 1024)
    silu_mul_kernel[(x.shape[0], triton.cdiv(hidden, block))](
        x,
        out,
        x.stride(0),
        x.stride(1),
        out.stride(0),
        out.stride(1),
        hidden,
        BLOCK_SIZE=block,
    )


def fused_add_rms_norm_setup(shape: tuple[int, int], dtype: torch.dtype) -> State:
    batch, hidden = shape
    return {
        "x": torch.randn((batch, hidden), device="cuda", dtype=dtype),
        "residual": torch.randn((batch, hidden), device="cuda", dtype=dtype),
        "weight": torch.randn((hidden,), device="cuda", dtype=dtype),
        "eps": 1e-5,
    }


def fused_add_rms_norm_vllm(state: State) -> None:
    vllm_ops.fused_add_rms_norm(
        state["x"],
        state["residual"],
        state["weight"],
        state["eps"],
    )


def build_fused_add_rms_norm_torch_compile() -> RunFn:
    @torch.compile(dynamic=False)
    def compiled(
        x: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        combined = x + residual
        variance = combined.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
        normalized = combined * torch.rsqrt(variance + eps)
        x.copy_(normalized.to(x.dtype) * weight)
        residual.copy_(combined)
        return x, residual

    return lambda state: compiled(
        state["x"],
        state["residual"],
        state["weight"],
        state["eps"],
    )


def fused_add_rms_norm_triton(state: State) -> None:
    x = state["x"]
    residual = state["residual"]
    hidden = x.shape[1]
    block = min(max(triton.next_power_of_2(hidden), 64), 1024)
    fused_add_rms_norm_kernel[(x.shape[0],)](
        x,
        residual,
        state["weight"],
        x.stride(0),
        x.stride(1),
        residual.stride(0),
        residual.stride(1),
        hidden,
        state["eps"],
        BLOCK_SIZE=block,
    )


def build_cos_sin_cache(max_position: int, head_size: int, dtype: torch.dtype) -> torch.Tensor:
    theta = torch.arange(0, head_size, 2, device="cuda", dtype=torch.float32)
    inv_freq = 1.0 / (10000.0 ** (theta / head_size))
    freqs = torch.outer(
        torch.arange(max_position, device="cuda", dtype=torch.float32),
        inv_freq,
    )
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1).to(dtype)


def rotary_setup(shape: tuple[int, int], dtype: torch.dtype) -> State:
    tokens, hidden = shape
    head_size = 64
    if hidden % head_size != 0:
        raise ValueError(
            f"rotary_embedding requires hidden size divisible by {head_size}, got {hidden}"
        )
    num_heads = hidden // head_size
    positions = torch.arange(tokens, device="cuda", dtype=torch.long).unsqueeze(0)
    return {
        "positions": positions,
        "positions_1d": positions[0].contiguous(),
        "q": torch.randn((1, tokens, num_heads, head_size), device="cuda", dtype=dtype),
        "k": torch.randn((1, tokens, num_heads, head_size), device="cuda", dtype=dtype),
        "cache": build_cos_sin_cache(max(2048, tokens), head_size, dtype),
        "head_size": head_size,
        "num_heads": num_heads,
        "is_neox": True,
    }


def rotary_vllm(state: State) -> None:
    vllm_ops.rotary_embedding(
        state["positions"],
        state["q"],
        state["k"],
        state["head_size"],
        state["cache"],
        state["is_neox"],
    )


def build_rotary_torch_compile() -> RunFn:
    @torch.compile(dynamic=False)
    def compiled(
        positions: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        cache: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        half = q.shape[-1] // 2
        selected = cache.index_select(0, positions.reshape(-1))
        cos = selected[:, :half].view(1, positions.shape[1], 1, half)
        sin = selected[:, half:].view(1, positions.shape[1], 1, half)
        q0, q1 = q[..., :half], q[..., half:]
        k0, k1 = k[..., :half], k[..., half:]
        q.copy_(torch.cat((q0 * cos - q1 * sin, q1 * cos + q0 * sin), dim=-1))
        k.copy_(torch.cat((k0 * cos - k1 * sin, k1 * cos + k0 * sin), dim=-1))
        return q, k

    return lambda state: compiled(
        state["positions"],
        state["q"],
        state["k"],
        state["cache"],
    )


def rotary_triton(state: State) -> None:
    q = state["q"].view(-1, int(state["head_size"]))
    k = state["k"].view(-1, int(state["head_size"]))
    half = int(state["head_size"]) // 2
    block = min(max(triton.next_power_of_2(half), 32), 256)
    rotary_embedding_kernel[(q.shape[0],)](
        q,
        k,
        state["positions_1d"],
        state["cache"],
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        state["cache"].stride(0),
        state["cache"].stride(1),
        int(state["num_heads"]),
        half,
        BLOCK_SIZE=block,
    )


def build_registry(
    bytes_per_element: int,
) -> dict[str, tuple[SetupFn, list[tuple[str, RunFn]], Callable[[tuple[int, int]], int]]]:
    return {
        "silu_and_mul": (
            silu_setup,
            [
                ("vllm-cuda", silu_vllm),
                ("torch-compile", build_silu_torch_compile()),
                ("triton-custom", silu_triton),
            ],
            lambda shape: shape[0] * shape[1] * 3 * bytes_per_element,
        ),
        "fused_add_rms_norm": (
            fused_add_rms_norm_setup,
            [
                ("vllm-cuda", fused_add_rms_norm_vllm),
                ("torch-compile", build_fused_add_rms_norm_torch_compile()),
                ("triton-custom", fused_add_rms_norm_triton),
            ],
            lambda shape: shape[0] * shape[1] * 5 * bytes_per_element,
        ),
        "rotary_embedding": (
            rotary_setup,
            [
                ("vllm-cuda", rotary_vllm),
                ("torch-compile", build_rotary_torch_compile()),
                ("triton-custom", rotary_triton),
            ],
            lambda shape: shape[0] * shape[1] * 4 * bytes_per_element,
        ),
    }
