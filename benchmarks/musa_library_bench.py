#!/usr/bin/env python3
"""Benchmark the installed MUSA public API against torch.topk."""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch
import torch_musa

import deep_select
from deep_select import deep_select_musa


def _sample(
    fn: Callable[[], None], iters: int, flush: torch.Tensor | None
) -> float:
    starts = [torch.musa.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.musa.Event(enable_timing=True) for _ in range(iters)]
    for index in range(iters):
        if flush is not None:
            flush.zero_()
        starts[index].record()
        fn()
        ends[index].record()
    torch.musa.synchronize()
    return sum(
        starts[index].elapsed_time(ends[index]) for index in range(iters)
    ) / iters


def _paired_samples(
    candidate: Callable[[], None],
    baseline: Callable[[], None],
    *,
    warmup: int,
    iters: int,
    rounds: int,
    flush: torch.Tensor | None,
) -> tuple[list[float], list[float]]:
    for _ in range(warmup):
        if flush is not None:
            flush.zero_()
        candidate()
        if flush is not None:
            flush.zero_()
        baseline()
    torch.musa.synchronize()

    candidate_samples = []
    baseline_samples = []
    for round_idx in range(rounds):
        if round_idx % 2 == 0:
            candidate_samples.append(_sample(candidate, iters, flush))
            baseline_samples.append(_sample(baseline, iters, flush))
        else:
            baseline_samples.append(_sample(baseline, iters, flush))
            candidate_samples.append(_sample(candidate, iters, flush))
    return candidate_samples, baseline_samples


@dataclass(frozen=True)
class Result:
    dtype: str
    index_dtype: str
    batch: int
    width: int
    topk: int
    backend: str
    deepselect_ms: float
    torch_ms: float
    speedup: float
    deepselect_tbps: float
    torch_tbps: float
    deepselect_samples_ms: list[float]
    torch_samples_ms: list[float]
    values_match: bool
    unique_indices: bool


def run_case(
    dtype: torch.dtype,
    index_dtype: torch.dtype,
    batch: int,
    width: int,
    topk: int,
    warmup: int,
    iters: int,
    rounds: int,
    flush: torch.Tensor | None,
    candidate_impl: str,
) -> Result:
    generator = torch.Generator(device="musa")
    generator.manual_seed(20261028 + batch * 17 + width + topk)
    input = torch.randn(
        batch, width, dtype=dtype, device="musa", generator=generator
    )
    output_alignment = 32 // index_dtype.itemsize
    output_stride = (
        (topk + output_alignment - 1) // output_alignment * output_alignment
    )
    output = torch.empty(
        batch, output_stride, dtype=index_dtype, device="musa"
    )[:, :topk]
    torch_values = torch.empty(batch, topk, dtype=dtype, device="musa")
    torch_indices = torch.empty(batch, topk, dtype=torch.int64, device="musa")

    if candidate_impl == "cpp":
        if dtype != torch.bfloat16:
            raise ValueError("MUSA C++ benchmark supports BF16 input")
        from deep_select.musa_cpp import launch_topk_bf16

        def candidate() -> None:
            launch_topk_bf16(input, output)

    else:

        def candidate() -> None:
            deep_select.topk(
                input,
                topk,
                output_idx=output,
                indices_type=index_dtype,
                return_value=False,
                abort_when_nan_found=False,
            )

    def baseline() -> None:
        torch.topk(
            input,
            topk,
            dim=1,
            largest=True,
            sorted=False,
            out=(torch_values, torch_indices),
        )

    candidate()
    torch.musa.synchronize()
    selected = torch.gather(input, 1, output.to(torch.int64))
    expected = torch.topk(input, topk, dim=1, sorted=True).values
    actual = torch.sort(selected, dim=1, descending=True).values
    values_match = torch.equal(actual.cpu(), expected.cpu())
    sorted_indices = torch.sort(output, dim=1).values
    unique_indices = bool(
        torch.all(sorted_indices[:, 1:] != sorted_indices[:, :-1]).item()
    )
    if not values_match or not unique_indices:
        raise AssertionError(
            f"incorrect result: values_match={values_match}, "
            f"unique_indices={unique_indices}, dtype={dtype}, "
            f"batch={batch}, width={width}"
        )

    candidate_samples, baseline_samples = _paired_samples(
        candidate,
        baseline,
        warmup=warmup,
        iters=iters,
        rounds=rounds,
        flush=flush,
    )
    candidate_ms = statistics.median(candidate_samples)
    baseline_ms = statistics.median(baseline_samples)
    total_bytes = (
        input.numel() * input.element_size()
        + output.numel() * output.element_size()
    )
    fast_path = candidate_impl == "cpp" or deep_select_musa._can_use_fast_path(
        input, topk, None, False, False, output, None, False, False
    )
    if candidate_impl in ("cpp", "public") and dtype in (
        torch.bfloat16,
        torch.float32,
    ):
        backend = "musa_cpp"
    else:
        backend = "torch_fallback"
    return Result(
        dtype="bfloat16" if dtype == torch.bfloat16 else "float32",
        index_dtype="int32" if index_dtype == torch.int32 else "int64",
        batch=batch,
        width=width,
        topk=topk,
        backend=backend,
        deepselect_ms=candidate_ms,
        torch_ms=baseline_ms,
        speedup=baseline_ms / candidate_ms,
        deepselect_tbps=total_bytes / (candidate_ms * 1e-3) / 1e12,
        torch_tbps=total_bytes / (baseline_ms * 1e-3) / 1e12,
        deepselect_samples_ms=candidate_samples,
        torch_samples_ms=baseline_samples,
        values_match=values_match,
        unique_indices=unique_indices,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtypes", default="bfloat16,float32")
    parser.add_argument("--index-dtypes", default="int32")
    parser.add_argument("--batches", default="1,4,16,64")
    parser.add_argument("--widths", default="32768,131072,204800")
    parser.add_argument("--topk", type=int, default=512)
    parser.add_argument(
        "--topks",
        help="comma-separated topk values; overrides --topk",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument(
        "--flush-mb",
        type=int,
        default=0,
        help="MUSA memory to zero before every timed launch; 8000 matches README CUDA policy",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate", choices=["public", "cpp"], default="public")
    args = parser.parse_args()

    dtype_map = {"bfloat16": torch.bfloat16, "float32": torch.float32}
    index_dtype_map = {"int32": torch.int32, "int64": torch.int64}
    flush = (
        torch.empty(args.flush_mb * 1024 * 1024, dtype=torch.uint8, device="musa")
        if args.flush_mb > 0
        else None
    )
    results = []
    topks = (
        [int(value) for value in args.topks.split(",")]
        if args.topks
        else [args.topk]
    )
    for dtype_text in args.dtypes.split(","):
        dtype = dtype_map[dtype_text.strip()]
        for index_dtype_text in args.index_dtypes.split(","):
            index_dtype = index_dtype_map[index_dtype_text.strip()]
            for topk in topks:
                for width_text in args.widths.split(","):
                    for batch_text in args.batches.split(","):
                        result = run_case(
                            dtype,
                            index_dtype,
                            int(batch_text),
                            int(width_text),
                            topk,
                            args.warmup,
                            args.iters,
                            args.rounds,
                            flush,
                            args.candidate,
                        )
                        results.append(asdict(result))
                        print(
                            json.dumps(asdict(result), sort_keys=True),
                            flush=True,
                        )

    payload = {
        "status": "passed",
        "device": torch.musa.get_device_name(),
        "capability": list(torch.musa.get_device_capability()),
        "torch": torch.__version__,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
