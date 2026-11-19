#!/usr/bin/env python3
"""Benchmark the README FP32 sampling scenario on MUSA."""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch
import torch_musa  # noqa: F401

import deep_select


@dataclass(frozen=True)
class Result:
    batch: int
    width: int
    topk: int
    deepselect_ms: float
    torch_ms: float
    speedup: float
    deepselect_tbps: float
    torch_tbps: float
    deepselect_samples_ms: list[float]
    torch_samples_ms: list[float]
    values_match: bool
    indices_match_values: bool


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
        candidate()
        baseline()
    torch.musa.synchronize()
    candidate_samples = []
    baseline_samples = []
    for round_index in range(rounds):
        if round_index % 2 == 0:
            candidate_samples.append(_sample(candidate, iters, flush))
            baseline_samples.append(_sample(baseline, iters, flush))
        else:
            baseline_samples.append(_sample(baseline, iters, flush))
            candidate_samples.append(_sample(candidate, iters, flush))
    return candidate_samples, baseline_samples


def run_case(
    batch: int,
    *,
    width: int,
    topk: int,
    warmup: int,
    iters: int,
    rounds: int,
    flush: torch.Tensor | None,
) -> Result:
    generator = torch.Generator(device="musa")
    generator.manual_seed(20261030 + batch)
    input = torch.randn(
        batch,
        width,
        dtype=torch.float32,
        device="musa",
        generator=generator,
    )
    output = torch.empty(
        batch, topk, dtype=torch.int64, device="musa"
    )
    torch_values = torch.empty(
        batch, topk, dtype=torch.float32, device="musa"
    )
    torch_indices = torch.empty(
        batch, topk, dtype=torch.int64, device="musa"
    )
    candidate_values: torch.Tensor | None = None
    candidate_indices: torch.Tensor | None = None

    def candidate() -> None:
        nonlocal candidate_values, candidate_indices
        candidate_values, candidate_indices = deep_select.topk(
            input,
            topk,
            sorted=True,
            indices_type=torch.int64,
            output_idx=output,
            return_value=True,
            abort_when_nan_found=False,
        )

    def baseline() -> None:
        torch.topk(
            input,
            topk,
            dim=1,
            largest=True,
            sorted=True,
            out=(torch_values, torch_indices),
        )

    candidate()
    baseline()
    torch.musa.synchronize()
    assert candidate_values is not None and candidate_indices is not None
    expected = torch.topk(input, topk, dim=1, sorted=True).values
    selected = torch.gather(input, 1, candidate_indices)
    values_match = torch.equal(candidate_values.cpu(), expected.cpu())
    indices_match_values = torch.equal(selected.cpu(), candidate_values.cpu())
    if not values_match or not indices_match_values:
        raise AssertionError(
            f"sampling mismatch for batch={batch}: "
            f"values={values_match}, indices={indices_match_values}"
        )

    candidate_samples, torch_samples = _paired_samples(
        candidate,
        baseline,
        warmup=warmup,
        iters=iters,
        rounds=rounds,
        flush=flush,
    )
    candidate_ms = statistics.median(candidate_samples)
    torch_ms = statistics.median(torch_samples)
    total_bytes = (
        input.numel() * input.element_size()
        + candidate_values.numel() * candidate_values.element_size()
        + candidate_indices.numel() * candidate_indices.element_size()
    )
    return Result(
        batch=batch,
        width=width,
        topk=topk,
        deepselect_ms=candidate_ms,
        torch_ms=torch_ms,
        speedup=torch_ms / candidate_ms,
        deepselect_tbps=total_bytes / (candidate_ms * 1e-3) / 1e12,
        torch_tbps=total_bytes / (torch_ms * 1e-3) / 1e12,
        deepselect_samples_ms=candidate_samples,
        torch_samples_ms=torch_samples,
        values_match=values_match,
        indices_match_values=indices_match_values,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", default="6,256,512,768,4096")
    parser.add_argument("--width", type=int, default=129280)
    parser.add_argument(
        "--widths",
        help="comma-separated vocab sizes; overrides --width",
    )
    parser.add_argument("--topk", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--flush-mb", type=int, default=8000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    flush = torch.empty(
        args.flush_mb * 1024 * 1024, dtype=torch.uint8, device="musa"
    )
    results = []
    widths = (
        [int(value) for value in args.widths.split(",")]
        if args.widths
        else [args.width]
    )
    for width in widths:
        for batch_text in args.batches.split(","):
            result = run_case(
                int(batch_text),
                width=width,
                topk=args.topk,
                warmup=args.warmup,
                iters=args.iters,
                rounds=args.rounds,
                flush=flush,
            )
            payload = asdict(result)
            results.append(payload)
            print(json.dumps(payload, sort_keys=True), flush=True)
    report = {
        "status": "passed",
        "device": torch.musa.get_device_name(),
        "capability": list(torch.musa.get_device_capability()),
        "torch": torch.__version__,
        "dtype": "float32",
        "index_dtype": "int64",
        "sorted_value": True,
        "return_value": True,
        "cold_cache_flush_mb": args.flush_mb,
        "widths": widths,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
