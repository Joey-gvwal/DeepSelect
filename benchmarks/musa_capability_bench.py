#!/usr/bin/env python3
"""Benchmark every native MUSA public capability against torch.topk."""

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


@dataclass(frozen=True)
class Case:
    mode: str
    dtype: torch.dtype
    index_dtype: torch.dtype
    batch: int
    width: int
    topk: int


@dataclass(frozen=True)
class Result:
    mode: str
    dtype: str
    index_dtype: str
    batch: int
    width: int
    topk: int
    deepselect_ms: float
    torch_ms: float
    speedup: float
    deepselect_samples_ms: list[float]
    torch_samples_ms: list[float]


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


def _aligned_output(case: Case) -> torch.Tensor:
    alignment = 32 // case.index_dtype.itemsize
    stride = (case.topk + alignment - 1) // alignment * alignment
    return torch.empty(
        case.batch, stride, dtype=case.index_dtype, device="musa"
    )[:, : case.topk]


def _mode_flags(mode: str) -> tuple[bool, bool, bool]:
    return (
        mode == "sorted_value",
        mode.startswith("sorted_index"),
        mode in ("values", "sorted_index_values", "sorted_value", "shortcut"),
    )


def run_case(
    case: Case,
    *,
    warmup: int,
    iters: int,
    rounds: int,
    flush: torch.Tensor | None,
) -> Result:
    generator = torch.Generator(device="musa")
    generator.manual_seed(
        20261029 + case.batch * 17 + case.width + case.topk
    )
    input = torch.randn(
        case.batch,
        case.width,
        dtype=case.dtype,
        device="musa",
        generator=generator,
    )
    output = _aligned_output(case)
    sorted_value, sorted_index, return_value = _mode_flags(case.mode)
    end = None
    offsets = None
    valid_length = case.width
    if case.mode == "end_offset":
        valid_length = case.width * 3 // 4
        end = torch.full(
            (case.batch,), valid_length, dtype=torch.int32, device="musa"
        )
        offsets = torch.arange(
            case.batch, dtype=torch.int32, device="musa"
        )
    offsets_i64 = offsets.to(torch.int64) if offsets is not None else None

    candidate_values: torch.Tensor | None = None
    candidate_indices: torch.Tensor | None = None

    def candidate() -> None:
        nonlocal candidate_values, candidate_indices
        candidate_values, candidate_indices = deep_select.topk(
            input,
            case.topk,
            sorted=sorted_value,
            sorted_index=sorted_index,
            end=end,
            output_idx=output,
            output_idx_offset=offsets,
            indices_type=case.index_dtype,
            return_value=return_value,
            abort_when_nan_found=True,
        )

    torch_values = torch.empty(
        case.batch, case.topk, dtype=case.dtype, device="musa"
    )
    torch_indices = torch.empty(
        case.batch, case.topk, dtype=torch.int64, device="musa"
    )
    sorted_indices = torch.empty_like(torch_indices)
    order = torch.empty_like(torch_indices)
    sorted_values = torch.empty_like(torch_values)
    offset_indices = torch.empty_like(torch_indices)

    def baseline() -> None:
        source = input[:, :valid_length]
        torch.topk(
            source,
            case.topk,
            dim=1,
            largest=True,
            sorted=sorted_value,
            out=(torch_values, torch_indices),
        )
        if sorted_index:
            torch.sort(torch_indices, dim=1, out=(sorted_indices, order))
            if return_value:
                torch.gather(torch_values, 1, order, out=sorted_values)
        if offsets is not None:
            assert offsets_i64 is not None
            torch.add(
                torch_indices,
                offsets_i64.unsqueeze(1),
                out=offset_indices,
            )

    candidate()
    torch.musa.synchronize()
    assert candidate_indices is not None
    raw_indices = candidate_indices.to(torch.int64)
    if offsets is not None:
        assert offsets_i64 is not None
        raw_indices = raw_indices - offsets_i64.unsqueeze(1)
    selected = torch.gather(input[:, :valid_length], 1, raw_indices)
    expected = torch.topk(
        input[:, :valid_length], case.topk, dim=1, sorted=True
    ).values
    torch.testing.assert_close(
        torch.sort(selected, dim=1, descending=True).values,
        expected,
        rtol=0,
        atol=0,
    )
    if candidate_values is not None:
        if sorted_value:
            torch.testing.assert_close(
                candidate_values, expected, rtol=0, atol=0
            )
        else:
            torch.testing.assert_close(
                torch.sort(candidate_values, dim=1, descending=True).values,
                expected,
                rtol=0,
                atol=0,
            )
        torch.testing.assert_close(candidate_values, selected, rtol=0, atol=0)
    if sorted_index:
        assert torch.all(
            candidate_indices[:, 1:] >= candidate_indices[:, :-1]
        ).item()

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
    return Result(
        mode=case.mode,
        dtype="bfloat16" if case.dtype == torch.bfloat16 else "float32",
        index_dtype=("int32" if case.index_dtype == torch.int32 else "int64"),
        batch=case.batch,
        width=case.width,
        topk=case.topk,
        deepselect_ms=candidate_ms,
        torch_ms=torch_ms,
        speedup=torch_ms / candidate_ms,
        deepselect_samples_ms=candidate_samples,
        torch_samples_ms=torch_samples,
    )


def _cases() -> list[Case]:
    cases = []
    common = (
        ("indices", torch.int32),
        ("values", torch.int32),
        ("sorted_index_indices", torch.int32),
        ("sorted_index_values", torch.int64),
        ("end_offset", torch.int64),
    )
    for dtype in (torch.bfloat16, torch.float32):
        for batch in (4, 512):
            for mode, index_dtype in common:
                cases.append(Case(mode, dtype, index_dtype, batch, 131072, 512))
            cases.append(Case("shortcut", dtype, torch.int64, batch, 512, 512))
    for batch in (4, 512):
        cases.append(
            Case("sorted_value", torch.float32, torch.int64, batch, 131072, 512)
        )
    cases.extend(
        [
            Case(
                "sorted_index_values",
                torch.bfloat16,
                torch.int64,
                4,
                131072,
                4096,
            ),
            Case(
                "sorted_index_values",
                torch.float32,
                torch.int64,
                4,
                131072,
                4096,
            ),
            Case(
                "sorted_value",
                torch.float32,
                torch.int64,
                4,
                131072,
                4096,
            ),
        ]
    )
    return cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--flush-mb", type=int, default=8000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    flush = torch.empty(
        args.flush_mb * 1024 * 1024, dtype=torch.uint8, device="musa"
    )
    results = []
    for case in _cases():
        result = run_case(
            case,
            warmup=args.warmup,
            iters=args.iters,
            rounds=args.rounds,
            flush=flush,
        )
        result_dict = asdict(result)
        results.append(result_dict)
        print(json.dumps(result_dict, sort_keys=True), flush=True)
    payload = {
        "status": "passed",
        "device": torch.musa.get_device_name(),
        "capability": list(torch.musa.get_device_capability()),
        "torch": torch.__version__,
        "cold_cache_flush_mb": args.flush_mb,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
