#!/usr/bin/env python3
"""One-file MUSA benchmark for DeepSelect."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch_musa

import deep_select


CASES = {
    "short": ((1, 1024, 128), (1, 2048, 256), (1, 4096, 512),
              (2, 4096, 512), (6, 4096, 512)),
    "wide": ((1, 16384, 512), (1, 131072, 512), (2, 65536, 512),
             (6, 131072, 512), (1, 1048576, 1024)),
}


def run_case(rows: int, width: int, topk: int, warmup: int, iters: int,
             profile: bool = False) -> dict:
    torch.manual_seed(9000 + rows + width + topk)
    cpu_input = torch.randn((rows, width), dtype=torch.bfloat16)
    musa_input = cpu_input.to("musa")

    def invoke():
        return deep_select.topk(
            musa_input, topk, indices_type=torch.int32,
            return_value=False, abort_when_nan_found=False,
        )

    for _ in range(warmup):
        _, indices = invoke()
    torch.musa.synchronize()

    profiler = None
    if profile:
        profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.MUSA],
            record_shapes=False,
        )
        profiler.__enter__()

    start = torch.musa.Event(enable_timing=True)
    end = torch.musa.Event(enable_timing=True)
    event_times = []
    wall_start = time.perf_counter()
    for _ in range(iters):
        start.record()
        _, indices = invoke()
        end.record()
        end.synchronize()
        event_times.append(start.elapsed_time(end))
    torch.musa.synchronize()
    wall_ms = (time.perf_counter() - wall_start) * 1000.0 / iters
    if profiler is not None:
        profiler.__exit__(None, None, None)

    event_times.sort()
    selected = cpu_input.gather(1, indices.cpu().long())
    reference = torch.topk(cpu_input, topk, dim=1, largest=True, sorted=False).values
    value_diff = torch.sort(selected.float().flatten()).values.sub(
        torch.sort(reference.float().flatten()).values
    ).abs().max().item()
    result = {
        "case": [rows, width, topk],
        "event_median_ms": event_times[len(event_times) // 2],
        "event_p10_ms": event_times[len(event_times) // 10],
        "event_p90_ms": event_times[(len(event_times) * 9) // 10],
        "wall_avg_ms": wall_ms,
        "max_sorted_value_diff": float(value_diff),
        "index_sum": int(indices.to(torch.int64).sum().item()),
    }
    if profiler is not None:
        result["musa_table"] = profiler.key_averages().table(
            sort_by="self_device_time_total", row_limit=20
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("short", "wide", "all"), default="short")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    suites = ("short", "wide") if args.suite == "all" else (args.suite,)
    report = {
        "device": torch.musa.get_device_name(0),
        "torch": torch.__version__,
        "torch_musa": torch_musa.__version__,
        "warmup": args.warmup,
        "iters": args.iters,
        "results": [],
    }
    for suite in suites:
        for case in CASES[suite]:
            result = run_case(*case, args.warmup, args.iters, args.profile)
            result["suite"] = suite
            report["results"].append(result)
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
