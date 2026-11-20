"""Benchmark the MUSA short-sequence TopK paths."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch_musa

import deep_select


DEFAULT_CASES = (
    (1, 1024, 128),
    (1, 2048, 256),
    (1, 4096, 512),
    (2, 4096, 512),
    (6, 4096, 512),
)


def measure_case(rows: int, width: int, topk: int, warmup: int, iters: int) -> dict:
    torch.manual_seed(9000 + rows + width + topk)
    cpu_input = torch.randn((rows, width), dtype=torch.bfloat16)
    musa_input = cpu_input.to("musa")

    for _ in range(warmup):
        _, indices = deep_select.topk(
            musa_input,
            topk,
            indices_type=torch.int32,
            return_value=False,
            abort_when_nan_found=False,
        )
    torch.musa.synchronize()

    start = torch.musa.Event(enable_timing=True)
    end = torch.musa.Event(enable_timing=True)
    event_times = []
    wall_start = time.perf_counter()
    for _ in range(iters):
        start.record()
        _, indices = deep_select.topk(
            musa_input,
            topk,
            indices_type=torch.int32,
            return_value=False,
            abort_when_nan_found=False,
        )
        end.record()
        end.synchronize()
        event_times.append(start.elapsed_time(end))
    torch.musa.synchronize()
    wall_ms = (time.perf_counter() - wall_start) * 1000.0 / iters

    event_times.sort()
    selected = cpu_input.gather(1, indices.cpu().long())
    reference = torch.topk(
        cpu_input, topk, dim=1, largest=True, sorted=False
    ).values
    value_diff = torch.sort(selected.float().flatten()).values.sub(
        torch.sort(reference.float().flatten()).values
    ).abs().max().item()

    return {
        "case": [rows, width, topk],
        "event_median_ms": event_times[len(event_times) // 2],
        "event_p10_ms": event_times[len(event_times) // 10],
        "event_p90_ms": event_times[(len(event_times) * 9) // 10],
        "wall_avg_ms": wall_ms,
        "max_sorted_value_diff": float(value_diff),
        "index_sum": int(indices.to(torch.int64).sum().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = {
        "device": torch.musa.get_device_name(0),
        "torch": torch.__version__,
        "torch_musa": torch_musa.__version__,
        "dtype": "bfloat16",
        "warmup": args.warmup,
        "iters": args.iters,
        "results": [
            measure_case(*case, args.warmup, args.iters) for case in DEFAULT_CASES
        ],
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
