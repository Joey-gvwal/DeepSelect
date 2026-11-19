#!/usr/bin/env python3
"""Tune the native low-batch FP32 multi-CTA partition count."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch
import torch_musa

from deep_select.musa_cpp import _module


def _sample(fn, iters: int, flush: torch.Tensor | None) -> float:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=6)
    parser.add_argument("--widths", default="129280,262144,524288,1048576")
    parser.add_argument("--parts", default="8,16,24,32,48,64")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--flush-mb", type=int, default=8000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    topk = 512
    flush = (
        torch.empty(
            args.flush_mb * 1024 * 1024, dtype=torch.uint8, device="musa"
        )
        if args.flush_mb > 0
        else None
    )
    module = _module()
    results = []
    for width_text in args.widths.split(","):
        width = int(width_text)
        generator = torch.Generator(device="musa")
        generator.manual_seed(20261030 + width)
        input = torch.randn(
            args.batch,
            width,
            dtype=torch.float32,
            device="musa",
            generator=generator,
        )
        expected = torch.topk(input, topk, dim=1, sorted=True).values
        for parts_text in args.parts.split(","):
            parts = int(parts_text)
            output = torch.empty(
                args.batch, topk, dtype=torch.int64, device="musa"
            )
            values = torch.empty(
                args.batch, topk, dtype=torch.float32, device="musa"
            )
            scratch_size = (
                args.batch * parts * 256
                + args.batch * parts
                + 7 * args.batch
            )
            scratch = torch.empty(
                scratch_size, dtype=torch.int32, device="musa"
            )

            def candidate() -> None:
                module.topk_fp32_multi(
                    input, output, scratch, parts, 256, values, False
                )
                module.sort_output(output, values, None, width, True)

            for _ in range(args.warmup):
                candidate()
            torch.musa.synchronize()
            candidate()
            torch.musa.synchronize()
            torch.testing.assert_close(values, expected, rtol=0, atol=0)
            torch.testing.assert_close(
                torch.gather(input, 1, output), expected, rtol=0, atol=0
            )
            samples = [
                _sample(candidate, args.iters, flush)
                for _ in range(args.rounds)
            ]
            result = {
                "batch": args.batch,
                "width": width,
                "topk": topk,
                "parts": parts,
                "milliseconds": statistics.median(samples),
                "samples_ms": samples,
            }
            results.append(result)
            print(json.dumps(result, sort_keys=True), flush=True)
    payload = {
        "status": "passed",
        "device": torch.musa.get_device_name(),
        "torch": torch.__version__,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
