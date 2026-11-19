"""Deterministic workload for MCU profiling of native MUSA TopK kernels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch_musa  # noqa: F401

from deep_select.musa_cpp import launch_topk_bf16


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=6)
    parser.add_argument("--width", type=int, default=1048576)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-artifact", action="append", default=[])
    args = parser.parse_args()

    generator = torch.Generator(device="musa")
    generator.manual_seed(20261028 + args.batch * 17 + args.width)
    input = torch.randn(
        args.batch,
        args.width,
        dtype=torch.bfloat16,
        device="musa",
        generator=generator,
    )
    indices = torch.empty(args.batch, 512, dtype=torch.int32, device="musa")

    for _ in range(args.warmup):
        launch_topk_bf16(input, indices)
    torch.musa.synchronize()
    for _ in range(args.iters):
        launch_topk_bf16(input, indices)
    torch.musa.synchronize()

    selected = torch.gather(input, 1, indices.to(torch.int64))
    reference = torch.topk(input, 512, dim=1).values
    correct = torch.equal(
        torch.sort(selected, dim=1, descending=True).values.cpu(),
        reference.cpu(),
    )
    if not correct:
        raise AssertionError("native MUSA TopK MCU input failed correctness")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "status": "passed",
                "device": torch.musa.get_device_name(),
                "capability": list(torch.musa.get_device_capability()),
                "torch": torch.__version__,
                "batch": args.batch,
                "width": args.width,
                "topk": 512,
                "correct": correct,
                "artifacts": args.expected_artifact,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
