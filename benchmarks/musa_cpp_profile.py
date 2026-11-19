#!/usr/bin/env python3
"""Collect Kineto and mcc -S evidence for the native MUSA TopK extension."""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
from pathlib import Path

import torch
import torch_musa

from deep_select.musa_cpp import _module, launch_topk_bf16


def _build_ninja() -> Path:
    candidates = sorted(
        (Path.home() / ".cache" / "torch_extensions").glob(
            "**/deep_select_musa_topk_*/build.ninja"
        ),
        key=lambda path: path.stat().st_mtime_ns,
    )
    if not candidates:
        raise RuntimeError("could not find the torch_musa JIT build.ninja")
    return candidates[-1]


def _compile_assembly(output: Path) -> tuple[list[str], list[str]]:
    ninja = _build_ninja()
    variables = {}
    for line in ninja.read_text().splitlines():
        if " = " in line:
            name, value = line.split(" = ", 1)
            variables[name.strip()] = value.strip()
    mcc = variables.get("mcc")
    flags = variables.get("musa_cflags")
    if not mcc or flags is None:
        raise RuntimeError(f"missing mcc/musa_cflags in {ninja}")
    source = (
        Path(__file__).resolve().parents[1]
        / "csrc"
        / "musa_kernels"
        / "topk.mu"
    )
    command = [mcc, *shlex.split(flags), "-S", str(source), "-o", str(output)]
    command_path = output.with_suffix(".mu.cmd")
    command_path.write_text(shlex.join(command) + "\n")
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    log_path = output.with_suffix(".mcc.log")
    log_path.write_text(completed.stdout + completed.stderr)
    if completed.returncode != 0:
        raise RuntimeError(f"mcc -S failed; see {log_path}")
    ninja_copy = output.parent / "build.ninja"
    shutil.copy2(ninja, ninja_copy)
    metadata = []
    for line in output.read_text(errors="replace").splitlines():
        if any(
            marker in line
            for marker in (
                ".kernel_name",
                ".cc_local_memory_size",
                ".private_memory_size",
                ".shared_memory_size",
                ".shared_reg_count",
                ".temp_reg_count",
                ".has_barrier",
            )
        ):
            metadata.append(line.strip())
    return metadata, [str(command_path), str(log_path), str(ninja_copy)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=6)
    parser.add_argument("--width", type=int, default=1048576)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    _module()
    input = torch.randn(
        args.batch, args.width, dtype=torch.bfloat16, device="musa"
    )
    indices = torch.empty(args.batch, 512, dtype=torch.int32, device="musa")
    for _ in range(10):
        launch_topk_bf16(input, indices)
    torch.musa.synchronize()

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.MUSA]
    ) as profiler:
        for _ in range(args.iters):
            launch_topk_bf16(input, indices)
        torch.musa.synchronize()
    trace_path = args.output.parent / "kineto-trace.json"
    table_path = args.output.parent / "kineto-table.txt"
    profiler.export_chrome_trace(str(trace_path))
    table_path.write_text(
        profiler.key_averages().table(
            sort_by="self_device_time_total", row_limit=30
        )
        + "\n"
    )

    assembly_path = args.output.parent / "topk_bf16.s"
    metadata, extra_artifacts = _compile_assembly(assembly_path)
    selected = torch.gather(input, 1, indices.to(torch.int64))
    reference = torch.topk(input, 512, dim=1).values
    correct = torch.equal(
        torch.sort(selected, dim=1, descending=True).values.cpu(),
        reference.cpu(),
    )
    if not correct:
        raise AssertionError("native MUSA TopK profile input failed correctness")

    artifacts = [str(trace_path), str(table_path), str(assembly_path), *extra_artifacts]
    args.output.write_text(
        json.dumps(
            {
                "status": "passed",
                "device": torch.musa.get_device_name(),
                "batch": args.batch,
                "width": args.width,
                "correct": correct,
                "assembly_metadata": metadata,
                "artifacts": artifacts,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
