#!/usr/bin/env python3
"""Single entry point for DeepSelect MUSA benchmarks and profiler evidence.

Examples:
  python benchmarks/musa_bench.py --suite short
  python benchmarks/musa_bench.py --suite all --output-dir bench-results
  python benchmarks/musa_bench.py --suite profile --batch 1 --width 1024
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def _run(script: str, args: list[str]) -> None:
    command = [sys.executable, str(ROOT / script), *args]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def _output(args: argparse.Namespace, name: str) -> str:
    if args.output is not None and args.suite != "all":
        return str(args.output)
    return str(args.output_dir / f"{name}.json")


def _common(args: argparse.Namespace) -> list[str]:
    return [
        "--warmup", str(args.warmup),
        "--iters", str(args.iters),
    ]


def run_suite(args: argparse.Namespace, suite: str) -> None:
    common = _common(args)
    if suite == "short":
        _run("musa_short_bench.py", [*common, "--output", _output(args, suite)])
    elif suite == "capability":
        _run(
            "musa_capability_bench.py",
            [*common, "--rounds", str(args.rounds), "--flush-mb", str(args.flush_mb),
             "--output", _output(args, suite)],
        )
    elif suite == "library":
        _run(
            "musa_library_bench.py",
            [*common, "--rounds", str(args.rounds), "--output", _output(args, suite)],
        )
    elif suite == "sampling":
        sampling_args = [*common, "--rounds", str(args.rounds),
                         "--flush-mb", str(args.flush_mb),
                         "--output", _output(args, suite)]
        if args.batches:
            sampling_args += ["--batches", args.batches]
        if args.widths:
            sampling_args += ["--widths", args.widths]
        _run(
            "musa_sampling_bench.py",
            sampling_args,
        )
    elif suite == "fp32_parts":
        _run(
            "musa_fp32_parts_bench.py",
            [
                "--warmup", str(args.warmup),
                "--iters", str(args.iters),
                "--rounds", str(args.rounds),
                "--output", _output(args, suite),
            ],
        )
    elif suite == "mcu":
        _run(
            "musa_cpp_mcu.py",
            [
                "--batch", str(args.batch),
                "--width", str(args.width),
                "--warmup", str(args.warmup),
                "--iters", str(args.iters),
                "--output", _output(args, suite),
            ],
        )
    elif suite == "profile":
        _run(
            "musa_cpp_profile.py",
            [
                "--batch", str(args.batch),
                "--width", str(args.width),
                "--iters", str(args.iters),
                "--output", _output(args, suite),
            ],
        )
    else:
        raise ValueError(f"unknown suite: {suite}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        choices=("short", "capability", "library", "sampling", "fp32_parts", "mcu", "profile", "all"),
        default="short",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("bench-results"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--flush-mb", type=int, default=0)
    parser.add_argument("--batches")
    parser.add_argument("--widths")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    suites = (
        ("short", "capability", "library", "sampling", "fp32_parts")
        if args.suite == "all" else (args.suite,)
    )
    for suite in suites:
        run_suite(args, suite)


if __name__ == "__main__":
    main()
