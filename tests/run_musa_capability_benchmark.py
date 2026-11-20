"""Build/install the MUSA wheel, then run a selected MUSA benchmark."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def _run(argv: list[str], *, cwd: Path, env: dict[str, str]) -> str:
    completed = subprocess.run(
        argv,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(argv)}\n"
            + completed.stdout
        )
    return completed.stdout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument(
        "--benchmark",
        choices=("short", "wide", "all"),
        default="short",
    )
    args = parser.parse_args()
    checkout = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    log = output.parent / f"{args.benchmark}-benchmark.log"
    env = dict(os.environ)
    env["DEEP_SELECT_BUILD_TARGET_PLATFORM"] = "MUSA"
    with tempfile.TemporaryDirectory(prefix="deepselect-capability-") as tmp_text:
        tmp = Path(tmp_text)
        source = tmp / "source"
        wheel_dir = tmp / "wheel"
        shutil.copytree(
            checkout,
            source,
            ignore=shutil.ignore_patterns(
                ".git", ".pipeline", "__pycache__", "*.pyc"
            ),
        )
        wheel_dir.mkdir()
        build_log = _run(
            [
                sys.executable,
                "-m",
                "pip",
                "wheel",
                "--no-build-isolation",
                "--no-deps",
                "--wheel-dir",
                str(wheel_dir),
                ".",
            ],
            cwd=source,
            env=env,
        )
        wheels = list(wheel_dir.glob("deep_select-*.whl"))
        if len(wheels) != 1:
            raise RuntimeError(f"expected one wheel, found {wheels}")
        install_log = _run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--force-reinstall",
                "--no-deps",
                str(wheels[0]),
            ],
            cwd=tmp,
            env=env,
        )
        bench_env = dict(env)
        bench_env.pop("PYTHONPATH", None)
        benchmark_script = "musa_bench.py"
        benchmark_argv = [
            sys.executable,
            str(source / "benchmarks" / benchmark_script),
            "--suite",
            args.benchmark,
            "--output",
            str(output),
            "--warmup",
            str(args.warmup),
            "--iters",
            str(args.iters),
        ]
        benchmark_log = _run(
            benchmark_argv,
            cwd=tmp,
            env=bench_env,
        )
    log.write_text("\n\n".join((build_log, install_log, benchmark_log)))


if __name__ == "__main__":
    main()
