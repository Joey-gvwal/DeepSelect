"""Build, install, and validate the MUSA wheel in an isolated source copy."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile


def _run(argv, *, cwd: Path, env: dict[str, str]) -> subprocess.CompletedProcess:
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
    return completed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    checkout = Path(__file__).resolve().parents[1]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    log_path = args.output.parent / "musa-wheel-validation.log"
    logs = []

    with tempfile.TemporaryDirectory(prefix="deepselect-musa-wheel-") as tmp_text:
        tmp = Path(tmp_text)
        source = tmp / "source"
        wheel_dir = tmp / "wheel"
        shutil.copytree(
            checkout,
            source,
            ignore=shutil.ignore_patterns(".git", ".pipeline", "__pycache__", "*.pyc"),
        )
        wheel_dir.mkdir()
        env = dict(os.environ)
        env["DEEP_SELECT_BUILD_TARGET_PLATFORM"] = "MUSA"
        build = _run(
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
        logs.append(build.stdout)
        wheels = list(wheel_dir.glob("deep_select-*.whl"))
        if len(wheels) != 1:
            raise RuntimeError(f"expected one wheel, found {wheels}")
        wheel = wheels[0]
        with zipfile.ZipFile(wheel) as archive:
            members = set(archive.namelist())
        required = {
            "deep_select/__init__.py",
            "deep_select/interface.py",
            "deep_select/deep_select_musa.py",
            "deep_select/musa_cpp.py",
        }
        missing = required - members
        if missing:
            raise RuntimeError(f"wheel is missing MUSA files: {sorted(missing)}")
        native_members = sorted(
            member
            for member in members
            if member.startswith("deep_select/deep_select_musa_cpp")
            and member.endswith(".so")
        )
        if len(native_members) != 1:
            raise RuntimeError(
                f"expected one AOT MUSA extension in wheel, found {native_members}"
            )
        legacy_source_members = sorted(
            member
            for member in members
            if member.startswith("deep_select/csrc/")
        )
        if legacy_source_members:
            raise RuntimeError(
                "MUSA sources must stay in top-level csrc/musa_kernels; "
                f"wheel contains legacy package paths: {legacy_source_members}"
            )

        install = _run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--force-reinstall",
                "--no-deps",
                str(wheel),
            ],
            cwd=tmp,
            env=env,
        )
        logs.append(install.stdout)

        test_env = dict(env)
        test_env.pop("PYTHONPATH", None)
        probe = _run(
            [
                sys.executable,
                "-c",
                "from deep_select.musa_cpp import _module; print(_module().__file__)",
            ],
            cwd=tmp,
            env=test_env,
        )
        logs.append(probe.stdout)
        native_extension = probe.stdout.strip().splitlines()[-1]
        if (
            not native_extension.endswith(".so")
            or "torch_extensions" in native_extension
        ):
            raise RuntimeError(
                f"wheel did not load its AOT MUSA extension: {native_extension}"
            )
        test = _run(
            [sys.executable, str(checkout / "tests" / "test_musa_backend.py")],
            cwd=tmp,
            env=test_env,
        )
        logs.append(test.stdout)
        tme_test = _run(
            [
                sys.executable,
                str(checkout / "tests" / "test_musa_tme_survivor.py"),
            ],
            cwd=tmp,
            env=test_env,
        )
        logs.append(tme_test.stdout)

        copied_wheel = args.output.parent / wheel.name
        shutil.copy2(wheel, copied_wheel)

    log_path.write_text("\n\n".join(logs))
    payload = {
        "status": "passed",
        "device": "musa",
        "python": sys.executable,
        "wheel": copied_wheel.as_posix(),
        "wheel_size": copied_wheel.stat().st_size,
        "required_members": sorted(required),
        "native_members": native_members,
        "legacy_source_members": legacy_source_members,
        "native_extension": native_extension,
        "tests": [
            "tests/test_musa_backend.py",
            "tests/test_musa_tme_survivor.py",
        ],
        "artifacts": [copied_wheel.as_posix(), log_path.as_posix()],
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
