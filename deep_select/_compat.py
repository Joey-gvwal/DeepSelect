"""Runtime compatibility checks for prebuilt `deepselect-musa` wheels.

The wheel is built against one exact MUSA/torch stack (encoded in the PEP 440
local version, e.g. ``0.0.1+musa5.2.0torch2.11.0.post1``).  Importing the MUSA
backend through a prebuilt wheel validates that stack and raises an actionable
error instead of letting the process crash inside a kernel launch.

A source checkout has no ``_build_info.py`` and therefore skips the checks.
"""
from __future__ import annotations

import json
import os
import platform
import sys


def build_info() -> dict:
    """Return the build metadata baked into a prebuilt wheel (or {})."""
    try:
        from . import _build_info as info  # type: ignore[import-not-found]
    except ImportError:
        return {}
    return {
        key: getattr(info, key)
        for key in (
            "PUBLIC_VERSION",
            "LOCAL_LABEL",
            "FULL_VERSION",
            "BUILD_PYTHON",
            "BUILD_TORCH",
            "BUILD_TORCH_PUBLIC",
            "BUILD_TORCH_MUSA",
            "BUILD_MUSA_HOME",
            "BUILD_MUSA_SDK",
            "BUILD_ARCH",
            "BUILD_GIT_HEAD",
            "BUILD_MUTLASS_COMMIT",
            "BUILD_TIME_UTC",
        )
        if hasattr(info, key)
    }


def _version_problems(info: dict) -> list[str]:
    problems: list[str] = []
    try:
        import torch
        import torch_musa
    except ImportError as exc:  # pragma: no cover - MUSA backend already imported
        return [f"cannot import torch/torch_musa: {exc}"]

    expected = info.get("BUILD_TORCH_MUSA", "")
    if expected and torch_musa.__version__ != expected:
        problems.append(
            f"torch_musa {torch_musa.__version__} != build {expected}"
        )

    expected_torch = info.get("BUILD_TORCH_PUBLIC", "")
    try:
        from packaging.version import Version

        installed_public = Version(torch.__version__).public
    except Exception:
        installed_public = torch.__version__.split("+", 1)[0]
    if expected_torch and installed_public != expected_torch:
        problems.append(f"torch {installed_public} != build {expected_torch}")

    musa_home = info.get("BUILD_MUSA_HOME", "")
    if musa_home and not os.path.isdir(musa_home):
        problems.append(f"MUSA_HOME {musa_home!r} is missing")
    return problems


def _reinstall_hint(info: dict) -> str:
    public = info.get("PUBLIC_VERSION", "<version>")
    local = info.get("LOCAL_LABEL", "")
    variant = info.get("INDEX_VARIANT") or ""
    if not variant and "torch" in local:
        import re

        match = re.match(r"musa(\d+)\.(\d+)\.[0-9]+torch(\d+)\.(\d+)", local)
        if match:
            variant = f"musa{match.group(1)}{match.group(2)}-torch{match.group(3)}{match.group(4)}"
    if not local:
        return (
            "Install the wheel published for this stack:\n"
            f"  pip install --upgrade 'deepselect-musa=={public}'\n"
            "If the default PyPI wheel does not match, install the build for your "
            "torch/MUSA stack from the project's release assets."
        )
    index = f"https://moorethreads.github.io/deepselect-whl/{variant}/" if variant else "..."
    return (
        "Install the wheel built for this stack:\n"
        f"  pip install --index-url {index} 'deepselect-musa=={public}+{local}'"
    )


def check(*, raise_on_error: bool = True) -> dict:
    """Validate the runtime stack against the prebuilt wheel metadata.

    Returns a JSON-serializable report. Raises ``RuntimeError`` on mismatch when
    ``raise_on_error`` is true (the default used by the backend loader).
    """
    info = build_info()
    report = {
        "distribution": "deepselect-musa",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "build": info,
        "problems": [],
    }
    if not info:
        report["mode"] = "source"
        return report
    report["mode"] = "wheel"
    report["problems"] = _version_problems(info)
    try:
        import torch

        report["torch"] = torch.__version__
        report["musa_available"] = bool(torch.musa.is_available())
        if report["musa_available"]:
            report["device"] = torch.musa.get_device_name(0)
            try:
                report["device_capability"] = list(torch.musa.get_device_capability(0))
            except Exception:
                pass
    except Exception as exc:  # pragma: no cover
        report["problems"].append(f"cannot inspect MUSA device: {exc}")

    if report["problems"]:
        report["hint"] = _reinstall_hint(info)
        if raise_on_error:
            raise RuntimeError(
                "deepselect-musa runtime does not match the wheel build:\n  "
                + "\n  ".join(report["problems"])
                + "\n"
                + report["hint"]
            )
    return report


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="DeepSelect MUSA environment check")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args()
    try:
        report = check(raise_on_error=False)
    except Exception as exc:  # pragma: no cover
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print("deepselect-musa environment")
        print(f"  python     : {report['python']}")
        print(f"  mode       : {report['mode']}")
        if report["build"]:
            print(f"  wheel      : {report['build'].get('FULL_VERSION')}")
            print(f"  build torch: {report['build'].get('BUILD_TORCH')}")
            print(f"  build musa : {report['build'].get('BUILD_MUSA_SDK')}")
        print(f"  torch       : {report.get('torch', '?')}")
        print(f"  device      : {report.get('device', '?')}")
        if report["problems"]:
            print("  problems    :")
            for item in report["problems"]:
                print(f"    - {item}")
            print(report.get("hint", ""))
        else:
            print("  status      : OK")
    return 1 if report["problems"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
