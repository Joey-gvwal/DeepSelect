#!/usr/bin/env python3
"""按 deep_ep 约定推导 DeepSelect-MUSA 的 wheel 版本/tag 字段。

公式：<public>+musa<MUSA_SDK>torch<torch_public>[-<pytag>-<abitag>-<plat>]

用法（构建容器内，已装 torch/torch_musa）：
    eval "$(python3 compose_wheel_tag.py --public-version 1.1.0 --export)"
    python3 compose_wheel_tag.py --public-version 1.1.0 --json
    python3 compose_wheel_tag.py --public-version 1.1.0 --print-filename

离线/测试可用显式参数覆盖推导：
    python3 compose_wheel_tag.py --public-version 1.1.0 \
        --torch-version 2.11.0.post1+musa5.2.0 --python-tag cp310
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import sysconfig
from pathlib import Path

try:
    from packaging.version import InvalidVersion, Version
except ImportError:  # pragma: no cover
    raise SystemExit("requires `packaging` (pip install packaging)")

DIST_NAME = "deepselect-musa"
LOCAL_RE = re.compile(
    r"^musa(?P<musa>\d+\.\d+\.\d+)torch(?P<torch>\d+\.\d+\.\d+(?:\.post\d+)?)"
    r"(?:\.b(?P<rebuild>\d+))?$"
)


def _torch_versions(args) -> tuple[str, str]:
    torch_version = args.torch_version
    torch_musa_version = args.torch_musa_version
    if not torch_version or not torch_musa_version:
        try:
            import torch  # type: ignore
            import torch_musa  # type: ignore

            torch_version = torch_version or torch.__version__
            torch_musa_version = torch_musa_version or torch_musa.__version__
        except Exception as exc:  # pragma: no cover
            raise SystemExit(
                f"cannot import torch/torch_musa ({exc}); pass --torch-version and "
                "--torch-musa-version explicitly"
            )
    return torch_version, torch_musa_version


def _musa_sdk(torch_musa_version: str) -> str:
    local = Version(torch_musa_version).local
    if not local:
        raise SystemExit(
            f"torch_musa version {torch_musa_version!r} has no local MUSA label, "
            "expected e.g. 2.11.0.post1+musa5.2.0"
        )
    match = re.match(r"musa(\d+\.\d+\.\d+)", local)
    if not match:
        raise SystemExit(f"unrecognized MUSA local label in {torch_musa_version!r}: {local}")
    return match.group(1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--public-version", required=True, help="DeepSelect 公共版本，如 1.1.0")
    ap.add_argument("--torch-version", default=None, help="默认取已安装 torch.__version__")
    ap.add_argument("--torch-musa-version", default=None, help="默认取已安装 torch_musa.__version__")
    ap.add_argument("--local-label", default="", help="显式 local label，如 musa5.2.0torch2.11.0.post1")
    ap.add_argument("--rebuild", type=int, default=0, help="同 SDK/torch 的运维重建计数（>0 时追加 .bN）")
    ap.add_argument("--python-tag", default=None, help="默认由当前解释器推导，如 cp310")
    ap.add_argument("--platform-tag", default=None, help="默认 sysconfig.get_platform()，应固定 linux_x86_64")
    ap.add_argument("--export", action="store_true", help="输出 shell export 行")
    ap.add_argument("--json", action="store_true", help="输出 JSON")
    ap.add_argument("--print-filename", action="store_true", help="只输出 wheel 文件名")
    args = ap.parse_args()

    public = Version(args.public_version).public
    if args.local_label:
        match = LOCAL_RE.match(args.local_label)
        if not match:
            raise SystemExit(f"local label does not match spec: {args.local_label!r}")
        local = args.local_label
        musa_sdk, torch_public = match.group("musa"), match.group("torch")
        torch_version = f"{torch_public}+musa{musa_sdk}"
        torch_musa_version = f"{torch_public}+musa{musa_sdk}"
    else:
        torch_version, torch_musa_version = _torch_versions(args)
        torch_public = Version(torch_version).public  # 去掉 +musa5.2.0
        musa_sdk = _musa_sdk(torch_musa_version)
        local = f"musa{musa_sdk}torch{torch_public}"
        if args.rebuild > 0:
            local += f".b{args.rebuild}"

    py_tag = args.python_tag or f"cp{sys.version_info.major}{sys.version_info.minor}"
    abi_tag = py_tag  # 普通 C++ 扩展；stable ABI 构建时改成 abi3
    plat_tag = args.platform_tag or (
        sysconfig.get_platform().replace("-", "_").replace(".", "_")
    )

    if not LOCAL_RE.match(local):
        raise SystemExit(f"local label does not match spec: {local!r}")
    version = f"{public}+{local}"
    filename = f"{DIST_NAME.replace('-', '_')}-{version}-{py_tag}-{abi_tag}-{plat_tag}.whl"

    payload = {
        "public_version": public,
        "local_label": local,
        "version": version,
        "dist_name": DIST_NAME,
        "module_name": "deep_select",
        "musa_sdk": musa_sdk,
        "torch_public": torch_public,
        "torch_musa_version": Version(torch_musa_version).public + (
            f"+{Version(torch_musa_version).local}" if Version(torch_musa_version).local else ""
        ),
        "python_tag": py_tag,
        "abi_tag": abi_tag,
        "platform_tag": plat_tag,
        "wheel_filename": filename,
    }
    # index variant: musa<major><minor>-torch<major><minor>，如 musa52-torch211
    musa_mm = "".join(musa_sdk.split(".")[:2])
    torch_mm = "".join(torch_public.split(".")[:2])
    payload["index_variant"] = f"musa{musa_mm}-torch{torch_mm}"

    if args.print_filename:
        print(filename)
        return 0
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0
    if args.export:
        print(f"export DEEP_SELECT_PUBLIC_VERSION={public}")
        print(f"export DEEP_SELECT_LOCAL_LABEL={local}")
        print(f"export DEEP_SELECT_VERSION={version}")
        print(f"export DEEP_SELECT_DIST_NAME={DIST_NAME}")
        print(f"export DEEP_SELECT_PY_TAG={py_tag}")
        print(f"export DEEP_SELECT_ABI_TAG={abi_tag}")
        print(f"export DEEP_SELECT_PLATFORM_TAG={plat_tag}")
        print(f"export DEEP_SELECT_INDEX_VARIANT={payload['index_variant']}")
        print(f"export DEEP_SELECT_WHEEL_FILENAME={filename}")
        return 0
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
