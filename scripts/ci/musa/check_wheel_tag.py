#!/usr/bin/env python3
"""校验 DeepSelect-MUSA wheel 的文件名 / METADATA / WHEEL / _build_info 四处一致。

用法：
    python3 check_wheel_tag.py dist/deepselect_musa-*.whl \
        --expected-local musa5.2.0torch2.11.0.post1

校验项：
1. 文件名符合 WHEEL-TAG-SPEC 公式（dist=deepselect_musa，local=musa…torch…[.bN]）
2. `packaging.utils.parse_wheel_filename` 可解析，且与期望 local 严格一致
3. wheel 内 METADATA `Name: deepselect-musa`、`Version:` == 文件名版本
4. WHEEL `Tag:` 集合 == 文件名推导出的 tag
5. wheel 内 `deep_select/_build_info.py`（P1 起存在）字段自洽：
   PUBLIC_VERSION / LOCAL_LABEL / BUILD_TORCH_PUBLIC / BUILD_MUSA_SDK
失败 exit 1，成功打印 JSON 报告。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from pathlib import Path

from packaging.utils import parse_wheel_filename
from packaging.version import Version

LOCAL_RE = re.compile(
    r"^musa(?P<musa>\d+\.\d+\.\d+)torch(?P<torch>\d+\.\d+\.\d+(?:\.post\d+)?)"
    r"(?:\.b(?P<rebuild>\d+))?$"
)
FILENAME_RE = re.compile(
    r"^deepselect_musa-(?P<version>[^-]+)-(?P<py>cp[0-9]+)-(?P<abi>cp[0-9]+|abi3)-"
    r"(?P<plat>[^-]+)\.whl$"
)


def _read_build_info(zf: zipfile.ZipFile) -> dict[str, str] | None:
    names = [n for n in zf.namelist() if n.endswith("deep_select/_build_info.py")]
    if not names:
        return None
    text = zf.read(names[0]).decode("utf-8", "replace")
    values: dict[str, str] = {}
    for key in ("PUBLIC_VERSION", "LOCAL_LABEL", "BUILD_TORCH_PUBLIC", "BUILD_TORCH_MUSA",
                "BUILD_MUSA_SDK", "BUILD_ARCH", "BUILD_GIT_HEAD", "BUILD_MUTLASS_COMMIT"):
        match = re.search(rf"^{key}\s*=\s*['\"]([^'\"]*)['\"]", text, re.M)
        if match:
            values[key] = match.group(1)
    return values


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("wheel", type=Path)
    ap.add_argument("--expected-local", default="", help="例如 musa5.2.0torch2.11.0.post1")
    ap.add_argument("--require-build-info", action="store_true",
                    help="P1 起 wheel 必须带 _build_info.py，缺失即失败")
    args = ap.parse_args()

    problems: list[str] = []
    report: dict = {"wheel": args.wheel.name}

    m = FILENAME_RE.match(args.wheel.name)
    if not m:
        problems.append(f"filename does not match spec: {args.wheel.name}")
        report["problems"] = problems
        print(json.dumps(report, indent=2))
        return 1
    version_str = m.group("version")
    report.update(
        version=version_str,
        python_tag=m.group("py"),
        abi_tag=m.group("abi"),
        platform_tag=m.group("plat"),
    )

    try:
        parsed = Version(version_str)
    except Exception as exc:
        problems.append(f"invalid PEP 440 version {version_str!r}: {exc}")
        parsed = None
    if parsed is not None:
        public = parsed.public
        local = str(parsed.local or "")
        report.update(public_version=public, local_label=local)
        if local and not LOCAL_RE.match(local):
            problems.append(f"local label does not match spec: {local!r}")
        if args.expected_local and local != args.expected_local:
            problems.append(f"local label {local!r} != expected {args.expected_local!r}")
        if not local and args.expected_local:
            problems.append(f"wheel has no local label but expected {args.expected_local!r}")
        if re.search(r"dev|git|unknown|20[0-9]{6}", public):
            problems.append(f"public version looks non-release: {public!r}")

    # 3/4: metadata + WHEEL tags
    try:
        with zipfile.ZipFile(args.wheel) as zf:
            metadata_name = next((n for n in zf.namelist() if n.endswith(".dist-info/METADATA")), None)
            wheel_name = next((n for n in zf.namelist() if n.endswith(".dist-info/WHEEL")), None)
            if not metadata_name or not wheel_name:
                problems.append("wheel is missing METADATA/WHEEL")
            else:
                meta = zf.read(metadata_name).decode("utf-8", "replace")
                got_name = re.search(r"^Name:\s*(\S+)", meta, re.M)
                got_ver = re.search(r"^Version:\s*(\S+)", meta, re.M)
                if not got_name or got_name.group(1).lower().replace("_", "-") != "deepselect-musa":
                    problems.append(f"METADATA Name mismatch: {got_name.group(1) if got_name else None!r}")
                if not got_ver or got_ver.group(1) != version_str:
                    problems.append(
                        f"METADATA Version {got_ver.group(1) if got_ver else None!r} != filename {version_str!r}"
                    )
                wheel_meta = zf.read(wheel_name).decode("utf-8", "replace")
                tags = {line.split(":", 1)[1].strip() for line in wheel_meta.splitlines()
                        if line.startswith("Tag:")}
                expected_tag = f"{m.group('py')}-{m.group('abi')}-{m.group('plat')}"
                if tags != {expected_tag}:
                    problems.append(f"WHEEL Tag {sorted(tags)} != expected {expected_tag!r}")
                info = _read_build_info(zf)
                report["build_info"] = info
                if info is None:
                    if args.require_build_info:
                        problems.append("missing deep_select/_build_info.py")
                elif parsed is not None:
                    local = str(parsed.local or "")
                    if info.get("LOCAL_LABEL") and info["LOCAL_LABEL"] != local:
                        problems.append(f"_build_info.LOCAL_LABEL {info['LOCAL_LABEL']!r} != {local!r}")
                    if info.get("PUBLIC_VERSION") and info["PUBLIC_VERSION"] != parsed.public:
                        problems.append(
                            f"_build_info.PUBLIC_VERSION {info['PUBLIC_VERSION']!r} != {parsed.public!r}"
                        )
                    if local:
                        if info.get("BUILD_TORCH_PUBLIC") and f"torch{info['BUILD_TORCH_PUBLIC']}" not in local:
                            problems.append(
                                f"_build_info.BUILD_TORCH_PUBLIC {info['BUILD_TORCH_PUBLIC']!r} not in local {local!r}"
                            )
                        if info.get("BUILD_MUSA_SDK") and f"musa{info['BUILD_MUSA_SDK']}" not in local:
                            problems.append(
                                f"_build_info.BUILD_MUSA_SDK {info['BUILD_MUSA_SDK']!r} not in local {local!r}"
                            )
    except zipfile.BadZipFile:
        problems.append("not a zip/wheel file")
    except StopIteration:
        pass

    # 2: packaging 解析（文件名 -> name/version/tags）
    try:
        pname, pver, _build, ptags = parse_wheel_filename(args.wheel.name)
        report["packaging"] = {"name": str(pname), "version": str(pver),
                               "tags": sorted(str(t) for t in ptags)}
        if str(pname) != "deepselect-musa":
            problems.append(f"parsed distribution name {str(pname)!r} != 'deepselect-musa'")
    except Exception as exc:
        problems.append(f"packaging.parse_wheel_filename failed: {exc}")

    report["problems"] = problems
    report["ok"] = not problems
    print(json.dumps(report, indent=2))
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
