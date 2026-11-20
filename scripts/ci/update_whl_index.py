#!/usr/bin/env python3
"""维护给 pip 用的静态 PEP 503 simple index（DeepSelect-MUSA，deep_ep 风格 tag）。

布局（`MooreThreads/deepselect-whl` 仓库 gh-pages 分支，按 (MUSA, torch) variant 分目录）：
    musa52-torch211/deepselect-musa/index.html    # pip install --index-url .../musa52-torch211/
    musa52-torch213/deepselect-musa/index.html
    index.html                                    # 根索引（人工浏览/排障）

每个 wheel 以绝对 URL + #sha256= 片段写入；资产托管在
`https://github.com/MooreThreads/deepselect-whl/releases/download/v<public>/<wheel>`。

重要：同一 variant 目录只应出现一条 torch 线（PEP 440 在只给公共版本时会选择最高
local version，混放可能装到 ABI 不匹配的 wheel，见 WHEEL-TAG-SPEC.md §5）。

用法：
    git clone --branch gh-pages https://github.com/MooreThreads/deepselect-whl.git whl
    python3 scripts/ci/update_whl_index.py --dist dist --repo whl
    cd whl && git add -A && git commit -m "index: deepselect-musa 1.1.0+musa5.2.0torch2.11.0.post1" && git push
"""
from __future__ import annotations

import argparse
import hashlib
import pathlib
import re

# deepselect_musa-1.1.0+musa5.2.0torch2.11.0.post1-cp310-cp310-linux_x86_64.whl
WHEEL_RE = re.compile(
    r"^deepselect_musa-(?P<public>[0-9]+(?:\.[0-9]+)*(?:(?:a|b|rc)[0-9]+)?"
    r"(?:\.post[0-9]+)?(?:\.dev[0-9]+)?)"
    r"\+(?P<musa>musa(?P<musa_ver>[0-9]+\.[0-9]+\.[0-9]+))"
    r"(?P<torch>torch(?P<torch_ver>[0-9]+\.[0-9]+\.[0-9]+(?:\.post[0-9]+)?))"
    r"(?:\.b(?P<rebuild>[0-9]+))?"
    r"-(?P<py>cp[0-9]+)-(?P<abi>cp[0-9]+|abi3)-(?P<plat>.+)\.whl$"
)


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def variant_of(match: re.Match[str]) -> str:
    """musa5.2.0 + torch2.11.0.post1 -> musa52-torch211"""
    musa_mm = "".join(match.group("musa_ver").split(".")[:2])
    torch_mm = "".join(match.group("torch_ver").split(".")[:2])
    return f"musa{musa_mm}-torch{torch_mm}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dist", type=pathlib.Path, default=pathlib.Path("dist"))
    ap.add_argument("--repo", type=pathlib.Path, default=pathlib.Path("deepselect-whl"))
    ap.add_argument("--project", default="deepselect-musa")
    ap.add_argument(
        "--release-base-url",
        default="https://github.com/MooreThreads/deepselect-whl/releases/download",
    )
    ap.add_argument("--only-variant", default="", help="只处理某个 variant，如 musa52-torch211")
    args = ap.parse_args()

    wheels = sorted(p for p in args.dist.rglob("*.whl") if p.is_file())
    if not wheels:
        raise SystemExit(f"no wheels under {args.dist}")

    indexed = 0
    skipped = 0
    for whl in wheels:
        match = WHEEL_RE.match(whl.name)
        if not match:
            print(f"skip (filename != spec): {whl.name}")
            skipped += 1
            continue
        variant = variant_of(match)
        if args.only_variant and variant != args.only_variant:
            continue
        public = match.group("public")
        index_dir = args.repo / variant / args.project
        index_dir.mkdir(parents=True, exist_ok=True)
        index_html = index_dir / "index.html"
        existing = index_html.read_text() if index_html.is_file() else ""
        if whl.name in existing:
            continue
        url = f"{args.release_base_url}/v{public}/{whl.name}#sha256={sha256(whl)}"
        with index_html.open("a") as f:
            f.write(f'<a href="{url}">{whl.name}</a><br>\n')
        print(f"index += {variant}/{whl.name}")
        indexed += 1

    root = args.repo / "index.html"
    if not root.is_file():
        variants = sorted(p.name for p in args.repo.iterdir()
                          if p.is_dir() and re.match(r"^musa[0-9]+-torch[0-9]+$", p.name))
        links = "\n".join(
            f'<a href="{v}/{args.project}/">{v}</a><br>' for v in variants
        )
        root.write_text(
            "<!DOCTYPE html><html><body><h1>DeepSelect MUSA wheels</h1>"
            f"{links}\n</body></html>\n"
        )
        print("root index created")

    print(f"done: {indexed} wheel(s) indexed, {skipped} skipped")


if __name__ == "__main__":
    main()
