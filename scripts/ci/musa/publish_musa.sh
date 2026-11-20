#!/usr/bin/env bash
# torchada 风格的“一条命令发布”兜底脚本（设计草案，基于 mccx-204 spike 实测配方）。
#
# 用途：没有 CI / 需要在 mccx-204 上手工发版时，与 release-musa.yml 做同样的事：
#   容器内构建 wheel -> 真机 GPU 门禁 -> GitHub Release -> 静态索引 -> 内网共享目录。
#   PyPI sdist 暂不发布（当前 sdist 不可构建，修复后见 DESIGN.md §8.2）。
#
# 前置：
#   - 在 mccx-204 上运行（有 MUSA 5.2 `-stack` 镜像、S5000、docker、gh）；
#   - 本地有干净的 DeepSelect 源码树（submodule 已 materialize：mutlass 必须有，
#     cutlass 对 MUSA 非必需；见 spike notes §2 的 codeload 兜底）；
#   - 已登录 gh（对 deepselect-whl 有写权限）。
#
# 用法：
#   ./scripts/ci/musa/publish_musa.sh 1.1.0              # 构建+测试+打印待发布清单
#   ./scripts/ci/musa/publish_musa.sh 1.1.0 --publish    # 真正发布（不可逆）
set -euo pipefail

VERSION="${1:?usage: $0 <public-version> [--publish]}"
shift || true
PUBLISH=false
[[ "${1:-}" == "--publish" ]] && PUBLISH=true

REPO_DIR="${REPO_DIR:-$(pwd)}"
WORK="${WORK:-/data/deepselect-release}"
OUT="$WORK/dist/$VERSION"
IMAGE="${IMAGE:-deepselect-build-musa:musa52-py310}"
LOCAL_LABEL="${LOCAL_LABEL:-musa5.2.0torch2.11.0.post1}"   # WHEEL-TAG-SPEC 公式
INDEX_VARIANT="${INDEX_VARIANT:-musa52-torch211}"
WHL_REPO="${WHL_REPO:-MooreThreads/deepselect-whl}"
INTERNAL_WHEEL_DIR="${INTERNAL_WHEEL_DIR:-/home/mccxadmin/action_runners/sgl-cache/whl-52/deepselect}"
DOCKER_GPU_ARGS="${DOCKER_GPU_ARGS:---runtime=mthreads --network=host --shm-size=32g}"

log() { printf '\n== %s ==\n' "$*"; }

[[ -d "$REPO_DIR/.git" ]] || { echo "not a git checkout: $REPO_DIR" >&2; exit 1; }
git -C "$REPO_DIR" diff --quiet || { echo "worktree is dirty; commit/stash first" >&2; exit 1; }
[[ -f "$REPO_DIR/csrc/3rdparty/mutlass/include/mutlass/version.h" ]] || {
  echo "mutlass headers missing; run: git submodule update --init csrc/3rdparty/mutlass" >&2; exit 1; }
rm -rf "$OUT"; mkdir -p "$OUT"

log "build wheel (image: $IMAGE)"
# 注意：spike 实测这个镜像/项目不能用 pip wheel 驱动构建（版本非确定），
# build_wheel.sh 内部已用 `python -m build --wheel --no-isolation`。
docker run --rm --network host \
  -e DEEP_SELECT_PUBLIC_VERSION="$VERSION" -e DEEP_SELECT_LOCAL_LABEL="$LOCAL_LABEL" \
  -v "$REPO_DIR:/src/DeepSelect:ro" -v "$OUT:/out" \
  "$IMAGE" --src /src/DeepSelect --out /out

log "GPU gate (smoke + project regression)"
WHEEL=$(ls "$OUT"/deep_select-*.whl "$OUT"/deepselect_musa-*.whl 2>/dev/null | head -1)
docker run --rm $DOCKER_GPU_ARGS -e MTHREADS_VISIBLE_DEVICES=0 \
  -v "$WHEEL:/tmp/deepselect.whl:ro" \
  -v "$REPO_DIR/scripts/ci/musa:/opt/ci:ro" \
  -v "$REPO_DIR/tests:/opt/deepselect-tests:ro" \
  "$IMAGE" bash -lc '
    set -euo pipefail
    python3 -m pip install --no-deps --force-reinstall --no-index /tmp/deepselect.whl
    python3 /opt/ci/smoke_test.py --expected-local-label "$LOCAL_LABEL" --expected-torch-musa 2.11.0.post1
    cd /opt/deepselect-tests && python3 test_musa_backend.py && python3 test_musa_tme_survivor.py
  '

log "checksums"
( cd "$OUT" && sha256sum *.whl > SHA256SUMS && cat SHA256SUMS )

log "copy to internal wheelhouse"
mkdir -p "$INTERNAL_WHEEL_DIR"
cp -f "$OUT"/*.whl "$INTERNAL_WHEEL_DIR/"
( cd "$INTERNAL_WHEEL_DIR" && sha256sum -c SHA256SUMS 2>/dev/null || true
  cat SHA256SUMS >> MANIFEST.txt; sort -u MANIFEST.txt -o MANIFEST.txt
  echo "internal install: pip install --no-index --no-deps $INTERNAL_WHEEL_DIR/$(basename "$WHEEL")" )

if ! $PUBLISH; then
  log "dry-run complete: $OUT"
  echo "re-run with --publish to push to GitHub Release + static index"
  exit 0
fi

log "GitHub release (wheel repo)"
gh release view "v$VERSION" --repo "$WHL_REPO" >/dev/null 2>&1 || \
  gh release create "v$VERSION" --repo "$WHL_REPO" --title "DeepSelect MUSA $VERSION" \
    --notes-file "$REPO_DIR/docs/release-notes-musa.md"
gh release upload "v$VERSION" --repo "$WHL_REPO" --clobber "$OUT"/*.whl "$OUT"/SHA256SUMS

log "update static index"
TMP_WHL=$(mktemp -d)
gh repo clone "$WHL_REPO" "$TMP_WHL/repo" -- --branch gh-pages --depth 1
python3 "$REPO_DIR/scripts/ci/update_whl_index.py" --dist "$OUT" --repo "$TMP_WHL/repo"
( cd "$TMP_WHL/repo" && git add -A && \
  git -c user.name="deepselect-bot" -c user.email="deepselect-bot@users.noreply.github.com" \
      commit -m "index: deepselect-musa $VERSION+$LOCAL_LABEL" && git push )
rm -rf "$TMP_WHL"

log "post-publish verify (from public index)"
python3 -m venv --system-site-packages --without-pip /tmp/verify-ds 2>/dev/null || true
python3 -m pip install --no-deps --target /tmp/verify-ds/lib/python*/site-packages \
  --index-url "https://moorethreads.github.io/deepselect-whl/${INDEX_VARIANT}/" \
  "deepselect-musa==${VERSION}+${LOCAL_LABEL}"
/tmp/verify-ds/bin/python "$REPO_DIR/scripts/ci/musa/smoke_test.py" \
  --expected-local-label "$LOCAL_LABEL" --expected-torch-musa 2.11.0.post1

log "published $VERSION+$LOCAL_LABEL"
echo "NOTE: PyPI sdist 暂不发布（见 DESIGN.md §8.2），修复 sdist 后再补。"
