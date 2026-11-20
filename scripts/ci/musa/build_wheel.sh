# Build a prebuilt DeepSelect MUSA wheel.
#
# Runs inside the `deepselect-build-musa` image (or any host with the MUSA 5.2
# stack, torch/torch_musa 2.11.0.post1, mcc, build/wheel/auditwheel installed).
#
# Usage:
#   build_wheel.sh --src /src/DeepSelect --out /out
#   build_wheel.sh --clone https://github.com/MooreThreads/DeepSelect.git --out /out
#
# Version/tag injection (see WHEEL-TAG-SPEC):
#   DEEP_SELECT_PUBLIC_VERSION   optional, defaults to deep_select/__version__.py
#   DEEP_SELECT_LOCAL_LABEL      optional, e.g. musa5.2.0torch2.11.0.post1
#                                (defaults to compose_wheel_tag.py output)
#   DEEP_SELECT_VERSION          optional full version (public[+local]); highest
#                                priority, e.g. 0.0.1+musa5.2.0torch2.11.0.post1
#   DEEP_SELECT_DIST_NAME        optional, defaults to deepselect-musa
#
# Never use `pip wheel .` / `pip install .` to build this project before the
# setup.py version fix lands in a release: setup.py used to embed a per-call
# timestamp and pip >= 25 validates the wheel filename against metadata.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

SRC_DIR=""
OUT_DIR="/out"
CLONE_URL=""
REF="main"
SKIP_SRC_SETUP=0
JOBS="${MAX_JOBS:-$(nproc)}"

log() { printf '\n### %s\n' "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --src) SRC_DIR="$2"; shift 2 ;;
    --out) OUT_DIR="$2"; shift 2 ;;
    --clone) CLONE_URL="$2"; shift 2 ;;
    --ref) REF="$2"; shift 2 ;;
    --skip-src-setup) SKIP_SRC_SETUP=1; shift ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ -n "$SRC_DIR" || -n "$CLONE_URL" ]] || die "pass --src DIR or --clone URL"
mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"

export DEEP_SELECT_BUILD_TARGET_PLATFORM=MUSA
export MAX_JOBS="$JOBS"
[[ -n "${MUSA_HOME:-}" ]] || export MUSA_HOME=/usr/local/musa
export PATH="$MUSA_HOME/bin:/home/mccxadmin/.local/bin:${PATH}"

# ------------------------------------------------------------------- toolchain
log "toolchain"
python3 -V
python3 - <<'PY'
import os, torch, torch_musa
print("torch", torch.__version__)
print("torch_musa", torch_musa.__version__)
print("MUSA_HOME", os.environ.get("MUSA_HOME"))
PY
command -v mcc && mcc --version | head -2 || die "mcc not found"

# ----------------------------------------------------------------- source setup
if [[ -n "$CLONE_URL" ]]; then
  SRC_DIR="${SRC_DIR:-/src/DeepSelect}"
  log "cloning $CLONE_URL ($REF) -> $SRC_DIR"
  rm -rf "$SRC_DIR"
  git clone --branch "$REF" "$CLONE_URL" "$SRC_DIR" \
    || die "git clone failed (use --src with an already-fetched tree on restricted networks)"
fi
SRC_DIR="$(cd "$SRC_DIR" && pwd)"
cd "$SRC_DIR"
[[ -f setup.py ]] || die "no setup.py in $SRC_DIR"

if [[ "$SKIP_SRC_SETUP" -eq 0 ]]; then
  log "submodules"
  if [[ -f csrc/3rdparty/mutlass/include/mutlass/version.h ]]; then
    echo "MUTLASS headers already present"
  elif [[ -e .gitmodules ]]; then
    git submodule update --init --depth 1 --recursive 2>/dev/null \
      || git submodule update --init --recursive \
      || echo "WARNING: submodule update failed; trying pinned codeload tarballs"
    if [[ ! -f csrc/3rdparty/mutlass/include/mutlass/version.h ]]; then
      log "submodule fallback: codeload tarballs at pinned commits"
      python3 - <<'PY'
import pathlib, subprocess, tarfile, urllib.request
out = subprocess.run(["git", "ls-tree", "HEAD", "csrc/3rdparty/"],
                     capture_output=True, text=True, check=True)
urls = {
    "csrc/3rdparty/cutlass": "https://codeload.github.com/NVIDIA/cutlass/tar.gz/",
    "csrc/3rdparty/mutlass": "https://codeload.github.com/MooreThreads/mutlass/tar.gz/",
}
for line in out.stdout.splitlines():
    meta, path = line.split("\t", 1)
    mode, typ, sha = meta.split()
    if typ != "commit":
        continue
    if (pathlib.Path(path) / "include").is_dir():
        continue
    if path not in urls:
        continue
    tmp = pathlib.Path("/tmp") / (pathlib.Path(path).name + ".tar.gz")
    print("downloading", urls[path] + sha)
    urllib.request.urlretrieve(urls[path] + sha, tmp)
    target = pathlib.Path(path); target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tmp) as tf:
        members = []
        for m in tf.getmembers():
            parts = pathlib.PurePosixPath(m.name).parts[1:]
            if parts:
                members.append(m.replace(name=str(pathlib.PurePosixPath(*parts))))
        tf.extractall(target, members=members)
    print("extracted", path)
PY
    fi
  fi
  [[ -f csrc/3rdparty/mutlass/include/mutlass/version.h ]] \
    || echo "WARNING: MUTLASS headers missing; MUSA build will likely fail"
fi

# ------------------------------------------------------- version / wheel tag
log "version and wheel tag (WHEEL-TAG-SPEC)"
PUBLIC_VERSION="${DEEP_SELECT_PUBLIC_VERSION:-$(python3 -c 'exec(open("deep_select/__version__.py").read()); print(__version__)')}"
if [[ -z "${DEEP_SELECT_VERSION:-}" ]]; then
  eval "$(python3 "$SCRIPT_DIR/compose_wheel_tag.py" --public-version "$PUBLIC_VERSION" ${DEEP_SELECT_LOCAL_LABEL:+--local-label "$DEEP_SELECT_LOCAL_LABEL"} --export)"
else
  # Full version provided explicitly; derive local label for build info/checks.
  DEEP_SELECT_PUBLIC_VERSION="$PUBLIC_VERSION"
  if [[ "$DEEP_SELECT_VERSION" == *+* ]]; then
    DEEP_SELECT_LOCAL_LABEL="${DEEP_SELECT_VERSION#*+}"
  else
    DEEP_SELECT_LOCAL_LABEL=""   # PyPI-compatible build: public version only
  fi
  DEEP_SELECT_DIST_NAME="${DEEP_SELECT_DIST_NAME:-deepselect-musa}"
fi
export DEEP_SELECT_VERSION DEEP_SELECT_DIST_NAME DEEP_SELECT_PUBLIC_VERSION DEEP_SELECT_LOCAL_LABEL
export DEEP_SELECT_INDEX_VARIANT="${DEEP_SELECT_INDEX_VARIANT:-}"
echo "dist=$DEEP_SELECT_DIST_NAME version=$DEEP_SELECT_VERSION local=$DEEP_SELECT_LOCAL_LABEL"

log "source revision"
{
  echo "git_head=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "mutlass_commit=$(git ls-tree HEAD csrc/3rdparty/mutlass 2>/dev/null | awk '{print $3}')"
  echo "cutlass_commit=$(git ls-tree HEAD csrc/3rdparty/cutlass 2>/dev/null | awk '{print $3}')"
  echo "build_python=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  echo "build_torch=$(python3 -c 'import torch; print(torch.__version__)')"
  echo "build_torch_public=$(python3 -c 'from packaging.version import Version; import torch; print(Version(torch.__version__).public)')"
  echo "build_torch_musa=$(python3 -c 'import torch_musa; print(torch_musa.__version__)')"
  echo "musa_home=${MUSA_HOME}"
  echo "musa_sdk=$(python3 -c 'from packaging.version import Version; import torch_musa, re; print(re.search(r"musa([0-9.]+)", str(Version(torch_musa.__version__).local)).group(1))')"
  echo "SOURCE_DATE_EPOCH=${SOURCE_DATE_EPOCH:-$(git log -1 --format=%ct 2>/dev/null || date +%s)}"
} | tee "$OUT_DIR/build-metadata.txt"

python3 "$SCRIPT_DIR/gen_build_info.py" \
  --output deep_select/_build_info.py \
  --public-version "$DEEP_SELECT_PUBLIC_VERSION" \
  --local-label "$DEEP_SELECT_LOCAL_LABEL" \
  --metadata "$OUT_DIR/build-metadata.txt"

# ------------------------------------------------------------------- build
log "clean previous build artifacts"
rm -rf build dist ./*.egg-info

log "build wheel (python -m build --wheel --no-isolation)"
START=$(date +%s)
set +e
python3 -m build --wheel --no-isolation --outdir "$OUT_DIR" . 2>&1 | tee "$OUT_DIR/build.log"
BUILD_RC=${PIPESTATUS[0]}
set -e
END=$(date +%s)
echo "BUILD_ELAPSED_SEC=$((END - START)) BUILD_RC=$BUILD_RC" | tee -a "$OUT_DIR/build.log"
if [[ "$BUILD_RC" -ne 0 ]]; then
  log "fallback: setup.py bdist_wheel"
  python3 setup.py bdist_wheel --dist-dir "$OUT_DIR" 2>&1 | tee "$OUT_DIR/build-bdist.log"
  BUILD_RC=${PIPESTATUS[0]}
fi
[[ "$BUILD_RC" -eq 0 ]] || die "wheel build failed"

rm -f deep_select/_build_info.py
WHEEL=$(ls -t "$OUT_DIR"/*.whl | head -1)
[[ -n "$WHEEL" ]] || die "no wheel produced in $OUT_DIR"
log "wheel: $WHEEL"

# ---------------------------------------------------------------- inspect/check
log "wheel tag check"
python3 "$SCRIPT_DIR/check_wheel_tag.py" "$WHEEL" \
  --expected-local "$DEEP_SELECT_LOCAL_LABEL" --require-build-info \
  | tee "$OUT_DIR/wheel-tag-check.json"

log "wheel contents"
unzip -l "$WHEEL" | tee "$OUT_DIR/wheel-contents.txt"
log "sha256"
( cd "$OUT_DIR" && sha256sum "$(basename "$WHEEL")" > SHA256SUMS && cat SHA256SUMS )

log "elf runtime deps"
rm -rf /tmp/wheel-inspect && mkdir -p /tmp/wheel-inspect
unzip -qo "$WHEEL" -d /tmp/wheel-inspect
SO=$(find /tmp/wheel-inspect/deep_select -name '*.so' | head -1)
{
  echo "### $SO"; ls -l "$SO"; file "$SO"
  echo "--- readelf -d ---"; readelf -d "$SO" | grep -E 'NEEDED|RPATH|RUNPATH|SONAME'
  echo "--- ldd ---"; ldd "$SO"
} | tee "$OUT_DIR/elf-deps.txt"

if command -v auditwheel >/dev/null 2>&1 || python3 -c 'import auditwheel' 2>/dev/null; then
  log "auditwheel show"
  python3 -m auditwheel show "$WHEEL" 2>&1 | tee "$OUT_DIR/auditwheel-show.txt" || true
fi

log "done"
echo "wheel : $WHEEL"
echo "size  : $(stat -c %s "$WHEEL") bytes"
echo "sha256: $(cd "$OUT_DIR" && sha256sum "$(basename "$WHEEL")" | cut -d' ' -f1)"
