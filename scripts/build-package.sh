#!/usr/bin/env bash
#
# Build the `ferroload` Python distribution locally: a wheel for the current
# platform + an sdist, into ./dist. Use for a smoke test or a manual upload.
#
# Cross-platform wheels (manylinux, macOS arm64/x86_64, Windows) are produced by
# CI — see .github/workflows/release.yml. This script is single-platform.
#
# Usage:
#   bash scripts/build-package.sh            # build into ./dist
#   bash scripts/build-package.sh --video    # include the video feature (needs ffmpeg)
set -euo pipefail
cd "$(dirname "$0")/.."

OUT="$(pwd)/dist"
FEATURES_ARG=()
if [[ "${1:-}" == "--video" ]]; then
  FEATURES_ARG=(--features video)
  echo "Building WITH the video feature (requires system ffmpeg + clang)."
fi

python -m pip install -U maturin twine >/dev/null

pushd crates/ferroload-py >/dev/null
maturin build --release --out "$OUT" "${FEATURES_ARG[@]}"
maturin sdist --out "$OUT"
popd >/dev/null

echo
echo "Artifacts in $OUT:"
ls -1 "$OUT"
echo
echo "Validate metadata:   python -m twine check $OUT/*"
echo "Publish (token):     python -m twine upload $OUT/*"
echo "Or with maturin:     (cd crates/ferroload-py && maturin publish)"
echo "Recommended:         push a 'v<version>' tag and let CI publish via Trusted Publishing."
