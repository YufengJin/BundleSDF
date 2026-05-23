#!/bin/bash
# Entrypoint for the BundleSDF container (DeepStream 7.1 base, CUDA 12.x, PyTorch 2.6).
# 1. Source env config (CUDA paths, torch lib, PYTHONPATH, build flags).
# 2. Build mycuda + BundleTrack on first start, but ONLY if their compiled artifacts
#    are missing. build.sh does a clean rebuild (rm -rf build), so guarding on the
#    actual .so files (which live on the live-mounted repo and therefore persist)
#    avoids a multi-minute recompile on every container restart. Set SKIP_BUILD=1
#    to force-skip.
# 3. Hand off to the base DeepStream entrypoint so its init still runs, then exec CMD.

set -e

# --- Source env config ---
if [ -f /etc/bundlesdf.env ]; then
    source /etc/bundlesdf.env
fi

# --- Build C++/CUDA modules on first start (only if artifacts are missing) ---
REPO=/workspace/BundleSDF
need_build=0
ls ${REPO}/BundleTrack/build/my_cpp*.so >/dev/null 2>&1 || need_build=1
ls ${REPO}/mycuda/gridencoder*.so       >/dev/null 2>&1 || need_build=1
if [ "${SKIP_BUILD:-0}" != "1" ] && [ -d "${REPO}" ] && [ "${need_build}" = "1" ]; then
    echo "[entrypoint] Building mycuda + BundleTrack (compiled artifacts missing)..."
    ( cd "${REPO}" && bash build.sh ) || echo "[entrypoint] WARNING: build.sh failed; set SKIP_BUILD=1 to skip."
fi

# --- Hand off to base DeepStream entrypoint (preserves its init), then exec CMD ---
exec /opt/nvidia/deepstream/deepstream-7.1/entrypoint.sh "$@"
