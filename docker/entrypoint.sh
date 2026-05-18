#!/bin/bash
# Entrypoint for BundleSDF CUDA 11.8 container (no ROS2).
# 1. Source env config (CUDA paths, torch lib, etc.)
# 2. Optionally build C++ submodules on first start
# 3. exec user command

set -e

# --- Source env config ---
if [ -f /etc/cu118.env ]; then
    source /etc/cu118.env >/dev/null 2>&1 || source /etc/cu118.env
fi

# --- Build mycuda + BundleTrack on first container start ---
# The sentinel /tmp/.bundlesdf_built is ephemeral (tmpfs); cleared on every
# container stop. Set SKIP_BUILD=1 to disable.
if [ "${SKIP_BUILD:-0}" != "1" ] && [ -d /workspace/BundleSDF ] && [ ! -f /tmp/.bundlesdf_built ]; then
    echo "[entrypoint] Building mycuda and BundleTrack (first start)..."
    (
        cd /workspace/BundleSDF && bash build.sh
    ) && touch /tmp/.bundlesdf_built || echo "[entrypoint] WARNING: build.sh failed; set SKIP_BUILD=1 to skip."
fi

# --- exec command ---
exec "$@"
