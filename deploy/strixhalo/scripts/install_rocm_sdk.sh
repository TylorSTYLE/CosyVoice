#!/usr/bin/env bash
# Resolve and install the newest TheRock nightly ROCm SDK tarball for gfx1151
# into a target dir (default /opt/rocm). x86_64 Linux only.
# Source pattern: hec-ovi/vllm-qwen scripts/install_rocm_sdk.sh (see NOTES.md).
set -euo pipefail

ROCM_DIR="${1:-/opt/rocm}"
BASE="https://therock-nightly-tarball.s3.amazonaws.com"
PREFIX="therock-dist-linux-gfx1151-7"   # ROCm 7.x, gfx1151 native

echo "[install_rocm_sdk] resolving newest '${PREFIX}*.tar.gz' from S3 ..."
KEY="$(curl -fsSL "${BASE}?list-type=2&prefix=${PREFIX}" \
        | tr '<' '\n' \
        | grep -oE "${PREFIX}[^<]*\.tar\.gz" \
        | sort -V | tail -n1)"

if [ -z "${KEY}" ]; then
  echo "[install_rocm_sdk] ERROR: no tarball found for prefix '${PREFIX}'." >&2
  echo "[install_rocm_sdk] check the bucket manually: ${BASE}?list-type=2&prefix=${PREFIX}" >&2
  exit 1
fi

echo "[install_rocm_sdk] selected: ${KEY}"
mkdir -p "${ROCM_DIR}"
curl -fSL --retry 3 --retry-delay 5 "${BASE}/${KEY}" -o /tmp/therock.tar.gz
echo "[install_rocm_sdk] extracting into ${ROCM_DIR} ..."
tar xzf /tmp/therock.tar.gz -C "${ROCM_DIR}" --strip-components=1
rm -f /tmp/therock.tar.gz

# Record exactly which tarball we used (for reproducibility / NOTES.md).
echo "${KEY}" > "${ROCM_DIR}/.therock-tarball"

if [ -x "${ROCM_DIR}/bin/rocminfo" ] || [ -e "${ROCM_DIR}/bin/rocminfo" ]; then
  echo "[install_rocm_sdk] OK: ROCm installed at ${ROCM_DIR} (rocminfo present), tarball=${KEY}"
else
  echo "[install_rocm_sdk] WARN: ${ROCM_DIR}/bin/rocminfo not found after extract — inspect layout:" >&2
  ls -la "${ROCM_DIR}" >&2 || true
fi
