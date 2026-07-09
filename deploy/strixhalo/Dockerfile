# syntax=docker/dockerfile:1
# CosyVoice on AMD Strix Halo (gfx1151) — multi-stage.
# BUILD/RUN ONLY on the Strix Halo server (x86_64, gfx1151). NEVER build on Mac
# (no /dev/kfd; QEMU amd64 emulation cannot reach the GPU).
# Build context = repo root:  docker build -f deploy/strixhalo/Dockerfile .
# Pinned sources & rationale: deploy/strixhalo/NOTES.md
#
# Stages:
#   rocm-torch  -> Phase 1: TheRock ROCm 7.x + gfx1151-native PyTorch + smoke
#   (vllm / runtime stages are added in Phase 2+ once Phase 1 gate passes)

# ============================================================
# Stage: rocm-torch  (Phase 1)
# ============================================================
FROM ubuntu:26.04 AS rocm-torch
ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git \
        build-essential cmake ninja-build pkg-config \
        libnuma1 libnuma-dev \
    && rm -rf /var/lib/apt/lists/*

# uv: fast resolver + provisions a standalone CPython 3.10 (Ubuntu 26.04 ships 3.14).
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# ---- TheRock ROCm SDK (nightly gfx1151 tarball) -> /opt/rocm ----
ENV ROCM_PATH=/opt/rocm
COPY deploy/strixhalo/scripts/install_rocm_sdk.sh /tmp/install_rocm_sdk.sh
RUN bash /tmp/install_rocm_sdk.sh /opt/rocm
ENV PATH="/opt/rocm/bin:${PATH}" \
    LD_LIBRARY_PATH="/opt/rocm/lib:/opt/rocm/lib64"

# ---- Python 3.10 venv (CosyVoice-friendly; cp310 wheels exist on the index) ----
ENV VIRTUAL_ENV=/opt/venv
RUN uv venv --python 3.10 /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# ---- gfx1151-native PyTorch from AMD prerelease index (NOT pytorch.org stock) ----
# first-index strategy: torch/torchaudio come from the rocm index; pure-python
# deps fall through to PyPI. If resolution fails, this is the first knob to touch.
RUN uv pip install --pre torch torchaudio \
        --index-url https://rocm.prereleases.amd.com/whl/gfx1151/ \
        --extra-index-url https://pypi.org/simple
RUN uv pip install numpy==1.26.4

# gfx1151 runtime env (from hec-ovi; harmless if a native build doesn't need them)
ENV HSA_OVERRIDE_GFX_VERSION=11.5.1 \
    PYTORCH_ROCM_ARCH=gfx1151 \
    VLLM_ROCM_USE_AITER=0 \
    ROCBLAS_USE_HIPBLASLT=1 \
    TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
    HIP_FORCE_DEV_KERNARG=1

COPY deploy/strixhalo/smoke_gpu.py /opt/smoke_gpu.py
# Smoke runs at `docker run` (needs /dev/kfd), not at build time.
CMD ["python", "/opt/smoke_gpu.py"]
