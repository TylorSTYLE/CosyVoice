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

# ---- Python 3.10 venv (CosyVoice-friendly; cp310 wheels exist on the index) ----
ENV VIRTUAL_ENV=/opt/venv
RUN uv venv --python 3.10 /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# ---- gfx1151-native PyTorch from AMD prerelease index ----
# The TheRock/AMD wheels are SELF-CONTAINED: `torch` pulls its ROCm runtime
# (rocm-sdk-libraries-gfx1151) from the SAME index, so NO system /opt/rocm and NO
# TheRock SDK tarball are needed for Phase 1. (The dev toolchain rocm-sdk-devel is
# added later in the vllm stage to source-build vLLM.) NEVER use pytorch.org stock
# ROCm wheels here — they SIGSEGV (exit 139) on gfx1151.
# Index confirmed to host torch(cp310) + rocm-sdk-libraries-gfx1151 + rocm-sdk-devel.
ARG TORCH_PIN="torch==2.9.1+rocm7.13.0rc2"
RUN uv pip install --pre ${TORCH_PIN} \
        --index-url https://rocm.prereleases.amd.com/whl/gfx1151/ \
        --extra-index-url https://pypi.org/simple
# CosyVoice needs numpy 1.26.4 (from PyPI; overrides any newer numpy pulled as a dep)
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
