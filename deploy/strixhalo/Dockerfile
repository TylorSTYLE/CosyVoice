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
        ca-certificates curl git ffmpeg \
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
# SINGLE index only. The gfx1151 index also hosts torch's pure-python deps
# (filelock, sympy, networkx, numpy, typing-extensions, ...) and the ROCm runtime,
# so PyPI is not needed. Do NOT add `--extra-index-url https://pypi.org/simple`:
# uv's first-index strategy would then resolve `torch` from PyPI (which lacks the
# `+rocm7.13.0rc2` local version) and fail. numpy 1.26.4 (CosyVoice's pin) is
# installed later in the runtime stage — the Phase 1 smoke doesn't need it.
ARG TORCH_PIN="torch==2.9.1+rocm7.13.0rc2"
RUN uv pip install --pre ${TORCH_PIN} \
        --index-url https://rocm.prereleases.amd.com/whl/gfx1151/

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

# ============================================================
# Stage: cosyvoice-eager  (Phase 2 revised — measure eager RTF before vLLM)
# Full CosyVoice3 pipeline on gfx1151 torch, NO vLLM / TensorRT / JIT.
# ============================================================
FROM rocm-torch AS cosyvoice-eager

# torchaudio: gfx1151 ROCm build. --no-deps so its exact-torch pin can't try to
# swap the already-installed torch 2.9.1 (index only has torchaudio 2.9.0).
RUN uv pip install --pre --no-deps torchaudio==2.9.0+rocm7.13.0rc2 \
        --index-url https://rocm.prereleases.amd.com/whl/gfx1151/

# CosyVoice runtime deps (CUDA/TRT/train stripped). constraints hold torch/torchaudio/
# numpy; unsafe-best-match lets the constrained torch versions resolve from the ROCm
# index while everything else comes from PyPI.
# setuptools<81 + wheel are pre-installed so legacy sdists (openai-whisper imports
# pkg_resources, dropped in setuptools 81) build via --no-build-isolation. The
# setuptools<81 pin in constraints keeps it from being upgraded mid-resolution.
COPY deploy/strixhalo/requirements-strix.txt deploy/strixhalo/constraints-strix.txt /tmp/
# Pre-seed build deps for legacy sdists built under --no-build-isolation:
#   setuptools<81 -> pkg_resources (openai-whisper);  numpy+cython -> pyworld.
# numpy is pinned to CosyVoice's version so it isn't churned by the main install.
RUN uv pip install "setuptools<81" wheel "numpy==1.26.4" cython
RUN uv pip install -r /tmp/requirements-strix.txt \
        --constraint /tmp/constraints-strix.txt \
        --extra-index-url https://rocm.prereleases.amd.com/whl/gfx1151/ \
        --index-strategy unsafe-best-match \
        --no-build-isolation
# openai-whisper --no-deps: keeps ROCm torch's triton 3.5.1 (whisper pins triton<3).
RUN uv pip install --no-deps --no-build-isolation openai-whisper==20231117

# App source. NOTE: third_party/Matcha-TTS must be populated on the host first:
#   git submodule update --init --recursive
WORKDIR /opt/CosyVoice
COPY cosyvoice /opt/CosyVoice/cosyvoice
COPY third_party /opt/CosyVoice/third_party
COPY asset /opt/CosyVoice/asset
COPY deploy/strixhalo/bench_cosyvoice.py /opt/CosyVoice/bench_cosyvoice.py
COPY deploy/strixhalo/server.py /opt/CosyVoice/server.py
ENV PYTHONPATH=/opt/CosyVoice:/opt/CosyVoice/third_party/Matcha-TTS \
    HF_HOME=/cache/hf \
    MODELSCOPE_CACHE=/cache/modelscope \
    MIOPEN_USER_DB_PATH=/cache/miopen \
    MIOPEN_CUSTOM_CACHE_DIR=/cache/miopen \
    COSYVOICE_MODEL_DIR=/models/Fun-CosyVoice3-0.5B \
    COSYVOICE_FP16=1

CMD ["python", "bench_cosyvoice.py"]

# ============================================================
# Stage: serve  (Phase 4 — OpenAI-compatible /v1/audio/speech)
# ============================================================
FROM cosyvoice-eager AS serve
ENV SERVE_PORT=8000
EXPOSE 8000
CMD ["python", "server.py"]
