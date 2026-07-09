# CosyVoice3 on AMD Strix Halo (gfx1151) — OpenAI-compatible Korean TTS

Runs **Fun-CosyVoice3-0.5B** on the Ryzen AI Max+ 395 iGPU (Radeon 8060S, gfx1151)
and serves an OpenAI-compatible `/v1/audio/speech`. Build/run **on the server only**
(x86_64 + gfx1151); the dev Mac is for code + `tests/` only.

## Key results (measured, see NOTES.md)
- ✅ Full pipeline (LLM → DiT flow → HiFi-GAN) runs on gfx1151, **no segfault**, intelligible Korean.
- ✅ Real-time is achievable: **fp16 RTF ≈ 0.45** on a warm (cached-shape) utterance.
- ⚠️ **Not consistently real-time**: each *new* utterance length pays a flow/HiFi-GAN
  MIOpen conv penalty (~7s → RTF 1.6–2.4). It's a gfx1151 MIOpen limitation (the conv
  falls back to an insufficient-workspace `GemmFwdRest` solver); `cudnn.benchmark`,
  `MIOPEN_FIND_MODE=NORMAL`, and streaming did **not** fix it.
- ❌ **vLLM is not the lever**: in fp16 the LLM is already fast (~2.2s incl. CPU frontend);
  the bottleneck is flow/HiFi-GAN, which vLLM does not touch. (The original
  "vLLM fixes the AR-decode memcpy bottleneck" hypothesis does not hold on this stack.)

## Stack
- Base `ubuntu:26.04`; **ROCm 7.x + gfx1151-native PyTorch** from the AMD prerelease
  pip index `https://rocm.prereleases.amd.com/whl/gfx1151/` (self-contained wheels —
  **not** pytorch.org stock wheels, which SIGSEGV on gfx1151). Python 3.10 via `uv`.
- `torch==2.9.1+rocm7.13.0rc2`, `torchaudio==2.9.0` (`--no-deps`), CosyVoice deps with
  CUDA/TensorRT/training bits stripped (`requirements-strix.txt`). Eager fp16, no vLLM/TRT/JIT.

## Prerequisites (server)
```bash
git clone --recursive <your-fork>           # or: git submodule update --init --recursive
# docker access: usermod -aG docker $USER (re-login) or use sudo
# git config --global --add safe.directory $(pwd)   # if 'dubious ownership'
```

## Build & run (docker compose)
```bash
cd deploy/strixhalo
cp .env.example .env            # optional; set HF_TOKEN / SERVE_PORT if needed
docker compose up -d --build
```
Device passthrough is `/dev/kfd` + `/dev/dri` (device-only, root container — same as
Ollama). If access fails, uncomment `group_add: ["993","44"]` in `docker-compose.yml`.

### Model download (one-off, into the named volume)
```bash
docker compose run --rm cosyvoice-tts \
  python -c "from huggingface_hub import snapshot_download; \
snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512', local_dir='/models/Fun-CosyVoice3-0.5B')"
```

## Use
```bash
curl -s http://localhost:8000/health
curl -s http://localhost:8000/v1/models

curl -s http://localhost:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"cosyvoice3-0.5b","input":"안녕하세요. 오늘 날씨가 정말 좋네요.","response_format":"mp3"}' \
  -o out.mp3
```
Body fields (OpenAI-compatible): `input` (required), `response_format` (`mp3`|`wav`),
`speed`, `voice`/`model` (accepted, currently a single built-in prompt voice). Korean is
produced cross-lingually from a short reference clip (`asset/zero_shot_prompt.wav`); drop
a Korean reference clip and set `COSYVOICE_PROMPT_WAV`/`COSYVOICE_PROMPT_TEXT` for KR-native timbre.

## Benchmark (RTF)
```bash
docker compose run --rm cosyvoice-tts python bench_cosyvoice.py --fp16
# --stream for streaming + time-to-first-chunk; prints per-run RTF and LLM vs flow+hift split
```

## Dev-box unit tests (Mac, CosyVoice mocked)
```bash
pip install fastapi pydantic soundfile numpy httpx pytest
pytest deploy/strixhalo/tests/
```

## Home Assistant integration
HA's **OpenAI-compatible TTS** (or the "OpenAI Conversation"/`rest_command` route) can point
at this server: set the TTS base URL to `http://<server>:8000/v1` and model `cosyvoice3-0.5b`;
HA calls `POST /v1/audio/speech` and plays the returned mp3. Given the MIOpen novel-shape
penalty, it's best for short, cache-warm phrases; for long/varied text expect ~1.5–2.4× RTF.

## Files
`Dockerfile` (multi-stage: `rocm-torch` → `cosyvoice-eager` → `serve`), `server.py`
(OpenAI API), `bench_cosyvoice.py` (RTF), `requirements-strix.txt` / `constraints-strix.txt`,
`smoke_gpu.py` (Phase 1), `docker-compose.yml`, `tests/`, `NOTES.md` (full decision log).
