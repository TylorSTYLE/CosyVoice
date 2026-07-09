#!/usr/bin/env python3
"""OpenAI-compatible TTS server for CosyVoice3 on Strix Halo (gfx1151).

Endpoints:
  GET  /health          -> {"status": "ok"}
  GET  /v1/models       -> OpenAI-style model list
  POST /v1/audio/speech -> synthesize speech (mp3/wav), OpenAI-compatible body

Runs the full CosyVoice3 pipeline eagerly (fp16) on the gfx1151 iGPU. See
deploy/strixhalo/NOTES.md for why vLLM is not used (LLM is not the bottleneck).

The audio-encoding and request/response logic is unit-tested on the dev Mac with
the CosyVoice engine mocked (deploy/strixhalo/tests/test_server.py); model loading
and synthesis only run on the server.
"""
import io
import os
import subprocess
from typing import Optional

import numpy as np
from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

# --- torchaudio 2.9 -> soundfile shim (no TorchCodec on gfx1151); see bench_cosyvoice.py ---
import soundfile as sf

try:
    import torch
    import torchaudio

    def _sf_load(filepath, *a, **k):
        data, samplerate = sf.read(filepath, dtype='float32', always_2d=True)
        return torch.from_numpy(data.T).contiguous(), samplerate

    def _sf_save(filepath, src, sample_rate, *a, **k):
        arr = src.detach().cpu().numpy()
        sf.write(filepath, arr.T if arr.ndim == 2 else arr, sample_rate)

    torchaudio.load = _sf_load
    torchaudio.save = _sf_save
except Exception:  # torch absent on the dev box during unit tests
    torch = None


# ----------------------------- config -----------------------------
MODEL_DIR = os.environ.get('COSYVOICE_MODEL_DIR', '/models/Fun-CosyVoice3-0.5B')
PROMPT_WAV = os.environ.get('COSYVOICE_PROMPT_WAV', 'asset/zero_shot_prompt.wav')
# CosyVoice3 requires <|endofprompt|> in prompt_text (see vllm_example.py).
PROMPT_TEXT = os.environ.get(
    'COSYVOICE_PROMPT_TEXT',
    'You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。')
FP16 = os.environ.get('COSYVOICE_FP16', '1') not in ('0', 'false', 'False', '')
MODEL_ID = os.environ.get('COSYVOICE_MODEL_ID', 'cosyvoice3-0.5b')
# Fixed seed -> deterministic token count -> stable flow/HiFi-GAN conv shapes -> MIOpen
# kernel-cache hits on repeated text (big win for canned phrases). Set -1 to randomize.
SEED = int(os.environ.get('COSYVOICE_SEED', '0'))
SUPPORTED_FORMATS = {'mp3': 'audio/mpeg', 'wav': 'audio/wav'}


# ----------------------------- audio encoding (pure, testable) -----------------------------
def float_to_pcm16(wav: np.ndarray) -> np.ndarray:
    """(N,) or (1,N) float in [-1,1] -> (N,) int16."""
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    return (np.clip(wav, -1.0, 1.0) * 32767.0).astype(np.int16)


def pcm16_to_wav_bytes(pcm: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, pcm, sr, format='WAV', subtype='PCM_16')
    return buf.getvalue()


def pcm16_to_mp3_bytes(pcm: np.ndarray, sr: int, bitrate: str = '128k') -> bytes:
    """Encode via ffmpeg (installed in the image). Raises if ffmpeg missing."""
    wav = pcm16_to_wav_bytes(pcm, sr)
    proc = subprocess.run(
        ['ffmpeg', '-hide_banner', '-loglevel', 'error',
         '-f', 'wav', '-i', 'pipe:0', '-f', 'mp3', '-b:a', bitrate, 'pipe:1'],
        input=wav, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    return proc.stdout


def encode(pcm: np.ndarray, sr: int, response_format: str) -> bytes:
    if response_format == 'wav':
        return pcm16_to_wav_bytes(pcm, sr)
    return pcm16_to_mp3_bytes(pcm, sr)


# ----------------------------- CosyVoice engine (server-only) -----------------------------
class CosyVoiceEngine:
    def __init__(self, model_dir=MODEL_DIR, prompt_wav=PROMPT_WAV,
                 prompt_text=PROMPT_TEXT, fp16=FP16, seed=SEED):
        import sys
        # Fail fast instead of silently running on CPU (e.g. compose device passthrough
        # not recursing into /dev/dri). Set COSYVOICE_REQUIRE_GPU=0 to allow CPU.
        require_gpu = os.environ.get('COSYVOICE_REQUIRE_GPU', '1') not in ('0', 'false', 'False', '')
        if require_gpu and not (torch is not None and torch.cuda.is_available()):
            raise RuntimeError(
                'GPU (ROCm/gfx1151) not visible to torch — refusing to run on CPU. '
                'Check device passthrough (/dev/kfd + /dev/dri/renderD128) and group_add. '
                'Set COSYVOICE_REQUIRE_GPU=0 to override.')
        sys.path.append('third_party/Matcha-TTS')
        from cosyvoice.cli.cosyvoice import AutoModel
        from cosyvoice.utils.common import set_all_random_seed
        self._set_seed = set_all_random_seed
        self.seed = seed
        self.prompt_wav = prompt_wav
        self.prompt_text = prompt_text
        self.model = AutoModel(model_dir=model_dir, load_trt=False, load_vllm=False, fp16=fp16)
        self.sr = self.model.sample_rate
        self._warmup()

    def _warmup(self):
        """Precompile base MIOpen kernels so the first real request isn't a cold ~40s compile."""
        try:
            self.synth('안녕하세요. 반갑습니다.')
        except Exception as e:  # pragma: no cover - warmup is best-effort
            import logging
            logging.warning('warmup synth failed (non-fatal): %s', e)

    def synth(self, text: str, speed: float = 1.0):
        """Return (pcm_int16 (N,), sample_rate)."""
        if self.seed >= 0:
            self._set_seed(self.seed)
        chunks = []
        for out in self.model.inference_zero_shot(
                text, self.prompt_text, self.prompt_wav, stream=False, speed=speed):
            chunks.append(out['tts_speech'])
        wav = torch.cat(chunks, dim=1)
        return float_to_pcm16(wav.squeeze(0).float().cpu().numpy()), self.sr


ENGINE: Optional[CosyVoiceEngine] = None  # lazily built on first request (or injected in tests)


def get_engine() -> CosyVoiceEngine:
    global ENGINE
    if ENGINE is None:
        ENGINE = CosyVoiceEngine()
    return ENGINE


# ----------------------------- API -----------------------------
class SpeechRequest(BaseModel):
    input: str
    model: Optional[str] = None
    voice: Optional[str] = 'default'
    response_format: Optional[str] = 'mp3'
    speed: Optional[float] = 1.0


app = FastAPI(title='CosyVoice3 OpenAI-compatible TTS (Strix Halo gfx1151)')


@app.get('/health')
def health():
    return {'status': 'ok'}


@app.get('/v1/models')
def list_models():
    return {'object': 'list',
            'data': [{'id': MODEL_ID, 'object': 'model', 'owned_by': 'cosyvoice'}]}


@app.post('/v1/audio/speech')
def create_speech(req: SpeechRequest):
    text = (req.input or '').strip()
    if not text:
        return JSONResponse(status_code=400, content={'error': {'message': 'input is required'}})
    fmt = (req.response_format or 'mp3').lower()
    if fmt not in SUPPORTED_FORMATS:
        return JSONResponse(status_code=400, content={'error': {
            'message': f'unsupported response_format {fmt!r}; supported: {sorted(SUPPORTED_FORMATS)}'}})
    speed = float(req.speed or 1.0)
    pcm, sr = get_engine().synth(text, speed=speed)
    audio = encode(pcm, sr, fmt)
    return Response(content=audio, media_type=SUPPORTED_FORMATS[fmt])


if __name__ == '__main__':
    import uvicorn
    port = int(os.environ.get('SERVE_PORT', '8000'))
    # Build the engine eagerly so the container is "ready" only once the model is loaded.
    get_engine()
    uvicorn.run(app, host='0.0.0.0', port=port)
