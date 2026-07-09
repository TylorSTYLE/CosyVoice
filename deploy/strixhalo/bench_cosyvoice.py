#!/usr/bin/env python3
"""Phase 2 (revised) — EAGER CosyVoice3 Korean RTF benchmark on gfx1151.

Runs the FULL pipeline (LLM autoregressive decode -> flow -> HiFi-GAN) with plain
ROCm torch, no vLLM / no TensorRT / no JIT, and measures RTF = synth_time / audio_len
for a Korean sentence. This is the baseline that decides whether vLLM is needed at all.

Run inside the container ON the Strix Halo server with GPU passthrough and the model
volume mounted. See deploy/strixhalo/README.md.
"""
import os
import sys
import time
import argparse

sys.path.append('third_party/Matcha-TTS')
import torch
import torchaudio
import soundfile as sf

# torchaudio 2.9 routes load()/save() through TorchCodec, which isn't installed
# (no ROCm/gfx1151 build). CosyVoice calls torchaudio.load(..., backend='soundfile');
# shim both to soundfile (already a dep) so we don't touch cosyvoice core.
def _sf_load(filepath, *args, **kwargs):
    data, samplerate = sf.read(filepath, dtype='float32', always_2d=True)  # (frames, ch)
    return torch.from_numpy(data.T).contiguous(), samplerate               # (ch, frames)


def _sf_save(filepath, src, sample_rate, *args, **kwargs):
    arr = src.detach().cpu().numpy()
    sf.write(filepath, arr.T if arr.ndim == 2 else arr, sample_rate)


torchaudio.load = _sf_load
torchaudio.save = _sf_save

# NOTE: tried torch.backends.cudnn.benchmark=True + MIOPEN_FIND_MODE=NORMAL to fix
# the flow/HiFi-GAN conv cost on gfx1151 — neither helped (MIOpen still falls back to
# the insufficient-workspace GemmFwdRest solver on novel shapes; NORMAL added a ~50s
# autotune penalty). Reverted to defaults. See NOTES.md Phase 2.

from cosyvoice.cli.cosyvoice import AutoModel
from cosyvoice.utils.common import set_all_random_seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_dir', default=os.environ.get('COSYVOICE_MODEL_DIR', '/models/Fun-CosyVoice3-0.5B'))
    ap.add_argument('--prompt_wav', default='asset/zero_shot_prompt.wav')
    ap.add_argument('--text', default='안녕하세요. 오늘 날씨가 정말 좋네요. 우리 함께 공원으로 산책하러 갈까요?')
    ap.add_argument('--mode', choices=['zero_shot', 'cross_lingual'], default='zero_shot')
    # CosyVoice3 asserts <|endofprompt|> (id 151646) is present in text/prompt_text.
    # Format mirrors vllm_example.py: "<instruct><|endofprompt|><prompt_transcript>".
    ap.add_argument('--prompt_text',
                    default='You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。',
                    help='CosyVoice3 requires <|endofprompt|> here (zero_shot mode)')
    ap.add_argument('--out', default='/out/cosy_kr_eager.wav')
    ap.add_argument('--warmup', type=int, default=1)
    ap.add_argument('--runs', type=int, default=3)
    ap.add_argument('--fp16', action='store_true', help='load model in fp16 (default fp32)')
    ap.add_argument('--stream', action='store_true', help='streaming synthesis (fixed chunk shapes)')
    args = ap.parse_args()

    print(f'torch {torch.__version__} | cuda(rocm) available: {torch.cuda.is_available()} | fp16={args.fp16}')
    if torch.cuda.is_available():
        print('device:', torch.cuda.get_device_name(0))

    t0 = time.time()
    # CosyVoice3 has NO load_jit param; pass only load_trt/load_vllm/fp16 (all False = eager).
    model = AutoModel(model_dir=args.model_dir, load_trt=False, load_vllm=False, fp16=args.fp16)
    print(f'[load] {time.time() - t0:.1f}s')

    # Split the pipeline: time token2wav (flow DiT + HiFi-GAN) vs the rest (LLM AR
    # decode + frontend). This decides whether vLLM (LLM-only) is even the right lever.
    t2w = {'t': 0.0}
    _orig_token2wav = model.model.token2wav

    def _timed_token2wav(*a, **k):
        s = time.perf_counter()
        r = _orig_token2wav(*a, **k)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t2w['t'] += time.perf_counter() - s
        return r

    model.model.token2wav = _timed_token2wav

    sr = model.sample_rate
    # This fork's inference_* take the prompt as a FILE PATH (see vllm_example.py);
    # the frontend loads it internally at 16k/24k. Do NOT pre-load to a tensor.
    print(f'sample_rate={sr}  mode={args.mode}  text="{args.text}"')

    def synth():
        """Return (wav, time_to_first_chunk). ttfc is the 'live' latency metric."""
        if args.mode == 'cross_lingual':
            gen = model.inference_cross_lingual(args.text, args.prompt_wav, stream=args.stream)
        else:
            gen = model.inference_zero_shot(args.text, args.prompt_text, args.prompt_wav, stream=args.stream)
        chunks, ttfc = [], None
        t0 = time.time()
        for o in gen:
            if ttfc is None:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                ttfc = time.time() - t0
            chunks.append(o['tts_speech'])
        return torch.cat(chunks, dim=1), ttfc

    for w in range(args.warmup):
        set_all_random_seed(0)
        _, _ = synth()
        print(f'[warmup {w}] done')

    best_rtf, wav = None, None
    for r in range(args.runs):
        set_all_random_seed(r)
        t2w['t'] = 0.0
        t = time.time()
        wav, ttfc = synth()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        el = time.time() - t
        dur = wav.shape[1] / sr
        rtf = el / dur
        flow_hift = t2w['t']
        llm_rest = el - flow_hift
        best_rtf = rtf if best_rtf is None else min(best_rtf, rtf)
        print(f'[run {r}] synth {el:.2f}s | audio {dur:.2f}s | RTF {rtf:.3f} '
              f'| ttfc {ttfc:.2f}s | flow+hift(token2wav) {flow_hift:.2f}s | llm+rest {llm_rest:.2f}s')

    print(f'[result] best RTF {best_rtf:.3f}  ({"REAL-TIME OK (<1.0)" if best_rtf < 1.0 else "SLOWER THAN REAL-TIME"})')

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torchaudio.save(args.out, wav, sr)
    print('saved', args.out)


if __name__ == '__main__':
    main()
