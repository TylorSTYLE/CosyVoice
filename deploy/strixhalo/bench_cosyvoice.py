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

from cosyvoice.cli.cosyvoice import AutoModel
from cosyvoice.utils.file_utils import load_wav
from cosyvoice.utils.common import set_all_random_seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_dir', default=os.environ.get('COSYVOICE_MODEL_DIR', '/models/Fun-CosyVoice3-0.5B'))
    ap.add_argument('--prompt_wav', default='asset/zero_shot_prompt.wav')
    ap.add_argument('--text', default='안녕하세요. 오늘 날씨가 정말 좋네요. 우리 함께 공원으로 산책하러 갈까요?')
    ap.add_argument('--mode', choices=['cross_lingual', 'zero_shot'], default='cross_lingual')
    ap.add_argument('--prompt_text', default='希望你以后能够做的比我还好呦。',
                    help='transcript of prompt_wav (only for --mode zero_shot)')
    ap.add_argument('--out', default='/out/cosy_kr_eager.wav')
    ap.add_argument('--warmup', type=int, default=1)
    ap.add_argument('--runs', type=int, default=3)
    args = ap.parse_args()

    print(f'torch {torch.__version__} | cuda(rocm) available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print('device:', torch.cuda.get_device_name(0))

    t0 = time.time()
    # CosyVoice3 has NO load_jit param; pass only load_trt/load_vllm/fp16 (all False = eager).
    model = AutoModel(model_dir=args.model_dir, load_trt=False, load_vllm=False, fp16=False)
    print(f'[load] {time.time() - t0:.1f}s')

    sr = model.sample_rate
    prompt = load_wav(args.prompt_wav, 16000)
    print(f'sample_rate={sr}  mode={args.mode}  text="{args.text}"')

    def synth():
        if args.mode == 'cross_lingual':
            gen = model.inference_cross_lingual(args.text, prompt, stream=False)
        else:
            gen = model.inference_zero_shot(args.text, args.prompt_text, prompt, stream=False)
        chunks = [o['tts_speech'] for o in gen]
        return torch.cat(chunks, dim=1)

    for w in range(args.warmup):
        set_all_random_seed(0)
        _ = synth()
        print(f'[warmup {w}] done')

    best_rtf, wav = None, None
    for r in range(args.runs):
        set_all_random_seed(r)
        t = time.time()
        wav = synth()
        el = time.time() - t
        dur = wav.shape[1] / sr
        rtf = el / dur
        best_rtf = rtf if best_rtf is None else min(best_rtf, rtf)
        print(f'[run {r}] synth {el:.2f}s | audio {dur:.2f}s | RTF {rtf:.3f}')

    print(f'[result] best RTF {best_rtf:.3f}  ({"REAL-TIME OK (<1.0)" if best_rtf < 1.0 else "SLOWER THAN REAL-TIME"})')

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torchaudio.save(args.out, wav, sr)
    print('saved', args.out)


if __name__ == '__main__':
    main()
