"""Local (dev Mac) unit tests for the OpenAI-compatible TTS server.

CosyVoice/torch are NOT exercised: we inject a fake engine and test the OpenAI
schema, audio encoding, and request/response logic. Run: pytest deploy/strixhalo/tests
Requires: fastapi, pydantic, soundfile, numpy, httpx, pytest (dev-only).
"""
import io
import os
import shutil
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import server  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


class FakeEngine:
    sr = 24000

    def synth(self, text, speed=1.0):
        n = int(self.sr * 0.5)  # 0.5s tone
        pcm = (np.sin(np.arange(n) * 0.05) * 10000).astype(np.int16)
        return pcm, self.sr


@pytest.fixture()
def client():
    server.ENGINE = FakeEngine()
    return TestClient(server.app)


def test_health(client):
    r = client.get('/health')
    assert r.status_code == 200 and r.json() == {'status': 'ok'}


def test_models(client):
    r = client.get('/v1/models')
    body = r.json()
    assert r.status_code == 200
    assert body['object'] == 'list'
    assert body['data'][0]['id'] == server.MODEL_ID


def test_speech_wav(client):
    r = client.post('/v1/audio/speech',
                    json={'input': '안녕하세요', 'response_format': 'wav'})
    assert r.status_code == 200
    assert r.headers['content-type'] == 'audio/wav'
    assert r.content[:4] == b'RIFF'
    # decodes back to the fake tone at 24k
    data, sr = _read(r.content)
    assert sr == 24000 and len(data) == int(24000 * 0.5)


def test_speech_defaults_to_mp3_contenttype(client, monkeypatch):
    # avoid requiring ffmpeg: stub the mp3 encoder
    monkeypatch.setattr(server, 'pcm16_to_mp3_bytes', lambda pcm, sr, **k: b'ID3fake')
    r = client.post('/v1/audio/speech', json={'input': '테스트'})
    assert r.status_code == 200
    assert r.headers['content-type'] == 'audio/mpeg'
    assert r.content == b'ID3fake'


def test_speech_empty_input(client):
    r = client.post('/v1/audio/speech', json={'input': '   '})
    assert r.status_code == 400


def test_speech_bad_format(client):
    r = client.post('/v1/audio/speech', json={'input': 'x', 'response_format': 'flac'})
    assert r.status_code == 400


def test_speech_missing_input_422(client):
    # pydantic rejects a body with no `input`
    r = client.post('/v1/audio/speech', json={'voice': 'default'})
    assert r.status_code == 422


def test_float_to_pcm16_clips():
    out = server.float_to_pcm16(np.array([0.0, 1.0, -1.0, 2.0, -2.0], dtype=np.float32))
    assert out.dtype == np.int16
    assert out[1] == 32767 and out[2] == -32767
    assert out[3] == 32767 and out[4] == -32767  # clipped


def test_wav_roundtrip():
    pcm = (np.sin(np.arange(1000) * 0.1) * 20000).astype(np.int16)
    data, sr = _read(server.pcm16_to_wav_bytes(pcm, 16000))
    assert sr == 16000 and len(data) == 1000


@pytest.mark.skipif(shutil.which('ffmpeg') is None, reason='ffmpeg not installed on dev box')
def test_mp3_encodes():
    pcm = (np.sin(np.arange(24000) * 0.05) * 10000).astype(np.int16)
    out = server.pcm16_to_mp3_bytes(pcm, 24000)
    assert isinstance(out, bytes) and len(out) > 100


def _read(b: bytes):
    import soundfile as sf
    data, sr = sf.read(io.BytesIO(b), dtype='int16')
    return data, sr
