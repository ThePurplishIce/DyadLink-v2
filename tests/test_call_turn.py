"""Tests for the video-call turn pipeline helpers.

The browser component emits utterances as base64 WAV clips; these tests
cover the Python-side payload shape and the WAV contract that
``deepseek_service._validate_audio`` relies on.
"""

import base64
import io
import wave

from modules.deepseek_service import _validate_audio
from modules.home import _call_reply_payload


def _make_wav_pcm16(seconds: float = 1.0, rate: int = 16000) -> bytes:
    """Build a PCM16 mono WAV identical in layout to the JS encoder."""
    frames = int(seconds * rate)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * frames)
    return buf.getvalue()


def test_call_reply_payload_shape() -> None:
    payload = _call_reply_payload(1234, "你好", b"mp3-bytes")
    assert payload == {
        "id": 1234,
        "text": "你好",
        "audio_b64": base64.b64encode(b"mp3-bytes").decode("ascii"),
    }


def test_call_reply_payload_without_audio() -> None:
    payload = _call_reply_payload("ts-1", "error text")
    assert payload["id"] == "ts-1"
    assert payload["audio_b64"] is None


def test_js_style_wav_passes_asr_validation() -> None:
    # Mirrors encodeWavPcm16() in components/vision/main.js: RIFF/WAVE,
    # PCM fmt chunk, 16 kHz mono 16-bit.
    wav = _make_wav_pcm16(seconds=2.0)
    assert wav.startswith(b"RIFF")
    duration_ms = _validate_audio(wav)
    assert 1900 <= duration_ms <= 2100


def test_utterance_wav_size_stays_under_limit() -> None:
    # 15 s cap enforced JS-side; even the max clip must fit MAX_AUDIO_BYTES.
    wav = _make_wav_pcm16(seconds=15.0)
    b64 = base64.b64encode(wav).decode("ascii")
    assert len(b64) < 700_000
