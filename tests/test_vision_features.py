from modules.vision_features import baseline_features, sanitize_capture


def _payload(**overrides):
    base = {
        "version": 1,
        "duration_s": 30.0,
        "frames": 900,
        "face_ratio": 0.98,
        "has_audio": True,
        "capture_id": 3,
        "captured_at": 1_700_000_000_000,
        "jaw_range": 0.42,
        "visual_syllable_rate": 210.0,
        "blink_rate": 18.0,
        "head_pitch_sd": 3.2,
        "head_yaw_sd": 4.1,
        "head_roll_sd": 2.0,
        "av_synchrony": 0.55,
        "av_lag_ms": 33,
        "articulating_silence_ratio": 0.3,
        "frozen_silence_ratio": 0.6,
        "silence_ratio": 0.4,
        "voiced_ratio": 0.6,
        "pause_per_min": 12.0,
        "mean_pause_s": 0.8,
    }
    base.update(overrides)
    return base


def test_sanitize_keeps_numeric_features() -> None:
    cleaned = sanitize_capture(_payload())
    assert cleaned["jaw_range"] == 0.42
    assert cleaned["av_synchrony"] == 0.55
    assert cleaned["capture_id"] == 3
    assert cleaned["has_audio"] is True
    assert cleaned["duration_s"] == 30.0


def test_sanitize_rejects_non_dict() -> None:
    assert sanitize_capture(None) == {}
    assert sanitize_capture("junk") == {}
    assert sanitize_capture([1, 2, 3]) == {}


def test_sanitize_requires_duration() -> None:
    payload = _payload()
    del payload["duration_s"]
    assert sanitize_capture(payload) == {}


def test_sanitize_clamps_out_of_range() -> None:
    cleaned = sanitize_capture(_payload(jaw_range=5.0, av_synchrony=2.0))
    assert cleaned["jaw_range"] == 1.0
    assert cleaned["av_synchrony"] == 1.0


def test_sanitize_drops_non_finite_and_bool() -> None:
    cleaned = sanitize_capture(
        _payload(jaw_range=float("nan"), blink_rate=True, head_yaw_sd=None)
    )
    assert "jaw_range" not in cleaned
    assert "blink_rate" not in cleaned
    assert "head_yaw_sd" not in cleaned


def test_sanitize_error_payload() -> None:
    assert sanitize_capture({"error": "too_short"}) == {"error": "too_short"}


def test_baseline_features_excludes_meta() -> None:
    features = baseline_features(sanitize_capture(_payload()))
    assert "jaw_range" in features
    assert "pause_per_min" in features
    assert "capture_id" not in features
    assert "duration_s" not in features
    assert "frames" not in features
    assert all(isinstance(v, float) for v in features.values())
