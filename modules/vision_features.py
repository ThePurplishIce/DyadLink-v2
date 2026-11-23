"""Validation and normalisation for multimodal capture payloads.

The browser component returns a flat dict of aggregate numbers. This module
bounds every field, drops non-numeric or out-of-range values, and produces
the canonical feature dict stored in the personal baseline. Nothing here
trusts the client blindly — but remember the component is first-party code
running in the page, so the checks are sanity bounds, not a security
boundary.
"""

from __future__ import annotations

import math
from typing import Any

# feature -> (min, max) sanity bounds
_BOUNDS: dict[str, tuple[float, float]] = {
    "jaw_range": (0.0, 1.0),
    "visual_syllable_rate": (0.0, 600.0),
    "blink_rate": (0.0, 120.0),
    "head_pitch_sd": (0.0, 180.0),
    "head_yaw_sd": (0.0, 180.0),
    "head_roll_sd": (0.0, 180.0),
    "av_synchrony": (-1.0, 1.0),
    "av_lag_ms": (-400.0, 400.0),
    "articulating_silence_ratio": (0.0, 1.0),
    "frozen_silence_ratio": (0.0, 1.0),
    "silence_ratio": (0.0, 1.0),
    "voiced_ratio": (0.0, 1.0),
    "pause_per_min": (0.0, 120.0),
    "mean_pause_s": (0.0, 30.0),
    "face_ratio": (0.0, 1.0),
    "silent_coverage": (0.0, 1.0),
    "duration_s": (0.0, 180.0),
    "pause_count": (0.0, 1000.0),
    # expression & gaze (Tier 1)
    "mouth_asymmetry": (0.0, 2.0),
    "expressivity_sd": (0.0, 1.0),
    "smile_ratio": (0.0, 1.0),
    "frown_ratio": (0.0, 1.0),
    "perclos": (0.0, 1.0),
    "blink_duration_ms": (0.0, 2000.0),
    "gaze_down_ratio": (0.0, 1.0),
    "gaze_shift_rate": (0.0, 120.0),
    # audio envelope (Tier 2)
    "longest_pause_s": (0.0, 60.0),
    "mean_voiced_segment_s": (0.0, 60.0),
    "speech_rate_proxy": (0.0, 600.0),
    "speech_tempo_cv": (0.0, 10.0),
    "loudness_sd": (0.0, 1.0),
    "drift_silence": (-1.0, 1.0),
    # head movement (Tier 3)
    "head_movement_energy": (0.0, 360.0),
    "nod_rate": (0.0, 120.0),
    "stillness_ratio": (0.0, 1.0),
}

# Keys copied verbatim into the stored capture record (non-feature meta).
_META_KEYS = ("capture_id", "captured_at", "version", "frames", "has_audio", "error")


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    if not math.isfinite(value):
        return None
    return value


def sanitize_capture(payload: Any) -> dict[str, Any]:
    """Return a cleaned capture dict, or ``{}`` when unusable."""
    if not isinstance(payload, dict):
        return {}
    if payload.get("error"):
        return {"error": str(payload.get("error"))[:60]}

    cleaned: dict[str, Any] = {}
    for key in _META_KEYS:
        value = payload.get(key)
        if key in {"capture_id", "frames", "version"}:
            number = _finite(value)
            if number is not None:
                cleaned[key] = int(number)
        elif key in {"captured_at"}:
            number = _finite(value)
            if number is not None:
                cleaned[key] = int(number)
        elif key == "has_audio":
            cleaned[key] = bool(value)
        elif key == "error":
            if value:
                cleaned[key] = str(value)[:60]

    for key, (low, high) in _BOUNDS.items():
        value = _finite(payload.get(key))
        if value is None:
            continue
        cleaned[key] = round(min(max(value, low), high), 4)

    if "duration_s" not in cleaned:
        return {}
    return cleaned


def baseline_features(capture: dict[str, Any]) -> dict[str, float]:
    """Subset of capture fields that feed the longitudinal baseline."""
    keys = (
        "jaw_range",
        "visual_syllable_rate",
        "blink_rate",
        "head_pitch_sd",
        "head_yaw_sd",
        "head_roll_sd",
        "av_synchrony",
        "articulating_silence_ratio",
        "frozen_silence_ratio",
        "silence_ratio",
        "voiced_ratio",
        "pause_per_min",
        "mean_pause_s",
        "longest_pause_s",
        "mean_voiced_segment_s",
        "speech_rate_proxy",
        "speech_tempo_cv",
        "loudness_sd",
        "drift_silence",
        "mouth_asymmetry",
        "expressivity_sd",
        "smile_ratio",
        "frown_ratio",
        "perclos",
        "blink_duration_ms",
        "gaze_down_ratio",
        "gaze_shift_rate",
        "head_movement_energy",
        "nod_rate",
        "stillness_ratio",
    )
    return {
        key: capture[key]
        for key in keys
        if isinstance(capture.get(key), (int, float))
        and not isinstance(capture.get(key), bool)
    }
