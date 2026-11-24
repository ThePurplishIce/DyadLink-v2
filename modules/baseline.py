"""Longitudinal personal baseline for numeric observation features.

Principle (MEGA_PLAN §7.3): compare a person with their *own* past, never
with a population cut-off. Each observation event — a voice recording
analysis or a multimodal vision capture — contributes one sample of
numeric features. Per feature we keep an exponentially weighted mean and
standard deviation (alpha = 0.3). Deviation flags switch on only after at
least ``MIN_SAMPLES`` observations of that feature; deviations are shown
as z-scores, never converted to percentages.
"""

from __future__ import annotations

import math
from typing import Any

from . import session_store

ALPHA = 0.3
MIN_SAMPLES = 4
Z_ATTENTION = 1.5
Z_NOTABLE = 2.0

# Canonical numeric features. Acoustic keys mirror ``audio_features``;
# vision keys mirror the browser component's output.
FEATURE_LABELS = {
    "pause_per_min": "speech_pause_rate",
    "mean_pause_s": "speech_mean_pause",
    "longest_pause_s": "speech_longest_pause",
    "mean_voiced_segment_s": "speech_voiced_segment",
    "speech_rate_proxy": "speech_rate_energy",
    "speech_tempo_cv": "speech_tempo_var",
    "loudness_sd": "speech_loudness_var",
    "drift_silence": "speech_drift_silence",
    "f0_sd_st": "speech_pitch_var",
    "silence_ratio": "speech_silence_ratio",
    "jaw_range": "vision_jaw_range",
    "visual_syllable_rate": "vision_syllable_rate",
    "blink_rate": "vision_blink_rate",
    "blink_duration_ms": "vision_blink_duration",
    "perclos": "vision_perclos",
    "head_pitch_sd": "vision_head_pitch",
    "head_yaw_sd": "vision_head_yaw",
    "head_roll_sd": "vision_head_roll",
    "head_movement_energy": "vision_head_energy",
    "nod_rate": "vision_nod_rate",
    "stillness_ratio": "vision_stillness",
    "av_synchrony": "vision_av_sync",
    "articulating_silence_ratio": "vision_articulating_silence",
    "frozen_silence_ratio": "vision_frozen_silence",
    "mouth_asymmetry": "vision_mouth_asymmetry",
    "expressivity_sd": "vision_expressivity",
    "smile_ratio": "vision_smile_ratio",
    "frown_ratio": "vision_frown_ratio",
    "gaze_down_ratio": "vision_gaze_down",
    "gaze_shift_rate": "vision_gaze_shift",
    "utterance_len": "text_utterance_len",
    "ttr": "text_ttr",
    "content_ratio": "text_content_ratio",
    "pronoun_ratio": "text_pronoun_ratio",
    "filler_ratio": "text_filler_ratio",
}

# Features whose *decrease* is the clinically interesting direction get a
# direction hint for wording only — z-scores stay direction-agnostic.
LOW_IS_NOTABLE = frozenset({"f0_sd_st", "av_synchrony"})


def record_sample(token: str, features: dict[str, Any]) -> None:
    session_store.append_feature_sample(token, features)


def _samples(token: str) -> list[dict]:
    return session_store.get_feature_samples(token)


def _ewma_stats(samples: list[dict]) -> dict[str, dict[str, float]]:
    """Per-feature EWMA stats across ``samples`` (chronological)."""
    online: dict[str, dict[str, float]] = {}
    counts: dict[str, int] = {}
    for sample in samples:
        for key, value in sample.items():
            if key == "ts" or not isinstance(value, (int, float)):
                continue
            state = online.setdefault(key, {"mean": float(value), "var": 0.0})
            n = counts.get(key, 0)
            if n:
                delta = float(value) - state["mean"]
                state["mean"] += ALPHA * delta
                state["var"] = ALPHA * delta * delta + (1.0 - ALPHA) * state["var"]
            counts[key] = n + 1
    return {
        key: {
            "n": float(counts[key]),
            "mean": state["mean"],
            "sd": math.sqrt(max(state["var"], 0.0)),
        }
        for key, state in online.items()
    }


def feature_stats(token: str) -> dict[str, dict[str, float]]:
    """Per-feature EWMA stats across all stored samples (chronological)."""
    return _ewma_stats(_samples(token))


def latest_sample(token: str) -> dict[str, float]:
    samples = _samples(token)
    if not samples:
        return {}
    latest = samples[-1]
    return {
        key: float(value)
        for key, value in latest.items()
        if key != "ts" and isinstance(value, (int, float))
    }


def zscore(feature: str, value: float, stats: dict[str, dict[str, float]]) -> float | None:
    entry = stats.get(feature)
    if not entry or entry.get("n", 0) < MIN_SAMPLES:
        return None
    mean = float(entry.get("mean") or 0.0)
    sd = float(entry.get("sd") or 0.0)
    # A near-zero historical variance is treated as sensor noise, not as a
    # licence to flag any wobble — floor the sd at 2% of |mean|.
    sd = max(sd, abs(mean) * 0.02, 1e-6)
    return (value - mean) / sd


def deviation_level(z: float | None) -> str:
    if z is None:
        return "none"
    if abs(z) > Z_NOTABLE:
        return "notable"
    if abs(z) > Z_ATTENTION:
        return "attention"
    return "none"


def evaluate(token: str) -> dict[str, dict[str, Any]]:
    """Latest-sample z-scores and deviation levels per feature.

    The baseline stats are computed over all samples *except* the newest
    one, so a fresh observation is always compared with the person's own
    past rather than with a mean that already includes itself.
    """
    samples = _samples(token)
    if not samples:
        return {}
    stats = _ewma_stats(samples[:-1])
    latest = {
        key: float(value)
        for key, value in samples[-1].items()
        if key != "ts" and isinstance(value, (int, float))
    }
    result: dict[str, dict[str, Any]] = {}
    for feature, value in latest.items():
        z = zscore(feature, value, stats)
        result[feature] = {
            "value": value,
            "z": z,
            "level": deviation_level(z),
            "n": stats.get(feature, {}).get("n", 0),
            "mean": stats.get(feature, {}).get("mean"),
        }
    return result
