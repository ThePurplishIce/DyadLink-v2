import pytest

from modules import baseline, session_store


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(session_store, "_STORE_PATH", tmp_path / "store.json")
    return tmp_path / "store.json"


def _feed(token: str, feature: str, values: list[float]) -> None:
    for value in values:
        baseline.record_sample(token, {feature: value})


def test_evaluate_empty_for_new_token(store) -> None:
    assert baseline.evaluate("tok-new") == {}
    assert baseline.latest_sample("tok-new") == {}


def test_flags_stay_off_below_min_samples(store) -> None:
    # MIN_SAMPLES-1 history samples + 1 newest => baseline n is too small.
    _feed("tok-a", "pause_per_min", [10.0] * baseline.MIN_SAMPLES)
    results = baseline.evaluate("tok-a")
    info = results["pause_per_min"]
    assert info["z"] is None
    assert info["level"] == "none"
    assert info["n"] == baseline.MIN_SAMPLES - 1


def test_deviation_flag_after_min_samples(store) -> None:
    _feed("tok-b", "pause_per_min", [10.0] * baseline.MIN_SAMPLES)
    baseline.record_sample("tok-b", {"pause_per_min": 40.0})
    results = baseline.evaluate("tok-b")
    info = results["pause_per_min"]
    assert info["z"] is not None and info["z"] > baseline.Z_NOTABLE
    assert info["level"] == "notable"


def test_attention_band_between_thresholds(store) -> None:
    # Build a stable baseline, then a moderate deviation.
    _feed("tok-c", "f0_sd_st", [5.0] * baseline.MIN_SAMPLES)
    stats = baseline.feature_stats("tok-c")
    mean = stats["f0_sd_st"]["mean"]
    sd = stats["f0_sd_st"]["sd"]
    # EWMA variance of identical samples is ~0; use varied samples instead.
    assert sd >= 0
    # Direct z-score unit check with synthetic stats.
    z = baseline.zscore(
        "f0_sd_st",
        mean + 1.7 * max(sd, 0.5),
        {"f0_sd_st": {"n": 5.0, "mean": mean, "sd": max(sd, 0.5)}},
    )
    assert z is not None
    assert baseline.deviation_level(z) == "attention"
    assert baseline.deviation_level(2.5) == "notable"
    assert baseline.deviation_level(0.5) == "none"
    assert baseline.deviation_level(None) == "none"


def test_tokens_are_isolated(store) -> None:
    _feed("tok-x", "blink_rate", [20.0] * (baseline.MIN_SAMPLES + 1))
    _feed("tok-y", "blink_rate", [20.0] * (baseline.MIN_SAMPLES + 1))
    baseline.record_sample("tok-y", {"blink_rate": 60.0})
    assert baseline.evaluate("tok-x")["blink_rate"]["z"] is None or (
        baseline.evaluate("tok-x")["blink_rate"]["level"] == "none"
    )
    assert baseline.evaluate("tok-y")["blink_rate"]["level"] == "notable"


def test_non_numeric_values_ignored(store) -> None:
    baseline.record_sample("tok-d", {"pause_per_min": "lots", "mean_pause_s": None})
    samples = session_store.get_feature_samples("tok-d")
    assert all(
        key == "ts" or isinstance(v, float)
        for sample in samples
        for key, v in sample.items()
    )


def test_feature_labels_resolve_in_all_languages() -> None:
    from modules.i18n import _TEXT

    for feature, label_key in baseline.FEATURE_LABELS.items():
        for lang, strings in _TEXT.items():
            assert label_key in strings, f"{label_key} ({feature}) missing in {lang}"
