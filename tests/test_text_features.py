from modules.text_features import analyze_text, merge_utterance_metrics


def test_cjk_utterance_metrics() -> None:
    # 9 CJK chars: 我(pronoun/function) + 8 content chars.
    result = analyze_text("我今日去街市買咗菜")
    assert result is not None
    assert result["utterance_len"] == 9
    assert result["ttr"] == 1.0
    assert result["pronoun_ratio"] == round(1 / 9, 4)
    assert result["content_ratio"] == round(8 / 9, 4)
    assert "filler_ratio" in result


def test_english_utterance_metrics() -> None:
    result = analyze_text("I went to the market and bought some vegetables")
    assert result is not None
    assert result["utterance_len"] == 9
    assert result["pronoun_ratio"] == round(1 / 9, 4)
    # Content words: went, market, bought, some, vegetables.
    assert result["content_ratio"] == round(5 / 9, 4)


def test_fillers_counted() -> None:
    result = analyze_text("嗯 um 我想想今天吃什麼好東西啦")
    assert result is not None
    assert result["filler_ratio"] > 0
    # Fillers are function words, not content.
    assert result["content_ratio"] < 1.0


def test_short_utterances() -> None:
    assert analyze_text("") is None
    assert analyze_text("好") is None
    assert analyze_text("hi ok") is None
    # Below the ratio threshold only length is reported.
    result = analyze_text("我哋去街")
    assert result is not None
    assert result["utterance_len"] == 4
    assert "ttr" not in result


def test_merge_weighted_mean() -> None:
    merged = merge_utterance_metrics(
        [
            {"utterance_len": 10, "ttr": 0.8, "content_ratio": 0.5},
            {"utterance_len": 30, "ttr": 0.5, "content_ratio": 0.7},
        ]
    )
    assert merged["ttr"] == round((0.8 * 10 + 0.5 * 30) / 40, 4)
    assert merged["content_ratio"] == round((0.5 * 10 + 0.7 * 30) / 40, 4)
    assert "utterance_len" not in merged


def test_merge_ignores_malformed() -> None:
    assert merge_utterance_metrics([]) == {}
    assert merge_utterance_metrics([{"ttr": 0.9}, {"utterance_len": 0}]) == {}
