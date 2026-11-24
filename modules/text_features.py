"""Content-level linguistic markers computed on user utterances.

No external segmenter is used: each CJK character counts as one token and
each Latin/digit run counts as one token. For Chinese text this makes the
diversity metric a *character*-diversity measure rather than a lexical TTR —
labels and report wording describe it as 用字多樣性 / vocabulary diversity,
not a validated clinical index.
"""

import math
import re

# Each CJK char = 1 token; each Latin/digit run = 1 token.
_TOKEN_RE = re.compile(r"[一-鿿]|[A-Za-z']+|\d+(?:\.\d+)?")

# Function words: particles, conjunctions, prepositions, auxiliaries,
# pronouns, hesitation fillers. Conservative list — content verbs/nouns
# are never filtered. (唔 is the Cantonese negation particle, a function
# word — not a filler.)
_ZH_FUNCTION = frozenset(
    "的了是在和就又都也不沒呢吧啊嗎嘛啦呀哦唉麼些這那其之而則與及或因所"
    "但於從向被把給對到再先已曾將會能可要應得著過按比為唔"
    "我你佢他她它您俺咱嗯呃"
)
_EN_FUNCTION = frozenset(
    (
        "the a an and or but if then of in on at for with to from by as is are "
        "was were be been being do does did have has had will would can could "
        "should shall may might must not no yes so too very just only also "
        "about over under between into out up down off again once here there "
        "when where why how what which who whom this that these those i you "
        "he she it we they me him her us them my your his their our mine "
        "yours ours theirs am um uh er erm hmm ah eh"
    ).split()
)

_ZH_PRONOUNS = frozenset("我你佢他她它您俺咱")
_EN_PRONOUNS = frozenset(
    (
        "i you he she it we they me him her us them my your his their our "
        "mine yours ours theirs"
    ).split()
)

_ZH_FILLERS = frozenset("嗯呃")
_EN_FILLERS = frozenset(("um uh er erm hmm ah eh").split())

# Ratios on very short utterances are noise; below this many tokens we only
# record the utterance length itself.
MIN_TOKENS_FOR_RATIOS = 8


def analyze_text(text: str) -> dict[str, float | int] | None:
    """Compute linguistic markers for one user utterance.

    Returns ``None`` when fewer than 3 tokens are present.
    """
    if not isinstance(text, str):
        return None
    raw_tokens = _TOKEN_RE.findall(text.lower())
    if len(raw_tokens) < 3:
        return None

    tokens = len(raw_tokens)
    types = len(set(raw_tokens))
    content = 0
    pronouns = 0
    fillers = 0
    for token in raw_tokens:
        if token in _ZH_FUNCTION or token in _EN_FUNCTION:
            is_function = True
        else:
            is_function = False
        if token in _ZH_PRONOUNS or token in _EN_PRONOUNS:
            pronouns += 1
        if token in _ZH_FILLERS or token in _EN_FILLERS:
            fillers += 1
        if not is_function:
            content += 1

    out: dict[str, float | int] = {
        "utterance_len": float(tokens),
    }
    if tokens >= MIN_TOKENS_FOR_RATIOS:
        out["ttr"] = round(types / tokens, 4)
        out["content_ratio"] = round(content / tokens, 4)
        out["pronoun_ratio"] = round(pronouns / tokens, 4)
        out["filler_ratio"] = round(fillers / tokens, 4)
    return out


# Baseline-tracked feature names produced above (kept separate from the
# vision-bounds table so bounds stay video-only).
TRACKED_FEATURES = frozenset(
    {"utterance_len", "ttr", "content_ratio", "pronoun_ratio", "filler_ratio"}
)


def merge_utterance_metrics(metrics: list[dict]) -> dict[str, float]:
    """Token-count-weighted mean of per-utterance ratio metrics.

    Used to summarize the recent conversation for the informant report —
    a weighted mean keeps a few long utterances from being diluted by many
    short acknowledgements.
    """
    totals: dict[str, float] = {}
    weights: dict[str, float] = {}
    for metric in metrics:
        n = metric.get("utterance_len")
        if not isinstance(n, (int, float)) or n <= 0:
            continue
        for key, value in metric.items():
            if key == "utterance_len" or not isinstance(value, (int, float)):
                continue
            totals[key] = totals.get(key, 0.0) + float(value) * n
            weights[key] = weights.get(key, 0.0) + n
    return {
        key: round(totals[key] / weights[key], 4)
        for key in totals
        if weights[key] > 0
    }
