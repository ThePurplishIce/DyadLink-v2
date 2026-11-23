from __future__ import annotations

import re
import unicodedata

from modules import nvidia_guard


# The deterministic layer only keeps patterns with near-zero false-positive
# rates in everyday conversation: literal code syntax, "produce this
# artifact" task phrasing, and canonical prompt-injection wordings.
# Everything semantic (is this *really* a task request?) is delegated to
# the NVIDIA guard model — see modules/nvidia_guard.py.
_ZERO_WIDTH = re.compile(r"[​-‏⁠﻿]")
_CODE_SYNTAX = re.compile(
    r"```|<\s*/?\s*(?:code|script)\b|#include|console\s*\.\s*log|"
    r"\b(?:def|class|import|from|print|function|const|let|var)\s+[A-Za-z_$]",
    re.IGNORECASE,
)
# "Produce this artifact" phrasing is only deterministic when it is clearly
# a request: an imperative at the start of the message, an explicit request
# marker (please / can you / help me / 幫我 / 請 …), or a translate-into-X
# construction. "My grandson taught me to write code" is left for the
# semantic layer because it lacks a request marker.
_PRODUCE_TASK = re.compile(
    r"^\s*(?:please\s+)?(?:write|generate|create|produce|draft|compose|"
    r"debug|fix|solve|translate|print|output|include|provide)\b"
    r".{0,40}?\b(?:code|program|script|essay|article|report|paper|"
    r"homework|assignment|equation|proof|translation|dissertation|"
    r"thesis)\b|"
    r"\b(?:please|can\s+you|could\s+you|would\s+you|help\s+me|"
    r"i\s+need\s+you\s+to|i\s+want\s+you\s+to)\b.{0,30}?"
    r"\b(?:write|generate|create|produce|draft|compose|solve|"
    r"translate)\b.{0,20}?\b(?:code|program|script|essay|article|"
    r"report|paper|homework|assignment|equation|proof|translation)\b|"
    r"\btranslate\b.{0,20}\b(?:into|to)\s+"
    r"(?:english|chinese|cantonese|mandarin|japanese|french|german|"
    r"spanish)\b|"
    r"(?:幫我|帮我|幫手|帮手|請|请|麻煩|麻烦|可唔可以|可不可以|能不能|"
    r"能夠|能够).{0,6}(?:寫|写|做|解答|計算|计算|翻譯|翻译|譯|译|"
    r"生成|提供|包含|包括).{0,8}(?:代碼|代码|程式碼|程式|程序|論文|"
    r"论文|作文|報告|报告|作業|作业|功課|功课|方程|題目|题目|"
    r"譯文|译文|翻譯)",
    re.IGNORECASE,
)
_INJECTION = re.compile(
    r"\b(?:ignore|disregard|forget|override|bypass)\s+"
    r"(?:all|previous|prior|above|earlier|your|the\s+above|these)\s+"
    r"(?:instructions?|rules?|restrictions?|prompts?)\b|"
    r"\b(?:prompt\s+injection|system\s+prompt|developer\s+mode)\b|"
    r"(?:忽略|無視|无视|繞過|绕过|解除).{0,16}"
    r"(?:規則|规则|指示|指令|限制|提示詞|提示词)",
    re.IGNORECASE,
)
_OUTPUT_DISALLOWED = re.compile(
    r"```|<\s*/?\s*(?:code|script)\b|#include|console\s*\.\s*log|"
    r"\b(?:python|javascript|typescript|java|c\+\+|c#|sql|html|css|bash|"
    r"programming|coding|compiler|compile|debug(?:ging)?|algorithm)\b|"
    r"\b(?:write|create|build|generate|develop|debug)\s+(?:a\s+)?"
    r"(?:computer\s+)?(?:program|script)\b|"
    r"\b(?:def|class|import|from|print|function|const|let|var)\s+[A-Za-z_$]|"
    r"代碼|代码|編程|编程|寫碼|写码|"
    r"解題|解题|做題|做题|解方程|數學|数学|求答案|"
    r"\b(?:solve|calculate|equation|homework|mathematics|worksheet|"
    r"integral|derivative|algebra|geometry)\b|"
    r"\b(?:prompt\s+injection|system\s+prompt|developer\s+mode)\b",
    re.IGNORECASE,
)


def _normalized(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    return _ZERO_WIDTH.sub("", normalized).strip()


def _deterministic_block(normalized: str) -> bool:
    """Instant refusal for patterns that are never legitimate here."""
    return bool(
        _CODE_SYNTAX.search(normalized)
        or _PRODUCE_TASK.search(normalized)
        or _INJECTION.search(normalized)
    )


def contains_disallowed_content(text: str) -> bool:
    """Catch code, math, and prompt-bypass content in model output."""
    normalized = _normalized(text)
    return bool(_OUTPUT_DISALLOWED.search(normalized))


def is_out_of_scope_request(text: str, has_life_context: bool = False) -> bool:
    """Refuse clearly off-scope user messages.

    Layer 1 (deterministic, always on): code syntax, explicit
    produce-this-artifact requests, canonical prompt-injection phrasings.

    Layer 2 (semantic): a NVIDIA-hosted guard model with custom scope
    categories judges everything else — polite questions, small talk,
    general knowledge questions, everyday requests all pass. Fails open
    when the guard model is unavailable so users are never wrongly blocked.
    """
    normalized = _normalized(text)
    if not normalized:
        return False
    if _deterministic_block(normalized):
        return True
    verdict = nvidia_guard.classify_message(normalized)
    return not verdict.safe
