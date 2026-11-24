"""Shared handling for multimodal capture results.

Used by the chat-page camera dialog. A capture is recorded at most once
per ``capture_id``; numeric features feed the personal baseline and a
short note is appended to the conversation so the event is visible in
history.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from modules import baseline, session_store
from modules.browser_history import current_token
from modules.i18n import tr
from modules.vision_features import baseline_features, sanitize_capture

_MAX_CAPTURES = 8
_MAX_NOTE_SCAN = 6
_LOADED_KEY = "vision_captures_loaded"


def _append_chat_note(cleaned: dict[str, Any]) -> None:
    capture_id = cleaned.get("capture_id")
    messages = st.session_state.setdefault("chat_messages", [])
    for message in reversed(messages[-_MAX_NOTE_SCAN:]):
        if message.get("vision_capture_id") == capture_id:
            return
    messages.append(
        {
            "role": "assistant",
            "content": tr("vision_chat_note"),
            "mode": "vision-note",
            "ui_only": True,
            "vision_capture_id": capture_id,
        }
    )


def record_result(res: Any) -> dict[str, Any] | None:
    """Consume a component ``ComponentResult``.

    Returns the cleaned capture dict when a *new* capture was recorded,
    ``None`` otherwise (no payload, error payload, or duplicate id).
    Error payloads surface through ``st.session_state["vision_error"]``.
    """
    payload = res.get("result") if hasattr(res, "get") else None
    if payload is None:
        return None
    cleaned = sanitize_capture(payload)
    if not cleaned:
        return None
    if cleaned.get("error"):
        st.session_state["vision_error"] = cleaned["error"]
        return None

    capture_id = cleaned.get("capture_id")
    if capture_id is None or capture_id == st.session_state.get(
        "vision_last_capture_id"
    ):
        return None
    st.session_state["vision_last_capture_id"] = capture_id
    st.session_state.pop("vision_error", None)

    captures = st.session_state.setdefault("vision_captures", [])
    captures.append(cleaned)
    del captures[:-_MAX_CAPTURES]

    token = current_token()
    if token:
        session_store.save_captures(token, captures)
        features = baseline_features(cleaned)
        if features:
            baseline.record_sample(token, features)

    _append_chat_note(cleaned)
    return cleaned


def latest_capture() -> dict[str, Any] | None:
    captures = st.session_state.get("vision_captures") or []
    return captures[-1] if captures else None


def sync_captures() -> None:
    """Persist captures into the token store; restore them once per session.

    Call once per page run (home main). Captures are aggregate numbers only
    — persisting them is consistent with the longitudinal personal baseline,
    which already survives refresh by design.
    """
    token = current_token()
    if not token:
        return
    captures = st.session_state.setdefault("vision_captures", [])
    if not st.session_state.get(_LOADED_KEY):
        st.session_state[_LOADED_KEY] = True
        if not captures:
            for item in session_store.get_captures(token):
                cleaned = sanitize_capture(item)
                if cleaned:
                    captures.append(cleaned)
            del captures[:-_MAX_CAPTURES]
            return
    if session_store.get_captures(token) != captures[-_MAX_CAPTURES:]:
        session_store.save_captures(token, captures[-_MAX_CAPTURES:])
