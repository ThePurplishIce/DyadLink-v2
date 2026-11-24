from __future__ import annotations

import base64
from datetime import date
from html import escape
import time

import streamlit as st
from streamlit.errors import StreamlitInvalidLayoutContextError

from modules import baseline
from modules.audio_features import analyze_wav
from modules.browser_history import current_token, sync_browser_history
from modules.consent import CONSENT_SESSION_KEY
from modules.deepseek_service import (
    DeepSeekBudgetExceeded,
    DeepSeekServiceError,
    generate_greeting,
    generate_informant_summary,
    generate_reply,
    generate_reply_stream,
    get_api_key,
    transcribe_audio,
)
from modules.emotion import is_emotionally_distressed
from modules.i18n import (
    get_response_language,
    get_speech_language,
    get_tts_voice,
    localize_service_error,
    tr,
)
from modules.crisis import detects_crisis
from modules.medical_safety import requires_medical_disclaimer
from modules.memory_extractor import extract_memory_from_input
from modules.observation_log import (
    initialize_observation_state,
    render_observation_panel,
    reset_observation_state,
    sensor_context_text,
)
from modules.scope_guard import contains_disallowed_content, is_out_of_scope_request
from modules.text_features import TRACKED_FEATURES, analyze_text
from modules.ui import render_brand
from modules.vision_component import mount_vision_observer
from modules.vision_session import record_result, sync_captures
from modules.tts_service import TTSError, autoplay_audio_html, synthesize_speech


MOCK_QUESTION_KEYS = (
    "mock_q1",
    "mock_q2",
    "mock_q3",
    "mock_q4",
)
SUMMARY_BUTTON_AFTER_TURNS = 4
INFORMANT_REPORT_INTERVAL = 5

# Stretch the camera button to the same height as the mic audio input in
# the bottom bar. The keyed container gives CSS a stable hook; flex rules
# cascade down through .stElementContainer wrappers to the <button> so it
# fills the (taller) audio column.
# The mic audio input's content height is the theme constant
# sizes.largestElementHeight = 4.25rem (68px). Set the camera button to
# exactly that height and centre it — deterministic instead of chasing
# Streamlit's shifting wrapper divs.
_BOTTOM_BAR_CSS = """
<style>
.stBottom .stHorizontalBlock,
[data-testid="stBottom"] .stHorizontalBlock { align-items: stretch !important; }
.stColumn:has(.st-key-cam_button_fill) {
  display: flex !important; flex-direction: column !important;
  justify-content: center !important;
}
.st-key-cam_button_fill .stButton button {
  height: 4.25rem !important; width: 100% !important;
}
</style>
"""


def _initialize_state() -> None:
    st.session_state.setdefault("chat_messages", [])
    st.session_state.setdefault("user_turns", 0)
    st.session_state["chat_completed"] = False
    st.session_state.setdefault("pending_transcript", None)
    st.session_state.setdefault("pending_speech_ms", None)
    st.session_state.setdefault("empathy_turns_left", 0)
    st.session_state.setdefault("input_error", None)
    st.session_state.setdefault("transcript_revision", 0)
    st.session_state.setdefault("deepseek_usage_tokens", 0)
    st.session_state.setdefault("asr_duration_ms", 0)
    st.session_state.setdefault("audio_input_revision", 0)
    st.session_state.setdefault("tts_autoplay_pending", None)
    st.session_state.setdefault("informant_report", None)
    st.session_state.setdefault("informant_report_turn", 0)
    st.session_state.setdefault("call_reply", None)
    initialize_observation_state()


def _reset_conversation() -> None:
    st.session_state["chat_messages"] = []
    st.session_state["user_turns"] = 0
    st.session_state["chat_completed"] = False
    st.session_state["pending_transcript"] = None
    st.session_state["pending_speech_ms"] = None
    st.session_state["empathy_turns_left"] = 0
    st.session_state["input_error"] = None
    st.session_state["transcript_revision"] += 1
    st.session_state["deepseek_usage_tokens"] = 0
    st.session_state["asr_duration_ms"] = 0
    st.session_state["audio_input_revision"] += 1
    st.session_state.pop("tts_audio_cache", None)
    st.session_state["tts_autoplay_pending"] = None
    st.session_state["informant_report"] = None
    st.session_state["informant_report_turn"] = 0
    st.session_state.pop("call_reply", None)
    reset_observation_state()


def _clear_inactive_transcript_fields() -> None:
    if st.session_state["pending_transcript"] is None:
        for key in list(st.session_state.keys()):
            if key.startswith("pending_transcript_edit_"):
                st.session_state.pop(key, None)


def _synchronize_mode(live_enabled: bool) -> None:
    previous_mode = st.session_state.get("active_api_mode")
    current_mode = "deepseek" if live_enabled else "local"
    if previous_mode is None:
        st.session_state["active_api_mode"] = current_mode
    elif previous_mode != current_mode:
        _reset_conversation()
        st.session_state["active_api_mode"] = current_mode
        st.session_state["mode_notice"] = tr("mode_changed")


def _mock_reply(user_turns: int) -> str:
    question_key = MOCK_QUESTION_KEYS[(user_turns - 1) % len(MOCK_QUESTION_KEYS)]
    return f"{tr('mock_thanks')} {tr(question_key)}"


def _record_observation(content: str) -> None:
    try:
        note = extract_memory_from_input(content)
    except Exception:
        return
    if isinstance(note, dict) and note.get("summary"):
        st.session_state.setdefault("auto_memory_log", []).append(note)


def _attach_text_metrics(content: str) -> None:
    features = analyze_text(content)
    if not features:
        return
    st.session_state["chat_messages"][-1]["textual"] = features
    token = current_token()
    if token:
        baseline.record_sample(
            token,
            {
                key: value
                for key, value in features.items()
                if key in TRACKED_FEATURES
            },
        )


def _append_user_message(
    content: str,
    api_key: str | None,
    live_enabled: bool,
    defer_reply: bool = True,
) -> None:
    clean_content = content.strip()
    if not clean_content:
        return
    st.session_state["session_had_input"] = True

    if detects_crisis(clean_content):
        st.session_state["crisis_flagged"] = True
        st.session_state["chat_messages"].append(
            {"role": "user", "content": clean_content, "scope_blocked": True}
        )
        st.session_state["chat_messages"].append(
            {
                "role": "assistant",
                "content": tr("crisis_response"),
                "mode": "crisis",
                "scope_blocked": True,
                "speech_language": get_speech_language(),
            }
        )
        st.session_state["tts_autoplay_pending"] = (
            len(st.session_state["chat_messages"]) - 1
        )
        return

    if requires_medical_disclaimer(clean_content):
        st.session_state["chat_messages"].append(
            {
                "role": "user",
                "content": clean_content,
                "scope_blocked": True,
                "medical_guarded": True,
            }
        )
        st.session_state["user_turns"] += 1
        st.session_state["chat_messages"].append(
            {
                "role": "assistant",
                "content": tr("medical_disclaimer"),
                "mode": "medical-guard",
                "scope_blocked": True,
                "speech_language": get_speech_language(),
            }
        )
        st.session_state["tts_autoplay_pending"] = (
            len(st.session_state["chat_messages"]) - 1
        )
        return

    has_life_context = any(
        message.get("role") == "user" and not message.get("scope_blocked")
        for message in st.session_state["chat_messages"]
    )
    if is_out_of_scope_request(clean_content, has_life_context):
        st.session_state["chat_messages"].append(
            {"role": "user", "content": clean_content, "scope_blocked": True}
        )
        st.session_state["chat_messages"].append(
            {
                "role": "assistant",
                "content": tr("scope_refusal"),
                "mode": "scope-guard",
                "scope_blocked": True,
                "speech_language": get_speech_language(),
            }
        )
        st.session_state["tts_autoplay_pending"] = (
            len(st.session_state["chat_messages"]) - 1
        )
        return

    st.session_state["chat_messages"].append(
        {"role": "user", "content": clean_content}
    )
    st.session_state["user_turns"] += 1
    user_turns = st.session_state["user_turns"]
    _record_observation(clean_content)
    _attach_text_metrics(clean_content)

    if is_emotionally_distressed(clean_content):
        st.session_state["empathy_turns_left"] = 3
    elif st.session_state["empathy_turns_left"] > 0:
        st.session_state["empathy_turns_left"] -= 1
    empathy_active = st.session_state["empathy_turns_left"] > 0

    if live_enabled and api_key and defer_reply:
        # Typed path: the reply streams during render so the user sees it
        # type out; _complete_pending_reply finishes the turn.
        st.session_state["pending_reply"] = {
            "user_turns": user_turns,
            "empathy": empathy_active,
        }
        st.session_state["chat_completed"] = False
        return

    if live_enabled and api_key:
        try:
            result = generate_reply(
                api_key,
                st.session_state["chat_messages"],
                user_turns,
                response_language=get_response_language(),
                empathy=empathy_active,
            )
            reply = result.content
            mode = "deepseek"
            if result.total_tokens is not None:
                st.session_state["deepseek_usage_tokens"] += result.total_tokens
            if contains_disallowed_content(reply):
                reply = tr("scope_refusal")
                mode = "scope-guard"
        except DeepSeekBudgetExceeded:
            st.session_state["input_error"] = tr("budget_fallback")
            reply = _mock_reply(user_turns)
            mode = "local-fallback"
        except DeepSeekServiceError as error:
            st.session_state["input_error"] = localize_service_error(str(error))
            reply = _mock_reply(user_turns)
            mode = "local-fallback"
    else:
        reply = _mock_reply(user_turns)
        mode = "local-demo"

    st.session_state["chat_messages"].append(
        {
            "role": "assistant",
            "content": reply,
            "mode": mode,
            "speech_language": get_speech_language(),
        }
    )
    st.session_state["tts_autoplay_pending"] = (
        len(st.session_state["chat_messages"]) - 1
    )
    st.session_state["chat_completed"] = False
    _maybe_generate_informant_report(api_key, live_enabled, user_turns)


def _complete_pending_reply(
    container: "st.delta_generator.DeltaGenerator",
    api_key: str | None,
    live_enabled: bool,
) -> None:
    """Render + finish a deferred reply (``pending_reply`` session key)."""
    pending = st.session_state.pop("pending_reply", None)
    if not pending or not (live_enabled and api_key):
        if pending:
            # Consent/key vanished between submit and render — fall back.
            reply = _mock_reply(pending["user_turns"])
            st.session_state["chat_messages"].append(
                {
                    "role": "assistant",
                    "content": reply,
                    "mode": "local-demo",
                    "speech_language": get_speech_language(),
                }
            )
            st.session_state["tts_autoplay_pending"] = (
                len(st.session_state["chat_messages"]) - 1
            )
        return

    user_turns = pending["user_turns"]
    meta: dict[str, int] = {}
    pieces: list[str] = []

    def _accumulate(stream):
        for piece in stream:
            pieces.append(piece)
            yield piece

    mode = "deepseek"
    with container:
        with st.chat_message("assistant"):
            try:
                stream = generate_reply_stream(
                    api_key,
                    st.session_state["chat_messages"],
                    user_turns,
                    response_language=get_response_language(),
                    empathy=pending["empathy"],
                    meta=meta,
                )
                st.write_stream(_accumulate(stream))
            except DeepSeekBudgetExceeded:
                st.session_state["input_error"] = tr("budget_fallback")
                mode = "local-fallback"
            except DeepSeekServiceError as error:
                st.session_state["input_error"] = localize_service_error(
                    str(error)
                )
                mode = "local-fallback"

    reply = "".join(pieces).strip()
    if mode == "local-fallback" and not reply:
        reply = _mock_reply(user_turns)
    if not reply:
        st.session_state["input_error"] = tr("call_error")
        reply = _mock_reply(user_turns)
        mode = "local-fallback"
    if mode == "deepseek" and contains_disallowed_content(reply):
        # Streamed text can't be unshown, but it never reaches history.
        reply = tr("scope_refusal")
        mode = "scope-guard"
    if isinstance(meta.get("total_tokens"), int):
        st.session_state["deepseek_usage_tokens"] += meta["total_tokens"]

    st.session_state["chat_messages"].append(
        {
            "role": "assistant",
            "content": reply,
            "mode": mode,
            "speech_language": get_speech_language(),
        }
    )
    st.session_state["tts_autoplay_pending"] = (
        len(st.session_state["chat_messages"]) - 1
    )
    st.session_state["chat_completed"] = False
    _maybe_generate_informant_report(api_key, live_enabled, user_turns)
    st.rerun()


def _generate_informant_report(
    api_key: str | None,
    live_enabled: bool,
    user_turns: int,
    manual: bool = False,
) -> bool:
    if not live_enabled or not api_key:
        if manual:
            st.session_state["input_error"] = tr("summary_requires_ai")
        return False

    try:
        with st.spinner(tr("summarizing")):
            report = generate_informant_summary(
                api_key,
                st.session_state["chat_messages"],
                response_language=get_response_language(),
                sensor_context=sensor_context_text(),
            )
    except DeepSeekBudgetExceeded:
        st.session_state["input_error"] = tr("budget_fallback")
        return False
    except DeepSeekServiceError as error:
        st.session_state["input_error"] = localize_service_error(str(error))
        return False

    total_tokens = report.get("total_tokens")
    if type(total_tokens) is int and total_tokens >= 0:
        st.session_state["deepseek_usage_tokens"] += total_tokens

    st.session_state["informant_report"] = {
        "turn": user_turns,
        "generated_at": time.time(),
        **report,
    }
    st.session_state["informant_report_turn"] = user_turns
    return True


def _maybe_generate_informant_report(
    api_key: str | None,
    live_enabled: bool,
    user_turns: int,
) -> None:
    if not live_enabled or not api_key:
        return
    if user_turns < INFORMANT_REPORT_INTERVAL:
        return
    if user_turns % INFORMANT_REPORT_INTERVAL != 0:
        return
    if st.session_state.get("informant_report_turn") == user_turns:
        return
    _generate_informant_report(api_key, live_enabled, user_turns, manual=False)


def _maybe_greet(api_key: str | None, live_enabled: bool) -> None:
    """One welcome-back greeting per session when prior history exists.

    Only fires before the session's first input — a greeting appended
    after a fresh turn would land below the user's own message.
    """
    if st.session_state.get("greeting_done"):
        return
    if st.session_state.get("session_had_input"):
        st.session_state["greeting_done"] = True
        return
    messages = st.session_state["chat_messages"]
    has_user = any(
        message.get("role") == "user" and not message.get("scope_blocked")
        for message in messages
    )
    if not has_user:
        return
    st.session_state["greeting_done"] = True
    greeting = None
    if live_enabled and api_key:
        try:
            greeting = generate_greeting(
                api_key,
                messages,
                response_language=get_response_language(),
            )
        except DeepSeekServiceError:
            greeting = None
    if not greeting:
        greeting = tr("greeting_fallback")
    st.session_state["chat_messages"].append(
        {
            "role": "assistant",
            "content": greeting,
            "mode": "greeting",
            "ui_only": True,
            "speech_language": get_speech_language(),
        }
    )


def _render_topic_card(api_key: str | None, live_enabled: bool) -> None:
    """Rotating daily prompt for an empty conversation."""
    if st.session_state["chat_messages"]:
        return
    topics = tr("daily_topics")
    if not isinstance(topics, list) or not topics:
        return
    index = date.today().toordinal() % len(topics)
    with st.container(border=True):
        st.markdown(
            f"**{escape(tr('topic_card_title'))}** — {escape(topics[index])}"
        )
        if st.button(
            tr("topic_start"),
            key="topic_card_start",
            type="primary",
            width="content",
        ):
            _append_user_message(topics[index], api_key, live_enabled)
            st.rerun()


def _render_conversation(api_key: str | None, live_enabled: bool) -> None:
    messages = st.session_state["chat_messages"]
    if not messages:
        return

    highlight_quote = st.session_state.get("highlight_quote") or ""
    for message_index, message in enumerate(messages):
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and message.get("mode") == "crisis":
                st.markdown(
                    f'<div class="crisis-card">{escape(message["content"])}</div>',
                    unsafe_allow_html=True,
                )
                st.caption(tr("crisis_badge"))
                _autoplay_assistant_message(message, message_index)
                continue
            if (
                message["role"] == "assistant"
                and message.get("mode") == "scope-guard"
            ):
                st.markdown(
                    f'<div class="scope-guide-card">{escape(message["content"])}</div>',
                    unsafe_allow_html=True,
                )
                st.caption(tr("scope_badge"))
                _autoplay_assistant_message(message, message_index)
                topic_keys = (
                    ("scope_topic_mood", "scope_prompt_mood"),
                    ("scope_topic_routine", "scope_prompt_routine"),
                    ("scope_topic_memory", "scope_prompt_memory"),
                )
                for label_key, prompt_key in topic_keys:
                    if st.button(
                        tr(label_key),
                        key=f"scope_topic_{message_index}_{label_key}",
                        width="stretch",
                    ):
                        _append_user_message(tr(prompt_key), api_key, live_enabled)
                        st.rerun()
                continue

            if (
                highlight_quote
                and message["role"] == "user"
                and isinstance(message.get("content"), str)
                and highlight_quote[:60] in message["content"]
            ):
                st.markdown(
                    f'<div class="msg-highlight">{escape(message["content"])}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(message["content"])
            if message["role"] == "assistant":
                if message.get("mode") == "deepseek":
                    st.caption(tr("ai_badge"))
                elif message.get("mode") in {"local-demo", "local-fallback"}:
                    st.caption(tr("local_badge"))
                elif message.get("mode") == "vision-note":
                    st.caption(tr("vision_note_badge"))
                elif message.get("mode") == "greeting":
                    st.caption(tr("greeting_badge"))
                _autoplay_assistant_message(message, message_index)


def _autoplay_assistant_message(
    message: dict[str, str],
    message_index: int,
) -> None:
    if st.session_state.get("tts_autoplay_pending") != message_index:
        return

    st.session_state["tts_autoplay_pending"] = None
    language = message.get("speech_language", get_speech_language())
    cache = st.session_state.setdefault("tts_audio_cache", {})
    cache_key = f"{message_index}:{language}"
    try:
        audio = cache.get(cache_key)
        if audio is None:
            with st.spinner(tr("speaking")):
                audio = synthesize_speech(
                    message["content"],
                    get_tts_voice(language),
                )
            cache[cache_key] = audio
        st.markdown(autoplay_audio_html(audio), unsafe_allow_html=True)
    except TTSError:
        st.warning(tr("tts_error"))


def _render_transcript_review() -> None:
    pending_transcript = st.session_state.get("pending_transcript")
    if pending_transcript is None:
        return
    transcript_key = (
        f"pending_transcript_edit_{st.session_state['transcript_revision']}"
    )

    with st.container(border=True):
        st.markdown(f"### {tr('transcript_title')}")
        st.caption(tr("transcript_help"))
        with st.form("voice_transcript_review"):
            edited_transcript = st.text_area(
                tr("transcript_label"),
                value=pending_transcript,
                height=130,
                max_chars=1500,
                key=transcript_key,
            )
            send_column, cancel_column = st.columns(2, gap="small")
            with send_column:
                send_transcript = st.form_submit_button(
                    tr("send"),
                    type="primary",
                    width="stretch",
                )
            with cancel_column:
                cancel_transcript = st.form_submit_button(
                    tr("cancel"),
                    width="stretch",
                )

    if send_transcript:
        api_key = get_api_key()
        live_enabled = bool(api_key and st.session_state.get(CONSENT_SESSION_KEY))
        st.session_state["pending_transcript"] = None
        speech_ms = st.session_state.pop("pending_speech_ms", None)
        _append_user_message(edited_transcript, api_key, live_enabled)
        _attach_speech_metric(edited_transcript, speech_ms)
        st.rerun()
    if cancel_transcript:
        st.session_state["pending_transcript"] = None
        st.session_state["pending_speech_ms"] = None
        st.session_state.pop("pending_vocal", None)
        st.session_state["transcript_revision"] += 1
        st.rerun()


_VOCAL_BASELINE_KEYS = (
    "pause_per_min",
    "mean_pause_s",
    "f0_sd_st",
    "silence_ratio",
)


def _attach_speech_metric(
    transcript: str,
    speech_ms: int | float | None,
) -> None:
    if not isinstance(speech_ms, (int, float)) or speech_ms <= 0:
        return
    chars = len("".join(transcript.split()))
    if not chars:
        return
    cpm = round(chars / (speech_ms / 60000.0), 1)
    for message in reversed(st.session_state["chat_messages"]):
        if message.get("role") == "user" and not message.get("scope_blocked"):
            message["speech_cpm"] = cpm
            message["speech_ms"] = round(speech_ms)
            vocal = st.session_state.pop("pending_vocal", None)
            if isinstance(vocal, dict):
                message["vocal"] = vocal
                token = current_token()
                if token:
                    features = {
                        key: vocal[key]
                        for key in _VOCAL_BASELINE_KEYS
                        if isinstance(vocal.get(key), (int, float))
                    }
                    if features:
                        baseline.record_sample(token, features)
            return


def _close_vision_frame(*_args: object, **_kwargs: object) -> None:
    st.session_state["vision_frame_open"] = False
    st.session_state.pop("call_reply", None)
    st.session_state.pop("vision_dialog_capture_done", None)


def _call_reply_payload(
    reply_id: object,
    text: str,
    audio_bytes: bytes | None = None,
) -> dict[str, object]:
    return {
        "id": reply_id,
        "text": text,
        "audio_b64": (
            base64.b64encode(audio_bytes).decode("ascii")
            if audio_bytes
            else None
        ),
    }


def _handle_call_utterance(
    res: object,
    api_key: str | None,
    live_enabled: bool,
) -> None:
    """Turn a finished call-mode utterance into a chat reply.

    Runs inside the camera dialog fragment. The WAV clip is transcribed by
    DeepSeek ASR, routed through the same guarded ``_append_user_message``
    pipeline as typed input (crisis/medical/scope guards included), and the
    reply is synthesised to speech and handed back to the component via
    ``data.call_reply``. A fragment rerun pushes the new data to the
    already-mounted component so it can play the reply.
    """
    utterance = res.get("utterance") if hasattr(res, "get") else None
    if not isinstance(utterance, dict):
        return
    utterance_id = utterance.get("ts")
    if (
        utterance_id is None
        or utterance_id == st.session_state.get("call_last_utterance_ts")
    ):
        return
    st.session_state["call_last_utterance_ts"] = utterance_id

    pcm_b64 = utterance.get("pcm")
    if not isinstance(pcm_b64, str) or not pcm_b64:
        return
    try:
        audio_bytes = base64.b64decode(pcm_b64, validate=True)
    except (ValueError, TypeError):
        return
    if not audio_bytes:
        return

    def _reply(text: str, audio: bytes | None = None) -> None:
        st.session_state["call_reply"] = _call_reply_payload(
            utterance_id, text, audio
        )
        try:
            st.rerun(scope="fragment")
        except StreamlitInvalidLayoutContextError:
            # The dialog body can also run inside a full-app rerun (e.g. a
            # chat message sent mid-call); fall back to an app-scoped rerun.
            st.rerun()

    if not live_enabled or not api_key:
        _reply(tr("call_needs_key"))
        return

    try:
        with st.spinner(tr("call_thinking")):
            transcription = transcribe_audio(audio_bytes, api_key)
    except DeepSeekBudgetExceeded:
        _reply(tr("budget_fallback"))
        return
    except DeepSeekServiceError as error:
        _reply(localize_service_error(str(error)))
        return

    st.session_state["asr_duration_ms"] += transcription.duration_ms
    st.session_state["pending_speech_ms"] = transcription.duration_ms
    try:
        st.session_state["pending_vocal"] = analyze_wav(audio_bytes)
    except Exception:
        st.session_state.pop("pending_vocal", None)

    # Call mode needs the reply synchronously (it feeds TTS right away),
    # so this path skips the typed-chat streaming deferral.
    _append_user_message(
        transcription.text, api_key, live_enabled, defer_reply=False
    )
    _attach_speech_metric(transcription.text, transcription.duration_ms)

    reply_message = st.session_state["chat_messages"][-1]
    if reply_message.get("role") != "assistant":
        _reply(tr("call_error"))
        return
    # The reply audio plays inside the call frame, so suppress the
    # page-level autoplay that _append_user_message queued.
    st.session_state["tts_autoplay_pending"] = None
    reply_text = reply_message.get("content") or ""
    reply_lang = reply_message.get("speech_language") or get_speech_language()
    reply_audio = None
    try:
        with st.spinner(tr("speaking")):
            reply_audio = synthesize_speech(
                reply_text,
                get_tts_voice(reply_lang),
            )
    except TTSError:
        pass
    _reply(reply_text, reply_audio)


def _vision_dialog_body() -> None:
    """Chat-page video-call frame: live preview + waveform + captions.

    Runs inside a ``st.dialog`` fragment, so component state updates only
    rerun this body — captures and call turns are handled without touching
    the page.
    """
    st.caption(tr("call_dialog_hint"))
    vision_error = st.session_state.pop("vision_error", None)
    if vision_error == "too_short":
        st.warning(tr("mm_error_too_short"))
    elif vision_error:
        st.warning(tr("mm_error_generic"))
    api_key = get_api_key()
    live_enabled = bool(
        api_key and st.session_state.get(CONSENT_SESSION_KEY, False)
    )
    res = mount_vision_observer(
        autostart=True,
        key="vision_dialog_capture",
        mode="call",
        call_reply=st.session_state.get("call_reply"),
    )
    capture = record_result(res)
    if capture is not None:
        # record_result only reports a capture once; remember it so the
        # confirmation + Done button survive later fragment reruns.
        st.session_state["vision_dialog_capture_done"] = True
    if st.session_state.get("vision_dialog_capture_done"):
        st.success(tr("vision_recorded_caption"))
        if st.button(
            tr("vision_done_close"),
            key="vision_dialog_done",
            type="primary",
            width="stretch",
        ):
            st.session_state["vision_frame_open"] = False
            st.session_state.pop("call_reply", None)
            st.session_state.pop("vision_dialog_capture_done", None)
            st.rerun()
    _handle_call_utterance(res, api_key, live_enabled)


def main() -> None:
    _initialize_state()
    sync_browser_history()
    sync_captures()
    api_key = get_api_key()
    consent = bool(st.session_state.get(CONSENT_SESSION_KEY, False))
    live_enabled = bool(api_key and consent)
    _maybe_greet(api_key, live_enabled)
    render_brand(tr("home_brand"))
    st.markdown(_BOTTOM_BAR_CSS, unsafe_allow_html=True)

    st.markdown(
        f'<h1 class="page-title">{tr("home_title")}</h1>'
        f'<p class="conversation-copy">{tr("home_intro")}</p>'
        f'<p class="conversation-copy">{tr("conversation_purpose")}</p>',
        unsafe_allow_html=True,
    )
    if not api_key:
        st.info(f"{tr('local_mode')} {tr('local_voice_off')}")

    _synchronize_mode(live_enabled)
    _clear_inactive_transcript_fields()
    mode_notice = st.session_state.pop("mode_notice", None)
    if mode_notice:
        st.info(mode_notice)
    if st.session_state["input_error"]:
        st.warning(st.session_state.pop("input_error"))
    if st.session_state.get("empathy_turns_left", 0) > 0:
        st.markdown(
            "<style>.stApp { background: #fdf7ee !important; }</style>",
            unsafe_allow_html=True,
        )
    if st.session_state.get("crisis_flagged"):
        st.markdown(
            f'<div class="crisis-banner">{escape(tr("crisis_banner"))}</div>',
            unsafe_allow_html=True,
        )

    conversation = st.container()
    _render_topic_card(api_key, live_enabled)
    with conversation:
        _render_conversation(api_key, live_enabled)
    _render_transcript_review()

    if (
        st.session_state["user_turns"] >= SUMMARY_BUTTON_AFTER_TURNS
        and st.session_state["pending_transcript"] is None
    ):
        if st.button(
            tr("summary_button"),
            type="primary",
            width="stretch",
            key="summary_button",
        ):
            _generate_informant_report(
                api_key,
                live_enabled,
                st.session_state["user_turns"],
                manual=True,
            )
            st.rerun()

    drawer_open = st.session_state.get("obs_drawer_open", False)
    with st.container(key="obs_drawer_btn"):
        if st.button(
            tr("observation_title"),
            key="obs_drawer_button",
            width="content",
        ):
            st.session_state["obs_drawer_open"] = not drawer_open
            st.rerun()
    if drawer_open:
        st.markdown(
            '<span class="obs-open-marker" aria-hidden="true"></span>',
            unsafe_allow_html=True,
        )
    with st.container(key="observation_drawer"):
        render_observation_panel()

    audio_disabled = (
        not live_enabled
        or st.session_state["pending_transcript"] is not None
    )
    with st.bottom:
        camera_column, mic_column = st.columns([1, 9], gap="small")
        with camera_column:
            with st.container(key="cam_button_fill"):
                camera_clicked = st.button(
                    tr("cam_button"),
                    key="vision_frame_toggle",
                    icon=":material/videocam:",
                    width="stretch",
                )
        with mic_column:
            recording = st.audio_input(
                label=tr("mic_speak" if live_enabled else "mic_disabled"),
                sample_rate=16000,
                key=f"voice_recording_{st.session_state['audio_input_revision']}",
                disabled=audio_disabled,
                label_visibility="collapsed",
                width="stretch",
            )
    if camera_clicked:
        st.session_state["vision_frame_open"] = not st.session_state.get(
            "vision_frame_open", False
        )
        if st.session_state["vision_frame_open"]:
            # Fresh call session — drop any stale reply/capture flag so an
            # old reply never replays inside the new frame.
            st.session_state.pop("call_reply", None)
            st.session_state.pop("vision_dialog_capture_done", None)

    vision_dialog = st.dialog(
        tr("vision_dialog_title"),
        width="large",
        dismissible=True,
        icon=":material/videocam:",
        on_dismiss=_close_vision_frame,
    )(_vision_dialog_body)
    if st.session_state.get("vision_frame_open"):
        vision_dialog()

    if recording is not None:
        st.session_state["audio_input_revision"] += 1
        if live_enabled and api_key:
            try:
                with st.spinner(tr("transcribing")):
                    transcription = transcribe_audio(recording.getvalue(), api_key)
                st.session_state["asr_duration_ms"] += transcription.duration_ms
                st.session_state["pending_speech_ms"] = transcription.duration_ms
                st.session_state["pending_vocal"] = analyze_wav(
                    recording.getvalue()
                )
                st.session_state["transcript_revision"] += 1
                st.session_state["pending_transcript"] = transcription.text
            except DeepSeekBudgetExceeded:
                st.session_state["input_error"] = tr("budget_fallback")
            except DeepSeekServiceError as error:
                st.session_state["input_error"] = localize_service_error(str(error))
        else:
            st.session_state["input_error"] = tr("local_voice_off")
        st.rerun()

    submission = st.chat_input(
        tr("chat_placeholder"),
        key="conversation_input",
        max_chars=1500,
        disabled=st.session_state["pending_transcript"] is not None,
    )

    if submission and submission.strip():
        _append_user_message(submission, api_key, live_enabled)
        st.rerun()

    _complete_pending_reply(conversation, api_key, live_enabled)


if __name__ == "__main__":
    main()
