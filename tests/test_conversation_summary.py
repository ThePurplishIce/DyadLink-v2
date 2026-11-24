from pathlib import Path
from unittest.mock import patch

import httpx
from streamlit.testing.v1 import AppTest

from modules.deepseek_service import (
    AssistantReply,
    generate_greeting,
    generate_informant_summary,
    generate_reply,
    generate_reply_stream,
    generate_summary,
)


ROOT = Path(__file__).resolve().parents[1]


def _completion_response(url: str, content: str, total_tokens: int = 0) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": content}}],
            "usage": {"total_tokens": total_tokens},
        },
        request=httpx.Request("POST", url),
    )


class _FakeStream:
    """Minimal stand-in for ``httpx.stream``'s response context manager."""

    def __init__(self, lines: list[str], status_code: int = 200) -> None:
        self._lines = lines
        self.status_code = status_code

    def __enter__(self) -> "_FakeStream":
        return self

    def __exit__(self, *args: object) -> bool:
        return False

    def iter_lines(self):
        yield from self._lines

    def read(self) -> None:
        return None


def test_reply_stream_yields_deltas_and_usage() -> None:
    lines = [
        'data: {"choices":[{"delta":{"content":"你好"}}]}',
        'data: {"choices":[{"delta":{"content":"呀"}}],'
        '"usage":{"total_tokens":42}}',
        "data: [DONE]",
    ]
    meta: dict[str, int] = {}
    messages = [{"role": "user", "content": "你好嗎"}]
    request_bodies = []
    original_stream = None

    def fake_stream(method, url, **kwargs):
        request_bodies.append(kwargs["json"])
        return _FakeStream(lines)

    with patch("modules.deepseek_service.httpx.stream", side_effect=fake_stream):
        chunks = list(
            generate_reply_stream(
                "test-key", messages, user_turns=1, meta=meta
            )
        )

    assert "".join(chunks) == "你好呀"
    assert meta["total_tokens"] == 42
    assert request_bodies[0]["stream"] is True
    assert request_bodies[0]["stream_options"] == {"include_usage": True}


def test_reply_stream_skips_ui_only_and_blocked() -> None:
    captured = []

    def fake_stream(method, url, **kwargs):
        captured.append(kwargs["json"])
        return _FakeStream(
            ['data: {"choices":[{"delta":{"content":"ok"}}]}', "data: [DONE]"]
        )

    messages = [
        {"role": "assistant", "content": "greeting", "ui_only": True},
        {"role": "user", "content": "blocked", "scope_blocked": True},
        {"role": "assistant", "content": "earlier"},
        {"role": "user", "content": "hello"},
    ]
    with patch("modules.deepseek_service.httpx.stream", side_effect=fake_stream):
        list(generate_reply_stream("test-key", messages, user_turns=1))

    api_messages = captured[0]["messages"][1:]
    contents = [m["content"] for m in api_messages]
    assert "greeting" not in contents
    assert "blocked" not in contents
    assert contents[-1] == "hello"


def test_generate_greeting_uses_recent_user_lines() -> None:
    request_bodies = []

    def fake_post(url: str, **kwargs: object) -> httpx.Response:
        request_bodies.append(kwargs["json"])
        return _completion_response(url, "歡迎返嚟！", 9)

    messages = [
        {"role": "assistant", "content": "How are you?", "mode": "deepseek"},
        {"role": "user", "content": "我孫仔上星期嚟探我"},
    ]
    with patch("modules.deepseek_service.httpx.post", side_effect=fake_post):
        greeting = generate_greeting("test-key", messages)

    assert greeting == "歡迎返嚟！"
    user_payload = request_bodies[0]["messages"][-1]["content"]
    assert "我孫仔上星期嚟探我" in user_payload


def test_deepseek_reply_continues_after_fifth_turn() -> None:
    request_bodies = []

    def fake_post(url: str, **kwargs: object) -> httpx.Response:
        request_bodies.append(kwargs["json"])
        return _completion_response(url, "We can keep talking.", 7)

    messages = [
        {"role": "assistant", "content": "Earlier question?", "mode": "deepseek"},
        {"role": "user", "content": "My answer."},
    ]
    with patch("modules.deepseek_service.httpx.post", side_effect=fake_post):
        result = generate_reply("test-key", messages, user_turns=12, response_language="English")

    assert result.content == "We can keep talking."
    assert result.complete is False
    request = request_bodies[0]
    assert "不限次數" in request["messages"][0]["content"]
    assert all(set(message) == {"role", "content"} for message in request["messages"][1:])


def test_deepseek_summary_uses_plain_text_and_strips_ui_metadata() -> None:
    request_bodies = []

    def fake_post(url: str, **kwargs: object) -> httpx.Response:
        request_bodies.append(kwargs["json"])
        return _completion_response(url, "You mentioned forgetting your keys.", 11)

    messages = [
        {"role": "user", "content": "I forgot my keys."},
        {
            "role": "assistant",
            "content": "What happened next?",
            "mode": "deepseek",
            "speech_language": "en",
        },
    ]
    with patch("modules.deepseek_service.httpx.post", side_effect=fake_post):
        result = generate_summary("test-key", messages, response_language="English")

    assert result.content == "You mentioned forgetting your keys."
    request = request_bodies[0]
    assert request["response_format"] == {"type": "text"}
    assert all(set(message) == {"role", "content"} for message in request["messages"][1:])
    assert "不可推斷認知能力" in request["messages"][0]["content"]
    assert "不是知情者報告" in request["messages"][0]["content"]


def test_deepseek_informant_summary_returns_structured_report() -> None:
    report_json = (
        '{"report_date":"2026-01-01","overall_impression":"User reported forgetting keys.",'
        '"ecog_observations":[{"domain":"Memory","observation":"Forgot keys yesterday.",'
        '"functional_impact":"Mild"}],"informant_concern_level":"Moderate",'
        '"recommendation_for_clinician":"Review memory routines."}'
    )
    request_bodies = []

    def fake_post(url: str, **kwargs: object) -> httpx.Response:
        request_bodies.append(kwargs["json"])
        return _completion_response(url, report_json, 19)

    messages = [
        {"role": "user", "content": "I forgot my keys yesterday."},
        {"role": "assistant", "content": "What happened next?"},
    ]
    with patch("modules.deepseek_service.httpx.post", side_effect=fake_post):
        report = generate_informant_summary(
            "test-key", messages, response_language="English"
        )

    assert report["overall_impression"] == "User reported forgetting keys."
    assert report["ecog_observations"][0]["domain"] == "Memory"
    assert report["ecog_observations"][0]["functional_impact"] == "mild"
    assert report["informant_concern_level"] == "moderate"
    assert report["recommendation_for_clinician"] == "Review memory routines."
    assert report["report_date"]
    assert request_bodies[0]["max_tokens"] == 900
    assert "AI Informant" in request_bodies[0]["messages"][0]["content"]


def test_informant_summary_injects_sensor_context_into_closing_message() -> None:
    report_json = (
        '{"overall_impression":"ok","ecog_observations":[],'
        '"informant_concern_level":"low","recommendation_for_clinician":""}'
    )
    request_bodies = []

    def fake_post(url: str, **kwargs: object) -> httpx.Response:
        request_bodies.append(kwargs["json"])
        return _completion_response(url, report_json, 5)

    messages = [{"role": "user", "content": "我今日去咗街市買餸。"}]
    with patch("modules.deepseek_service.httpx.post", side_effect=fake_post):
        generate_informant_summary(
            "test-key",
            messages,
            sensor_context="Camera captures n=1, latest 41s: blink_rate=22.0",
        )
        generate_informant_summary("test-key", messages)

    closing_with = request_bodies[0]["messages"][-1]["content"]
    closing_without = request_bodies[1]["messages"][-1]["content"]
    assert "blink_rate=22.0" in closing_with
    assert "裝置端觀察數值" in closing_with
    assert "裝置端觀察數值" not in closing_without


def test_summary_requires_ai_and_chat_continues_without_local_fallback() -> None:
    with patch("modules.deepseek_service.get_api_key", return_value=None):
        with patch("modules.tts_service.synthesize_speech", return_value=b"mock-audio"):
            app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=20)
            next(button for button in app.button if button.label == "同意並進入").click().run(
                timeout=20
            )

            for turn in range(1, 5):
                app.chat_input[0].set_value(
                    f"I forgot my keys on day {turn}."
                ).run(timeout=20)
                assert app.session_state["user_turns"] == turn
                assert app.session_state["chat_completed"] is False
                assert any(button.label == "產生知情者摘要" for button in app.button) == (turn == 4)

            message_count = len(app.session_state["chat_messages"])
            next(button for button in app.button if button.label == "產生知情者摘要").click().run(
                timeout=20
            )
            assert app.session_state["user_turns"] == 4
            assert len(app.session_state["chat_messages"]) == message_count
            assert any("啟用 DeepSeek" in warning.value for warning in app.warning)
            assert any(button.label == "產生知情者摘要" for button in app.button)

            for turn in range(5, 9):
                app.chat_input[0].set_value(
                    f"I forgot my keys on day {turn}."
                ).run(timeout=20)
                assert app.session_state["user_turns"] == turn
                assert app.session_state["chat_completed"] is False
                assert any(button.label == "產生知情者摘要" for button in app.button)

            assert not app.exception


def test_informant_report_generates_automatically_at_fifth_turn() -> None:
    report = {
        "report_date": "2026-01-01",
        "overall_impression": "User reported repeated forgetting.",
        "ecog_observations": [
            {
                "domain": "Memory",
                "observation": "Forgot keys on multiple days.",
                "functional_impact": "mild",
            }
        ],
        "informant_concern_level": "moderate",
        "recommendation_for_clinician": "Review memory routines.",
        "total_tokens": 30,
    }
    with patch("modules.deepseek_service.get_api_key", return_value="test-key"):
        with patch(
            "modules.deepseek_service.generate_reply_stream",
            side_effect=lambda *args, **kwargs: iter(["We can keep talking."]),
        ):
            with patch(
                "modules.deepseek_service.generate_informant_summary",
                return_value=report,
            ) as generate_report:
                note_counter = {"count": 0}

                def fake_extract(text: str) -> dict[str, object]:
                    note_counter["count"] += 1
                    return {
                        "id": f"obs-{note_counter['count']}",
                        "theme": "memory",
                        "summary": text[:80],
                        "ecog_dimension": "memory",
                        "risk_flag": "normal",
                        "source": "deepseek",
                    }

                with patch(
                    "modules.memory_extractor.extract_memory_from_input",
                    side_effect=fake_extract,
                ):
                    with patch(
                        "modules.tts_service.synthesize_speech",
                        return_value=b"mock-audio",
                    ):
                        app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=20)
                        next(
                            button
                            for button in app.button
                            if button.label == "同意並進入"
                        ).click().run(timeout=20)

                        for turn in range(1, 6):
                            app.chat_input[0].set_value(
                                f"I forgot my keys on day {turn}."
                            ).run(timeout=20)
                            assert app.session_state["user_turns"] == turn

                        assert generate_report.call_count == 1
                        assert app.session_state["informant_report"]["turn"] == 5
                        assert (
                            app.session_state["informant_report"]["overall_impression"]
                            == "User reported repeated forgetting."
                        )
                        assert not app.exception
