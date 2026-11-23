from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from modules.dashboard import _USER_COPY
from modules.i18n import _TEXT, tr
from modules.research_data import PAPER_CITATION, PAPER_DOI


ROOT = Path(__file__).resolve().parents[1]


def test_new_ui_labels_are_translated_and_missing_keys_are_readable() -> None:
    keys = (
        "scope_topic_mood",
        "scope_topic_routine",
        "scope_topic_memory",
        "chart_aha_annotation",
        "observation_personal_care",
        "informant_report_title",
        "informant_impact_severe",
    )
    for translations in _TEXT.values():
        for key in keys:
            assert key in translations
            assert translations[key] != key

    with patch("modules.i18n.get_ui_language", return_value="zh-Hant"):
        assert tr("untranslated_test_key") == "Untranslated test key"


def test_auto_observation_log_replaces_manual_save_flow() -> None:
    with patch("modules.deepseek_service.get_api_key", return_value=None):
        with patch("modules.tts_service.synthesize_speech", return_value=b"mock-audio"):
            app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=20)
            next(button for button in app.button if button.label == "同意並進入").click().run(
                timeout=20
            )
            assert not any(
                button.label == "保存這句生活經歷" for button in app.button
            )

            app.chat_input[0].set_value("I forgot my keys yesterday.").run(timeout=20)
            log = app.session_state["auto_memory_log"]
            assert len(log) == 1
            assert log[0]["source"] == "local-fallback"
            assert log[0]["risk_flag"] in {"normal", "mild", "concern"}
            assert log[0]["ecog_dimension"] in {
                "memory",
                "planning",
                "financial",
                "shopping",
                "medication",
                "personal_care",
                "communication",
                "daily_activities",
                "none",
            }
            assert not any(
                button.label == "保存這句生活經歷" for button in app.button
            )
            assert not app.exception


def test_dashboard_shows_citation_and_doi() -> None:
    assert "Cheng, W. W." in PAPER_CITATION
    assert "Version 3" in PAPER_CITATION
    assert PAPER_DOI == "https://doi.org/10.21203/rs.3.rs-10884877/v3"

    with patch("modules.deepseek_service.get_api_key", return_value=None):
        app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=20)
        next(
            button for button in app.button if button.label == "同意並進入"
        ).click().run(timeout=20)
        app.switch_page("modules/dashboard.py").run(timeout=30)

    assert any("Cheng, W. W." in element.value for element in app.markdown)
    assert any(PAPER_DOI in element.value for element in app.markdown)
    assert not app.exception


def test_missing_key_uses_local_demo_and_fresh_session_requires_consent() -> None:
    with patch("modules.deepseek_service.get_api_key", return_value=None):
        with patch("modules.tts_service.synthesize_speech", return_value=b"mock-audio"):
            app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=20)
            next(button for button in app.button if button.label == "同意並進入").click().run(
                timeout=20
            )
            assert any("本機示範模式" in item.value for item in app.info)

            app.chat_input[0].set_value("I forgot my keys yesterday.").run(timeout=20)
            assert app.session_state["chat_messages"][-1]["mode"] == "local-demo"

            refreshed_session = AppTest.from_file(str(ROOT / "app.py")).run(timeout=20)
            assert any(
                button.label == "同意並進入" for button in refreshed_session.button
            )


def test_research_page_defaults_to_plain_language_and_folded_details() -> None:
    hidden_terms = ("C-index", "ΔC", "95% CI", "IPCW", "Cox", "elastic net")
    user_keys = (
        "title",
        "slogan",
        "story",
        "scale",
        "step_base",
        "step_availability",
        "step_ecog",
        "caveat",
        "analogy_title",
        "analogy",
    )
    for copy in _USER_COPY.values():
        visible_copy = " ".join(copy[key] for key in user_keys)
        assert not any(term in visible_copy for term in hidden_terms)

    with patch("modules.deepseek_service.get_api_key", return_value=None):
        app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=20)
        next(button for button in app.button if button.label == "同意並進入").click().run(
            timeout=20
        )
        app.switch_page("modules/dashboard.py").run(timeout=20)

    assert [progress.value for progress in app.progress] == [68, 73, 76]
    academic_panel = next(
        expander for expander in app.expander if "研究依據" in expander.label
    )
    assert academic_panel.proto.expanded is False
    assert not app.exception
