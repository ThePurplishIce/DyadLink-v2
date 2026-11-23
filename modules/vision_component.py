"""Multimodal observation custom component (Streamlit components v2).

The browser-side module lives in ``components/vision/main.js`` and is
registered once here as a trusted bidi component. It captures camera +
microphone on-device with MediaPipe Face Landmarker and returns only
aggregate numeric features — no video or raw audio leaves the browser.
"""

from __future__ import annotations

from pathlib import Path

from streamlit.components.v2 import component as _v2_component

_JS_PATH = (
    Path(__file__).resolve().parents[1] / "components" / "vision" / "main.js"
)

_VISION_CSS = """
.dlv { font-family: inherit; }
.dlv-stage {
  position: relative;
  width: 100%;
  max-width: 420px;
  aspect-ratio: 4 / 3;
  margin: 0 auto 0.6rem;
  border: 1px solid #d7e3ec;
  border-radius: 1rem;
  overflow: hidden;
  background: #0f2233;
}
.dlv-video {
  width: 100%;
  height: 100%;
  object-fit: cover;
  transform: scaleX(-1);
  display: block;
}
.dlv-overlay {
  position: absolute;
  left: 0; right: 0; bottom: 0;
  padding: 0.6rem 0.8rem;
  background: linear-gradient(180deg, transparent, rgba(10, 30, 45, 0.82));
  color: #eef6f9;
}
.dlv-status { font-size: 0.95rem; font-weight: 600; min-height: 1.2em; }
.dlv-statusrow {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}
.dlv-face {
  width: 0.7rem;
  height: 0.7rem;
  border-radius: 50%;
  background: rgba(255, 255, 255, 0.35);
  flex: none;
  transition: background 0.2s, box-shadow 0.2s;
}
.dlv-face.on {
  background: #35c4ac;
  box-shadow: 0 0 8px rgba(53, 196, 172, 0.9);
}
.dlv-progress {
  margin-top: 0.4rem;
  height: 0.5rem;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.22);
  overflow: hidden;
}
.dlv-progress-fill {
  height: 100%;
  width: 0%;
  border-radius: 999px;
  background: #35c4ac;
  transition: width 0.2s linear;
}
.dlv-wave {
  display: block;
  width: 100%;
  height: 3.4rem;
  margin-top: 0.55rem;
  border-radius: 0.8rem;
  background: #0d2438;
}
.dlv-caption {
  margin-top: 0.45rem;
  min-height: 1.3em;
  text-align: center;
  color: #244b69;
  font-size: 0.98rem;
  font-weight: 650;
  line-height: 1.45;
}
.dlv-unmute {
  display: block;
  margin: 0.4rem auto 0;
  padding: 0.35rem 1rem;
  border: 1px solid #c7d6e0;
  border-radius: 999px;
  background: #fff;
  color: #244b69;
  font-size: 0.85rem;
  font-weight: 700;
  cursor: pointer;
}
.dlv-controls {
  display: flex;
  align-items: center;
  gap: 0.7rem;
  flex-wrap: wrap;
  justify-content: center;
}
.dlv-btn {
  min-height: 2.8rem;
  padding: 0.45rem 1.4rem;
  border: none;
  border-radius: 0.9rem;
  background: linear-gradient(135deg, #168a78, #1a9a84);
  color: #fff;
  font-size: 1rem;
  font-weight: 700;
  cursor: pointer;
  box-shadow: 0 6px 16px rgba(22, 138, 120, 0.25);
}
.dlv-btn:disabled { opacity: 0.55; cursor: default; }
.dlv-btn-secondary {
  background: #eef3f6;
  color: #244b69;
  border: 1px solid #c7d6e0;
  box-shadow: none;
}
.dlv-duration {
  font-size: 1rem;
  font-weight: 600;
  padding: 0.4rem 0.6rem;
  border: 1px solid #c7d6e0;
  border-radius: 0.7rem;
  background: #fff;
  color: #18344a;
}
.dlv-note { width: 100%; text-align: center; color: #526d80; font-size: 0.9rem; }
.dlv-error {
  margin-top: 0.6rem;
  padding: 0.7rem 0.9rem;
  border: 1px solid #f0c4c4;
  border-left: 5px solid #b03030;
  border-radius: 0.8rem;
  background: #fdecec;
  color: #6d1f1f;
  font-size: 0.95rem;
  font-weight: 600;
}
.dlv-done { color: #126c5e; font-weight: 700; text-align: center; }
"""

_VISION_HTML = """
<div class="dlv">
  <div class="dlv-stage">
    <video class="dlv-video" autoplay muted playsinline></video>
    <div class="dlv-overlay">
      <div class="dlv-statusrow">
        <span class="dlv-face"></span>
        <div class="dlv-status"></div>
      </div>
      <div class="dlv-progress"><div class="dlv-progress-fill"></div></div>
    </div>
  </div>
  <canvas class="dlv-wave"></canvas>
  <div class="dlv-caption"></div>
  <div class="dlv-controls">
    <button type="button" class="dlv-btn dlv-start"></button>
    <button type="button" class="dlv-btn dlv-btn-secondary dlv-stop" hidden></button>
    <select class="dlv-duration" aria-label="duration">
      <option value="15">15 s</option>
      <option value="30" selected>30 s</option>
      <option value="60">60 s</option>
    </select>
    <span class="dlv-note"></span>
  </div>
  <button type="button" class="dlv-unmute" hidden></button>
  <div class="dlv-error" hidden></div>
  <div class="dlv-done" hidden></div>
</div>
"""


def _component_strings() -> dict[str, str]:
    from modules.i18n import tr

    return {
        "start": tr("mm_start"),
        "stop": tr("mm_stop"),
        "loading_model": tr("mm_loading_model"),
        "requesting_media": tr("mm_requesting_media"),
        "privacy_note": tr("mm_privacy_note"),
        "done_hint": tr("mm_done_hint"),
        "capturing": tr("mm_capturing"),
        "idle_hint": tr("mm_idle_hint"),
        "enable_sound": tr("mm_enable_sound"),
        "error_insecure": tr("mm_error_insecure"),
        "error_permission": tr("mm_error_permission"),
        "error_no_device": tr("mm_error_no_device"),
        "error_generic": tr("mm_error_generic"),
        "call_start": tr("call_start"),
        "call_end": tr("call_end"),
        "call_listening": tr("call_listening"),
        "call_waiting": tr("call_waiting"),
        "call_speaking": tr("call_speaking"),
        "call_error": tr("call_error"),
    }


def mount_vision_observer(
    *,
    autostart: bool = False,
    key: str = "dyadlink_vision_observer",
    mode: str = "observe",
    call_reply: dict | None = None,
):
    """Mount the observation component and return its ``ComponentResult``.

    ``autostart=True`` begins camera+mic capture as soon as the component
    mounts (used by the chat-page camera dialog). ``mode="call"`` enables
    the conversational loop: finished utterances are emitted as
    ``utterance`` state and replies arrive through ``data.call_reply``.
    ``on_result_change``/``on_utterance_change`` keep both attributes
    present on the result.
    """
    return vision_observer(
        key=key,
        data={
            "strings": _component_strings(),
            "modelUrl": "",
            "autostart": autostart,
            "mode": mode,
            "call_reply": call_reply,
        },
        width="stretch",
        height="content",
        on_result_change=lambda: None,
        on_utterance_change=lambda: None,
    )


def _load_js() -> str:
    return _JS_PATH.read_text(encoding="utf-8")


vision_observer = _v2_component(
    name="dyadlink_vision_observer",
    html=_VISION_HTML,
    css=_VISION_CSS,
    js=_load_js(),
    isolate_styles=True,
)
