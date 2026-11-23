import math
from datetime import datetime
from html import escape

import streamlit as st

from modules import baseline
from modules.browser_history import current_token
from modules.i18n import tr


DIMENSION_LABEL_KEYS = {
    "memory": "observation_memory",
    "planning": "observation_planning",
    "financial": "observation_finances",
    "shopping": "observation_shopping",
    "medication": "observation_medication",
    "personal_care": "observation_personal_care",
    "communication": "observation_communication",
    "daily_activities": "observation_daily_tasks",
    "none": "observation_general",
}
FLAG_LABEL_KEYS = {
    "normal": "observation_flag_normal",
    "mild": "observation_flag_mild",
    "concern": "observation_flag_concern",
}
FLAG_STYLES = {
    "normal": "background:#eef6f3;color:#126c5e;border-color:#b9ddd4",
    "mild": "background:#fdf6e7;color:#a66c18;border-color:#ecd9b0",
    "concern": "background:#fdecec;color:#8a2d2d;border-color:#f0c4c4",
}
CONCERN_STYLES = {
    "low": "background:#eef6f3;color:#126c5e;border-color:#b9ddd4",
    "moderate": "background:#fdf6e7;color:#a66c18;border-color:#ecd9b0",
    "high": "background:#fdecec;color:#8a2d2d;border-color:#f0c4c4",
}
IMPACT_STYLES = {
    "none": "background:#eef6f3;color:#126c5e;border-color:#b9ddd4",
    "mild": "background:#fdf6e7;color:#a66c18;border-color:#ecd9b0",
    "moderate": "background:#fde8d8;color:#a34d12;border-color:#f2c9a4",
    "severe": "background:#fdecec;color:#8a2d2d;border-color:#f0c4c4",
}
_LEVEL_STYLES = {
    "attention": "background:#fdf6e7;color:#a66c18;border-color:#ecd9b0",
    "notable": "background:#fdecec;color:#8a2d2d;border-color:#f0c4c4",
}

# Capture features grouped into clinician-recognisable domains. The drawer
# renders one section per group; all of them are exported and summarised
# for the informant report.
_VISION_GROUPS = (
    (
        "vision_group_fluency",
        (
            "speech_rate_proxy",
            "visual_syllable_rate",
            "mean_voiced_segment_s",
            "pause_per_min",
            "mean_pause_s",
            "longest_pause_s",
            "silence_ratio",
            "articulating_silence_ratio",
            "frozen_silence_ratio",
            "speech_tempo_cv",
            "loudness_sd",
            "drift_silence",
            "av_synchrony",
        ),
    ),
    (
        "vision_group_psychomotor",
        (
            "jaw_range",
            "head_movement_energy",
            "stillness_ratio",
            "head_pitch_sd",
            "head_yaw_sd",
            "head_roll_sd",
            "nod_rate",
            "mouth_asymmetry",
            "blink_duration_ms",
        ),
    ),
    (
        "vision_group_affect",
        (
            "expressivity_sd",
            "smile_ratio",
            "frown_ratio",
        ),
    ),
    (
        "vision_group_alertness",
        (
            "perclos",
            "blink_rate",
            "gaze_down_ratio",
            "gaze_shift_rate",
        ),
    ),
)
_GROUP_EN_NAMES = {
    "vision_group_fluency": "Fluency",
    "vision_group_psychomotor": "Psychomotor",
    "vision_group_affect": "Affect",
    "vision_group_alertness": "Alertness",
}
_CAPTURE_FEATURES = tuple(
    feature for _, features in _VISION_GROUPS for feature in features
)
_TWO_DP = frozenset(
    {
        "jaw_range",
        "av_synchrony",
        "articulating_silence_ratio",
        "frozen_silence_ratio",
        "silence_ratio",
        "voiced_ratio",
        "mean_pause_s",
        "longest_pause_s",
        "mean_voiced_segment_s",
        "mouth_asymmetry",
        "expressivity_sd",
        "smile_ratio",
        "frown_ratio",
        "perclos",
        "gaze_down_ratio",
        "stillness_ratio",
        "speech_tempo_cv",
        "loudness_sd",
    }
)
# Signed metrics show a + sign so direction is readable at a glance.
_SIGNED = frozenset({"drift_silence", "av_synchrony"})
_VOCAL_CONTEXT_KEYS = ("silence_ratio", "pause_per_min", "mean_pause_s", "f0_sd_st")


def initialize_observation_state() -> None:
    st.session_state.setdefault("auto_memory_log", [])


def reset_observation_state() -> None:
    st.session_state["auto_memory_log"] = []


def _domain_label_key(domain: object) -> str:
    text = str(domain or "").casefold()
    if "memory" in text or "memories" in text:
        return "observation_memory"
    if "plan" in text or "organiz" in text or "schedule" in text:
        return "observation_planning"
    if "financ" in text or "money" in text or "bill" in text:
        return "observation_finances"
    if "shop" in text or "errand" in text:
        return "observation_shopping"
    if "medication" in text or "medicine" in text or "drug" in text:
        return "observation_medication"
    if "personal" in text or "care" in text or "hygiene" in text or "dressing" in text:
        return "observation_personal_care"
    if "communicat" in text or "speech" in text or "language" in text:
        return "observation_communication"
    if "daily" in text or "activit" in text or "household" in text or "routine" in text:
        return "observation_daily_tasks"
    return "observation_general"


def _badge(label_key: str, style: str) -> None:
    st.markdown(
        f'<span class="obs-flag" style="{style}">{tr(label_key)}</span>',
        unsafe_allow_html=True,
    )


def _format_report_text(report: dict[str, object]) -> str:
    lines = []
    if st.session_state.get("crisis_flagged"):
        lines += [f"**{tr('informant_report_crisis')}**", ""]
    lines += [
        f"# {tr('informant_report_title')}",
        f"{tr('informant_report_date')}: {report.get('report_date', '')}",
        tr("informant_report_turn").format(turn=report.get("turn", "")),
        "",
        f"## {tr('informant_report_overall')}",
        str(report.get("overall_impression", "")),
        "",
        f"## {tr('informant_report_observations')}",
    ]
    observations = report.get("ecog_observations") or []
    if not observations:
        lines.append(tr("informant_report_no_observations"))
    for observation in observations:
        domain = tr(_domain_label_key(observation.get("domain")))
        impact = str(observation.get("functional_impact", ""))
        lines.append(f"- [{domain} | {impact}] {observation.get('observation', '')}")
    lines += [
        "",
        f"{tr('informant_report_concern')}: "
        f"{report.get('informant_concern_level', '')}",
    ]
    vocals = _vocal_samples()
    if vocals:
        latest = vocals[-1]
        lines += [
            "",
            f"## {tr('informant_report_acoustic')}",
            tr(
                "speech_pause",
                pct=round(latest.get("silence_ratio", 0) * 100),
                rate=latest.get("pause_per_min", 0),
                length=latest.get("mean_pause_s", 0),
            ),
            tr("speech_pitch", sd=latest.get("f0_sd_st", 0)),
        ]
    captures = st.session_state.get("vision_captures") or []
    if captures:
        latest = captures[-1]
        rows = [
            f"- {tr(baseline.FEATURE_LABELS.get(feature, feature))}: {latest[feature]}"
            for feature in _CAPTURE_FEATURES
            if isinstance(latest.get(feature), (int, float))
        ]
        if rows:
            duration = latest.get("duration_s")
            meta = tr(
                "vision_capture_meta",
                seconds=round(float(duration), 1)
                if isinstance(duration, (int, float))
                else "?",
                count=len(captures),
            )
            lines += ["", f"## {tr('informant_report_multimodal')}", meta, *rows]
    lines += [
        "",
        f"## {tr('informant_report_recommendation')}",
        str(report.get("recommendation_for_clinician", "")),
    ]
    return "\n".join(lines)


_RADAR_DIMENSIONS = (
    "memory",
    "planning",
    "financial",
    "shopping",
    "medication",
    "personal_care",
    "communication",
    "daily_activities",
)
_FLAG_SCORES = {"normal": 0, "mild": 1, "concern": 2}
_RADAR_LABEL_KEYS = {"financial": "observation_finances_short"}


def _render_ecog_radar() -> None:
    notes = st.session_state.get("auto_memory_log", [])
    if not notes:
        return
    scores = {dimension: 0 for dimension in _RADAR_DIMENSIONS}
    for note in notes:
        dimension = note.get("ecog_dimension")
        score = _FLAG_SCORES.get(note.get("risk_flag", "normal"), 0)
        if dimension in scores:
            scores[dimension] = max(scores[dimension], score)
    if not any(scores.values()):
        return

    cx, cy, radius = 130.0, 118.0, 78.0
    label_r = radius + 16.0
    count = len(_RADAR_DIMENSIONS)
    rings = ""
    for level in (1, 2):
        pts = " ".join(
            f"{cx + radius * level / 2 * math.sin(2 * math.pi * i / count):.1f},"
            f"{cy - radius * level / 2 * math.cos(2 * math.pi * i / count):.1f}"
            for i in range(count)
        )
        rings += (
            f'<polygon points="{pts}" fill="none" stroke="#d7e3ec" '
            f'stroke-width="1"/>'
        )
    axes = ""
    labels = ""
    for i, dimension in enumerate(_RADAR_DIMENSIONS):
        angle = 2 * math.pi * i / count
        x = cx + radius * math.sin(angle)
        y = cy - radius * math.cos(angle)
        axes += (
            f'<line x1="{cx}" y1="{cy}" x2="{x:.1f}" y2="{y:.1f}" '
            f'stroke="#d7e3ec" stroke-width="1"/>'
        )
        lx = cx + label_r * math.sin(angle)
        ly = cy - label_r * math.cos(angle)
        anchor = "middle"
        if math.sin(angle) > 0.35:
            anchor = "start"
        elif math.sin(angle) < -0.35:
            anchor = "end"
        color = "#8a2d2d" if scores[dimension] >= 2 else "#334f63"
        label_key = _RADAR_LABEL_KEYS.get(
            dimension, DIMENSION_LABEL_KEYS[dimension]
        )
        labels += (
            f'<text x="{lx:.1f}" y="{ly + 3:.1f}" text-anchor="{anchor}" '
            f'font-size="9" fill="{color}" font-weight="700">'
            f'{tr(label_key)}</text>'
        )
    polygon = " ".join(
        f"{cx + radius * (scores[d] / 2) * math.sin(2 * math.pi * i / count):.1f},"
        f"{cy - radius * (scores[d] / 2) * math.cos(2 * math.pi * i / count):.1f}"
        for i, d in enumerate(_RADAR_DIMENSIONS)
    )
    dots = ""
    for i, dimension in enumerate(_RADAR_DIMENSIONS):
        score = scores[dimension]
        if not score:
            continue
        angle = 2 * math.pi * i / count
        x = cx + radius * (score / 2) * math.sin(angle)
        y = cy - radius * (score / 2) * math.cos(angle)
        fill = "#8a2d2d" if score >= 2 else "#a66c18"
        dots += f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{fill}"/>'
    st.markdown(f"**{tr('radar_title')}**")
    st.markdown(
        f'<svg viewBox="0 0 260 236" style="max-width:100%;display:block" '
        f'role="img">{rings}{axes}'
        f'<polygon points="{polygon}" fill="rgba(168,50,50,0.22)" '
        f'stroke="#8a2d2d" stroke-width="2"/>{dots}{labels}</svg>',
        unsafe_allow_html=True,
    )
    st.caption(tr("radar_caption"))


def _render_informant_report() -> None:
    report = st.session_state.get("informant_report")
    if not report:
        st.caption(tr("informant_report_empty"))
        return

    with st.container(border=True):
        if st.session_state.get("crisis_flagged"):
            st.markdown(
                f'<div class="crisis-card">{tr("informant_report_crisis")}</div>',
                unsafe_allow_html=True,
            )
        st.markdown(f"**{tr('informant_report_title')}**")

        concern = str(report.get("informant_concern_level", "low")).casefold()
        _badge(
            f"informant_concern_{concern}",
            CONCERN_STYLES.get(concern, CONCERN_STYLES["low"]),
        )
        st.caption(tr("informant_report_turn").format(turn=report.get("turn", "")))

        overall = report.get("overall_impression", "")
        if overall:
            st.markdown(f"**{tr('informant_report_overall')}**")
            st.text(overall)

        observations = report.get("ecog_observations") or []
        st.markdown(f"**{tr('informant_report_observations')}**")
        if not observations:
            st.caption(tr("informant_report_no_observations"))
        for observation in observations:
            impact = str(observation.get("functional_impact", "none")).casefold()
            _badge(
                f"informant_impact_{impact}",
                IMPACT_STYLES.get(impact, IMPACT_STYLES["none"]),
            )
            st.markdown(f"**{tr(_domain_label_key(observation.get('domain')))}**")
            st.text(observation.get("observation", ""))

        recommendation = report.get("recommendation_for_clinician", "")
        if recommendation:
            st.markdown(f"**{tr('informant_report_recommendation')}**")
            st.text(recommendation)

        st.download_button(
            tr("report_download"),
            data=_format_report_text(report),
            file_name=(
                f"dyadlink_informant_report_"
                f"{report.get('report_date') or 'latest'}.md"
            ),
            mime="text/markdown",
            width="stretch",
        )


def _vocal_samples() -> list[dict[str, float]]:
    return [
        message["vocal"]
        for message in st.session_state.get("chat_messages", [])
        if message.get("role") == "user"
        and isinstance(message.get("vocal"), dict)
    ]


def _render_speech_metrics() -> None:
    samples = [
        message["speech_cpm"]
        for message in st.session_state.get("chat_messages", [])
        if message.get("role") == "user"
        and isinstance(message.get("speech_cpm"), (int, float))
        and message["speech_cpm"] > 0
    ]
    vocals = _vocal_samples()
    if not samples and not vocals:
        return

    st.markdown(f"**{tr('speech_title')}**")
    if samples:
        latest = samples[-1]
        st.caption(tr("speech_rate", rate=latest))
        if len(samples) >= 2:
            baseline = sum(samples[:-1]) / (len(samples) - 1)
            if baseline > 0:
                pct = round((latest - baseline) / baseline * 100)
                if pct <= -10:
                    st.caption(tr("speech_trend_slower", pct=abs(pct)))
                elif pct >= 10:
                    st.caption(tr("speech_trend_faster", pct=pct))
                else:
                    st.caption(tr("speech_trend_stable"))
    if vocals:
        latest = vocals[-1]
        st.caption(
            tr(
                "speech_pause",
                pct=round(latest.get("silence_ratio", 0) * 100),
                rate=latest.get("pause_per_min", 0),
                length=latest.get("mean_pause_s", 0),
            )
        )
        if latest.get("f0_sd_st", 0) > 0:
            st.caption(tr("speech_pitch", sd=latest["f0_sd_st"]))
        if len(vocals) >= 2:
            base_sil = [v.get("silence_ratio", 0) for v in vocals[:-1]]
            base_sil = [v for v in base_sil if v]
            if base_sil:
                baseline = sum(base_sil) / len(base_sil)
                if baseline > 0:
                    pct = round(
                        (latest.get("silence_ratio", 0) - baseline)
                        / baseline
                        * 100
                    )
                    if pct >= 15:
                        st.caption(tr("speech_pause_more", pct=pct))
                    elif pct <= -15:
                        st.caption(tr("speech_pause_less", pct=abs(pct)))
            base_f0 = [
                v.get("f0_sd_st", 0) for v in vocals[:-1] if v.get("f0_sd_st")
            ]
            if base_f0 and latest.get("f0_sd_st", 0) > 0:
                baseline = sum(base_f0) / len(base_f0)
                if baseline > 0 and latest["f0_sd_st"] < baseline * 0.85:
                    st.caption(tr("speech_pitch_flatter"))


def _format_feature(feature: str, value: float) -> str:
    if feature in _SIGNED:
        return f"{value:+.2f}"
    if feature == "blink_duration_ms":
        return f"{value:.0f}"
    return f"{value:.2f}" if feature in _TWO_DP else f"{value:.1f}"


def _feature_card(feature: str, value: float, zinfo: dict) -> str:
    level = zinfo.get("level", "none")
    badge = ""
    if level in _LEVEL_STYLES and zinfo.get("z") is not None:
        badge = (
            f'<span class="obs-flag" style="{_LEVEL_STYLES[level]}">'
            f"z={zinfo['z']:+.1f}</span> "
        )
    return (
        f"<div class='history-entry'><span class='history-meta'>"
        f"{escape(tr(baseline.FEATURE_LABELS.get(feature, feature)))} {badge}</span>"
        f"<span class='history-text'>{_format_feature(feature, value)}</span></div>"
    )


def _render_vision_metrics() -> None:
    captures = st.session_state.get("vision_captures") or []
    if not captures:
        return
    latest = captures[-1]
    if not any(
        isinstance(latest.get(feature), (int, float))
        for feature in _CAPTURE_FEATURES
    ):
        return

    st.markdown(f"**{tr('vision_metrics_title')}**")
    token = current_token()
    zmap = baseline.evaluate(token) if token else {}

    face_ratio = latest.get("face_ratio")
    if isinstance(face_ratio, (int, float)):
        audio_key = "vision_audio_yes" if latest.get("has_audio") else "vision_audio_no"
        st.caption(
            f"{tr('vision_quality')} · "
            + tr(
                "vision_quality_meta",
                pct=round(float(face_ratio) * 100),
                audio=tr(audio_key),
            )
        )
        if face_ratio < 0.5:
            st.caption(tr("vision_quality_weak"))

    for group_key, features in _VISION_GROUPS:
        flat = [
            (feature, float(latest[feature]))
            for feature in features
            if isinstance(latest.get(feature), (int, float))
        ]
        if not flat:
            continue
        st.caption(f"**{tr(group_key)}**")
        cols = st.columns(3, gap="small")
        flagged: list[str] = []
        for index, (feature, value) in enumerate(flat):
            zinfo = zmap.get(feature) or {}
            if zinfo.get("level") in _LEVEL_STYLES and zinfo.get("z") is not None:
                arrow = "↑" if zinfo["z"] > 0 else "↓"
                flagged.append(
                    f"{tr(baseline.FEATURE_LABELS.get(feature, feature))} "
                    f"{arrow} z={zinfo['z']:+.1f}"
                )
            cols[index % 3].markdown(
                _feature_card(feature, value, zinfo),
                unsafe_allow_html=True,
            )
        if flagged:
            st.caption(tr("vision_group_flags", items="、".join(flagged)))

    duration = latest.get("duration_s")
    st.caption(
        tr(
            "vision_capture_meta",
            seconds=round(float(duration), 1)
            if isinstance(duration, (int, float))
            else "?",
            count=len(captures),
        )
    )


def sensor_context_text() -> str:
    """Compact English digest of device-side metrics for the LLM report.

    Only aggregate numbers are included — the same values already shown in
    the observation drawer — so no raw audio/video ever leaves the device.
    """
    lines: list[str] = []
    captures = st.session_state.get("vision_captures") or []
    if captures:
        latest = captures[-1]
        duration = latest.get("duration_s")
        header = f"Camera captures n={len(captures)}"
        if isinstance(duration, (int, float)):
            header += f", latest {float(duration):.0f}s"
        face_ratio = latest.get("face_ratio")
        if isinstance(face_ratio, (int, float)):
            header += f", face_ratio={face_ratio}"
        lines.append(header)
        for group_key, features in _VISION_GROUPS:
            parts = [
                f"{feature}={latest[feature]}"
                for feature in features
                if isinstance(latest.get(feature), (int, float))
            ]
            if parts:
                lines.append(
                    f"{_GROUP_EN_NAMES[group_key]}: " + ", ".join(parts)
                )

    vocals = _vocal_samples()
    if vocals:
        latest = vocals[-1]
        parts = [
            f"{key}={latest[key]}"
            for key in _VOCAL_CONTEXT_KEYS
            if isinstance(latest.get(key), (int, float))
        ]
        if parts:
            lines.append(
                f"Speech acoustics n={len(vocals)}: " + ", ".join(parts)
            )

    token = current_token()
    if token:
        flagged = [
            (feature, info["z"], info["level"])
            for feature, info in baseline.evaluate(token).items()
            if info.get("level") in _LEVEL_STYLES and info.get("z") is not None
        ]
        if flagged:
            lines.append(
                "Personal-baseline deviations (z vs own history): "
                + ", ".join(f"{f} z={z:+.1f} ({lvl})" for f, z, lvl in flagged)
            )
    return "\n".join(lines)


def _render_baseline() -> None:
    token = current_token()
    if not token:
        return
    results = baseline.evaluate(token)
    if not results:
        return

    st.markdown(f"**{tr('baseline_title')}**")
    flagged = []
    for feature in baseline.FEATURE_LABELS:
        info = results.get(feature)
        if not info or info.get("level", "none") == "none":
            continue
        flagged.append((feature, info))
    for feature, info in flagged:
        style = (
            FLAG_STYLES["concern"]
            if info.get("level") == "notable"
            else FLAG_STYLES["mild"]
        )
        z = info.get("z")
        ztxt = f" z={z:+.1f}" if z is not None else ""
        st.markdown(
            f'<span class="obs-flag" style="{style}">'
            f"{escape(tr(baseline.FEATURE_LABELS[feature]))}{ztxt}</span>",
            unsafe_allow_html=True,
        )
    n = int(max((info.get("n") or 0 for info in results.values()), default=0))
    st.caption(tr("baseline_samples", n=n))
    st.caption(tr("baseline_caption"))


def render_observation_panel() -> None:
    with st.container(border=True):
        st.markdown(f"### {tr('observation_title')}")
        _render_informant_report()
        _render_speech_metrics()
        _render_vision_metrics()
        _render_baseline()
        _render_ecog_radar()
        st.markdown(f"**{tr('observation_notes_title')}**")

        notes = st.session_state.get("auto_memory_log", [])
        if not notes:
            st.caption(tr("observation_empty"))
            return

        current_date: str | None = None
        for index in reversed(range(len(notes))):
            note = notes[index]
            created_at = note.get("created_at")
            if isinstance(created_at, (int, float)) and created_at > 0:
                date_label = datetime.fromtimestamp(created_at).strftime("%Y-%m-%d")
                time_label = datetime.fromtimestamp(created_at).strftime("%H:%M")
            else:
                date_label = ""
                time_label = ""
            if date_label and date_label != current_date:
                st.markdown(f"**{date_label}**")
                current_date = date_label

            dimension = note.get("ecog_dimension", "none")
            flag = note.get("risk_flag", "normal")
            dimension_key = DIMENSION_LABEL_KEYS.get(dimension, "observation_general")
            flag_key = FLAG_LABEL_KEYS.get(flag, "observation_flag_normal")
            flag_style = FLAG_STYLES.get(flag, FLAG_STYLES["normal"])

            with st.container(border=True):
                _badge(flag_key, flag_style)
                st.caption(
                    f"{time_label} · {tr(dimension_key)}"
                    if time_label
                    else tr(dimension_key)
                )
                st.write(note.get("summary") or note.get("source_quote", ""))
                theme = note.get("theme")
                if isinstance(theme, str) and theme and theme != dimension:
                    st.caption(theme)
                if st.button(
                    tr("observation_locate"),
                    key=f"locate_observation_{note.get('id', index)}",
                    width="stretch",
                ):
                    st.session_state["highlight_quote"] = (
                        note.get("source_quote") or note.get("summary") or ""
                    )
                    st.rerun()
                if st.button(
                    tr("observation_delete"),
                    key=f"delete_observation_{note.get('id', index)}",
                    width="stretch",
                ):
                    notes.pop(index)
                    st.rerun()

        if st.button(
            tr("observation_delete_all"),
            key="delete_all_observations",
            width="stretch",
        ):
            notes.clear()
            st.rerun()
