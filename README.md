# 伴知 DyadLink v2

*A transparent, consent-first everyday-life observation demo — research
preliminary submission for Hackbridge × neXus AI Hackathon 2027 (Voice &
Multimodal track, provisional).*

> **What this is:** a Streamlit research-demonstration app that lets a person
> *choose* to be observed during everyday conversation — with speech acoustics
> and optional on-device camera cues — and produces plain-language observation
> notes for an authorised reviewer.
>
> **What this is not:** a diagnostic tool, an individual risk predictor, a
> covert cognitive test, or medical advice. The research numbers shown are
> fixed cohort-level results from published work; they are **not** scores
> computed from anything a visitor says or does.

---

## Features (what is actually built)

| Area | Status |
| --- | --- |
| Guided everyday-life conversation (memory, planning, money, medication…) | ✅ working |
| Safety scope guard: deterministic layer for code syntax & injection + semantic guard via NVIDIA NIM (`nvidia/nemotron-3.5-lightning-30b-a3b`, ~10k req/day free) — fails open when unavailable | ✅ working |
| Medical-safety pre-filter + crisis-language override with support banner | ✅ working |
| Speech acoustics from WAV (pauses, pitch variability) — NumPy only, no raw audio retained | ✅ working |
| Browser-side multimodal observation (MediaPipe Face Landmarker, camera+mic analysed on-device; ~30 aggregate metrics grouped into fluency / psychomotor / affect / alertness domains, only numbers returned) — camera dialog on the chat page, results shown in the observation drawer | ✅ implemented — **experimental, needs real-device verification** |
| Video-call mode in the camera dialog: speak → auto-pause → DeepSeek ASR → guarded reply → spoken answer in-frame | ✅ implemented — turn-based (not token-streamed); VAD + WAV upload |
| Personal longitudinal baseline: EWMA z-scores vs *your own* past sessions | ✅ working |
| API budget guard: per-minute rate, per-session daily cap, server-wide daily token budget with local fallback | ✅ working |
| Token-keyed bounded persistence (messages, consent, feature samples) | ✅ working |
| Four UI languages: Cantonese, Mainland Mandarin, Taiwan Mandarin, English | ✅ working |
| Research dashboard: plain-language progress + collapsed academic detail | ✅ working |

Planned / not yet verified: MediaPipe capture quality on real devices,
mainland-China reachability of CDN/model endpoints (video fallback required),
formal pilot data for acoustic claims.

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux
pip install -r requirements.txt
cp .env.example .env            # add DEEPSEEK_API_KEY for live mode
streamlit run app.py
```

Without `DEEPSEEK_API_KEY` the app runs in a clearly labelled **local demo
mode** — conversation, guards and UI all work; only live DeepSeek
transcription/chat is disabled.

## Configuration

See [`.env.example`](.env.example). Key variables:

- `DEEPSEEK_API_KEY` — live conversation, ASR, summaries, informant reports.
- `DEEPSEEK_MODEL` — defaults to `deepseek-flash`.
- `NVIDIA_API_KEY` — semantic scope-guard calls to `nvidia/nemotron-3.5-lightning-30b-a3b` on build.nvidia.com NIM (~40 rpm / ~10k req/day free tier).
- `GUARD_API_KEY` / `GUARD_BASE_URL` / `GUARD_MODEL` — point the guard at another OpenAI-compatible endpoint instead. All optional — without any key the deterministic layer still blocks code syntax/injection and ambiguous messages fail open.
- `DYADLINK_RATE_PER_MINUTE`, `DYADLINK_SESSION_CALLS_PER_DAY` — per-visitor caps.
- `DYADLINK_DAILY_TOKEN_BUDGET`, `DYADLINK_GLOBAL_CALLS_PER_DAY` — server-wide caps so judge traffic cannot exhaust the key. When a cap trips, visitors get a polite local-fallback message (`budget_fallback`).

## Architecture

```
app.py                     consent gate → sidebar → st.navigation
modules/home.py            conversation, ASR, TTS, safety guards, observation notes
modules/scope_guard.py     deterministic scope patterns + delegates semantics
modules/nvidia_guard.py    NVIDIA NIM guard client (fail-open)
modules/deepseek_service.py chat/transcription/summary/report + budget guard
modules/rate_limit.py      JSON-file rate + token-budget guards (temp dir)
modules/baseline.py        EWMA personal baseline, z-score deviation flags
modules/session_store.py   token-keyed bounded persistence (?c= token)
modules/audio_features.py  NumPy acoustic features from WAV
modules/vision_component.py Streamlit v2 bidi component registration
modules/vision_features.py payload validation/bounds
components/vision/main.js  on-device MediaPipe capture + feature computation
```

See [ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full data-flow and trust
boundaries, and [DEPLOYMENT.md](docs/DEPLOYMENT.md) for hosting notes
(including the mainland-reachability risk and video fallback).

## Privacy & ethics commitments

- Full-screen consent gate before anything runs; withdrawal clears state.
- Conversation text, transcripts and numeric features only — **raw audio is
  never stored**, **video is never stored or uploaded**.
- Multimodal cues (mouth movement, blinks, head pose, pauses, AV synchrony)
  are computed **on the device**; only aggregate numbers reach the server.
- Do not enter names, phone numbers, or other identifiers.
- Crisis language triggers support resources *before* any observation flow.
- Outputs are observation notes for human reviewers — never labels shown to
  the person being observed, never a diagnosis.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

Covers the safety regression set (scope guard, medical pre-filter, crisis
override, model-output filtering), budget guards, baseline statistics,
vision-payload validation, consent gating and translation completeness.

## Prior-work disclosure

DyadLink v1 (this repository's UI/UX foundation and conversation engine) was
built before this hackathon. v2 adds the multimodal capture pipeline,
personal-baseline statistics and deployment hardening described in
`MEGA_PLAN.md`. We will disclose this split explicitly in the submission,
per the competition's prior-work rules (TBC at kick-off 2026-11-29).

## 中文說明

伴知 DyadLink v2 是一個**研究示範應用**：在知情同意下，透過日常對話與
（可選的）裝置端多模態觀察，為有需要的照顧者／研究者產生淺白的觀察紀錄。

- **不是**診斷工具、個人風險預測、隱蔽認知測驗或醫療建議。
- 儀表板上的研究數字是已發表研究的**群體層級**結果，不是訪客個人的分數。
- 語音與多模態指標只與**使用者自己過往的紀錄**比較（個人基線），不設群體門檻。
- 相機／麥克風影像與聲音只在裝置上分析，不上傳、不儲存；只有匯總數字回傳。
- 危機言論會優先觸發支援資源；所有輸出供人工審閱。

---

*Built for Hackbridge × neXus AI Hackathon 2027 — University Track.*
