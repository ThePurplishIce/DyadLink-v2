# Architecture — DyadLink v2

## Design goals

1. **Transparency** — the person being observed knows what is measured and why.
2. **Minimisation** — no raw audio retained, no video leaves the device, only
   aggregate numeric features are stored.
3. **Honesty** — cohort research numbers are labelled as such; individual
   outputs are observation notes for human review, never diagnoses.
4. **Resilience** — every external call is budget-guarded; a local demo mode
   keeps the app usable with zero API spend.

## Request flow

```
Browser ──► app.py
              ├─ consent gate (modules/consent.py)  → st.stop() until agreed
              ├─ token persistence (?c=…, session_store.py)
              ├─ sidebar (language, consent controls, research link)
              └─ st.navigation
                   ├─ home.py        conversation + ASR + TTS + guards + notes
                   │                 (camera dialog → on-device vision capture;
                   │                 metrics shown in the observation drawer)
                   ├─ history.py     persisted conversation history
                   └─ dashboard.py   research explainer
```

## Trust boundaries

| Boundary | Rule |
| --- | --- |
| Browser → server | Only text transcripts and aggregate feature numbers cross. Video/audio never leave the device for the vision path; WAV bytes go to DeepSeek ASR only after consent, and are not stored. |
| Server → DeepSeek | Chat history (sanitised, UI metadata stripped), audio for transcription, summary/report prompts. Every call passes `rate_limit.check_allowed()` first and `record_call()` after. |
| Server → guard endpoint | The latest user message only, sent to `nvidia/nemotron-3.5-lightning-30b-a3b` on NVIDIA NIM for semantic scope classification (custom categories: off-scope work requests, prompt manipulation). `GUARD_*` envs can point at another OpenAI-compatible endpoint. Fail-open: no key, rate limit, timeout or unparsable output all degrade to "safe" — the deterministic layer in `scope_guard` still blocks code syntax/injection without it, and the output sanitizer still filters replies. |
| Model → UI | All model output passes through `sanitize_*` filters; disallowed content (code, math, injection) is blocked *before* the API call where possible and *after* it regardless. |
| Features → baseline | `session_store.append_feature_sample` keeps ≤200 samples/token, numeric fields only, 24 keys max. |
| Component JS | `components/vision/main.js` is first-party trusted code; it contains no user/model content — all UI strings arrive via `data.strings`. |

## Personal baseline (`modules/baseline.py`)

- Each observation event (voice turn or vision capture) appends one numeric
  sample per token.
- Per feature: EWMA mean/variance (α = 0.3).
- `evaluate()` compares the **latest** sample against stats computed over all
  **prior** samples — a new observation never contaminates its own baseline.
- Deviation flags require ≥4 prior samples; near-zero variance is floored at
  2 % of |mean| to avoid noise-amplification.
- z-scores only — no percentages, no population cut-offs, no labels shown to
  the observed person.

## Rate & budget guard (`modules/rate_limit.py`)

Two layers, both keyed by the `?c=` token unless noted:

1. **Per-visitor** — sliding-window calls/minute, min interval, calls/day.
2. **Server-wide** — global calls/day and a global daily *token* budget, so a
   room of judges cannot burn the DeepSeek key.

State is a small JSON file in the temp dir (same pattern as `session_store`),
pruned to 26 h retention / 2000 tokens. `0` disables a guard.
`DeepSeekBudgetExceeded` is caught in `home.py` and surfaced as the
`budget_fallback` message — the app degrades to local mode rather than erroring.

## Multimodal capture (`components/vision/main.js`)

- MediaPipe Face Landmarker (`@mediapipe/tasks-vision` from CDN, WASM + `.task`
  model from Google storage) runs entirely in-browser at ~30 Hz.
- Per frame: `jawOpen`/`eyeBlink*` blendshapes, facial transformation matrix →
  pitch/yaw/roll Euler angles, mic RMS envelope.
- On stop, aggregate features are computed and sent via `setStateValue`:
  - `jaw_range` (P95−P5 of jawOpen), `visual_syllable_rate` (jaw band-passed
    2–8 Hz peak rate), `blink_rate` (hysteresis state machine),
    `head_*_sd` (Euler-angle SDs),
  - cross-modal: voiced/silent classification mirroring
    `audio_features.py`, articulating-vs-frozen silence split (jaw velocity
    during silence vs the speaker's own speech variability), and AV
    synchrony via normalised cross-correlation with lag limited to ±300 ms.
- Python side (`vision_features.py`) re-validates every field against sanity
  bounds before anything is stored — defence in depth, not a security
  boundary (the component is first-party).

### Video-call mode (`mode="call"`)

The chat-page camera dialog mounts the same component in call mode, which
adds a turn-based conversation loop on top of the observation pipeline:

- An `AudioWorkletNode` taps the mic stream (post-AEC) into a 20 s rolling
  PCM buffer; an energy VAD (adaptive noise floor, 350 ms speech onset,
  900 ms endpoint, 15 s cap) detects utterance boundaries.
- Each finished utterance is resampled to 16 kHz WAV, base64-encoded and
  emitted via `setStateValue("utterance", …)` — the **only** path where
  audio leaves the device, and only after consent, exactly like the mic
  button flow.
- `home._handle_call_utterance` transcribes via DeepSeek ASR, then routes
  the text through the *same* `_append_user_message` pipeline as typed
  input — crisis, medical and scope guards all apply. The assistant reply
  is synthesised with Edge TTS and returned through `data.call_reply`;
  the component plays it and resumes listening (VAD is muted while the
  reply plays to avoid echo turns).
- Component sessions are keyed on the `.dlv` DOM node
  (`root.__dlvSession`), because Streamlit re-invokes the component
  function whenever `data` changes; this keeps camera/listeners alive
  across reply updates instead of restarting capture.

## Known limitations

- MediaPipe CDN + model download requires network access to jsdelivr/unpkg
  and `storage.googleapis.com` — a mainland-China reachability risk; the
  recorded-video fallback covers demo day.
- Euler decomposition assumes MediaPipe's column-major 4×4 convention.
- Feature computation is heuristic; thresholds (blink hysteresis, syllable
  band, silence split) are documented in code and tunable.
- No claim is made that these cues measure cognition — they are descriptive
  observations for human reviewers.
