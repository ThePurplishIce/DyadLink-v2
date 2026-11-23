# Demo video script — DyadLink v2 (~3 min)

Narration beats are in **English**; record a zh-Hant voiceover for the
bilingual submission if time permits.

## 0:00–0:20 — Hook & honesty

> "What if someone has no informant — no spouse, no clinician — but still
> wants to be understood? 伴知 DyadLink lets a person *choose* to be
> observed during everyday conversation. It is not a diagnosis and not a
> risk score — it is transparent observation with consent."

Show: title card → consent screen → "what this is / is not" panel.

## 0:20–0:55 — Conversation with guards

> "The conversation stays in everyday life — memory, planning, shopping,
> medication routines. Try to ask it for code or medical dosage and it
> refuses, before the model even answers."

Show: home page → ask a normal life question → then a blocked request
(e.g. "write me a python script") → scope refusal. Mention crisis override
in one line (banner appears for crisis language).

## 0:55–1:35 — Speech acoustics

> "With consent, a short voice note is analysed on-device — pauses and pitch
> variability, computed with NumPy. The audio itself is never stored."

Show: `st.bottom.audio_input` record → transcript review → observation
drawer speech metrics.

## 1:35–2:15 — Multimodal observation (the new part)

> "Optional multimodal mode: the camera and microphone stay on *your*
> device. MediaPipe measures mouth movement, blinks, head motion, and
> audio-visual synchrony — only summary numbers are sent back."

Show: camera button on the chat page → observe 15–30 s → open the
observation drawer → vision metric cards with z-score badges → explain
"compared only with your own past sessions".

## 2:15–2:40 — Personal baseline & research honesty

> "Every observation feeds a personal baseline — exponentially weighted,
> flagged only after several sessions, never compared to other people.
> The dashboard shows what the *research* found — clearly labelled as
> cohort-level, not a score for this visitor."

Show: observation drawer baseline section → dashboard plain-language view →
expand academic details (C-index values, citation).

## 2:40–3:00 — Safety, privacy, limits

> "Consent is explicit and revocable. API spend is budget-capped so the demo
> survives a room of judges. And we're upfront: this is a research
> demonstration — the vision pipeline is experimental and verified
> synthetically so far."

Show: consent withdrawal → budget fallback note → README/docs mention of
limitations. End card: repo + "prior work disclosed".

## Recording checklist

- HTTPS/local URL so camera permission prompt looks legit.
- Pre-seed a few conversation turns so the drawer isn't empty.
- Have the video fallback exported at 1080p before leaving for the venue.
