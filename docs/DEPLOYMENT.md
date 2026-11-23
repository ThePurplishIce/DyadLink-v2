# Deployment notes — DyadLink v2

## Options

### Streamlit Community Cloud (fastest)

- New app → connect repo → main file `app.py`.
- Add `DEEPSEEK_API_KEY` (and optionally `DEEPSEEK_MODEL`, budget vars) under
  *Settings → Secrets* as a TOML block:

  ```toml
  DEEPSEEK_API_KEY = "sk-..."
  DYADLINK_DAILY_TOKEN_BUDGET = 500000
  ```

- Resource note: the app is CPU-light; vision inference happens in the
  *visitor's* browser, not on the server.

### Render (container)

`render.yaml` + `Dockerfile` are included:

```bash
docker build -t dyadlink .
docker run -p 8501:8501 -e DEEPSEEK_API_KEY=sk-... dyadlink
```

On Render: New → Blueprint → point at the repo; set `DEEPSEEK_API_KEY` as an
env var. The service needs a persistent-enough disk only for the temp-dir
JSON stores — they are rebuildable caches, not databases.

### Any VM / VPS

```bash
pip install -r requirements.txt
DEEPSEEK_API_KEY=sk-... streamlit run app.py --server.port 8501 \
  --server.address 0.0.0.0 --server.headless true
```

Put it behind HTTPS (Caddy/nginx/Cloudflare Tunnel) — camera/mic capture in
the vision page **requires a secure context** (HTTPS or localhost).

## Mainland-China reachability risk

The app itself is static-ish Streamlit; the risky external calls are:

| Endpoint | Used for | Fallback |
| --- | --- | --- |
| `api.deepseek.com` | chat, ASR, reports | local demo mode (no key) |
| `cdn.jsdelivr.net` / `unpkg.com` | MediaPipe tasks-vision JS | none — vision page degrades with an error banner |
| `storage.googleapis.com` | face_landmarker `.task` model | same |
| `edge-tts` service | spoken replies | subtitles still work |

Mitigations:

1. **Video is the primary fallback** — record the full demo before travel;
   judges accept a video when the live link is unreachable.
2. Test the deployed URL from a mainland network (or ask a teammate there)
   *before* submission; if DeepSeek is unreachable, run with no API key so
   the demo still works end-to-end in local mode.
3. Consider a mirror/proxy for the MediaPipe CDN assets if vision must work
   in-region (self-host `vision_bundle.mjs` + wasm + `.task` under `/static`
   and point `LIB_CDNS`/`DEFAULT_MODEL_URL` at them).

## Budget protection for demo day

A room full of judges can burn the API key quickly. Defaults already cap:

- 12 calls/minute and 300 calls/day **per visitor token**;
- 500 000 tokens/day and 5 000 calls/day **across all visitors**.

Tune via env vars (see `.env.example`). When a cap trips, visitors see a
polite local-fallback notice instead of an error — the demo keeps working.

## Pre-submission checklist

- [ ] Hackathon rules confirmed (deadline, format, prior-work disclosure).
- [ ] Live URL reachable from the judges' network; video backup recorded.
- [ ] `DEEPSEEK_API_KEY` set; budget caps sane for expected traffic.
- [ ] Vision page smoke-tested on a real phone/laptop camera (HTTPS).
- [ ] Consent screen reviewed; crisis banner verified with a test phrase.
- [ ] `pytest` green.
