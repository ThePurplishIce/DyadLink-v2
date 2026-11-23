// DyadLink — multimodal observation component (Streamlit v2 bidi component).
//
// Everything runs in the browser: camera + microphone are analysed on-device
// with MediaPipe Face Landmarker. No video frames and no raw audio ever leave
// the device — only aggregate numeric features are sent to Python via
// setStateValue().
//
// Frame features follow MEGA_PLAN §7 (expanded — clinician-readable groups):
//   Fluency     visual_syllable_rate, speech_rate_proxy, pause/voiced-segment
//               stats, silence decomposition, tempo CV, loudness SD, drift
//   Psychomotor jaw_range, head euler SDs, movement energy, stillness, nods,
//               mouth asymmetry, blink duration
//   Affect      expressivity_sd, smile/frown time ratios
//   Alertness   PERCLOS, blink rate, gaze direction & shift rate
// Cross-modal (§7.2):
//   articulating/frozen silence decomposition and AV synchrony on a common
//   30 Hz time base with lag search limited to ±300 ms.

const MP_VERSION = "0.10.14";
const LIB_CDNS = [
  `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}/vision_bundle.mjs`,
  `https://unpkg.com/@mediapipe/tasks-vision@${MP_VERSION}/vision_bundle.mjs`,
];
const WASM_CDNS = [
  `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}/wasm`,
  `https://unpkg.com/@mediapipe/tasks-vision@${MP_VERSION}/wasm`,
];
const DEFAULT_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

const TARGET_HZ = 30;
const TICK_MS = 1000 / TARGET_HZ;
const MAX_LAG_FRAMES = 9; // ±300 ms at 30 Hz
const MIN_PAUSE_S = 0.25;
const BLINK_HIGH = 0.5;
const BLINK_LOW = 0.35;
const BLINK_ARM = 0.3;
const BLINK_MIN_GAP_MS = 150;
const MIN_SPEECH_FRAMES = 20;

const clamp = (v, lo, hi) => Math.min(Math.max(v, lo), hi);
const r3 = (v) => (v == null ? null : Math.round(v * 1000) / 1000);
const r4 = (v) => (v == null ? null : Math.round(v * 10000) / 10000);

function percentile(sorted, q) {
  if (!sorted.length) return null;
  const i = (sorted.length - 1) * q;
  const lo = Math.floor(i);
  const hi = Math.ceil(i);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (i - lo);
}

function median(values) {
  if (!values.length) return null;
  const s = [...values].sort((a, b) => a - b);
  return percentile(s, 0.5);
}

function stddev(values) {
  if (values.length < 2) return 0;
  const m = values.reduce((a, b) => a + b, 0) / values.length;
  return Math.sqrt(values.reduce((a, b) => a + (b - m) * (b - m), 0) / (values.length - 1));
}

// Linear interpolation of `values` (may contain null) onto `grid` times.
// Gaps longer than maxGap are left null.
function resample(times, values, grid, maxGap) {
  const out = new Array(grid.length).fill(null);
  let j = 0;
  for (let i = 0; i < grid.length; i++) {
    const gt = grid[i];
    while (j < times.length - 1 && times[j + 1] < gt) j++;
    const candidates = [];
    for (let k = Math.max(0, j - 1); k <= Math.min(times.length - 1, j + 1); k++) {
      candidates.push(k);
    }
    let best = -1;
    let bestDt = Infinity;
    for (const k of candidates) {
      const dt = Math.abs(times[k] - gt);
      if (dt < bestDt) {
        bestDt = dt;
        best = k;
      }
    }
    if (best >= 0 && values[best] != null && isFinite(values[best]) && bestDt <= maxGap) {
      out[i] = values[best];
    }
  }
  return out;
}

function movingAverage(x, w) {
  const out = new Array(x.length).fill(null);
  const half = Math.floor(w / 2);
  for (let i = 0; i < x.length; i++) {
    if (x[i] == null) continue;
    let sum = 0;
    let n = 0;
    for (let k = Math.max(0, i - half); k <= Math.min(x.length - 1, i + half); k++) {
      if (x[k] != null) {
        sum += x[k];
        n++;
      }
    }
    if (n) out[i] = sum / n;
  }
  return out;
}

// RBJ biquad bandpass, constant 0 dB peak gain, applied forward-only.
function bandpass(x, fs, f1, f2) {
  const f0 = Math.sqrt(f1 * f2);
  const q = f0 / (f2 - f1);
  const w0 = (2 * Math.PI * f0) / fs;
  const alpha = Math.sin(w0) / (2 * q);
  const a0 = 1 + alpha;
  const b0 = alpha / a0;
  const b1 = 0;
  const b2 = -alpha / a0;
  const a1 = (-2 * Math.cos(w0)) / a0;
  const a2 = (1 - alpha) / a0;
  const out = new Array(x.length).fill(0);
  let x1 = 0;
  let x2 = 0;
  let y1 = 0;
  let y2 = 0;
  for (let i = 0; i < x.length; i++) {
    const xi = x[i] == null ? 0 : x[i];
    const yi = b0 * xi + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
    out[i] = yi;
    x2 = x1;
    x1 = xi;
    y2 = y1;
    y1 = yi;
  }
  return out;
}

function detectPeaks(x, minDist) {
  const peaks = [];
  const valid = x.map((v, i) => (v == null ? null : i)).filter((i) => i != null);
  if (valid.length < 3) return peaks;
  const vals = valid.map((i) => x[i]);
  const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
  const sd = stddev(vals);
  const thr = mean + 0.25 * sd;
  let last = -minDist - 1;
  for (let n = 1; n < valid.length - 1; n++) {
    const i = valid[n];
    if (x[i] == null || x[i] <= thr) continue;
    if (x[i] >= x[valid[n - 1]] && x[i] > x[valid[n + 1]] && i - last >= minDist) {
      peaks.push(i);
      last = i;
    }
  }
  return peaks;
}

function categoryScore(categories, name) {
  for (const c of categories) {
    if (c && c.categoryName === name && typeof c.score === "number") {
      return c.score;
    }
  }
  return null;
}

function eulerFromMatrix(data) {
  // MediaPipe facialTransformationMatrixes are 4x4 column-major:
  // R[row][col] = data[col * 4 + row].
  if (!data || data.length < 16) return null;
  const r20 = data[0 * 4 + 2];
  const r10 = data[0 * 4 + 1];
  const r00 = data[0 * 4 + 0];
  const r21 = data[1 * 4 + 2];
  const r22 = data[2 * 4 + 2];
  const pitch = Math.asin(clamp(-r20, -1, 1));
  const yaw = Math.atan2(r10, r00);
  const roll = Math.atan2(r21, r22);
  const d = 180 / Math.PI;
  return { pitch: pitch * d, yaw: yaw * d, roll: roll * d };
}

function pearson(a, b) {
  const n = Math.min(a.length, b.length);
  if (n < 8) return null;
  let ma = 0;
  let mb = 0;
  for (let i = 0; i < n; i++) {
    ma += a[i];
    mb += b[i];
  }
  ma /= n;
  mb /= n;
  let num = 0;
  let da = 0;
  let db = 0;
  for (let i = 0; i < n; i++) {
    const xa = a[i] - ma;
    const xb = b[i] - mb;
    num += xa * xb;
    da += xa * xa;
    db += xb * xb;
  }
  if (da <= 0 || db <= 0) return null;
  return num / Math.sqrt(da * db);
}

// PCM -> 16 kHz Int16 -> WAV helpers for voice-turn upload.
function resampleTo16k(float32, srcRate) {
  if (!float32.length) return new Int16Array(0);
  if (srcRate === 16000) {
    const out = new Int16Array(float32.length);
    for (let i = 0; i < float32.length; i++) {
      out[i] = clamp(Math.round(float32[i] * 32767), -32768, 32767);
    }
    return out;
  }
  const ratio = srcRate / 16000;
  const outLen = Math.max(1, Math.floor(float32.length / ratio));
  const out = new Int16Array(outLen);
  for (let i = 0; i < outLen; i++) {
    const pos = i * ratio;
    const i0 = Math.min(float32.length - 1, Math.floor(pos));
    const i1 = Math.min(float32.length - 1, i0 + 1);
    const v = float32[i0] + (float32[i1] - float32[i0]) * (pos - i0);
    out[i] = clamp(Math.round(v * 32767), -32768, 32767);
  }
  return out;
}

function encodeWavPcm16(pcm, sampleRate) {
  const dataLen = pcm.length * 2;
  const buf = new ArrayBuffer(44 + dataLen);
  const v = new DataView(buf);
  const wstr = (off, s) => {
    for (let i = 0; i < s.length; i++) v.setUint8(off + i, s.charCodeAt(i));
  };
  wstr(0, "RIFF");
  v.setUint32(4, 36 + dataLen, true);
  wstr(8, "WAVE");
  wstr(12, "fmt ");
  v.setUint32(16, 16, true);
  v.setUint16(20, 1, true); // PCM
  v.setUint16(22, 1, true); // mono
  v.setUint32(24, sampleRate, true);
  v.setUint32(28, sampleRate * 2, true);
  v.setUint16(32, 2, true);
  v.setUint16(34, 16, true);
  wstr(36, "data");
  v.setUint32(40, dataLen, true);
  for (let i = 0; i < pcm.length; i++) {
    v.setInt16(44 + i * 2, pcm[i], true);
  }
  return new Uint8Array(buf);
}

function bytesToBase64(bytes) {
  let s = "";
  const CH = 0x8000;
  for (let i = 0; i < bytes.length; i += CH) {
    s += String.fromCharCode.apply(null, bytes.subarray(i, i + CH));
  }
  return btoa(s);
}

// One live capture session per mounted `.dlv` root. Streamlit re-invokes the
// component function whenever `data` changes; keeping session state on the
// DOM node prevents camera/listener resets and duplicate handlers.
function createSession(root, parentElement, cfg, setStateValue) {
  let L = cfg.strings || {};
  const modelUrl = cfg.modelUrl || DEFAULT_MODEL_URL;
  const callMode = cfg.mode === "call";

  const els = {};
  for (const sel of [
    ".dlv-video",
    ".dlv-status",
    ".dlv-face",
    ".dlv-caption",
    ".dlv-wave",
    ".dlv-progress-fill",
    ".dlv-start",
    ".dlv-stop",
    ".dlv-duration",
    ".dlv-note",
    ".dlv-unmute",
    ".dlv-error",
    ".dlv-done",
  ]) {
    els[sel] = root.querySelector(sel);
  }
  if (!els[".dlv-start"]) return { cleanup: () => {}, update: () => {} };

  const video = els[".dlv-video"];
  const statusEl = els[".dlv-status"];
  const faceEl = els[".dlv-face"];
  const captionEl = els[".dlv-caption"];
  const waveCanvas = els[".dlv-wave"];
  const fillEl = els[".dlv-progress-fill"];
  const startBtn = els[".dlv-start"];
  const stopBtn = els[".dlv-stop"];
  const durationSel = els[".dlv-duration"];
  const noteEl = els[".dlv-note"];
  const unmuteBtn = els[".dlv-unmute"];
  const errorEl = els[".dlv-error"];
  const doneEl = els[".dlv-done"];
  const waveCtx = waveCanvas ? waveCanvas.getContext("2d") : null;
  const WAVE_LEN = 240; // ~8 s at 30 Hz
  let waveData = new Array(WAVE_LEN).fill(0);

  const text = (el, key, fallback) => {
    if (el) el.textContent = L[key] || fallback || "";
  };
  text(startBtn, callMode ? "call_start" : "start", callMode ? "Start call" : "Start observation");
  text(stopBtn, callMode ? "call_end" : "stop", "Stop");
  text(noteEl, "privacy_note", "Video and audio stay on this device.");
  text(doneEl, "done_hint", "");
  text(unmuteBtn, "enable_sound", "Tap for sound");
  text(captionEl, "idle_hint", "");
  if (callMode && durationSel) durationSel.hidden = true;
  if (errorEl) errorEl.hidden = true;

  let landmarker = null;
  let lib = null;
  let stream = null;
  let audioCtx = null;
  let analyser = null;
  let audioBuf = null;
  let timer = null;
  let autoTimer = null;
  let samples = [];
  let running = false;
  let startTs = 0;

  // Call-mode voice-turn state: a 16 kHz PCM tap feeds a lightweight energy
  // VAD; completed utterances are encoded to WAV and uploaded for ASR.
  let pcmTap = null;
  let pcmMute = null;
  let pcmChunks = [];
  let pcmTotal = 0;
  let pcmTapOk = false;
  let replyAudio = null;
  let lastReplyId = null;
  const vad = {
    noise: 0.012,
    inSpeech: false,
    speechTicks: 0,
    silenceTicks: 0,
    speechStartSample: 0,
    awaitingReply: false,
    awaitingSince: 0,
  };

  function setStatus(msg) {
    if (statusEl && statusEl.textContent !== (msg || "")) {
      statusEl.textContent = msg || "";
    }
  }

  function showError(msg) {
    if (errorEl) {
      errorEl.textContent = msg || "";
      errorEl.hidden = !msg;
    }
  }

  function setProgress(frac) {
    if (fillEl) fillEl.style.width = `${Math.round(clamp(frac, 0, 1) * 100)}%`;
  }

  function setFace(found) {
    if (faceEl) faceEl.classList.toggle("on", !!found);
  }

  function setCaption(msg) {
    if (captionEl) captionEl.textContent = msg || "";
  }

  function drawWave() {
    if (!waveCanvas || !waveCtx) return;
    const dpr = window.devicePixelRatio || 1;
    const w = waveCanvas.clientWidth;
    const h = waveCanvas.clientHeight;
    if (!w || !h) return;
    if (waveCanvas.width !== Math.round(w * dpr)) {
      waveCanvas.width = Math.round(w * dpr);
      waveCanvas.height = Math.round(h * dpr);
    }
    const ctx = waveCtx;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    const n = waveData.length;
    const barW = w / n;
    const mid = h / 2;
    const rounded = typeof ctx.roundRect === "function";
    for (let i = 0; i < n; i++) {
      const v = waveData[i];
      const bh = Math.max(1.5, v * h * 0.92);
      const x = i * barW + barW * 0.18;
      const bw = barW * 0.64;
      const alpha = 0.35 + v * 0.65;
      ctx.fillStyle = `rgba(53, 196, 172, ${alpha})`;
      ctx.beginPath();
      if (rounded) {
        ctx.roundRect(x, mid - bh / 2, bw, bh, barW * 0.3);
      } else {
        ctx.rect(x, mid - bh / 2, bw, bh);
      }
      ctx.fill();
    }
  }

  function pushWave(rms) {
    const v = rms == null ? 0 : clamp(rms * 6, 0, 1);
    waveData.push(v);
    if (waveData.length > WAVE_LEN) waveData.shift();
    drawWave();
  }

  function setRunningUI(on) {
    if (startBtn) startBtn.hidden = on;
    if (stopBtn) stopBtn.hidden = !on;
    if (durationSel) durationSel.disabled = on;
    if (doneEl) doneEl.hidden = on;
  }

  async function importVision() {
    let lastErr = null;
    for (let i = 0; i < LIB_CDNS.length; i++) {
      try {
        lib = await import(LIB_CDNS[i]);
        return LIB_CDNS[i];
      } catch (err) {
        lastErr = err;
      }
    }
    throw lastErr || new Error("mediapipe_load_failed");
  }

  async function ensureLandmarker() {
    if (landmarker) return;
    setStatus(L.loading_model || "Loading model…");
    const bundleUrl = await importVision();
    const pkgBase = bundleUrl.slice(0, bundleUrl.lastIndexOf("/"));
    const wasmBase = `${pkgBase}/wasm`;
    const fileset = await lib.FilesetResolver.forVisionTasks(wasmBase);
    landmarker = await lib.FaceLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: modelUrl, delegate: "GPU" },
      outputFaceBlendshapes: true,
      outputFacialTransformationMatrixes: true,
      runningMode: "VIDEO",
      numFaces: 1,
    });
  }

  async function start() {
    if (running) return;
    showError(null);
    startBtn.disabled = true;
    try {
      if (!window.isSecureContext) {
        throw new Error(L.error_insecure || "Camera needs HTTPS or localhost.");
      }
      await ensureLandmarker();
      setStatus(L.requesting_media || "Requesting camera and microphone…");
      stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: "user",
        },
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
          channelCount: 1,
        },
      });
      video.srcObject = stream;
      await video.play();

      audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      if (audioCtx.state === "suspended") await audioCtx.resume();
      if (unmuteBtn) unmuteBtn.hidden = audioCtx.state !== "suspended";
      const source = audioCtx.createMediaStreamSource(stream);
      analyser = audioCtx.createAnalyser();
      analyser.fftSize = 2048;
      analyser.smoothingTimeConstant = 0;
      source.connect(analyser);
      audioBuf = new Float32Array(analyser.fftSize);
      if (callMode) await setupPcmTap(source);

      samples = [];
      waveData = new Array(WAVE_LEN).fill(0);
      pcmChunks = [];
      pcmTotal = 0;
      startTs = performance.now();
      running = true;
      setRunningUI(true);
      setFace(false);
      if (callMode) {
        setCaption("");
        setStatus(L.call_listening || "Listening — just speak");
      } else {
        setCaption(L.capturing || "Observing…");
      }
      const durationMs = callMode
        ? 0
        : (parseFloat(durationSel ? durationSel.value : "30") || 30) * 1000;
      timer = setInterval(() => tick(durationMs), TICK_MS);
      if (!callMode) autoTimer = setTimeout(finish, durationMs + 400);
    } catch (err) {
      cleanupMedia();
      const name = err && err.name ? err.name : "";
      if (name === "NotAllowedError" || name === "SecurityError") {
        showError(L.error_permission || "Camera or microphone permission was denied.");
      } else if (name === "NotFoundError" || name === "OverconstrainedError") {
        showError(L.error_no_device || "No usable camera or microphone found.");
      } else {
        showError((L.error_generic || "Observation failed.") + " " + (err && err.message ? err.message : ""));
      }
      setStatus("");
      setRunningUI(false);
    } finally {
      startBtn.disabled = false;
    }
  }

  function tick(durationMs) {
    if (!running) return;
    const t = performance.now();
    const elapsedS = Math.floor((t - startTs) / 1000);
    if (!callMode) {
      setProgress((t - startTs) / durationMs);
      setCaption(`${L.capturing || "Observing…"} · ${elapsedS}s`);
    }

    let rms = null;
    let faceFound = false;
    if (analyser && audioBuf) {
      analyser.getFloatTimeDomainData(audioBuf);
      let sum = 0;
      for (let i = 0; i < audioBuf.length; i++) sum += audioBuf[i] * audioBuf[i];
      rms = Math.sqrt(sum / audioBuf.length);
    }

    let jaw = null;
    let blink = null;
    let pitch = null;
    let yaw = null;
    let roll = null;
    let smileL = null;
    let smileR = null;
    let frown = null;
    let brow = null;
    let mouthL = null;
    let mouthR = null;
    let gazeD = null;
    let gazeU = null;
    let gazeL = null;
    let gazeR = null;
    if (video && video.readyState >= 2 && landmarker) {
      try {
        const res = landmarker.detectForVideo(video, t);
        const faces = res && res.faceBlendshapes;
        if (faces && faces.length) {
          faceFound = true;
          const cats = faces[0].categories || [];
          jaw = categoryScore(cats, "jawOpen");
          const bl = categoryScore(cats, "eyeBlinkLeft");
          const br = categoryScore(cats, "eyeBlinkRight");
          if (bl != null || br != null) blink = Math.max(bl || 0, br || 0);
          smileL = categoryScore(cats, "mouthSmileLeft");
          smileR = categoryScore(cats, "mouthSmileRight");
          const fl = categoryScore(cats, "mouthFrownLeft");
          const fr = categoryScore(cats, "mouthFrownRight");
          if (fl != null || fr != null) frown = Math.max(fl || 0, fr || 0);
          brow = categoryScore(cats, "browInnerUp");
          mouthL = categoryScore(cats, "mouthLeft");
          mouthR = categoryScore(cats, "mouthRight");
          // Gaze composites: outward/inward blendshapes combine per side.
          gazeD =
            (categoryScore(cats, "eyeLookDownLeft") || 0) +
            (categoryScore(cats, "eyeLookDownRight") || 0);
          gazeU =
            (categoryScore(cats, "eyeLookUpLeft") || 0) +
            (categoryScore(cats, "eyeLookUpRight") || 0);
          gazeL =
            (categoryScore(cats, "eyeLookOutLeft") || 0) +
            (categoryScore(cats, "eyeLookInRight") || 0);
          gazeR =
            (categoryScore(cats, "eyeLookOutRight") || 0) +
            (categoryScore(cats, "eyeLookInLeft") || 0);
          const mats = res.facialTransformationMatrixes;
          if (mats && mats.length) {
            const e = eulerFromMatrix(mats[0].data || mats[0]);
            if (e) {
              pitch = e.pitch;
              yaw = e.yaw;
              roll = e.roll;
            }
          }
        }
      } catch (err) {
        // Skip corrupted frames; detection continues next tick.
      }
    }

    setFace(faceFound);
    pushWave(rms);
    samples.push({
      t, rms, jaw, blink, pitch, yaw, roll,
      smileL, smileR, frown, brow, mouthL, mouthR,
      gazeD, gazeU, gazeL, gazeR,
    });
    if (callMode) {
      vadUpdate(rms, t);
      callCaption();
    }
  }

  function cleanupMedia() {
    running = false;
    if (timer) {
      clearInterval(timer);
      timer = null;
    }
    if (autoTimer) {
      clearTimeout(autoTimer);
      autoTimer = null;
    }
    if (stream) {
      stream.getTracks().forEach((tr) => tr.stop());
      stream = null;
    }
    if (video) video.srcObject = null;
    if (audioCtx) {
      try {
        audioCtx.close();
      } catch (err) {
        /* already closed */
      }
      audioCtx = null;
      analyser = null;
      audioBuf = null;
    }
    pcmChunks = [];
    pcmTotal = 0;
    pcmTapOk = false;
    if (pcmTap) {
      try {
        pcmTap.disconnect();
      } catch (err) {
        /* ignore */
      }
      pcmTap = null;
    }
    if (pcmMute) {
      try {
        pcmMute.disconnect();
      } catch (err) {
        /* ignore */
      }
      pcmMute = null;
    }
    vad.inSpeech = false;
    vad.speechTicks = 0;
    vad.silenceTicks = 0;
    vad.awaitingReply = false;
  }

  // ---------- call mode: PCM tap + energy VAD + reply playback ----------

  async function setupPcmTap(sourceNode) {
    try {
      const workletCode =
        "class Tap extends AudioWorkletProcessor{" +
        "process(inputs){const ch=inputs[0]&&inputs[0][0];" +
        "if(ch&&ch.length)this.port.postMessage(ch.slice(0));return true}}" +
        "registerProcessor('dlv-pcm-tap',Tap);";
      const url = URL.createObjectURL(new Blob([workletCode], { type: "text/javascript" }));
      await audioCtx.audioWorklet.addModule(url);
      URL.revokeObjectURL(url);
      pcmTap = new AudioWorkletNode(audioCtx, "dlv-pcm-tap");
      pcmTap.port.onmessage = (ev) => {
        const d = ev.data;
        if (!d || !d.length) return;
        pcmChunks.push({ start: pcmTotal, data: d });
        pcmTotal += d.length;
        const keepFrom = pcmTotal - audioCtx.sampleRate * 20;
        while (
          pcmChunks.length &&
          pcmChunks[0].start + pcmChunks[0].data.length < keepFrom
        ) {
          pcmChunks.shift();
        }
      };
      sourceNode.connect(pcmTap);
      pcmMute = audioCtx.createGain();
      pcmMute.gain.value = 0;
      pcmTap.connect(pcmMute);
      pcmMute.connect(audioCtx.destination);
      pcmTapOk = true;
    } catch (err) {
      pcmTapOk = false;
    }
  }

  function vadUpdate(rms, t) {
    if (!pcmTapOk || !audioCtx || !running) return;
    if (vad.awaitingReply) {
      if (t - vad.awaitingSince > 45000) vad.awaitingReply = false;
      return;
    }
    if (replyAudio && !replyAudio.paused) return;
    if (rms == null || !isFinite(rms)) return;
    const thr = Math.max(vad.noise * 3.0, 0.01);
    const loud = rms > thr;
    if (!vad.inSpeech) {
      vad.noise = clamp(vad.noise * 0.97 + Math.min(rms, thr) * 0.03, 0.002, 0.08);
      if (loud) {
        vad.inSpeech = true;
        vad.speechTicks = 1;
        vad.silenceTicks = 0;
        vad.speechStartSample = Math.max(
          0,
          pcmTotal - Math.floor(audioCtx.sampleRate * 0.2),
        );
      }
      return;
    }
    if (loud) {
      vad.speechTicks += 1;
      vad.silenceTicks = 0;
    } else {
      vad.silenceTicks += 1;
    }
    const speechMs = vad.speechTicks * TICK_MS;
    const silenceMs = vad.silenceTicks * TICK_MS;
    const durS = (pcmTotal - vad.speechStartSample) / audioCtx.sampleRate;
    if ((silenceMs >= 900 && speechMs >= 350) || durS >= 15) {
      emitUtterance();
    }
  }

  function emitUtterance() {
    vad.inSpeech = false;
    vad.speechTicks = 0;
    vad.silenceTicks = 0;
    if (!pcmTapOk || !audioCtx) return;
    const endSample = pcmTotal;
    const startSample = Math.max(
      0,
      Math.min(vad.speechStartSample, endSample - audioCtx.sampleRate * 16),
    );
    let total = 0;
    const parts = [];
    for (const c of pcmChunks) {
      const cs = c.start;
      const ce = c.start + c.data.length;
      if (ce <= startSample || cs >= endSample) continue;
      const a = Math.max(0, startSample - cs);
      const b = Math.min(c.data.length, endSample - cs);
      if (b > a) {
        parts.push(c.data.subarray(a, b));
        total += b - a;
      }
    }
    if (total < audioCtx.sampleRate * 0.35) return;
    const merged = new Float32Array(total);
    let off = 0;
    for (const p of parts) {
      merged.set(p, off);
      off += p.length;
    }
    const pcm16 = resampleTo16k(merged, audioCtx.sampleRate);
    const wav = encodeWavPcm16(pcm16, 16000);
    vad.awaitingReply = true;
    vad.awaitingSince = performance.now();
    vad.speechStartSample = 0;
    setStatus(L.call_waiting || "Sent — waiting for reply…");
    setStateValue("utterance", {
      pcm: bytesToBase64(wav),
      ts: Date.now(),
      dur_ms: Math.round((total / audioCtx.sampleRate) * 1000),
    });
  }

  // Live call state goes to the status line (top overlay); the caption
  // line keeps the latest reply as a persistent subtitle.
  function callCaption() {
    if (replyAudio && !replyAudio.paused) {
      setStatus(L.call_speaking || "DyadLink is speaking…");
    } else if (vad.awaitingReply) {
      setStatus(L.call_waiting || "Sent — waiting for reply…");
    } else {
      setStatus(L.call_listening || "Listening — just speak");
    }
  }

  function playReply(reply) {
    lastReplyId = reply.id;
    vad.awaitingReply = false;
    let txt = reply.text || "";
    if (txt.length > 600) txt = `${txt.slice(0, 597)}…`;
    setCaption(txt ? `DyadLink: ${txt}` : L.call_error || "");
    if (reply.audio_b64) {
      try {
        const bin = atob(reply.audio_b64);
        const bytes = new Uint8Array(bin.length);
        for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
        const url = URL.createObjectURL(new Blob([bytes], { type: "audio/mpeg" }));
        if (replyAudio) {
          try {
            replyAudio.pause();
          } catch (err) {
            /* ignore */
          }
          if (replyAudio._url) URL.revokeObjectURL(replyAudio._url);
        }
        replyAudio = new Audio(url);
        replyAudio._url = url;
        replyAudio.onended = () => {
          URL.revokeObjectURL(url);
          replyAudio = null;
          if (running) callCaption();
        };
        replyAudio.onerror = () => {
          URL.revokeObjectURL(url);
          replyAudio = null;
          if (running) callCaption();
        };
        const p = replyAudio.play();
        if (p && p.catch) p.catch(() => {});
      } catch (err) {
        /* ignore */
      }
    } else if (running) {
      setTimeout(() => {
        if (running && !vad.awaitingReply) callCaption();
      }, 5000);
    }
  }

  function applyLabels() {
    text(startBtn, callMode ? "call_start" : "start", callMode ? "Start call" : "Start observation");
    text(stopBtn, callMode ? "call_end" : "stop", "Stop");
    text(noteEl, "privacy_note", "Video and audio stay on this device.");
    text(doneEl, "done_hint", "");
    text(unmuteBtn, "enable_sound", "Tap for sound");
    if (!running) text(captionEl, "idle_hint", "");
  }

  function update(newCfg) {
    const L2 = newCfg && newCfg.strings ? newCfg.strings : null;
    if (L2 && Object.keys(L2).length) L = L2;
    applyLabels();
    const reply = newCfg ? newCfg.call_reply : null;
    if (reply && reply.id != null && reply.id !== lastReplyId) {
      playReply(reply);
    }
  }

  function finish() {
    if (!running) return;
    cleanupMedia();
    const payload = computeFeatures(samples);
    payload.captured_at = Date.now();
    // Wall-clock id: unique across component remounts (dialog open/close,
    // language switch), so Python-side dedupe never rejects a real capture.
    payload.capture_id = payload.captured_at;
    setStateValue("result", payload);
    setRunningUI(false);
    setProgress(1);
    setCaption("");
    if (doneEl) {
      doneEl.textContent = L.done_hint || "";
      doneEl.hidden = false;
    }
    setStatus("");
  }

  function computeFeatures(s) {
    const out = { version: 1 };
    const n = s.length;
    if (n < 30) {
      out.error = "too_short";
      return out;
    }
    const t0 = s[0].t;
    const t1 = s[n - 1].t;
    const durationS = (t1 - t0) / 1000;
    out.duration_s = r3(durationS);
    out.frames = n;

    const rmsVals = [];
    const jawVals = [];
    const pitchVals = [];
    const yawVals = [];
    const rollVals = [];
    const blinkVals = [];
    for (const f of s) {
      if (f.rms != null && isFinite(f.rms)) rmsVals.push(f.rms);
      if (f.jaw != null && isFinite(f.jaw)) jawVals.push(f.jaw);
      if (f.pitch != null && isFinite(f.pitch)) pitchVals.push(f.pitch);
      if (f.yaw != null && isFinite(f.yaw)) yawVals.push(f.yaw);
      if (f.roll != null && isFinite(f.roll)) rollVals.push(f.roll);
      if (f.blink != null && isFinite(f.blink)) blinkVals.push(f.blink);
    }
    out.face_ratio = r3(jawVals.length / n);
    out.has_audio = rmsVals.length > n * 0.5;

    // ---------- audio envelope (mirrors modules/audio_features.py) ----------
    let floor = null;
    let voicedFlags = null;
    if (out.has_audio) {
      const sorted = [...rmsVals].sort((a, b) => a - b);
      const noise = percentile(sorted, 0.2);
      const peak = percentile(sorted, 0.9);
      floor = Math.max(Math.min(noise * 2.5, peak * 0.3), 0.008);
      voicedFlags = s.map((f) => (f.rms != null ? f.rms > floor : null));
      const voicedCount = voicedFlags.filter((v) => v === true).length;
      out.voiced_ratio = r3(voicedCount / n);
      if (voicedCount / n >= 0.05) {
        const hop = TICK_MS / 1000;
        let pauses = 0;
        let pauseLens = [];
        let run = 0;
        for (const v of voicedFlags) {
          if (v === false) {
            run += 1;
          } else {
            if (run * hop >= MIN_PAUSE_S) {
              pauses += 1;
              pauseLens.push(run * hop);
            }
            run = 0;
          }
        }
        if (run * hop >= MIN_PAUSE_S) {
          pauses += 1;
          pauseLens.push(run * hop);
        }
        out.silence_ratio = r3(1 - out.voiced_ratio);
        out.pause_count = pauses;
        out.pause_per_min = Math.round((pauses / (durationS / 60)) * 10) / 10;
        out.mean_pause_s =
          Math.round((pauseLens.length ? pauseLens.reduce((a, b) => a + b, 0) / pauseLens.length : 0) * 100) / 100;
        if (pauseLens.length) {
          out.longest_pause_s = Math.round(Math.max(...pauseLens) * 100) / 100;
        }

        // Voiced-segment stats: mean phrase length (>=2 frames = ~67 ms).
        let segRun = 0;
        const segLens = [];
        for (const v of voicedFlags) {
          if (v === true) {
            segRun += 1;
          } else {
            if (segRun >= 2) segLens.push(segRun * hop);
            segRun = 0;
          }
        }
        if (segRun >= 2) segLens.push(segRun * hop);
        if (segLens.length >= 2) {
          out.mean_voiced_segment_s =
            Math.round((segLens.reduce((a, b) => a + b, 0) / segLens.length) * 100) / 100;
        }

        // Within-capture drift: silence ratio in the second half minus the
        // first half — a rough fatigue signal, not a score.
        const mid = Math.floor(voicedFlags.length / 2);
        const halves = [voicedFlags.slice(0, mid), voicedFlags.slice(mid)];
        if (halves[0].length >= TARGET_HZ && halves[1].length >= TARGET_HZ) {
          const sil = (arr) =>
            1 - arr.filter((v) => v === true).length / arr.filter((v) => v != null).length;
          const s0 = sil(halves[0]);
          const s1 = sil(halves[1]);
          if (isFinite(s0) && isFinite(s1)) out.drift_silence = r3(s1 - s0);
        }
      }
    }

    // ---------- resample jaw + rms onto a uniform 30 Hz grid ----------
    const times = s.map((f) => f.t);
    const grid = [];
    for (let i = 0; i <= Math.floor((t1 - t0) / TICK_MS); i++) grid.push(t0 + i * TICK_MS);
    const maxGap = TICK_MS * 2.5;
    const jawGrid = resample(times, s.map((f) => f.jaw), grid, maxGap);
    const rmsGrid = resample(times, s.map((f) => f.rms), grid, maxGap);
    const voicedGrid = voicedFlags
      ? resample(times, voicedFlags.map((v) => (v === true ? 1 : v === false ? 0 : null)), grid, maxGap)
      : null;

    // ---------- jaw_range ----------
    if (jawVals.length >= 10) {
      const js = [...jawVals].sort((a, b) => a - b);
      out.jaw_range = r4(percentile(js, 0.95) - percentile(js, 0.05));
    }

    // ---------- visual_syllable_rate ----------
    const jawSmooth = movingAverage(jawGrid, 5);
    const band = bandpass(jawSmooth.map((v) => (v == null ? 0 : v)), TARGET_HZ, 2, 8);
    const trimN = Math.floor(TARGET_HZ * 0.4);
    const bandTrim = band.slice(trimN, Math.max(trimN + 1, band.length - trimN));
    const peaks = detectPeaks(bandTrim, 4);
    const usable = bandTrim.filter((v) => v != null).length;
    if (usable > TARGET_HZ * 3 && durationS > 3) {
      out.visual_syllable_rate = Math.round((peaks.length / (usable / TARGET_HZ)) * 60 * 10) / 10;
    }

    // ---------- energy-envelope speech rate, tempo CV, loudness SD ----------
    if (voicedGrid && durationS > 3) {
      const eBand = bandpass(
        rmsGrid.map((v) => (v == null ? 0 : v)), TARGET_HZ, 2, 8
      );
      const ePeaks = detectPeaks(
        eBand.slice(trimN, Math.max(trimN + 1, eBand.length - trimN)), 4
      ).map((i) => i + trimN);
      const vPeaks = ePeaks.filter((i) => voicedGrid[i] === 1);
      const voicedSec = voicedGrid.filter((v) => v === 1).length / TARGET_HZ;
      if (voicedSec > 2 && vPeaks.length >= 3) {
        out.speech_rate_proxy =
          Math.round((vPeaks.length / voicedSec) * 60 * 10) / 10;
        const intervals = [];
        for (let k = 1; k < vPeaks.length; k++) {
          intervals.push((vPeaks[k] - vPeaks[k - 1]) / TARGET_HZ);
        }
        const meanIv = intervals.reduce((a, b) => a + b, 0) / intervals.length;
        if (meanIv > 0) {
          out.speech_tempo_cv = r3(stddev(intervals) / meanIv);
        }
      }
      const voicedRms = rmsGrid.filter(
        (v, i) => voicedGrid[i] === 1 && v != null
      );
      if (voicedRms.length >= 10) {
        out.loudness_sd = r4(stddev(voicedRms));
      }
    }

    // ---------- blink_rate + duration + PERCLOS (hysteresis) ----------
    {
      let state = "open";
      let count = 0;
      let lastEvent = -Infinity;
      let closeStart = 0;
      const blinkLens = [];
      for (const f of s) {
        if (f.blink == null) continue;
        if (state === "open" && f.blink > BLINK_HIGH) {
          state = "closed";
          closeStart = f.t;
        } else if (state === "closed" && f.blink < BLINK_LOW) {
          if (f.t - lastEvent > BLINK_MIN_GAP_MS) {
            count += 1;
            lastEvent = f.t;
            blinkLens.push(f.t - closeStart);
          }
          state = "open";
        }
      }
      if (blinkVals.length > n * 0.3 && durationS > 3) {
        out.blink_rate = Math.round((count / durationS) * 60 * 10) / 10;
        // PERCLOS: fraction of time the eyes are >=80% closed.
        const closed = blinkVals.filter((v) => v > 0.8).length;
        out.perclos = r3(closed / blinkVals.length);
        if (blinkLens.length >= 3) {
          out.blink_duration_ms = Math.round(
            blinkLens.reduce((a, b) => a + b, 0) / blinkLens.length
          );
        }
      }
    }

    // ---------- head euler SDs (degrees) ----------
    const sdPitch = stddev(pitchVals);
    const sdYaw = stddev(yawVals);
    const sdRoll = stddev(rollVals);
    if (pitchVals.length > n * 0.3) {
      out.head_pitch_sd = Math.round(sdPitch * 100) / 100;
      out.head_yaw_sd = Math.round(sdYaw * 100) / 100;
      out.head_roll_sd = Math.round(sdRoll * 100) / 100;
    }

    // ---------- head movement energy, stillness, nods ----------
    {
      let prevT = null;
      let prevP = null;
      let prevY = null;
      let prevR = null;
      const speeds = [];
      for (const f of s) {
        if (f.pitch == null || f.yaw == null || f.roll == null) {
          prevT = null;
          continue;
        }
        if (prevT != null && f.t - prevT > 0 && f.t - prevT < 200) {
          const deg =
            (Math.abs(f.pitch - prevP) +
              Math.abs(f.yaw - prevY) +
              Math.abs(f.roll - prevR)) /
            ((f.t - prevT) / 1000);
          speeds.push(deg);
        }
        prevT = f.t;
        prevP = f.pitch;
        prevY = f.yaw;
        prevR = f.roll;
      }
      if (speeds.length > n * 0.2) {
        out.head_movement_energy =
          Math.round((speeds.reduce((a, b) => a + b, 0) / speeds.length) * 10) / 10;
        out.stillness_ratio = r3(speeds.filter((v) => v < 3).length / speeds.length);
      }
      const nodGrid = resample(times, s.map((f) => f.pitch), grid, maxGap);
      const nodBand = bandpass(
        nodGrid.map((v) => (v == null ? 0 : v)), TARGET_HZ, 0.5, 3
      );
      const nodTrim = nodBand.slice(
        trimN, Math.max(trimN + 1, nodBand.length - trimN)
      );
      const nodPeaks = detectPeaks(nodTrim, 8);
      const nodUsable = nodTrim.length;
      if (nodUsable > TARGET_HZ * 3 && durationS > 3) {
        out.nod_rate =
          Math.round((nodPeaks.length / (nodUsable / TARGET_HZ)) * 60 * 10) / 10;
      }
    }

    // ---------- silence decomposition + AV synchrony ----------
    if (voicedGrid && floor != null && rmsGrid.some((v) => v != null)) {
      const silentIdx = [];
      const voicedIdx = [];
      for (let i = 0; i < grid.length; i++) {
        if (voicedGrid[i] === 0) silentIdx.push(i);
        else if (voicedGrid[i] === 1) voicedIdx.push(i);
      }
      // jaw velocity per grid step
      const jawVel = new Array(grid.length).fill(null);
      for (let i = 1; i < grid.length; i++) {
        if (jawGrid[i] != null && jawGrid[i - 1] != null) {
          jawVel[i] = Math.abs(jawGrid[i] - jawGrid[i - 1]);
        }
      }
      const voicedVel = voicedIdx.map((i) => jawVel[i]).filter((v) => v != null);
      const speechP75 = percentile([...voicedVel].sort((a, b) => a - b), 0.75);
      if (voicedVel.length >= MIN_SPEECH_FRAMES && speechP75 != null && speechP75 > 0) {
        const thr = Math.max(3 * speechP75, 0.01);
        let art = 0;
        let froz = 0;
        let covered = 0;
        for (const i of silentIdx) {
          const v = jawVel[i];
          if (v == null) continue;
          covered += 1;
          if (v > thr) art += 1;
          else froz += 1;
        }
        if (covered >= 10) {
          out.articulating_silence_ratio = r4(art / covered);
          out.frozen_silence_ratio = r4(froz / covered);
          out.silent_coverage = r3(covered / Math.max(silentIdx.length, 1));
        }
      }

      // AV synchrony: normalized cross-correlation, lag ±MAX_LAG_FRAMES
      if (voicedIdx.length >= MIN_SPEECH_FRAMES && durationS >= 5) {
        const aFull = rmsGrid.map((v) => (v == null ? null : v));
        const jFull = jawGrid.map((v) => (v == null ? null : v));
        // restrict to contiguous span where both exist
        let lo = 0;
        let hi = grid.length - 1;
        while (lo < grid.length && (aFull[lo] == null || jFull[lo] == null)) lo++;
        while (hi > lo && (aFull[hi] == null || jFull[hi] == null)) hi--;
        if (hi - lo > TARGET_HZ * 4) {
          const a = aFull.slice(lo, hi + 1);
          const j = jSmooth(jFull.slice(lo, hi + 1), 3);
          let best = null;
          let bestLag = 0;
          for (let lag = -MAX_LAG_FRAMES; lag <= MAX_LAG_FRAMES; lag++) {
            let xs = [];
            let ys = [];
            for (let i = 0; i < a.length; i++) {
              const k = i + lag;
              if (k < 0 || k >= j.length) continue;
              if (a[i] == null || j[k] == null) continue;
              xs.push(a[i]);
              ys.push(j[k]);
            }
            const c = pearson(xs, ys);
            if (c != null && (best == null || c > best)) {
              best = c;
              bestLag = lag;
            }
          }
          if (best != null) {
            out.av_synchrony = r4(best);
            out.av_lag_ms = Math.round(bestLag * (1000 / TARGET_HZ));
          }
        }
      }
    }

    // ---------- expression, asymmetry, gaze ----------
    {
      const smileMean = [];
      const frownArr = [];
      const browArr = [];
      const asymArr = [];
      let gazeShifts = 0;
      let gazeDownFrames = 0;
      let gazeFrames = 0;
      let lastDir = null;
      for (const f of s) {
        if (f.smileL != null && f.smileR != null) {
          smileMean.push((f.smileL + f.smileR) / 2);
        }
        if (f.frown != null) frownArr.push(f.frown);
        if (f.brow != null) browArr.push(f.brow);
        if (
          f.smileL != null &&
          f.smileR != null &&
          f.mouthL != null &&
          f.mouthR != null
        ) {
          asymArr.push(
            Math.abs(f.smileL - f.smileR) + Math.abs(f.mouthL - f.mouthR)
          );
        }
        if (
          f.gazeD != null &&
          f.gazeU != null &&
          f.gazeL != null &&
          f.gazeR != null
        ) {
          gazeFrames += 1;
          const dirs = [
            ["D", f.gazeD],
            ["U", f.gazeU],
            ["L", f.gazeL],
            ["R", f.gazeR],
          ];
          dirs.sort((a, b) => b[1] - a[1]);
          const dir = dirs[0][1] > 0.5 ? dirs[0][0] : "C";
          if (dir === "D") gazeDownFrames += 1;
          if (dir !== "C") {
            if (lastDir != null && dir !== lastDir) gazeShifts += 1;
            lastDir = dir;
          }
        }
      }
      if (asymArr.length > n * 0.3) {
        out.mouth_asymmetry = r4(
          asymArr.reduce((a, b) => a + b, 0) / asymArr.length
        );
      }
      if (smileMean.length > n * 0.3) {
        out.smile_ratio = r3(
          smileMean.filter((v) => v > 0.4).length / smileMean.length
        );
      }
      if (frownArr.length > n * 0.3) {
        out.frown_ratio = r3(
          frownArr.filter((v) => v > 0.4).length / frownArr.length
        );
      }
      // Expressivity: mean temporal SD of the key action units — a flat
      // profile indicates reduced facial dynamism.
      const series = [smileMean, frownArr, browArr, jawVals].filter(
        (arr) => arr.length >= 10
      );
      if (series.length >= 3) {
        out.expressivity_sd = r4(
          series.reduce((a, arr) => a + stddev(arr), 0) / series.length
        );
      }
      if (gazeFrames > n * 0.3 && durationS > 3) {
        out.gaze_down_ratio = r3(gazeDownFrames / gazeFrames);
        out.gaze_shift_rate =
          Math.round((gazeShifts / durationS) * 60 * 10) / 10;
      }
    }

    return out;
  }

  function jSmooth(arr, w) {
    return movingAverage(arr, w);
  }

  if (startBtn) startBtn.addEventListener("click", start);
  if (stopBtn) stopBtn.addEventListener("click", () => finish());
  if (unmuteBtn) {
    unmuteBtn.addEventListener("click", () => {
      if (audioCtx && audioCtx.state === "suspended") {
        audioCtx.resume().then(() => {
          if (unmuteBtn) unmuteBtn.hidden = true;
        });
      } else if (unmuteBtn) {
        unmuteBtn.hidden = true;
      }
    });
  }
  window.addEventListener("pagehide", cleanupMedia);

  // ChatGPT-style inline capture: the host page can request an immediate
  // start (e.g. from a camera button) by passing data.autostart.
  if (cfg.autostart) {
    setTimeout(() => {
      if (!running) start();
    }, 80);
  }

  const session = {
    update,
    cleanup: () => {
      window.removeEventListener("pagehide", cleanupMedia);
      if (replyAudio) {
        try {
          replyAudio.pause();
        } catch (err) {
          /* ignore */
        }
        if (replyAudio._url) URL.revokeObjectURL(replyAudio._url);
        replyAudio = null;
      }
      cleanupMedia();
      if (landmarker && typeof landmarker.close === "function") {
        try {
          landmarker.close();
        } catch (err) {
          /* ignore */
        }
      }
    },
  };
  return session;
}

export default function DyadLinkVisionObserver(component) {
  const { data, parentElement } = component;
  const cfg = data && typeof data === "object" ? data : {};
  const root = parentElement.querySelector(".dlv");
  if (!root) return () => {};
  const existing = root.__dlvSession;
  if (existing) {
    existing.update(cfg);
    return existing.cleanup;
  }
  const session = createSession(root, parentElement, cfg, component.setStateValue);
  root.__dlvSession = session;
  return session.cleanup;
}
