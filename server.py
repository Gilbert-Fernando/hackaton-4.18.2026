from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from src.predict_realtime import PredictionSmoother, predict_from_hands
from utils.preprocess import landmarks_to_feature_vector

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))


@dataclass
class _Hand:
    feature_vector: np.ndarray


class _Point(BaseModel):
    x: float
    y: float
    z: float = 0.0


class PredictRequest(BaseModel):
    hands: list[list[_Point]] = Field(default_factory=list)


def _features_from_landmarks(hand_points: list[_Point]) -> np.ndarray:
    class _Lm:
        def __init__(self, x: float, y: float, z: float) -> None:
            self.x = float(x)
            self.y = float(y)
            self.z = float(z)

    class _Tmp:
        def __init__(self, points: list[_Point]) -> None:
            self.landmark = [_Lm(p.x, p.y, p.z) for p in points]

    return landmarks_to_feature_vector(_Tmp(hand_points))


def _load_models() -> dict[str, Any]:
    models_dir = os.path.join(PROJECT_ROOT, "models")
    return {
        "letters_model": joblib.load(os.path.join(models_dir, "letters_model.pkl")),
        "letters_encoder": joblib.load(os.path.join(models_dir, "letters_encoder.pkl")),
        "phrases_model": joblib.load(os.path.join(models_dir, "phrases_model.pkl")),
        "phrases_encoder": joblib.load(os.path.join(models_dir, "phrases_encoder.pkl")),
    }


app = FastAPI()
_models = _load_models()
_letters_smoother = PredictionSmoother(window_size=10, dominance=0.7)
_phrases_smoother = PredictionSmoother(window_size=10, dominance=0.7)
_last_req_ts = 0.0


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(
        """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>ASL Realtime (Browser Camera)</title>
    <style>
      body{margin:0;background:#000;color:#fff;font-family:system-ui,Segoe UI,Arial;}
      .wrap{max-width:1100px;margin:0 auto;padding:12px;}
      .grid{display:grid;grid-template-columns:2fr 1fr;gap:12px;align-items:start;}
      .card{border:1px solid #222;border-radius:10px;overflow:hidden}
      .panel{border:1px solid #222;border-radius:10px;padding:10px;background:#0a0a0a}
      video,canvas{width:100%;display:block;background:#000}
      .row{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
      button{background:#111;border:1px solid #333;color:#fff;border-radius:10px;padding:8px 10px;cursor:pointer}
      button:hover{background:#151515}
      .muted{color:#a3a3a3}
      .bar{height:10px;background:#111;border:1px solid #222;border-radius:999px;overflow:hidden}
      .bar > div{height:100%;width:0;background:#22c55e}
      textarea{width:100%;min-height:90px;background:#050505;color:#fff;border:1px solid #222;border-radius:10px;padding:10px;resize:vertical}
      pre{white-space:pre-wrap;word-break:break-word;font-size:12px;margin:0}
    </style>
    <script src="https://cdn.jsdelivr.net/npm/@mediapipe/drawing_utils/drawing_utils.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/@mediapipe/hands/hands.js"></script>
  </head>
  <body>
    <div class="wrap">
      <div class="grid">
        <div class="card">
          <div style="position:relative">
            <video id="video" playsinline autoplay muted></video>
            <canvas id="overlay" style="position:absolute;left:0;top:0"></canvas>
          </div>
        </div>

        <div class="panel">
          <div class="row" style="justify-content:space-between">
            <div><b>Status:</b> <span id="status" class="muted">Starting…</span> <span class="muted">| FPS</span> <span id="fps">0.0</span></div>
            <button id="ttsEnable">Enable TTS</button>
          </div>

          <div style="margin-top:10px">
            <div><b>Label:</b> <span id="label">—</span></div>
            <div><b>Mode:</b> <span id="mode">—</span> <span class="muted">• Hands</span> <span id="hands">0</span></div>
            <div><b>Conf:</b> <span id="conf">0.00</span></div>
          </div>

          <div style="margin-top:12px"><b>Output</b> <span class="muted">(commit behavior)</span></div>
          <textarea id="built" readonly></textarea>
          <div style="margin-top:10px">
            <div class="row" style="justify-content:space-between"><span class="muted">Commit in</span><span id="stableSeconds" class="muted">—</span></div>
            <div class="bar" style="margin-top:6px"><div id="stableBar"></div></div>
          </div>
          <div class="row" style="margin-top:8px">
            <button id="backspace">Backspace</button>
            <button id="reset">Reset</button>
          </div>

          <div style="margin-top:12px" class="muted">Raw /predict response</div>
          <pre id="raw">—</pre>
        </div>
      </div>
    </div>

    <script>
      const statusEl = document.getElementById('status');
      const fpsEl = document.getElementById('fps');
      const labelEl = document.getElementById('label');
      const modeEl = document.getElementById('mode');
      const handsEl = document.getElementById('hands');
      const confEl = document.getElementById('conf');
      const stableSecondsEl = document.getElementById('stableSeconds');
      const stableBarEl = document.getElementById('stableBar');
      const builtEl = document.getElementById('built');
      const ttsEnableBtn = document.getElementById('ttsEnable');
      const rawEl = document.getElementById('raw');
      const videoEl = document.getElementById('video');
      const canvasEl = document.getElementById('overlay');
      const ctx = canvasEl.getContext('2d');

      const CONFIDENCE_THRESHOLD = 0.60;
      const STABLE_SECONDS = 1.2;
      const COOLDOWN_SECONDS = 1.5;

      const ui = {
        built: '',
        pendingLabel: '',
        pendingSince: 0,
        lastCommitted: '',
        lastCommitTs: 0,
        enterArmed: true,
        enterReleaseSince: null,
        ttsEnabled: false,
        lastFpsTs: 0,
      };

      function speak(text) {
        if (!ui.ttsEnabled) return;
        try {
          const u = new SpeechSynthesisUtterance(text);
          u.rate = 1.0;
          window.speechSynthesis.cancel();
          window.speechSynthesis.speak(u);
        } catch (e) {}
      }

      function updateStableUi(secondsLeft, progress) {
        stableSecondsEl.textContent = secondsLeft === null ? '—' : secondsLeft.toFixed(1) + 's';
        const pct = Math.max(0, Math.min(1, progress ?? 0)) * 100;
        stableBarEl.style.width = pct + '%';
      }

      function updateBuilt() {
        builtEl.value = ui.built;
      }

      function commitIfStable(payload) {
        const label = String(payload.label || '').trim();
        const rawLabel = String(payload.raw_label || '').trim();
        const confidence = Number(payload.confidence ?? 0);
        const rawConfidence = Number(payload.raw_confidence ?? 0);
        const now = Date.now();
        const upper = label.toUpperCase();
        const rawUpper = rawLabel.toUpperCase();

        const inCooldown =
          upper !== 'ENTER' &&
          upper !== 'SPACE' &&
          label &&
          label === ui.lastCommitted &&
          now - ui.lastCommitTs < COOLDOWN_SECONDS * 1000;

        if (!label || confidence < CONFIDENCE_THRESHOLD) {
          if (!inCooldown) {
            ui.pendingLabel = '';
            ui.pendingSince = 0;
          }

          if (rawUpper !== 'ENTER' || rawConfidence < CONFIDENCE_THRESHOLD) {
            if (ui.enterReleaseSince === null) ui.enterReleaseSince = now;
            if (now - ui.enterReleaseSince >= 250) ui.enterArmed = true;
          } else {
            ui.enterReleaseSince = null;
          }

          updateStableUi(null, 0);
          return;
        }

        if (rawUpper !== 'ENTER' || rawConfidence < CONFIDENCE_THRESHOLD) {
          if (ui.enterReleaseSince === null) ui.enterReleaseSince = now;
          if (now - ui.enterReleaseSince >= 250) ui.enterArmed = true;
        } else {
          ui.enterReleaseSince = null;
        }

        if (label !== ui.pendingLabel) {
          ui.pendingLabel = label;
          ui.pendingSince = now;
          updateStableUi(STABLE_SECONDS, 0);
          return;
        }

        const elapsed = (now - ui.pendingSince) / 1000;
        const left = Math.max(0, STABLE_SECONDS - elapsed);
        updateStableUi(left, Math.min(1, elapsed / STABLE_SECONDS));
        if (elapsed < STABLE_SECONDS) return;

        if (upper === 'ENTER') {
          if (!ui.enterArmed) return;
          ui.enterArmed = false;
          ui.built += '\n';
          updateBuilt();
          ui.lastCommitted = label;
          ui.lastCommitTs = now;
          speak('Enter');
          return;
        }

        if (upper === 'SPACE') {
          ui.built += ' ';
          updateBuilt();
          ui.lastCommitted = label;
          ui.lastCommitTs = now;
          speak('Space');
          return;
        }

        if (inCooldown) return;

        ui.built += label;
        updateBuilt();
        ui.lastCommitted = label;
        ui.lastCommitTs = now;
        speak(label);
      }

      document.getElementById('reset').onclick = () => {
        ui.built = '';
        ui.lastCommitted = '';
        ui.pendingLabel = '';
        ui.pendingSince = 0;
        ui.enterArmed = true;
        ui.enterReleaseSince = null;
        updateBuilt();
      };

      document.getElementById('backspace').onclick = () => {
        ui.built = ui.built.slice(0, -1);
        updateBuilt();
      };

      ttsEnableBtn.onclick = () => {
        ui.ttsEnabled = !ui.ttsEnabled;
        ttsEnableBtn.textContent = ui.ttsEnabled ? 'Disable TTS' : 'Enable TTS';
      };

      window.addEventListener('error', (ev) => {
        try {
          statusEl.textContent = 'Error: ' + (ev && ev.message ? ev.message : 'unknown');
        } catch (e) {}
      });

      let inflight = false;
      let lastSent = 0;

      async function sendToBackend(hands) {
        const now = performance.now();
        if (inflight) return;
        if (now - lastSent < 90) return;
        lastSent = now;
        inflight = true;

        try {
          const res = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ hands })
          });
          if (!res.ok) throw new Error('HTTP ' + res.status);
          const payload = await res.json();
          rawEl.textContent = JSON.stringify(payload, null, 2);

          statusEl.textContent = 'Live';
          labelEl.textContent = payload.label || '—';
          modeEl.textContent = payload.mode || '—';
          handsEl.textContent = String(payload.hands ?? 0);
          confEl.textContent = Number(payload.confidence ?? 0).toFixed(2);

          const ts = Number(payload.ts ?? 0);
          if (ts && ui.lastFpsTs && ts !== ui.lastFpsTs) {
            fpsEl.textContent = (1000 / Math.max(1, ts - ui.lastFpsTs)).toFixed(1);
          }
          ui.lastFpsTs = ts || ui.lastFpsTs;

          commitIfStable(payload);
        } catch (e) {
          statusEl.textContent = 'Error: ' + (e && e.message ? e.message : String(e));
        } finally {
          inflight = false;
        }
      }

      function resizeCanvas() {
        const w = videoEl.videoWidth || 0;
        const h = videoEl.videoHeight || 0;
        if (!w || !h) return;
        canvasEl.width = w;
        canvasEl.height = h;
        canvasEl.style.width = '100%';
        canvasEl.style.height = 'auto';
      }

      if (typeof Hands === 'undefined') {
        statusEl.textContent = 'Error: MediaPipe Hands failed to load';
        throw new Error('MediaPipe Hands failed to load');
      }

      const hands = new Hands({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
      });

      hands.setOptions({
        maxNumHands: 2,
        modelComplexity: 1,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });

      hands.onResults((results) => {
        resizeCanvas();
        ctx.save();
        ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
        ctx.drawImage(results.image, 0, 0, canvasEl.width, canvasEl.height);

        const outHands = [];
        if (results.multiHandLandmarks && results.multiHandLandmarks.length) {
          for (const lms of results.multiHandLandmarks.slice(0, 2)) {
            drawConnectors(ctx, lms, HAND_CONNECTIONS, { color: '#22c55e', lineWidth: 2 });
            drawLandmarks(ctx, lms, { color: '#60a5fa', lineWidth: 1 });
            outHands.push(lms.map(p => ({ x: p.x, y: p.y, z: p.z || 0 })));
          }
        } else {
          ctx.font = '32px system-ui';
          ctx.fillStyle = 'red';
          ctx.fillText('No hand detected', 20, 50);
        }
        ctx.restore();

        sendToBackend(outHands);
      });

      updateBuilt();
      statusEl.textContent = 'Starting camera…';

      async function startCamera() {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
          throw new Error('getUserMedia not available');
        }
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: { ideal: 960 }, height: { ideal: 540 } },
          audio: false,
        });
        videoEl.srcObject = stream;
        await videoEl.play();
      }

      let running = true;
      async function frameLoop() {
        if (!running) return;
        try {
          if (videoEl.readyState >= 2) {
            await hands.send({ image: videoEl });
          }
        } catch (e) {
          statusEl.textContent = 'Error: ' + (e && e.message ? e.message : String(e));
          running = false;
          return;
        }
        requestAnimationFrame(frameLoop);
      }

      (async () => {
        try {
          await startCamera();
          statusEl.textContent = 'Camera live';
          requestAnimationFrame(frameLoop);
        } catch (e) {
          statusEl.textContent = 'Error: ' + (e && e.message ? e.message : String(e));
          rawEl.textContent = String(e);
        }
      })();
    </script>
  </body>
</html>
"""
    )


@app.post("/predict")
async def predict(req: PredictRequest) -> JSONResponse:
    global _last_req_ts

    hands_in = (req.hands or [])[:2]
    hands_objs: list[_Hand] = []

    for pts in hands_in:
        if len(pts) != 21:
            continue
        fv = _features_from_landmarks(pts)
        hands_objs.append(_Hand(feature_vector=fv))

    payload = predict_from_hands(
        hands_objs,
        letters_model=_models["letters_model"],
        letters_encoder=_models["letters_encoder"],
        phrases_model=_models["phrases_model"],
        phrases_encoder=_models["phrases_encoder"],
        letters_smoother=_letters_smoother,
        phrases_smoother=_phrases_smoother,
    )

    now = time.time()
    ts = int(now * 1000)
    fps = 0.0
    if _last_req_ts:
        fps = 1.0 / max(1e-6, now - _last_req_ts)
    _last_req_ts = now

    payload["ts"] = ts
    payload["fps"] = float(fps)

    return JSONResponse(payload)


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"ok": True})
