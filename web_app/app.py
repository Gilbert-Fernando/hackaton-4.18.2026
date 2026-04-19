from __future__ import annotations

import asyncio
import os
import time
from collections import Counter, deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple

import cv2
import joblib
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

import sys

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.hand_tracking import HandTracker


@dataclass
class SmoothedPrediction:
    label: str
    confidence: float


class PredictionSmoother:
    def __init__(self, window_size: int = 12, min_count: int = 6) -> None:
        self._items: Deque[Tuple[str, float]] = deque(maxlen=window_size)
        self._min_count = min_count

    def push(self, label: str, confidence: float) -> None:
        self._items.append((label, float(confidence)))

    def get(self) -> Optional[SmoothedPrediction]:
        if len(self._items) < self._min_count:
            return None

        labels = [lbl for (lbl, _conf) in self._items]
        winner, winner_count = Counter(labels).most_common(1)[0]
        if winner_count < self._min_count:
            return None

        winner_confs = [conf for (lbl, conf) in self._items if lbl == winner]
        return SmoothedPrediction(label=winner, confidence=float(np.mean(winner_confs)))


def predict_label(model, encoder, features: np.ndarray) -> Tuple[str, float]:
    x = features.reshape(1, -1)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)[0]
        idx = int(np.argmax(proba))
        label = str(encoder.inverse_transform([idx])[0])
        return label, float(proba[idx])

    idx = int(model.predict(x)[0])
    label = str(encoder.inverse_transform([idx])[0])
    return label, 1.0


class ConnectionManager:
    def __init__(self) -> None:
        self._clients: set[WebSocket] = set()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self._clients.add(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        self._clients.discard(websocket)

    async def broadcast_json(self, payload: dict) -> None:
        if not self._clients:
            return

        dead: list[WebSocket] = []
        for ws in list(self._clients):
            try:
                await ws.send_json(payload)
            except Exception:
                dead.append(ws)

        for ws in dead:
            self.disconnect(ws)


class RealtimeEngine:
    def __init__(self, camera_index: int = 0) -> None:
        models_dir = os.path.join(PROJECT_ROOT, "models")

        self.letters_model = joblib.load(os.path.join(models_dir, "letters_model.pkl"))
        self.letters_encoder = joblib.load(os.path.join(models_dir, "letters_encoder.pkl"))
        self.phrases_model = joblib.load(os.path.join(models_dir, "phrases_model.pkl"))
        self.phrases_encoder = joblib.load(os.path.join(models_dir, "phrases_encoder.pkl"))

        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open webcam (index {camera_index})")

        self.tracker = HandTracker(max_num_hands=2)
        self.letters_smoother = PredictionSmoother(window_size=12, min_count=6)
        self.phrases_smoother = PredictionSmoother(window_size=12, min_count=6)

        self._lock = asyncio.Lock()
        self._latest_jpeg: Optional[bytes] = None
        self._latest_payload: dict = {
            "label": "",
            "confidence": 0.0,
            "mode": "",
            "hands": 0,
            "ts": 0,
            "fps": 0.0,
        }

        self._running = False

    async def start(self, manager: ConnectionManager) -> None:
        if self._running:
            return
        self._running = True

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._run_loop, manager)

    def stop(self) -> None:
        self._running = False

    def _run_loop(self, manager: ConnectionManager) -> None:
        last_t = time.time()
        fps = 0.0
        ema_alpha = 0.15

        while self._running:
            ok, frame = self.cap.read()
            if not ok:
                continue

            frame = cv2.flip(frame, 1)

            hands = self.tracker.detect_hands(frame)
            mode_text = ""
            label = ""
            conf = 0.0

            if len(hands) == 1:
                mode_text = "LETTERS"
                label, conf = predict_label(
                    self.letters_model, self.letters_encoder, hands[0].feature_vector
                )
                self.letters_smoother.push(label, conf)
                smoothed = self.letters_smoother.get()
            elif len(hands) == 2:
                mode_text = "PHRASES"
                features_2hand = np.concatenate(
                    [hands[0].feature_vector, hands[1].feature_vector]
                )
                label, conf = predict_label(
                    self.phrases_model, self.phrases_encoder, features_2hand
                )
                self.phrases_smoother.push(label, conf)
                smoothed = self.phrases_smoother.get()
            else:
                smoothed = None

            if smoothed is not None:
                label = smoothed.label
                conf = smoothed.confidence

            now = time.time()
            inst_fps = 1.0 / max(1e-6, now - last_t)
            last_t = now
            fps = (1 - ema_alpha) * fps + ema_alpha * inst_fps

            cv2.putText(
                frame,
                f"{label}",
                (20, 70),
                cv2.FONT_HERSHEY_DUPLEX,
                2.2,
                (0, 0, 0),
                4,
            )
            cv2.putText(
                frame,
                f"conf: {conf:.2f}",
                (20, 110),
                cv2.FONT_HERSHEY_DUPLEX,
                0.8,
                (0, 0, 0),
                2,
            )

            ok2, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ok2:
                continue
            jpeg = buf.tobytes()

            payload = {
                "label": label,
                "confidence": float(conf),
                "mode": mode_text,
                "hands": int(len(hands)),
                "ts": int(now * 1000),
                "fps": float(fps),
            }

            asyncio.run(self._update_and_broadcast(manager, jpeg, payload))

        self.tracker.close()
        self.cap.release()

    async def _update_and_broadcast(
        self, manager: ConnectionManager, jpeg: bytes, payload: dict
    ) -> None:
        async with self._lock:
            self._latest_jpeg = jpeg
            self._latest_payload = payload

        await manager.broadcast_json(payload)

    async def get_latest_jpeg(self) -> Optional[bytes]:
        async with self._lock:
            return self._latest_jpeg


app = FastAPI()
manager = ConnectionManager()
engine: Optional[RealtimeEngine] = None

app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")


@app.on_event("startup")
async def startup() -> None:
    global engine
    engine = RealtimeEngine(camera_index=int(os.environ.get("ASL_CAMERA", "0")))
    asyncio.create_task(engine.start(manager))


@app.on_event("shutdown")
async def shutdown() -> None:
    if engine is not None:
        engine.stop()


@app.get("/")
async def index() -> HTMLResponse:
    with open(os.path.join(os.path.dirname(__file__), "static", "index.html"), "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.get("/video")
async def video() -> StreamingResponse:
    async def gen():
        while True:
            if engine is None:
                await asyncio.sleep(0.05)
                continue
            jpeg = await engine.get_latest_jpeg()
            if jpeg is None:
                await asyncio.sleep(0.01)
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            )
            await asyncio.sleep(0.03)

    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.websocket("/ws")
async def ws(websocket: WebSocket) -> None:
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
