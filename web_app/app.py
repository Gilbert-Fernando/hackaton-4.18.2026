from __future__ import annotations

import asyncio
import os
import threading
import time
from typing import Optional

import cv2
import joblib
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

import sys

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.hand_tracking import HandTracker
from src.predict_realtime import PredictionSmoother, predict_from_hands
from utils.visualization import draw_landmark_points


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
            self.cap.release()
            self.cap = None

            for idx in range(0, 6):
                if idx == camera_index:
                    continue
                cap = cv2.VideoCapture(idx)
                if cap.isOpened():
                    self.cap = cap
                    break
                cap.release()

            if self.cap is None:
                raise RuntimeError(f"Could not open webcam (tried indices 0-5; preferred {camera_index})")

        self.tracker = HandTracker(max_num_hands=2)
        self.letters_smoother = PredictionSmoother(window_size=10, dominance=0.7)
        self.phrases_smoother = PredictionSmoother(window_size=10, dominance=0.7)

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
        self._thread: Optional[threading.Thread] = None

    async def start(self, manager: ConnectionManager) -> None:
        if self._running:
            return
        self._running = True

        loop = asyncio.get_running_loop()
        self._thread = threading.Thread(target=self._run_loop, args=(manager, loop), daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False

        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def _run_loop(self, manager: ConnectionManager, loop: asyncio.AbstractEventLoop) -> None:
        last_t = time.time()
        fps = 0.0
        ema_alpha = 0.15

        while self._running:
            ok, frame = self.cap.read()
            if not ok:
                continue

            frame = cv2.flip(frame, 1)

            hands = self.tracker.detect_hands(frame)
            overlay = frame.copy()

            if len(hands) == 0:
                cv2.putText(
                    overlay,
                    "No hand detected",
                    (10, 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    2,
                )
            else:
                for hnd in hands:
                    draw_landmark_points(overlay, hnd.landmarks_xy)
            payload = predict_from_hands(
                hands,
                letters_model=self.letters_model,
                letters_encoder=self.letters_encoder,
                phrases_model=self.phrases_model,
                phrases_encoder=self.phrases_encoder,
                letters_smoother=self.letters_smoother,
                phrases_smoother=self.phrases_smoother,
            )

            now = time.time()
            inst_fps = 1.0 / max(1e-6, now - last_t)
            last_t = now
            fps = (1 - ema_alpha) * fps + ema_alpha * inst_fps

            cv2.putText(
                overlay,
                f"{payload['label']}",
                (20, 70),
                cv2.FONT_HERSHEY_DUPLEX,
                2.2,
                (0, 0, 0),
                4,
            )
            cv2.putText(
                overlay,
                f"conf: {float(payload['confidence']):.2f}",
                (20, 110),
                cv2.FONT_HERSHEY_DUPLEX,
                0.8,
                (0, 0, 0),
                2,
            )

            ok2, buf = cv2.imencode(".jpg", overlay, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ok2:
                continue
            jpeg = buf.tobytes()

            payload["ts"] = int(now * 1000)
            payload["fps"] = float(fps)

            fut = asyncio.run_coroutine_threadsafe(
                self._update_and_broadcast(manager, jpeg, payload),
                loop,
            )
            try:
                fut.result(timeout=1.0)
            except Exception:
                pass

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

    async def get_latest_payload(self) -> dict:
        async with self._lock:
            return dict(self._latest_payload)


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


@app.get("/debug")
async def debug() -> JSONResponse:
    index_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    try:
        with open(index_path, "r", encoding="utf-8") as f:
            html = f.read()
    except Exception as e:
        return JSONResponse({"app_file": __file__, "index_path": index_path, "error": str(e)})

    return JSONResponse(
        {
            "app_file": __file__,
            "index_path": index_path,
            "has_pullState": "pullState" in html,
            "html_prefix": html[:120],
        }
    )


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


@app.get("/state")
async def state() -> JSONResponse:
    if engine is None:
        return JSONResponse({"label": "", "confidence": 0.0, "mode": "", "hands": 0, "ts": 0, "fps": 0.0})
    payload = await engine.get_latest_payload()
    return JSONResponse(payload)


@app.websocket("/ws")
async def ws(websocket: WebSocket) -> None:
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
