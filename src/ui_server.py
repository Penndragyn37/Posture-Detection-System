"""
ui_server.py
FastAPI + WebSocket server that streams annotated video frames and
detection state to the web UI in real time.

Start with: python run.py
Then open: http://localhost:8080
"""
from __future__ import annotations
import asyncio
import base64
import copy
import json
import threading
from pathlib import Path
from typing import Optional, Set

import cv2
import numpy as np

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse
    from fastapi.staticfiles import StaticFiles
    import uvicorn
    _FASTAPI_OK = True
except ImportError:
    _FASTAPI_OK = False

from .pipeline import Pipeline
from .intent_filter import SystemState


def create_app(pipeline: Pipeline, image_source: str | None = None) -> "FastAPI":
    if not _FASTAPI_OK:
        raise ImportError(
            "FastAPI/uvicorn not installed. Run: pip install fastapi uvicorn"
        )

    app = FastAPI(title="PDSDA — Posture Detection System")

    # Serve static files (index.html)
    static_dir = Path(__file__).parent.parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    _connections: Set[WebSocket] = set()
    _connections_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Broadcast helpers (called from the pipeline thread)
    # ------------------------------------------------------------------

    def _encode_frame(frame: np.ndarray, quality: int = 70) -> str:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # Scale down for streaming speed
        h, w = frame_bgr.shape[:2]
        scale = min(1.0, 800 / w)
        if scale < 1.0:
            frame_bgr = cv2.resize(
                frame_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
            )
        _, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return base64.b64encode(buf).decode("utf-8")

    async def _broadcast(message: dict) -> None:
        dead = set()
        async with _connections_lock:
            for ws in _connections:
                try:
                    await ws.send_text(json.dumps(message))
                except Exception:
                    dead.add(ws)
            for ws in dead:
                _connections.discard(ws)

    def _pipeline_callback(frame: np.ndarray, result, state) -> None:
        """Called from pipeline thread — schedule broadcast on the event loop."""
        try:
            loop = app.state.loop
        except AttributeError:
            return

        msg_frame = {
            "type":  "frame",
            "data":  _encode_frame(frame),
        }
        msg_data = {
            "type":    "detection",
            "result":  result.to_dict(),
            "state":   state.as_dict(),
            "history": pipeline.history[:10],
            "latency":        pipeline.latency_stats,
            "paused":         pipeline.is_paused,
            "image_mode":     pipeline.image_mode,
            "image_progress": pipeline.image_progress,
        }

        asyncio.run_coroutine_threadsafe(_broadcast(msg_frame), loop)
        asyncio.run_coroutine_threadsafe(_broadcast(msg_data), loop)

    pipeline.on_frame = _pipeline_callback

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    @app.on_event("startup")
    async def on_startup():
        app.state.loop = asyncio.get_event_loop()

    @app.get("/", response_class=HTMLResponse)
    async def index():
        html_path = Path(__file__).parent.parent / "static" / "index.html"
        return HTMLResponse(content=html_path.read_text())

    @app.get("/api/status")
    async def status():
        return {
            "approach":    pipeline.approach,
            "model":       pipeline.config.get("gemma4", {}).get("model", "gemma4:26b"),
            "running":     pipeline._running,
            "paused":      pipeline.is_paused,
            "log_path":    pipeline.log_path,
            "state":       pipeline.latest_state.as_dict() if pipeline.latest_state else {},
            "camera_id":   pipeline.config.get("camera", {}).get("device_id", 0),
            "latency":       pipeline.latency_stats,
            "ollama_host":   pipeline.ollama_host,
            "image_mode":    pipeline.image_mode,
            "source_type":   pipeline.source_type,
            "image_progress": pipeline.image_progress,
        }

    @app.get("/api/cameras")
    async def cameras():
        return {
            "cameras":           pipeline.available_cameras,
            "last_camera_name":  pipeline._last_camera_name,
        }

    @app.get("/api/models")
    async def models():
        """Return available models: local Ollama models + Claude API option."""
        import requests as _req
        available = []
        # Always offer Claude API
        available.append({"id": "claude_api", "label": "Claude API (cloud, no GPU)", "approach": "claude_api"})
        # Query Ollama for installed models
        ollama_host = pipeline.config.get("gemma4", {}).get("ollama_host", "http://localhost:11434")
        try:
            r = _req.get(f"{ollama_host}/api/tags", timeout=3)
            if r.ok:
                for m in r.json().get("models", []):
                    name = m.get("name", "")
                    approach = "gemma4_mobile" if "e4b" in name.lower() else "gemma4_server"
                    size_mb = m.get("size", 0) // (1024*1024)
                    label = f"{name}  ({size_mb:,} MB)" if size_mb else name
                    available.append({"id": name, "label": label, "approach": approach})
        except Exception:
            pass
        return {"models": available, "active": pipeline.config.get("gemma4", {}).get("model", "gemma4:26b")}

    @app.get("/api/ping_host")
    async def ping_host(host: str = "http://localhost:11434"):
        """Test connectivity and list models at any Ollama endpoint."""
        import requests as _req
        try:
            h = host.strip().rstrip("/")
            if not h.startswith("http"):
                h = "http://" + h
            r = _req.get(f"{h}/api/tags", timeout=4)
            if r.ok:
                models = [m.get("name","") for m in r.json().get("models", [])]
                return {"ok": True, "host": h, "models": models, "count": len(models)}
            return {"ok": False, "host": h, "error": f"HTTP {r.status_code}"}
        except Exception as e:
            return {"ok": False, "host": host, "error": str(e)}

    @app.websocket("/ws")
    async def ws_endpoint(websocket: WebSocket):
        await websocket.accept()
        async with _connections_lock:
            _connections.add(websocket)
        try:
            while True:
                raw = await websocket.receive_text()
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                action = msg.get("action")
                if action == "cancel":
                    pipeline.cancel()
                elif action == "confirm_now":
                    pipeline.confirm_now()
                elif action == "set_approach":
                    new_approach = msg.get("approach", pipeline.approach)
                    threading.Thread(target=pipeline.set_approach, args=(new_approach,), daemon=True).start()
                elif action == "set_model":
                    model_id = msg.get("model_id", "")
                    if model_id:
                        threading.Thread(target=pipeline.set_model, args=(model_id,), daemon=True).start()
                elif action == "set_ollama_host":
                    h = msg.get("host", "")
                    if h:
                        threading.Thread(target=pipeline.set_ollama_host, args=(h,), daemon=True).start()
                elif action == "next_image":
                    pipeline.advance_image()
                elif action == "prev_image":
                    pipeline.previous_image()
                elif action == "set_source":
                    src_type = msg.get("source_type", "camera")
                    src_val  = msg.get("value", 0)
                    threading.Thread(
                        target=pipeline.set_source,
                        args=(src_type, src_val),
                        daemon=True,
                    ).start()
                elif action == "pause":
                    pipeline.pause()
                elif action == "resume":
                    pipeline.resume()
                elif action == "set_camera":
                    device_id = int(msg.get("device_id", 0))
                    threading.Thread(
                        target=pipeline.set_camera,
                        args=(device_id,),
                        daemon=True,
                    ).start()
                elif action == "clear_error":
                    pipeline._intent.clear_error()
                elif action == "reconnect_camera":
                    threading.Thread(
                        target=pipeline.reconnect_camera,
                        daemon=True,
                    ).start()

        except WebSocketDisconnect:
            pass
        finally:
            async with _connections_lock:
                _connections.discard(websocket)

    return app


def run_server(pipeline: Pipeline, host: str = "0.0.0.0", port: int = 8080) -> None:
    app = create_app(pipeline)
    uvicorn.run(app, host=host, port=port, log_level="warning")
