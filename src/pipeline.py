"""
pipeline.py
Main orchestrator: camera capture -> detector -> intent filter -> logger -> UI.
Satisfies SRS FR-001 to FR-036 as the integration layer.
"""
from __future__ import annotations
import copy
import os
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import yaml

from .output_schema import PostureResult
from .detector_base import PostureDetector
from .intent_filter import IntentFilter, IntentState, SystemState
from .camera_utils import list_cameras
from .image_player import ImagePlayer
from .evaluation_logger import EvaluationLogger




class FrameGrabber:
    """
    Runs a dedicated background thread that continuously drains the camera
    buffer at full capture fps (typically 30). The inference loop calls
    get_latest() to get the most recent frame rather than a stale buffered one.

    Without this, a 2-second Gemma 4 inference causes 60 frames to queue up
    in the OpenCV buffer. The next cap.read() returns a frame from 60 frames
    ago — often a dark or transitional frame the model reads as absent.
    """

    def __init__(self, capture):
        self._cap   = capture
        self._frame = None
        self._lock  = threading.Lock()
        self._ok    = False
        self._running = True
        self._thread  = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        _consecutive_fails = 0
        _FAIL_THRESHOLD    = 5   # CAM-03: mark disconnected after 5 consecutive
                                  # failed reads (covers both ret=False and cases
                                  # where AVFoundation returns a frozen/black frame
                                  # — the pipeline's _loop also checks brightness)
        while self._running:
            ret, frame = self._cap.read()
            if ret and frame is not None:
                with self._lock:
                    self._frame = frame
                    self._ok    = True
                _consecutive_fails = 0
            else:
                _consecutive_fails += 1
                if _consecutive_fails >= _FAIL_THRESHOLD:
                    with self._lock:
                        self._ok = False   # signal pipeline to call set_error()

    def get_latest(self):
        """Return (ok, frame_bgr) — always the most recently captured frame."""
        with self._lock:
            return self._ok, (self._frame.copy() if self._frame is not None else None)

    def stop(self):
        self._running = False
        self._thread.join(timeout=2.0)   # CAM-01 fix: wait for current cap.read()
                                          # to complete before the caller releases
                                          # the VideoCapture object. Without this,
                                          # release() races cap.read() across threads,
                                          # which is undefined behaviour in OpenCV and
                                          # causes a crash on camera switch.

class Pipeline:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self._approach    = self.config.get("approach", "gemma4_server")
        self._detector: Optional[PostureDetector] = None
        self._capture     = None
        self._running     = False
        self._thread: Optional[threading.Thread] = None
        self._grabber: Optional[FrameGrabber] = None
        self._image_player: Optional[ImagePlayer] = None

        # State
        self.latest_result:   Optional[PostureResult] = None  # raw (every frame)
        self._display_result: Optional[PostureResult] = None  # held: last confident result
        self._paused:         bool                      = False
        self._latency_history: list                     = []   # recent inference latencies
        self.latest_state:    Optional[IntentState]   = None
        self.latest_frame:    Optional[np.ndarray]    = None
        self._frame_lock    = threading.Lock()

        # Recent history for UI log panel
        self._history: list[dict] = []
        self._history_max = 20

        # UI callbacks
        self.on_frame:  Optional[Callable] = None   # (frame_rgb, result, state)
        self.on_result: Optional[Callable] = None   # (result, state)

        # Last confirmed active camera name — used to reconnect by name on Linux
        # where device indices can change after a USB replug
        self._last_camera_name: Optional[str] = None

        # Sub-systems
        self._logger = EvaluationLogger(self.config.get("evaluation", {}))
        self._intent = IntentFilter(
            self.config.get("intent", {}),
            on_adjust=self._on_adjust,
            on_state_change=self._on_state_change,
        )

        self._load_detector()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def preflight(self) -> dict:
        """Run startup checks. Returns dict of {check: bool}."""
        results = {}

        # Camera (FR-004)
        cam_cfg  = self.config.get("camera", {})
        cap_test = cv2.VideoCapture(cam_cfg.get("device_id", 0))
        results["camera"] = cap_test.isOpened()
        cap_test.release()

        # Detector (FR-004)
        try:
            results["detector"] = self._detector.health_check()
        except Exception as exc:
            results["detector"] = False
            results["detector_error"] = str(exc)

        # Disk space (FR-029)
        import shutil
        free_gb = shutil.disk_usage(self._logger.log_path).free / (1024 ** 3)
        results["disk_ok"] = free_gb > 0.5   # warn if <500 MB

        # PID lock (FR-032)
        lock_file = Path("/tmp/pdsda.pid")
        if lock_file.exists():
            try:
                pid = int(lock_file.read_text())
                import os, signal
                os.kill(pid, 0)   # raises if process dead
                results["pid_lock"] = False
                results["pid_lock_error"] = f"Process {pid} already running"
            except (ProcessLookupError, ValueError):
                lock_file.unlink(missing_ok=True)
                results["pid_lock"] = True
        else:
            results["pid_lock"] = True

        return results

    def start(self, image_source: str | None = None) -> None:
        # PID lock
        lock_file = Path("/tmp/pdsda.pid")
        try:
            import os
            lock_file.write_text(str(os.getpid()))
        except Exception:
            pass

        fps = self.config.get("sampling", {}).get("fps", 1)

        if image_source:
            # Image player mode — no camera needed
            self._image_player = ImagePlayer(image_source, fps=fps, loop=False)
            self._capture      = None
            self._grabber      = None
            print(f"[PDSDA] Image player mode: {self._image_player.total} images from {image_source}")
        else:
            cam_cfg = self.config.get("camera", {})
            self._capture = cv2.VideoCapture(cam_cfg.get("device_id", 0))
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH,  cam_cfg.get("width",  1280))
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg.get("height", 720))

            if not self._capture.isOpened():
                raise RuntimeError("Camera could not be opened (SRS FR-004)")

            self._grabber      = FrameGrabber(self._capture)
            self._image_player = None

        self._running = True
        self._intent.start()

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._grabber:
            self._grabber.stop()
        if self._image_player:
            self._image_player.stop()
        if self._capture:
            self._capture.release()
        if self._detector:
            self._detector.teardown()
        Path("/tmp/pdsda.pid").unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # User actions forwarded to intent filter
    # ------------------------------------------------------------------

    def cancel(self) -> None:
        self._intent.cancel()

    def confirm_now(self) -> None:
        self._intent.confirm_now()

    def set_camera(self, device_id: int) -> None:
        """
        Hot-swap the active camera without restarting the pipeline.
        Delegates to set_source(), which performs the swap safely (builds
        the new VideoCapture/FrameGrabber before tearing down the old one,
        avoiding the race condition where _loop could observe both the old
        and new source as None simultaneously).
        """
        self.set_source("camera", device_id)

    def set_source(self, source_type: str, value: str | int = 0) -> None:
        """
        Switch between live camera and image player at runtime.
        source_type: "camera" | "images"
        value:       camera device_id (int) for camera mode,
                     folder/file path (str) for image mode
        """
        fps = self.config.get("sampling", {}).get("fps", 1)

        # Build the new source FIRST, before touching any existing
        # reference. This avoids a window where both self._grabber and
        # self._image_player are None at once, which the _loop thread
        # could observe mid-iteration and crash on.
        new_capture = None
        new_grabber = None
        new_player  = None
        error_msg   = None

        if source_type == "images":
            path = str(value)
            try:
                new_player = ImagePlayer(path, fps=fps, loop=False)
                print(f"[PDSDA] Loaded image player: {new_player.total} images from {path}")
            except FileNotFoundError as e:
                error_msg = f"Image path not found: {path}"
                print(f"[PDSDA] {e}")
        else:
            device_id = int(value) if value is not None else 0
            cam_cfg = self.config.get("camera", {})
            import platform as _platform
            if _platform.system() == "Windows":
                new_capture = cv2.VideoCapture(device_id, cv2.CAP_DSHOW)
            else:
                new_capture = cv2.VideoCapture(device_id)
            new_capture.set(cv2.CAP_PROP_FRAME_WIDTH,  cam_cfg.get("width",  1280))
            new_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg.get("height",  720))
            if not new_capture.isOpened():
                # CAM-03: on macOS, a just-replugged camera may not be ready
                # immediately. Retry once after a brief delay before giving up.
                import time as _time
                new_capture.release()
                _time.sleep(1.5)
                if _platform.system() == "Windows":
                    new_capture = cv2.VideoCapture(device_id, cv2.CAP_DSHOW)
                else:
                    new_capture = cv2.VideoCapture(device_id)
                new_capture.set(cv2.CAP_PROP_FRAME_WIDTH,  cam_cfg.get("width",  1280))
                new_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg.get("height",  720))

            if not new_capture.isOpened():
                from .camera_utils import list_cameras
                available = [str(c["id"]) for c in list_cameras()]
                avail_str = ", ".join(available) if available else "none found"
                error_msg = (
                    f"Camera {device_id} could not be opened (tried twice). "
                    f"Available cameras: [{avail_str}]. "
                    f"Wait a moment after replugging, then try again."
                )
                new_capture.release()
                new_capture = None
            else:
                new_grabber = FrameGrabber(new_capture)

        if error_msg:
            self._intent.set_error(error_msg)
            return   # keep the existing source running; nothing to swap in

        # Atomic-ish swap: stop the old source only after the new one is
        # fully ready, and reassign all three references back to back so
        # _loop never observes a state where every source is None.
        old_grabber, old_player, old_capture = self._grabber, self._image_player, self._capture

        self._grabber      = new_grabber
        self._image_player = new_player
        self._capture       = new_capture
        self._paused        = False   # switching sources is an active user
                                       # action; always resume immediately,
                                       # even if the previous source had
                                       # auto-paused on completion

        if old_grabber:
            old_grabber.stop()
        if old_player:
            old_player.stop()
        if old_capture:
            old_capture.release()

        if source_type == "images":
            self._intent.clear_error()
            print(f"[PDSDA] Switched to image player ({new_player.total} images)")
        else:
            device_id = int(value) if value is not None else 0
            self.config.setdefault("camera", {})["device_id"] = device_id
            # Remember the camera name so reconnect_camera() can find it by name
            # if the device index changes after a USB replug (common on Linux)
            from .camera_utils import list_cameras
            for cam in list_cameras():
                if cam["id"] == device_id:
                    self._last_camera_name = cam["label"]
                    break
            self._intent.clear_error()
            print(f"[PDSDA] Switched to camera {value}"
                  + (f" ({self._last_camera_name})" if self._last_camera_name else ""))

    @property
    def source_type(self) -> str:
        return "images" if self._image_player is not None else "camera"

    @property
    def available_cameras(self) -> list[dict]:
        return list_cameras()

    @property
    def image_mode(self) -> bool:
        return self._image_player is not None

    @property
    def image_progress(self) -> dict | None:
        return self._image_player.progress if self._image_player else None

    def advance_image(self) -> None:
        """Manually step to the next test image."""
        if self._image_player:
            self._image_player.advance()

    def previous_image(self) -> None:
        """Step back to the previous test image."""
        if self._image_player:
            self._image_player.previous()

    @property
    def latency_stats(self) -> dict:
        h = self._latency_history
        if not h:
            return {"current": None, "mean": None, "min": None, "max": None, "history": []}
        return {
            "current": round(h[-1]),
            "mean":    round(sum(h) / len(h)),
            "min":     round(min(h)),
            "max":     round(max(h)),
            "history": [round(v) for v in h[-20:]],
        }

    @property
    def is_paused(self) -> bool:
        return self._paused

    def pause(self) -> None:
        """Pause inference without stopping the camera or WebSocket."""
        self._paused = True
        if self._intent:
            self._intent.set_error("Paused by user")

    def resume(self) -> None:
        """Resume inference after pause."""
        self._paused = False
        if self._intent:
            self._intent.clear_error()

    def reconnect_camera(self) -> None:
        """
        Re-open the last active camera.

        On Linux, USB cameras often re-enumerate to a different /dev/videoN
        index after a physical replug. This method searches the current camera
        list by name first, then falls back to the stored device_id. This
        ensures the correct camera is reconnected even if its index changed.
        """
        from .camera_utils import list_cameras
        cameras = list_cameras()
        target_id = self.config.get("camera", {}).get("device_id", 0)

        if self._last_camera_name:
            # Search by name — handles Linux index re-enumeration after replug
            for cam in cameras:
                if cam["label"] == self._last_camera_name:
                    if cam["id"] != target_id:
                        print(f"[PDSDA] Camera re-enumerated: "
                              f"{self._last_camera_name} moved from "
                              f"[{target_id}] to [{cam['id']}] — updating")
                        target_id = cam["id"]
                    break
            else:
                print(f"[PDSDA] Camera '{self._last_camera_name}' not found "
                      f"in current device list — trying stored id [{target_id}]")

        print(f"[PDSDA] Reconnecting to camera [{target_id}]...")
        self.set_camera(target_id)

    def set_approach(self, approach: str) -> None:
        """Hot-swap inference approach without restarting the camera."""
        self.config["approach"] = approach
        self._approach = approach
        if self._detector:
            self._detector.teardown()
        self._load_detector()

    def set_model(self, model_id: str) -> None:
        """
        Switch the active model by name.
        - "claude_api" → Claude API path
        - "gemma4:e4b" → Gemma 4 mobile path
        - anything else → Gemma 4 server path
        """
        if model_id == "claude_api":
            self.config["approach"] = "claude_api"
        else:
            approach = "gemma4_mobile" if "e4b" in model_id.lower() else "gemma4_server"
            self.config["approach"] = approach
            self.config.setdefault("gemma4", {})["model"] = model_id
        self._approach = self.config["approach"]
        if self._detector:
            self._detector.teardown()
        self._load_detector()

    def set_ollama_host(self, host: str) -> None:
        """
        Point the Gemma 4 detector at a different Ollama endpoint.
        Useful for:
          - localhost:11434  — Mac default
          - 192.168.x.x:11434  — phone or tablet on same Wi-Fi
        Changing the host restarts the detector; the camera is not affected.
        """
        host = host.strip().rstrip("/")
        if not host.startswith("http"):
            host = "http://" + host
        self.config.setdefault("gemma4", {})["ollama_host"] = host
        if self._detector:
            self._detector.teardown()
        self._load_detector()

    @property
    def ollama_host(self) -> str:
        return self.config.get("gemma4", {}).get("ollama_host", "http://localhost:11434")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def approach(self) -> str:
        return self._approach

    @property
    def history(self) -> list[dict]:
        return list(self._history)

    @property
    def log_path(self) -> str:
        return self._logger.log_path

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_detector(self) -> None:
        approach = self._approach
        if approach in ("gemma4_server", "gemma4_mobile"):
            from .detectors.gemma4_detector import Gemma4Detector
            self._detector = Gemma4Detector(self.config.get("gemma4", {}))
        elif approach == "claude_api":
            from .detectors.claude_detector import ClaudeDetector
            self._detector = ClaudeDetector(self.config.get("claude_api", {}))
        else:
            raise ValueError(f"Unknown approach: {approach}")

    def _loop(self) -> None:
        fps      = self.config.get("sampling", {}).get("fps", 1)
        interval = 1.0 / max(fps, 0.1)
        conf_min = self.config.get("intent", {}).get("confidence_min", 0.70)

        while self._running:
            t0 = time.monotonic()

            # Pause mode
            if self._paused:
                time.sleep(0.2)
                continue

            # Get next frame from whichever source is currently active.
            # Capture the reference ONCE per iteration rather than reading
            # self._image_player repeatedly — set_source() can swap sources
            # from another thread at any time, and re-reading the attribute
            # multiple times within one iteration risked observing a stale
            # or inconsistent value between checks.
            image_player = self._image_player
            grabber      = self._grabber

            if image_player is not None:
                if image_player.is_finished:
                    print("[PDSDA] All images processed. Pausing.")
                    self._paused = True
                    continue
                ret, frame_bgr = image_player.get_latest()
            elif grabber is not None:
                ret, frame_bgr = grabber.get_latest()
            else:
                time.sleep(0.2)   # no source active yet (mid-switch)
                continue

            if not ret or frame_bgr is None:
                self._intent.set_error("Camera disconnected — frame read failed (SRS FR-005)")
                time.sleep(0.5)
                continue

            # CAM-03: detect nearly-black frames as a disconnect signal.
            # On macOS, a disconnected USB camera often returns very dark
            # frames (mean brightness < 5/255) rather than ret=False.
            # After 3 consecutive dark frames, surface as an error so the UI
            # shows the camera as disconnected rather than silently showing absent.
            if not hasattr(self, '_dark_frame_count'):
                self._dark_frame_count = 0
            import numpy as _np
            if frame_bgr is not None and _np.mean(frame_bgr) < 5.0:
                self._dark_frame_count += 1
                if self._dark_frame_count >= 3:
                    self._intent.set_error("Camera disconnected — dark frames detected (SRS FR-005)")
            else:
                self._dark_frame_count = 0

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            with self._frame_lock:
                self.latest_frame = frame_rgb

            # Inference
            result = self._detector.detect(frame_rgb)
            self.latest_result = result

            if image_player is not None:
                # Image-player mode: each image is an independent test sample
                # with no temporal relationship to the one before it. The
                # 5-frame threshold, countdown, and cooldown only make sense
                # for a continuous video stream, so they are bypassed
                # entirely here. The classification is taken at face value
                # for every single image.
                state = IntentState(
                    system_state=SystemState.EVALUATING,
                    current_posture=result.posture,
                )
            else:
                # Live camera mode: full 7-state intent detection machine.
                state = self._intent.process(result)
            self.latest_state = state

            # Log every frame; include ground truth and source filename when
            # running from images, so individual misclassifications can be
            # traced back to the exact file for later inspection.
            ground_truth = image_player.current_ground_truth if image_player is not None else None
            image_name   = image_player.current_name          if image_player is not None else None
            self._logger.log(result, state.system_state.value,
                             frame=frame_rgb, ground_truth=ground_truth, image_name=image_name)

            # Track latency (confident frames only)
            if not result.is_error() and result.confidence >= conf_min:
                self._latency_history.append(result.latency_ms)
                if len(self._latency_history) > 30:
                    self._latency_history.pop(0)

            conf_min = self.config.get("intent", {}).get("confidence_min", 0.70)

            if image_player is not None:
                # No holding across unrelated images. Always display this
                # image's own result, and log every image to history
                # regardless of confidence, since each one is a discrete
                # benchmark sample the user wants visible, not noise to
                # suppress in a continuous feed.
                display  = result
                is_held  = False
                self._push_history(result, state)
            else:
                # Live camera mode: hold last confident result on screen so
                # a single low-confidence or error frame doesn't flicker the
                # display to "absent".
                if result.confidence >= conf_min and not result.is_error():
                    self._display_result = result
                display = self._display_result if self._display_result is not None else result
                is_held = (self._display_result is not None and result is not self._display_result
                           and (result.is_error() or result.confidence < conf_min))
                if result.confidence >= conf_min and not result.is_error():
                    self._push_history(result, state)

            annotated = self._annotate(frame_rgb.copy(), display, held=is_held)

            # Notify UI with the display (held) result, not the raw one
            if self.on_frame:
                self.on_frame(annotated, display, state)
            if self.on_result:
                self.on_result(display, state)

            # Image player mode: advance only now that this image has been
            # through one full inference + logging cycle. This guarantees
            # every image in the folder gets exactly one inference call,
            # regardless of how long inference took (no skipped frames).
            # Re-check self._image_player (not the local) here: if the user
            # switched sources mid-cycle, we want to advance/stop the NEW
            # state correctly rather than acting on a stale local reference.
            if self._image_player is image_player and image_player is not None:
                image_player.mark_consumed()
                if not image_player.advance():
                    print(f"[PDSDA] All {image_player.total} images processed. Pausing.")
                    self._paused = True
                continue   # skip the fps-interval sleep below; advance() already paced this

            # Sleep remainder of polling interval (camera mode only)
            elapsed = time.monotonic() - t0
            time.sleep(max(0.0, interval - elapsed))

    def _annotate(self, frame: np.ndarray, result: PostureResult, held: bool = False) -> np.ndarray:
        """Burn detection result onto the frame as a lightweight overlay."""
        h, w = frame.shape[:2]

        # Posture colour
        color = {
            "sitting":  (29, 158, 117),   # teal
            "standing": (99, 153, 34),    # green
            "absent":   (136, 135, 128),  # gray
        }.get(result.posture, (200, 50, 50))

        # Semi-transparent banner at bottom
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 70), (w, h), (15, 25, 35), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Posture label
        label = result.posture.upper()
        cv2.putText(frame, label, (20, h - 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2, cv2.LINE_AA)

        # Confidence bar
        bar_x, bar_y, bar_h = 20, h - 22, 8
        bar_w = int((w - 40) * result.confidence)
        cv2.rectangle(frame, (bar_x, bar_y), (w - 20, bar_y + bar_h), (50, 50, 50), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), color, -1)

        # Latency badge top-right
        lat_text = f"{result.latency_ms:.0f}ms"
        cv2.putText(frame, lat_text, (w - 120, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1, cv2.LINE_AA)

        # "HELD" indicator: shown when displaying a previous result due to low confidence
        if held:
            cv2.putText(frame, "HELD", (12, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (186, 117, 23), 2, cv2.LINE_AA)

        # Confidence value
        conf_text = f"{result.confidence:.0%}"
        cv2.putText(frame, conf_text, (w - 100, h - 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1, cv2.LINE_AA)

        return frame

    def _push_history(self, result: PostureResult, state: IntentState) -> None:
        entry = {
            "posture":    result.posture,
            "confidence": round(result.confidence, 2),
            "latency_ms": result.latency_ms,
            "approach":   result.approach,
            "state":      state.system_state.value,
            "timestamp":  result.timestamp,
            "reasoning":  result.reasoning,
        }
        self._history.insert(0, entry)
        if len(self._history) > self._history_max:
            self._history.pop()

    def _on_adjust(self, posture: str) -> None:
        print(f"[PDSDA] DESK CONTROL SIGNAL emitted: target posture = {posture}")

    def _on_state_change(self, state: IntentState) -> None:
        self.latest_state = state


