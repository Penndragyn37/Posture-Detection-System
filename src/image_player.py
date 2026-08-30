"""
image_player.py
Replaces FrameGrabber when running with static images instead of a live camera.

Usage:
    python run.py --images ./test_images/

KEY DESIGN: advancement is driven by inference completion, not a wall-clock
timer. The pipeline calls advance() once it has finished processing the
current image (inference + logging). This guarantees every single image
in the folder receives exactly one inference call, regardless of how slow
or fast that inference is. A fixed-fps timer would skip images whenever
inference takes longer than the fps interval, which is the normal case
for Gemma 4 (1-3s per frame) at the default 1 fps setting.

Ground truth is parsed from the filename if the name starts with a known
posture label, e.g.:
    sitting_desk_daylight_01.jpg   -> ground truth = sitting
    standing_001.jpg               -> ground truth = standing
    absent_no_person.jpg           -> ground truth = absent
    001.jpg                        -> ground truth = unknown
"""
from __future__ import annotations
import glob
import threading
from pathlib import Path

import cv2
import numpy as np


_VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_LABELS    = {"sitting", "standing", "absent"}


def _parse_ground_truth(filename: str) -> str | None:
    """Extract posture label from filename if present."""
    stem = Path(filename).stem.lower()
    for label in _LABELS:
        if stem.startswith(label):
            return label
    return None


class ImagePlayer:
    """
    Reads images from a directory and serves them as frames to the pipeline.
    Advances ONLY when advance() is called explicitly by the pipeline after
    completing inference on the current image, or by the user via the
    next_image / prev_image UI controls. There is no internal timer.
    """

    def __init__(self, path: str, fps: float = 1.0, loop: bool = False):
        # fps is accepted for API compatibility but no longer drives a timer.
        p = Path(path)
        if p.is_file():
            self._paths = [str(p)]
        elif p.is_dir():
            self._paths = sorted(
                f for f in glob.glob(str(p / "**" / "*"), recursive=True)
                if Path(f).suffix.lower() in _VALID_EXT
            )
        else:
            raise FileNotFoundError(f"Image path not found: {path}")

        if not self._paths:
            raise FileNotFoundError(f"No images found in: {path}")

        self._loop      = loop
        self._idx        = 0
        self._frame: np.ndarray | None = None
        self._ok         = False
        self._lock       = threading.Lock()
        self._finished   = False
        self._consumed   = False   # has the current image been through one inference cycle?

        self._load(0)

    # ------------------------------------------------------------------
    def _load(self, idx: int) -> bool:
        path = self._paths[idx]
        img  = cv2.imread(path)
        if img is None:
            return False
        with self._lock:
            self._frame    = img          # BGR, same as VideoCapture
            self._ok       = True
            self._idx      = idx
            self._consumed = False
        return True

    # ------------------------------------------------------------------
    def get_latest(self) -> tuple[bool, np.ndarray | None]:
        """Return (ok, frame_bgr) — the current image, unchanged until advance()."""
        with self._lock:
            return self._ok, (self._frame.copy() if self._frame is not None else None)

    def mark_consumed(self) -> None:
        """
        Call after one full inference + logging cycle on the current image.
        Used by the pipeline to know it is safe to advance.
        """
        with self._lock:
            self._consumed = True

    @property
    def was_consumed(self) -> bool:
        with self._lock:
            return self._consumed

    def advance(self) -> bool:
        """
        Move to the next image. Returns False if at the end (and not looping).
        Called automatically by the pipeline after mark_consumed(), or
        manually via the next_image UI action.
        """
        next_idx = self._idx + 1
        if next_idx >= len(self._paths):
            if self._loop:
                return self._load(0)
            self._finished = True
            return False
        return self._load(next_idx)

    def previous(self) -> bool:
        """Step back to the previous image (manual UI control only)."""
        if self._idx <= 0:
            return False
        self._finished = False
        return self._load(self._idx - 1)

    # ------------------------------------------------------------------
    @property
    def current_name(self) -> str:
        return Path(self._paths[self._idx]).name

    @property
    def current_ground_truth(self) -> str | None:
        return _parse_ground_truth(self.current_name)

    @property
    def index(self) -> int:
        return self._idx

    @property
    def total(self) -> int:
        return len(self._paths)

    @property
    def is_finished(self) -> bool:
        return self._finished

    @property
    def progress(self) -> dict:
        return {
            "index":        self._idx + 1,
            "total":        len(self._paths),
            "name":         self.current_name,
            "ground_truth": self.current_ground_truth,
            "finished":     self._finished,
            "loop":         self._loop,
        }

    def stop(self) -> None:
        # No background thread to stop in this version; kept for API
        # compatibility with code that calls image_player.stop().
        pass
