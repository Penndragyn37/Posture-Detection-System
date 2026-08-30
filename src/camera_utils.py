"""
camera_utils.py
Enumerate available camera devices with human-readable names.

Name resolution strategy (tried in order, first to succeed wins):
  macOS  1. avfoundation via ffmpeg  (most reliable index-to-name mapping)
         2. system_profiler JSON     (fallback — can mis-order on Apple Silicon)
  Linux  1. v4l2-ctl --list-devices  (requires v4l2-utils; maps /dev/videoN → name)
  Any    N. "Camera N"               (final fallback — always works)
"""
from __future__ import annotations
import platform
import re
import subprocess
import cv2


# ── macOS helpers ─────────────────────────────────────────────────────────────

def _names_via_ffmpeg() -> dict[int, str]:
    """
    Query ffmpeg for AVFoundation camera names.
    Returns {0: "FaceTime HD Camera", 1: "Logitech C920", ...}
    Index matches OpenCV VideoCapture(i) exactly because both use AVFoundation.
    """
    try:
        result = subprocess.run(
            ["ffmpeg", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
            capture_output=True, text=True, timeout=5
        )
        # ffmpeg writes device list to stderr
        output = result.stderr

        names: dict[int, str] = {}
        # Match lines like: [AVFoundation indev @ ...] [0] FaceTime HD Camera
        pattern = re.compile(r"\[(\d+)\]\s+(.+)")
        in_video_section = False
        for line in output.splitlines():
            if "AVFoundation video devices" in line:
                in_video_section = True
                continue
            if "AVFoundation audio devices" in line:
                break
            if in_video_section:
                m = pattern.search(line)
                if m:
                    names[int(m.group(1))] = m.group(2).strip()
        return names
    except Exception:
        return {}


def _names_via_system_profiler() -> dict[int, str]:
    """
    Query macOS system_profiler for camera display names.
    Less reliable than ffmpeg for index ordering on Apple Silicon.
    """
    try:
        import json
        result = subprocess.run(
            ["system_profiler", "SPCameraDataType", "-json"],
            capture_output=True, text=True, timeout=5
        )
        data = json.loads(result.stdout)
        cameras = data.get("SPCameraDataType", [])
        return {i: c.get("_name", f"Camera {i}") for i, c in enumerate(cameras)}
    except Exception:
        return {}


def _macos_camera_names() -> dict[int, str]:
    """Try ffmpeg first, fall back to system_profiler."""
    names = _names_via_ffmpeg()
    if names:
        return names
    return _names_via_system_profiler()


# ── Linux helpers ─────────────────────────────────────────────────────────────

def _linux_camera_names() -> dict[int, str]:
    """
    Use v4l2-ctl to map /dev/videoN indices to device names.
    Requires: apt install v4l-utils
    Returns {0: "Integrated Camera", 2: "Logitech C920", ...}
    Only indices that have a VideoCapture-openable device are useful.
    """
    try:
        result = subprocess.run(
            ["v4l2-ctl", "--list-devices"],
            capture_output=True, text=True, timeout=5
        )
        # Output format:
        #   Integrated Camera (usb-0000:00:1a.0-1.6):
        #       /dev/video0
        #       /dev/video1
        names: dict[int, str] = {}
        current_name = ""
        for line in result.stdout.splitlines():
            stripped = line.strip()
            if stripped.startswith("/dev/video"):
                try:
                    idx = int(stripped.replace("/dev/video", ""))
                    if current_name:
                        names[idx] = current_name
                except ValueError:
                    pass
            elif stripped and not stripped.startswith("/dev/"):
                # Device name line (strip trailing " (usb-...)" detail)
                current_name = re.sub(r"\s*\(.*\)\s*:?\s*$", "", stripped).strip()
        return names
    except Exception:
        return {}


# ── Public API ────────────────────────────────────────────────────────────────

def list_cameras(max_index: int = 8) -> list[dict]:
    """
    Try VideoCapture indices 0..max_index-1, return those that open successfully.
    Each entry: {"id": N, "label": "FaceTime HD Camera (0)"}

    Name resolution order:
      macOS → ffmpeg → system_profiler → "Camera N"
      Linux → v4l2-ctl → "Camera N"
    """
    system = platform.system()
    if system == "Darwin":
        names = _macos_camera_names()
    elif system == "Linux":
        names = _linux_camera_names()
    else:
        names = {}

    cameras = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if not cap.isOpened():
            cap.release()
            continue

        ret, _ = cap.read()
        cap.release()
        if not ret:
            continue

        name  = names.get(i, f"Camera {i}")
        label = f"{name}  ·  [{i}]"   # e.g. "FaceTime HD Camera  ·  [0]"
        cameras.append({"id": i, "label": label})

    return cameras
