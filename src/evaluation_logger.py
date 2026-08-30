"""
evaluation_logger.py
Writes per-frame detection results to a CSV log file for benchmarking.
Satisfies SRS FR-028, FR-029 (continuous, crash-safe logging).
"""
from __future__ import annotations
import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from .output_schema import PostureResult


_FIELDS = [
    "frame_id", "timestamp", "approach",
    "predicted", "confidence", "latency_ms",
    "system_state", "reasoning",
    "cpu_pct", "ram_mb", "gpu_pct",
    "ground_truth", "correct", "image_name",
]


class EvaluationLogger:
    def __init__(self, config: dict):
        log_dir = Path(config.get("log_path", "./logs"))
        log_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file   = log_dir / f"detection_{ts}.csv"
        self.frame_count = 0
        self._save_frames = config.get("save_frames", False)
        self._frame_dir   = log_dir / f"frames_{ts}" if self._save_frames else None

        if self._frame_dir:
            self._frame_dir.mkdir(parents=True, exist_ok=True)

        # Write CSV header
        with open(self.log_file, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=_FIELDS).writeheader()

    # ------------------------------------------------------------------
    def log(
        self,
        result: PostureResult,
        system_state: str,
        ground_truth: Optional[str] = None,
        frame=None,
        image_name: Optional[str] = None,
    ) -> None:
        self.frame_count += 1
        fid = f"frame_{self.frame_count:06d}"

        cpu_pct = ram_mb = gpu_pct = None
        try:
            import psutil
            cpu_pct = round(psutil.cpu_percent(interval=None), 1)
            ram_mb  = round(psutil.Process().memory_info().rss / (1024 ** 2), 1)
        except ImportError:
            pass

        try:
            import platform, subprocess
            if platform.system() == "Linux":
                # NVIDIA GPU (Linux / Windows WSL)
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                    timeout=1, text=True,
                ).strip()
                gpu_pct = float(out.split("\n")[0])
            # macOS Apple Silicon: Metal GPU utilisation via powermetrics (requires sudo)
            # Skipped here to avoid sudo requirement; monitor via Activity Monitor instead.
        except Exception:
            pass

        row = {
            "frame_id":    fid,
            "timestamp":   result.timestamp,
            "approach":    result.approach,
            "predicted":   result.posture,
            "confidence":  round(result.confidence, 4),
            "latency_ms":  result.latency_ms,
            "system_state": system_state,
            "reasoning":   (result.reasoning or "")[:200],
            "cpu_pct":     cpu_pct,
            "ram_mb":      ram_mb,
            "gpu_pct":     gpu_pct,
            "ground_truth": ground_truth,
            "correct":     (ground_truth == result.posture) if ground_truth else None,
            "image_name":  image_name,
        }

        # Append row — open/close every write so a crash does not corrupt data (FR-029)
        with open(self.log_file, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=_FIELDS).writerow(row)

        # Optional frame save
        if self._save_frames and frame is not None:
            try:
                import cv2
                import numpy as np
                p = self._frame_dir / f"{fid}_{result.posture}.jpg"
                cv2.imwrite(str(p), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            except Exception:
                pass

    @property
    def log_path(self) -> str:
        return str(self.log_file)
