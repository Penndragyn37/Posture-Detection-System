"""
detector_base.py
Abstract base class that all inference approach modules must implement.
Satisfies SRS FR-011 (PostureDetector interface).
"""
from abc import ABC, abstractmethod
import numpy as np
from .output_schema import PostureResult


class PostureDetector(ABC):
    """
    Common interface for all posture detection approaches.
    Swap approaches by changing the 'approach' key in config.yaml — no code changes needed.
    """

    @abstractmethod
    def detect(self, frame: np.ndarray) -> PostureResult:
        """
        Classify posture from a preprocessed RGB frame.

        Args:
            frame: np.ndarray of shape (H, W, 3), dtype uint8, RGB colour order.

        Returns:
            PostureResult with posture class, confidence, latency, and reasoning.
        """
        ...

    @abstractmethod
    def health_check(self) -> bool:
        """
        Return True if the detector is ready to accept frames.
        Called during startup pre-flight (SRS FR-004).
        """
        ...

    @abstractmethod
    def teardown(self) -> None:
        """
        Release any held resources (API sessions, model handles, etc.).
        Called when the pipeline stops (SRS FR-005 graceful shutdown).
        """
        ...
