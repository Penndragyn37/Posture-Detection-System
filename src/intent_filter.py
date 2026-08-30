"""
intent_filter.py
Seven-state intent detection machine.

Filters transient posture events (shoe-tying, floor pickup) using the
5-frame consecutive confirmation rule, then runs a visual countdown before
emitting any desk adjustment signal.

Satisfies SRS FR-014–FR-026, NFR-007, NFR-011–NFR-013 (safety invariant).
State machine documented in WP3.0 and the interactive diagram artifact.
"""
from __future__ import annotations
import copy
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

from .output_schema import PostureResult


class SystemState(Enum):
    IDLE               = "idle"
    MONITORING         = "monitoring"
    TRANSITION_PENDING = "transition_pending"
    SIGNALING          = "signaling"
    ADJUSTING          = "adjusting"
    COOLDOWN           = "cooldown"
    ERROR              = "error"
    EVALUATING         = "evaluating"   # image-player mode: independent per-image
                                         # classification, state machine bypassed


# Human-readable labels for the UI
STATE_LABELS = {
    SystemState.IDLE:               "Idle",
    SystemState.MONITORING:         "Monitoring",
    SystemState.TRANSITION_PENDING: "Change detected...",
    SystemState.SIGNALING:          "Adjusting soon",
    SystemState.ADJUSTING:          "Adjusting",
    SystemState.COOLDOWN:           "Cooldown",
    SystemState.ERROR:              "Error",
    SystemState.EVALUATING:         "Evaluating image",
}

STATE_COLORS = {
    SystemState.IDLE:               "#888780",
    SystemState.MONITORING:         "#1D9E75",
    SystemState.TRANSITION_PENDING: "#BA7517",
    SystemState.SIGNALING:          "#185FA5",
    SystemState.ADJUSTING:          "#3B6D11",
    SystemState.COOLDOWN:           "#534AB7",
    SystemState.ERROR:              "#A32D2D",
    SystemState.EVALUATING:         "#5E6AD2",
}


@dataclass
class IntentState:
    system_state:       SystemState = SystemState.IDLE
    current_posture:    str         = "unknown"
    pending_posture:    str         = ""
    consecutive_frames: int         = 0
    countdown_remaining: float      = 0.0
    cooldown_remaining:  float      = 0.0
    last_error:         str         = ""

    def as_dict(self) -> dict:
        return {
            "system_state":        self.system_state.value,
            "state_label":         STATE_LABELS[self.system_state],
            "state_color":         STATE_COLORS[self.system_state],
            "current_posture":     self.current_posture,
            "pending_posture":     self.pending_posture,
            "consecutive_frames":  self.consecutive_frames,
            "countdown_remaining": round(self.countdown_remaining, 1),
            "cooldown_remaining":  round(self.cooldown_remaining, 1),
            "last_error":          self.last_error,
        }


class IntentFilter:
    """
    SAFETY INVARIANT (SRS NFR-011, NFR-013):
        The ADJUSTING state is ONLY reachable from SIGNALING.
        SIGNALING is ONLY reachable from TRANSITION_PENDING after threshold.
        No path from MONITORING directly to ADJUSTING exists.
    """

    def __init__(
        self,
        config: dict,
        on_adjust: Optional[Callable[[str], None]] = None,
        on_state_change: Optional[Callable[[IntentState], None]] = None,
    ):
        self.threshold          = int(config.get("threshold_frames", 5))
        self.cooldown_seconds   = float(config.get("cooldown_seconds", 60))
        self.countdown_seconds  = float(config.get("countdown_seconds", 10))
        self.confidence_min     = float(config.get("confidence_min", 0.70))

        self._state     = IntentState()
        self._lock      = threading.Lock()
        self._low_conf_streak = 0

        self.on_adjust       = on_adjust
        self.on_state_change = on_state_change

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        with self._lock:
            self._state.system_state    = SystemState.MONITORING
            self._state.current_posture = "unknown"
            self._notify()

    def process(self, result: PostureResult) -> IntentState:
        """Feed a PostureResult into the state machine. Thread-safe."""
        with self._lock:
            st = self._state.system_state

            # States that reject incoming frames
            if st in (SystemState.IDLE, SystemState.ADJUSTING):
                return copy.copy(self._state)

            posture    = result.posture
            confidence = result.confidence

            # Low-confidence gating (SRS FR-012, FR-013)
            if confidence < self.confidence_min:
                self._low_conf_streak += 1
                if self._low_conf_streak >= 10 and st != SystemState.ERROR:
                    self._state.last_error = "Sustained low confidence — check camera/lighting"
                    self._state.system_state = SystemState.ERROR
                    self._notify()
                return copy.copy(self._state)
            else:
                self._low_conf_streak = 0

            if st == SystemState.MONITORING:
                self._handle_monitoring(posture)

            elif st == SystemState.TRANSITION_PENDING:
                self._handle_transition(posture, confidence)

            elif st == SystemState.SIGNALING:
                # Posture reverted during countdown — auto-cancel
                if posture == self._state.current_posture:
                    self._cancel_internal()

            elif st == SystemState.COOLDOWN:
                pass  # ignore frames during cooldown

            elif st == SystemState.ERROR:
                pass  # error must be cleared externally

            return copy.copy(self._state)

    def cancel(self) -> None:
        """User clicked Cancel or pressed Esc."""
        with self._lock:
            if self._state.system_state in (
                SystemState.TRANSITION_PENDING, SystemState.SIGNALING
            ):
                self._cancel_internal()

    def confirm_now(self) -> None:
        """User clicked Adjust Now — skip remaining countdown."""
        with self._lock:
            if self._state.system_state == SystemState.SIGNALING:
                self._state.countdown_remaining = 0.0

    def clear_error(self) -> None:
        with self._lock:
            if self._state.system_state == SystemState.ERROR:
                self._state.system_state       = SystemState.MONITORING
                self._state.last_error         = ""
                self._low_conf_streak          = 0
                self._state.consecutive_frames = 0   # T1/F1: reset debounce counter on error recovery
                self._state.pending_posture    = ""   # T1/F1: clear stale pending posture on error recovery
                self._notify()

    def set_error(self, message: str) -> None:
        with self._lock:
            self._state.system_state = SystemState.ERROR
            self._state.last_error   = message
            self._notify()

    @property
    def state(self) -> IntentState:
        with self._lock:
            return copy.copy(self._state)

    # ------------------------------------------------------------------
    # Internal state transitions (all called within self._lock)
    # ------------------------------------------------------------------

    def _handle_monitoring(self, posture: str) -> None:
        if posture != self._state.current_posture and posture != "unknown":
            if self._state.pending_posture == posture:
                self._state.consecutive_frames += 1
            else:
                self._state.pending_posture    = posture
                self._state.consecutive_frames = 1

            self._state.system_state = SystemState.TRANSITION_PENDING
            self._notify()

            if self._state.consecutive_frames >= self.threshold:
                self._enter_signaling()
        else:
            # Same as current — reset any pending count
            self._state.pending_posture    = ""
            self._state.consecutive_frames = 0

    def _handle_transition(self, posture: str, confidence: float) -> None:
        if posture == self._state.pending_posture:
            self._state.consecutive_frames += 1
            self._notify()
            if self._state.consecutive_frames >= self.threshold:
                self._enter_signaling()

        elif posture == self._state.current_posture:
            # Reverted before threshold — back to monitoring (FR-015)
            self._state.system_state       = SystemState.MONITORING
            self._state.pending_posture    = ""
            self._state.consecutive_frames = 0
            self._notify()

        else:
            # Changed to a THIRD posture — restart counter
            self._state.pending_posture    = posture
            self._state.consecutive_frames = 1
            self._notify()

    def _enter_signaling(self) -> None:
        """
        ONLY valid entry point to SIGNALING state.
        Spawns countdown thread. ADJUSTING is only reachable via this path.
        """
        self._state.system_state       = SystemState.SIGNALING
        self._state.countdown_remaining = self.countdown_seconds
        self._notify()

        t = threading.Thread(target=self._countdown_thread, daemon=True)
        t.start()

    def _countdown_thread(self) -> None:
        tick = 0.1
        while True:
            time.sleep(tick)
            with self._lock:
                if self._state.system_state != SystemState.SIGNALING:
                    return
                self._state.countdown_remaining = max(0.0, self._state.countdown_remaining - tick)
                self._notify()
                if self._state.countdown_remaining <= 0.0:
                    # SAFETY INVARIANT: ADJUSTING only reached from here
                    self._trigger_adjustment()
                    return

    def _trigger_adjustment(self) -> None:
        """Called within lock. Only reachable from _countdown_thread."""
        confirmed_posture               = self._state.pending_posture
        self._state.system_state        = SystemState.ADJUSTING
        self._state.current_posture     = confirmed_posture
        self._state.pending_posture     = ""
        self._state.consecutive_frames  = 0
        self._notify()

        if self.on_adjust:
            # Fire callback outside lock via thread to avoid deadlock
            threading.Thread(
                target=self.on_adjust, args=(confirmed_posture,), daemon=True
            ).start()

        # Immediately enter cooldown (SRS FR-025)
        self._state.system_state       = SystemState.COOLDOWN
        self._state.cooldown_remaining = self.cooldown_seconds
        self._notify()

        t = threading.Thread(target=self._cooldown_thread, daemon=True)
        t.start()

    def _cooldown_thread(self) -> None:
        tick = 1.0
        while True:
            time.sleep(tick)
            with self._lock:
                if self._state.system_state != SystemState.COOLDOWN:
                    return
                self._state.cooldown_remaining = max(0.0, self._state.cooldown_remaining - tick)
                self._notify()
                if self._state.cooldown_remaining <= 0.0:
                    self._state.system_state = SystemState.MONITORING
                    self._notify()
                    return

    def _cancel_internal(self) -> None:
        """Cancel without acquiring the lock (already held by caller)."""
        self._state.system_state       = SystemState.MONITORING
        self._state.pending_posture    = ""
        self._state.consecutive_frames = 0
        self._state.countdown_remaining = 0.0
        self._notify()

    def _notify(self) -> None:
        if self.on_state_change:
            snapshot = copy.copy(self._state)
            threading.Thread(
                target=self.on_state_change, args=(snapshot,), daemon=True
            ).start()
