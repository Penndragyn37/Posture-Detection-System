"""
claude_detector.py
Proof-of-concept path: claude-sonnet-4-6 via the Anthropic Messages API.
Runs on any PC with a webcam and internet connection — no GPU required.

Satisfies SRS FR-010, FR-011, NFR-003, NFR-015 (privacy disclosure required at setup).
Cloud connectivity risks documented in WP2.1 literature review.
"""
from __future__ import annotations
import base64
import json
import time
import cv2
import numpy as np
from datetime import datetime, timezone

from ..detector_base import PostureDetector
from ..output_schema import PostureResult

try:
    import anthropic
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False

_SYSTEM_PROMPT = (
    "You are a posture classification system. "
    "Look at the image and classify the person's posture as EXACTLY one of: "
    "sitting, standing, or absent.\n\n"
    "Rules:\n"
    "- sitting: a PERSON is clearly visible and seated — hips below shoulder level. "
    "This includes any position ON a chair or seat, regardless of how they are "
    "positioned on it: crouching on a chair, kneeling on a chair, curled up on a "
    "chair, or lying across a chair all count as sitting.\n"
    "- standing: a PERSON is clearly visible and upright on their feet, hips above "
    "knee level.\n"
    "- absent: NO person is visible. This includes: empty rooms, an empty chair "
    "with nobody in it, a desk with no one present, or any scene with only "
    "furniture and no human body. An empty office chair = absent.\n\n"
    "Edge case mappings — use these when the posture is unusual:\n"
    "- Person crouching or squatting on the floor → absent "
    "(not using the desk; treat as not present)\n"
    "- Person kneeling on the floor → absent\n"
    "- Person lying on the floor → absent\n"
    "- Person crouching ON a chair or seat → sitting\n"
    "- Person lying across a chair or seat → sitting\n"
    "- Person bending far forward from a chair (e.g. picking something up) → sitting "
    "(temporary movement, still occupying the chair)\n"
    "- Person bending far forward while standing → standing\n\n"
    "IMPORTANT: a chair, desk, or other furniture alone does NOT mean sitting. "
    "A person must be visible and occupying the chair for it to be sitting.\n\n"
    "Return ONLY valid JSON with no other text:\n"
    '{"posture": "sitting"|"standing"|"absent", "confidence": 0.0-1.0, "reasoning": "brief explanation"}'
)

_USER_TEXT = (
    "Classify the posture in this image using the rules in the system prompt. "
    "Return only the JSON object: "
    '{"posture": "sitting"|"standing"|"absent", "confidence": 0.0-1.0, "reasoning": "..."}'
)


class ClaudeDetector(PostureDetector):
    """
    Cloud-based proof-of-concept. Frames are encoded as base64 JPEG and sent
    to Anthropic servers for inference.

    PRIVACY NOTE (SRS NFR-015): the user must acknowledge before first use
    that camera frames will be transmitted to Anthropic servers. This
    disclosure is shown in the UI startup flow.
    """

    def __init__(self, config: dict):
        if not _ANTHROPIC_AVAILABLE:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")

        self.model        = config.get("model", "claude-sonnet-4-6")
        self.max_tokens   = config.get("max_tokens", 150)
        self.timeout      = config.get("timeout", 10)
        self.approach_name = "claude_api"
        self._client      = anthropic.Anthropic()

    # ------------------------------------------------------------------
    def detect(self, frame: np.ndarray) -> PostureResult:
        t_start = time.perf_counter()
        frame_b64 = self._encode(frame)

        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=_SYSTEM_PROMPT,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": frame_b64,
                            },
                        },
                        {"type": "text", "text": _USER_TEXT},
                    ],
                }],
            )
            raw = response.content[0].text.strip()

        except anthropic.APIConnectionError:
            return PostureResult.error(self.approach_name, "No internet connection", self._elapsed(t_start))
        except anthropic.AuthenticationError:
            return PostureResult.error(self.approach_name, "API key invalid or expired", self._elapsed(t_start))
        except anthropic.RateLimitError:
            return PostureResult.error(self.approach_name, "Rate limit (429) — backing off", self._elapsed(t_start))
        except anthropic.APIStatusError as exc:
            return PostureResult.error(self.approach_name, f"API {exc.status_code}", self._elapsed(t_start))
        except Exception as exc:
            return PostureResult.error(self.approach_name, str(exc), self._elapsed(t_start))

        try:
            clean = raw.strip("`").strip()
            if clean.lower().startswith("json"):
                clean = clean[4:].strip()
            data = json.loads(clean)
        except json.JSONDecodeError:
            return PostureResult.error(
                self.approach_name,
                f"JSON parse failed — raw: {raw[:120]}",
                self._elapsed(t_start),
            )

        posture = str(data.get("posture", "absent")).lower()
        if posture not in ("sitting", "standing", "absent"):
            posture = "absent"

        return PostureResult(
            posture=posture,
            confidence=min(1.0, max(0.0, float(data.get("confidence", 0.5)))),
            approach=self.approach_name,
            latency_ms=self._elapsed(t_start),
            timestamp=datetime.now(timezone.utc).isoformat(),
            reasoning=data.get("reasoning"),
        )

    # ------------------------------------------------------------------
    def health_check(self) -> bool:
        """
        Validate API key with a minimal call.
        Note: this also constitutes a network connectivity check (SRS FR-004).
        """
        if not _ANTHROPIC_AVAILABLE:
            return False
        try:
            self._client.messages.create(
                model=self.model,
                max_tokens=5,
                messages=[{"role": "user", "content": "ping"}],
            )
            return True
        except anthropic.AuthenticationError:
            return False
        except Exception:
            return False

    def teardown(self) -> None:
        pass

    # ------------------------------------------------------------------
    @staticmethod
    def _encode(frame: np.ndarray) -> str:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        _, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buf).decode("utf-8")

    @staticmethod
    def _elapsed(t_start: float) -> float:
        return round((time.perf_counter() - t_start) * 1000, 1)
