"""
output_schema.py
Canonical output type for all PostureDetector implementations.
Satisfies SRS FR-027 (shared JSON schema).
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional
import json


@dataclass
class PostureResult:
    posture: str            # "sitting" | "standing" | "absent"
    confidence: float       # 0.0 – 1.0
    approach: str           # "gemma4_server" | "gemma4_mobile" | "claude_api"
    latency_ms: float       # wall-clock ms from frame receipt to result
    timestamp: str          # ISO 8601 UTC
    reasoning: Optional[str] = None   # VLM explanation; None for non-VLM approaches

    # Convenience ---------------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    def is_low_confidence(self, threshold: float = 0.70) -> bool:
        return self.confidence < threshold

    def is_error(self) -> bool:
        return self.reasoning is not None and self.reasoning.startswith("ERROR:")

    @classmethod
    def error(cls, approach: str, message: str, latency_ms: float = 0.0) -> "PostureResult":
        from datetime import datetime, timezone
        return cls(
            posture="absent",
            confidence=0.0,
            approach=approach,
            latency_ms=round(latency_ms, 1),
            timestamp=datetime.now(timezone.utc).isoformat(),
            reasoning=f"ERROR: {message}",
        )
