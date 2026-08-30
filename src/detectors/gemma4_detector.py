"""
gemma4_detector.py
Gemma 4 via Ollama /api/chat.

KEY FIX for E4B thinking model:
  The assistant turn is prefilled with "{" which forces the model to continue
  generating JSON immediately, bypassing the thinking/reasoning chain entirely.
  The response content will be the rest of the JSON (without the opening brace),
  so we prepend "{" when reading it back.

  This works for all Gemma 4 variants (26B MoE, E4B) regardless of whether
  think:false is respected by the Ollama version in use.
"""
from __future__ import annotations
import base64, json, re, time
import cv2
import numpy as np
import requests
from datetime import datetime, timezone

from ..detector_base import PostureDetector
from ..output_schema import PostureResult


def _extract_fields_loosely(raw: str) -> dict | None:
    """
    Fallback extractor used when json.loads() fails on an otherwise
    well-formed-looking response.

    Root cause this addresses: the model's "reasoning" text occasionally
    contains a character that breaks strict JSON string parsing (most
    commonly an unescaped quote inside a descriptive sentence). The
    posture and confidence fields are short, constrained values that are
    very unlikely to contain such characters, so they can be extracted
    independently via targeted regex even when the whole-string JSON
    parse fails. The reasoning field is then captured best-effort.
    """
    posture_m = re.search(r'"posture"\s*:\s*"(sitting|standing|absent)"', raw, re.IGNORECASE)
    conf_m    = re.search(r'"confidence"\s*:\s*([0-9.]+)', raw)
    if not (posture_m and conf_m):
        return None

    reasoning = None
    reasoning_m = re.search(r'"reasoning"\s*:\s*"(.*)', raw, re.DOTALL)
    if reasoning_m:
        reasoning = reasoning_m.group(1).rstrip()
        reasoning = re.sub(r'"\s*\}?\s*$', '', reasoning)   # trim trailing "} if present

    return {
        "posture":    posture_m.group(1).lower(),
        "confidence": float(conf_m.group(1)),
        "reasoning":  reasoning,
    }


_USER_PROMPT = (
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
    'Return ONLY the values for this JSON (do not add the opening brace — it is already there):\n'
    '"posture":"sitting","confidence":0.92,"reasoning":"brief reason"}'
)


class Gemma4Detector(PostureDetector):

    def __init__(self, config: dict):
        self.model         = config.get("model", "gemma4:26b")
        self.ollama_host   = config.get("ollama_host", "http://localhost:11434").rstrip("/")
        self.timeout       = config.get("timeout", 15)
        self.temperature   = config.get("temperature", 0.1)
        self.num_predict   = config.get("num_predict", 200)
        self.force_json    = config.get("force_json", True)
        self.think         = config.get("think", False)
        self.use_prefill   = config.get("use_prefill", True)
        self.approach_name = "gemma4_mobile" if "e4b" in self.model.lower() else "gemma4_server"

    # ------------------------------------------------------------------ #
    def detect(self, frame: np.ndarray) -> PostureResult:
        t_start = time.perf_counter()
        frame_b64 = self._encode(frame)

        # Build the messages array. The assistant prefill (an opening "{")
        # forces the model to skip its thinking phase and continue directly
        # into JSON. This is the most reliable fix for the E4B/26B thinking
        # behaviour; use_prefill=False reverts to a plain two-message
        # request for comparison/ablation purposes.
        messages = [
            {
                "role": "user",
                "content": _USER_PROMPT,
                "images": [frame_b64],
            },
        ]
        if self.use_prefill:
            messages.append({"role": "assistant", "content": "{"})

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": self.think,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.num_predict,
            },
        }
        if self.force_json:
            payload["format"] = "json"

        try:
            resp = requests.post(
                f"{self.ollama_host}/api/chat",
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            body = resp.json()
            msg  = body.get("message", {})

            # When use_prefill is True, the model continues from the "{" prefill
            # and Ollama returns only the new tokens, so we prepend the brace
            # back. When use_prefill is False, content should already be
            # complete JSON (if force_json worked) or plain text otherwise.
            content = msg.get("content", "").strip()
            if content:
                if self.use_prefill and not content.startswith("{"):
                    raw = "{" + content
                else:
                    raw = content
            else:
                # thinking:false not respected — try the thinking field as fallback
                thinking = msg.get("thinking", "")
                m = re.search(r'\{[^{}]*"posture"\s*:[^{}]*\}', thinking)
                raw = m.group(0) if m else ""
                if raw:
                    print(f"[PDSDA] Extracted JSON from thinking field (think:false ignored)")

        except requests.Timeout:
            return PostureResult.error(
                self.approach_name,
                f"Timeout after {self.timeout}s — raise 'timeout' in config.yaml",
                self._ms(t_start),
            )
        except requests.ConnectionError:
            return PostureResult.error(self.approach_name, "Cannot reach Ollama — run: ollama serve", self._ms(t_start))
        except requests.RequestException as exc:
            return PostureResult.error(self.approach_name, str(exc), self._ms(t_start))

        if not raw:
            print(f"[PDSDA] Empty response. Full body: {json.dumps(body)[:400]}")
            print(f"[PDSDA] Verify: ollama show {self.model}")
            return PostureResult.error(
                self.approach_name,
                f"Empty response from Ollama — check terminal for details",
                self._ms(t_start),
            )

        # ── Parse JSON ──────────────────────────────────────────────── #
        try:
            clean = raw.strip().strip("`")
            if clean.lower().startswith("json"):
                clean = clean[4:].strip()
            data = json.loads(clean)
        except json.JSONDecodeError:
            # Strict parse failed. This commonly happens when the model's
            # reasoning text contains an unescaped quote or similar character
            # that breaks JSON string boundaries, even though posture and
            # confidence are perfectly readable. Try a loose field-by-field
            # extraction before giving up.
            loose = _extract_fields_loosely(raw)
            if loose:
                print(f"[PDSDA] Strict JSON parse failed but recovered fields via regex fallback "
                      f"(posture={loose['posture']}, confidence={loose['confidence']})")
                data = loose
            else:
                # Log the FULL raw text (not a 120-char slice) so the actual
                # cause is visible for diagnosis, not hidden by truncation.
                print(f"[PDSDA] JSON parse failed and regex fallback found no fields. "
                      f"Full raw response below:\n{raw}")
                return PostureResult.error(
                    self.approach_name,
                    f"JSON parse failed, no recoverable fields — raw ({len(raw)} chars): {raw[:300]}",
                    self._ms(t_start),
                )

        posture = str(data.get("posture", "absent")).lower().strip()
        if posture not in ("sitting", "standing", "absent"):
            posture = "absent"

        return PostureResult(
            posture=posture,
            confidence=min(1.0, max(0.0, float(data.get("confidence", 0.5)))),
            approach=self.approach_name,
            latency_ms=self._ms(t_start),
            timestamp=datetime.now(timezone.utc).isoformat(),
            reasoning=data.get("reasoning"),
        )

    # ------------------------------------------------------------------ #
    def health_check(self) -> bool:
        try:
            resp = requests.get(f"{self.ollama_host}/api/tags", timeout=5)
            if not resp.ok:
                return False
            models = [m.get("name", "") for m in resp.json().get("models", [])]
            base   = self.model.split(":")[0].lower()
            found  = any(base in m.lower() for m in models)
            if not found:
                print(f"[PDSDA] '{self.model}' not found. Pull: ollama pull {self.model}")
            return found
        except Exception:
            return False

    def teardown(self) -> None:
        pass

    # ------------------------------------------------------------------ #
    @staticmethod
    def _encode(frame: np.ndarray) -> str:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        _, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buf).decode("utf-8")

    @staticmethod
    def _ms(t_start: float) -> float:
        return round((time.perf_counter() - t_start) * 1000, 1)
