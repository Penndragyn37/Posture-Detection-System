#!/usr/bin/env python3
"""
test_ollama.py — run before the full pipeline to verify Ollama and Gemma 4 vision.
Usage:  python test_ollama.py
"""
import sys, json, base64, requests
import numpy as np
import cv2

OLLAMA = "http://localhost:11434"
MODEL  = None  # auto-detected from config or command line
USE_JSON_FORMAT = True  # matches gemma4_detector.py's use_json_format setting

# Load model from config if available
try:
    import yaml
    cfg = yaml.safe_load(open("config/config.yaml"))
    gemma_cfg = cfg.get("gemma4", {})
    MODEL = gemma_cfg.get("model", "gemma4:26b")
    USE_JSON_FORMAT = gemma_cfg.get("use_json_format", True)
except Exception:
    MODEL = sys.argv[1] if len(sys.argv) > 1 else "gemma4:26b"

print(f"\nPDSDA Ollama diagnostic  (model: {MODEL}, use_json_format: {USE_JSON_FORMAT})\n" + "="*50)

def chk(label, ok, detail=""):
    print(f"  [{'OK  ' if ok else 'FAIL'}] {label}" + (f"  — {detail}" if detail else ""))
    return ok

# ── 1. Ollama running ──────────────────────────────────────────────────
try:
    r = requests.get(f"{OLLAMA}/api/tags", timeout=3)
    models = [m["name"] for m in r.json().get("models", [])] if r.ok else []
    if not chk("Ollama reachable", r.ok, f"{len(models)} model(s) loaded"):
        print("\nFix:  ollama serve")
        sys.exit(1)
    print(f"         Installed models: {models}")
except Exception as e:
    chk("Ollama reachable", False, str(e))
    print("\nFix:  ollama serve")
    sys.exit(1)

# ── 2. Model present ──────────────────────────────────────────────────
base = MODEL.split(":")[0].lower()
found_name = next((m for m in models if base in m.lower()), None)
if not chk(f"Model '{MODEL}'", bool(found_name), found_name or f"not found — run: ollama pull {MODEL}"):
    sys.exit(1)

# ── 3. Text-only call ─────────────────────────────────────────────────
print("\n  Testing text inference...")
text_payload = {
    "model": MODEL,
    "messages": [{"role":"user","content":'Reply with only: {"ok":true}'}],
    "stream": False,
    "options": {"num_predict": 20}
}
if USE_JSON_FORMAT:
    text_payload["format"] = "json"
r = requests.post(f"{OLLAMA}/api/chat", json=text_payload, timeout=30)
raw_text = r.json().get("message",{}).get("content","").strip()
chk("Text inference", bool(raw_text), raw_text[:60] if raw_text else "empty response")

# ── 4. Vision call ────────────────────────────────────────────────────
print("\n  Testing vision inference (sending test image)...")
# Simple 64x64 blue rectangle
img = np.zeros((64,64,3), dtype=np.uint8); img[:,:] = [100,150,200]
_, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 50])
b64 = base64.b64encode(buf).decode()

vision_prompt = (
    "Look at this image. Classify the posture as one of: sitting, standing, absent. "
    'Return ONLY JSON: {"posture":"absent","confidence":0.9,"reasoning":"test"}'
)

vision_payload = {
    "model": MODEL,
    "messages": [{"role":"user","content": vision_prompt,"images":[b64]}],
    "stream": False,
    "options": {"temperature":0.1, "num_predict":100},
}
if USE_JSON_FORMAT:
    vision_payload["format"] = "json"
r = requests.post(f"{OLLAMA}/api/chat", json=vision_payload, timeout=45)

body = r.json()
raw_vision = body.get("message",{}).get("content","").strip()

if raw_vision:
    try:
        data = json.loads(raw_vision)
        chk("Vision inference", True,
            f"posture={data.get('posture')}, confidence={data.get('confidence')}")
        print(f"\n  Raw: {raw_vision[:120]}")
        print(f"\n{'='*50}")
        print("  All checks passed — run:  python run.py\n")
    except json.JSONDecodeError:
        chk("Vision inference — JSON parse", False, f"raw: {raw_vision[:120]}")
else:
    chk("Vision inference", False, "empty response — model may not support vision")
    print(f"\n  Full Ollama body: {json.dumps(body)[:400]}")
    print(f"\n  Try:  ollama show {MODEL}")
    print(f"  Look for 'projector' or 'vision' in the output.")
    print(f"  If absent, this is a text-only model. Try:  ollama pull llava")
