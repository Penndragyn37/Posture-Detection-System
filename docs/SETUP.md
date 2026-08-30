# PDSDA — Application Setup Guide

Detailed install, configuration, and operation reference for the posture
detection application itself (as opposed to the project-level overview in
the root [`README.md`](../README.md)).

---

## Hardware notes (tuned for MacBook Pro M5 Pro, 48 GB)

The shipped `config/config.yaml` defaults were tuned on this hardware profile:

| Spec | M5 Pro value | Implication for PDSDA |
|---|---|---|
| GPU cores | Up to 20-core GPU | Each core has a Neural Accelerator for AI |
| AI compute | 4x peak vs M4 Pro | Gemma 4 inference is fast out of the box |
| Unified memory | 48 GB | Run Gemma 4 26B MoE (~15 GB) with 33 GB to spare |
| Memory bandwidth | 307 GB/s | LLM inference is memory-bandwidth-bound; this is excellent |
| Architecture | Fusion Architecture (dual-die) | Designed explicitly for professional AI workloads |

**Expected performance** for posture detection with `gemma4:26b`:
- Around 75 tokens/second via Ollama with Metal (llama.cpp backend)
- A posture classification response is roughly 80 tokens, so approximately
  1–2 seconds per frame — well within the 1 fps polling rate
- With Ollama 0.19+ MLX preview enabled, significantly faster

If you're running on different hardware (Linux + NVIDIA GPU, as in the
capstone's EC2 benchmark, or an older Mac), adjust `gemma4.timeout`,
`gemma4.image_tokens`, and `sampling.fps` in `config/config.yaml` accordingly
— see the config reference table below.

---

## Quick start

### Step 1 — Camera permission (required)
macOS: System Settings > Privacy & Security > Camera > enable Terminal (or
your IDE). Without this, OpenCV opens the camera but returns no frames.

### Step 2 — Install Ollama
Download from [ollama.ai](https://ollama.ai) and install.

### Step 3 — Pull Gemma 4
```bash
# Primary choice — 26B MoE: ~15 GB loaded, runs on 48 GB unified memory with room to spare
ollama pull gemma4:26b

# Optional maximum quality — 31B Dense: ~22 GB loaded
# NOTE: set OLLAMA_FLASH_ATTENTION=0 to avoid a known hang bug in Ollama 0.19
ollama pull gemma4:31b
```

### Step 4 — Install Python dependencies
```bash
pip install -r requirements.txt
```
If `opencv-python` fails to install: `brew install cmake pkg-config` first
(macOS), then retry.

### Step 5 — Run
```bash
# Start Ollama (if not already running via the menu bar app)
ollama serve

# Run PDSDA with Gemma 4 (browser opens automatically)
python run.py --approach gemma4_server
```

**Or with Claude API** (no Ollama needed, any machine, internet required):
```bash
export ANTHROPIC_API_KEY=sk-ant-...
python run.py --approach claude_api
```

---

## Ollama environment variables (recommended)

Add to your shell profile (`~/.zshrc` or `~/.bashrc`) before running PDSDA
for best stability:

```bash
export OLLAMA_FLASH_ATTENTION=0      # avoids known Gemma 4 hang on complex prompts
export OLLAMA_NUM_PARALLEL=1         # PDSDA sends one frame at a time
export OLLAMA_MAX_LOADED_MODELS=1    # keep Gemma 4 loaded between frames
```

## Ollama MLX preview (optional — Apple Silicon only, extra speed)

Ollama 0.19+ ships a preview MLX backend that leverages Apple Silicon Neural
Accelerators more aggressively:

```bash
OLLAMA_USE_MLX=1 ollama serve
```

MLX support for Gemma 4 vision models is still in preview as of June 2026.
If you see errors, omit `OLLAMA_USE_MLX=1` — the standard Metal backend
works fine.

---

## Model selection guide

| Model | Size loaded | Image tokens | Use when |
|---|---|---|---|
| `gemma4:26b` | ~15 GB | 560 (config default) | Best balance; recommended starting point |
| `gemma4:31b` | ~22 GB | 560 | Maximum quality; set `OLLAMA_FLASH_ATTENTION=0` |
| `gemma4:e4b` | ~3 GB | 280 | Mobile/edge deployment; lower accuracy, higher latency in this project's benchmark |
| Claude API | 0 GB (cloud) | auto | Highest accuracy in this project's benchmark; internet required, per-request cost |

See the root README's results table and the final report
(`docs/final_report/`) for the full accuracy/latency/cost tradeoff.

---

## Configuration reference (`config/config.yaml`)

| Key | Default | Notes |
|---|---|---|
| `gemma4.model` | `gemma4:26b` | Change to `gemma4:31b` for max quality, `gemma4:e4b` for mobile |
| `gemma4.image_tokens` | `560` | Higher than the 280 minimum; raise/lower for detail vs. speed |
| `gemma4.timeout` | `8` | Seconds; raise on slower hardware |
| `gemma4.force_json` | `true` | Forces valid JSON output; do not disable for real runs |
| `gemma4.think` | `false` | Asks Ollama to skip chain-of-thought (not always honoured) |
| `gemma4.use_prefill` | `true` | The actual fix for the thinking-model empty-response bug |
| `gemma4.temperature` | `0.1` | Low = consistent, reproducible classifications |
| `gemma4.num_predict` | `200` | Max output tokens; headroom for the reasoning field |
| `sampling.fps` | `1` | Increase to `2` if inference is consistently under 500 ms |
| `intent.threshold_frames` | `5` | Consecutive frames of consistent posture to confirm |
| `intent.countdown_seconds` | `10` | Desk won't adjust without this warning period |
| `intent.cooldown_seconds` | `60` | Lock-out after every adjustment (oscillation prevention) |
| `intent.confidence_min` | `0.70` | Frames below this neither increment nor reset the counter |

Every Gemma 4 inference toggle above has a full explanation, the symptom it
fixes, and a copy-paste test to verify it, in
[`OLLAMA_TOGGLE_GUIDE.md`](OLLAMA_TOGGLE_GUIDE.md). Read this before
benchmarking, especially the thinking-model prefill fix.

---

## UI controls

| Control | Action |
|---|---|
| Approach selector (top right) | Hot-swap Gemma 4 / Claude API without restart |
| **Cancel** button | Abort a pending adjustment during countdown |
| **Adjust now** | Skip remaining countdown |
| **Esc** key | Cancel keyboard shortcut |

---

## Evaluation logs

`logs/detection_YYYYMMDD_HHMMSS.csv` — fields:
`frame_id, timestamp, approach, predicted, confidence, latency_ms,
system_state, reasoning, cpu_pct, ram_mb, gpu_pct, ground_truth, correct`

GPU utilisation (`gpu_pct`) is null on macOS (`nvidia-smi` not available);
monitor Metal GPU usage via Activity Monitor > GPU History instead. On the
Linux/NVIDIA benchmark path, `gpu_pct` is populated normally.

Historical logs from the capstone evaluation (dev/test runs for all three
approaches, plus UAT logs T2–T8) are committed under
[`data/evaluation_logs/`](../data/evaluation_logs/) — analyse them with
`scripts/analyse_results.py`.

---

## Utility scripts

```bash
# Compute accuracy / per-class F1 / confusion matrix / latency stats from a log
python scripts/analyse_results.py data/evaluation_logs/claude_test_detection_20260620_115948.csv

# Compare two approaches side by side
python scripts/analyse_results.py data/evaluation_logs/gemma4_26b_test_detection_20260624_015454.csv data/evaluation_logs/claude_test_detection_20260620_115948.csv

# Pre-flight check that Ollama + Gemma 4 vision are working before a full run
python tests/test_ollama.py

# Organize captured frames (config: save_frames: true) into a labeled dataset
python scripts/collect_dataset.py label logs/frames_20260618_120000/ sitting
python scripts/collect_dataset.py check ./dataset/
```
