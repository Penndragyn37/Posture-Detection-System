# Posture Detection System (PDSDA)
**Camera-Based Posture Detection for Smart Desk Automation**
MEng Capstone · Robotics & Intelligent Autonomous Systems · University of Cincinnati

Real-time camera-based posture detection (sitting / standing / absent) using
zero-shot vision-language models, with a 5-frame intent filter, visual
countdown signaling, and a live web UI — evaluated across cloud (Claude API),
server GPU (Gemma 4 26B), and mobile NPU (Gemma 4 E4B) inference approaches.

## Results summary

| Approach | Accuracy | Mean latency | NFR target met |
|---|---|---|---|
| Claude API | 99.4% | 2,681 ms | ✅ NFR-003 (≤3,000 ms) |
| Gemma 4 26B (EC2 g5.2xlarge) | 98.5% | 1,855 ms | ✅ NFR-001 (≤2,000 ms) |
| Gemma 4 E4B (mobile NPU) | 96.4% | 5,593 ms | ❌ NFR-002 (≤3,000 ms) |

All three approaches exceed the 90% accuracy target on a 337-frame held-out
synthetic test set. Gemma 4 26B is recommended for production deployment
(best balance of accuracy, latency, and privacy). Full methodology, error
analysis, and discussion are in the final report under `docs/final_report/`.

## Quick start

### 1. Camera permission
Grant your terminal/IDE camera access (macOS: System Settings > Privacy &
Security > Camera).

### 2. Install Ollama (for local Gemma 4 inference)
Download from [ollama.ai](https://ollama.ai), then pull a model:
```bash
ollama pull gemma4:26b
```

### 3. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 4. Run
```bash
# Local Gemma 4 (requires Ollama running)
ollama serve
python run.py --approach gemma4_server

# Cloud Claude API (no Ollama needed, internet required)
export ANTHROPIC_API_KEY=sk-ant-...
python run.py --approach claude_api
```

Recommended Ollama environment variables and per-toggle explanations are in
[`docs/OLLAMA_TOGGLE_GUIDE.md`](docs/OLLAMA_TOGGLE_GUIDE.md).

## Repository structure

```
.
├── run.py                    Entry point
├── requirements.txt
├── config/
│   └── config.yaml           All tunable settings (camera, sampling, intent filter, models)
├── src/
│   ├── output_schema.py      PostureResult dataclass (shared JSON schema)
│   ├── detector_base.py      PostureDetector ABC (SRS FR-011)
│   ├── intent_filter.py      Intent state machine: 5-frame rule + countdown + cooldown
│   ├── evaluation_logger.py  Per-frame CSV logging (SRS FR-028, FR-029)
│   ├── pipeline.py           Main orchestrator (FrameGrabber, camera lifecycle)
│   ├── ui_server.py          FastAPI WebSocket server
│   ├── camera_utils.py
│   └── detectors/
│       ├── gemma4_detector.py   Ollama + Gemma 4 (local/edge)
│       └── claude_detector.py   Anthropic Messages API (cloud)
├── static/
│   └── index.html            Web UI: live video, state machine, controls
├── scripts/
│   ├── analyse_results.py    Accuracy / F1 / confusion matrix / latency stats from CSV logs
│   └── collect_dataset.py    Organize captured frames into a labeled dataset
├── tests/
│   └── test_ollama.py        Pre-flight diagnostic for Ollama + Gemma 4 vision
├── data/
│   └── evaluation_logs/      Dev/test CSV logs for all three approaches + UAT logs (T2–T8)
└── docs/
    ├── proposal/              Capstone proposal
    ├── schedule/              Project schedule (PDF + Gantt chart component)
    ├── requirements/          SRS + requirements work packages (WP1.1, WP1.2)
    ├── architecture/          Literature review, system architecture, FTA, FMEA (WP2.1–WP3.0)
    ├── verification/          Data collection log, reliability test record, validation plan (WP4, WP5.0)
    ├── final_report/          Final capstone report + annotated bibliography
    └── OLLAMA_TOGGLE_GUIDE.md Explanation and test procedure for every Gemma 4 inference toggle
```

## Safety invariant (SRS NFR-011)

The desk control signal is **never emitted** without the user first receiving
a 10-second visual countdown with a cancel option. The `ADJUSTING` state is
reachable only from `SIGNALING` — there is no direct code path from
`MONITORING` to `ADJUSTING`. This is verified by reliability test T1 (code
inspection) and traced to fault-tree minimum cut set MCS-1
(`docs/architecture/WP2_3_Fault_Tree_Analysis.docx`).

## Scope

This project covers the **perception and intent-detection layers only**
(Change Request CR-01). Integration with motorized desk hardware is out of
scope and identified as future work — see the final report, Section 5.4.

## License

Not yet specified — add a `LICENSE` file before treating this as open source.
