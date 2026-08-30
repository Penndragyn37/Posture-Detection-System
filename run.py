#!/usr/bin/env python3
"""
run.py  —  PDSDA entry point
Usage:
  python run.py                              # live camera
  python run.py --images ./test_images/      # run from a folder of images
  python run.py --images ./test_images/sitting_001.jpg  # single image
  python run.py --config path/to/config.yaml
  python run.py --approach claude_api        # override approach
  python run.py --port 8080
"""
import argparse
import sys
import threading
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import Pipeline
from src.ui_server import run_server


def parse_args():
    p = argparse.ArgumentParser(description="PDSDA — Posture Detection System")
    p.add_argument("--config",   default="config/config.yaml", help="Path to config YAML")
    p.add_argument("--approach", default=None,
                   choices=["gemma4_server", "gemma4_mobile", "claude_api"],
                   help="Override approach from config")
    p.add_argument("--port", type=int, default=None, help="Override UI port (default 8080)")
    p.add_argument("--no-browser", action="store_true", help="Do not auto-open browser")
    p.add_argument("--images", default=None, metavar="PATH",
                   help="Path to image file or folder; runs without a camera")
    p.add_argument("--loop",   action="store_true",
                   help="Loop through images continuously (default: stop at end)")
    p.add_argument("--camera", type=int, default=None, metavar="DEVICE_ID",
                   help="Override camera device ID from config (e.g. --camera 4)")
    return p.parse_args()


def main():
    args = parse_args()

    # Load pipeline
    print("[PDSDA] Loading configuration from", args.config)
    pipeline = Pipeline(config_path=args.config)

    # CLI overrides
    if args.approach:
        pipeline.config["approach"] = args.approach
        pipeline._approach = args.approach
        pipeline._load_detector()
        print(f"[PDSDA] Approach overridden to: {args.approach}")

    if args.camera is not None:
        pipeline.config.setdefault("camera", {})["device_id"] = args.camera
        print(f"[PDSDA] Camera overridden to device {args.camera}")

    port = args.port or pipeline.config.get("ui", {}).get("port", 8080)
    host = pipeline.config.get("ui", {}).get("host", "0.0.0.0")

    # Pre-flight checks (skip camera check when using image player)
    print("[PDSDA] Running pre-flight checks…")
    checks = pipeline.preflight()
    all_ok = True
    skip_camera = bool(args.images)
    for check, result in checks.items():
        if check.endswith("_error"):
            continue
        if skip_camera and check == "camera":
            print(f"  [SKIP] camera  (image player mode)")
            continue
        status = "OK" if result else "FAIL"
        print(f"  [{status}] {check}")
        if not result and check not in ("gpu", "disk_ok"):
            if check in ("camera", "detector", "pid_lock"):
                if not (skip_camera and check == "camera"):
                    all_ok = False

    if not all_ok:
        print("\n[PDSDA] Pre-flight failed. Check errors above and retry.")
        print("  Tip: If approach=claude_api, ensure ANTHROPIC_API_KEY is set in your environment.")
        print("  Tip: If approach=gemma4_server, ensure Ollama is running: ollama serve")
        sys.exit(1)

    print(f"\n[PDSDA] All checks passed. Starting pipeline (approach: {pipeline.approach})")
    print(f"[PDSDA] Evaluation log: {pipeline.log_path}")

    # Start pipeline (camera or image player)
    pipeline.start(image_source=args.images)
    if args.images:
        print(f"[PDSDA] Image player started ({pipeline.image_progress['total']} images).")
    else:
        print("[PDSDA] Camera pipeline started.")

    # Open browser
    if not args.no_browser:
        import webbrowser, time
        threading.Timer(1.5, lambda: webbrowser.open(f"http://localhost:{port}")).start()

    # Start UI server (blocking)
    print(f"[PDSDA] UI server starting at http://{host}:{port}")
    print("[PDSDA] Press Ctrl+C to stop.\n")

    try:
        run_server(pipeline, host=host, port=port)
    except KeyboardInterrupt:
        pass
    finally:
        print("\n[PDSDA] Shutting down…")
        pipeline.stop()
        print("[PDSDA] Done.")


if __name__ == "__main__":
    main()
