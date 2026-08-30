#!/usr/bin/env python3
"""
collect_dataset.py
Helps you organise frames captured via save_frames: true into a properly
named dataset folder ready for --images mode evaluation.

Usage:
    # After a recording session (save_frames: true in config.yaml):
    python collect_dataset.py label    logs/frames_20260618_120000/  sitting
    python collect_dataset.py label    logs/frames_20260618_121000/  standing
    python collect_dataset.py label    logs/frames_20260618_122000/  absent

    # Check dataset is ready for evaluation:
    python collect_dataset.py check    ./dataset/

    # Print a collection checklist:
    python collect_dataset.py checklist
"""
import sys, shutil, csv
from pathlib import Path


CLASSES   = ["sitting", "standing", "absent"]
LIGHTINGS = ["L1_daylight", "L2_artificial", "L3_dim", "L4_mixed", "L5_backlit"]
DISTANCES = ["near", "mid", "far"]
TARGET_TOTAL   = 270
TARGET_PER_CLASS = 90


# ── label command ────────────────────────────────────────────────────────────
def label(src_dir: str, posture: str) -> None:
    if posture not in CLASSES:
        print(f"ERROR: posture must be one of {CLASSES}")
        sys.exit(1)

    src = Path(src_dir)
    frames = sorted(src.glob("*.jpg")) + sorted(src.glob("*.png"))
    if not frames:
        print(f"No images found in {src_dir}")
        sys.exit(1)

    dst = Path("dataset") / posture
    dst.mkdir(parents=True, exist_ok=True)

    # Ask for condition codes to embed in filenames
    print(f"\nLabelling {len(frames)} frames as '{posture}'")
    print("Lighting: " + "  ".join(f"{i+1}={l}" for i, l in enumerate(LIGHTINGS)))
    lighting_idx = int(input("Lighting condition (1-5): ").strip()) - 1
    lighting = LIGHTINGS[lighting_idx]

    print("Distance: 1=near (0.8-1.2m)  2=mid (1.5-2.0m)  3=far (2.5-3.0m)")
    dist_idx  = int(input("Distance band (1-3): ").strip()) - 1
    distance  = DISTANCES[dist_idx]

    bg = input("Background (1=plain, 2=office, 3=high_contrast): ").strip()
    background = ["B1_plain", "B2_office", "B3_contrast"][int(bg) - 1]

    existing = len(list(dst.glob("*.jpg")))
    copied = 0
    for i, frame in enumerate(frames):
        n = existing + i + 1
        new_name = f"{posture}_{lighting}_{background}_{distance}_{n:04d}.jpg"
        dst_path = dst / new_name
        if not dst_path.exists():
            shutil.copy(frame, dst_path)
            copied += 1

    print(f"\nCopied {copied} frames -> dataset/{posture}/")
    print(f"Total in class: {existing + copied} / {TARGET_PER_CLASS} needed")
    print(f"  (Target: 90/class for development set + held-out test — CR-09)")


# ── check command ────────────────────────────────────────────────────────────
def check(dataset_dir: str) -> None:
    base = Path(dataset_dir)
    print(f"\nDataset readiness check: {dataset_dir}\n")

    all_files = []
    for cls in CLASSES:
        cls_dir = base / cls
        if not cls_dir.exists():
            files = []
        else:
            files = list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.png"))
        pct  = len(files) / TARGET_PER_CLASS * 100
        bar  = "#" * int(pct / 5) + "-" * (20 - int(pct / 5))
        status = "READY" if len(files) >= TARGET_PER_CLASS else f"need {TARGET_PER_CLASS - len(files)} more  (target: {TARGET_PER_CLASS})"
        print(f"  {cls:<10} [{bar}] {len(files):>4}/{TARGET_PER_CLASS}  {status}")
        all_files.extend(files)

    print(f"\n  Total: {len(all_files)} / {TARGET_TOTAL} frames")

    if len(all_files) >= TARGET_TOTAL:
        print("\n  Dataset is READY for evaluation.")
        print(f"  Run:  python run.py --images {dataset_dir}")
    else:
        print(f"\n  Dataset not yet complete ({TARGET_TOTAL - len(all_files)} frames remaining).")

    # Lighting / distance coverage summary
    print("\n  Coverage by condition:")
    lighting_counts: dict = {}
    for f in all_files:
        for L in LIGHTINGS:
            if L in f.stem:
                lighting_counts[L] = lighting_counts.get(L, 0) + 1
    for L in LIGHTINGS:
        n = lighting_counts.get(L, 0)
        print(f"    {L:<20} {n} frames")


# ── checklist command ────────────────────────────────────────────────────────
def checklist() -> None:
    print("""
DATASET COLLECTION CHECKLIST
═══════════════════════════════════════════════════════════════════
Before you start:
  [ ] config/config.yaml:  set  save_frames: true
  [ ] Create tape/marker on floor for capture zone
  [ ] Camera at 100-120 cm height, horizontal ±5 degrees
  [ ] Resolution 1280×720, 1 fps in config.yaml

For each session:
  1. Run:   python run.py --approach gemma4_server
  2. Open browser at http://localhost:8080
  3. Adopt posture and hold it for 30+ seconds per lighting/distance setup
  4. Ctrl+C to stop.  Frames saved in  logs/frames_TIMESTAMP/
  5. Run:   python collect_dataset.py label  logs/frames_TIMESTAMP/  <posture>

POSTURE RULES:
  sitting  Seated on chair, hips below shoulders, torso visible
  standing Upright, hips above knees, feet on floor
  absent   Nobody visible — point camera at empty desk

SESSION PLAN (3 classes × 45 condition combos × 2 frames = 270 total — CR-09):
  For each class (sitting / standing / absent), cover every combo of:
    5 lighting (L1-L5) × 3 backgrounds (B1-B3) × 3 distances (near/mid/far) = 45 combos
  Capture 2 frames per combo = 90 frames per class.

  Suggested session groupings (one session = one lighting × background pairing):
  ┌──────────────┬──────────────────┬────────────────┬───────────────────────┐
  │ Session      │ Lighting          │ Background     │ Combos (×3 dist each) │
  ├──────────────┼──────────────────┼────────────────┼───────────────────────┤
  │ S01          │ L1 daylight       │ B1 plain       │ 3 (near/mid/far)      │
  │ S02          │ L1 daylight       │ B2 office      │ 3                     │
  │ S03          │ L1 daylight       │ B3 contrast    │ 3                     │
  │ S04          │ L2 artificial     │ B1 plain       │ 3                     │
  │ S05          │ L2 artificial     │ B2 office      │ 3                     │
  │ S06          │ L2 artificial     │ B3 contrast    │ 3                     │
  │ S07          │ L3 dim            │ B1 plain       │ 3                     │
  │ S08          │ L3 dim            │ B2 office      │ 3                     │
  │ S09          │ L3 dim            │ B3 contrast    │ 3                     │
  │ S10          │ L4 mixed          │ B1 plain       │ 3                     │
  │ S11          │ L4 mixed          │ B2 office      │ 3                     │
  │ S12          │ L4 mixed          │ B3 contrast    │ 3                     │
  │ S13          │ L5 backlit        │ B1 plain       │ 3                     │
  │ S14          │ L5 backlit        │ B2 office      │ 3                     │
  │ S15          │ L5 backlit        │ B3 contrast    │ 3                     │
  └──────────────┴──────────────────┴────────────────┴───────────────────────┘
  Repeat all 15 sessions for each of the 3 classes = 45 sessions total.
  Capture 2 frames per combo per class: hold each posture for ~10 seconds,
  run.py saves 2 frames automatically at 1 fps with save_frames: true.

ABSENT SETUP:
  Just point the camera at an empty desk in the same position and lighting
  you use for the other classes. Walk out of frame. Wait 30 seconds.

TOTAL TIME ESTIMATE:
  45 sessions × ~2 min setup + 10s capture ≈ 90 minutes total
  (same wall-clock time, one-third the frames — dataset is correctly sized for zero-shot VLM)

After collecting:
  python collect_dataset.py check  ./dataset/
  python run.py --images ./dataset/test/  --approach gemma4_server
  python run.py --images ./dataset/test/  --approach gemma4_mobile
  python run.py --images ./dataset/test/  --approach claude_api
  python analyse_results.py  logs/detection_TIMESTAMP.csv
""")


# ── main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    cmd = sys.argv[1]
    if cmd == "label" and len(sys.argv) == 4:
        label(sys.argv[2], sys.argv[3])
    elif cmd == "check" and len(sys.argv) == 3:
        check(sys.argv[2])
    elif cmd == "checklist":
        checklist()
    else:
        print(__doc__)
        sys.exit(1)
