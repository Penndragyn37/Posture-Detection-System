#!/usr/bin/env python3
"""
analyse_results.py
Computes classification accuracy, per-class precision/recall/F1, confusion
matrix, and latency statistics from a PDSDA evaluation CSV log.

Usage:
    # Analyse a single run log:
    python analyse_results.py logs/detection_20260618_040913.csv

    # Compare two approaches side by side:
    python analyse_results.py logs/gemma4_run.csv logs/claude_run.csv

    # Show only frames with ground truth (image-player mode):
    python analyse_results.py logs/detection_20260618.csv --eval-only

Output: terminal report + optional results_summary.csv
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

CLASSES = ["sitting", "standing", "absent"]


# ─────────────────────────────────────────────────────────────────────────────
# Core metrics
# ─────────────────────────────────────────────────────────────────────────────

def load_csv(path: str, eval_only: bool = False) -> list[dict]:
    """Load rows from an evaluation CSV. If eval_only, skip rows with no ground truth."""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if eval_only and not row.get("ground_truth"):
                continue
            rows.append(row)
    return rows


def confusion_matrix(rows: list[dict]) -> dict:
    """
    Returns a nested dict: cm[actual][predicted] = count.
    Only includes rows that have a ground_truth label.
    """
    cm = {c: {p: 0 for p in CLASSES} for c in CLASSES}
    for row in rows:
        gt  = row.get("ground_truth", "").strip().lower()
        pred = row.get("predicted", "").strip().lower()
        if gt in CLASSES and pred in CLASSES:
            cm[gt][pred] += 1
    return cm


def class_metrics(cm: dict) -> dict:
    """Compute per-class precision, recall, F1 from the confusion matrix."""
    metrics = {}
    for cls in CLASSES:
        tp = cm[cls][cls]
        fp = sum(cm[other][cls] for other in CLASSES if other != cls)
        fn = sum(cm[cls][other] for other in CLASSES if other != cls)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)
        support   = sum(cm[cls].values())

        metrics[cls] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(precision, 4),
            "recall":    round(recall,    4),
            "f1":        round(f1,        4),
            "support":   support,
        }
    return metrics


def overall_accuracy(cm: dict) -> float:
    correct = sum(cm[c][c] for c in CLASSES)
    total   = sum(cm[r][p] for r in CLASSES for p in CLASSES)
    return round(correct / total, 4) if total > 0 else 0.0


def macro_f1(metrics: dict) -> float:
    scores = [metrics[c]["f1"] for c in CLASSES if metrics[c]["support"] > 0]
    return round(sum(scores) / len(scores), 4) if scores else 0.0


def latency_stats(rows: list[dict]) -> dict:
    """Latency stats across all rows that have a valid latency_ms value."""
    lats = []
    for row in rows:
        try:
            v = float(row.get("latency_ms", "") or "")
            if v > 0:
                lats.append(v)
        except (ValueError, TypeError):
            pass
    if not lats:
        return {"n": 0}
    lats_sorted = sorted(lats)
    n = len(lats_sorted)
    return {
        "n":    n,
        "mean": round(sum(lats) / n),
        "p50":  round(lats_sorted[n // 2]),
        "p95":  round(lats_sorted[int(n * 0.95)]),
        "max":  round(max(lats)),
        "min":  round(min(lats)),
    }


def approach_name(rows: list[dict]) -> str:
    """Best-guess approach label from the first non-empty approach field."""
    for row in rows:
        a = row.get("approach", "").strip()
        if a:
            return a
    return "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# Misclassification breakdown
# ─────────────────────────────────────────────────────────────────────────────

def misclassifications(rows: list[dict]) -> list[dict]:
    """Return rows where ground truth ≠ predicted (both present)."""
    errors = []
    for row in rows:
        gt   = row.get("ground_truth", "").strip().lower()
        pred = row.get("predicted",    "").strip().lower()
        if gt in CLASSES and pred in CLASSES and gt != pred:
            errors.append(row)
    return errors


def error_pattern_summary(errors: list[dict]) -> dict:
    """Group misclassifications by (actual → predicted) pair."""
    patterns = defaultdict(list)
    for row in errors:
        key = f"{row['ground_truth']} → {row['predicted']}"
        patterns[key].append(row.get("image_name") or row.get("frame_id", ""))
    return dict(patterns)


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def _bar(value: float, width: int = 20) -> str:
    filled = int(round(value * width))
    return "█" * filled + "░" * (width - filled)


def print_report(log_path: str, rows: list[dict], eval_rows: list[dict]) -> dict:
    """Print a formatted report to stdout. Returns summary dict."""
    approach = approach_name(rows)
    cm   = confusion_matrix(eval_rows)
    mets = class_metrics(cm)
    acc  = overall_accuracy(cm)
    mf1  = macro_f1(mets)
    lat  = latency_stats(rows)
    errs = misclassifications(eval_rows)

    total_eval = sum(sum(cm[r].values()) for r in CLASSES)

    w = 65
    print("\n" + "═" * w)
    print(f"  PDSDA Results — {Path(log_path).name}")
    print(f"  Approach: {approach}")
    print("═" * w)

    # ── Accuracy ──────────────────────────────────────────────────────
    target_met = acc >= 0.90
    print(f"\n  Overall accuracy:  {acc:.1%}  {_bar(acc)}  {'✓ target met' if target_met else '✗ below 90% target'}")
    print(f"  Macro F1:          {mf1:.4f}")
    print(f"  Evaluated frames:  {total_eval}  ({len(rows)} total in log)")

    # ── Per-class metrics ─────────────────────────────────────────────
    print(f"\n  {'Class':<12} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Supp':>6}  {'≥0.88?':>7}")
    print("  " + "─" * 50)
    for cls in CLASSES:
        m = mets[cls]
        ok = "✓" if m["f1"] >= 0.88 else "✗"
        print(f"  {cls:<12} {m['precision']:>6.3f} {m['recall']:>6.3f} {m['f1']:>6.3f} {m['support']:>6}  {ok:>7}")

    # ── Confusion matrix ──────────────────────────────────────────────
    print(f"\n  Confusion matrix (rows = actual, cols = predicted):")
    header = f"  {'actual \\ pred':<14}" + "".join(f"{c:>10}" for c in CLASSES)
    print(header)
    print("  " + "─" * (14 + 10 * len(CLASSES)))
    for actual in CLASSES:
        row_str = f"  {actual:<14}"
        for pred in CLASSES:
            n = cm[actual][pred]
            mark = f"[{n}]" if actual == pred else f" {n} "
            row_str += f"{mark:>10}"
        print(row_str)

    # ── Latency ───────────────────────────────────────────────────────
    if lat["n"] > 0:
        print(f"\n  Latency (ms)  mean={lat['mean']}  p50={lat['p50']}  p95={lat['p95']}  max={lat['max']}  n={lat['n']}")
        nfr001_target = 2000
        met = lat["mean"] <= nfr001_target
        print(f"  SRS NFR-001 mean ≤ {nfr001_target}ms:  {'✓' if met else '✗'}")

    # ── Misclassifications ────────────────────────────────────────────
    if errs:
        patterns = error_pattern_summary(errs)
        print(f"\n  Misclassifications: {len(errs)} / {total_eval} ({len(errs)/total_eval:.1%})")
        for pattern, files in sorted(patterns.items(), key=lambda x: -len(x[1])):
            print(f"    {pattern:<28}  ×{len(files)}")
            # Show up to 3 example filenames for the most common error
            for fname in files[:3]:
                print(f"      {fname}")
            if len(files) > 3:
                print(f"      … and {len(files)-3} more")
    else:
        print(f"\n  No misclassifications in {total_eval} evaluated frames. ✓")

    print("\n" + "═" * w + "\n")

    return {
        "log":          log_path,
        "approach":     approach,
        "accuracy":     acc,
        "macro_f1":     mf1,
        "target_met":   target_met,
        "total_frames": len(rows),
        "eval_frames":  total_eval,
        "errors":       len(errs),
        "latency_mean": lat.get("mean"),
        "latency_p95":  lat.get("p95"),
        **{f"{cls}_f1": mets[cls]["f1"] for cls in CLASSES},
        **{f"{cls}_precision": mets[cls]["precision"] for cls in CLASSES},
        **{f"{cls}_recall":    mets[cls]["recall"]    for cls in CLASSES},
    }


def write_summary_csv(summaries: list[dict], out_path: str = "results_summary.csv") -> None:
    if not summaries:
        return
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)
    print(f"  Summary written to {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyse PDSDA evaluation CSV logs — accuracy, F1, confusion matrix, latency."
    )
    parser.add_argument("logs", nargs="+", metavar="LOG_CSV",
                        help="One or more detection_*.csv files from logs/")
    parser.add_argument("--eval-only", action="store_true",
                        help="Only score frames that have a ground_truth label (image-player mode)")
    parser.add_argument("--save", action="store_true",
                        help="Write results_summary.csv with one row per log file")
    args = parser.parse_args()

    summaries = []
    for log_path in args.logs:
        if not Path(log_path).exists():
            print(f"ERROR: file not found: {log_path}", file=sys.stderr)
            continue

        all_rows  = load_csv(log_path, eval_only=False)
        eval_rows = load_csv(log_path, eval_only=True)

        if not eval_rows:
            print(f"\nWARNING: {log_path} has no rows with ground_truth labels.")
            print("  Run with  --images ./dataset/test/  to get ground truth from filenames.\n")
            continue

        summary = print_report(log_path, all_rows, eval_rows)
        summaries.append(summary)

    if args.save and summaries:
        write_summary_csv(summaries)

    if len(summaries) > 1:
        # Side-by-side comparison row
        print("COMPARISON SUMMARY")
        print(f"  {'Approach':<22} {'Accuracy':>9} {'Macro F1':>9} {'Mean lat':>9} {'Errors':>7}")
        print("  " + "─" * 60)
        for s in summaries:
            lat = f"{s['latency_mean']}ms" if s.get("latency_mean") else "—"
            print(f"  {s['approach']:<22} {s['accuracy']:>8.1%} {s['macro_f1']:>9.4f} {lat:>9} {s['errors']:>7}")
        print()


if __name__ == "__main__":
    main()
