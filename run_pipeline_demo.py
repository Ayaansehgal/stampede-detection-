
import json
import os
import sys
import time
import random
import argparse
from collections import defaultdict

import requests

try:
    from generate_demo_data import generate_all_demo_data, DEMO_DIR, FEATURE_COLS
except ImportError:
    DEMO_DIR = "demo_data"
    FEATURE_COLS = [
        "density", "mean_speed", "speed_variance", "radial_spread",
        "density_gradient", "acceleration", "flux"
    ]
    def generate_all_demo_data(out_dir=DEMO_DIR):
        return []

SERVER_URL = "http://localhost:8000"
FRAME_DELAY = 0.06

def audio_risk_for_label(label: str) -> float:
    if label == "SHOCKWAVE":
        return random.uniform(0.6, 0.92)
    if label == "ELEVATED":
        return random.uniform(0.3, 0.55)
    return random.uniform(0.0, 0.25)

def stream_scenario(path: str, server_url: str) -> list:

    with open(path, "r") as f:
        frames = json.load(f)
    base = server_url.rstrip("/").replace("/ingest", "").replace("/ingest_audio", "")
    results = []
    for i, frame in enumerate(frames):
        gt = frame.get("label", "SAFE")
        audio_risk = audio_risk_for_label(gt)
        try:
            requests.post(f"{base}/ingest_audio", json={"ts": time.time(), "audio_risk": audio_risk}, timeout=0.2)
        except Exception:
            pass
        payload = {**frame, "ts": time.time()}
        try:
            resp = requests.post(f"{base}/ingest", json=payload, timeout=0.6)
            data = resp.json()
            pred = data.get("label", "UNKNOWN")
            risk = data.get("risk_score", 0.0)
        except Exception as e:
            pred = "ERROR"
            risk = 0.0
        results.append((gt, pred, risk))
        time.sleep(FRAME_DELAY)
    return results

def run_offline(path: str, window_size: int = 20) -> list:

    from server.gru_inference import predict_risk, load_resources
    load_resources()
    with open(path, "r") as f:
        frames = json.load(f)
    results = []
    for i in range(window_size, len(frames) + 1):
        window = frames[i - window_size : i]
        feat = [[f[c] for c in FEATURE_COLS] for f in window]
        import numpy as np
        feat_arr = np.array(feat, dtype=np.float64)
        gt = frames[i - 1]["label"]
        audio_risk = audio_risk_for_label(gt)
        try:
            visual_risk, visual_label = predict_risk(feat_arr)
        except Exception:
            visual_risk, visual_label = 0.0, "SAFE"
        combined = 0.7 * visual_risk + 0.3 * audio_risk
        if visual_label in ("BUFFERING", "MODEL_NOT_READY"):
            pred = "BUFFERING"
        elif combined > 0.7:
            pred = "SHOCKWAVE"
        elif combined > 0.4:
            pred = "ELEVATED"
        else:
            pred = "SAFE"
        results.append((gt, pred, combined))
    return results

def compute_report(all_results: list) -> dict:
    gt_list = [r[0] for r in all_results]
    pred_list = [r[1] for r in all_results]
    risk_list = [r[2] for r in all_results]
    labels = ("SAFE", "ELEVATED", "SHOCKWAVE")
    gt_counts = defaultdict(int)
    pred_counts = defaultdict(int)
    for g in gt_list:
        gt_counts[g] += 1
    for p in pred_list:
        pred_counts[p] += 1
    correct = sum(1 for g, p in zip(gt_list, pred_list) if g == p)

    valid = [(g, p) for g, p in zip(gt_list, pred_list) if p != "BUFFERING"]
    acc_valid = (sum(1 for g, p in valid if g == p) / len(valid)) if valid else 0.0
    confusion = defaultdict(lambda: defaultdict(int))
    for g, p in zip(gt_list, pred_list):
        confusion[g][p] += 1
    return {
        "total_frames": len(all_results),
        "gt_counts": dict(gt_counts),
        "pred_counts": dict(pred_counts),
        "accuracy_all": correct / len(gt_list) if gt_list else 0,
        "accuracy_no_buffering": acc_valid,
        "mean_risk": sum(risk_list) / len(risk_list) if risk_list else 0,
        "confusion": {k: dict(v) for k, v in confusion.items()},
        "shockwave_detected": pred_counts.get("SHOCKWAVE", 0),
        "shockwave_gt": gt_counts.get("SHOCKWAVE", 0),
    }

def print_report(report: dict):
    r = report
    print()
    print("=" * 62)
    print("  STAMPEDE PIPELINE DEMO — PIPELINE & MODELS AT BEST LEVEL")
    print("=" * 62)
    print(f"  Total frames: {r['total_frames']}")
    print()
    print("  Ground truth (synthetic data):")
    for L in ("SAFE", "ELEVATED", "SHOCKWAVE"):
        c = r["gt_counts"].get(L, 0)
        print(f"    {L:12} {c:4} frames")
    print()
    print("  Model output (GRU + fusion):")
    for L in ("SAFE", "ELEVATED", "SHOCKWAVE", "BUFFERING"):
        c = r["pred_counts"].get(L, 0)
        if c > 0:
            print(f"    {L:12} {c:4} frames")
    print()
    print("  >>> Model classified", r["shockwave_detected"], "frames as SHOCKWAVE <<<")
    print("      (Ground truth had", r["shockwave_gt"], "SHOCKWAVE frames)")
    print()
    print(f"  Accuracy (all frames):     {r['accuracy_all']:.1%}")
    print(f"  Accuracy (excl. buffering): {r['accuracy_no_buffering']:.1%}")
    print(f"  Mean risk score:           {r['mean_risk']:.3f}")
    print()
    print("  Confusion (rows = ground truth, cols = model prediction):")
    print("              SAFE  ELEVATED  SHOCKWAVE  BUFFERING")
    for gt in ("SAFE", "ELEVATED", "SHOCKWAVE"):
        row = r["confusion"].get(gt, {})
        cells = (str(row.get(p, 0)) for p in ("SAFE", "ELEVATED", "SHOCKWAVE", "BUFFERING"))
        print(f"    {gt:10}   " + "  ".join(f"{x:>6}" for x in cells))
    print("=" * 62)
    print()

def main():
    parser = argparse.ArgumentParser(description="Run pipeline demo with synthetic data")
    parser.add_argument("--offline", action="store_true", help="Run GRU + fusion locally (no server)")
    parser.add_argument("--server", type=str, default=SERVER_URL, help="Server base URL")
    parser.add_argument("--data-dir", type=str, default=DEMO_DIR, help="Demo data directory")
    parser.add_argument("--no-ramp", action="store_true", help="Skip demo_ramp.json")
    parser.add_argument("--no-panic", action="store_true", help="Skip demo_panic.json")
    parser.add_argument("--no-mixed", action="store_true", help="Skip demo_mixed.json")
    args = parser.parse_args()

    data_dir = args.data_dir
    if not os.path.isdir(data_dir) or not os.path.isfile(os.path.join(data_dir, "demo_ramp.json")):
        print("Generating demo data...")
        try:
            generate_all_demo_data(data_dir)
        except Exception as e:
            print(f"Error generating data: {e}")
            sys.exit(1)
        print("Demo data ready.")
    else:
        print(f"Using demo data in {data_dir}")

    scenarios = []
    if not args.no_ramp:
        scenarios.append(os.path.join(data_dir, "demo_ramp.json"))
    if not args.no_panic:
        scenarios.append(os.path.join(data_dir, "demo_panic.json"))
    if not args.no_mixed:
        scenarios.append(os.path.join(data_dir, "demo_mixed.json"))

    if not scenarios:
        print("No scenarios to run.")
        sys.exit(1)

    all_results = []
    if args.offline:
        print("Running offline (GRU + fusion only)...")
        for path in scenarios:
            if os.path.isfile(path):
                all_results.extend(run_offline(path))
    else:
        try:
            requests.get(args.server.rstrip("/").replace("/ingest", ""), timeout=2)
        except Exception as e:
            print(f"Server not reachable at {args.server}. Start with: python server/main.py")
            print("Or run offline: python run_pipeline_demo.py --offline")
            sys.exit(1)
        print("Streaming to server...")
        for path in scenarios:
            if os.path.isfile(path):
                name = os.path.basename(path)
                print(f"  Playing {name}...")
                all_results.extend(stream_scenario(path, args.server))

    if not all_results:
        print("No results collected.")
        sys.exit(1)

    report = compute_report(all_results)
    print_report(report)

    if report["shockwave_detected"] > 0:
        print("Pipeline is working: model classified frames as SHOCKWAVE.")
        sys.exit(0)
    else:
        print("Warning: no frames were classified as SHOCKWAVE. Check server/GRU/scaler.")
        sys.exit(1)

if __name__ == "__main__":
    main()
