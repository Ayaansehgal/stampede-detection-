
import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np
import requests

FEATURE_COLS = [
    "density", "mean_speed", "speed_variance", "radial_spread",
    "density_gradient", "acceleration", "flux"
]

LABEL_RANGES = {
    "SAFE": {
        "density": (0.05, 0.8),
        "mean_speed": (0.3, 1.2),
        "speed_variance": (0.05, 0.4),
        "radial_spread": (3.0, 6.0),
        "density_gradient": (1.0, 4.0),
        "acceleration": (0.5, 3.0),
        "flux": (10.0, 80.0),
    },
    "ELEVATED": {
        "density": (0.8, 2.0),
        "mean_speed": (1.2, 2.2),
        "speed_variance": (0.3, 0.8),
        "radial_spread": (5.0, 9.0),
        "density_gradient": (3.5, 6.0),
        "acceleration": (3.0, 8.0),
        "flux": (80.0, 180.0),
    },
    "SHOCKWAVE": {
        "density": (1.5, 4.0),
        "mean_speed": (2.0, 3.5),
        "speed_variance": (0.5, 1.2),
        "radial_spread": (7.0, 14.0),
        "density_gradient": (5.0, 10.0),
        "acceleration": (6.0, 25.0),
        "flux": (150.0, 250.0),
    },
}

def generate_frame_for_label(label: str, frame_idx: int, base_ts: float, dt: float = 0.1, noise: float = 0.1):

    rng = LABEL_RANGES.get(label, LABEL_RANGES["SAFE"])
    row = {"ts": base_ts + frame_idx * dt, "frame": frame_idx, "label": label}
    for col in FEATURE_COLS:
        lo, hi = rng[col]
        val = random.uniform(lo, hi) * (1 + random.uniform(-noise, noise))
        row[col] = max(0.0, val)
    row["is_shockwave"] = label == "SHOCKWAVE"
    return row

def generate_scenario_sequence(
    num_frames: int,
    pattern: str,
    base_ts: float = None,
    dt: float = 0.1,
) -> list:

    if base_ts is None:
        base_ts = time.time()
    out = []
    rng = random.Random()
    if pattern == "safe":
        for i in range(num_frames):
            out.append(generate_frame_for_label("SAFE", i, base_ts, dt, noise=0.02))
    elif pattern == "panic":
        for i in range(num_frames):
            out.append(generate_frame_for_label("SHOCKWAVE", i, base_ts, dt, noise=0.02))
    elif pattern == "elevated":
        for i in range(num_frames):
            out.append(generate_frame_for_label("ELEVATED", i, base_ts, dt, noise=0.02))
    elif pattern == "elevated_then_shockwave":

        t = num_frames // 3
        for i in range(num_frames):
            label = "ELEVATED" if i < t else "SHOCKWAVE"
            out.append(generate_frame_for_label(label, i, base_ts, dt, noise=0.02))
    elif pattern == "mixed":

        labels = ["SAFE", "ELEVATED", "SHOCKWAVE"]
        label = rng.choice(labels)
        for i in range(num_frames):
            if rng.random() < 0.05:
                label = rng.choice(labels)
            out.append(generate_frame_for_label(label, i, base_ts, dt, noise=0.02))
    else:
        raise ValueError(f"Unknown pattern: {pattern}")
    return out

def generate_scenarios_via_simulation(out_dir: str, num_safe: int = 3, num_panic: int = 3, steps: int = 200) -> list:

    try:
        import simulation
    except ImportError:
        return []
    os.makedirs(out_dir, exist_ok=True)
    sim = simulation.Simulation()
    paths = []
    for i in range(num_safe):
        sim.reset_scenario("safe")
        data = sim.run_scenario(steps, "safe")
        p = os.path.join(out_dir, f"sim_safe_{i:02d}.json")
        with open(p, "w") as f:
            json.dump(data, f, indent=0)
        paths.append(p)
    for i in range(num_panic):
        sim.reset_scenario("panic")
        data = sim.run_scenario(steps, "panic")
        p = os.path.join(out_dir, f"sim_panic_{i:02d}.json")
        with open(p, "w") as f:
            json.dump(data, f, indent=0)
        paths.append(p)
    return paths

def generate_all_synthetic_scenarios(
    out_dir: str,
    num_safe: int = 5,
    num_panic: int = 5,
    num_elevated: int = 3,
    num_mixed: int = 2,
    num_ramp: int = 2,
    frames_per_scenario: int = 100,
    use_simulation: bool = False,
) -> list:

    os.makedirs(out_dir, exist_ok=True)
    paths = []
    idx = 0
    for _ in range(num_safe):
        data = generate_scenario_sequence(frames_per_scenario, "safe")
        p = os.path.join(out_dir, f"scenario_{idx:04d}.json")
        with open(p, "w") as f:
            json.dump(data, f, indent=0)
        paths.append(p)
        idx += 1
    for _ in range(num_panic):
        data = generate_scenario_sequence(frames_per_scenario, "panic")
        p = os.path.join(out_dir, f"scenario_{idx:04d}.json")
        with open(p, "w") as f:
            json.dump(data, f, indent=0)
        paths.append(p)
        idx += 1
    for _ in range(num_elevated):
        data = generate_scenario_sequence(frames_per_scenario, "elevated")
        p = os.path.join(out_dir, f"scenario_{idx:04d}.json")
        with open(p, "w") as f:
            json.dump(data, f, indent=0)
        paths.append(p)
        idx += 1
    for _ in range(num_ramp):
        data = generate_scenario_sequence(frames_per_scenario, "elevated_then_shockwave")
        p = os.path.join(out_dir, f"scenario_{idx:04d}.json")
        with open(p, "w") as f:
            json.dump(data, f, indent=0)
        paths.append(p)
        idx += 1
    for _ in range(num_mixed):
        data = generate_scenario_sequence(frames_per_scenario, "mixed")
        p = os.path.join(out_dir, f"scenario_{idx:04d}.json")
        with open(p, "w") as f:
            json.dump(data, f, indent=0)
        paths.append(p)
        idx += 1
    if use_simulation:
        sim_paths = generate_scenarios_via_simulation(out_dir, num_safe=3, num_panic=3, steps=frames_per_scenario)
        paths.extend(sim_paths)
    return paths

def generate_synthetic_crowd_image(count: int, width: int = 640, height: int = 480, sigma: float = 12.0) -> np.ndarray:

    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (40, 45, 50)
    r = int(max(2, sigma))
    for _ in range(count):
        cx = random.randint(r, width - r - 1)
        cy = random.randint(r, height - r - 1)
        for dy in range(-r * 2, r * 2 + 1):
            for dx in range(-r * 2, r * 2 + 1):
                x, y = cx + dx, cy + dy
                if 0 <= x < width and 0 <= y < height:
                    d2 = dx * dx + dy * dy
                    intensity = max(0, 255 * np.exp(-d2 / (2 * sigma * sigma)))
                    for c in range(3):
                        img[y, x, c] = min(255, img[y, x, c] + int(intensity * 0.6))
    return img

def generate_synthetic_image_set(out_dir: str, counts: list, width: int = 640, height: int = 480) -> list:

    os.makedirs(out_dir, exist_ok=True)
    try:
        import cv2
    except ImportError:
        print("Warning: opencv-python not installed; skipping synthetic image generation.")
        return []
    paths = []
    for i, count in enumerate(counts):
        img = generate_synthetic_crowd_image(count, width, height)
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        p = os.path.join(out_dir, f"synthetic_crowd_{i:03d}_n{count}.png")
        cv2.imwrite(p, bgr)
        paths.append(p)
    return paths

def synthetic_audio_risk_for_label(label: str) -> float:

    if label == "SHOCKWAVE":
        return random.uniform(0.55, 0.95)
    if label == "ELEVATED":
        return random.uniform(0.25, 0.55)
    return random.uniform(0.0, 0.25)

def _scenario_files(scenario_dir: str, include_sim: bool = True):

    import glob
    files = sorted(glob.glob(os.path.join(scenario_dir, "scenario_*.json")))
    if include_sim:
        files += sorted(glob.glob(os.path.join(scenario_dir, "sim_*.json")))
    return files

def stream_scenario_and_collect(
    scenario_path: str,
    server_url: str,
    frame_delay: float = 0.05,
    use_synthetic_audio: bool = True,
) -> list:

    with open(scenario_path, "r") as f:
        frames = json.load(f)
    if not server_url.endswith("/"):
        server_url = server_url.rstrip("/")
    ingest_url = f"{server_url}/ingest"
    audio_url = f"{server_url}/ingest_audio"
    results = []
    for i, frame in enumerate(frames):
        gt_label = frame.get("label", "SAFE")
        if use_synthetic_audio:
            audio_risk = synthetic_audio_risk_for_label(gt_label)
            try:
                requests.post(audio_url, json={"ts": time.time(), "audio_risk": audio_risk}, timeout=0.2)
            except Exception:
                pass
        payload = {**frame, "ts": time.time()}
        try:
            resp = requests.post(ingest_url, json=payload, timeout=0.5)
            data = resp.json()
            pred_label = data.get("label", "UNKNOWN")
            risk = data.get("risk_score", 0.0)
            v_risk = data.get("visual_score", 0.0)
            a_risk = data.get("audio_risk", 0.0)
        except Exception as e:
            pred_label = "ERROR"
            risk = v_risk = a_risk = 0.0
        results.append((gt_label, pred_label, risk, v_risk, a_risk))
        time.sleep(frame_delay)
    return results

def run_demo_with_server(
    scenario_dir: str,
    server_url: str = "http://localhost:8000",
    max_scenarios: int = None,
    frame_delay: float = 0.05,
) -> dict:

    files = _scenario_files(scenario_dir)
    if max_scenarios is not None:
        files = files[:max_scenarios]
    if not files:
        print(f"No scenario_*.json found in {scenario_dir}. Run with --generate-only first.")
        return {}
    try:
        requests.get(server_url if not server_url.endswith("/ingest") else server_url.replace("/ingest", ""), timeout=2)
    except Exception as e:
        print(f"Server not reachable at {server_url}: {e}")
        print("Start server with: python server/main.py  or  python run_all.py")
        return {}
    all_results = []
    for path in files:
        results = stream_scenario_and_collect(path, server_url, frame_delay=frame_delay)
        all_results.extend(results)
    return compute_metrics(all_results)

def compute_metrics(results: list) -> dict:

    if not results:
        return {}
    gt_list = [r[0] for r in results]
    pred_list = [r[1] for r in results]
    labels = ["SAFE", "ELEVATED", "SHOCKWAVE"]

    gt_bin = [1 if g in ("ELEVATED", "SHOCKWAVE") else 0 for g in gt_list]
    pred_bin = [1 if p in ("ELEVATED", "SHOCKWAVE") else 0 for p in pred_list]
    acc_bin = sum(1 for a, b in zip(gt_bin, pred_bin) if a == b) / len(gt_bin)

    acc_multi = sum(1 for a, b in zip(gt_list, pred_list) if a == b) / len(gt_list)

    confusion = defaultdict(lambda: defaultdict(int))
    for g, p in zip(gt_list, pred_list):
        confusion[g][p] += 1

    prec = {}
    rec = {}
    for L in labels:
        tp = sum(1 for g, p in zip(gt_list, pred_list) if g == L and p == L)
        pred_L = sum(1 for p in pred_list if p == L)
        gt_L = sum(1 for g in gt_list if g == L)
        prec[L] = tp / pred_L if pred_L else 0.0
        rec[L] = tp / gt_L if gt_L else 0.0
    risk_scores = [r[2] for r in results]
    return {
        "num_frames": len(results),
        "accuracy_binary": acc_bin,
        "accuracy_multiclass": acc_multi,
        "precision": prec,
        "recall": rec,
        "confusion": {k: dict(v) for k, v in confusion.items()},
        "mean_risk_score": float(np.mean(risk_scores)),
        "results_sample": results[:5],
    }

def print_metrics(metrics: dict):

    if not metrics:
        return
    print("\n" + "=" * 60)
    print("  STAMPEDE DETECTION — DEMO METRICS")
    print("=" * 60)
    print(f"  Frames evaluated:  {metrics['num_frames']}")
    print(f"  Accuracy (binary): {metrics['accuracy_binary']:.2%}")
    print(f"  Accuracy (3-class): {metrics['accuracy_multiclass']:.2%}")
    print(f"  Mean risk score:   {metrics['mean_risk_score']:.3f}")
    print("\n  Per-label Precision / Recall:")
    for L in ("SAFE", "ELEVATED", "SHOCKWAVE"):
        p = metrics["precision"].get(L, 0)
        r = metrics["recall"].get(L, 0)
        print(f"    {L:12}  P={p:.2%}  R={r:.2%}")
    print("\n  Confusion (rows=GT, cols=Pred):")
    for gt in ("SAFE", "ELEVATED", "SHOCKWAVE"):
        row = metrics["confusion"].get(gt, {})
        cells = [str(row.get(p, 0)) for p in ("SAFE", "ELEVATED", "SHOCKWAVE")]
        print(f"    {gt:12}  " + "  ".join(cells))
    print("=" * 60 + "\n")

def run_offline_test(scenario_dir: str, max_scenarios: int = 5, window_size: int = 20) -> dict:

    try:
        from server.gru_inference import predict_risk, load_resources
        load_resources()
    except Exception as e:
        print(f"Offline test: could not load GRU: {e}")
        return {}
    files = _scenario_files(scenario_dir)[:max_scenarios]
    if not files:
        print(f"No scenario_*.json in {scenario_dir}. Run --generate-only first.")
        return {}
    results = []
    for path in files:
        with open(path, "r") as f:
            frames = json.load(f)
        for i in range(window_size, len(frames) + 1):
            window = frames[i - window_size:i]
            feat_arr = np.array([[f[c] for c in FEATURE_COLS] for f in window])
            gt_label = frames[i - 1]["label"]
            audio_risk = synthetic_audio_risk_for_label(gt_label)
            try:
                visual_risk, visual_label = predict_risk(feat_arr)
            except Exception:
                visual_risk, visual_label = 0.0, "SAFE"
            combined = 0.7 * visual_risk + 0.3 * audio_risk
            if visual_label == "BUFFERING" or visual_label == "MODEL_NOT_READY":
                pred_label = "BUFFERING"
            elif combined > 0.7:
                pred_label = "SHOCKWAVE"
            elif combined > 0.4:
                pred_label = "ELEVATED"
            else:
                pred_label = "SAFE"
            results.append((gt_label, pred_label, combined, visual_risk, audio_risk))
    return compute_metrics(results)

def test_csrnet_on_synthetic_images(image_dir: str, model_path: str = "models/model_csrnet.pth", max_images: int = 10) -> dict:

    import glob
    from PIL import Image
    images = sorted(glob.glob(os.path.join(image_dir, "*.png")))[:max_images]
    if not images:
        return {}
    try:
        from predict import CrowdCounter
        counter = CrowdCounter(model_path)
    except Exception as e:
        print(f"CSRNet test skipped: {e}")
        return {}
    out = []
    for path in images:

        base = os.path.basename(path)
        true_count = None
        if "_n" in base:
            try:
                true_count = int(base.split("_n")[1].replace(".png", ""))
            except ValueError:
                pass
        r = counter.predict(Image.open(path).convert("RGB"))
        out.append({"path": path, "pred_count": r["count"], "true_count": true_count})
    return {"predictions": out}

def main():
    parser = argparse.ArgumentParser(description="Stampede detection: synthetic data & demo")
    parser.add_argument("--generate-only", action="store_true", help="Only generate synthetic data")
    parser.add_argument("--out-dir", type=str, default="synthetic_data", help="Output dir for scenario JSONs")
    parser.add_argument("--demo", action="store_true", help="Stream to server and print metrics")
    parser.add_argument("--server", type=str, default="http://localhost:8000", help="Server URL for demo")
    parser.add_argument("--test-offline", action="store_true", help="Test GRU + fusion offline (no server)")
    parser.add_argument("--max-scenarios", type=int, default=None, help="Max scenarios to stream in demo")
    parser.add_argument("--frame-delay", type=float, default=0.05, help="Delay between frames when streaming (s)")
    parser.add_argument("--generate-images", action="store_true", help="Also generate synthetic crowd images")
    parser.add_argument("--images-dir", type=str, default="synthetic_data/images", help="Dir for synthetic images")
    parser.add_argument("--test-csrnet", action="store_true", help="Run CSRNet on synthetic images (after --generate-images)")
    parser.add_argument("--use-simulation", action="store_true", help="Also generate scenarios via simulation.py (sim_*.json)")
    parser.add_argument("--full-demo", action="store_true", help="Generate data, then run demo (requires server for demo part)")
    args = parser.parse_args()

    if args.full_demo:
        print("Full demo: generating synthetic data...")
        generate_all_synthetic_scenarios(args.out_dir, use_simulation=args.use_simulation)
        print("Starting demo (streaming to server)...")
        metrics = run_demo_with_server(args.out_dir, server_url=args.server, max_scenarios=args.max_scenarios, frame_delay=args.frame_delay)
        print_metrics(metrics)
        sys.exit(0 if (metrics and metrics.get("accuracy_binary", 0) >= 0.5) else 1)

    if args.generate_only or (not args.demo and not args.test_offline and not args.test_csrnet):

        paths = generate_all_synthetic_scenarios(args.out_dir, use_simulation=args.use_simulation)
        print(f"Generated {len(paths)} scenario JSONs in {args.out_dir}")
        if args.generate_images:
            img_paths = generate_synthetic_image_set(
                args.images_dir,
                counts=[5, 10, 20, 30, 50, 80, 120],
            )
            print(f"Generated {len(img_paths)} synthetic crowd images in {args.images_dir}")
        if not args.demo and not args.test_offline:
            print("\nNext: start server (python server/main.py) then run:")
            print("  python test_synthetic_demo.py --demo")
        return

    if not os.path.isdir(args.out_dir):
        print(f"Generating synthetic data in {args.out_dir}...")
        generate_all_synthetic_scenarios(args.out_dir)

    if args.demo:
        metrics = run_demo_with_server(
            args.out_dir,
            server_url=args.server,
            max_scenarios=args.max_scenarios,
            frame_delay=args.frame_delay,
        )
        print_metrics(metrics)
        if metrics:
            sys.exit(0 if metrics.get("accuracy_binary", 0) >= 0.5 else 1)

    if args.test_offline:
        metrics = run_offline_test(args.out_dir, max_scenarios=args.max_scenarios or 5)
        print_metrics(metrics)
        if metrics:
            sys.exit(0 if metrics.get("accuracy_binary", 0) >= 0.5 else 1)

    if args.test_csrnet and os.path.isdir(args.images_dir):
        res = test_csrnet_on_synthetic_images(args.images_dir, max_images=15)
        if res:
            print("\nCSRNet on synthetic images:")
            for p in res.get("predictions", [])[:10]:
                tc = p.get("true_count")
                tc_str = f" (true ~{tc})" if tc is not None else ""
                print(f"  {os.path.basename(p['path'])}: count={p['pred_count']}{tc_str}")

if __name__ == "__main__":
    main()
