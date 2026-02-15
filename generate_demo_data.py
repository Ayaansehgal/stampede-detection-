
import json
import os
import time
import random

DEMO_DIR = "demo_data"
FEATURE_COLS = [
    "density", "mean_speed", "speed_variance", "radial_spread",
    "density_gradient", "acceleration", "flux"
]

SAFE_RANGES = {
    "density": (0.05, 0.25),
    "mean_speed": (0.5, 1.2),
    "speed_variance": (0.05, 0.25),
    "radial_spread": (3.0, 6.0),
    "density_gradient": (1.0, 4.0),
    "acceleration": (0.3, 2.5),
    "flux": (15.0, 70.0),
}
ELEVATED_RANGES = {
    "density": (0.2, 0.5),
    "mean_speed": (1.2, 2.0),
    "speed_variance": (0.25, 0.55),
    "radial_spread": (6.0, 9.0),
    "density_gradient": (4.0, 6.5),
    "acceleration": (3.0, 8.0),
    "flux": (80.0, 140.0),
}

SHOCKWAVE_RANGES = {
    "density": (0.15, 0.35),
    "mean_speed": (2.0, 3.0),
    "speed_variance": (0.5, 1.2),
    "radial_spread": (9.0, 14.0),
    "density_gradient": (6.0, 10.0),
    "acceleration": (8.0, 26.0),
    "flux": (150.0, 220.0),
}

def _frame(label: str, frame_idx: int, base_ts: float, dt: float = 0.1, noise: float = 0.08):
    rng = SAFE_RANGES if label == "SAFE" else (ELEVATED_RANGES if label == "ELEVATED" else SHOCKWAVE_RANGES)
    row = {"ts": base_ts + frame_idx * dt, "frame": frame_idx, "label": label}
    for col in FEATURE_COLS:
        lo, hi = rng[col]
        val = random.uniform(lo, hi) * (1 + random.uniform(-noise, noise))
        row[col] = max(0.0, val)
    row["is_shockwave"] = label == "SHOCKWAVE"
    return row

def generate_ramp_scenario(num_safe=30, num_elevated=25, num_shockwave=45, base_ts=None):

    if base_ts is None:
        base_ts = time.time()
    frames = []
    idx = 0
    for _ in range(num_safe):
        frames.append(_frame("SAFE", idx, base_ts))
        idx += 1
    for _ in range(num_elevated):
        frames.append(_frame("ELEVATED", idx, base_ts))
        idx += 1
    for _ in range(num_shockwave):
        frames.append(_frame("SHOCKWAVE", idx, base_ts))
        idx += 1
    return frames

def generate_panic_scenario(num_frames=120, base_ts=None):

    if base_ts is None:
        base_ts = time.time()
    frames = []
    buffer = 25
    for i in range(num_frames):
        label = "SAFE" if i < buffer else "SHOCKWAVE"
        frames.append(_frame(label, i, base_ts))
    return frames

def generate_mixed_scenario(num_frames=150, base_ts=None):

    if base_ts is None:
        base_ts = time.time()
    frames = []
    block = 0
    for i in range(num_frames):
        if i < 25:
            label = "SAFE"
        elif i < 55:
            label = "SHOCKWAVE"
        elif i < 80:
            label = "SAFE"
        elif i < 120:
            label = "SHOCKWAVE"
        else:
            label = "SAFE"
        frames.append(_frame(label, i, base_ts))
    return frames

def generate_all_demo_data(out_dir=DEMO_DIR):
    os.makedirs(out_dir, exist_ok=True)
    base = time.time()

    ramp = generate_ramp_scenario(30, 25, 45, base)
    with open(os.path.join(out_dir, "demo_ramp.json"), "w") as f:
        json.dump(ramp, f, indent=0)
    print(f"  demo_ramp.json: {len(ramp)} frames (SAFE -> ELEVATED -> SHOCKWAVE)")

    panic = generate_panic_scenario(120, base + 100)
    with open(os.path.join(out_dir, "demo_panic.json"), "w") as f:
        json.dump(panic, f, indent=0)
    print(f"  demo_panic.json: {len(panic)} frames (short SAFE then SHOCKWAVE)")

    mixed = generate_mixed_scenario(150, base + 200)
    with open(os.path.join(out_dir, "demo_mixed.json"), "w") as f:
        json.dump(mixed, f, indent=0)
    print(f"  demo_mixed.json: {len(mixed)} frames (SAFE / SHOCKWAVE blocks)")

    return [
        os.path.join(out_dir, "demo_ramp.json"),
        os.path.join(out_dir, "demo_panic.json"),
        os.path.join(out_dir, "demo_mixed.json"),
    ]

if __name__ == "__main__":
    print("Generating demo data (values tuned for SHOCKWAVE classification)...")
    generate_all_demo_data()
    print("Done. Run: python run_pipeline_demo.py")
