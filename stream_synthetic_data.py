import json
import time
import requests
import glob
import os
import random
import argparse

SERVER_URL = "http://localhost:8000"
DATA_DIR = "synthetic_data"

def _find_scenario_files(data_dir=None):
    d = data_dir or DATA_DIR
    files = sorted(glob.glob(os.path.join(d, "scenario_*.json")))
    files += sorted(glob.glob(os.path.join(d, "sim_*.json")))
    return files

def _pick_panic_scenario(files):

    for path in files:
        try:
            with open(path, "r") as f:
                data = json.load(f)
            if any(isinstance(d, dict) and d.get("label") == "SHOCKWAVE" for d in (data if isinstance(data, list) else [data])):
                return path
        except Exception:
            continue
    return None

def _generate_panic_scenario():

    import simulation
    sim = simulation.Simulation()
    sim.reset_scenario("panic")
    return sim.run_scenario(200, "panic")

def stream_synthetic_data(scenario_index=0, panic=False, server_url=None, data_dir=None):
    server_url = server_url or SERVER_URL
    data_dir = data_dir or DATA_DIR
    print("=== Streaming Synthetic Data to Server ===")

    base_check = (server_url or "").rstrip("/").replace("/ingest", "").replace("/ingest_audio", "") or server_url
    try:
        requests.get(base_check, timeout=2)
        print("Server is reachable.")
    except Exception:
        print(f"Error: Server not running at {server_url}. Please run 'server/main.py' or 'run_all.py' first.")
        return

    scenario_data = None
    scenario_path = None

    if panic:
        os.makedirs(data_dir, exist_ok=True)
        files = _find_scenario_files(data_dir)
        scenario_path = _pick_panic_scenario(files)
        if scenario_path:
            with open(scenario_path, "r") as f:
                scenario_data = json.load(f)
            print(f"Playing SHOCKWAVE scenario: {scenario_path}")
        else:
            print("No existing panic scenario found. Generating panic scenario...")
            scenario_data = _generate_panic_scenario()
            scenario_path = os.path.join(data_dir, "scenario_panic_live.json")
            with open(scenario_path, "w") as f:
                json.dump(scenario_data, f)
            print("Generated and playing panic (SHOCKWAVE) scenario.")
    else:
        files = _find_scenario_files(data_dir)
        if not files:
            print("No synthetic data found. Generating one panic scenario as fallback...")
            os.makedirs(data_dir, exist_ok=True)
            scenario_data = _generate_panic_scenario()
            scenario_path = os.path.join(data_dir, "scenario_test.json")
            with open(scenario_path, "w") as f:
                json.dump(scenario_data, f)
        else:
            idx = min(scenario_index, len(files) - 1)
            scenario_path = files[idx]
            with open(scenario_path, "r") as f:
                scenario_data = json.load(f)
            print(f"Found {len(files)} scenarios. Playing #{idx}: {os.path.basename(scenario_path)}")

    if not scenario_data:
        print("No scenario data to play.")
        return
    base = server_url.rstrip("/")
    if base.endswith("/ingest"):
        base = base.replace("/ingest", "")
    print(f"Loaded scenario with {len(scenario_data)} frames.")
    print("Press Ctrl+C to stop.\n")

    try:
        for i, frame in enumerate(scenario_data):

            is_shockwave = frame.get("label") == "SHOCKWAVE"
            if is_shockwave:
                audio_risk = random.uniform(0.6, 0.95)
            else:
                audio_risk = random.uniform(0.0, 0.3)
            try:
                requests.post(f"{base}/ingest_audio", json={"ts": time.time(), "audio_risk": audio_risk}, timeout=0.2)
            except Exception:
                pass

            payload = frame.copy()
            payload["ts"] = time.time()
            try:
                resp = requests.post(f"{base}/ingest", json=payload, timeout=0.5)
                data = resp.json()

                print(f"Frame {i:03d} | VisRisk: {frame['label']} ({frame.get('is_shockwave')}) -> Server: {data.get('label')} | Score: {data.get('risk_score'):.2f}")
            except Exception as e:
                print(f"Error making request: {e}")

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nStopped.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stream synthetic visual + audio data to stampede server")
    parser.add_argument("--panic", "--shockwave", dest="panic", action="store_true", help="Run a SHOCKWAVE/panic scenario")
    parser.add_argument("--scenario", type=int, default=0, metavar="N", help="Scenario index to play (default: 0)")
    parser.add_argument("--server", type=str, default=SERVER_URL, help="Server base URL")
    parser.add_argument("--data-dir", type=str, default=DATA_DIR, help="Directory with scenario_*.json files")
    args = parser.parse_args()
    stream_synthetic_data(scenario_index=args.scenario, panic=args.panic, server_url=args.server, data_dir=args.data_dir)
