
import os
import sys
import time
import json
import random
import subprocess
import webbrowser
import requests

DEMO_DIR = "demo_data"
SERVER_URL = "http://localhost:8000"
FRAME_DELAY = 0.08

def ensure_demo_data():
    if not os.path.isdir(DEMO_DIR) or not os.path.isfile(os.path.join(DEMO_DIR, "demo_ramp.json")):
        print("Generating demo data...")
        try:
            from generate_demo_data import generate_all_demo_data
            generate_all_demo_data(DEMO_DIR)
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)

def wait_for_server(timeout=15):
    base = SERVER_URL.rstrip("/").replace("/ingest", "")
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(base, timeout=1)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False

def stream_demo_to_server():

    base = SERVER_URL.rstrip("/").replace("/ingest", "").replace("/ingest_audio", "")
    files = [
        os.path.join(DEMO_DIR, "demo_ramp.json"),
        os.path.join(DEMO_DIR, "demo_panic.json"),
    ]
    for path in files:
        if not os.path.isfile(path):
            continue
        with open(path, "r") as f:
            frames = json.load(f)
        name = os.path.basename(path)
        print(f"  Streaming {name} ({len(frames)} frames)...")
        for i, frame in enumerate(frames):
            gt = frame.get("label", "SAFE")
            audio = random.uniform(0.6, 0.9) if gt == "SHOCKWAVE" else random.uniform(0.0, 0.3)
            try:
                requests.post(f"{base}/ingest_audio", json={"ts": time.time(), "audio_risk": audio}, timeout=0.3)
            except Exception:
                pass
            payload = {**frame, "ts": time.time()}
            try:
                requests.post(f"{base}/ingest", json=payload, timeout=0.5)
            except Exception:
                pass
            time.sleep(FRAME_DELAY)
        print(f"  Done {name}.")
    print("Demo stream finished. Dashboard will show the last state. Keep the browser open.")

def main():
    print("=" * 56)
    print("  HACKATHON DEMO — Stampede Detection")
    print("=" * 56)
    ensure_demo_data()

    print("\n[1/3] Starting server (dashboard at http://localhost:8000)...")
    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000"],
        cwd=os.getcwd(),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if not wait_for_server():
        print("Server failed to start. Check if port 8000 is free.")
        server.terminate()
        sys.exit(1)
    print("      Server is up.")

    print("[2/3] Opening dashboard in browser...")
    try:
        webbrowser.open("http://localhost:8000")
    except Exception:
        print("      Open http://localhost:8000 in your browser.")
    time.sleep(1.5)

    print("[3/3] Streaming demo data (SAFE -> ELEVATED -> SHOCKWAVE)...")
    try:
        stream_demo_to_server()
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        print("\nServer is still running. Close the browser or press Ctrl+C in this window to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        server.terminate()
        print("Done.")

if __name__ == "__main__":
    main()
