import json
import time
import requests
import glob
import os
import random

SERVER_URL = "http://localhost:8000"

def stream_synthetic_data():
    print("=== Streaming Synthetic Data to Server ===")
    
    # Quick check if server is up
    try:
        requests.get(f"{SERVER_URL}/")
        print("Server is reachable.")
    except:
        print(f"Error: Server not running at {SERVER_URL}. Please run 'server/main.py' or 'run_all.py' first.")
        return

    # Find JSON files
    files = sorted(glob.glob("synthetic_data/scenario_*.json"))
    if not files:
        print("No synthetic data found in 'synthetic_data/'. Running simulation first...")
        import simulation
        sim = simulation.Simulation()
        sim.reset_scenario("panic")
        data = sim.run_scenario(200, "panic")
        with open("synthetic_data/scenario_test.json", "w") as f:
            json.dump(data, f)
        files = ["synthetic_data/scenario_test.json"]

    print(f"Found {len(files)} scenarios. Playing the first one...")
    
    with open(files[0], "r") as f:
        scenario_data = json.load(f)
        
    print(f"Loaded scenario with {len(scenario_data)} frames.")
    print("Press Ctrl+C to stop.\n")
    
    try:
        for i, frame in enumerate(scenario_data):
            # 1. Simulate Audio Risk based on Visual Label (for testing fusion)
            # If visual says SHOCKWAVE, make audio high too (with some noise)
            is_shockwave = frame.get("label") == "SHOCKWAVE"
            if is_shockwave:
                audio_risk = random.uniform(0.6, 0.95)
            else:
                audio_risk = random.uniform(0.0, 0.3)
                
            # Send Audio
            try:
                requests.post(f"{SERVER_URL}/ingest_audio", json={
                    "ts": time.time(),
                    "audio_risk": audio_risk
                }, timeout=0.1)
            except: pass

            # 2. Send Visual Features (GRU Input)
            payload = frame.copy()
            payload["ts"] = time.time()
            
            try:
                resp = requests.post(f"{SERVER_URL}/ingest", json=payload, timeout=0.1)
                data = resp.json()
                
                # feedback
                print(f"Frame {i:03d} | VisRisk: {frame['label']} ({frame.get('is_shockwave')}) -> Server: {data.get('label')} | Score: {data.get('risk_score'):.2f}")
            except Exception as e:
                print(f"Error making request: {e}")

            # Sleep to mimic real-time (0.1s per frame)
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nStopped.")

if __name__ == "__main__":
    stream_synthetic_data()
