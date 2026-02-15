import requests
import time
import random

SERVER_URL = "http://localhost:8000"

def force_shockwave():
    print("=== FORCING SHOCKWAVE ALERT ===")

    print("1. Injecting Audio Panic...")
    base_ts = time.time()

    try:
        requests.post(f"{SERVER_URL}/ingest_audio", json={
            "ts": base_ts,
            "audio_risk": 0.98
        })
    except:
        print("Error sending audio (server might be down)")
        return

    print("2. Injecting Visual Chaos (5 frames)...")

    for i in range(6):
        payload = {
            "ts": base_ts + (i * 0.1),
            "density": 0.95,
            "mean_speed": 5.0,
            "speed_variance": 3.0,
            "radial_spread": 0.9,
            "density_gradient": 0.8,
            "acceleration": 2.0,
            "flux": 200.0
        }

        try:
            resp = requests.post(f"{SERVER_URL}/ingest", json=payload)
            data = resp.json()
            print(f"   Frame {i+1}: Label={data.get('label')} | Risk={data.get('risk_score'):.2f}")

            if data.get("label") == "SHOCKWAVE":
                print("\n>>> SUCCESS: SHOCKWAVE TRIGGERED! <<<")
                print(f"Final Response: {data}")
                break

        except Exception as e:
            print(f"Error: {e}")

        time.sleep(0.1)

if __name__ == "__main__":
    force_shockwave()
