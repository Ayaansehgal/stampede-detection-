import requests
import time
import random

SERVER_URL = "http://localhost:8000"

def force_shockwave():
    print("=== FORCING SHOCKWAVE ALERT ===")
    
    # 1. Send High-Risk AUDIO (Panic)
    #    We send it slightly in the future to cover the visual frames
    print("1. Injecting Audio Panic...")
    base_ts = time.time()
    
    try:
        requests.post(f"{SERVER_URL}/ingest_audio", json={
            "ts": base_ts,
            "audio_risk": 0.98  # Extremely high (Screaming)
        })
    except:
        print("Error sending audio (server might be down)")
        return

    # 2. Send Extreme VISUAL Data (Fill Buffer)
    #    We need 5 frames (since we lowered buffer size)
    print("2. Injecting Visual Chaos (5 frames)...")
    
    for i in range(6): # Send 6 just to be sure
        payload = {
            "ts": base_ts + (i * 0.1), # Increment time slightly
            "density": 0.95,           # Packed crowd
            "mean_speed": 5.0,         # Running fast
            "speed_variance": 3.0,     # Chaotic movement
            "radial_spread": 0.9,      # Scattering
            "density_gradient": 0.8,   # Uneven density
            "acceleration": 2.0,       # Sudden bursts
            "flux": 200.0              # High energy
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
