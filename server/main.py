from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import json
import os
from .storage import store_event
from .gru_inference import predict_risk
from .blockchain import generate_hash, log_to_blockchain
from .state_manager import StateManager

app = FastAPI()

# Allow CORS for local testing if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

state_manager = StateManager()

@app.get("/", response_class=HTMLResponse)
async def read_dashboard():
    with open("server/dashboard.html", "r") as f:
        return f.read()

@app.get("/events")
async def get_events():
    events = []
    if os.path.exists("blockchain_ledger.jsonl"):
        with open("blockchain_ledger.jsonl", "r") as f:
            for line in f:
                if line.strip():
                    try:
                        events.append(json.loads(line))
                    except:
                        pass
    # Return last 50 events
    return events[-50:]

@app.post("/ingest_audio")
async def ingest_audio(data: dict):
    ts = data.get("ts", 0.0)
    risk = data.get("audio_risk", 0.0)
    state_manager.update_audio(risk, ts)
    return {"status": "ok"}

@app.post("/ingest")
async def ingest(data: dict):

    # file_path = store_event(data) # Moved to end

    feature_cols = ["density", "mean_speed", "speed_variance", "radial_spread", "density_gradient", "acceleration", "flux"]
    
    try:
        current_features = [float(data.get(c, 0.0)) for c in feature_cols]
    except ValueError:
        return {"status": "error", "message": "Invalid feature values"}

    state_manager.add_frame(current_features)
    
    feature_window = state_manager.get_buffer()
    
    # 1. Visual/GRU Risk
    if feature_window is not None:
        visual_risk, visual_label = predict_risk(feature_window)
    else:
        visual_risk = 0.0
        visual_label = "BUFFERING"

    # 2. Audio Risk
    current_ts = data.get("ts", 0.0)
    audio_risk = state_manager.get_audio_risk(current_ts)
    
    # 3. Sensor Fusion Logic
    # Weighted average: 70% Visual (more reliable for spatial), 30% Audio (confirmatory)
    combined_score = (0.7 * visual_risk) + (0.3 * audio_risk)
    
    # Boost: If BOTH provide strong signals, amplify the risk
    if visual_risk > 0.4 and audio_risk > 0.6:
        combined_score = max(combined_score, 0.75) # Force into SHOCKWAVE territory
    
    # Final Decision
    if combined_score > 0.7:
        final_label = "SHOCKWAVE"
    elif combined_score > 0.4:
        final_label = "ELEVATED"
    else:
        final_label = "SAFE"
        
    # Override if buffering
    if visual_label == "BUFFERING":
        final_label = "BUFFERING"
        # We KEEP the calc score for debugging, but set label to BUFFERING
        # combined_score = 0.0  <-- Removed this so we can see the potential score in logs

    # Perform Logging
    if state_manager.should_log(final_label):
        event_payload = {
            "timestamp": data.get("ts"),
            "risk_score": combined_score,
            "visual_risk": visual_risk,
            "audio_risk": audio_risk,
            "label": final_label,
            # "file_path": file_path # We'll get this in a sec
        }
        event_hash = generate_hash(event_payload)
        event_payload["hash"] = event_hash
        log_to_blockchain(event_payload)

    # Prepare Response Data
    response_data = {
        "risk_score": combined_score,
        "visual_score": visual_risk,
        "audio_score": audio_risk,
        "label": final_label,
        "status": "processed"
    }
    
    # Store complete event (Input + Output)
    full_log = data.copy()
    full_log.update(response_data)
    store_event(full_log)

    return response_data

