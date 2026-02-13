from fastapi import FastAPI
from .storage import store_event
from .gru_inference import predict_risk
from .blockchain import generate_hash, log_to_blockchain
from .state_manager import StateManager

app = FastAPI()
state_manager = StateManager()

@app.post("/ingest")
async def ingest(data: dict):

    file_path = store_event(data)

    feature_cols = ["density", "mean_speed", "speed_variance", "radial_spread", "density_gradient", "acceleration", "flux"]
    
    try:
        current_features = [float(data.get(c, 0.0)) for c in feature_cols]
    except ValueError:
        return {"status": "error", "message": "Invalid feature values"}

    state_manager.add_frame(current_features)
    
    feature_window = state_manager.get_buffer()
    
    if feature_window is not None:
        risk_score, label = predict_risk(feature_window)
    else:
        risk_score = 0.0
        label = "BUFFERING"

    if state_manager.should_log(label):

        event_payload = {
            "timestamp": data.get("ts"),
            "risk_score": risk_score,
            "label": label,
            "file_path": file_path
        }

        event_hash = generate_hash(event_payload)
        event_payload["hash"] = event_hash

        log_to_blockchain(event_payload)

    return {
        "risk_score": risk_score,
        "label": label,
        "status": "processed"
    }