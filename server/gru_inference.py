import torch
import torch.nn as nn
import joblib
import numpy as np
import os

MODEL_PATH = "models/gru_shockwave_model.pth"
SCALER_PATH = "models/feature_scaler.pkl"
DEVICE = "cpu"
FEATURE_COLUMNS = [
    "density", "mean_speed", "speed_variance", "radial_spread",
    "density_gradient", "acceleration", "flux"
]

class GRUClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        return self.fc(out).squeeze(1)

model = None
scaler = None

def load_resources():
    global model, scaler
    if model is None:
        if os.path.exists(MODEL_PATH):
            model = GRUClassifier(input_size=len(FEATURE_COLUMNS)).to(DEVICE)
            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            model.eval()
            print("GRU Model Loaded")
        else:
            print(f"Warning: {MODEL_PATH} not found")
    
    if scaler is None:
        if os.path.exists(SCALER_PATH):
            scaler = joblib.load(SCALER_PATH)
            print("Scaler Loaded")
        else:
            print(f"Warning: {SCALER_PATH} not found")

def predict_risk(feature_sequence):
    """
    feature_sequence: numpy array of shape (20, 7)
    """
    load_resources()
    
    if model is None or scaler is None:
        return 0.0, "MODEL_NOT_READY"
        
    if feature_sequence.shape != (20, 7):
        return 0.0, "BUFFER_NOT_FULL"
    
    features_scaled = scaler.transform(feature_sequence)
    
    input_tensor = torch.tensor(features_scaled, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        output = model(input_tensor)
        risk_score = torch.sigmoid(output).item()
        
    label = "SHOCKWAVE" if risk_score > 0.75 else "SAFE"
    return risk_score, label