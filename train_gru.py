import os
import json
import glob
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

DATA_DIR = "synthetic_data"
SEQ_LEN = 20
BATCH_SIZE = 64
EPOCHS = 25
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FEATURE_COLUMNS = [
    "density",
    "mean_speed",
    "speed_variance",
    "radial_spread",
    "density_gradient",
    "acceleration",
    "flux"
]

class ShockwaveDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class GRUClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size,
                          batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        return self.fc(out).squeeze(1)

def load_scenarios():
    all_sequences = []
    all_labels = []

    json_files = glob.glob(os.path.join(DATA_DIR, "*.json"))

    for file in json_files:
        with open(file, "r") as f:
            scenario = json.load(f)

        features = []
        labels = []

        for frame in scenario:
            row = [frame[col] for col in FEATURE_COLUMNS]
            features.append(row)
            labels.append(1 if frame["is_shockwave"] else 0)

        features = np.array(features)
        labels = np.array(labels)

        for i in range(len(features) - SEQ_LEN):
            window = features[i:i+SEQ_LEN]

            label = labels[i+SEQ_LEN-1]

            all_sequences.append(window)
            all_labels.append(label)

    return np.array(all_sequences), np.array(all_labels)

def main():

    print("Loading scenarios...")
    X, y = load_scenarios()

    print(f"Total sequences created: {len(X)}")
    print(f"Shockwave samples: {np.sum(y)}")

    print("Normalizing features...")

    scaler = StandardScaler()

    X_flat = X.reshape(-1, X.shape[-1])
    X_scaled = scaler.fit_transform(X_flat)
    X = X_scaled.reshape(X.shape)

    joblib.dump(scaler, "feature_scaler.pkl")
    print("Scaler saved as feature_scaler.pkl")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    train_dataset = ShockwaveDataset(X_train, y_train)
    val_dataset = ShockwaveDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=True)

    val_loader = DataLoader(val_dataset,
                            batch_size=BATCH_SIZE)

    model = GRUClassifier(input_size=len(FEATURE_COLUMNS)).to(DEVICE)

    pos_weight = torch.tensor(
        (len(y_train) - np.sum(y_train)) / np.sum(y_train)
    ).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print("Training...\n")

    for epoch in range(EPOCHS):

        model.train()
        total_loss = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                outputs = model(xb)
                preds = torch.sigmoid(outputs) > 0.5
                correct += (preds == yb.bool()).sum().item()
                total += yb.size(0)

        acc = correct / total if total > 0 else 0

        print(f"Epoch {epoch+1}/{EPOCHS} "
              f"Loss: {total_loss/len(train_loader):.4f} "
              f"Val Acc: {acc:.4f}")

    torch.save(model.state_dict(), "gru_shockwave_model.pth")
    print("\nModel saved as gru_shockwave_model.pth")

if __name__ == "__main__":
    main()
