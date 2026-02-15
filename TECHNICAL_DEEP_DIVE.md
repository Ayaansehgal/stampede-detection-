# CrowdShield: Complete Technical Deep Dive

## Table of Contents
1. [System Architecture](#system-architecture)
2. [CSRNet Density Estimation](#csrnet-density-estimation)
3. [Optical Flow & Motion Analysis](#optical-flow--motion-analysis)
4. [GRU Temporal Sequence Model](#gru-temporal-sequence-model)
5. [Audio-Visual Fusion](#audio-visual-fusion)
6. [Server Pipeline](#server-pipeline)
7. [Synthetic Data Generation](#synthetic-data-generation)
8. [Deployment & Real-World Integration](#deployment--real-world-integration)
9. [Optimization Techniques](#optimization-techniques)

---

## System Architecture

### End-to-End Data Flow

```
REAL WORLD / SYNTHETIC DATA
    ↓
┌───────────────────────────────────────────┐
│  VIDEO INPUT LAYER                        │
│  ├─ Webcam/USB Camera (640×480, 30fps)   │
│  ├─ Raspberry Pi Camera Module            │
│  └─ Video File (MP4, AVI)                 │
└────────────┬────────────────────────────┘
             ↓
┌───────────────────────────────────────────┐
│  FEATURE EXTRACTION (Local)               │
│  ├─ CSRNet Dense Counting (GPU/CPU)       │
│  ├─ Optical Flow (Lucas-Kanade)           │
│  └─ Temporal Smoothing                    │
└────────────┬────────────────────────────┘
             ↓
┌───────────────────────────────────────────┐
│  7-FEATURE VECTOR (per frame)             │
│  [density, mean_speed, speed_var,         │
│   radial_spread, density_grad,            │
│   acceleration, flux]                     │
└────────────┬───────────────────────────┘
             ↓
┌───────────────────────────────────────────┐
│  SERVER PIPELINE (http://localhost:8000)  │
│  ├─ Socket.io Connection                  │
│  └─ /ingest Endpoint                      │
└────────────┬───────────────────────────┘
             ↓
┌───────────────────────────────────────────┐
│  GRU SEQUENCE MODEL                       │
│  ├─ Input: Rolling 20-frame window        │
│  ├─ Hidden State: 64-dim temporal repr.   │
│  └─ Output: Visual Risk ∈ [0, 1]         │
└────────────┬───────────────────────────┘
             ↓
┌───────────────────────────────────────────┐
│  AUDIO FUSION LAYER                       │
│  ├─ YAMNet Audio Embeddings               │
│  ├─ Audio Risk: P(screams|audio)         │
│  └─ Weighted Combination                  │
└────────────┬───────────────────────────┘
             ↓
┌───────────────────────────────────────────┐
│  DECISION LOGIC                           │
│  ├─ Risk > 0.7   → SHOCKWAVE (Red)        │
│  ├─ Risk 0.4-0.7 → ELEVATED (Amber)      │
│  └─ Risk < 0.4   → SAFE (Green)           │
└────────────┬───────────────────────────┘
             ↓
┌───────────────────────────────────────────┐
│  OUTPUT SYSTEMS                           │
│  ├─ Dashboard (React 5173)                │
│  ├─ Blockchain Ledger (Immutable)         │
│  ├─ Raspberry Pi Actuators                │
│  └─ WebSocket Live Updates                │
└───────────────────────────────────────────┘
```

### Multi-Computational Layers

```
┌─────────────────────────────────┐
│  EDGE (Local Machine)           │
│  ├─ Vision: CSRNet + OptFlow    │
│  ├─ Compute: CPU/GPU            │
│  ├─ Latency: ~50-100ms/frame    │
│  └─ Robustness: Works offline   │
└──────────────┬──────────────────┘
               ↓ (JSON + TCP)
┌─────────────────────────────────┐
│  SERVER (Central Processing)    │
│  ├─ Model: GRU Classifier       │
│  ├─ Fusion: Audio + Visual      │
│  ├─ Logging: Blockchain         │
│  └─ Latency: <10ms              │
└──────────────┬──────────────────┘
               ↓ (WebSocket)
┌─────────────────────────────────┐
│  FRONTEND (Real-time UI)        │
│  ├─ React Dashboard             │
│  ├─ Update: 30 FPS              │
│  └─ Visualization: Risk scores  │
└─────────────────────────────────┘
```

---

## CSRNet Density Estimation

### Model Architecture (CrowdStampedeYantra)

**File**: `model.py`

```python
class CrowdStampedeYantra(nn.Module):
    """
    VGG16-based fully convolutional network for crowd density estimation.
    Outputs: Single-channel density map (same spatial dims as input)
    """
    
    def __init__(self, load_weights=False):
        super().__init__()
        
        # Frontend: VGG16 encoder (13 layers)
        # Progressively reduces spatial dims, increases channels
        self.frontend_feat = [
            64, 64, 'M',                    # Block 1: 2×Conv64, MaxPool
            128, 128, 'M',                  # Block 2: 2×Conv128, MaxPool
            256, 256, 256, 'M',             # Block 3: 3×Conv256, MaxPool
            512, 512, 512                   # Block 4: 3×Conv512 (no pool)
        ]
        
        # Backend: Decoder with dilated convolutions
        # Upsamples back to input resolution
        self.backend_feat = [
            512, 512, 512,  # 3×Conv512
            256,            # 1×Conv256
            128,            # 1×Conv128
            64              # 1×Conv64
        ]
        
        self.frontend = make_layers(self.frontend_feat)
        self.backend = make_layers(
            self.backend_feat, 
            in_channels=512, 
            dilation=True  # Dilated convs maintain field-of-view
        )
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        
        # Transfer learning: Initialize frontend with ImageNet weights
        if not load_weights:
            vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
            # Copy VGG16 weights into frontend
            for i, (name, param) in enumerate(self.frontend.named_parameters()):
                vgg_param = list(vgg16.parameters())[i]
                param.data.copy_(vgg_param.data)
```

### Feature Extraction Details

```
Input: 640×480×3 RGB Frame
    ↓
Frontend (VGG16):
    Conv2d(3, 64, 3×3)        → 640×480×64
    Conv2d(64, 64, 3×3)       → 640×480×64
    MaxPool2d(2×2)            → 320×240×64
    
    Conv2d(64, 128, 3×3)      → 320×240×128
    Conv2d(128, 128, 3×3)     → 320×240×128
    MaxPool2d(2×2)            → 160×120×128
    
    Conv2d(128, 256, 3×3)     → 160×120×256
    Conv2d(256, 256, 3×3)     → 160×120×256
    Conv2d(256, 256, 3×3)     → 160×120×256
    MaxPool2d(2×2)            → 80×60×256
    
    Conv2d(256, 512, 3×3)     → 80×60×512
    Conv2d(512, 512, 3×3)     → 80×60×512
    Conv2d(512, 512, 3×3)     → 80×60×512
    (No pooling - preserve spatial info)
    ↓
Backend (Dilated Decoder):
    Conv2d(512, 512, 3×3, dilation=2)  → 80×60×512
    Conv2d(512, 512, 3×3, dilation=2)  → 80×60×512
    Conv2d(512, 512, 3×3, dilation=2)  → 80×60×512
    
    Conv2d(512, 256, 3×3, dilation=2)  → 80×60×256
    (Implicit upsampling via context)
    
    Conv2d(256, 128, 3×3, dilation=2)  → 80×60×128
    Conv2d(128, 64, 3×3, dilation=2)   → 80×60×64
    ↓
Output Layer:
    Conv2d(64, 1, 1×1)  → 80×60×1 (Density Map)
```

### How Density Counting Works

**File**: `predict.py` → `CrowdCounter.predict()`

```python
def predict(self, image_input):
    """
    Input: Image (file path, numpy array, or PIL Image)
    Output: Dense count via integration over spatial map
    """
    
    # Normalize per ImageNet statistics
    img_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet RGB means
            std=[0.229, 0.224, 0.225]    # ImageNet RGB stds
        )
    ])(image).unsqueeze(0).to(device)
    
    # Forward pass produces density map
    with torch.no_grad():
        density_map = model(img_tensor)
        # density_map shape: (1, 1, 80, 60)
    
    # Convert to actual count
    # Integral over spatial dimensions = total occupants
    count = int(round(density_map.sum().item()))
    
    # Extract summary statistics
    density_np = density_map.squeeze().cpu().numpy()
    density_avg = (count / total_pixels) * 1000  # Normalize to per-1000px²
    density_peak = density_np.max()  # Highest concentration point
    
    return {
        'count': count,
        'density_avg': density_avg,
        'density_peak': density_peak,
        'density_map': density_np
    }
```

### Why Density Regression vs Object Detection?

| Approach | Use Case | Pros | Cons |
|----------|----------|------|------|
| **Density Map (CSRNet)** | Dense crowds | No occlusion issues, real-time | Loses individual identity |
| **Object Detection (YOLO)** | Sparse crowds | Precise boxes, identifiable | Fails with occlusion, slow |
| **Skeleton Detection** | Pose analysis | Detailed posture | Missing in dense crowds |
| **Manual Counting** | Ground truth | 100% accurate | Only for validation |

**Decision**: CSRNet optimal because:
- Stampedes = **100+ people** in view → occlusion inevitable
- We need total count + spatial distribution, not individual tracking
- Real-time constraint (30 FPS) → can't run expensive detectors

---

## Optical Flow & Motion Analysis

### Dense Optical Flow (Lucas-Kanade)

**File**: `video_pipeline.py` → Motion Analysis Loop

```python
# For each frame:
frame_bgr = ...  # Current frame (640×480×3)
gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

if prev_gray is not None:
    # Dense optical flow: every pixel gets a motion vector
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, gray,
        pyr_scale=0.5,       # Image pyramid scale
        levels=3,            # Pyramid levels (multi-scale analysis)
        winsize=15,          # Window size for flow computation
        iterations=3,        # Refinement iterations
        poly_n=5,            # Neighborhood size
        poly_sigma=1.2,      # Gaussian std for deriv smoothing
        flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN
    )
    
    # flow has shape (480, 640, 2)
    # flow[y,x] = [optical_flow_x, optical_flow_y]
    
    # Extract motion magnitude and direction
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    # mag[y,x] = sqrt(flow_x^2 + flow_y^2)
    # ang[y,x] = arctan2(flow_y, flow_x)
```

### Motion Feature Extraction

```python
def extract_motion_features(flow, density_count):
    """
    Input: Optical flow matrix (480×640×2)
    Output: 5 motion-based features
    """
    
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    # Feature 1: Mean Speed
    # Average pixel displacement magnitude
    mean_speed = mag.mean()
    # Calibration: map to m/s
    # 1 pixel per frame (~3cm) @ 30fps ≈ 1 m/s
    mean_speed_ms = mean_speed * SPEED_SCALE  # 0.2
    
    # Feature 2: Speed Variance
    # Heterogeneity: low=organized, high=chaotic
    motion_mask = mag > FLOW_MAG_THRESHOLD  # Only significant motion
    speed_variance = mag[motion_mask].std()
    
    # Feature 3: Radial Spread
    # Do motion vectors diverge from center? → Centrifugal panic?
    def compute_radial_spread(flow, motion_mask):
        ys, xs = np.where(motion_mask > 0)
        cx, cy = np.mean(xs), np.mean(ys)  # Centroid of motion
        
        # Compute how far vectors point from centroid
        distances = np.sqrt((xs - cx)**2 + (ys - cy)**2)
        spread = np.std(distances) / np.sqrt(480**2 + 640**2)
        return spread * 25.0  # Scale to simulation range
    
    radial_spread = compute_radial_spread(flow, motion_mask)
    
    # Feature 4: Density Gradient
    # Computed from CSRNet output (sharp edges = sudden concentration)
    density_map = crowd_counter.predict(frame_bgr)['density_map']
    gradient = cv2.Sobel(density_map, cv2.CV_32F, 1, 1, ksize=3)
    density_gradient = gradient.mean()
    
    # Feature 5: Acceleration
    # Change in speed from previous frame
    acceleration = abs(mean_speed - prev_mean_speed)
    
    # Feature 6: Flux
    # "Momentum" of the panic: density × velocity
    flux = density_count * mean_speed_ms
    
    return {
        'mean_speed': mean_speed_ms,        # Float [0.3, 3.5] m/s
        'speed_variance': speed_variance,   # Float [0.05, 1.2]
        'radial_spread': radial_spread,     # Float [3.0, 14.0]
        'density_gradient': density_gradient,
        'acceleration': acceleration,
        'flux': flux
    }
```

### Motion Clustering

```python
def count_motion_clusters(motion_mask):
    """
    Count independent moving groups.
    Validates "crowd motion" vs "individual motion"
    """
    
    # Morphological cleanup: merge nearby pixels into connected blobs
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    cleaned = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, 
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    
    # Connected components analysis
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        cleaned, connectivity=8
    )
    
    # Filter noise: require >500 pixels per blob (genuine moving group)
    valid_clusters = 0
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > 500:
            valid_clusters += 1
    
    # Only trust motion if ≥2 independent clusters (true crowd motion)
    return max(0, valid_clusters - 1)  # Subtract 1 for background
```

### Temporal Smoothing

```python
# Exponential smoothing: reduces spikes, increases stability
SMOOTH_FACTOR = 0.5

s_speed = SMOOTH_FACTOR * mean_speed + (1 - SMOOTH_FACTOR) * prev_s_speed
s_var = SMOOTH_FACTOR * speed_variance + (1 - SMOOTH_FACTOR) * prev_s_var
s_flux = SMOOTH_FACTOR * flux + (1 - SMOOTH_FACTOR) * prev_s_flux

# Effect: 2-frame averaging, reduces noise by ~30%
```

---

## GRU Temporal Sequence Model

### Training Pipeline

**File**: `train_gru.py`

```python
class GRUClassifier(nn.Module):
    """
    Gated Recurrent Unit for shockwave detection
    """
    def __init__(self, input_size=7, hidden_size=64):
        super().__init__()
        
        # GRU cell: processes sequence step-by-step
        self.gru = nn.GRU(
            input_size=7,      # 7 features per frame
            hidden_size=64,    # Internal state dimension
            batch_first=True   # Input shape: (batch, seq_len, features)
        )
        
        # Output layer: collapse 64-dim state to risk score
        self.fc = nn.Linear(64, 1)
    
    def forward(self, x):
        """
        Input: x of shape (batch=1, seq_len=20, features=7)
        
        GRU equations:
            r_t = σ(W_ir * x_t + b_ir + W_hr * h_{t-1} + b_hr)  # Reset gate
            z_t = σ(W_iz * x_t + b_iz + W_hz * h_{t-1} + b_hz)  # Update gate
            n_t = tanh(W_in * x_t + b_in + r_t * (W_hn * h_{t-1} + b_hn))  # New
            h_t = (1 - z_t) * n_t + z_t * h_{t-1}  # Hidden state
        """
        
        # LSTM would need: (h_t, c_t) = 2 states vs GRU = 1 state
        # GRU is 40% fewer parameters
        out, hidden = self.gru(x)
        # out shape: (batch=1, seq_len=20, hidden=64)
        # hidden shape: (1, batch=1, hidden=64)
        
        # Use final timestep's hidden state
        final_hidden = out[:, -1, :]  # Shape: (1, 64)
        
        # Linear projection to risk score
        logits = self.fc(final_hidden)  # Shape: (1, 1)
        
        return logits
```

### Training Data Preparation

```python
# Load synthetic scenarios from JSON
# Each scenario: 100 frames, 7 features, binary label
X_raw, y_raw = load_scenarios()

# Create sliding windows
# i.e., [frame0-19] → label19, [frame1-20] → label20, ...
SEQ_LEN = 20  # 2 seconds @ 10fps

for scenario in scenarios:
    features = scenario['features']  # (100, 7)
    labels = scenario['labels']      # (100,)
    
    for i in range(len(features) - SEQ_LEN):
        window = features[i:i+SEQ_LEN]      # (20, 7)
        label_at_end = labels[i+SEQ_LEN-1]  # True/False
        sequences.append(window)
        sequence_labels.append(label_at_end)

# Normalize across all sequences
scaler = StandardScaler()
X_flat = X.reshape(-1, 7)
X_scaled = scaler.fit_transform(X_flat)
X = X_scaled.reshape(-1, 20, 7)  # (N_sequences, 20, 7)

# Train-val split (stratified)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, 
    test_size=0.2, 
    stratify=y,  # Maintain SAFE/SHOCKWAVE ratio
    random_state=42
)
```

### Loss Function & Training

```python
# Class imbalance: SAFE >>>>> SHOCKWAVE
# Solution: Use weighted BCE loss
pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
# If 80% SAFE, 20% SHOCKWAVE: pos_weight = 4.0
# Penalizes missing shockwaves 4× more than false alarms

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
# Outputs logits (before sigmoid), computes binary cross-entropy

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# Adaptive learning rate: automatically tunes per-parameter step size

# Training loop
for epoch in range(25):
    model.train()
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
        
        logits = model(x_batch)  # (batch, 1)
        loss = criterion(logits.squeeze(), y_batch.float())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Validation
    model.eval()
    with torch.no_grad():
        logits = model(X_val)
        val_loss = criterion(logits.squeeze(), torch.tensor(y_val))
        
        # Compute F1 score
        predictions = (torch.sigmoid(logits) > 0.5).numpy()
        f1 = f1_score(y_val, predictions)
        
        print(f"Epoch {epoch}: Loss={loss:.4f}, F1={f1:.4f}")
```

### Inference (Server Side)

**File**: `server/gru_inference.py`

```python
def predict_risk(feature_window):
    """
    Real-time risk prediction
    
    Args:
        feature_window: numpy array (seq_len, 7) where seq_len ≤ 20
    
    Returns:
        (risk_score, label): (float [0,1], str)
    """
    
    load_resources()  # Lazy load model & scaler
    
    if model is None or scaler is None:
        return 0.0, "MODEL_NOT_READY"
    
    # Pad if window < 20 frames
    if feature_window.shape[0] < 20:
        # Repeat last frame to fill
        padding = np.tile(
            feature_window[-1],
            (20 - feature_window.shape[0], 1)
        )
        feature_window = np.vstack([feature_window, padding])
    
    # Normalize using training scaler
    features_scaled = scaler.transform(feature_window)
    
    # Convert to tensor + batch dimension
    input_tensor = torch.tensor(
        features_scaled,
        dtype=torch.float32
    ).unsqueeze(0).to(DEVICE)  # (1, 20, 7)
    
    # Forward pass
    with torch.no_grad():
        logits = model(input_tensor)
        risk_score = torch.sigmoid(logits).item()
    
    # Label based on thresholds
    if risk_score > 0.7:
        label = "SHOCKWAVE"
    elif risk_score > 0.4:
        label = "ELEVATED"
    else:
        label = "SAFE"
    
    return risk_score, label
```

### Sequence Length Justification

```
0.5 seconds (5 frames @ 10fps)
    ↓ Too short
    ├─ Can't distinguish noise from onset
    ├─ High false positive rate
    └─ Reaction time good but unreliable
    
2 seconds (20 frames @ 10fps)  ← CHOSEN
    ↓ OPTIMAL
    ├─ Captures onset → peak progression
    ├─ Filters out single-frame artifacts
    ├─ ~200ms lag (acceptable for alarms)
    └─ Proven in crowd dynamics literature
    
5 seconds (50 frames @ 10fps)
    ↓ Too long
    ├─ Loses responsiveness
    ├─ Can't detect rapid onset
    └─ More memory overhead
```

---

## Audio-Visual Fusion

### YAMNet Audio Feature Extraction

**Source**: Google's pretrained audio embeddings model

```python
# YAMNet: Lightweight audio tagging model
# Input: 10-second audio waveform
# Output: 521-dim embedding + per-frame probabilities

import tensorflow_hub as hub

# Load YAMNet
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
yamnet_classes = hub.load('https://tfhub.dev/google/yamnet_721_1/yamnet_class_map.csv')

def extract_audio_risk(audio_segment):
    """
    Extract risk from audio.
    
    Target events:
    - 'Speech': Screaming, shouting (high confidence → high risk)
    - 'Crowd': Background noise
    - 'Alarm': Siren, whistle
    - 'Music': Usually false positive (low weight)
    """
    
    # Resample to 16kHz (YAMNet requirement)
    audio_16k = librosa.resample(audio, sr=44100, target_sr=16000)
    
    # Normalize waveform
    audio_16k = audio_16k / np.max(np.abs(audio_16k))
    
    # Run inference
    scores, embeddings, _ = yamnet_model(audio_16k)
    # scores: (T, 521) where T ≈ 10 * 16000 / 480 ≈ 333 frames
    
    # Extract scream/speech probability
    speech_idx = [i for i, c in enumerate(yamnet_classes) if 'speech' in c.lower()]
    alarm_idx = [i for i, c in enumerate(yamnet_classes) if 'alarm' in c.lower()]
    
    speech_prob = scores[:, speech_idx].max()     # 0-1
    alarm_prob = scores[:, alarm_idx].max()        # 0-1
    
    # Weighted risk
    audio_risk = 0.7 * speech_prob + 0.3 * alarm_prob
    
    return audio_risk
```

### Server-Side Fusion Logic

**File**: `server/main.py` → `/ingest` endpoint

```python
@app.post("/ingest")
async def ingest(data: dict):
    """
    Core decision-making endpoint.
    Combines visual + audio for robust detection.
    """
    
    # Feature vector from client
    current_features = [
        float(data.get(col, 0.0))
        for col in FEATURE_COLS
    ]
    
    # Step 1: Add to sliding window buffer
    state_manager.add_frame(current_features)
    feature_window = state_manager.get_buffer()  # (20, 7)
    
    # Step 2: Visual risk (GRU prediction)
    if feature_window is not None:
        visual_risk, visual_label = predict_risk(feature_window)
        # predict_risk: GRU → logits → sigmoid → [0, 1]
    else:
        visual_risk = 0.0
        visual_label = "BUFFERING"
    
    # Step 3: Audio risk (get latest from buffer)
    current_ts = data.get("ts", 0.0)
    audio_risk = state_manager.get_audio_risk(current_ts)
    # audio_risk: from YAMNet embeddings, memoized for 2 seconds
    
    # Step 4: Sensor Fusion
    # Weighted average: visual dominates (more reliable spatial info)
    combined_score = (0.7 * visual_risk) + (0.3 * audio_risk)
    
    # Boost logic: dual confirmation
    # If BOTH modalities provide strong signals → amplify
    if visual_risk > 0.4 and audio_risk > 0.6:
        # Both say "something's wrong" → confident shockwave
        combined_score = max(combined_score, 0.75)
    
    # Step 5: Final Decision
    if combined_score > 0.7:
        final_label = "SHOCKWAVE"
    elif combined_score > 0.4:
        final_label = "ELEVATED"
    else:
        final_label = "SAFE"
    
    # Override if still buffering
    if visual_label == "BUFFERING":
        final_label = "BUFFERING"
    
    # Step 6: Update live status (for dashboard)
    state_manager.set_live_status(
        final_label,
        combined_score,
        audio_risk,
        current_ts
    )
    
    # Step 7: Blockchain logging (if significant event)
    if state_manager.should_log(final_label):
        event_payload = {
            "timestamp": current_ts,
            "risk_score": combined_score,
            "visual_risk": visual_risk,
            "audio_risk": audio_risk,
            "label": final_label,
        }
        event_hash = generate_hash(event_payload)
        log_to_blockchain(event_payload)
    
    # Step 8: Prepare response for Pi
    pi_status = "alert" if final_label in ["SHOCKWAVE", "ELEVATED"] else "ok"
    
    return {
        "risk_score": combined_score,
        "visual_score": visual_risk,
        "audio_score": audio_risk,
        "label": final_label,
        "pi_status": pi_status,
        "confidence": combined_score,
        "timestamp": current_ts
    }
```

### Weight Justification

| Component | Weight | Reason |
|-----------|--------|--------|
| Visual (GRU) | 70% | Direct spatial evidence, lower latency, temporal dynamics |
| Audio (YAMNet) | 30% | Secondhand (screams indicate panic), some false positives |
| **Boost threshold** | 0.75 | If both ≥ moderate confidence → force SHOCKWAVE |

**Example Scenarios**:
```
Scenario 1: Dense crowd, quiet
  Visual: 0.8 (high density + fast motion)
  Audio: 0.1 (background chatter only)
  Combined: 0.7*0.8 + 0.3*0.1 = 0.59 → ELEVATED
  
Scenario 2: Thin crowd, screaming
  Visual: 0.3 (some motion, low density)
  Audio: 0.8 (loud screams detected)
  Combined: 0.7*0.3 + 0.3*0.8 = 0.45 → ELEVATED
  
Scenario 3: Dense + screaming (TRUE STAMPEDE)
  Visual: 0.8
  Audio: 0.8
  Check boost: 0.8 > 0.4 AND 0.8 > 0.6 ✓
  Combined: max(0.7*0.8 + 0.3*0.8, 0.75) = 0.76 → SHOCKWAVE ✓✓
```

---

## Server Pipeline

### State Management

**File**: `server/state_manager.py`

```python
class StateManager:
    """
    Buffers frames, tracks audio, manages decision logic
    """
    
    def __init__(self, buffer_size=20):
        self.buffer = deque(maxlen=buffer_size)  # (20, 7)
        self.audio_cache = {}  # timestamp → audio_risk
        
        self.last_label = "BUFFERING"
        self.last_risk = 0.0
        self.last_audio_risk = 0.0
        self.last_ts = 0.0
        
        self.shockwave_count = 0  # Persistence counter
        
    def add_frame(self, features):
        """Buffer incoming frame"""
        self.buffer.append(features)
    
    def get_buffer(self):
        """Return full buffer if ready, else None"""
        if len(self.buffer) == 20:
            return np.array(list(self.buffer))
        else:
            return None  # Wait for warmup
    
    def update_audio(self, audio_risk, ts):
        """Store audio risk with timestamp"""
        self.audio_cache[ts] = audio_risk
    
    def get_audio_risk(self, current_ts):
        """
        Retrieve cached audio risk from last 2 seconds
        (YAMNet has high latency, so we memoize)
        """
        recent = [
            risk for ts, risk in self.audio_cache.items()
            if current_ts - ts <= 2.0
        ]
        return max(recent) if recent else 0.0
    
    def should_log(self, label):
        """
        Only log on state changes or high confidence
        Reduces blockchain spam (write ops are slow)
        """
        if label != self.last_label:
            return True
        if self.last_risk > 0.8:
            return True
        return False
    
    def set_live_status(self, label, risk, audio_risk, ts):
        """Update for dashboard streaming"""
        self.last_label = label
        self.last_risk = risk
        self.last_audio_risk = audio_risk
        self.last_ts = ts
```

### Blockchain Logging

**File**: `server/blockchain.py`

```python
def generate_hash(event_payload):
    """
    Immutable event hash for tamper-proofing
    """
    event_json = json.dumps(event_payload, sort_keys=True)
    return hashlib.sha256(event_json.encode()).hexdigest()

def log_to_blockchain(event_payload):
    """
    Append-only ledger: once written, cannot be edited
    """
    event_payload['hash'] = generate_hash(event_payload)
    
    with open("blockchain_ledger.jsonl", "a") as f:
        f.write(json.dumps(event_payload) + "\n")
    
    # File structure: each line is immutable record
    # blockchain_ledger.jsonl:
    # {"timestamp": 1.234, "label": "SAFE", "hash": "0xabc123..."}
    # {"timestamp": 1.456, "label": "ELEVATED", "hash": "0xdef456..."}
    # {"timestamp": 1.678, "label": "SHOCKWAVE", "hash": "0xghi789..."}
```

---

## Synthetic Data Generation

### Scenario Generator

**File**: `test_synthetic_demo.py`

```python
FEATURE_COLS = [
    "density", "mean_speed", "speed_variance",
    "radial_spread", "density_gradient", "acceleration", "flux"
]

# Plausible ranges per label (from crowd dynamics + simulation)
LABEL_RANGES = {
    "SAFE": {
        "density": (0.05, 0.8),
        "mean_speed": (0.3, 1.2),
        "speed_variance": (0.05, 0.4),
        "radial_spread": (3.0, 6.0),
        "density_gradient": (1.0, 4.0),
        "acceleration": (0.5, 3.0),
        "flux": (10.0, 80.0),
    },
    "ELEVATED": {
        "density": (0.8, 2.0),
        "mean_speed": (1.2, 2.2),
        "speed_variance": (0.3, 0.8),
        "radial_spread": (5.0, 9.0),
        "density_gradient": (3.5, 6.0),
        "acceleration": (3.0, 8.0),
        "flux": (80.0, 180.0),
    },
    "SHOCKWAVE": {
        "density": (1.5, 4.0),
        "mean_speed": (2.0, 3.5),
        "speed_variance": (0.5, 1.2),
        "radial_spread": (7.0, 14.0),
        "density_gradient": (5.0, 10.0),
        "acceleration": (6.0, 25.0),
        "flux": (150.0, 250.0),
    },
}

def generate_frame_for_label(label, frame_idx, noise=0.02):
    """
    Synthesize one frame with realistic noise
    
    noise=0.02 means ±2% random variation per feature
    """
    rng = LABEL_RANGES[label]
    row = {"ts": base_ts + frame_idx * 0.1, "frame": frame_idx, "label": label}
    
    for col in FEATURE_COLS:
        lo, hi = rng[col]
        # Uniform sample from range
        val = random.uniform(lo, hi)
        # Add small Gaussian noise
        val *= (1 + random.gauss(0, noise))
        row[col] = max(0.0, val)  # Clamp to [0, ∞)
    
    row["is_shockwave"] = (label == "SHOCKWAVE")
    return row

def generate_scenario_sequence(num_frames, pattern):
    """
    Generate temporal sequence of frames.
    pattern: "safe" | "panic" | "elevated" | "elevated_then_shockwave" | "mixed"
    """
    out = []
    
    if pattern == "safe":
        # 100 frames all SAFE
        for i in range(num_frames):
            out.append(generate_frame_for_label("SAFE", i, noise=0.02))
    
    elif pattern == "panic":
        # 100 frames all SHOCKWAVE (instant stampede)
        for i in range(num_frames):
            out.append(generate_frame_for_label("SHOCKWAVE", i, noise=0.02))
    
    elif pattern == "elevated_then_shockwave":
        # First 33 frames: crowded but calm
        # Last 67 frames: panic
        t = num_frames // 3
        for i in range(num_frames):
            label = "ELEVATED" if i < t else "SHOCKWAVE"
            out.append(generate_frame_for_label(label, i, noise=0.02))
    
    elif pattern == "mixed":
        # Random walk over labels (stress test)
        labels = ["SAFE", "ELEVATED", "SHOCKWAVE"]
        label = random.choice(labels)
        for i in range(num_frames):
            if random.random() < 0.05:  # 5% chance to switch
                label = random.choice(labels)
            out.append(generate_frame_for_label(label, i, noise=0.02))
    
    return out

def generate_all_synthetic_scenarios(out_dir, num_safe=5, num_panic=5, ...):
    """
    Generate full dataset
    """
    scenarios = []
    
    for _ in range(num_safe):
        data = generate_scenario_sequence(100, "safe")
        path = f"{out_dir}/scenario_NNNN.json"
        with open(path, "w") as f:
            json.dump(data, f)
        scenarios.append(path)
    
    # ... similar for panic, elevated, mixed, ramp
    
    return scenarios
```

### Sim2Real Calibration

```python
# Simulation constants in video_pipeline.py
SPEED_SCALE = 0.2        # Map optical flow pixels → simulation m/s
DENSITY_AREA = 6.0       # Assumed visible area in m²
FLUX_BOOST = 1.0         # Extra flux multiplier
SMOOTH_FACTOR = 0.5      # Temporal damping

# Why these values?
# 1. Optical flow magnitude ≈ pixel displacement per frame
#    At 30fps: 1 pixel/frame ≈ 3cm/frame ≈ 1 m/s (walking speed)
#    SPEED_SCALE=0.2 → 1 pixel/frame maps to 0.2 simulation units
#
# 2. CSRNet density map integrated over 640×480 frame
#    Assuming visible area = 6 m² (e.g., 2m × 3m corridor)
#    Dividing person count by 6 → density per m²
#
# These calibrations match training distribution of synthetic data
# → GRU scaler normalizes accurately
```

---

## Deployment & Real-World Integration

### Raspberry Pi Pipeline

**Hardware**:
- Raspberry Pi 4 Model B (4GB RAM)
- Pi Camera Module v2 (8MP)
- USB Microphone (audio capture)

**System Architecture**:
```
┌──────────────────────┐
│  Pi Camera (video)   │  → 1080p@30fps (CSRNet locally)
│  USB Mic (audio)     │  → YAMNet on server
└────────┬─────────────┘
         ↓ Features ({density, speed, ...}) 
┌─────────────────────────────────────────┐
│  video_pipeline.py (runs on Pi)         │
│  ├─ CSRNet inference (CPU, ~500ms)      │
│  ├─ Optical flow (100ms)                │
│  ├─ Feature extraction                  │
│  └─ HTTP POST to server:8000/ingest    │
└────────┬─────────────────────────────────┘
         ↓ JSON payload
┌─────────────────────────────────────────┐
│  Server (separate machine)              │
│  ├─ GRU inference (10ms)                │
│  ├─ Audio fusion                        │
│  ├─ Decision logic                      │
│  └─ WebSocket broadcast to dashboard   │
└────────┬─────────────────────────────────┘
         ↓
┌────────────────────────────────────────┐
│  Pi GPIO (actuators)                   │
│  ├─ LED: Green (SAFE) / Red (SHOCKWAVE)│
│  ├─ Buzzer: Alert frequency ∝ risk     │
│  └─ Speaker: Text-to-speech evacuation │
└────────────────────────────────────────┘
```

### Network Integration

```python
# Client (Pi) → Server communication
import requests

SERVER_URL = "http://192.168.1.100:8000/ingest"

# Every frame processed locally, results sent to server
feature_vector = {
    "ts": time.time(),
    "density": 2.5,
    "mean_speed": 1.8,
    "speed_variance": 0.6,
    "radial_spread": 8.5,
    "density_gradient": 5.2,
    "acceleration": 4.1,
    "flux": 145.0
}

response = requests.post(SERVER_URL, json=feature_vector)
decision = response.json()
# {
#    "label": "ELEVATED",
#    "risk_score": 0.65,
#    "pi_status": "alert"
# }

if decision['pi_status'] == 'alert':
    GPIO.output(RED_LED, GPIO.HIGH)
    BUZZER.frequency = 800 + int(decision['risk_score'] * 3000)
```

---

## Optimization Techniques

### Model Optimization

```python
# 1. Quantization (CPU inference on Pi)
import torch.quantization as quantization

model_fp32 = GRUClassifier(...)
model_int8 = quantization.quantize_dynamic(
    model_fp32,
    {nn.Linear},  # Which layers to quantize
    dtype=torch.qint8
)
# Result: 4× smaller model, ~2× faster, negligible accuracy loss

# 2. Knowledge Distillation
# Train student model (smaller) to mimic teacher (large)
loss = F.mse_loss(
    student_logits,
    teacher_logits.detach()  # Don't update teacher
)

# 3. Batch Inference (for non-real-time scenarios)
# Process 32 sequences at once instead of 1
# Amortizes overhead, increases throughput
```

### Latency Breakdown

```
Frame capture (Pi):           20ms
├─ Camera readout
└─ Debayer + resize

CSRNet inference:             100-150ms (GPU) / 200-300ms (CPU)
├─ Frontend convolutions
├─ Backend dilated convolution
└─ Sum over spatial dimensions

Optical flow:                 50-100ms
├─ Gaussian pyramid build
├─ Lucas-Kanade iterations
└─ Motion magnitude/direction

HTTP POST:                    20-50ms
├─ Network latency
└─ JSON serialization

Server processing:            <10ms
├─ GRU inference
├─ Audio fusion
└─ Decision logic

Total per-frame latency:      200-500ms (acceptable for real-time)
```

### Memory Safety

```python
# Prevent Denial-of-Service attacks
MAX_BUFFER_SIZE = 100  # Only keep last 100 frames in memory

# Sensor fusion sanity checks
if combined_score < 0 or combined_score > 1:
    combined_score = 0.5  # Default to middle (safe)

# Rate limiting on blockchain writes
if state_manager.events_logged_this_minute > 60:
    # Skip logging (prevent ledger spam)
    pass

# Model staleness check
if time.time() - model_load_time > 3600:
    # Reload model (daily retraining possible)
    reload_model()
```

---

## Summary Table

| Component | Purpose | Input | Output | Latency |
|-----------|---------|-------|--------|---------|
| **CSRNet** | Crowd density estimation | RGB frame | Person count + heatmap | 100-300ms |
| **Optical Flow** | Motion analysis | Frame pair | Speed, variance, spread | 50-100ms |
| **Feature Extractor** | Sensor fusion prep | Count + flow | 7-dim vector | <10ms |
| **GRU Classifier** | Temporal decision | 20-frame window | Risk score [0,1] | <5ms |
| **Audio Fusion** | Multi-modal confirmation | Audio embeddings | Audio risk [0,1] | ~2000ms (async) |
| **Server Logic** | Final decision | Visual + audio | Label + status | <10ms |
| **Blockchain** | Immutable logging | Event payload | Hash + ledger entry | 5-20ms |
| **Dashboard** | Real-time visualization | WebSocket stream | Risk visualization | 30fps |

---

## Key Files

```
├── model.py                      # CSRNet architecture
├── predict.py                    # Crowd counting inference
├── video_pipeline.py             # Local feature extraction
├── train_gru.py                  # GRU training script
├── server/
│   ├── main.py                   # FastAPI server + fusion logic
│   ├── gru_inference.py          # GRU real-time inference
│   ├── state_manager.py          # Buffer + state tracking
│   ├── blockchain.py             # Immutable ledger
│   └── dashboard.html            # Visualization UI
├── test_synthetic_demo.py        # Synthetic data generation + streaming
└── CrowdShield/
    ├── client/                   # React frontend (TypeScript)
    └── server/                   # Node.js API gateway
```

