# Hackathon Demo — How to Show It Working

## Quick run (one command)

From the project folder:

```bash
python run_hackathon_demo.py
```

This will:

1. Start the server
2. Open **http://localhost:8000** in your browser (dashboard)
3. Stream synthetic crowd data so the dashboard shows **SAFE → ELEVATED → SHOCKWAVE** live

**Put the browser tab on the big screen / projector.** Judges will see the status change and the risk meter go up.

---

## Step-by-step (if you prefer to run things separately)

### Option A: Synthetic data demo (no webcam needed)

1. **Terminal 1 — start server**
   ```bash
   python server/main.py
   ```
   Or: `python -m uvicorn server.main:app --host 0.0.0.0 --port 8000`

2. **Browser**  
   Open **http://localhost:8000** and put it on the big screen.

3. **Terminal 2 — stream demo data**
   ```bash
   python stream_synthetic_data.py --panic
   ```
   Or use the full demo data:
   ```bash
   python run_pipeline_demo.py
   ```
   The dashboard will update: **SAFE** (green) → then **SHOCKWAVE** (red) with risk score going up.

### Option B: Live webcam demo

1. **Terminal 1 — start server**
   ```bash
   python server/main.py
   ```

2. **Browser**  
   Open **http://localhost:8000** (dashboard on big screen).

3. **Terminal 2 — run video pipeline (webcam)**
   ```bash
   python video_pipeline.py
   ```
   A window will open with the camera feed and overlay: **Count**, **Speed**, **Risk [SAFE/ELEVATED/SHOCKWAVE]**.  
   Move around or have 2–3 people walk quickly to push the risk up so judges see **SHOCKWAVE** or **ELEVATED**.

---

## What to say (talking points)

- **“This is a stampede detection system for crowded spaces.”**
- **“We use a multi-model pipeline: density from a crowd-counting network, velocity from optical flow, and a GRU that predicts risk from a short time window. Audio from YAMNet can cross-check for panic sounds.”**
- **“The dashboard shows the current status: green = SAFE, orange = ELEVATED, red = SHOCKWAVE. Events are logged so you can audit them later.”**
- **“Right now we’re streaming synthetic data that mimics a crowd going from safe to shockwave so you can see the system react. In production this would be a live camera and optional microphone.”**

---

## Troubleshooting

- **“WAITING...” and nothing changes**  
  Start the stream: in another terminal run `python stream_synthetic_data.py --panic` or `python run_pipeline_demo.py`.

- **Port 8000 in use**  
  Stop the other app using 8000, or change the port in `server/main.py` and in the browser URL.

- **No webcam**  
  Use the synthetic demo only: `python run_hackathon_demo.py` or Option A above.

- **Run the demo again**  
  Run `python run_hackathon_demo.py` again; it will stream the same demo data so the dashboard goes SAFE → SHOCKWAVE again.
