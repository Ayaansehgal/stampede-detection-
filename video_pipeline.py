import cv2
import torch
import numpy as np
import requests
import time
import argparse
from PIL import Image
from predict import CrowdCounter

try:
    from picamera.array import PiRGBArray
    from picamera import PiCamera
    PICAMERA_AVAILABLE = True
except ImportError:
    PICAMERA_AVAILABLE = False

class PiCameraStream:
    def __init__(self, resolution=(640, 480), framerate=32):
        self.camera = PiCamera()
        self.camera.resolution = resolution
        self.camera.framerate = framerate
        self.rawCapture = PiRGBArray(self.camera, size=resolution)
        self.stream = self.camera.capture_continuous(self.rawCapture,
            format="bgr", use_video_port=True)
        self.frame = None
        self.stopped = False

    def start(self):
        # start input thread (not implemented here for simplicity, using blocking read)
        return self

    def read(self):
        # We need to simulate the blocking behavior of cv2.VideoCapture.read()
        # capture_continuous is a generator
        try:
           frame = next(self.stream)
           image = frame.array
           self.rawCapture.truncate(0) 
           return True, image
        except Exception as e:
           print(f"PiCamera Error: {e}")
           return False, None

    def release(self):
        self.camera.close()

# --- CONFIG ---
SERVER_URL = "http://localhost:8000/ingest"
MODEL_PATH = "models/model_csrnet.pth"
RESIZE_W, RESIZE_H = 640, 480
SKIP_FRAMES = 5  # Run AI every N frames

# --- CALIBRATION (Sim2Real Bridge) ---
# These scale webcam values INTO simulation-like ranges so the scaler works.
# Simulation: density ~1-5 p/m², speed ~0.5-3 m/s, flux ~30-200
SPEED_SCALE = 0.2        # Reduced from 0.5 - maps optical flow pixels → ~simulation m/s
DENSITY_AREA = 6.0       # Assumed visible area in m² (small webcam FOV)
FLUX_BOOST = 1.0         # Extra flux multiplier (tunable)
SMOOTH_FACTOR = 0.5      # Increased from 0.3 - more temporal damping to reduce spikes

# Optical Flow tuning
FLOW_MAG_THRESHOLD = 1.5   # Lowered from 5.0 — captures real walking motion
MIN_MOTION_COVERAGE = 0.02 # At least 2% of pixels must be moving (anti-proximity)
MIN_MOTION_CLUSTERS = 2    # Need ≥2 independent moving blobs for "crowd" motion

# Crowd guard (soft, not hard)
SOFT_CROWD_MIN = 2.0     # Below this count, risk is dampened (not zeroed)
SOFT_CROWD_FULL = 6.0    # At or above this count, full risk passes through

# Temporal consistency (anti-jitter)
SHOCKWAVE_PERSISTENCE = 5  # Must see high risk for N consecutive frames before declaring SHOCKWAVE


def count_motion_clusters(motion_mask):
    """Count independent moving groups using connected components."""
    # Morphological cleanup to merge nearby pixels into blobs
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    cleaned = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    
    # Filter out tiny blobs (noise) — require at least 500 pixels
    valid_clusters = 0
    for i in range(1, num_labels):  # skip background (label 0)
        if stats[i, cv2.CC_STAT_AREA] > 500:
            valid_clusters += 1
    
    return valid_clusters


def compute_radial_spread(flow, motion_mask):
    """Compute spatial spread of motion from optical flow vectors."""
    ys, xs = np.where(motion_mask > 0)
    if len(xs) < 10:
        return 0.0
    
    # Centroid of moving pixels
    cx, cy = np.mean(xs), np.mean(ys)
    distances = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    
    # Normalize by frame diagonal so it's scale-independent
    diag = np.sqrt(RESIZE_W ** 2 + RESIZE_H ** 2)
    spread = np.std(distances) / diag
    
    # Scale to simulation-like range (sim radial_spread ~3-8)
    return spread * 25.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="0", help="Video source (0 for webcam, or path to mp4)")
    parser.add_argument("--picamera", action="store_true", help="Use Raspberry Pi Camera Module")
    args = parser.parse_args()

    # Hardware (Camera)
    if args.picamera:
        if not PICAMERA_AVAILABLE:
            print("Error: picamera library not found. Install with 'pip install picamera'")
            return
        print("Initializing Pi Camera...")
        cap = PiCameraStream(resolution=(RESIZE_W, RESIZE_H))
        # Warmup
        time.sleep(2.0)
    else:
        source = int(args.source) if args.source.isdigit() else args.source
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"Error: Could not open video source {source}")
            return

    # Model (Vision)
    print("Loading Vision Model...")
    try:
        crowd_counter = CrowdCounter(model_path=MODEL_PATH)
    except Exception as e:
        print(f"Failed to load model from {MODEL_PATH}: {e}")
        return

    print(f"Starting pipeline. Sending data to {SERVER_URL}")
    print("Press 'q' to quit.")

    prev_gray = None
    
    # State variables for smoothing
    s_speed = 0.0
    s_var = 0.0
    s_flux = 0.0
    s_acc = 0.0
    s_radial = 0.0
    
    # Temporal consistency tracker
    shockwave_counter = 0
    
    # Store last density to reuse between inference frames
    last_count = 0.0
    last_density_metric = 0.0
    last_density_gradient = 0.0

    frame_idx = 0
    last_print = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize for performance
        frame = cv2.resize(frame, (RESIZE_W, RESIZE_H))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1. Optical Flow (Motion/Speed) - Run every frame for smoothness
        raw_speed = 0.0
        raw_var = 0.0
        raw_radial = 0.0
        motion_clusters = 0
        
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            
            # Create motion mask with lower threshold
            motion_mask = (magnitude > FLOW_MAG_THRESHOLD).astype(np.uint8) * 255
            
            # Check pixel coverage — prevent single-close-person false positives
            total_pixels = RESIZE_W * RESIZE_H
            motion_pixel_count = np.count_nonzero(motion_mask)
            motion_coverage = motion_pixel_count / total_pixels
            
            # Count independent moving blobs
            motion_clusters = count_motion_clusters(motion_mask)
            
            # Only register motion if it covers enough of the frame
            # AND isn't just one big blob (person right in front of camera)
            if motion_coverage > MIN_MOTION_COVERAGE:
                valid_magnitudes = magnitude[magnitude > FLOW_MAG_THRESHOLD]
                
                if len(valid_magnitudes) > 0:
                    raw_speed = np.mean(valid_magnitudes) * SPEED_SCALE
                    raw_var = np.var(valid_magnitudes) * SPEED_SCALE
                
                # If just ONE giant blob covers >30% of frame, it's a proximity artifact
                if motion_clusters <= 1 and motion_coverage > 0.30:
                    raw_speed *= 0.05  # Almost zero — this is someone in front of the camera
                    raw_var *= 0.05
                
                raw_radial = compute_radial_spread(flow, motion_mask)
            
        acceleration = abs(raw_speed - s_speed)
        
        # Cap speed at realistic maximum (prevents optical flow spikes from dominating)
        raw_speed = min(raw_speed, 6.0)  # Max ~6 m/s for human crowd movement
        
        # 2. Crowd Counting (Density) - Run every N frames
        if frame_idx % SKIP_FRAMES == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            
            try:
                result = crowd_counter.predict(pil_image)
                last_count = result['count']
                density_map = result['density_map']
                
                # Density = people per m² (using assumed webcam visible area)
                last_density_metric = last_count / DENSITY_AREA
                last_density_gradient = np.std(density_map)
            except:
                pass 
        
        # 3. Flux — approximate "sum of all agent speeds" like simulation
        #    Simulation: flux = sum(speeds) ≈ N_agents × mean_speed
        #    We approximate: count × speed × boost
        effective_count = max(last_count, motion_clusters)  # Use whichever is higher
        raw_flux = effective_count * raw_speed * FLUX_BOOST

        # Apply Smoothing (EMA)
        s_speed = s_speed * (1.0 - SMOOTH_FACTOR) + raw_speed * SMOOTH_FACTOR
        s_var = s_var * (1.0 - SMOOTH_FACTOR) + raw_var * SMOOTH_FACTOR
        s_flux = s_flux * (1.0 - SMOOTH_FACTOR) + raw_flux * SMOOTH_FACTOR
        s_acc = s_acc * (1.0 - SMOOTH_FACTOR) + acceleration * SMOOTH_FACTOR
        s_radial = s_radial * (1.0 - SMOOTH_FACTOR) + raw_radial * SMOOTH_FACTOR

        prev_gray = gray
        frame_idx += 1

        # 4. Construct features (all scaled to simulation-like ranges)
        feat_density = float(last_density_metric)
        feat_speed = float(s_speed)
        feat_var = float(s_var)
        feat_radial = float(s_radial)
        feat_gradient = float(last_density_gradient)
        feat_acc = float(s_acc)
        feat_flux = float(s_flux)

        # 5. LOCAL HEURISTIC — adapted from simulation.py formula
        #    Denominators tuned UP (less sensitive) to prevent false positives from normal contact
        #    (handshakes, pushes, people standing close together)
        heuristic_score = (
            0.3 * (feat_var / 10.0) +     # Further reduced sensitivity
            0.3 * (feat_acc / 20.0) +     # Further reduced sensitivity
            0.2 * (feat_density / 3.0) +  # Density kept same (actual crowd metric)
            0.2 * (feat_flux / 600.0)     # Further reduced sensitivity
        )
        heuristic_score = max(0.0, min(1.0, heuristic_score))  # Clamp to [0, 1]
        
        # Additional guard: minimum speed threshold — stampedes have sustained high speed
        # Handshakes/pushes are brief and slower
        if feat_speed < 2.0:  # Below 2 m/s equivalent, cap heuristic
            heuristic_score *= 0.5
        
        # CROWD SYNERGY: Real stampedes require BOTH high speed AND high density
        # Single person moving fast = low density, should not trigger
        if feat_speed > 3.0 and feat_density < 1.0:
            heuristic_score *= 0.3  # High speed but low crowd → heavily dampen

        # 6. Optional GRU Server (boost heuristic if available)
        gru_score = None
        gru_label = None
        payload = {
            "ts": time.time(),
            "density": feat_density,
            "mean_speed": feat_speed,
            "speed_variance": feat_var,
            "radial_spread": feat_radial,
            "density_gradient": feat_gradient,
            "acceleration": feat_acc,
            "flux": feat_flux
        }
        
        try:
            resp = requests.post(SERVER_URL, json=payload, timeout=0.2)
            if resp.status_code == 200:
                data = resp.json()
                gru_label = data.get("label", "Unknown")
                gru_score = data.get("risk_score", 0.0)
        except requests.exceptions.ConnectionError:
            pass  # Server not running — that's fine, heuristic handles it
        except Exception as e:
            if frame_idx % 100 == 0:  # Don't spam logs
                print(f"[WARN] Server error: {e}")

        # 7. Blend scores: if GRU is available, weighted average; otherwise pure heuristic
        if gru_score is not None and gru_label not in ("BUFFERING", "MODEL_NOT_READY", "BUFFER_NOT_FULL"):
            # 40% heuristic + 60% GRU (GRU is more sophisticated when it works)
            risk_score = 0.4 * heuristic_score + 0.6 * gru_score
        else:
            risk_score = heuristic_score

        # 8. Soft Crowd Guard: dampen (don't zero) risk for low-count scenes
        if effective_count < SOFT_CROWD_MIN:
            crowd_factor = 0.05
        elif effective_count < SOFT_CROWD_FULL:
            crowd_factor = (effective_count - SOFT_CROWD_MIN) / (SOFT_CROWD_FULL - SOFT_CROWD_MIN)
        else:
            crowd_factor = 1.0
        
        risk_score *= crowd_factor
        
        # Classify
        if risk_score > 0.7:
            risk_label = "SHOCKWAVE"
        elif risk_score > 0.4:
            risk_label = "ELEVATED"
        else:
            risk_label = "SAFE"

        # Anti-single-person guard: if only 1 cluster detected, cap risk
        if motion_clusters <= 1 and risk_label == "SHOCKWAVE":
            risk_score = min(risk_score, 0.35)
            risk_label = "ELEVATED"
        
        # Temporal consistency: SHOCKWAVE must persist for N frames
        if risk_label == "SHOCKWAVE":
            shockwave_counter += 1
            if shockwave_counter < SHOCKWAVE_PERSISTENCE:
                risk_label = "ELEVATED"  # Downgrade until confirmed
        else:
            shockwave_counter = 0  # Reset counter

        # Calibration Log (Every 1s)
        gru_str = f"GRU:{gru_score:.2f}" if gru_score is not None else "GRU:OFF"
        if time.time() - last_print > 1.0:
            print(f"[CALIB] Count:{last_count:.0f} Clusters:{motion_clusters} | "
                  f"Dens:{feat_density:.2f} Speed:{feat_speed:.2f} Flux:{feat_flux:.1f} | "
                  f"Heur:{heuristic_score:.2f} {gru_str} | "
                  f"Risk:{risk_score:.2f} [{risk_label}] (cf:{crowd_factor:.2f})")
            last_print = time.time()

        # 6. Visualization
        if risk_label == "SAFE":
            color = (0, 255, 0)
        elif risk_label == "ELEVATED":
            color = (0, 165, 255)  # Orange
        elif risk_label == "SHOCKWAVE":
            color = (0, 0, 255)    # Red
        elif risk_label == "BUFFERING":
            color = (0, 255, 255)  # Yellow
        else:
            color = (200, 200, 200)  # Gray for READY/MODEL_NOT_READY
        
        cv2.putText(frame, f"Count: {last_count:.0f} ({motion_clusters} clusters)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Speed: {s_speed:.2f}  Flux: {s_flux:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Risk: {risk_score:.2f} [{risk_label}]", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        cv2.imshow("Stampede Detection (No-Sensor Mode)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
