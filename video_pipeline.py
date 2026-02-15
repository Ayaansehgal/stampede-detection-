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

        return self

    def read(self):

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

SERVER_URL = "http://localhost:8000/ingest"
MODEL_PATH = "models/model_csrnet.pth"
RESIZE_W, RESIZE_H = 640, 480
SKIP_FRAMES = 5

SPEED_SCALE = 0.2
DENSITY_AREA = 6.0
FLUX_BOOST = 1.0
SMOOTH_FACTOR = 0.5

FLOW_MAG_THRESHOLD = 1.5
MIN_MOTION_COVERAGE = 0.02
MIN_MOTION_CLUSTERS = 2

SOFT_CROWD_MIN = 2.0
SOFT_CROWD_FULL = 6.0

SHOCKWAVE_PERSISTENCE = 5

def count_motion_clusters(motion_mask):

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    cleaned = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)

    valid_clusters = 0
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > 500:
            valid_clusters += 1

    return valid_clusters

def compute_radial_spread(flow, motion_mask):

    ys, xs = np.where(motion_mask > 0)
    if len(xs) < 10:
        return 0.0

    cx, cy = np.mean(xs), np.mean(ys)
    distances = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)

    diag = np.sqrt(RESIZE_W ** 2 + RESIZE_H ** 2)
    spread = np.std(distances) / diag

    return spread * 25.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="0", help="Video source (0 for webcam, or path to mp4)")
    parser.add_argument("--picamera", action="store_true", help="Use Raspberry Pi Camera Module")
    args = parser.parse_args()

    if args.picamera:
        if not PICAMERA_AVAILABLE:
            print("Error: picamera library not found. Install with 'pip install picamera'")
            return
        print("Initializing Pi Camera...")
        cap = PiCameraStream(resolution=(RESIZE_W, RESIZE_H))

        time.sleep(2.0)
    else:
        source = int(args.source) if args.source.isdigit() else args.source
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"Error: Could not open video source {source}")
            return

    print("Loading Vision Model...")
    try:
        crowd_counter = CrowdCounter(model_path=MODEL_PATH)
    except Exception as e:
        print(f"Failed to load model from {MODEL_PATH}: {e}")
        return

    print(f"Starting pipeline. Sending data to {SERVER_URL}")
    print("Press 'q' to quit.")

    prev_gray = None

    s_speed = 0.0
    s_var = 0.0
    s_flux = 0.0
    s_acc = 0.0
    s_radial = 0.0

    shockwave_counter = 0

    last_count = 0.0
    last_density_metric = 0.0
    last_density_gradient = 0.0

    frame_idx = 0
    last_print = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (RESIZE_W, RESIZE_H))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        raw_speed = 0.0
        raw_var = 0.0
        raw_radial = 0.0
        motion_clusters = 0

        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

            motion_mask = (magnitude > FLOW_MAG_THRESHOLD).astype(np.uint8) * 255

            total_pixels = RESIZE_W * RESIZE_H
            motion_pixel_count = np.count_nonzero(motion_mask)
            motion_coverage = motion_pixel_count / total_pixels

            motion_clusters = count_motion_clusters(motion_mask)

            if motion_coverage > MIN_MOTION_COVERAGE:
                valid_magnitudes = magnitude[magnitude > FLOW_MAG_THRESHOLD]

                if len(valid_magnitudes) > 0:
                    raw_speed = np.mean(valid_magnitudes) * SPEED_SCALE
                    raw_var = np.var(valid_magnitudes) * SPEED_SCALE

                if motion_clusters <= 1 and motion_coverage > 0.30:
                    raw_speed *= 0.05
                    raw_var *= 0.05

                raw_radial = compute_radial_spread(flow, motion_mask)

        acceleration = abs(raw_speed - s_speed)

        raw_speed = min(raw_speed, 6.0)

        if frame_idx % SKIP_FRAMES == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)

            try:
                result = crowd_counter.predict(pil_image)
                last_count = result['count']
                density_map = result['density_map']

                last_density_metric = last_count / DENSITY_AREA
                last_density_gradient = np.std(density_map)
            except:
                pass

        effective_count = max(last_count, motion_clusters)
        raw_flux = effective_count * raw_speed * FLUX_BOOST

        s_speed = s_speed * (1.0 - SMOOTH_FACTOR) + raw_speed * SMOOTH_FACTOR
        s_var = s_var * (1.0 - SMOOTH_FACTOR) + raw_var * SMOOTH_FACTOR
        s_flux = s_flux * (1.0 - SMOOTH_FACTOR) + raw_flux * SMOOTH_FACTOR
        s_acc = s_acc * (1.0 - SMOOTH_FACTOR) + acceleration * SMOOTH_FACTOR
        s_radial = s_radial * (1.0 - SMOOTH_FACTOR) + raw_radial * SMOOTH_FACTOR

        prev_gray = gray
        frame_idx += 1

        feat_density = float(last_density_metric)
        feat_speed = float(s_speed)
        feat_var = float(s_var)
        feat_radial = float(s_radial)
        feat_gradient = float(last_density_gradient)
        feat_acc = float(s_acc)
        feat_flux = float(s_flux)

        heuristic_score = (
            0.3 * (feat_var / 10.0) +
            0.3 * (feat_acc / 20.0) +
            0.2 * (feat_density / 3.0) +
            0.2 * (feat_flux / 600.0)
        )
        heuristic_score = max(0.0, min(1.0, heuristic_score))

        if feat_speed < 2.0:
            heuristic_score *= 0.5

        if feat_speed > 3.0 and feat_density < 1.0:
            heuristic_score *= 0.3

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
            pass
        except Exception as e:
            if frame_idx % 100 == 0:
                print(f"[WARN] Server error: {e}")

        if gru_score is not None and gru_label not in ("BUFFERING", "MODEL_NOT_READY", "BUFFER_NOT_FULL"):

            risk_score = 0.4 * heuristic_score + 0.6 * gru_score
        else:
            risk_score = heuristic_score

        if effective_count < SOFT_CROWD_MIN:
            crowd_factor = 0.05
        elif effective_count < SOFT_CROWD_FULL:
            crowd_factor = (effective_count - SOFT_CROWD_MIN) / (SOFT_CROWD_FULL - SOFT_CROWD_MIN)
        else:
            crowd_factor = 1.0

        risk_score *= crowd_factor

        if risk_score > 0.7:
            risk_label = "SHOCKWAVE"
        elif risk_score > 0.4:
            risk_label = "ELEVATED"
        else:
            risk_label = "SAFE"

        if motion_clusters <= 1 and risk_label == "SHOCKWAVE":
            risk_score = min(risk_score, 0.35)
            risk_label = "ELEVATED"

        if risk_label == "SHOCKWAVE":
            shockwave_counter += 1
            if shockwave_counter < SHOCKWAVE_PERSISTENCE:
                risk_label = "ELEVATED"
        else:
            shockwave_counter = 0

        gru_str = f"GRU:{gru_score:.2f}" if gru_score is not None else "GRU:OFF"
        if time.time() - last_print > 1.0:
            print(f"[CALIB] Count:{last_count:.0f} Clusters:{motion_clusters} | "
                  f"Dens:{feat_density:.2f} Speed:{feat_speed:.2f} Flux:{feat_flux:.1f} | "
                  f"Heur:{heuristic_score:.2f} {gru_str} | "
                  f"Risk:{risk_score:.2f} [{risk_label}] (cf:{crowd_factor:.2f})")
            last_print = time.time()

        if risk_label == "SAFE":
            color = (0, 255, 0)
        elif risk_label == "ELEVATED":
            color = (0, 165, 255)
        elif risk_label == "SHOCKWAVE":
            color = (0, 0, 255)
        elif risk_label == "BUFFERING":
            color = (0, 255, 255)
        else:
            color = (200, 200, 200)

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
