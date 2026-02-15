import serial
import struct
import time
import numpy as np
import argparse
import random
import threading

LD2450_BAUDRATE = 256000
FRAME_HEADER = b'\xAA\xFF\x03\x00'
FRAME_ENDER = b'\x55\xCC'
DT = 0.1

class HLK_LD2450:
    def __init__(self, port, baudrate=LD2450_BAUDRATE):
        self.port = port
        self.baudrate = baudrate
        self.ser = None
        self.running = False
        self.targets = []
        self.lock = threading.Lock()

    def start(self):
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=0.1)
            self.running = True
            self.thread = threading.Thread(target=self._read_loop, daemon=True)
            self.thread.start()
            print(f"[RADAR] Connected to {self.port} at {self.baudrate} baud")
        except Exception as e:
            print(f"[RADAR] Error connecting to sensor: {e}")

    def stop(self):
        self.running = False
        if self.ser:
            self.ser.close()

    def get_latest_targets(self):
        with self.lock:
            return self.targets.copy()

    def _read_loop(self):
        buffer = b''
        while self.running:
            try:
                data = self.ser.read(1024)
                if not data:
                    continue
                buffer += data

                while len(buffer) >= 30:
                    start_idx = buffer.find(FRAME_HEADER)
                    if start_idx == -1:
                        buffer = buffer[-4:]
                        break

                    if start_idx > 0:
                        buffer = buffer[start_idx:]

                    if len(buffer) < 30:
                        break

                    if buffer[28:30] != FRAME_ENDER:
                        buffer = buffer[4:]
                        continue

                    new_targets = []
                    for i in range(3):
                        base = 4 + (i * 8)
                        x = struct.unpack('<h', buffer[base:base+2])[0]
                        y = struct.unpack('<h', buffer[base+2:base+4])[0]
                        speed = struct.unpack('<h', buffer[base+4:base+6])[0]

                        if not (x == 0 and y == 0 and speed == 0):
                            new_targets.append({
                                'x': x,
                                'y': y,
                                'speed': abs(speed)
                            })

                    with self.lock:
                        self.targets = new_targets

                    buffer = buffer[30:]

            except Exception as e:
                print(f"[RADAR] Read error: {e}")
                time.sleep(0.1)

class FeatureExtractor:
    def __init__(self):
        self.prev_mean_speed = 0.0

    def compute(self, targets, vision_density=None):
        if not targets:
            return {
                "density": 0.0,
                "mean_speed": 0.0,
                "speed_variance": 0.0,
                "radial_spread": 0.0,
                "density_gradient": 0.0,
                "acceleration": 0.0,
                "flux": 0.0
            }

        pos_x = np.array([t['x'] for t in targets]) / 1000.0
        pos_y = np.array([t['y'] for t in targets]) / 1000.0
        speeds = np.array([t['speed'] for t in targets]) / 100.0

        mean_speed = float(np.mean(speeds))

        speed_variance = float(np.var(speeds))

        if len(targets) > 1:
            centroid_x = np.mean(pos_x)
            centroid_y = np.mean(pos_y)
            dists_from_center = np.sqrt((pos_x - centroid_x)**2 + (pos_y - centroid_y)**2)
            radial_spread = float(np.mean(dists_from_center))
        else:
            radial_spread = 0.0

        if len(targets) > 1:
            from scipy.spatial.distance import pdist
            points = np.column_stack((pos_x, pos_y))
            pairwise_dists = pdist(points)
            density_gradient = float(np.std(pairwise_dists)) if len(pairwise_dists) > 0 else 0.0
        else:
            density_gradient = 0.0

        acceleration = (mean_speed - self.prev_mean_speed) / DT
        self.prev_mean_speed = mean_speed

        flux = 0.0
        if vision_density is not None:
             flux = float(vision_density) * mean_speed
        else:
             flux = float(np.sum(speeds))

        return {
            "density": float(vision_density) if vision_density is not None else 0.0,
            "mean_speed": mean_speed,
            "speed_variance": speed_variance,
            "radial_spread": radial_spread,
            "density_gradient": density_gradient,
            "acceleration": acceleration,
            "flux": flux
        }

class MockLD2450(HLK_LD2450):
    def __init__(self):
        super().__init__("MOCK")
        self.running = False

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._mock_loop, daemon=True)
        self.thread.start()
        print("[RADAR] MOCK MODE: Generating synthetic targets...")

    def _mock_loop(self):
        while self.running:
            n_targets = random.choice([1, 2, 3])
            new_targets = []
            for _ in range(n_targets):
                new_targets.append({
                    'x': random.randint(-1000, 1000),
                    'y': random.randint(500, 3000),
                    'speed': random.randint(50, 150)
                })

            with self.lock:
                self.targets = new_targets

            time.sleep(DT)

def main():
    parser = argparse.ArgumentParser(description="Radar Pipeline")
    parser.add_argument('--port', default='COM3', help="Serial port for HLK-LD2450")
    parser.add_argument('--test', action='store_true', help="Run in mock mode (no hardware needed)")
    args = parser.parse_args()

    if args.test:
        radar = MockLD2450()
    else:
        radar = HLK_LD2450(port=args.port)

    extractor = FeatureExtractor()

    radar.start()

    print(f"{'Mean Spd':>10} | {'Var':>8} | {'Spread':>8} | {'Grad':>8} | {'Accel':>8} | {'Flux':>8}")
    print("-" * 75)

    try:
        while True:
            time.sleep(DT)

            raw_targets = radar.get_latest_targets()

            dummy_density = 2.5
            features = extractor.compute(raw_targets, vision_density=dummy_density)

            print(f"{features['mean_speed']:10.2f} | "
                  f"{features['speed_variance']:8.2f} | "
                  f"{features['radial_spread']:8.2f} | "
                  f"{features['density_gradient']:8.2f} | "
                  f"{features['acceleration']:8.2f} | "
                  f"{features['flux']:8.2f}")

    except KeyboardInterrupt:
        print("\nStopping...")
        radar.stop()

if __name__ == "__main__":
    main()
