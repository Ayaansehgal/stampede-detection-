import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import sounddevice as sd
import requests
import time
import argparse

SAMPLE_RATE = 16000
AUDIO_DURATION = 0.975
SERVER_URL = "http://localhost:8000/ingest_audio"

import os
LOCAL_MODEL_PATH = os.path.join("models", "yamnet")
if os.path.exists(LOCAL_MODEL_PATH):
    MODEL_PATH = LOCAL_MODEL_PATH
    print(f"Using local audio model: {MODEL_PATH}")
else:
    MODEL_PATH = 'https://tfhub.dev/google/yamnet/1'
    print(f"Using remote audio model: {MODEL_PATH}")

PANIC_LABELS = {
    'Screaming': 1.0,
    'Shout': 0.8,
    'Crowd': 0.6,
    'Children shouting': 0.9,
    'Battle cry': 1.0,
    'Whoop': 0.7
}

NEGATIVE_LABELS = {
    'Laughter': 0.8,
    'Music': 0.7,
    'Cheering': 0.6,
    'Speech': 0.3
}

class AudioClassifier:
    def __init__(self):
        print(f"Loading YAMNet model from {MODEL_PATH}...")
        self.model = hub.load(MODEL_PATH)
        print("Model loaded.")

        class_names = self.model.class_names()
        self.target_indices = {}
        self.negative_indices = {}

        for i, name_tensor in enumerate(class_names):
            name = name_tensor.numpy().decode('utf-8')
            if name in PANIC_LABELS:
                self.target_indices[i] = PANIC_LABELS[name]
                print(f"Monitoring panic class: {name} (idx {i})")
            if name in NEGATIVE_LABELS:
                self.negative_indices[i] = NEGATIVE_LABELS[name]

        print(f"Monitoring {len(self.target_indices)} panic classes and {len(self.negative_indices)} negative classes.")

    def predict(self, waveform):

        scores, embeddings, spectrogram = self.model(waveform)

        prediction = np.mean(scores.numpy(), axis=0)

        panic_score = 0.0
        details = {}

        for idx, weight in self.target_indices.items():
            score = prediction[idx]
            weighted_score = score * weight
            if weighted_score > panic_score:
                panic_score = weighted_score

            if score > 0.01:
                class_name = [k for k, v in PANIC_LABELS.items() if v == weight][0]

                details[idx] = score

        neg_score = 0.0
        for idx, weight in self.negative_indices.items():
            score = prediction[idx]
            weighted_score = score * weight
            if weighted_score > neg_score:
                neg_score = weighted_score

        if neg_score > 0.3 and panic_score < 0.8:
            damping = 1.0 - (neg_score * 0.8)
            panic_score *= max(0.2, damping)

        return panic_score, details

def cls_callback(indata, frames, time_info, status):

    if status:
        print(status)

    pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=None, help="Audio device ID")
    args = parser.parse_args()

    detector = AudioClassifier()

    block_len = int(SAMPLE_RATE * AUDIO_DURATION)

    print(f"Starting audio stream on device {args.device}...")
    print("Press Ctrl+C to stop.")

    try:
        with sd.InputStream(device=args.device, channels=1, samplerate=SAMPLE_RATE, blocksize=block_len) as stream:
            while True:

                data, overflow = stream.read(block_len)
                if overflow:
                    print("Values ignored (buffer overflow)")

                waveform = data[:, 0].astype(np.float32)

                start = time.time()
                risk, details = detector.predict(waveform)
                infer_time = (time.time() - start) * 1000

                payload = {
                    "ts": time.time(),
                    "audio_risk": float(risk)
                }

                try:
                    requests.post(SERVER_URL, json=payload, timeout=0.1)
                except:
                    pass

                bar_len = int(risk * 20)
                bar = "#" * bar_len + "-" * (20 - bar_len)
                print(f"[AUDIO] Risk: {risk:.2f} [{bar}] ({infer_time:.0f}ms)")

    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    main()
