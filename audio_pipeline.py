import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import sounddevice as sd
import requests
import time
import argparse

# --- CONFIG ---
SAMPLE_RATE = 16000     # YAMNet requires 16kHz
AUDIO_DURATION = 0.975  # YAMNet inference window (~1 second)
SERVER_URL = "http://localhost:8000/ingest_audio"

# Check for local model first, otherwise use TFHub URL
import os
LOCAL_MODEL_PATH = os.path.join("models", "yamnet")
if os.path.exists(LOCAL_MODEL_PATH):
    MODEL_PATH = LOCAL_MODEL_PATH
    print(f"Using local audio model: {MODEL_PATH}")
else:
    MODEL_PATH = 'https://tfhub.dev/google/yamnet/1'
    print(f"Using remote audio model: {MODEL_PATH}")

# --- AUDIOSET CLASSES ---
# We look for these specific classes to detect panic/danger
# Weights can be tuned based on importance
PANIC_LABELS = {
    'Screaming': 1.0,
    'Shout': 0.8,
    'Crowd': 0.6,
    'Children shouting': 0.9,
    'Battle cry': 1.0,
    'Whoop': 0.7
}

# Negative classes (filter false positives)
# If these are high, we dampen the panic score
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
        
        # Build index maps for faster lookup
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
        # YAMNet expects float32 mono audio in [-1.0, 1.0] at 16kHz
        # waveform shape: (N,)
        scores, embeddings, spectrogram = self.model(waveform)
        
        # scores shape: (N_patches, 521) -> take max over time patches
        prediction = np.mean(scores.numpy(), axis=0)
        
        # Calculate panic score (weighted max)
        panic_score = 0.0
        details = {}
        
        for idx, weight in self.target_indices.items():
            score = prediction[idx]
            weighted_score = score * weight
            if weighted_score > panic_score:
                panic_score = weighted_score
            
            # Keep details for logging
            if score > 0.01:
                class_name = [k for k, v in PANIC_LABELS.items() if v == weight][0] # Simple reverse lookup (imperfect but fine for logging)
                # Actually better to look up name from class_names if we kept it, 
                # but for now we just want the score
                details[idx] = score

        # Calculate negative score
        neg_score = 0.0
        for idx, weight in self.negative_indices.items():
            score = prediction[idx]
            weighted_score = score * weight
            if weighted_score > neg_score:
                neg_score = weighted_score

        # Dampen panic if valid negative sounds are present (e.g. loud music)
        # But don't dampen if panic is overwhelmingly high (>0.8) — e.g. screaming over music
        if neg_score > 0.3 and panic_score < 0.8:
            damping = 1.0 - (neg_score * 0.8) # e.g. neg=0.5 -> damp=0.6
            panic_score *= max(0.2, damping)

        return panic_score, details

def cls_callback(indata, frames, time_info, status):
    """Callback for sounddevice listener"""
    if status:
        print(status)
    
    # Store incoming audio in a buffer (handled by main loop via queue if needed, 
    # but for simplicity we can just process in blocks if the callback is fast enough. 
    # YAMNet inference is ~30ms on CPU, so it might block audio thread. 
    # Better approach: main loop reads from stream.)
    pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=None, help="Audio device ID")
    args = parser.parse_args()
    
    detector = AudioClassifier()
    
    # buffer metrics
    block_len = int(SAMPLE_RATE * AUDIO_DURATION)
    
    print(f"Starting audio stream on device {args.device}...")
    print("Press Ctrl+C to stop.")

    try:
        with sd.InputStream(device=args.device, channels=1, samplerate=SAMPLE_RATE, blocksize=block_len) as stream:
            while True:
                # Read audio block (blocking call)
                data, overflow = stream.read(block_len)
                if overflow:
                    print("Values ignored (buffer overflow)")
                
                # Convert to mono float32
                waveform = data[:, 0].astype(np.float32)
                
                # Predict
                start = time.time()
                risk, details = detector.predict(waveform)
                infer_time = (time.time() - start) * 1000
                
                # Send to server
                payload = {
                    "ts": time.time(),
                    "audio_risk": float(risk)
                }
                
                try:
                    requests.post(SERVER_URL, json=payload, timeout=0.1)
                except:
                    pass
                
                # Log
                # Basic ASCII bar for visual feedback
                bar_len = int(risk * 20)
                bar = "#" * bar_len + "-" * (20 - bar_len)
                print(f"[AUDIO] Risk: {risk:.2f} [{bar}] ({infer_time:.0f}ms)")
                
    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    main()
