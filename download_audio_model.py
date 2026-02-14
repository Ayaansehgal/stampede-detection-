import tensorflow as tf
import tensorflow_hub as hub
import os
import shutil

def download_model():
    print("=== Downloading Audio Model to Local Folder ===")
    
    MODEL_URL = 'https://tfhub.dev/google/yamnet/1'
    LOCAL_PATH = os.path.join("models", "yamnet")
    
    # Create models directory if not exists
    os.makedirs(LOCAL_PATH, exist_ok=True)
    
    print(f"Downloading from: {MODEL_URL}")
    print(f"Saving to: {LOCAL_PATH}")
    
    try:
        # Load from Hub (uses cache momentarily)
        model = hub.load(MODEL_URL)
        
        # Save explicitly to our local folder
        tf.saved_model.save(model, LOCAL_PATH)
        
        print("\nSUCCESS: YAMNet model saved to 'models/yamnet/'")
        print("You can see the 'saved_model.pb' file there now.")
        
    except Exception as e:
        print(f"\nERROR: Failed to download/save model: {e}")

if __name__ == "__main__":
    download_model()
