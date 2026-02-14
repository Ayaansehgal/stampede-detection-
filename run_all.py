import subprocess
import time
import sys
import os

def main():
    print("=== Starting Stampede Detection System ===")
    
    # 1. Start Server
    print("[1/3] Starting API Server...")
    # Using python -m uvicorn to ensure it uses the same python environment
    server_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000"],
        cwd=os.getcwd()
    )
    
    # Wait for server to potentialy start
    time.sleep(3)
    
    # 2. Start Audio Pipeline
    print("[2/3] Starting Audio Pipeline...")
    audio_process = subprocess.Popen(
        [sys.executable, "audio_pipeline.py"],
        cwd=os.getcwd()
    )
    
    # 3. Start Video Pipeline
    print("[3/3] Starting Video Pipeline...")
    video_process = subprocess.Popen(
        [sys.executable, "video_pipeline.py"],
        cwd=os.getcwd()
    )
    
    print("\nAll systems running. Press Ctrl+C to stop.\n")
    
    try:
        while True:
            time.sleep(1)
            # Check if processes are still alive
            if server_process.poll() is not None:
                print("Server process ended unexpectedly.")
                break
            if audio_process.poll() is not None:
                print("Audio process ended unexpectedly.")
                break
            if video_process.poll() is not None:
                print("Video process ended unexpectedly.")
                break
                
    except KeyboardInterrupt:
        print("\nStopping all services...")
    finally:
        video_process.terminate()
        audio_process.terminate()
        server_process.terminate()
        print("Shutdown complete.")

if __name__ == "__main__":
    main()
