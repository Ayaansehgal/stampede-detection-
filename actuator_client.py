import requests
import time
import sys

# --- CONFIG ---
SERVER_URL = "http://localhost:8000/events" # Poll the events log or the ingest response
# On Pi, you would import RPi.GPIO as GPIO
# import RPi.GPIO as GPIO

def actuator_loop():
    print("=== Actuator Client Started ===")
    print("Listening for SHOCKWAVE events...")
    
    # Setup GPIO (Mock)
    # GPIO.setmode(GPIO.BCM)
    # GPIO.setup(18, GPIO.OUT) # Red Light / Alarm
    
    last_processed_hash = None
    
    try:
        while True:
            try:
                # Poll for latest events
                resp = requests.get(SERVER_URL, timeout=1)
                if resp.status_code == 200:
                    events = resp.json()
                    
                    if events:
                        latest_event = events[-1]
                        current_hash = latest_event.get("hash")
                        
                        # Check if it's a new event we haven't acted on
                        if current_hash != last_processed_hash:
                            label = latest_event.get("label")
                            risk = latest_event.get("risk_score")
                            
                            print(f"New Event: {label} (Risk: {risk:.2f})")
                            
                            if label == "SHOCKWAVE":
                                trigger_alarm(True)
                            elif label == "ELEVATED":
                                trigger_warning(True)
                            else:
                                trigger_alarm(False)
                                
                            last_processed_hash = current_hash
            except Exception as e:
                print(f"Connection error: {e}")
                
            time.sleep(1.0)
            
    except KeyboardInterrupt:
        print("\nExiting...")
        # GPIO.cleanup()

def trigger_alarm(state):
    if state:
        print(">>> ACTUATOR TRIGGERED: RED LIGHT & SIREN ON <<<")
        # GPIO.output(18, GPIO.HIGH)
    else:
        print(">>> Actuator Safe: Systems Normal <<<")
        # GPIO.output(18, GPIO.LOW)

def trigger_warning(state):
    if state:
        print(">>> ACTUATOR WARNING: YELLOW LIGHT FLASHING <<<")

if __name__ == "__main__":
    actuator_loop()
