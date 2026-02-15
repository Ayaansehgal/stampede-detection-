import requests
import time
import sys

SERVER_URL = "http://localhost:8000/events"

def actuator_loop():
    print("=== Actuator Client Started ===")
    print("Listening for SHOCKWAVE events...")

    last_processed_hash = None

    try:
        while True:
            try:

                resp = requests.get(SERVER_URL, timeout=1)
                if resp.status_code == 200:
                    events = resp.json()

                    if events:
                        latest_event = events[-1]
                        current_hash = latest_event.get("hash")

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

def trigger_alarm(state):
    if state:
        print(">>> ACTUATOR TRIGGERED: RED LIGHT & SIREN ON <<<")

    else:
        print(">>> Actuator Safe: Systems Normal <<<")

def trigger_warning(state):
    if state:
        print(">>> ACTUATOR WARNING: YELLOW LIGHT FLASHING <<<")

if __name__ == "__main__":
    actuator_loop()
