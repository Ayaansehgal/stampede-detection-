import json
import os
from datetime import datetime

EVENT_DIR = "server/events"
os.makedirs(EVENT_DIR, exist_ok=True)

def store_event(data):
    filename = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".json"
    path = os.path.join(EVENT_DIR, filename)

    with open(path, "w") as f:
        json.dump(data, f)

    return path
