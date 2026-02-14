from collections import deque
import numpy as np

class StateManager:
    def __init__(self):
        self.previous_state = "SAFE"
        self.feature_buffer = deque(maxlen=5)
        
        # Audio State
        self.latest_audio_risk = 0.0
        self.last_audio_ts = 0.0

    def add_frame(self, features):
        self.feature_buffer.append(features)

    def update_audio(self, risk, ts):
        self.latest_audio_risk = risk
        self.last_audio_ts = ts

    def get_audio_risk(self, current_ts, timeout=2.0):
        # Only return audio risk if it's fresh (within timeout seconds)
        if current_ts - self.last_audio_ts < timeout:
            return self.latest_audio_risk
        return 0.0

    def get_buffer(self):
        if len(self.feature_buffer) < 5:
            return None
        return np.array(list(self.feature_buffer))

    def should_log(self, current_state):
        if current_state == "SHOCKWAVE" and self.previous_state != "SHOCKWAVE":
            self.previous_state = current_state
            return True

        self.previous_state = current_state
        return False