from collections import deque
import numpy as np

class StateManager:
    def __init__(self):
        self.previous_state = "SAFE"
        self.feature_buffer = deque(maxlen=20)

    def add_frame(self, features):
        self.feature_buffer.append(features)

    def get_buffer(self):
        if len(self.feature_buffer) < 20:
            return None
        return np.array(list(self.feature_buffer))

    def should_log(self, current_state):
        if current_state == "SHOCKWAVE" and self.previous_state == "SAFE":
            self.previous_state = current_state
            return True

        self.previous_state = current_state
        return False