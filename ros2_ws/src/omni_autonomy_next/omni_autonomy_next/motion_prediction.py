"""Predict delayed holonomic motion from timestamped body-frame commands."""
from collections import deque
import math
import numpy as np


class DelayedMotionPredictor:
    def __init__(self, delay_sec=.3, response_sec=.15):
        if not np.isfinite([delay_sec, response_sec]).all() or not (
                0. < delay_sec <= .5 and 0. < response_sec <= .3):
            raise ValueError('invalid motion prediction time constants')
        self.delay_sec, self.response_sec = float(delay_sec), float(response_sec)
        self.history = deque(maxlen=128)

    @property
    def horizon(self):
        return self.delay_sec+self.response_sec

    def record(self, now, command):
        command = np.asarray(command, dtype=float)
        if not math.isfinite(now) or command.shape != (3,) or not np.isfinite(command).all():
            self.history.clear()
            return
        if self.history and (now < self.history[-1][0] or now-self.history[-1][0] > .2):
            self.history.clear()
        if self.history and now == self.history[-1][0]:
            self.history.pop()
        self.history.append((float(now), command.copy()))
        # Retain the sample immediately preceding the integration window.
        while len(self.history) > 2 and self.history[1][0] < now-self.horizon-.1:
            self.history.popleft()

    def predict(self, pose, velocity, now):
        pose, velocity = np.asarray(pose, dtype=float), np.asarray(velocity, dtype=float)
        if pose.shape != (3,) or velocity.shape != (3,) or not np.isfinite(
                np.r_[pose, velocity, now]).all():
            raise ValueError('invalid measured state')
        state, predicted_velocity = pose.copy(), velocity.copy()
        count = math.ceil(self.horizon/.02)
        dt = self.horizon/count
        alpha = 1.-math.exp(-dt/self.response_sec)
        history = tuple(self.history)
        if history and not 0. <= now-history[-1][0] <= .2:
            history = ()
        # Unknown command history must not predict a moving robot stops by itself.
        fallback = velocity if not history else np.zeros(3)
        for i in range(count):
            at = now+(i+.5)*dt-self.delay_sec
            command = fallback
            for stamp, value in history:
                if stamp > at:
                    break
                command = value
            predicted_velocity += (command-predicted_velocity)*alpha
            cosine, sine = math.cos(state[2]), math.sin(state[2])
            state += dt*np.array([
                cosine*predicted_velocity[0]-sine*predicted_velocity[1],
                sine*predicted_velocity[0]+cosine*predicted_velocity[1], predicted_velocity[2]])
        return state, predicted_velocity
