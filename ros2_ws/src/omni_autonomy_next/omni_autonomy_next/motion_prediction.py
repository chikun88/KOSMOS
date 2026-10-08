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
        history = tuple(self.history)
        if history and not 0. <= now-history[-1][0] <= .2:
            history = ()
        # Unknown command history must not predict a moving robot stops by itself.
        # A recent first command does not describe the earlier delayed input.
        # Use the measured twist until a timestamped command is available;
        # assuming zero here invents braking immediately after startup/reset.
        fallback = velocity
        # Integrate each delayed command at its actual timestamp. Rounding a
        # change to the nearest 20 ms step adds phase error precisely when a
        # fast path starts braking or turning. Keep the bounded integration
        # step, but split it at every command transition inside the horizon.
        boundaries = sorted(set(np.linspace(0., self.horizon, count+1).tolist()
            + [stamp+self.delay_sec-now for stamp, _ in history
               if 0. < stamp+self.delay_sec-now < self.horizon]))
        history_index = -1
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            dt = end-start
            at = now+.5*(start+end)-self.delay_sec
            while (history_index+1 < len(history)
                   and history[history_index+1][0] <= at):
                history_index += 1
            command = fallback
            if history_index >= 0:
                command = history[history_index][1]
            delta = predicted_velocity-command
            response_integral = -self.response_sec*math.expm1(-dt/self.response_sec)
            if predicted_velocity[2] == 0. and command[2] == 0.:
                # Integrate straight responses analytically, including time
                # constants much shorter than the nominal integration step.
                body = command[:2]*dt+delta[:2]*response_integral
                cosine, sine = math.cos(state[2]), math.sin(state[2])
                state[:2] += (cosine*body[0]-sine*body[1],
                              sine*body[0]+cosine*body[1])
                predicted_velocity = command+delta*math.exp(-dt/self.response_sec)
                continue
            # The first-order body velocity and integrated yaw have analytic
            # solutions. Three-point Gauss quadrature integrates translation
            # in that rotating frame, avoiding the old right-velocity /
            # left-yaw Euler bias (centimetres at 3.5 m/s). This only improves
            # numerical integration; delay/gain identification remains needed.
            steps = (dt,)
            if dt > self.response_sec and np.max(np.abs(delta)) > 1.e-8:
                # Resolve a fast transient without making work grow as 1/tau.
                # After 18 tau its remaining amplitude is < 1.6e-8 of delta;
                # the tail can use the original bounded step. At most 37
                # substeps are added, and default tau=.15 never enters here.
                transient = min(dt, 18.*self.response_sec)
                parts = min(36, math.ceil(2.*(transient/self.response_sec)))
                steps = (transient/parts,)*parts
                if transient < dt:
                    steps += (dt-transient,)
            for step in steps:
                delta = predicted_velocity-command
                dx = dy = 0.
                for fraction, weight in ((.1127016653792583, 5./18.),
                                         (.5, 4./9.),
                                         (.8872983346207417, 5./18.)):
                    elapsed = fraction*step
                    decay = math.exp(-elapsed/self.response_sec)
                    turn = (command[2]*elapsed
                            - delta[2]*self.response_sec*math.expm1(-elapsed/self.response_sec))
                    yaw = state[2]+turn
                    vx, vy = command[:2]+delta[:2]*decay
                    cosine, sine = math.cos(yaw), math.sin(yaw)
                    dx += weight*(cosine*vx-sine*vy)
                    dy += weight*(sine*vx+cosine*vy)
                response_integral = -self.response_sec*math.expm1(-step/self.response_sec)
                state += (step*dx, step*dy, command[2]*step+delta[2]*response_integral)
                predicted_velocity = command+delta*math.exp(-step/self.response_sec)
        return state, predicted_velocity
