"""Pure safety/dynamics core used by the ROS node and deterministic tests."""

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np

from .motor_kinematics import allocate_omni4_wheel_budget


@dataclass(frozen=True)
class MotionLimits:
    linear: float
    lateral: float
    angular: float
    linear_accel: float
    angular_accel: float
    linear_jerk: float
    angular_jerk: float

    def bounded_by(self, hard: 'MotionLimits') -> 'MotionLimits':
        values = {
            name: max(0.0, min(float(getattr(self, name)), float(getattr(hard, name))))
            for name in self.__dataclass_fields__
        }
        return MotionLimits(**values)


@dataclass(frozen=True)
class GuardHealth:
    armed: bool = False
    emergency_stop: bool = False
    tracking_ok: bool = False
    motor_link_ok: bool = False
    auto_engaged: bool = False
    rl_policy_ok: bool = True


@dataclass(frozen=True)
class GuardResult:
    velocity: tuple[float, float, float]
    acceleration: tuple[float, float, float]
    allowed: bool
    reason: str
    applied_scale: float
    profile: str
    red_zone: bool


class RuntimeGuard:
    """Fail-closed command gate with holonomic wheel-budget allocation."""

    def __init__(
        self,
        *,
        profiles: Mapping[str, MotionLimits],
        hard_limits: MotionLimits,
        default_profile: str,
        command_timeout_sec: float,
        red_zone_speed_scale: float,
        wheel_radius: float,
        wheel_positions: Sequence[Sequence[float]],
        wheel_drive_angles_rad: Sequence[float],
        wheel_signs: Sequence[float],
        max_wheel_speed: float,
        translation_budget_share: float = 0.45,
        profile_max_wheel_speeds=None,
    ):
        if default_profile not in profiles:
            raise ValueError(f'unknown default profile: {default_profile}')
        if command_timeout_sec <= 0.0:
            raise ValueError('command_timeout_sec must be positive')
        self.hard_limits = hard_limits
        self.profiles = {
            name: limits.bounded_by(hard_limits)
            for name, limits in profiles.items()
        }
        self.default_profile = default_profile
        self.command_timeout_sec = float(command_timeout_sec)
        self.red_zone_speed_scale = float(np.clip(red_zone_speed_scale, 0.0, 1.0))
        self.wheel_radius = float(wheel_radius)
        self.wheel_positions = np.asarray(wheel_positions, dtype=float)
        self.wheel_drive_angles = np.asarray(wheel_drive_angles_rad, dtype=float)
        self.wheel_signs = np.asarray(wheel_signs, dtype=float)
        self.max_wheel_speed = float(max_wheel_speed)
        self.profile_max_wheel_speeds = dict(profile_max_wheel_speeds or {})
        # Fraction of every wheel's limit kept for translation while yaw is
        # also requested, so goal-yaw regulation cannot starve progress.
        self.translation_budget_share = float(translation_budget_share)
        self.velocity = np.zeros(3, dtype=float)
        self.acceleration = np.zeros(3, dtype=float)
        self.last_step_time = None

    def reset(self) -> None:
        self.velocity.fill(0.0)
        self.acceleration.fill(0.0)
        self.last_step_time = None

    def effective_scale(
        self, user_scale: float, red_zone: bool, rl_scale: float = 1.0
    ) -> float:
        values = (float(user_scale), float(rl_scale))
        if not all(math.isfinite(value) for value in values):
            return 0.0
        scale = float(np.clip(values[0], 0.0, 1.0)) * float(
            np.clip(values[1], 0.0, 1.0)
        )
        if red_zone:
            scale = min(scale, self.red_zone_speed_scale)
        return scale

    def resolve_profile(self, profile: str) -> str:
        return profile if profile in self.profiles else self.default_profile

    def reference_scale(self, health: GuardHealth, user_scale: float,
                        red_zone: bool, rl_scale: float = 1.0) -> float:
        """Planner clock permission, independent of downstream command age.

        A stale output must stop the motors, but cannot also prohibit an
        otherwise healthy controller from generating its replacement. Nav2's
        monitor stops publishing repeated zeros after stop_pub_timeout, making
        applied_scale=0 a feedback deadlock at rest. All physical health gates
        still freeze this clock; step() still rejects every stale command.
        """
        if self._health_reason(health) != 'ACTIVE':
            return 0.0
        return self.effective_scale(user_scale, red_zone, rl_scale)

    def planner_speed_percentage(
        self,
        *,
        profile: str,
        user_scale: float,
        red_zone: bool,
        reference: Sequence[float],
        rl_scale: float = 1.0,
        minimum_percentage: float = 1.0,
    ) -> float:
        """Largest planner speed percentage this guard passes unclipped.

        ``reference`` is the controller's own configured (vx, vy, wz) maximum.
        A ``nav2_msgs/SpeedLimit`` percentage rescales all three of those by a
        single common ratio, so only the smallest per-axis ratio guarantees
        that no axis is planned faster than this guard will execute.  Without
        it this guard is a hidden governor: the controller keeps scoring
        trajectories inside an envelope that is silently reduced afterwards,
        so its predicted motion describes a robot that does not exist.

        The result never reaches 0.0 because Nav2 reads a zero speed limit as
        "no limit"; a fully closed gate is enforced by :meth:`step`, not here.
        """
        limits = self.profiles[self.resolve_profile(profile)]
        scale = self.effective_scale(user_scale, red_zone, rl_scale)
        axis_limits = (limits.linear, limits.lateral, limits.angular)
        reference_values = np.asarray(reference, dtype=float)
        if reference_values.shape != (3,):
            raise ValueError('reference must contain [vx, vy, wz] maxima')
        ratios = [
            (limit * scale) / float(maximum)
            for limit, maximum in zip(axis_limits, reference_values)
            if float(maximum) > 0.0
        ]
        fraction = min(ratios) if ratios else 1.0
        return float(np.clip(100.0 * fraction, float(minimum_percentage), 100.0))

    @staticmethod
    def _health_reason(health: GuardHealth) -> str:
        if health.emergency_stop:
            return 'EMERGENCY_STOP'
        if not health.armed:
            return 'DISARMED'
        if not health.tracking_ok:
            return 'LOCALIZATION_UNHEALTHY'
        if not health.rl_policy_ok:
            return 'RL_POLICY_UNHEALTHY'
        if not health.motor_link_ok:
            return 'MOTOR_LINK_UNHEALTHY'
        if not health.auto_engaged:
            return 'AUTO_NOT_ENGAGED'
        return 'ACTIVE'

    @staticmethod
    def _clamp_translation(target: np.ndarray, x_max: float, y_max: float) -> None:
        if x_max <= 0.0 or y_max <= 0.0:
            target[:2] = 0.0
            return
        ellipse = math.hypot(target[0] / x_max, target[1] / y_max)
        if ellipse > 1.0:
            target[:2] /= ellipse

    @staticmethod
    def _limit_vector(vector: np.ndarray, maximum: float) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        if maximum <= 0.0:
            return np.zeros_like(vector)
        if norm > maximum:
            return vector * (maximum / norm)
        return vector

    def _dynamic_limit(
        self, target: np.ndarray, limits: MotionLimits, dt: float
    ) -> tuple[np.ndarray, np.ndarray]:
        # A stop from upstream Collision Monitor is intentionally immediate.
        if float(np.linalg.norm(target)) < 1.0e-9:
            return np.zeros(3), np.zeros(3)

        error = target - self.velocity
        desired_accel = error / dt
        # Ramp acceleration down BEFORE reaching target velocity. Merely
        # slew-limiting error/dt holds maximum acceleration until the crossing,
        # then the anti-overshoot clamp removes it in one tick (a jerk spike).
        # Invert the discrete sum of the remaining jerk-limited ramp, using
        # this tick's dt so the result also covers timer jitter.
        norm = float(np.linalg.norm(error[:2]))
        if norm > 0.:
            desired_accel[:2] = error[:2] / norm * self._capture_acceleration(
                norm, limits.linear_jerk, dt)
        desired_accel[2] = math.copysign(self._capture_acceleration(
            abs(float(error[2])), limits.angular_jerk, dt), float(error[2]))
        desired_accel[:2] = self._limit_vector(
            desired_accel[:2], limits.linear_accel
        )
        desired_accel[2] = float(np.clip(
            desired_accel[2], -limits.angular_accel, limits.angular_accel
        ))

        accel_delta = desired_accel - self.acceleration
        accel_delta[:2] = self._limit_vector(
            accel_delta[:2], limits.linear_jerk * dt
        )
        accel_delta[2] = float(np.clip(
            accel_delta[2],
            -limits.angular_jerk * dt,
            limits.angular_jerk * dt,
        ))
        acceleration = self.acceleration + accel_delta
        velocity = self.velocity + acceleration * dt

        # Never overshoot a component through the target during rate limiting.
        for i in range(3):
            before = target[i] - self.velocity[i]
            after = target[i] - velocity[i]
            if before == 0.0 or before * after < 0.0:
                velocity[i] = target[i]
                acceleration[i] = (velocity[i] - self.velocity[i]) / dt
        return velocity, acceleration

    @staticmethod
    def _capture_acceleration(error: float, jerk: float, dt: float) -> float:
        """Largest acceleration whose discrete ramp to zero fits the error."""
        if error <= 0.0 or jerk <= 0.0:
            return 0.0
        step = jerk * dt
        count = math.floor((math.sqrt(1.0 + 8.0 * error / (step * dt)) - 1.0) / 2.0)
        return error / (dt * (count + 1)) + .5 * count * step

    def step(
        self,
        command: Sequence[float],
        *,
        now_sec: float,
        command_age_sec: float,
        health: GuardHealth,
        profile: str,
        user_scale: float,
        red_zone: bool,
        rl_scale: float = 1.0,
    ) -> GuardResult:
        reason = self._health_reason(health)
        if command_age_sec > self.command_timeout_sec:
            reason = 'STALE_COMMAND'
        if reason != 'ACTIVE':
            self.velocity.fill(0.0)
            self.acceleration.fill(0.0)
            self.last_step_time = float(now_sec)
            return GuardResult(
                (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), False, reason,
                0.0, self.resolve_profile(profile), bool(red_zone),
            )

        selected = self.resolve_profile(profile)
        limits = self.profiles[selected]
        scale = self.effective_scale(user_scale, red_zone, rl_scale)

        target = np.asarray(command, dtype=float).copy()
        if target.shape != (3,) or not np.all(np.isfinite(target)):
            target = np.zeros(3)
            reason = 'INVALID_COMMAND'
        self._clamp_translation(
            target, limits.linear * scale, limits.lateral * scale
        )
        target[2] = float(np.clip(
            target[2], -limits.angular * scale, limits.angular * scale
        ))
        target[:] = allocate_omni4_wheel_budget(
            float(target[0]), float(target[1]), float(target[2]),
            wheel_radius=self.wheel_radius,
            wheel_positions=self.wheel_positions,
            wheel_drive_angles=self.wheel_drive_angles,
            wheel_signs=self.wheel_signs,
            maximum=self.profile_max_wheel_speeds.get(selected, self.max_wheel_speed),
            translation_budget_share=self.translation_budget_share,
        )

        if reason == 'INVALID_COMMAND':
            self.velocity.fill(0.0)
            self.acceleration.fill(0.0)
            self.last_step_time = float(now_sec)
            return GuardResult(
                (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), False, reason,
                0.0, selected, bool(red_zone),
            )

        if self.last_step_time is None:
            dt = 0.01
        else:
            dt = float(np.clip(now_sec - self.last_step_time, 0.001, 0.05))
        velocity, acceleration = self._dynamic_limit(target, limits, dt)
        self.velocity = velocity
        self.acceleration = acceleration
        self.last_step_time = float(now_sec)
        return GuardResult(
            tuple(float(v) for v in velocity),
            tuple(float(v) for v in acceleration),
            True, 'ACTIVE', scale, selected, bool(red_zone),
        )
