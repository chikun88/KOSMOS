from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math

import numpy as np


def wrap_angle(value):
    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


@dataclass(frozen=True)
class SimProfile:
    """Motion limits and the deployed gates the command actually passes.

    The speed limits come from ``runtime.yaml`` and the gate values from
    ``nav2_next.yaml`` via :func:`run_campaign.deployed_profile`.  The tracking
    gains belong to this offline model rather than to any deployed file.
    """

    speed: float = 2.00
    lateral_speed: float = 2.00
    angular_speed: float = 1.30
    acceleration: float = 0.85
    angular_acceleration: float = 2.0
    lookahead: float = 0.55
    position_gain: float = 2.1
    yaw_gain: float = 2.2
    # Offline stand-in for MPPI's critic balance.  CostCritic only outweighs
    # PathAlign/PathFollow very close to collision, so the commanded direction
    # turns away from a wall inside this band and by at most this fraction.
    # A full-authority turn cannot reach a goal that is itself 47 mm from a
    # wall, and no turn at all cannot hold the 40 mm bucket lane.
    #
    # These four constants were calibrated against the only recorded field
    # timings, the pose 4 -> 5 and 5 -> 4 round trip at 26.66 s and 26.01 s.
    # No rosbag of a failing run survives, so the calibration is deliberately
    # conservative: authority 0.22 makes every bucket-lane transit succeed in
    # 13-19 s, which contradicts both those timings and the reported field
    # behaviour, while 0.30 reproduces the observed frequent failure at
    # comparable duration.  The response is knife-edge between the two, which
    # is itself the finding: the lane is too tight for the margin available.
    repulsion_edge: float = 0.06
    repulsion_authority: float = 0.30
    speed_slowdown_edge: float = 0.30
    speed_slowdown_floor: float = 0.30
    # velocity_smoother -> RL residual -> Collision Monitor -> RuntimeGuard ->
    # UART.  One controller period plus the measured link delay.
    command_latency_sec: float = 0.10
    # collision_monitor SlowZone
    slowdown_ratio: float = 0.80
    slow_zone_margin: float = 0.15
    # collision_monitor FootprintApproach
    approach_horizon_sec: float = 1.2
    approach_step_sec: float = 0.05
    # controller_server progress_checker / goal_checker
    progress_radius: float = 0.04
    progress_angle: float = 0.08
    progress_time_allowance: float = 8.0
    xy_goal_tolerance: float = 0.04
    yaw_goal_tolerance: float = 0.035


@dataclass(frozen=True)
class EpisodeResult:
    success: bool
    collision: bool
    timeout: bool
    aborted: bool
    final_position_error: float
    final_yaw_error: float
    elapsed: float
    path_length: float
    minimum_clearance: float
    braked_fraction: float
    command_reversals: int


@dataclass(frozen=True)
class ControlObservation:
    """Small, simulator-independent state exposed to a residual controller."""

    remaining_distance: float
    clearance_margin: float
    turn_error: float
    speed_fraction: float
    goal_clearance_margin: float
    yaw_error: float


@dataclass(frozen=True)
class ControlAdjustment:
    """Bounded adjustment which cannot raise the deployed motion limits."""

    speed_scale: float = 1.0
    clearance_push: float = 1.0


def _path_length(path):
    return float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())


def _control_observation(
    position, target, goal, velocity, body_clearance, goal_clearance,
    yaw, goal_yaw, profile,
):
    direction = np.asarray(target, dtype=float) - np.asarray(position, dtype=float)
    speed = float(np.linalg.norm(velocity))
    if float(np.linalg.norm(direction)) > 1.0e-9 and speed > 1.0e-3:
        desired_heading = math.atan2(direction[1], direction[0])
        velocity_heading = math.atan2(velocity[1], velocity[0])
        turn_error = abs(wrap_angle(desired_heading - velocity_heading))
    else:
        turn_error = 0.0
    return ControlObservation(
        remaining_distance=float(np.linalg.norm(np.asarray(goal) - position)),
        clearance_margin=max(0.0, float(body_clearance)),
        turn_error=turn_error,
        speed_fraction=speed / max(profile.speed, 1.0e-6),
        goal_clearance_margin=max(0.0, float(goal_clearance)),
        yaw_error=abs(wrap_angle(float(goal_yaw) - float(yaw))),
    )


def _validate_adjustment(adjustment):
    if not isinstance(adjustment, ControlAdjustment):
        raise TypeError('controller must return ControlAdjustment')
    if not 0.0 < adjustment.speed_scale <= 1.0:
        raise ValueError('controller speed_scale must be within (0, 1]')
    if not 1.0 <= adjustment.clearance_push <= 2.0:
        raise ValueError('controller clearance_push must be within [1, 2]')


STEP_COST = 0.06
BRAKE_COST = 0.05


def _transition_reward(previous, current, success, collision, timeout, aborted, braked):
    """Dense progress with terminal penalties, tuned to match the promotion gate.

    The step and brake costs are deliberately of the same order.  An earlier
    balance charged 0.25 per braked step against 0.02 per step, so slowing down
    everywhere was an almost free way to avoid the collision brake: the learner
    took it, slowed 45 already-solved routes and rescued none.  Trading time for
    brake avoidance now only pays when it prevents an abort.
    """
    reward = (
        30.0 * (previous.remaining_distance - current.remaining_distance)
        - STEP_COST
    )
    reward -= 0.8 * max(0.0, 0.06 - current.clearance_margin)
    if braked:
        reward -= BRAKE_COST
    if collision:
        reward -= 100.0
    elif aborted:
        reward -= 60.0
    elif timeout:
        reward -= 30.0
    elif success:
        reward += 50.0
    return reward


# Largest displacement of any body point between two projected poses.  The
# deployed monitor steps every 0.05 s; sampling by distance instead keeps the
# projection from stepping over a millimetric margin next to the bucket while
# staying affordable in the open field.
_PROJECTION_STEP_M = 0.025
_MAX_PROJECTION_SAMPLES = 24
def _projection_times(clearance, speed, yaw_rate, radius, horizon, step):
    """Sample times for a swept-footprint projection, or None when it is moot.

    No body point can close more than ``speed + |yaw_rate| * radius`` per
    second, so a pose whose clearance exceeds that bound over the whole horizon
    cannot touch a wall and needs no projection at all.
    """
    closing = float(speed) + abs(float(yaw_rate)) * float(radius)
    if closing <= 1.0e-9:
        return None
    if closing * horizon < clearance:
        return None
    deployed = max(1, int(round(horizon / max(float(step), 1.0e-6))))
    needed = int(math.ceil(closing * horizon / _PROJECTION_STEP_M))
    samples = max(1, min(needed, deployed, _MAX_PROJECTION_SAMPLES))
    return np.linspace(horizon / samples, horizon, samples)


def _certified_projection_time(clearance, times, values, closing):
    """Return a clear prefix, including the gaps between projected poses.

    Every body point moves at most ``closing * dt`` in an interval. Endpoint
    clearances certify the interval only when their safe neighbourhoods cover
    it; otherwise retain the strictly clear prefix from the previous endpoint.
    Discrete positive samples alone may miss a corner crossing a thin wall.
    """
    previous_time = 0.0
    previous_clearance = float(clearance)
    for time, value in zip(times, values):
        step = float(time) - previous_time
        if previous_clearance + value <= closing * step:
            safe = np.nextafter(max(0.0, previous_clearance) / closing, 0.0)
            return min(float(time), previous_time + safe)
        previous_time = float(time)
        previous_clearance = float(value)
    return previous_time


def _project_constant_twist(position, velocity, angular_velocity, times):
    """Project a held body-frame twist, starting with map-frame velocity.

    Body axes rotate with yaw, so a combined translation/rotation follows an
    arc. ``sinc`` keeps the closed form continuous at zero angular velocity.
    """
    velocity = np.asarray(velocity, dtype=float)
    times = np.asarray(times, dtype=float)
    angles = float(angular_velocity) * times
    along = times * np.sinc(angles / np.pi)
    across = times * .5 * angles * np.sinc(angles / (2.0 * np.pi)) ** 2
    perpendicular = np.asarray([-velocity[1], velocity[0]])
    return (np.asarray(position)[None, :] + along[:, None] * velocity[None, :]
            + across[:, None] * perpendicular[None, :])


def _approach_scale(
    field, position, yaw, velocity, angular_velocity, profile, clearance
):
    """Conservative continuous-sweep stand-in for Monitor ``approach``.

    Nav2 projects the commanded velocity forward and, when the footprint would
    collide inside ``time_before_collision``, scales the command so the robot
    stops at the obstacle instead of suppressing every direction.  Motion away
    from the obstacle therefore stays possible, which is why the deployed
    configuration replaced its static stop polygon with this check.
    """
    if clearance <= 0.0:
        return 0.0
    speed = float(np.linalg.norm(velocity))
    closing = speed + abs(float(angular_velocity)) * field.body.radius
    times = _projection_times(
        clearance, speed, angular_velocity, field.body.radius,
        float(profile.approach_horizon_sec), float(profile.approach_step_sec),
    )
    if times is None:
        return 1.0
    predicted = _project_constant_twist(position, velocity, angular_velocity, times)
    predicted_yaw = yaw + angular_velocity * times
    # Magnitudes certify the gaps between samples as well as sampled poses.
    values = field.body_clearance_batch(
        predicted, predicted_yaw, cap=closing * float(times[0])
    )
    reachable = _certified_projection_time(clearance, times, values, closing)
    return float(max(0.0, reachable / float(profile.approach_horizon_sec)))


def _yaw_rate_gate(field, position, yaw, angular_velocity, profile, clearance):
    """Reduce a rotation to the angle the footprint can actually sweep.

    Both deployed command producers now do exactly this before Collision
    Monitor sees the twist -- ``rl_residual.limit_yaw_rate``, applied in
    ``rl_policy_node`` and in ``trajectory_tracker_node`` -- so the model has
    to do it too or the gate measures a controller that is not deployed.

    It used to return zero on any predicted contact.  That is the wrong shape
    as well as the wrong magnitude: this field's firing poses have 10-24
    degrees of rotational headroom against a horizon that sweeps 89 degrees at
    the profile's yaw limit, so "touches at some point in 1.2 s" is true of
    essentially every rotation beside a wall, and zeroing it left the base
    translating to the goal and then sitting there unable to turn.  Measured
    on eight goal pairs, four seeds each: 16/32 arrivals zeroing against
    24/32 reducing, with 1->2 going from four progress aborts at 21-25 s to
    four arrivals at 11.4-11.9 s.
    """
    if clearance <= 0.0:
        return 0.0
    horizon = float(profile.approach_horizon_sec)
    times = _projection_times(
        clearance, 0.0, angular_velocity, field.body.radius,
        horizon, float(profile.approach_step_sec),
    )
    if times is None:
        return angular_velocity
    predicted = np.repeat(position[None, :], len(times), axis=0)
    closing = abs(float(angular_velocity)) * field.body.radius
    values = field.body_clearance_batch(
        predicted, yaw + angular_velocity * times,
        cap=closing * float(times[0]),
    )
    reachable = _certified_projection_time(clearance, times, values, closing)
    return math.copysign(
        min(abs(angular_velocity), abs(angular_velocity) * reachable / horizon),
        angular_velocity,
    )


def simulate_episode(field, path, start_yaw, goal_yaw, profile, rng, controller=None):
    if path is None or len(path) < 2:
        return EpisodeResult(
            False, False, False, False, math.inf, math.inf, 0.0, 0.0, 0.0, 0.0, 0,
        )
    dt = 0.05
    length = _path_length(path)
    # An offline safety stop, not a modelled gate.  The goal bridge imposes no
    # per-goal navigation timeout, so on the field the discriminator is the
    # progress checker.  A tight wall-clock budget here would penalise the
    # steady crawl that keeps the collision brake from zeroing the command,
    # which is the behaviour the learner has to be free to find.
    max_time = 20.0 + 8.0 * length / max(profile.speed, 0.1)
    position = np.asarray(path[0], dtype=float).copy()
    yaw = float(start_yaw)
    velocity = np.zeros(2)
    angular_velocity = 0.0
    drift = np.zeros(2)
    index = 0
    minimum_clearance = math.inf
    elapsed = 0.0
    braked_steps = 0
    total_steps = 0
    reversals = 0
    previous_command = np.zeros(2)
    latency_steps = max(0, int(round(float(profile.command_latency_sec) / dt)))
    pending = deque(
        [(np.zeros(2), 0.0)] * latency_steps, maxlen=max(1, latency_steps + 1)
    )
    goal_clearance = field.body_clearance(path[-1], goal_yaw)
    progress_anchor = position.copy()
    progress_anchor_yaw = yaw
    progress_timer = 0.0
    if controller is not None:
        controller.begin_episode()
    while elapsed < max_time:
        total_steps += 1
        drift += rng.normal(0.0, 0.00015, 2)
        estimated = position + drift + rng.normal(0.0, 0.006, 2)
        estimated_yaw = yaw + rng.normal(0.0, 0.004)
        while index + 1 < len(path) and np.linalg.norm(path[index + 1] - estimated) < 0.22:
            index += 1
        target_index = index
        distance_sum = 0.0
        while target_index + 1 < len(path) and distance_sum < profile.lookahead:
            distance_sum += float(np.linalg.norm(path[target_index + 1] - path[target_index]))
            target_index += 1
        # Do not cut across the inside of a path corner.  This mirrors MPPI's
        # CostCritic footprint rejection in the lightweight offline model.
        while target_index > index and not field.segment_safe(
            estimated, path[target_index], estimated_yaw,
            margin=0.5 * field.planning_margin,
        ):
            target_index -= 1
        target = path[target_index]
        error = target - estimated
        remaining = float(np.linalg.norm(path[-1] - estimated))
        desired_map = profile.position_gain * error
        speed_limit = min(profile.speed, math.sqrt(max(0.0, 2.0 * profile.acceleration * remaining)))
        body_clearance = field.body_clearance(estimated, estimated_yaw)
        observation = _control_observation(
            estimated, target, path[-1], velocity, body_clearance, goal_clearance,
            estimated_yaw, goal_yaw, profile,
        )
        adjustment = (
            controller.select_adjustment(observation)
            if controller is not None else ControlAdjustment()
        )
        _validate_adjustment(adjustment)
        speed_limit *= adjustment.speed_scale
        slowdown_edge = float(profile.speed_slowdown_edge)
        if body_clearance < slowdown_edge:
            speed_limit *= float(np.clip(
                body_clearance / max(0.01, slowdown_edge),
                float(profile.speed_slowdown_floor), 1.0,
            ))
        repulsion_edge = float(profile.repulsion_edge)
        if body_clearance < repulsion_edge:
            # MPPI's CostCritic outweighs PathAlignCritic once the footprint
            # nears collision, so the commanded direction turns away from the
            # wall rather than only being scaled down.
            _, gradient = field.clearance_and_gradient(estimated)
            gradient_norm = float(np.linalg.norm(gradient))
            path_norm = float(np.linalg.norm(desired_map))
            if gradient_norm > 1.0e-6 and path_norm > 1.0e-9:
                weight = float(np.clip(
                    (repulsion_edge - body_clearance) / repulsion_edge
                    * float(profile.repulsion_authority)
                    * adjustment.clearance_push,
                    0.0, 1.0,
                ))
                desired_map = (
                    (1.0 - weight) * desired_map / path_norm
                    + weight * gradient / gradient_norm
                ) * path_norm
        norm = float(np.linalg.norm(desired_map))
        if norm > speed_limit:
            desired_map *= speed_limit / norm
        # Independent body-axis cap captures the lower lateral traction limit.
        c, s = math.cos(estimated_yaw), math.sin(estimated_yaw)
        desired_body = np.asarray([
            c * desired_map[0] + s * desired_map[1],
            -s * desired_map[0] + c * desired_map[1],
        ])
        ellipse = math.hypot(
            desired_body[0] / max(profile.speed, 1.0e-6),
            desired_body[1] / max(profile.lateral_speed, 1.0e-6),
        )
        if ellipse > 1.0:
            desired_body /= ellipse
        desired_map = np.asarray([
            c * desired_body[0] - s * desired_body[1],
            s * desired_body[0] + c * desired_body[1],
        ])

        yaw_error = wrap_angle(goal_yaw - estimated_yaw)
        desired_w = float(np.clip(
            profile.yaw_gain * yaw_error,
            -profile.angular_speed, profile.angular_speed,
        ))
        desired_w = _yaw_rate_gate(
            field, estimated, estimated_yaw, desired_w, profile, body_clearance
        )

        # Collision Monitor: proximity brake then the directional approach gate.
        if body_clearance <= float(profile.slow_zone_margin):
            desired_map = desired_map * float(profile.slowdown_ratio)
            desired_w *= float(profile.slowdown_ratio)
        scale = _approach_scale(
            field, estimated, estimated_yaw, desired_map, desired_w, profile,
            body_clearance,
        )
        if scale < 1.0:
            braked_steps += 1
        desired_map = desired_map * scale
        desired_w *= scale

        pending.append((desired_map.copy(), desired_w))
        commanded_map, commanded_w = pending.popleft() if latency_steps else (
            desired_map, desired_w
        )
        if (
            float(np.linalg.norm(commanded_map)) > 0.05
            and float(np.linalg.norm(previous_command)) > 0.05
            and float(commanded_map @ previous_command) < 0.0
        ):
            reversals += 1
        previous_command = commanded_map.copy()

        accel = commanded_map - velocity
        accel_norm = float(np.linalg.norm(accel))
        if accel_norm > profile.acceleration * dt:
            accel *= profile.acceleration * dt / accel_norm
        velocity += accel
        dw = float(np.clip(
            commanded_w - angular_velocity,
            -profile.angular_acceleration * dt,
            profile.angular_acceleration * dt,
        ))
        angular_velocity += dw

        slip = np.clip(rng.normal(1.0, 0.018, 2), 0.90, 1.04)
        position += velocity * slip * dt + rng.normal(0.0, 0.0004, 2)
        yaw = wrap_angle(yaw + angular_velocity * rng.normal(1.0, 0.012) * dt)
        elapsed += dt

        clearance = field.body_clearance(position, yaw)
        minimum_clearance = min(minimum_clearance, clearance)
        collision = clearance <= 0.0
        position_error = float(np.linalg.norm(path[-1] - position))
        final_yaw_error = abs(wrap_angle(goal_yaw - yaw))
        # nav2_controller::SimpleGoalChecker has no velocity condition.
        success = (
            not collision
            and position_error <= float(profile.xy_goal_tolerance)
            and final_yaw_error <= float(profile.yaw_goal_tolerance)
        )
        # nav2_controller::PoseProgressChecker aborts the whole action when the
        # robot fails to move far enough within its allowance.
        progress_timer += dt
        if (
            float(np.linalg.norm(position - progress_anchor))
            > float(profile.progress_radius)
            or abs(wrap_angle(yaw - progress_anchor_yaw))
            > float(profile.progress_angle)
        ):
            progress_anchor = position.copy()
            progress_anchor_yaw = yaw
            progress_timer = 0.0
        aborted = (
            not collision
            and not success
            and progress_timer >= float(profile.progress_time_allowance)
        )
        timeout = (
            not collision and not success and not aborted
            and elapsed >= max_time
        )
        if controller is not None:
            next_observation = _control_observation(
                position, target, path[-1], velocity,
                clearance, goal_clearance, yaw, goal_yaw, profile,
            )
            controller.observe_transition(
                _transition_reward(
                    observation, next_observation, success, collision, timeout,
                    aborted, scale < 1.0,
                ),
                next_observation,
                success or collision or timeout or aborted,
            )
        if collision:
            return EpisodeResult(
                False, True, False, False, math.inf, math.inf, elapsed, length,
                minimum_clearance, braked_steps / total_steps, reversals,
            )
        if success:
            return EpisodeResult(
                True, False, False, False, position_error, final_yaw_error,
                elapsed, length, minimum_clearance,
                braked_steps / total_steps, reversals,
            )
        if aborted:
            return EpisodeResult(
                False, False, False, True, position_error, final_yaw_error,
                elapsed, length, minimum_clearance,
                braked_steps / total_steps, reversals,
            )
    return EpisodeResult(
        False, False, True, False,
        float(np.linalg.norm(path[-1] - position)),
        abs(wrap_angle(goal_yaw - yaw)), elapsed, length, minimum_clearance,
        braked_steps / max(1, total_steps), reversals,
    )
