#!/usr/bin/env python3
"""Closed-loop replay of the tracker control law against the measured plant.

Run with no argument for the behaviour before the 2026-08-07 fixes, or
``--fixed`` to replay the same legs with them enabled.

Why this exists: the base weaved in yaw while translating, but the synthetic
demo never showed it.  The demo only ever ran the goal 4 <-> 5 shuttle, and
that leg is parallel to the costmap grid axis.  Every other leg is diagonal,
where SmacPlanner2D returns a staircase of cell centres; ``resample_spacing_m``
is 0.05 m, the same as the grid pitch, so the staircase survives into
``path_tangents`` and the feedforward velocity direction swings the full 45
degrees of the 8-connected grid.  Measured here: 31-44 direction reversals and
+-0.53 m/s of commanded lateral velocity per leg.

Plant is the drivebase model this repo measured against: 120 ms dead time plus
an 80 ms first-order lag.  Sensors are sampled at the rates the real nodes
publish, with the measurement wheel's encoder quantisation, because the
tracker feeds that raw one-sample difference back with a gain near 0.5.

``fixes`` knobs, and what each one turned out to be worth:

  smooth-path       shipped.  Kills the staircase.  Cross-track on a turning
                    leg 50.2 -> 13.0 mm, rate limiter 99.9 -> 10.6 % of ticks.
  reference-lead    shipped.  Sample the reference at t+lag, matching the
                    state prediction.  Along-track lag 32 -> 4 mm.
  velocity-filter   shipped.  Low-pass the measurement-wheel twist before it
                    is differentiated into the command.
  yaw-anchor        investigated, NOT shipped.  Carrying the yaw ramp across
                    replans instead of re-anchoring it on the measured yaw
                    changed nothing measurable; goal yaw already settles at
                    0.00-0.01 deg.
  body-rate-limit   investigated, NOT shipped.  Rate-limiting in the world
                    frame was a candidate for the saturated limiter, but the
                    saturation came from the staircase; body-frame limiting
                    is also the right frame for motor current.

``--rotation`` answers a different question on the same plant: how much time a
leg costs, and how smooth the rotation is, for each way of deciding the yaw
profile.  ``yaw_mode`` selects it:

  ramp      the deployed default.  Retire the yaw linearly in arc length.
  plan-raw  the arc-length DP of ``omni_yaw.plan_yaw_profile``, interpolated
            straight from its 0.30 m knots.  This is what ``optimize_yaw``
            shipped as, and its yaw rate is a staircase.
  plan      the same DP with the knots rounded off (``smooth_yaw_profile``)
            and the angular rate/acceleration folded into the speed profile.

The rotation metrics are what "smoothly" has to mean for a command: the
angular acceleration the tracker demands, and how often the output rate
limiter has to refuse it.  A yaw staircase is invisible in yaw *error* until
it saturates, so measuring the demand is the point.
"""
import json
import math
import sys
from collections import deque
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1] / 'ros2_ws' / 'src' / 'omni_autonomy_next'
sys.path.insert(0, str(ROOT))

from omni_autonomy_next.omni_yaw import OmniEnvelope, wrap  # noqa: E402
from omni_autonomy_next.omni_yaw import (  # noqa: E402
    direction_speed_limits, plan_yaw_profile, smooth_yaw_profile,
)
from omni_autonomy_next.rl_residual import CadClearanceModel  # noqa: E402
from omni_autonomy_next.trajectory_tracker_node import (  # noqa: E402
    Trajectory, path_tangents, resample, smooth_path, terminal_translation,
    terminal_yaw_reference, to_body,
)

robot = yaml.safe_load((ROOT / 'config' / 'robot.yaml').read_text())
ENVELOPE = OmniEnvelope(robot['robot']['drivetrain'])
guard = yaml.safe_load(
    (ROOT / 'config' / 'runtime.yaml').read_text())['runtime_guard'][
        'ros__parameters']
PROFILES = json.loads(guard['profiles_json'])
PROFILE = PROFILES[guard['default_profile']]
SPEED, LATERAL, YAW_LIMIT = (float(PROFILE['linear']),
                             float(PROFILE['lateral']),
                             float(PROFILE['angular']))
nav2 = yaml.safe_load((ROOT / 'config' / 'nav2_next.yaml').read_text())
ACCEL = float(nav2['velocity_smoother']['ros__parameters']['max_accel'][0])
YAW_ACCEL = float(nav2['velocity_smoother']['ros__parameters']['max_accel'][2])
GOAL = nav2['controller_server']['ros__parameters']['goal_checker']
XY_TOLERANCE = float(GOAL['xy_goal_tolerance'])
YAW_TOLERANCE = float(GOAL['yaw_goal_tolerance'])
YAW_SMOOTHING = 0.25       # trajectory_tracker yaw_smoothing_m


def profile_limits(name):
    limits = PROFILES[name]
    return (float(limits['linear']), float(limits['lateral']),
            float(limits['angular']))


def clearance_model():
    """The deployed ten-vertex footprint against the CAD walls, or None."""
    try:
        return CadClearanceModel.from_yaml(
            str(ROOT / 'config' / 'field_cad.yaml'),
            str(ROOT / 'config' / 'competition_footprints.yaml'), 'NORMAL')
    except (OSError, ValueError) as error:      # pragma: no cover - diagnostic
        print(f'  (footprint clearance unavailable: {error})')
        return None

POSITION_GAIN, POSITION_DAMPING, YAW_GAIN = 0.8, 0.10, 1.0
FEEDBACK_DELAY, LATERAL_ACCEL = 0.20, 1.2
TERMINAL_ZONE, TERMINAL_SPEED = 0.10, 0.30   # trajectory_tracker defaults
MAX_PREDICTED_YAW, MAX_LEAD = 0.35, 0.35
# The plant is integrated at 100 Hz.  The control law runs at the rate the
# deployed tracker actually publishes, which is a different number: the
# tracker was cut 100 -> 30 -> 20 Hz so collision_monitor's swept-footprint
# check would fit its budget (2026-09-01, -09-02).  Replaying the control law
# at the plant step hides the zero-order hold and makes every gain in the loop
# act five times more often than it does on the robot.
PERIOD = 0.01
CONTROL_HZ = 20.0          # trajectory_tracker control_rate_hz
DEAD_TIME, TAU = 0.120, 0.080
POSE_RATE, ODOM_RATE, REPLAN_RATE, ICP_RATE = 60.0, 50.0, 1.0, 12.0
YAW_COUNT_RAD = 0.000351          # measurement-wheel yaw per encoder count


def build(points, current_yaw, goal_yaw, entry_speed, smoothing=0.0,
          yaw_mode='ramp', profile=(SPEED, LATERAL, YAW_LIMIT),
          clearance=None, angular_limits=True,
          terminal_zone=TERMINAL_ZONE, terminal_speed=TERMINAL_SPEED,
          yaw_settles_early=True):
    """``_build_trajectory`` for one plan, at the given yaw mode."""
    speed_max, lateral_max, yaw_max = profile
    points = resample(points, 0.05)
    points = smooth_path(points, 0.05, smoothing)
    tangents = path_tangents(points)
    arclength = np.concatenate((
        [0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))))
    # Retire the yaw by the time the terminal approach starts, as the tracker
    # does, so the last stretch is a pure translation settle.
    cutoff = (max(arclength[-1] - terminal_zone, 0.5 * arclength[-1])
              if yaw_settles_early else arclength[-1])
    if yaw_mode == 'ramp':
        fraction = np.clip(arclength / max(cutoff, 1.0e-6), 0.0, 1.0)
        yaws = current_yaw + wrap(goal_yaw - current_yaw) * fraction
    else:
        yaws = planned_yaw(points, tangents, arclength, current_yaw, goal_yaw,
                           profile, clearance,
                           smoothing=(YAW_SMOOTHING if yaw_mode == 'plan'
                                      else 0.0), cutoff=cutoff)
    limits, _ = direction_speed_limits(
        tangents, yaws, arclength, envelope=ENVELOPE,
        ellipse_x=speed_max, ellipse_y=lateral_max)
    limits = np.where((arclength[-1] - arclength) <= terminal_zone,
                      np.minimum(limits, terminal_speed), limits)
    return Trajectory(points, yaws, limits, acceleration=ACCEL,
                      lateral_acceleration=LATERAL_ACCEL,
                      entry_speed=entry_speed,
                      angular_speed=yaw_max if angular_limits else None,
                      angular_acceleration=(YAW_ACCEL if angular_limits
                                            else None))


def planned_yaw(points, tangents, arclength, current_yaw, goal_yaw, profile,
                clearance, smoothing, cutoff=None):
    """``TrajectoryTracker._plan_yaw`` with the deployed parameters."""
    speed_max, lateral_max, yaw_max = profile
    total = float(arclength[-1] if cutoff is None else cutoff)
    count = max(2, int(round(total / 0.30)) + 1)
    targets = np.linspace(0.0, total, count)
    indices = np.searchsorted(arclength, targets).clip(0, len(points) - 1)
    step_lengths = np.diff(targets)
    max_step = np.maximum(yaw_max * step_lengths / max(speed_max, 1.0e-3),
                          math.radians(5.0))
    coarse = plan_yaw_profile(
        points[indices], tangents[indices], envelope=ENVELOPE,
        ellipse_x=speed_max, ellipse_y=lateral_max, current_yaw=current_yaw,
        goal_yaw=goal_yaw, clearance_model=clearance, minimum_clearance=0.02,
        clearance_weight=1.6, max_yaw_step=max_step)
    planned = np.interp(
        np.minimum(arclength, total), arclength[indices], coarse)
    return smooth_yaw_profile(planned, arclength, smoothing)


def grid_path(start, goal, cell=0.05):
    """SmacPlanner2D's cell centres, with the deployed goal snap applied.

    The planner quantises to cells and stops up to its 0.20 m tolerance short,
    so ``_build_trajectory`` appends the real goal from
    ``/navigation/active_goal``.  Without that last step the probe measures
    arrival against a cell centre up to 35 mm from the goal the goal checker
    is using.
    """
    start, goal = np.asarray(start, float), np.asarray(goal, float)
    steps = max(2, int(np.linalg.norm(goal - start) / cell) + 1)
    raw = start + np.linspace(0, 1, steps)[:, None] * (goal - start)
    return np.vstack((np.round(raw / cell) * cell, goal[None, :]))


def simulate(start, goal, goal_yaw, *, duration=12.0, initial_yaw_error=0.0,
             fixes=(), seed=0, yaw_disturbance=0.0, control_hz=CONTROL_HZ,
             plant_gain=1.0, plant_yaw_gain=None,
             position_gain=POSITION_GAIN, yaw_gain=YAW_GAIN,
             odom_noise_mps=0.0, odom_noise_radps=0.0,
             profile=None, yaw_mode='ramp', clearance=None,
             angular_limits=True, terminal_zone=TERMINAL_ZONE,
             terminal_speed=TERMINAL_SPEED, yaw_settles_early=True):
    """Replay ``_tick``.  ``fixes`` names the corrections to enable."""
    rng = np.random.default_rng(seed)
    # How much of the commanded velocity the drivebase actually delivers.
    # The gateway converts m/s to wheel command units with a *derived*
    # constant (auto_units_per_mps = 7202, obtained by declaring full manual
    # stick to be 0.55 m/s), so this ratio is unverified on the real robot.
    # It is the one plant property a feed-forward controller cannot absorb for
    # free: the reference is executed at gain * plan and the error has to be
    # removed through 200 ms of dead time.
    gain = np.array([
        float(plant_gain), float(plant_gain),
        float(plant_gain if plant_yaw_gain is None else plant_yaw_gain)])
    # The command holds between control ticks, so the plant keeps running on
    # the last one.  Snap the period to a whole number of plant steps.
    control_stride = max(1, int(round(1.0 / (float(control_hz) * PERIOD))))
    control_period = control_stride * PERIOD
    limits = profile_limits(profile or guard['default_profile'])
    SPEED, LATERAL, YAW_LIMIT = limits
    plan = dict(yaw_mode=yaw_mode, profile=limits, clearance=clearance,
                angular_limits=angular_limits, terminal_zone=terminal_zone,
                terminal_speed=terminal_speed,
                yaw_settles_early=yaw_settles_early)
    smoothing = 0.12 if 'smooth-path' in fixes else 0.0
    vel_tau = 0.06 if 'velocity-filter' in fixes else 0.0
    filtered = np.zeros(3)
    anchor_yaw = 'yaw-anchor' in fixes      # keep the yaw ramp across replans
    align_ref = 'reference-lead' in fixes   # sample the reference at t+tau
    body_rate = 'body-rate-limit' in fixes  # rate-limit in the world frame

    truth = np.array([start[0], start[1], initial_yaw_error], float)
    actual, command = np.zeros(3), np.zeros(3)
    pipeline = deque([np.zeros(3)] * max(1, int(round(DEAD_TIME / PERIOD))))
    pose, odom_twist = truth.copy(), np.zeros(3)
    odom_prev, odom_prev_t, residual = truth.copy(), 0.0, 0.0
    planned_yaw_ref = None                  # survives replans when fixed

    trajectory = build(grid_path(start, goal), truth[2], goal_yaw, 0.0,
                       smoothing, **plan)
    reference_time, rows = 0.0, []
    yaw_demand = 0.0

    for step in range(int(duration / PERIOD)):
        now = step * PERIOD
        stride = max(1, int(round(1.0 / (ODOM_RATE * PERIOD))))
        if step % stride == 0:
            dt = now - odom_prev_t
            if dt > 1e-9:
                delta = truth - odom_prev
                body = to_body(delta[:2], odom_prev[2])
                raw = wrap(delta[2]) + residual
                dyaw = YAW_COUNT_RAD * round(raw / YAW_COUNT_RAD)
                residual = raw - dyaw
                odom_twist = np.array([body[0] / dt, body[1] / dt, dyaw / dt])
                odom_twist = odom_twist + np.array([
                    rng.normal(0.0, odom_noise_mps),
                    rng.normal(0.0, odom_noise_mps),
                    rng.normal(0.0, odom_noise_radps)])
                # _on_odom filters in the subscription callback, so the filter
                # advances once per odometry sample, not once per control tick.
                if vel_tau > 0.0:
                    filtered = filtered + (1.0 - math.exp(-dt / vel_tau)) * (
                        odom_twist - filtered)
            odom_prev, odom_prev_t = truth.copy(), now
        c, s = math.cos(pose[2]), math.sin(pose[2])
        pose = pose + PERIOD * np.array([
            odom_twist[0] * c - odom_twist[1] * s,
            odom_twist[0] * s + odom_twist[1] * c, odom_twist[2]])
        if step % max(1, int(round(1.0 / (ICP_RATE * PERIOD)))) == 0:
            pose = truth + np.array([0.0, 0.0, rng.normal(0.0, 0.0025)])

        if step and step % int(round(1.0 / (REPLAN_RATE * PERIOD))) == 0:
            # The deployed build anchors the yaw ramp on the measured yaw.
            start_yaw = (planned_yaw_ref if (anchor_yaw and planned_yaw_ref
                                             is not None) else pose[2])
            trajectory = build(grid_path(pose[:2], goal), start_yaw, goal_yaw,
                               float(np.linalg.norm(odom_twist[:2])),
                               smoothing, **plan)
            reference_time = trajectory.time_at_arclength(
                trajectory.project(pose[:2]))

        # One control tick.  Between ticks the command holds and the plant
        # keeps integrating it, which is the lag the 20 Hz loop really has.
        if step % control_stride == 0:
            measured = filtered if vel_tau > 0.0 else odom_twist
            c, s = math.cos(pose[2]), math.sin(pose[2])
            world_measured = np.array([measured[0] * c - measured[1] * s,
                                       measured[0] * s + measured[1] * c])
            position = pose[:2] + world_measured * FEEDBACK_DELAY
            yaw_lead = float(np.clip(measured[2] * FEEDBACK_DELAY,
                                     -MAX_PREDICTED_YAW, MAX_PREDICTED_YAW))
            predicted_yaw = pose[2] + yaw_lead

            reference_time = min(reference_time + control_period,
                                 trajectory.duration)
            actual_s = trajectory.project(position)
            sample_time = reference_time
            if align_ref:
                sample_time = min(reference_time + FEEDBACK_DELAY,
                                  trajectory.duration)
            reference = trajectory.sample(sample_time)
            if reference[4] > actual_s + MAX_LEAD:
                sample_time = trajectory.time_at_arclength(actual_s + MAX_LEAD)
                reference_time = max(0., sample_time - (
                    FEEDBACK_DELAY if align_ref else 0.))
                reference = trajectory.sample(sample_time)
            ref_position, ref_velocity, ref_yaw, ref_yaw_rate, _ = reference
            planned_yaw_ref = ref_yaw
            # metric only: where the profile says the base should be *now*
            now_ref = trajectory.sample(reference_time)

            error = ref_position - position
            world_velocity = (
                ref_velocity + position_gain * error
                + POSITION_DAMPING * (ref_velocity - world_measured))
            ref_yaw, ref_yaw_rate = terminal_yaw_reference(
                ref_yaw, ref_yaw_rate, float(trajectory.yaws[-1]),
                trajectory.length - actual_s, terminal_zone)
            yaw_error = wrap(ref_yaw - predicted_yaw)
            yaw_command = ref_yaw_rate + yaw_gain * yaw_error
            world_velocity = terminal_translation(
                world_velocity, position, trajectory.points[-1],
                trajectory.length - actual_s, terminal_zone,
                position_gain, terminal_speed)
            vx, vy = to_body(world_velocity, predicted_yaw)
            ellipse = math.hypot(vx / SPEED, vy / LATERAL)
            if ellipse > 1.0:
                vx, vy = vx / ellipse, vy / ellipse
            yaw_command = float(np.clip(yaw_command, -YAW_LIMIT, YAW_LIMIT))
            cost = ENVELOPE.wheel_cost(vx, vy, yaw_command)
            if cost > ENVELOPE.max_wheel:
                k = ENVELOPE.max_wheel / cost
                vx, vy, yaw_command = vx * k, vy * k, yaw_command * k

            # rate limit
            target = np.array([vx, vy])
            if body_rate:
                # compare in the world frame: rotating the body frame is not
                # physical acceleration
                prev_world = np.array([
                    command[0] * c - command[1] * s,
                    command[0] * s + command[1] * c])
                cp, sp = math.cos(predicted_yaw), math.sin(predicted_yaw)
                want_world = np.array([
                    target[0] * cp - target[1] * sp,
                    target[0] * sp + target[1] * cp])
                d = want_world - prev_world
                n = float(np.linalg.norm(d))
                if n > ACCEL * control_period:
                    want_world = prev_world + d * (ACCEL * control_period / n)
                target = to_body(want_world, predicted_yaw)
                limited = float(n > ACCEL * control_period)
            else:
                d = target - command[:2]
                n = float(np.linalg.norm(d))
                limited = float(n > ACCEL * control_period)
                if n > ACCEL * control_period:
                    target = command[:2] + d * (ACCEL * control_period / n)
            yaw_demand = abs(yaw_command - command[2]) / control_period
            yaw_saturated = float(yaw_demand > YAW_ACCEL + 1.0e-9)
            yaw_budget = YAW_ACCEL * control_period
            yaw_out = command[2] + float(np.clip(
                yaw_command - command[2], -yaw_budget, yaw_budget))
            command = np.array([target[0], target[1], yaw_out])

        pipeline.append(command.copy())
        delayed = pipeline.popleft()
        actual += (delayed * gain - actual) * (PERIOD / TAU)
        ct, st = math.cos(truth[2]), math.sin(truth[2])
        spin = yaw_disturbance if now > 1.0 else 0.0
        truth = truth + PERIOD * np.array([
            actual[0] * ct - actual[1] * st,
            actual[0] * st + actual[1] * ct, actual[2] + spin])

        # along-track / cross-track error against the reference point
        tangent = now_ref[1] / max(float(np.linalg.norm(now_ref[1])), 1e-9)
        normal = np.array([-tangent[1], tangent[0]])
        gap = now_ref[0] - truth[:2]
        rows.append((now, truth[2], yaw_command, ref_yaw,
                     float(gap @ tangent), float(gap @ normal), limited,
                     float(np.linalg.norm(np.asarray(goal, float) - truth[:2])),
                     yaw_saturated, yaw_demand,
                     abs(wrap(goal_yaw - truth[2])),
                     float(actual[0]), float(actual[1]), float(actual[2])))
    return np.array(rows)


def arrival(history):
    """First instant the deployed goal checker would accept, or None.

    Match the deployed goal checker, including physical stopping speed when
    StoppedGoalChecker is selected. A momentary position hit is insufficient.
    """
    inside = (history[:, 7] <= XY_TOLERANCE) & (history[:, 10] <= YAW_TOLERANCE)
    if GOAL.get('plugin', '').endswith('StoppedGoalChecker'):
        inside &= np.hypot(history[:, 11], history[:, 12]) <= float(GOAL['trans_stopped_velocity'])
        inside &= np.abs(history[:, 13]) <= float(GOAL['rot_stopped_velocity'])
    hit = np.flatnonzero(inside)
    return float(history[hit[0], 0]) if len(hit) else None


LEGS = [
    ('4->5  straight, yaw held', (-0.81, 1.34), (-0.81, -2.31), 0.0,
     math.radians(6.0), 0.0),
    ('3->6  translate + 11 deg turn', (-3.88, -0.55), (-0.81, 0.23), -0.20,
     0.0, 0.0),
    ('4->5  with a 0.15 rad/s yaw push', (-0.81, 1.34), (-0.81, -2.31), 0.0,
     0.0, 0.15),
]


def report(fixes):
    for name, start, goal, goal_yaw, err, push in LEGS:
        h = simulate(start, goal, goal_yaw, initial_yaw_error=err, fixes=fixes,
                     yaw_disturbance=push)
        run = (h[:, 0] > 1.0) & (h[:, 0] < 9.0)
        yaw_dev = np.degrees(h[run, 1] - h[run, 3])
        print(f'  {name}')
        print(f'      yaw vs reference : mean {yaw_dev.mean():+6.2f} deg   '
              f'peak-to-peak {yaw_dev.max() - yaw_dev.min():5.2f} deg')
        print(f'      along-track lag  : {h[run, 4].mean() * 1000:+6.1f} mm   '
              f'cross-track {np.abs(h[run, 5]).max() * 1000:5.1f} mm')
        print(f'      accel limiter on : {100 * h[run, 6].mean():5.1f} % of '
              f'ticks        final error {h[-1, 7] * 1000:5.1f} mm')


def sweep(fixes=()):
    """How the loop responds to real measurement-wheel velocity noise.

    The synthetic demo the tracker was validated in had exact odometry, so
    this axis was never exercised before the drivebase was connected.
    """
    print('\nStraight leg 4->5, measurement-wheel velocity noise swept:')
    print(f'  {"noise m/s":>9} {"rad/s":>7} {"yaw p-p":>10} {"ring":>8} '
          f'{"cross-track":>12} {"limiter":>9}')
    for nv, nw in [(0.0, 0.0), (0.01, 0.02), (0.02, 0.04), (0.04, 0.08),
                   (0.06, 0.12)]:
        h = simulate((-0.81, 1.34), (-0.81, -2.31), 0.0, seed=3, fixes=fixes,
                     odom_noise_mps=nv, odom_noise_radps=nw)
        run = (h[:, 0] > 1.0) & (h[:, 0] < 8.0)
        yaw = np.degrees(h[run, 1] - h[run, 3])
        span = h[run, 0][-1] - h[run, 0][0]
        freq = np.sum(np.diff(np.sign(yaw - yaw.mean())) != 0) / span / 2
        print(f'  {nv:9.2f} {nw:7.2f} {yaw.max() - yaw.min():8.2f} deg '
              f'{freq:6.2f} Hz {np.abs(h[run, 5]).max() * 1000:9.1f} mm '
              f'{100 * h[run, 6].mean():7.1f} %')


ROTATION_LEGS = [
    ('1->4  diagonal + 90 deg', (-1.80, 4.75), (-0.81, 1.34),
     -0.5 * math.pi, 0.0),
    ('1->2  along -x + 90 deg', (-1.80, 4.75), (-4.19, 4.69),
     -0.5 * math.pi, 0.0),
    ('4->5  along -y, yaw held', (-0.81, 1.34), (-0.81, -2.31), 0.0, 0.0),
    ('3->6  diagonal + 11 deg', (-3.88, -0.55), (-0.81, 0.23), 0.0, -0.20),
]

ROTATION_MODES = [
    ('ramp      ', 'ramp', None),
    ('plan-raw  ', 'plan-raw', None),
    ('plan      ', 'plan', None),
    ('plan sprint', 'plan', 'sprint'),
    ('ramp sprint', 'ramp', 'sprint'),
]

FIXES = ('smooth-path', 'reference-lead', 'velocity-filter')


def rotation(duration=20.0):
    """Leg time and rotation smoothness for each way of choosing the yaw."""
    clearance = clearance_model()
    print(f'Plant: {DEAD_TIME * 1000:.0f} ms dead time + {TAU * 1000:.0f} ms '
          f'lag.  Goal checker {XY_TOLERANCE * 1000:.0f} mm / '
          f'{math.degrees(YAW_TOLERANCE):.1f} deg.')
    print(f'Angular acceleration available to the command: {YAW_ACCEL:.2f} '
          f'rad/s^2.\n')
    for name, start, goal, start_yaw, goal_yaw in ROTATION_LEGS:
        print(f'  {name}   ({np.linalg.norm(np.subtract(goal, start)):.2f} m, '
              f'{math.degrees(wrap(goal_yaw - start_yaw)):+.0f} deg)')
        print(f'      {"yaw mode":11} {"arrive":>7} {"plan":>6} {"peak v":>7} '
              f'{"yaw accel demand":>17} {"limiter":>8} {"yaw p-p":>8} '
              f'{"cross":>7}')
        for label, mode, profile in ROTATION_MODES:
            history = simulate(
                start, goal, goal_yaw, duration=duration, fixes=FIXES,
                initial_yaw_error=start_yaw, yaw_mode=mode,
                profile=profile, clearance=clearance)
            reference = build(
                grid_path(start, goal), start_yaw, goal_yaw, 0.0, 0.12,
                yaw_mode=mode, clearance=clearance,
                profile=profile_limits(profile or guard['default_profile']))
            reached = arrival(history)
            # Skip the launch step: at t=0 the command is zero and the profile
            # already wants a finite yaw rate, so the first 0.1 s always shows
            # the whole rate limiter's worth of demand.  The question here is
            # the rotation while under way.
            window = ((history[:, 0] > 0.5)
                      & (history[:, 0] < (reached if reached else duration)))
            # wrap: the plant integrates yaw without wrapping while the
            # reference is wrapped, so a leg that crosses +-pi would otherwise
            # report a 360 deg artefact as a tracking error.
            yaw_dev = np.degrees([
                wrap(row[1] - row[3]) for row in history[window]])
            yaw_dev = np.asarray(yaw_dev)
            print(f'      {label:11} '
                  f'{("%.2f s" % reached) if reached else "  none":>7} '
                  f'{reference.duration:5.2f}s {np.max(reference.speed):6.3f} '
                  f'{np.max(history[window, 9]):9.2f} rad/s^2 '
                  f'{100 * history[window, 8].mean():6.1f} % '
                  f'{yaw_dev.max() - yaw_dev.min():7.2f}d '
                  f'{np.abs(history[window, 5]).max() * 1000:5.1f}mm')
        print()


def terminal(duration=22.0):
    """What the low-speed final approach costs, and what it buys.

    The last ``terminal_approach_m`` is driven by a pure position P loop capped
    at ``terminal_speed``.  It exists because the drivebase carries 120 ms of
    dead time and an 80 ms lag, so a profile that brings the *command* to zero
    at the goal leaves the machine still moving: measured 130 mm past a pose
    with 59 mm of clearance.  Latency compensation (``feedback_delay_sec``) now
    removes that overshoot by predicting the pose the command will act on, so
    the question is how much of the low-speed crawl is still paying for
    anything.  Overshoot is what decides it, not arrival time.
    """
    print(f'  {"zone":>5} {"cap":>5} {"arrive":>8} {"overshoot":>10} '
          f'{"settled":>8} {"yaw at/after":>13}')
    for name, start, goal, start_yaw, goal_yaw in ROTATION_LEGS:
        print(f'  {name}')
        for zone, cap in [(0.16, 0.16), (0.16, 0.35), (0.12, 0.30),
                          (0.10, 0.30), (0.10, 0.35), (0.08, 0.50),
                          (0.0, 0.0)]:
            history = simulate(
                start, goal, goal_yaw, duration=duration, fixes=FIXES,
                initial_yaw_error=start_yaw, terminal_zone=zone,
                terminal_speed=cap)
            reached = arrival(history)
            if reached is None:
                print(f'  {zone:5.2f} {cap:5.2f} {"    none":>8}')
                continue
            after = history[history[:, 0] >= reached]
            settle = after[after[:, 0] <= reached + 2.5]
            # overshoot: how far it goes back out after its closest approach
            closest = int(np.argmin(settle[:, 7]))
            print(f'  {zone:5.2f} {cap:5.2f} {reached:6.2f} s '
                  f'{settle[closest:, 7].max() * 1000:7.1f} mm '
                  f'{settle[-1, 7] * 1000:5.1f} mm '
                  f'{math.degrees(settle[0, 10]):6.2f}/'
                  f'{math.degrees(settle[-1, 10]):.2f} deg')
        print()


MARGIN_LEGS = [
    ('4->5  yaw held', (-0.81, 1.34), (-0.81, -2.31), 0.0,
     dict(initial_yaw_error=math.radians(6.0), duration=16.0)),
    ('1->4  +90 deg', (-1.80, 4.75), (-0.81, 1.34), -0.5 * math.pi,
     dict(duration=18.0)),
]
MARGIN_GAINS = (1.0, 2.0, 2.5, 2.8, 3.2)
MARGIN_YAW_GAINS = (2.6, 2.0, 1.6, 1.2)


def margin():
    """How much drivebase gain error the yaw loop survives, per ``yaw_gain``.

    The drivebase gain is delivered speed / commanded speed.  It is set by
    ``auto_units_per_mps`` in the gateway (and its copy in ``robomas_uart``),
    which was *derived* by declaring full manual stick to be 0.55 m/s rather
    than measured, so it is an unknown of the deployed system rather than a
    property of it.

    Yaw is carried by the feed-forward rate, so ``yaw_gain`` only removes the
    residual: the number it really buys is margin against this error.  A gain
    error multiplies the loop gain directly, and the loop is delay limited
    (120 ms dead time + 80 ms lag + the 20 Hz hold), so past some gain it
    self-oscillates.  That is what a bystander sees as the base wobbling in
    yaw while it translates.

    Reported per cell: yaw peak-to-peak against the reference, and the share
    of ticks where the output angular-acceleration limiter was saturated.
    Saturation is the signature of a demand the base cannot execute.
    """
    print('Plant: %.0f ms dead time + %.0f ms lag, control at %.0f Hz, '
          'measurement-wheel noise 0.02 m/s / 0.04 rad/s.'
          % (DEAD_TIME * 1000, TAU * 1000, CONTROL_HZ))
    print('Cell: yaw peak-to-peak [deg] / angular-accel limiter saturated '
          '[% of ticks].\n')
    for name, start, goal, goal_yaw, options in MARGIN_LEGS:
        print('  %s' % name)
        print('      %-10s' % 'yaw_gain'
              + ''.join('%14s' % ('gain x%.1f' % g) for g in MARGIN_GAINS))
        for yaw_gain in MARGIN_YAW_GAINS:
            cells = []
            for gain in MARGIN_GAINS:
                history = simulate(
                    start, goal, goal_yaw, fixes=FIXES, seed=3,
                    plant_gain=gain, yaw_gain=yaw_gain, odom_noise_mps=0.02,
                    odom_noise_radps=0.04, **options)
                window = (history[:, 0] > 1.0) & (history[:, 0] < 9.0)
                yaw = np.degrees(history[window, 1] - history[window, 3])
                cells.append('%9.1f/%3.0f%%' % (
                    yaw.max() - yaw.min(), 100 * history[window, 8].mean()))
            print('      %-10.1f' % yaw_gain + ''.join(
                '%14s' % cell for cell in cells))
        print()
    print('  Deployed yaw_gain is 1.0. This table isolates yaw feedback;')
    print('  use test_hardware_tracking.py for the deployed control tick,')
    print('  longer delays, wheel noise, and sustained settling after arrival.')
    print('  Measure the real gain with: python3 run.py '
          'check-drive-directions --accept-motor-risk')


if __name__ == '__main__':
    if '--margin' in sys.argv:
        print('YAW LOOP MARGIN AGAINST DRIVEBASE GAIN ERROR\n')
        margin()
    elif '--terminal' in sys.argv:
        print('COST OF THE LOW-SPEED FINAL APPROACH\n')
        terminal()
    elif '--rotation' in sys.argv:
        print('ROTATION AND LEG TIME BY YAW MODE\n')
        rotation()
    elif '--fixed' in sys.argv:
        print('WITH FIXES (path smoothing + reference at t+lag + velocity '
              'filter)\n')
        report(('smooth-path', 'reference-lead', 'velocity-filter'))
        sweep(('smooth-path', 'reference-lead', 'velocity-filter'))
    else:
        print('AS DEPLOYED (optimize_yaw=false)\n')
        report(())
        sweep()
