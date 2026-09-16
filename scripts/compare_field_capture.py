#!/usr/bin/env python3
"""Offline paired control-tick replay. No ROS nodes, publishers or motors.

Field logs establish routes, .74 operator scale and gain warnings, but do
not contain synchronized samples. These are uncertainty sweeps, not fitted
hardware trajectories or measured speed improvements.
"""
import json
import math
import sys
import ast
import yaml
from collections import deque
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / 'ros2_ws/src/omni_autonomy_next'
sys.path[:0] = [str(PACKAGE), str(PACKAGE / 'test')]
from test_smooth_arrival import make_node, TRACKER_DEFAULTS
from omni_autonomy_next import trajectory_tracker_node as tracker
from omni_autonomy_next.runtime_guard import RuntimeGuard, MotionLimits, GuardHealth


def original_guard():
    source = ROOT/'evidence/runtime_guard_before.py'
    tree = ast.parse(source.read_text())
    function = next(n for n in ast.walk(tree)
                    if isinstance(n, ast.FunctionDef) and n.name == '_dynamic_limit')
    namespace = dict(np=np, MotionLimits=MotionLimits)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), namespace)
    return type('OriginalGuard', (RuntimeGuard,), {'_dynamic_limit': namespace['_dynamic_limit']})


def make_guard(cls):
    cfg = yaml.safe_load((PACKAGE/'config/runtime.yaml').read_text())['runtime_guard']['ros__parameters']
    drive = yaml.safe_load((PACKAGE/'config/robot.yaml').read_text())['robot']['drivetrain']
    hard = MotionLimits(*(cfg['hard_max_'+k] for k in (
        'linear_speed', 'lateral_speed', 'angular_speed', 'linear_acceleration',
        'angular_acceleration', 'linear_jerk', 'angular_jerk')))
    return cls(profiles={k: MotionLimits(**v) for k, v in json.loads(cfg['profiles_json']).items()},
        hard_limits=hard, default_profile='balanced', command_timeout_sec=.25,
        red_zone_speed_scale=.7, wheel_radius=drive['wheel_radius'],
        wheel_positions=drive['wheel_positions'], wheel_drive_angles_rad=np.radians(drive['wheel_drive_angles_deg']),
        wheel_signs=drive['wheel_signs'], max_wheel_speed=drive['max_wheel_speed'],
        profile_max_wheel_speeds=drive.get('profile_max_wheel_speeds', {}),
        translation_budget_share=drive['translation_budget_share'])


def replay(cls, gain, delay, tau, direction, scale, distance=2.):
    goal = np.array([distance*math.cos(direction), distance*math.sin(direction), 0.])
    node = make_node(goal=goal)
    node.speed_scale = scale
    actual = np.zeros(3)
    truth = node.pose.copy()
    pipeline = deque(np.zeros(3) for _ in range(round(delay/.01)))
    rng = np.random.default_rng(13)
    records = []
    guard = make_guard(cls)
    safe_commands = []
    for step in range(1600):
        now = step*.01
        with patch.object(tracker.time, 'monotonic', return_value=now):
            node.pose = truth.copy()
            node.pose[2] += rng.normal(0., .0025)
            node.velocity_stamp = node.pose_stamp = now
            if step % 2 == 0:
                sample = actual+rng.normal(0., [.02, .02, .04])
                node.velocity += (1.-math.exp(-.02/TRACKER_DEFAULTS['velocity_filter_sec']))*(sample-node.velocity)
            if step % 100 == 0:
                tracker.TrajectoryTracker._build_trajectory(node, np.array([node.pose[:2], goal[:2]]), goal[2])
            if step % 5 == 0:
                tracker.TrajectoryTracker._tick(node)
        safe = guard.step(node.command, now_sec=now, command_age_sec=0.,
            health=GuardHealth(True, False, True, True, True),
            profile='balanced', user_scale=scale, red_zone=False)
        safe_commands.append(safe.velocity)
        pipeline.append(np.array(safe.velocity))
        actual += (pipeline.popleft()*gain-actual)*(.01/tau)
        c, s = math.cos(truth[2]), math.sin(truth[2])
        truth += .01*np.array([actual[0]*c-actual[1]*s, actual[0]*s+actual[1]*c, actual[2]])
        records.append([now, np.linalg.norm(truth[:2]-goal[:2]), np.linalg.norm(actual[:2]),
                        float(np.dot(truth[:2]-goal[:2], goal[:2])/distance)])
    data = np.asarray(records)
    # Report sustained arrival, not a transient crossing of the goal band.
    settled = (data[:, 1] <= .015) & (data[:, 2] <= .025)
    arrival = None
    for i in range(len(settled)-50):
        if np.all(settled[i:i+50]):
            arrival = round(float(data[i, 0]), 3)
            break
    commands = np.asarray(node.commands)
    accel = np.diff(commands[:, :2], axis=0)/.05
    safe_accel = np.diff(np.asarray(safe_commands)[:, :2], axis=0)/.01
    return dict(arrival_sec=arrival, tail_error_m=float(np.max(data[-300:, 1])),
                overshoot_m=float(max(0., np.max(data[:, 3]))),
                peak_command_accel=float(np.max(np.linalg.norm(accel, axis=1))),
                safe_accel_variation=float(np.sum(np.linalg.norm(np.diff(safe_accel, axis=0), axis=1))),
                command_accel_variation=float(np.sum(np.linalg.norm(np.diff(accel, axis=0), axis=1))))


def main():
    candidate = RuntimeGuard
    original = original_guard()
    records = []
    for gain, delay, tau in [(1., .12, .08), (1.4, .12, .08),
                             (2.8, .2, .12), (2.8, .3, .15), (3.2, .2, .12)]:
        for direction in (0., math.pi/4, math.pi/2):
            for scale in (.74, 1.):
                before = replay(original, gain, delay, tau, direction, scale)
                after = replay(candidate, gain, delay, tau, direction, scale)
                records.append(dict(gain=gain, delay=delay, tau=tau,
                    direction=direction, scale=scale, before=before, after=after))
    steps = []
    for direction in (0., math.pi/4, math.pi/2):
        pair = {}
        for label, cls in [('before', original), ('after', candidate)]:
            guard = make_guard(cls)
            velocities = [np.zeros(3)]
            for i in range(150):
                target = (.5*math.cos(direction), .5*math.sin(direction), .6)
                velocities.append(guard.step(target, now_sec=i*.01, command_age_sec=0.,
                    health=GuardHealth(True, False, True, True, True),
                    profile='balanced', user_scale=.74, red_zone=False).velocity)
            accelerations = np.diff(velocities, axis=0)/.01
            jerks = np.diff(accelerations, axis=0)/.01
            pair[label] = dict(peak_linear_jerk=float(np.max(np.linalg.norm(jerks[:, :2],axis=1))),
                peak_angular_jerk=float(np.max(np.abs(jerks[:, 2]))))
        steps.append(dict(direction=direction, **pair))
    report = dict(kind='offline uncertain-plant comparison, not measured hardware', cases=records, steps=steps)
    # Gate on every case, not only the fastest direction or nominal drive.
    report['accepted'] = all(
        r['after']['arrival_sec'] is not None
        and r['before']['arrival_sec'] is not None
        and r['after']['arrival_sec'] <= r['before']['arrival_sec'] + .031
        and r['after']['tail_error_m'] < .04
        and r['after']['safe_accel_variation'] < r['before']['safe_accel_variation']
        and r['after']['overshoot_m'] <= r['before']['overshoot_m'] + .01
        for r in records)
    path = ROOT/'docs/field_capture_validation.json'
    path.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    if not report['accepted']:
        raise SystemExit('Paired replay acceptance failed')


if __name__ == '__main__':
    main()
