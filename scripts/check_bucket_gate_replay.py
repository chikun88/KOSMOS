#!/usr/bin/env python3
"""Closed-loop fixed-gate replay using recorded Nav2 path geometry; no robot I/O.

The baseline uses 16 cm gate disks and the previous uniform 0.20 m/s slowdown.
The candidate uses the native BT's bounded outer passage and turn-only speed
planning. Replanning preserves recorded detours, permitting direct connectors
only after a full-body CAD sweep. This does not run Smac, TF, localization or
the live collision monitor.
"""
import argparse
import ctypes
from collections import deque
import json
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import yaml

from compare_field_response import ROOT, make_node, make_guard, transmitted_response
from omni_autonomy_next import trajectory_tracker_node as tracker
from omni_autonomy_next.config import calibrated_tracking_parameters
from omni_autonomy_next.config import load_robot
from omni_autonomy_next.fixed_gate_speed import FixedGateSpeed
from omni_autonomy_next.runtime_guard import RuntimeGuard, GuardHealth
from omni_autonomy_next.staged_heading import remaining_path
from omni_autonomy_next.rl_residual import CadClearanceModel

CONFIG = ROOT/'ros2_ws/src/omni_autonomy_next/config'


def replay(case, enabled, delay=.2, tau=.12, mirror=False, gain_scale=1., mode=None,
           trace_output=None, gate_tuning=None, legacy_tracker_budget=False,
           response_gains=None):
    mode = mode or ('fast' if enabled else 'original')
    points = np.asarray(case['points'], float).copy()
    gates = np.asarray(case['gates'], float).copy()
    goal = np.asarray(case['goal'], float).copy()
    yaw = case['yaw']
    if mirror:
        points[:, 0] *= -1.
        gates[:, 0] *= -1.
        goal[0] *= -1.
        goal[2] = tracker.wrap(math.pi-goal[2])
        yaw = tracker.wrap(math.pi-yaw)
    # Smac ends at the goal's grid cell; the production tracker snaps to the
    # exact saved target. Match only gates still present in this first plan.
    indices = []
    selected = []
    for gate in gates:
        i = int(np.argmin(np.linalg.norm(points-gate, axis=1)))
        if np.linalg.norm(points[i]-gate) < .05 and (not indices or i > indices[-1]):
            indices.append(i)
            selected.append(gate)
    node = make_node(pose=(*points[0], yaw), goal=goal)
    node.clearance = CadClearanceModel.from_yaml(CONFIG/'field_planning.yaml', CONFIG/'competition_footprints.yaml')
    node.fixed_gate_speed = FixedGateSpeed.from_yaml(CONFIG/'routes.yaml') if mode != 'original' else None
    if gate_tuning is not None and node.fixed_gate_speed is not None:
        for name, value in gate_tuning.items():
            if name not in {'speed', 'transit_speed', 'entry_hold_m', 'exit_hold_m'}:
                raise ValueError(f'unsupported gate tuning: {name}')
            setattr(node.fixed_gate_speed, name, float(value))
    routes = yaml.safe_load((CONFIG/'routes.yaml').read_text())
    transit = routes['fixed_bucket_transit']['waypoints']
    if mode == 'slow':
        node.fixed_gate_speed.corner_only = False
        extra = [[sign*p['x'], p['y']] for p in transit for sign in (-1., 1.)]
        node.fixed_gate_speed.gates = np.vstack((node.fixed_gate_speed.gates, extra))
    passage = None
    if mode == 'fast':
        from ament_index_python.packages import get_package_prefix
        library = ctypes.CDLL(str(Path(get_package_prefix('omni_route_bt'))/
                                  'lib/libomni_remove_passed_bucket_goals_bt_node.so'))
        passage = library.omni_outer_gate_passed
        passage.argtypes = [ctypes.c_double]*7
        passage.restype = ctypes.c_bool
    outer_gates = {(sign*p['x'], p['y']) for p in transit if p.get('outer_y_passage')
                   for sign in (-1., 1.)}
    robot = yaml.safe_load((CONFIG/'robot.yaml').read_text())['robot']
    tuning = calibrated_tracking_parameters(robot, hardware=True, motion_mode='simultaneous')
    original_parameter = node.get_parameter
    node.get_parameter = lambda name: (SimpleNamespace(value=tuning[name])
        if name in tuning else original_parameter(name))
    guard = make_guard(RuntimeGuard)
    profile = guard.profiles['sprint']
    node.speed_limit, node.lateral_limit, node.yaw_limit = (
        profile.linear, profile.lateral, profile.angular)
    node.profile_name = 'sprint'
    drive = load_robot(str(CONFIG/'robot.yaml'))['drivetrain']
    node.profiles = {name: dict(linear=p.linear, lateral=p.lateral, angular=p.angular, linear_accel=p.linear_accel)
                     for name, p in guard.profiles.items()}
    node.default_max_wheel_speed = drive['max_wheel_speed']
    node.profile_max_wheel_speeds = ({} if legacy_tracker_budget else
                                    drive['profile_max_wheel_speeds'])
    node.default_acceleration = node.acceleration
    node.deceleration = node.acceleration
    node.profile_linear_accelerations = drive.get('profile_linear_accelerations', {})
    tracker.TrajectoryTracker._apply_profile(node, 'sprint')
    fitted = json.loads((ROOT/'docs/field_response_fit_20260914.json').read_text())
    gains = np.array([axis['gain'] for axis in fitted['runs'][-1]['fit']])*gain_scale
    if response_gains is not None:
        gains = np.asarray(response_gains, dtype=float)*gain_scale
    drive = robot['drivetrain']
    calibration = np.array([drive['linear_command_scale']]*2 + [drive['angular_command_scale']])
    truth, actual = node.pose.copy(), np.zeros(3)
    model = CadClearanceModel.from_yaml(CONFIG/'field_planning.yaml', CONFIG/'competition_footprints.yaml')
    pipeline = deque(np.zeros(3) for _ in range(round(delay/.01)))
    gate_index = 0
    passed = []
    rows = []
    settled_at = None
    arrived = None
    rng = np.random.default_rng(13)
    for step in range(6000):
        now = step*.01
        with patch.object(tracker.time, 'monotonic', return_value=now):
            node.pose = truth.copy()
            node.pose[:2] += rng.normal(0., .005, 2)
            node.pose[2] += rng.normal(0., .0025)
            node.pose_stamp = node.velocity_stamp = now
            node.velocity += (1.-math.exp(-.01/.06))*(actual-node.velocity)
            if gate_index < len(selected):
                gate = selected[gate_index]
                reached = np.linalg.norm(node.pose[:2]-gate) <= .16
                if passage is not None and tuple(gate) in outer_gates:
                    next_y = selected[gate_index+1][1] if gate_index+1 < len(selected) else goal[1]
                    reached |= passage(*node.pose, *gate, next_y, .45)
                if reached:
                    passed.append(now)
                    gate_index += 1
            if step % 100 == 0:
                first = 0 if gate_index == 0 else indices[gate_index-1]
                if gate_index < len(indices):
                    end = indices[gate_index]
                    incoming = remaining_path(points[first:end+1], node.pose[:2])
                    # Once an outer passage is complete, a real planner can
                    # connect directly to the next gate. Reprojecting onto the
                    # old leg would invent a return to its already passed start.
                    # Use that connector only when a dense full-body CAD sweep
                    # is clear; otherwise retain the recorded obstacle detour.
                    count = max(2, int(np.linalg.norm(points[end]-node.pose[:2])/.01)+2)
                    direct = np.linspace(node.pose[:2], points[end], count)
                    margin = model.clearance_over_poses(direct, np.full(count, node.pose[2]), cap=.08)
                    if np.min(margin) > .025 + .005 + model.radius*.02:
                        incoming = direct
                    plan = np.vstack((incoming, points[end+1:]))
                else:
                    plan = remaining_path(points[first:], node.pose[:2])
                tracker.TrajectoryTracker._build_trajectory(node, plan, goal[2])
                if mode == 'slow' and node.trajectory is not None:
                    # Previous version limited the reference only; do not
                    # quietly give the baseline the new full-command bound.
                    node.trajectory.gate_turn_limits = None
            if step % 5 == 0:
                tracker.TrajectoryTracker._tick(node)
        safe = guard.step(node.command, now_sec=now, command_age_sec=0.,
            health=GuardHealth(True, False, True, True, True), profile='sprint',
            user_scale=1., red_zone=False)
        _, response, _ = transmitted_response(np.asarray(safe.velocity)*calibration, gains)
        pipeline.append(response)
        actual += (pipeline.popleft()-actual)*(.01/tau)
        c, s = math.cos(truth[2]), math.sin(truth[2])
        truth += .01*np.array([actual[0]*c-actual[1]*s,
                              actual[0]*s+actual[1]*c, actual[2]])
        rows.append([now, *truth, *actual])
        settled = (np.linalg.norm(truth[:2]-goal[:2]) <= .015
                   and abs(tracker.wrap(truth[2]-goal[2])) <= .015
                   and np.linalg.norm(actual[:2]) <= .025 and abs(actual[2]) <= .025)
        settled_at = (now if settled_at is None else settled_at) if settled else None
        if settled_at is not None and now-settled_at >= .5 and gate_index == len(selected):
            arrived = round(settled_at, 2)
            break
    data = np.asarray(rows)
    if trace_output is not None:
        np.save(trace_output, data)
    direction = np.sign(goal[1]-points[0, 1])
    backtrack = float(np.maximum(0., -direction*np.diff(data[:, 2])).sum())
    margins = model.clearance_over_poses(data[::5, 1:3], data[::5, 3])
    speed = np.linalg.norm(data[:, 4:6], axis=1)
    middle = speed[(np.abs(data[:, 1]) > 1.45) & (data[:, 2] > -.9) & (data[:, 2] < -.2)]
    return dict(arrival_sec=arrived, passed_gates=len(passed), required_gates=len(selected),
                gate_times_sec=passed, backward_y_m=backtrack,
                min_cad_clearance_m=float(min(margins)),
                min_clearance_pose=data[::5][int(np.argmin(margins)), :4].tolist(),
                peak_speed_m_s=float(max(speed)),
                transit_speed_median_m_s=float(np.median(middle)) if len(middle) else None,
                final_error_m=float(np.linalg.norm(truth[:2]-goal[:2])),
                states=sorted(set(node.statuses)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'docs/bucket_fast_replay_20260915.json')
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()
    fixture = json.loads((ROOT/'docs/bucket_gate_fixture_20260915.json').read_text())
    results = []
    for index, case in enumerate(fixture['cases']):
        conditions = [(False, .2, .12, 1.)] if args.quick else [
            (mirror, delay, tau, gain) for mirror in (False, True)
            for delay, tau, gain in ((.1, .08, 1.), (.2, .12, 1.), (.3, .15, 1.1))]
        for mirror, delay, tau, gain in conditions:
            result = dict(case=index, mirror=mirror, delay_sec=delay, tau_sec=tau, gain_scale=gain)
            for label, enabled in [('before', False), ('after', True)]:
                result[label] = replay(case, enabled, delay, tau, mirror, gain,
                                       mode='fast' if enabled else 'slow')
            results.append(result)
            print(json.dumps(result), flush=True)
    accepted = all(r['after']['arrival_sec'] is not None
                   and r['after']['backward_y_m'] < .10
                   and r['after']['min_cad_clearance_m'] > 0.
                   and r['before']['arrival_sec'] is not None
                   and r['after']['arrival_sec'] < r['before']['arrival_sec']
                   and r['after']['transit_speed_median_m_s'] > 1.2*r['before']['transit_speed_median_m_s']
                   for r in results)
    args.output.write_text(json.dumps(dict(session=fixture['session'], accepted=accepted,
        limitations=__doc__, cases=results), indent=2)+'\n')
    if not accepted:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
