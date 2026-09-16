#!/usr/bin/env python3
"""Paired offline replay of actual tracker/guard code; no ROS publishers.

Uncertain gain and delay are sensitivity tests, not hardware measurements.
The straight routes here do not model CAD obstacles or Collision Monitor.
"""
import argparse
import ast
from collections import deque
import json
import math
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / 'ros2_ws/src/omni_autonomy_next'
sys.path[:0] = [str(PACKAGE), str(PACKAGE/'test')]
from test_smooth_arrival import make_node, TRACKER_DEFAULTS
from omni_autonomy_next import trajectory_tracker_node as tracker
from omni_autonomy_next.runtime_guard import RuntimeGuard, GuardHealth
from compare_field_capture import make_guard


def baseline_builder():
    source = ROOT/'evidence/simultaneous_20260908/trajectory_tracker_before.py'
    tree = ast.parse(source.read_text())
    method = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == '_build_trajectory')
    namespace = vars(tracker).copy()
    trajectory = next(n for n in tree.body
                      if isinstance(n, ast.ClassDef) and n.name == 'Trajectory')
    exec(compile(ast.Module(body=[trajectory], type_ignores=[]), str(source), 'exec'), namespace)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), namespace)
    return namespace['_build_trajectory']


def replay(builder, length, angle, direction, gain, delay, tau, scale):
    goal = np.array([length*math.cos(direction), length*math.sin(direction), angle])
    node = make_node(goal=goal)
    node.speed_scale = scale
    actual = np.zeros(3)
    truth = node.pose.copy()
    pipeline = deque(np.zeros(3) for _ in range(round(delay/.01)))
    rng = np.random.default_rng(13)
    guard = make_guard(RuntimeGuard)
    records, safe_commands = [], []
    duration = None
    for step in range(1800):
        now = step*.01
        with patch.object(tracker.time, 'monotonic', return_value=now):
            node.pose = truth.copy()
            node.pose[2] += rng.normal(0., .0025)
            node.velocity_stamp = node.pose_stamp = now
            if step % 2 == 0:
                sample = actual+rng.normal(0., [.02, .02, .04])
                node.velocity += (1.-math.exp(-.02/TRACKER_DEFAULTS['velocity_filter_sec']))*(sample-node.velocity)
            if step % 100 == 0:
                builder(node, np.array([node.pose[:2], goal[:2]]), goal[2])
                if duration is None:
                    duration = node.trajectory.duration
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
        records.append([now, np.linalg.norm(truth[:2]-goal[:2]),
                        abs(tracker.wrap(truth[2]-angle)), np.linalg.norm(actual[:2]), abs(actual[2])])
    data = np.asarray(records)
    settled = (data[:, 1] <= .015) & (data[:, 2] <= .02) & (data[:, 3] <= .025) & (data[:, 4] <= .04)
    arrival = next((round(float(data[i, 0]), 3) for i in range(len(settled)-49)
                    if np.all(settled[i:i+50])), None)
    end = len(data) if arrival is None else min(len(data), round((arrival+.5)/.01))
    acceleration = np.diff(np.asarray(safe_commands)[:end], axis=0)/.01
    return dict(arrival_sec=arrival, plan_duration_sec=duration,
        tail_position_error_m=float(np.max(data[-300:, 1])),
        tail_yaw_error_rad=float(np.max(data[-300:, 2])),
        linear_accel_variation=float(np.sum(np.linalg.norm(np.diff(acceleration[:, :2], axis=0), axis=1))),
        yaw_accel_variation=float(np.sum(np.abs(np.diff(acceleration[:, 2])))),
        peak_yaw_accel=float(np.max(np.abs(acceleration[:, 2]))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--output', default=str(ROOT/'evidence/simultaneous_comparison.json'))
    args = parser.parse_args()
    builders = [('before', baseline_builder()), ('after', tracker.TrajectoryTracker._build_trajectory)]
    scenarios = [(2., math.pi/2, 0.), (3., -math.pi/2, math.pi/4),
                 (.3, math.pi/2, 0.), (2., 0., math.pi/2)]
    conditions = [(1., .12, .08, 1.)]
    if not args.quick:
        scenarios += [(.04, -math.pi, 0.), (2., math.pi, math.pi/2)]
        conditions += [(1.4, .12, .08, .74), (2.8, .2, .12, .74),
                       (3.2, .2, .12, 1.), (2.8, .3, .15, .74)]
    rows = []
    for length, angle, direction in scenarios:
        for gain, delay, tau, scale in conditions:
            row = dict(length=length, angle=angle, direction=direction,
                       gain=gain, delay=delay, tau=tau, scale=scale)
            for label, builder in builders:
                row[label] = replay(builder, length, angle, direction, gain, delay, tau, scale)
            rows.append(row)
            print(json.dumps(row), flush=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(dict(scope=__doc__, scenarios=rows), indent=2))


if __name__ == '__main__':
    main()
