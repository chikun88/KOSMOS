#!/usr/bin/env python3
"""Read and test the actual trajectory source without ROS or motor communication."""
import ast
import json
import math
import sys
from pathlib import Path
from typing import Optional
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / 'ros2_ws/src/omni_autonomy_next'
sys.path.insert(0, str(PACKAGE))
from omni_autonomy_next import omni_yaw

path = PACKAGE / 'omni_autonomy_next/trajectory_tracker_node.py'
names = {'resample', 'path_tangents', 'menger_curvature', 'yaw_derivative', 'Trajectory'}
tree = ast.parse(path.read_text(encoding='utf-8'))
pure = ast.Module(body=[n for n in tree.body if getattr(n, 'name', '') in names], type_ignores=[])
namespace = dict(vars(omni_yaw), Optional=Optional)
exec(compile(pure, str(path), 'exec'), namespace)
records = []
for length in (.02, .04, .1, .3, 1., 3.):
    for angle in (math.pi / 2, -math.pi, .2):
        points = namespace['resample'](np.array([[0., 0.], [length, 0.]]), .05)
        yaws = angle * np.minimum(points[:, 0] / max(length - .1, length / 2), 1.)
        trajectory = namespace['Trajectory'](points, yaws, np.full(len(points), .78),
            acceleration=.85, lateral_acceleration=1.2, entry_speed=0.,
            angular_speed=1.3, angular_acceleration=2.)
        times = np.linspace(0., trajectory.duration, 2001)
        rates = np.array([trajectory.sample(t)[3] for t in times])
        peak = float(np.max(np.abs(rates)))
        acceleration = float(np.max(np.abs(np.diff(rates) / np.diff(times))))
        assert peak <= 1.3 + 1.e-8, (length, angle, peak)
        assert acceleration <= 2. + 1.e-5, (length, angle, acceleration)
        records.append(dict(length_m=length, turn_rad=angle,
            peak_radps=round(peak, 6), peak_radps2=round(acceleration, 6)))
for count in (1, 2):
    plan = namespace['Trajectory'](np.zeros((count, 2)), np.full(count, math.pi),
        np.full(count, .78), acceleration=.85, lateral_acceleration=1.2, entry_speed=0.)
    assert plan.project(np.zeros(2)) == plan.time_at_arclength(0.) == 0.
    assert np.linalg.norm(plan.sample(1.)[1]) == 0.
    assert abs(plan.sample(1.)[2]) == math.pi
print(json.dumps(dict(result='PASS', source=str(path), cases=records), indent=2))
