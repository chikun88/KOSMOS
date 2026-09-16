#!/usr/bin/env python3
"""Check the deployed trajectory math without ROS or motor communication.

The gateway host has no ROS. Extract only the numerical definitions from the
actual tracker source so this check does not duplicate the implementation.
"""
import ast
import json
import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / 'ros2_ws' / 'src' / 'omni_autonomy_next'
sys.path.insert(0, str(PACKAGE))
from omni_autonomy_next import omni_yaw


def main():
    path = PACKAGE / 'omni_autonomy_next' / 'trajectory_tracker_node.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    names = {'resample', 'path_tangents', 'menger_curvature', 'Trajectory'}
    pure = ast.Module(body=[node for node in tree.body
                           if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                           and node.name in names], type_ignores=[])
    namespace = dict(vars(omni_yaw), Optional=Optional)
    exec(compile(pure, str(path), 'exec'), namespace)
    results = []
    for length in (.005, .02, .04, .06, 2.):
        points = namespace['resample'](np.array([[0., 0.], [length, 0.]]), .05)
        plan = namespace['Trajectory'](
            points, np.zeros(len(points)), np.full(len(points), .78),
            acceleration=.85, lateral_acceleration=1.2, entry_speed=0.)
        if length < .1:
            assert abs(plan.duration - 2 * math.sqrt(length / .85)) < 1.e-8
        assert abs(plan.sample(plan.duration)[4] - length) < 1.e-8
        assert np.linalg.norm(plan.sample(plan.duration)[1]) < 1.e-8
        inverse_error = max(abs(plan.time_at_arclength(plan.sample(t)[4]) - t)
                            for t in np.linspace(0., plan.duration, 51))
        assert inverse_error < 1.e-7
        for fraction in (.03, .12, .88, .97):
            t, dt = fraction * plan.duration, 1.e-6
            derivative = (plan.sample(t + dt)[0] - plan.sample(t - dt)[0]) / (2 * dt)
            assert np.linalg.norm(derivative - plan.sample(t)[1]) < 1.e-6
        assert abs(plan.project(np.array([length * .37, .01])) - length * .37) < 1.e-9
        results.append(dict(length_m=length, duration_s=round(plan.duration, 6),
                            peak_mps=round(float(max(plan.speed)), 6)))
    print(json.dumps(dict(result='PASS', source=str(path), cases=results), indent=2))


if __name__ == '__main__':
    main()
