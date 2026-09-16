#!/usr/bin/env python3
"""Empty 30 m corridor scenarios; not physical 4 m/s validation."""
import hashlib
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml
import compare_field_response as plant
from omni_autonomy_next.config import calibrated_tracking_parameters
from omni_autonomy_next.rl_residual import CadClearanceModel


def evaluate(job):
    direction, yaw, delay, tau, gains = job
    robot = yaml.safe_load((plant.ROOT/'ros2_ws/src/omni_autonomy_next/config/robot.yaml').read_text())['robot']
    make_node = plant.make_node
    original_clearance = plant.tracker.sprint_turn_clearance
    results = {}
    for label in ('before', 'after'):
        rotating_feedback = []
        def node_factory(*args, **kwargs):
            node = make_node(*args, **kwargs)
            node.clearance = CadClearanceModel([[[-200., -200.], [200., -200.]]], robot['footprint'])
            def record(heading, measured, vx, vy):
                if abs(measured[2]) >= .05:
                    rotating_feedback.append(float(np.linalg.norm(measured[:2])))
            node._track_flow = record
            return node
        policy = original_clearance if label == 'after' else lambda p, y, m, **kw: np.zeros(len(p), dtype=bool)
        with patch.object(plant, 'make_node', node_factory), patch.object(plant.tracker, 'sprint_turn_clearance', policy):
            result = plant.replay(gains, [.38, .34], delay, tau, direction, yaw, 1.,
                overrides=calibrated_tracking_parameters(robot, hardware=True, motion_mode='simultaneous'),
                distance=30., profile='sprint', duration_sec=60.)
        result['filtered_feedback_peak_while_rotating_m_s'] = max(rotating_feedback, default=0.)
        results[label] = result
    return dict(direction_rad=direction, goal_yaw_rad=yaw, delay_sec=delay, tau_sec=tau, **results)


def main():
    root = plant.ROOT
    audit = root/'docs/latest_four_mps_audit_20260916.json'
    gains = [r['gain'] for r in json.loads(audit.read_text())['runs'][0]['exploratory_fit']]
    files = [Path(__file__), audit, root/'scripts/compare_field_response.py', root/'scripts/compare_field_capture.py',
             root/'ros2_ws/src/omni_autonomy_next/test/test_smooth_arrival.py']
    package = root/'ros2_ws/src/omni_autonomy_next'
    files += list((package/'omni_autonomy_next').glob('*.py'))+list((package/'config').glob('*.yaml'))
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    jobs = [(d, y, delay, tau, gains) for d, y in ((0., math.pi/2), (math.pi/4, -math.pi/2),
              (math.pi/2, math.pi/2), (-math.pi/4, -math.pi/2)) for delay, tau in ((.2, .12), (.3, .15))]
    results = []
    with ProcessPoolExecutor(max_workers=2) as pool:
        for result in pool.map(evaluate, jobs):
            results.append(result)
            print(len(results), [(k, result[k]['peak_speed_m_s'], result[k]['arrival_sec'],
                                 result[k]['cross_track_peak_m']) for k in ('before', 'after')], flush=True)
    for p in files:
        if hashlib.sha256(p.read_bytes()).hexdigest() != hashes[str(p)]:
            raise RuntimeError(f'Source changed: {p}')
    (root/'docs/open_sprint_turning_replay_20260916.json').write_text(json.dumps(dict(
        physical_target_verified=False, sources=hashes,
        limitations='Hypothetical empty 30 m corridor, production tracker/guard/UART. Low-speed gains '
        'extrapolated with assumed delay. No live collision monitor, localization faults, load or slip. '
        'Joint rotation/speed metric uses filtered simulated feedback with abs(yaw_rate)>=0.05 rad/s.',
        cases=results), indent=2)+'\n')


if __name__ == '__main__':
    main()
