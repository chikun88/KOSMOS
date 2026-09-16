#!/usr/bin/env python3
"""Replay the 243 recorded test scenarios with the current cruise configuration.

Offline tracker/guard/calibrated UART model; no publishers or hardware I/O.
This does not model CAD obstacles, localization errors, or motor capability.
"""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json

import yaml

from compare_field_response import ROOT, replay
from omni_autonomy_next.config import calibrated_tracking_parameters


def evaluate(case):
    result = replay(case['gains'], [.38, .34], case['delay'], case['tau'],
        case['direction'], case['yaw'], case['scale'], profile=case['profile'],
        distance=case['distance'], duration_sec=30. if case['distance'] > 4. else 20.,
        profile_limits=case['limits'], overrides=case['tuning'])
    passed = (result['arrival_sec'] is not None
        and result['tail_position_error_m'] < .015
        and result['tail_yaw_error_rad'] < .015
        and result['cross_track_peak_m'] < .06
        and result['physical_acceleration_peak_m_s2'] < 1.3
        and result['wheel_command_peak'] <= 8000)
    return dict(case=case, result=result, passed=passed)


def main():
    config = ROOT/'ros2_ws/src/omni_autonomy_next/config'
    package = config.parent/'omni_autonomy_next'
    paths = [config/'runtime.yaml', config/'robot.yaml', config/'routes.yaml',
        config/'nav2_next.yaml', ROOT/'docs/two_mps_replay_20260915.json',
        ROOT/'scripts/compare_field_response.py', ROOT/'scripts/check_cruise_floor.py',
        *[package/name for name in ('trajectory_tracker_node.py', 'runtime_guard.py',
          'motor_udp_bridge_node.py', 'robomas_uart.py')]]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    runtime = yaml.safe_load(paths[0].read_text())['runtime_guard']['ros__parameters']
    profiles = json.loads(runtime['profiles_json'])
    robot = yaml.safe_load(paths[1].read_text())['robot']
    tuning = calibrated_tracking_parameters(robot, hardware=True, motion_mode='simultaneous')
    jobs = []
    for stored in json.loads(paths[4].read_text())['cases']:
        case = {k: v for k, v in stored.items() if k not in ('before', 'after')}
        case['limits'] = [profiles[case['profile']][k] for k in ('linear', 'lateral', 'angular')]
        case['tuning'] = tuning
        jobs.append(case)
    rows = []
    with ProcessPoolExecutor(max_workers=3) as pool:
        for row in pool.map(evaluate, jobs):
            rows.append(row)
            print(len(rows), '/', len(jobs), row['passed'], flush=True)
    for p in paths:
        assert hashlib.sha256(p.read_bytes()).hexdigest() == hashes[str(p.relative_to(ROOT))]
    passed = all(row['passed'] for row in rows)
    (ROOT/'docs/cruise_floor_replay_20260915.json').write_text(json.dumps(dict(
        limitations=__doc__, sources=hashes, accepted=passed, cases=rows), indent=2)+'\n')
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
