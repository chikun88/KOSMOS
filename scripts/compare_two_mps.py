#!/usr/bin/env python3
"""Paired offline control replay for 2 m/s cruise speed tuning.

No publishers or hardware I/O. Gains come from completed physical recordings;
delay/tau are uncertainty scenarios. This does not measure motor capability.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math

import numpy as np
import yaml

from compare_field_response import ROOT, replay
from omni_autonomy_next.config import calibrated_tracking_parameters

BASELINE = {'precision': [.55, .50, .75], 'balanced': [1.05, 1.05, 1.3],
            'sprint': [1.10, 1.10, 1.6]}
BASELINE_TUNING = {'position_gain': 1.6, 'yaw_gain': 1.6, 'feedback_delay_sec': .32}


def evaluate(case):
    args = (case['gains'], [.38, .34], case['delay'], case['tau'],
            case['direction'], case['yaw'], case['scale'])
    kwargs = dict(profile=case['profile'], distance=case['distance'],
                  duration_sec=30. if case['distance'] > 4. else 20.)
    case['before'] = replay(*args, **kwargs, profile_limits=BASELINE[case['profile']],
                            overrides=BASELINE_TUNING, bridge_limits=(1.1, 1.8), wheel_limit=15.709120382)
    case['after'] = replay(*args, **kwargs, profile_limits=case['limits'], overrides=case['tuning'])
    return case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--output', type=str, default='docs/two_mps_replay_20260915.json')
    args = parser.parse_args()
    config = ROOT / 'ros2_ws/src/omni_autonomy_next/config'
    paths = [ROOT / 'docs/field_response_fit_20260914.json', config / 'robot.yaml',
             config / 'runtime.yaml', config / 'nav2_next.yaml',
             ROOT / 'scripts/compare_field_response.py', ROOT / 'scripts/compare_two_mps.py',
             ROOT / 'ros2_ws/src/omni_autonomy_next/omni_autonomy_next/trajectory_tracker_node.py',
             ROOT / 'ros2_ws/src/omni_autonomy_next/omni_autonomy_next/runtime_guard.py',
             ROOT / 'ros2_ws/src/omni_autonomy_next/omni_autonomy_next/motor_udp_bridge_node.py',
             ROOT / 'ros2_ws/src/omni_autonomy_next/omni_autonomy_next/robomas_uart.py']
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    fitted = json.loads(paths[0].read_text())
    robot = yaml.safe_load(paths[1].read_text())['robot']
    tuning = calibrated_tracking_parameters(robot, hardware=True, motion_mode='simultaneous')
    runtime = yaml.safe_load(paths[2].read_text())['runtime_guard']['ros__parameters']
    profiles = json.loads(runtime['profiles_json'])
    jobs = []

    def add(run, distance, direction, yaw, delay, tau, scale):
        for profile in BASELINE:
            jobs.append(dict(session=run['session_id'], gains=[a['gain'] for a in run['fit']],
                distance=distance, direction=direction, yaw=yaw, delay=delay, tau=tau,
                scale=scale, profile=profile, tuning=tuning,
                limits=[profiles[profile][k] for k in ('linear', 'lateral', 'angular')]))

    for run in fitted['runs'][-1:] if args.quick else fitted['runs']:
        for delay, tau in ([(.1, .08)] if args.quick else [(.1, .08), (.2, .12), (.3, .15)]):
            for distance in ([8.] if args.quick else [2., 4.]):
                for direction, yaw in [(0., 0.), (math.pi/2, 0.),
                                       (math.pi/4, math.pi/2), (math.pi/2, -math.pi)]:
                    add(run, distance, direction, yaw, delay, tau, 1.)
    if not args.quick:
        # Long straight axes allow the 2 m/s cruise to be reached before braking.
        for run in fitted['runs']:
            for delay, tau in [(.1, .08), (.2, .12), (.3, .15)]:
                for direction in [0., math.pi/2]:
                    add(run, 8., direction, 0., delay, tau, 1.)
        for distance in [0., .1, .3]:
            for yaw in [0., math.pi/2, -math.pi]:
                for delay, tau in [(.1, .08), (.3, .15)]:
                    add(fitted['runs'][-1], distance, math.pi/4, yaw, delay, tau, 1.)
        # The existing operator/red-zone scale must still permit settling.
        for scale in [.1, .35, .7]:
            add(fitted['runs'][-1], .3, 0., 0., .3, .15, scale)
    cases = []
    with ProcessPoolExecutor(max_workers=3) as pool:
        for case in pool.map(evaluate, jobs):
            cases.append(case)
            print(len(cases), '/', len(jobs), case['profile'], case['distance'],
                  case['before']['arrival_sec'], case['after']['arrival_sec'], flush=True)
    failed = [i for i, c in enumerate(cases) if not (
        c['after']['arrival_sec'] is not None
        and c['after']['tail_position_error_m'] < .015
        and c['after']['tail_yaw_error_rad'] < .015
        and c['after']['cross_track_peak_m'] < .06
        and c['after']['physical_acceleration_peak_m_s2'] < 1.3
        and c['after']['wheel_command_peak'] <= 8000)]
    summaries = {}
    for profile in BASELINE:
        long = [c for c in cases if c['profile'] == profile and c['distance'] >= 2.]
        change = [(1-c['after']['arrival_sec']/c['before']['arrival_sec'])*100
                  for c in long if c['after']['arrival_sec'] and c['before']['arrival_sec']]
        cruise = [c for c in long if c['yaw'] == 0.]
        summaries[profile] = dict(
            arrival_improvement_percent=dict(min=min(change), median=float(np.median(change)), max=max(change)),
            straight_middle_p10_improvement_percent=[
                (c['after']['middle_speed_p10_m_s']/c['before']['middle_speed_p10_m_s']-1)*100
                for c in cruise],
            straight_peak_improvement_percent=[
                (c['after']['peak_speed_m_s']/c['before']['peak_speed_m_s']-1)*100 for c in cruise])
    for p in paths:
        if hashlib.sha256(p.read_bytes()).hexdigest() != hashes[str(p.relative_to(ROOT))]:
            raise RuntimeError(f'Source changed during comparison: {p}')
    report = dict(kind='Offline fitted-response comparison; not physical validation',
        limitations='No CAD obstacles, live collision monitor, independent localization, tire slip, motor RPM/current/temperature.',
        baseline=BASELINE, baseline_tuning=BASELINE_TUNING,
        sources=hashes, accepted=not failed, failed_case_indices=failed,
        summaries=summaries, cases=cases)
    (ROOT / args.output).write_text(json.dumps(report, indent=2)+'\n')
    print('accepted', report['accepted'], 'failed', failed, flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
