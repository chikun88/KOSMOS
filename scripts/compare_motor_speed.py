#!/usr/bin/env python3
"""Compare configured speed profiles with prior profiles using fitted run gains.

Offline production control ticks only; no ROS nodes or motor publishers.
"""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math

import numpy as np
import yaml
from compare_field_response import ROOT, replay
from omni_autonomy_next.config import calibrated_tracking_parameters

BASELINE = {'balanced': [.78, .702, 1.3], 'sprint': [.95, .57, 1.6]}


def evaluate(case):
    args = (case['gains'], [.38, .34], case['delay'], case['tau'],
            case['direction'], case['yaw'], case['scale'])
    kwargs = dict(overrides=case['tuning'], profile=case['profile'], distance=case['distance'])
    case['before'] = replay(*args, **kwargs, profile_limits=BASELINE[case['profile']])
    case['after'] = replay(*args, **kwargs, profile_limits=case['limits'])
    return case


def main():
    config = ROOT / 'ros2_ws/src/omni_autonomy_next/config'
    paths = [ROOT / 'docs/field_response_fit_20260914.json', config / 'robot.yaml',
             config / 'runtime.yaml', config / 'nav2_next.yaml']
    fitted = json.loads(paths[0].read_text())
    robot = yaml.safe_load(paths[1].read_text())['robot']
    tuning = calibrated_tracking_parameters(robot, hardware=True, motion_mode='simultaneous')
    runtime = yaml.safe_load(paths[2].read_text())['runtime_guard']['ros__parameters']
    profiles = json.loads(runtime['profiles_json'])
    jobs = []
    for run in fitted['runs']:
        gains = [axis['gain'] for axis in run['fit']]
        for delay, tau in [(.1, .08), (.2, .12), (.3, .15)]:
            for direction, yaw in [(0., 0.), (math.pi/2, 0.),
                                   (math.pi/4, math.pi/2), (math.pi/2, -math.pi)]:
                for profile in BASELINE:
                    jobs.append(dict(session=run['session_id'], gains=gains, delay=delay,
                        tau=tau, direction=direction, yaw=yaw, profile=profile,
                        limits=[profiles[profile][k] for k in ('linear', 'lateral', 'angular')],
                        scale=1., distance=2., tuning=tuning))
    # Close goals and in-place turns: both delay extremes, latest held-out fit.
    template = jobs[-1]
    for distance in [0., .1, .3]:
        for yaw in [0., math.pi/2, -math.pi]:
            for delay, tau in [(.1, .08), (.3, .15)]:
                for profile in BASELINE:
                    jobs.append(dict(template, distance=distance, yaw=yaw, direction=math.pi/4,
                        delay=delay, tau=tau, profile=profile,
                        limits=[profiles[profile][k] for k in ('linear', 'lateral', 'angular')]))
    with ProcessPoolExecutor(max_workers=3) as pool:
        cases = []
        for case in pool.map(evaluate, jobs):
            cases.append(case)
            print(len(cases), case['profile'], case['distance'],
                  case['before']['arrival_sec'], case['after']['arrival_sec'], flush=True)
    accepted = all(c['after']['arrival_sec'] is not None
        and c['after']['tail_position_error_m'] < .015
        and c['after']['tail_yaw_error_rad'] < .015
        and c['after']['cross_track_peak_m'] < .06
        and c['after']['physical_acceleration_peak_m_s2'] < 1.3
        and c['after']['wheel_command_peak'] <= 8000 for c in cases)
    changes = [(1-c['after']['arrival_sec']/c['before']['arrival_sec'])*100
        for c in cases[:48] if c['after']['arrival_sec'] and c['before']['arrival_sec']]
    cruise = [c for c in cases[:48] if c['yaw'] == 0.]
    accepted = accepted and all(c['after']['peak_speed_m_s'] > c['before']['peak_speed_m_s']
        and c['after']['wheel_command_peak'] > c['before']['wheel_command_peak'] for c in cruise)
    report = dict(kind='Offline fitted-response speed comparison; not physical validation',
        limitations='No CAD obstacles, tire slip, independent localization, physical current/temperature/RPM or live collision monitor.',
        baseline=BASELINE, sources={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        accepted=accepted, long_move_improvement_percent=dict(min=min(changes),
            median=float(np.median(changes)), max=max(changes)), cases=cases)
    (ROOT / 'docs/motor_speed_replay_20260914.json').write_text(json.dumps(report, indent=2)+'\n')
    print('accepted', accepted, report['long_move_improvement_percent'])
    if not accepted:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
