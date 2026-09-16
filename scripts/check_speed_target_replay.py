#!/usr/bin/env python3
"""Compare hypothetical speed ceilings without changing deployed configuration.

Runs the production tracker/guard/bridge in a publisher-free harness. UART
saturation remains enforced at each hypothetical command cap. A long empty runway deliberately removes the
field-length constraint; fitted high-speed response is only an extrapolation.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
from unittest.mock import patch

import yaml

import compare_field_response as plant
from omni_autonomy_next.config import calibrated_tracking_parameters
from omni_autonomy_next import robomas_uart


def evaluate(case):
    # Every override is in memory; never edit YAML or instantiate a ROS node.
    original = plant.make_guard
    def make_guard(cls):
        guard = original(cls)
        guard.hard_limits = replace(guard.hard_limits,
                                   linear=case['target'], lateral=case['target'])
        guard.profiles['balanced'] = replace(guard.profiles['balanced'],
                                             linear=case['target'], lateral=case['target'])
        return guard

    with patch.object(plant, 'make_guard', make_guard), patch.object(
            robomas_uart, 'AUTO_WHEEL_LIMIT', case['uart_limit']):
        result = plant.replay(
            case['gains'], case['calibration'], case['delay'], case['tau'],
            case['direction'], 0., 1., overrides=case['tuning'],
            distance=case['distance'], profile_limits=(case['target'], case['target'], 1.3),
            bridge_limits=(case['target'], 1.8),
            wheel_limit=case['target']*math.cos(math.pi/4)/case['wheel_radius'],
            duration_sec=case['duration'])
    return dict(**case, result=result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--targets', type=float, nargs='+', default=[2., 2.9, 6.])
    parser.add_argument('--uart-limits', type=int, nargs='+',
                        default=[robomas_uart.AUTO_WHEEL_LIMIT],
                        help='Hypothetical caps only; not hardware-approved ratings')
    parser.add_argument('--distance', type=float, default=60.)
    parser.add_argument('--duration', type=float, default=60.)
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    if any(not math.isfinite(v) or v <= 0 for v in
           [*args.targets, args.distance, args.duration]) or args.workers < 1:
        parser.error('targets, distance, duration and workers must be positive and finite')
    if any(v <= 0 or v > 32767 for v in args.uart_limits):
        parser.error('hypothetical UART limits must fit a positive signed int16')
    root = plant.ROOT
    paths = [root/'ros2_ws/src/omni_autonomy_next/config'/f for f in
             ('robot.yaml', 'runtime.yaml', 'nav2_next.yaml')]
    paths += [Path(__file__).resolve(), root/'scripts/compare_field_response.py',
              args.audit.resolve()]
    paths += sorted((root/'ros2_ws/src/omni_autonomy_next/omni_autonomy_next').glob('*.py'))
    paths += [root/'scripts/compare_field_capture.py',
              root/'ros2_ws/src/omni_autonomy_next/test/test_smooth_arrival.py']
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    audit = json.loads(args.audit.read_text())
    robot = yaml.safe_load(paths[0].read_text())['robot']
    drive = robot['drivetrain']
    tuning = calibrated_tracking_parameters(robot, hardware=True, motion_mode='simultaneous')
    # The latest two successfully fitted sessions provide two empirical gains.
    # Their recording gaps are preserved as a limitation, not hidden by replay.
    runs = [r for r in audit['runs'] if r['exploratory_fit']
            and r['settings']['operation_mode'] == 'hardware'][:2]
    if not runs:
        parser.error('audit contains no fitted hardware runs')
    jobs = [dict(session=r['session'], gains=[a['gain'] for a in r['exploratory_fit']],
                 recording_dropped=r['recording_at_snapshot']['dropped'],
                 calibration=[drive['linear_command_scale'], drive['angular_command_scale']],
                 tuning=tuning, wheel_radius=drive['wheel_radius'],
                 target=t, uart_limit=u, direction=d, distance=args.distance, duration=args.duration,
                 delay=.2, tau=.12)
            for r in runs for d in (0., math.pi/2) for t in args.targets for u in args.uart_limits]
    cases = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for case in pool.map(evaluate, jobs):
            cases.append(case)
            r = case['result']
            print(len(cases), '/', len(jobs), case['session'], case['target'],
                  'peak', round(r['peak_speed_m_s'], 3),
                  'arrival', r['arrival_sec'], 'UART', r['wheel_command_peak'],
                  '/', case['uart_limit'], flush=True)
    for p in paths:
        if hashlib.sha256(p.read_bytes()).hexdigest() != hashes[str(p)]:
            raise RuntimeError(f'Source changed during replay: {p}')
    report = dict(kind='Offline hypothetical speed ceiling comparison',
                  physical_target_verified=False, deployed_configuration_changed=False,
                  limitations='60 m default is a hypothetical empty runway, not the real field. '
                  'No collision monitor, CAD obstacles, independent localization, slip, motor '
                  'RPM/current/temperature or high-speed calibration. Delay/tau are scenarios. '
                  'Recorded gain fits contain dropped messages. UART saturation is enforced '
                  'at the hypothetical cap, which is not a hardware rating or deployment recommendation.',
                  sources=hashes, cases=cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
