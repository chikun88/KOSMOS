#!/usr/bin/env python3
"""Compare acceleration alone using production control and recorded route geometry.

No ROS nodes or motor I/O. Braking, speed limits and guard are held fixed.
"""
import json
import hashlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from unittest.mock import patch

import check_bucket_gate_replay as route
import compare_field_response as straight
from omni_autonomy_next.config import calibrated_tracking_parameters
import yaml

ROOT = Path(__file__).resolve().parents[1]


def evaluate(job):
    case, acceleration, delay, tau, gains = job
    module = route if case.get('points') else straight
    loader = module.load_robot
    def configured(path):
        robot = loader(path)
        robot['drivetrain']['profile_linear_accelerations'] = {case.get('profile', 'sprint'): acceleration}
        return robot
    with patch.object(module, 'load_robot', configured):
        if case.get('points'):
            result = route.replay(case, True, delay, tau, mode='fast', response_gains=gains)
        else:
            robot = yaml.safe_load((route.CONFIG/'robot.yaml').read_text())['robot']
            result = straight.replay(gains, [.38, .34], delay, tau, 0., 0., 1.,
                overrides=calibrated_tracking_parameters(robot, hardware=True, motion_mode='simultaneous'),
                distance=case['distance_m'], profile=case.get('profile', 'sprint'), duration_sec=30.)
    return dict(case=case.get('plan_sequence', case.get('distance_m')),
                profile=case.get('profile', 'sprint'), kind='recorded_geometry' if case.get('points') else 'empty_straight',
                acceleration=acceleration, delay=delay, tau=tau, result=result)


def main():
    fixture = ROOT/'docs/acceleration_routes_20260916.json'
    audit = ROOT/'docs/acceleration_audit_20260916.json'
    runs = json.loads(audit.read_text())['runs']
    fitted = next(r for r in runs if r['exploratory_fit'])
    gains = [a['gain'] for a in fitted['exploratory_fit']]
    cases = [c for r in json.loads(fixture.read_text())['runs'] for c in r['cases']]
    cases += [dict(distance_m=d) for d in (2., 4., 8.)]
    cases += [dict(distance_m=d, profile='balanced') for d in (2., 4.)]
    files = [Path(__file__), fixture, audit, ROOT/'scripts/check_bucket_gate_replay.py',
             ROOT/'scripts/compare_field_response.py']
    files += list(route.CONFIG.glob('*.yaml'))
    files += list((ROOT/'ros2_ws/src/omni_autonomy_next/omni_autonomy_next').glob('*.py'))
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    jobs = [(c, a, delay, tau, gains) for c in cases for a in ((.85, 1.8) if c.get('profile') == 'balanced' else (.85, 1.8, 3.))
            for delay, tau in ((.2, .12), (.3, .15))]
    results = []
    with ProcessPoolExecutor(max_workers=2) as pool:
        for r in pool.map(evaluate, jobs):
            results.append(r)
            print(len(results),len(jobs),r['kind'],r['case'],r['acceleration'],r['delay'],
                  r['result']['arrival_sec'],r['result']['peak_speed_m_s'], flush=True)
    for p in files:
        if hashlib.sha256(p.read_bytes()).hexdigest() != hashes[str(p.relative_to(ROOT))]:
            raise RuntimeError(f'source changed: {p}')
    (ROOT/'docs/acceleration_replay_20260916.json').write_text(json.dumps(dict(
        sources=hashes, gain_session=fitted['session'], physical_improvement_verified=False,
        limitations='Acceleration-only comparison, guard held fixed. Latest route geometry; '
        'gain from previous incomplete run because latest fit rejected. '
        'No live Nav2, RL, collision monitor, sensor faults or load/slip model; '
        'delay/tau are assumptions. Straight paths have no obstacles.', cases=results),indent=2)+'\n')


if __name__ == '__main__':
    main()
