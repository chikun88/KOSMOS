#!/usr/bin/env python3
"""Offline removal of sprint's fixed turn caps; never deploys or publishes."""
import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from unittest.mock import patch

import numpy as np
import check_bucket_gate_replay as route


def evaluate(job):
    case, gains, delay, tau = job
    before = route.replay(case, True, delay, tau, mode='fast', response_gains=gains)
    # Bypass only the extra turn budgets. Keep the production wheel envelope,
    # angular/linear acceleration, terminal braking, guard and UART saturation.
    with patch.object(route.tracker, 'sprint_turn_clearance',
                      lambda points, yaws, model, **kw: np.ones(len(points), dtype=bool)):
        after = route.replay(case, True, delay, tau, mode='fast', response_gains=gains)
    feasible = before['arrival_sec'] is not None and before['min_cad_clearance_m'] >= .015
    clearance_regression = feasible and after['min_cad_clearance_m'] < .015
    return dict(plan_sequence=case['plan_sequence'], delay_sec=delay, tau_sec=tau,
        before=before, after=after, baseline_feasible=feasible,
        clearance_regression=clearance_regression,
        regression=feasible and (clearance_regression or after['arrival_sec'] is None
            or after['arrival_sec'] > before['arrival_sec']+.15))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    runs = json.loads(args.audit.read_text())['runs']
    gains = {r['session']: [a['gain'] for a in r['exploratory_fit']]
             for r in runs if r['exploratory_fit']}
    fixture = json.loads(args.fixture.read_text())
    files = [Path(__file__).resolve(), args.fixture.resolve(), args.audit.resolve(),
        route.ROOT/'scripts/check_bucket_gate_replay.py', route.ROOT/'scripts/compare_field_response.py',
        route.ROOT/'scripts/compare_field_capture.py',
        route.ROOT/'ros2_ws/src/omni_autonomy_next/test/test_smooth_arrival.py']
    files += list(route.CONFIG.glob('*.yaml'))
    files += list((route.CONFIG.parent/'omni_autonomy_next').glob('*.py'))
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    jobs = [(case, gains[run['session']], d, t) for run in fixture['runs']
            for case in run['cases'] for d, t in ((.2, .12), (.3, .15))]
    results = []
    with ProcessPoolExecutor(max_workers=2) as pool:
        for r in pool.map(evaluate, jobs):
            results.append(r)
            print(len(results), r['plan_sequence'], r['delay_sec'],
                [(k, round(r[k]['peak_speed_m_s'], 3), r[k]['arrival_sec'],
                  round(r[k]['min_cad_clearance_m'], 3)) for k in ('before', 'after')], flush=True)
    for p in files:
        if hashlib.sha256(p.read_bytes()).hexdigest() != hashes[str(p)]:
            raise RuntimeError(f'Source changed: {p}')
    report = dict(deployed=False, physical_target_verified=False, sources=hashes,
        limitations='Offline recorded geometry and low-speed gain extrapolation; delay scenarios. '
        'No live collision monitor, localization faults, manual intervention, load or slip. '
        'CAD overlap is a model result, not evidence of an actual collision.',
        regressions=sum(r['regression'] for r in results),
        clearance_regressions=sum(r['clearance_regression'] for r in results), cases=results)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    if report['regressions']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
