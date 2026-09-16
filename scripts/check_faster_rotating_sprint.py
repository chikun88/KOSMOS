#!/usr/bin/env python3
"""Compare faster rotating sprint with the current predictive policy; no robot I/O."""
import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from unittest.mock import patch
import check_bucket_gate_replay as route
from check_turn_budget_candidates import candidate_method

BUILD = route.tracker.TrajectoryTracker._build_trajectory



def evaluate(job):
    session, case, gains, delay, tau = job
    results = {}
    for label in ('before', 'after'):
        build = (candidate_method(BUILD,
            "turn_speed = 4.00", "turn_speed = 3.00")
            if label == 'before' else BUILD)
        with patch.object(route.tracker.TrajectoryTracker, '_build_trajectory', build):
            results[label] = route.replay(case, True, delay, tau, mode='fast', response_gains=gains)
    before, after = results['before'], results['after']
    feasible = before['arrival_sec'] is not None and before['min_cad_clearance_m'] >= .015
    regression = feasible and (after['arrival_sec'] is None or after['min_cad_clearance_m'] < .015
        or after['arrival_sec'] > before['arrival_sec']+.15)
    return dict(session=session, plan_sequence=case['plan_sequence'], delay_sec=delay,
                tau_sec=tau, baseline_feasible=feasible, regression=regression, **results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = route.ROOT
    files = [Path(__file__).resolve(), root/'scripts/check_bucket_gate_replay.py', root/'scripts/check_turn_budget_candidates.py',
        root/'scripts/compare_field_response.py', root/'scripts/compare_field_capture.py',
        route.CONFIG.parent/'test/test_smooth_arrival.py']
    files += sorted(route.CONFIG.glob('*.yaml'))
    files += sorted((route.CONFIG.parent/'omni_autonomy_next').glob('*.py'))
    jobs = []
    for prefix in ('unrestricted_turn','latest_speed_retry','latest_four_mps'):
        fixture, audit = (root/f'docs/{prefix}_{suffix}_20260916.json' for suffix in ('routes','audit'))
        files += [fixture,audit]
        gains = {r['session']:[a['gain'] for a in r['exploratory_fit']]
                 for r in json.loads(audit.read_text())['runs'] if r['exploratory_fit']}
        for run in json.loads(fixture.read_text())['runs']:
            for case in run['cases']:
                jobs.extend((run['session'],case,gains[run['session']],d,t)
                            for d,t in ((.2,.12),(.3,.15)))
    hashes = {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    results = []
    with ProcessPoolExecutor(max_workers=2) as pool:
        for r in pool.map(evaluate,jobs):
            results.append(r)
            print(len(results),r['plan_sequence'],r['delay_sec'],
                [(k,round(r[k]['peak_speed_m_s'],3),r[k]['arrival_sec'],
                  round(r[k]['min_cad_clearance_m'],3)) for k in ('before','after')],
                'regression=',r['regression'],flush=True)
    for name,digest in hashes.items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'Source changed: {name}')
    report = dict(physical_target_verified=False,sources=hashes,
        limitations='Recorded geometry, low-speed gain extrapolation, assumed delays; '
        'no live collision monitor, localization faults, load or slip. '
        'Pre-existing failing cases are not counted as successful.',
        baseline_feasible=sum(r['baseline_feasible'] for r in results),
        regressions=sum(r['regression'] for r in results),cases=results)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    if report['regressions']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
