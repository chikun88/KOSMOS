#!/usr/bin/env python3
"""Compare corridor-aware sprint turns with legacy turn caps, without robot I/O."""
import hashlib
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from unittest.mock import patch

import numpy as np
import check_bucket_gate_replay as route


def evaluate(job):
    case, gains, delay, tau = job
    with patch.object(route.tracker, 'sprint_turn_clearance',
                      lambda points, yaws, model, **kwargs: np.zeros(len(points), dtype=bool)):
        before = route.replay(case, True, delay, tau, mode='fast', response_gains=gains)
    after = route.replay(case, True, delay, tau, mode='fast', response_gains=gains)
    feasible_before = before['arrival_sec'] is not None and before['min_cad_clearance_m'] >= .015
    regression = feasible_before and (after['arrival_sec'] is None
        or after['min_cad_clearance_m'] < .015
        or after['arrival_sec'] > before['arrival_sec']+.15)
    return dict(plan_sequence=case['plan_sequence'], delay=delay, tau=tau,
                before=before, after=after, baseline_feasible=feasible_before,
                regression=regression)


def main():
    root = route.ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', type=Path, default=root/'docs/latest_four_mps_routes_20260916.json')
    parser.add_argument('--audit', type=Path, default=root/'docs/latest_four_mps_audit_20260916.json')
    parser.add_argument('--output', type=Path, default=root/'docs/sprint_turning_replay_20260916.json')
    args = parser.parse_args()
    fixture, audit = args.fixture.resolve(), args.audit.resolve()
    gains = [r['gain'] for r in json.loads(audit.read_text())['runs'][0]['exploratory_fit']]
    cases = json.loads(fixture.read_text())['runs'][0]['cases']
    sources = [Path(__file__), fixture, audit, root/'scripts/check_bucket_gate_replay.py',
               root/'scripts/compare_field_response.py', root/'scripts/compare_field_capture.py',
               root/'ros2_ws/src/omni_autonomy_next/test/test_smooth_arrival.py']
    sources += list(route.CONFIG.glob('*.yaml'))
    sources += list((route.CONFIG.parent/'omni_autonomy_next').glob('*.py'))
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    jobs = [(case, gains, delay, tau) for case in cases for delay, tau in ((.2, .12), (.3, .15))]
    results = []
    with ProcessPoolExecutor(max_workers=2) as pool:
        for result in pool.map(evaluate, jobs):
            results.append(result)
            print(len(results), result['plan_sequence'], result['delay'],
                  [(k, round(result[k]['peak_speed_m_s'], 3), result[k]['arrival_sec'],
                    round(result[k]['min_cad_clearance_m'], 3)) for k in ('before', 'after')], flush=True)
    for path in sources:
        if hashlib.sha256(path.read_bytes()).hexdigest() != hashes[str(path)]:
            raise RuntimeError(f'Source changed: {path}')
    report = dict(physical_target_verified=False, sources=hashes,
        limitations='Recorded geometry and low-speed gain extrapolation. CAD eligibility and post-run '
        'clearance, no live collision monitor, RL, localization faults or slip. Baseline disables only '
        'the new open-corridor turn budget. Pre-existing infeasible cases are retained.',
        regressions=sum(r['regression'] for r in results), cases=results)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    if report['regressions']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
