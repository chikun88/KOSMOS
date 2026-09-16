#!/usr/bin/env python3
"""Replay candidate sprint turn caps in memory; never publish or deploy them."""
import argparse
import hashlib
import inspect
import json
import math
import textwrap
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from unittest.mock import patch

import check_bucket_gate_replay as route

BUILD = route.tracker.TrajectoryTracker._build_trajectory
TICK = route.tracker.TrajectoryTracker._tick
CALIBRATION = route.calibrated_tracking_parameters


def legacy_tuning(*args, **kwargs):
    return {**CALIBRATION(*args, **kwargs), 'predictive_sprint': False}


def candidate_method(method, old, new):
    source = textwrap.dedent(inspect.getsource(method))
    if source.count(old) != 1:
        raise RuntimeError('Production implementation changed; review the candidate substitution')
    namespace = {}
    exec(compile(source.replace(old, new), '<offline-turn-candidate>', 'exec'),
         method.__globals__, namespace)
    return namespace[method.__name__]


def evaluate(job):
    session, case, gains, delay, tau, speed, lookahead = job
    with patch.object(route, 'calibrated_tracking_parameters', legacy_tuning):
        before = route.replay(case, True, delay, tau, mode='fast', response_gains=gains)
    build = candidate_method(BUILD,
        "turn_speed = (3.00 if self.get_parameter('predictive_sprint').value else 1.30)",
        f'turn_speed = {speed!r}')
    tick = (TICK if lookahead is None else candidate_method(TICK,
        "lag = float(self.get_parameter('feedback_delay_sec').value)", f'lag = {lookahead!r}'))
    with patch.object(route.tracker.TrajectoryTracker, '_build_trajectory', build), \
         patch.object(route.tracker.TrajectoryTracker, '_tick', tick), \
         patch.object(route, 'calibrated_tracking_parameters', legacy_tuning):
        after = route.replay(case, True, delay, tau, mode='fast', response_gains=gains)
    feasible = before['arrival_sec'] is not None and before['min_cad_clearance_m'] >= .015
    clearance_regression = feasible and after['min_cad_clearance_m'] < .015
    arrival_regression = feasible and (after['arrival_sec'] is None or
        after['arrival_sec'] > before['arrival_sec'] + .15)
    return dict(session=session, plan_sequence=case['plan_sequence'], delay_sec=delay,
        tau_sec=tau, before=before, after=after, baseline_feasible=feasible,
        clearance_regression=clearance_regression, arrival_regression=arrival_regression,
        regression=bool(clearance_regression or arrival_regression))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--turn-speed', type=float, required=True)
    parser.add_argument('--feedback-lookahead', type=float)
    parser.add_argument('--cases', type=int, nargs='+', help='Optional plan-sequence subset')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not math.isfinite(args.turn_speed) or not 1.3 < args.turn_speed <= 4.:
        parser.error('--turn-speed must be finite and in (1.3, 4.0]')
    if args.feedback_lookahead is not None and (not math.isfinite(args.feedback_lookahead)
                                               or not 0. < args.feedback_lookahead <= 1.):
        parser.error('--feedback-lookahead must be finite and in (0, 1]')
    root = route.ROOT
    package = route.CONFIG.parent
    files = [Path(__file__).resolve(), root/'scripts/check_bucket_gate_replay.py',
        root/'scripts/compare_field_response.py', root/'scripts/compare_field_capture.py',
        package/'test/test_smooth_arrival.py']
    files += sorted(route.CONFIG.glob('*.yaml'))
    files += sorted((package/'omni_autonomy_next').glob('*.py'))
    jobs = []
    for prefix in ('unrestricted_turn', 'latest_speed_retry', 'latest_four_mps'):
        fixture = root/f'docs/{prefix}_routes_20260916.json'
        audit = root/f'docs/{prefix}_audit_20260916.json'
        files += [fixture, audit]
        gains = {r['session']: [a['gain'] for a in r['exploratory_fit']]
                 for r in json.loads(audit.read_text())['runs'] if r['exploratory_fit']}
        for run in json.loads(fixture.read_text())['runs']:
            for case in run['cases']:
                if args.cases and case['plan_sequence'] not in args.cases:
                    continue
                for delay, tau in ((.2, .12), (.3, .15)):
                    jobs.append((run['session'], case, gains[run['session']], delay, tau,
                                 args.turn_speed, args.feedback_lookahead))
    if not jobs:
        parser.error('No matching cases')
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    results = []
    with ProcessPoolExecutor(max_workers=2) as pool:
        for result in pool.map(evaluate, jobs):
            results.append(result)
            print(len(results), result['plan_sequence'], result['delay_sec'],
                [(key, round(result[key]['peak_speed_m_s'], 3), result[key]['arrival_sec'],
                  round(result[key]['min_cad_clearance_m'], 3)) for key in ('before', 'after')],
                'regression=', result['regression'], flush=True)
    for relative, digest in hashes.items():
        if hashlib.sha256((root/relative).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'Source changed during replay: {relative}')
    report = dict(deployed=False, physical_target_verified=False, sources=hashes,
        turn_speed_m_s=args.turn_speed, feedback_lookahead_sec=args.feedback_lookahead,
        limitations='Recorded geometry with extrapolated low-speed gains and assumed delays. '
        'No live collision monitor, localization faults, manual intervention, load or slip. '
        'CAD overlap is a model result, not an observed physical collision. '
        'Cases failing the baseline are reported, not counted as passing.',
        baseline_feasible=sum(r['baseline_feasible'] for r in results),
        clearance_regressions=sum(r['clearance_regression'] for r in results),
        regressions=sum(r['regression'] for r in results), cases=results)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    if report['regressions']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
