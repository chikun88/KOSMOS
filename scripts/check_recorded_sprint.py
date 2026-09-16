#!/usr/bin/env python3
"""Compare the lost sprint wheel-budget fix on recorded route geometry.

Read-only hardware logs and publisher-free production control replay. This is
not a physical 4 m/s result or a live Nav2/collision-monitor acceptance test.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from check_bucket_gate_replay import CONFIG, ROOT, replay


def pose_values(pose):
    p, q = pose['position'], pose['orientation']
    yaw = math.atan2(2*(q['w']*q['z']+q['x']*q['y']),
                     1-2*(q['y']**2+q['z']**2))
    return [p['x'], p['y'], yaw]


def extract(directory):
    manifest = json.loads((directory/'manifest.json').read_text())
    if manifest['settings']['operation_mode'] != 'hardware':
        raise ValueError('hardware recording required')
    topics = ('/plan', '/localization/pose', '/navigation/active_goal')
    needles = [json.dumps(t).encode() for t in topics]
    records, hashes = [], {}
    for path in sorted(directory.glob('samples-*.jsonl')):
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for raw in stream:
                digest.update(raw)
                if not any(needle in raw for needle in needles):
                    continue
                row = json.loads(raw)
                if row.get('topic') not in topics or not row.get('source_ros_ns'):
                    continue
                if abs(row['received_ros_ns']-row['received_unix_ns']) > 20_000_000:
                    continue
                records.append(row)
        hashes[path.name] = digest.hexdigest()
    records.sort(key=lambda row: (row['source_ros_ns'], row['sequence']))
    goal, pose, pose_at, goal_sequence = None, None, None, None
    chosen = {}
    for row in records:
        value, stamp = row['value'], row['source_ros_ns']
        if value['header']['frame_id'] != 'map':
            continue
        if row['topic'] == '/localization/pose':
            pose, pose_at = pose_values(value['pose']['pose']), stamp
        elif row['topic'] == '/navigation/active_goal':
            goal, goal_sequence = pose_values(value['pose']), row['sequence']
        elif goal is not None and pose is not None and goal_sequence not in chosen:
            points = np.array([[p['pose']['position']['x'], p['pose']['position']['y']]
                               for p in value['poses']])
            if len(points) < 2 or stamp-pose_at > 300_000_000:
                continue
            if (np.linalg.norm(points[0]-pose[:2]) > .3
                    or np.linalg.norm(points[-1]-goal[:2]) > .3):
                continue
            length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
            if length < .5:
                continue
            chosen[goal_sequence] = dict(
                session=directory.name, plan_sequence=row['sequence'],
                goal_sequence=goal_sequence, points=points.tolist(), gates=[],
                goal=goal, yaw=pose[2], length_m=length,
                pose_age_sec=(stamp-pose_at)*1e-9,
                start_position_discrepancy_m=float(np.linalg.norm(points[0]-pose[:2])))
    return dict(session=directory.name, raw_sha256=hashes, cases=list(chosen.values()))


def evaluate(job):
    case, gains, delay, tau = job
    metrics = {}
    for label, legacy in [('before', True), ('after', False)]:
        metrics[label] = replay(case, True, delay, tau, mode='fast',
                                legacy_tracker_budget=legacy, response_gains=gains)
    before, after = metrics['before'], metrics['after']
    # Report pre-existing infeasible cases instead of filtering them away.
    safe_before = before['arrival_sec'] is not None and before['min_cad_clearance_m'] >= .015
    safe_after = after['arrival_sec'] is not None and after['min_cad_clearance_m'] >= .015
    return dict(session=case['session'], plan_sequence=case['plan_sequence'],
                length_m=case['length_m'], delay_sec=delay, tau_sec=tau,
                **metrics, baseline_feasible=safe_before, candidate_feasible=safe_after,
                regression=bool(safe_before and (not safe_after
                    or after['min_cad_clearance_m'] < before['min_cad_clearance_m']-.005
                    or after['arrival_sec'] > before['arrival_sec']+.15)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directories', type=Path, nargs='*')
    parser.add_argument('--fixture', type=Path, default=ROOT/'docs/recorded_sprint_fixture_20260916.json')
    parser.add_argument('--audit', type=Path, default=ROOT/'docs/four_mps_audit_20260916.json')
    parser.add_argument('--output', type=Path, default=ROOT/'docs/recorded_sprint_replay_20260916.json')
    parser.add_argument('--extract-only', action='store_true')
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    args.fixture = args.fixture.resolve()
    args.audit = args.audit.resolve()
    if args.workers < 1:
        parser.error('workers must be positive')
    if args.directories:
        fixture = dict(runs=[extract(d) for d in args.directories])
        args.fixture.write_text(json.dumps(fixture, indent=2)+'\n')
    else:
        fixture = json.loads(args.fixture.read_text())
    if args.extract_only:
        for run in fixture['runs']:
            print(run['session'], [(c['plan_sequence'], round(c['length_m'], 2)) for c in run['cases']])
        return
    audit = json.loads(args.audit.read_text())
    gains = {r['session']: [a['gain'] for a in r['exploratory_fit']]
             for r in audit['runs'] if r['exploratory_fit']}
    paths = [Path(__file__).resolve(), args.fixture, args.audit,
             ROOT/'scripts/check_bucket_gate_replay.py', ROOT/'scripts/compare_field_response.py',
             ROOT/'scripts/compare_field_capture.py']
    paths += list((ROOT/'ros2_ws/src/omni_autonomy_next/omni_autonomy_next').glob('*.py'))
    paths += list(CONFIG.glob('*.yaml'))
    paths += [ROOT/'ros2_ws/src/omni_autonomy_next/test/test_smooth_arrival.py']
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    conditions = [(.2, .12)] if args.quick else [(.1, .08), (.2, .12), (.3, .15)]
    jobs = [(c, gains[r['session']], delay, tau) for r in fixture['runs']
            for c in r['cases'] for delay, tau in conditions]
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(evaluate, jobs):
            results.append(result)
            print(len(results), '/', len(jobs), result['plan_sequence'],
                  {k: {m: round(v[m], 3) if v[m] is not None else None
                       for m in ('peak_speed_m_s', 'arrival_sec', 'min_cad_clearance_m')}
                   for k, v in result.items() if k in ('before', 'after')}, flush=True)
    for path in paths:
        if hashlib.sha256(path.read_bytes()).hexdigest() != hashes[str(path)]:
            raise RuntimeError(f'source changed during replay: {path}')
    report = dict(kind=__doc__, sources=hashes, quick=args.quick,
        physical_target_verified=False,
        limitations='Recorded path geometry, source-time pose/goal association, current sprint settings. '
        'No live Smac, RL, collision monitor, localization jumps, manual/reverse modes or moving obstacles. '
        'Starts stationary at the first recorded path point; retains its detours. CAD checked afterwards. '
        'Gains from incomplete low-speed recordings; delay/tau scenarios, no high-speed identification. '
        'No comparison of real before/after runs. No claim of all-route acceptance.',
        regressions=sum(r['regression'] for r in results),
        candidate_infeasible=sum(not r['candidate_feasible'] for r in results), cases=results)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    if report['regressions']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
