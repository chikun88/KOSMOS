#!/usr/bin/env python3
"""Offline acceleration comparison; no publishers or robot I/O."""
import hashlib
import json
from pathlib import Path
import numpy as np
from check_acceleration_replay import evaluate, ROOT
from omni_autonomy_next.trajectory_tracker_node import Trajectory


def main():
    audit = ROOT/'docs/acceleration_audit_20260916.json'
    fixture = ROOT/'docs/acceleration_routes_20260916.json'
    fitted = next(r for r in json.loads(audit.read_text())['runs'] if r['exploratory_fit'])
    gains = [a['gain'] for a in fitted['exploratory_fit']]
    cases = [c for r in json.loads(fixture.read_text())['runs'] for c in r['cases']]
    cases += [dict(distance_m=d) for d in (8., 16.)]
    results = []
    for case in cases:
        for delay, tau in ((.2, .12), (.3, .15)):
            pair = {label: evaluate((case, accel, delay, tau, gains))
                    for label, accel in (('before', 3.), ('after', 3.3))}
            before, after = (pair[k]['result'] for k in ('before', 'after'))
            feasible = before['arrival_sec'] is not None and before.get('min_cad_clearance_m', 1.) >= .015
            pair['baseline_feasible'] = feasible
            pair['regression'] = feasible and (after['arrival_sec'] is None or
                after['arrival_sec'] > before['arrival_sec'] + .15 or
                after.get('min_cad_clearance_m', 1.) < .015)
            results.append(pair)
            print(case.get('plan_sequence', case.get('distance_m')), delay,
                  before['arrival_sec'], after['arrival_sec'], pair['regression'], flush=True)
    plans = []
    for distance in (4., 8., 16., 30.):
        x = np.linspace(0., distance, round(distance/.01)+1)
        for accel in (3., 3.3):
            plan = Trajectory(np.c_[x, np.zeros_like(x)], np.zeros_like(x),
                np.full_like(x, 4.), acceleration=accel, deceleration=.85,
                lateral_acceleration=1.2, entry_speed=0.)
            cruise = (plan.speed[:-1] >= 4.-1.e-8) & (plan.speed[1:] >= 4.-1.e-8)
            plans.append(dict(distance_m=distance, acceleration=accel,
                cruise_at_4_m_s_sec=float(np.diff(plan.time)[cruise].sum())))
    files = [Path(__file__), audit, fixture]
    files += list((ROOT/'ros2_ws/src/omni_autonomy_next/config').glob('*.yaml'))
    report = dict(physical_improvement_verified=False,
        limitations='Acceleration-only comparison with current guard (3.5) held fixed. '
        'Recorded geometry and low-speed response extrapolation; assumed delay; no live obstacle monitor or slip. '
        'Plan cruise is an ideal straight trajectory, not measured robot speed.',
        sources={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        regressions=sum(p['regression'] for p in results), cases=results, ideal_plans=plans)
    (ROOT/'docs/longer_cruise_20260916.json').write_text(json.dumps(report, indent=2)+'\n')
    if report['regressions']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
