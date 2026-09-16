#!/usr/bin/env python3
"""Offline comparison of start-exit and bucket turn speed; never publishes commands.

Bucket paths are recorded; the start-exit path is a constructed fixed-gate case.
Uses the existing calibrated tracker/guard/UART replay, not live Nav2 or LiDAR.
"""
import argparse
import json
import math

from check_bucket_gate_replay import ROOT, CONFIG, replay
from omni_autonomy_next.fixed_gate_speed import FixedGateSpeed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()
    fixture = json.loads((ROOT/'docs/bucket_gate_fixture_20260915.json').read_text())
    cases = [('bucket_' + str(i), case) for i, case in enumerate(fixture['cases'])]
    cases.append(('start_exit', dict(
        points=[[-1.8, 4.75], [-1.8, 4.1], [-2.8, 4.1], [-4.19, 4.69]],
        gates=[[-1.8, 4.1]], goal=[-4.19, 4.69, 0.], yaw=-math.pi/2)))
    baseline = dict(speed=.20, transit_speed=.40, entry_hold_m=.25, exit_hold_m=.25)
    configured = FixedGateSpeed.from_yaml(CONFIG/'routes.yaml')
    candidate = {name: getattr(configured, name) for name in baseline}
    conditions = [(False, .3, .15, 1.1)] if args.quick else [
        (mirror, delay, tau, gain) for mirror in (False, True)
        for delay, tau, gain in ((.1, .08, 1.), (.2, .12, 1.), (.3, .15, 1.1))]
    results = []
    for name, case in cases:
        for mirror, delay, tau, gain in conditions:
            row = dict(case=name, mirror=mirror, delay_sec=delay, tau_sec=tau, gain_scale=gain)
            for label, tuning in [('before', baseline), ('after', candidate)]:
                row[label] = replay(case, True, delay, tau, mirror, gain, gate_tuning=tuning)
            before, after = row['before'], row['after']
            row['accepted'] = (
                before['arrival_sec'] is not None and after['arrival_sec'] is not None
                and after['arrival_sec'] < before['arrival_sec']
                and after['passed_gates'] == after['required_gates']
                and after['final_error_m'] < .015
                and after['min_cad_clearance_m'] >= .015
                and after['min_cad_clearance_m'] >= before['min_cad_clearance_m']-.005
                and (name == 'start_exit' or after['backward_y_m'] < .10))
            # The start path intentionally goes south then north, so summed
            # reverse-Y travel is not a backtracking metric for that case.
            results.append(row)
            print(json.dumps(row), flush=True)
    accepted = all(row['accepted'] for row in results)
    output = ROOT/'docs/local_speed_replay_20260915.json'
    output.write_text(json.dumps(dict(
        accepted=accepted, quick=args.quick, limitations=__doc__,
        baseline=baseline, candidate=candidate, cases=results), indent=2)+'\n')
    if not accepted:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
