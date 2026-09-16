#!/usr/bin/env python3
"""Compare 3/4 m/s turn budgets in a hypothetical straight corridor; no robot I/O."""
import argparse
import hashlib
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import yaml
import numpy as np
from check_faster_rotating_sprint import BUILD, candidate_method
import compare_field_response as plant
from omni_autonomy_next.config import calibrated_tracking_parameters
from omni_autonomy_next.rl_residual import CadClearanceModel


def evaluate(job):
    direction, yaw, delay, tau, gains, distance = job
    robot = yaml.safe_load((plant.ROOT/'ros2_ws/src/omni_autonomy_next/config/robot.yaml').read_text())['robot']
    make_node = plant.make_node
    results = {}
    for label in ('before', 'after'):
        def factory(*args,**kwargs):
            node = make_node(*args,**kwargs)
            tangent = np.array([math.cos(direction), math.sin(direction)])
            normal = np.array([-tangent[1], tangent[0]])
            walls = [[(-200*tangent+offset*normal).tolist(),
                      (200*tangent+offset*normal).tolist()] for offset in (-.8, .8)]
            node.clearance = CadClearanceModel(walls, robot['footprint'])
            return node
        build = (candidate_method(BUILD,
            "turn_speed = 4.00", "turn_speed = 3.00")
            if label == 'before' else BUILD)
        with patch.object(plant,'make_node',factory), \
             patch.object(plant.tracker.TrajectoryTracker, '_build_trajectory', build):
            overrides = calibrated_tracking_parameters(robot,hardware=True,motion_mode='simultaneous')
            results[label] = plant.replay(gains,[.38,.34],delay,tau,direction,yaw,1.,
                overrides=overrides,distance=distance,profile='sprint',duration_sec=60.)
    before, after = results['before'], results['after']
    radius = max(math.hypot(*p) for p in robot['footprint'])
    for result in results.values():
        # A circumscribed circle bounds the footprint for every yaw, including
        # between model time samples; subtract maximum lateral deviation.
        result['wall_clearance_lower_bound_m'] = .8-radius-result['cross_track_peak_m']
    accepted = (after['arrival_sec'] is not None and before['arrival_sec'] is not None
                and after['arrival_sec'] <= before['arrival_sec']+.15
                and after['wheel_command_peak'] <= 10000
                and after['wall_clearance_lower_bound_m'] >= .015)
    baseline_feasible = before['arrival_sec'] is not None and before['wall_clearance_lower_bound_m'] >= .015
    regression = baseline_feasible and not accepted
    return dict(baseline_feasible=baseline_feasible, regression=regression, direction_rad=direction,goal_yaw_rad=yaw,delay_sec=delay,tau_sec=tau,
                distance_m=distance,accepted=accepted,**results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--distance',type=float,default=30.)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if not math.isfinite(args.distance) or not 1. <= args.distance <= 100.:
        parser.error('distance must be in [1,100] metres')
    root = plant.ROOT
    audit = root/'docs/latest_four_mps_audit_20260916.json'
    gains = [a['gain'] for a in json.loads(audit.read_text())['runs'][0]['exploratory_fit']]
    package = root/'ros2_ws/src/omni_autonomy_next'
    files = [Path(__file__).resolve(),audit,root/'scripts/compare_field_response.py',
        root/'scripts/compare_field_capture.py', root/'scripts/check_faster_rotating_sprint.py',
        root/'scripts/check_turn_budget_candidates.py', package/'test/test_smooth_arrival.py']
    files += sorted((package/'omni_autonomy_next').glob('*.py'))+sorted((package/'config').glob('*.yaml'))
    hashes = {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    jobs = [(direction,yaw,d,t,gains,args.distance)
        for direction in (0.,math.pi/2,math.pi,-math.pi/2)
        for yaw in (math.pi/2,-math.pi/2)
        for d,t in ((.2,.12),(.3,.15))]
    results = []
    with ProcessPoolExecutor(max_workers=2) as pool:
        for r in pool.map(evaluate,jobs):
            results.append(r)
            print(r['direction_rad'],r['delay_sec'],r['after']['simultaneous_motion'],
                  'cross_track=',r['after']['cross_track_peak_m'],'accepted=',r['accepted'],flush=True)
    for name,digest in hashes.items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'Source changed: {name}')
    report = dict(physical_target_verified=False,accepted=not any(r['regression'] for r in results),
        baseline_feasible=sum(r['baseline_feasible'] for r in results),
        regressions=sum(r['regression'] for r in results),
        sources=hashes,criterion='Arrival without >0.15 s regression, UART <=10000, conservative wall clearance >=0.015 m',
        limitations='Hypothetical 1.6 m wide corridor, low-speed gain extrapolation and assumed delay. '
        'Not the competition field or physical validation; no load, slip, live obstacle monitor or localization faults.',cases=results)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    if not report['accepted']:
        raise SystemExit(1)


if __name__=='__main__':
    main()
