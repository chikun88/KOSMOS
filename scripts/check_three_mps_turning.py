#!/usr/bin/env python3
"""Require sustained 3 m/s while rotating in an explicitly empty model corridor."""
import argparse
import hashlib
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import yaml
import compare_field_response as plant
from omni_autonomy_next.config import calibrated_tracking_parameters
from omni_autonomy_next.rl_residual import CadClearanceModel


def evaluate(job):
    direction, yaw, delay, tau, gains, distance = job
    robot = yaml.safe_load((plant.ROOT/'ros2_ws/src/omni_autonomy_next/config/robot.yaml').read_text())['robot']
    make_node = plant.make_node
    results = {}
    for label, enabled in (('before',False),('after',True)):
        def factory(*args,**kwargs):
            node = make_node(*args,**kwargs)
            node.clearance = CadClearanceModel([[[-200.,-200.],[200.,-200.]]],robot['footprint'])
            get = node.get_parameter
            node.get_parameter = lambda name: SimpleNamespace(value=enabled) if name=='predictive_sprint' else get(name)
            return node
        with patch.object(plant,'make_node',factory):
            overrides = calibrated_tracking_parameters(robot,hardware=True,motion_mode='simultaneous')
            overrides['predictive_sprint'] = enabled
            results[label] = plant.replay(gains,[.38,.34],delay,tau,direction,yaw,1.,
                overrides=overrides,distance=distance,profile='sprint',duration_sec=60.)
    after = results['after']
    accepted = (after['arrival_sec'] is not None and after['wheel_command_peak'] <= 10000
                and after['cross_track_peak_m'] <= .35
                and after['simultaneous_motion']['simultaneous_longest_sec'] >= 1.)
    return dict(direction_rad=direction,goal_yaw_rad=yaw,delay_sec=delay,tau_sec=tau,
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
        root/'scripts/compare_field_capture.py',package/'test/test_smooth_arrival.py']
    files += sorted((package/'omni_autonomy_next').glob('*.py'))+sorted((package/'config').glob('*.yaml'))
    hashes = {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    jobs = [(direction,yaw,d,t,gains,args.distance) for direction,yaw in (
        (0.,math.pi/2),(math.pi/4,-math.pi/2),(math.pi/2,math.pi/2),(-math.pi/4,-math.pi/2),
        (math.pi,math.pi/2),(3*math.pi/4,-math.pi/2),(-math.pi/2,math.pi/2),(-3*math.pi/4,-math.pi/2))
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
    report = dict(physical_target_verified=False,accepted=all(r['accepted'] for r in results),
        sources=hashes,criterion='model truth speed >=3 m/s AND abs(yaw rate)>=0.05 rad/s continuously for >=1 s; arrival and <=0.35 m cross-track error',
        limitations='Hypothetical empty corridor, low-speed gain extrapolation and assumed delay. '
        'Not the competition field or physical validation; no load, slip, live obstacle monitor or localization faults.',cases=results)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    if not report['accepted']:
        raise SystemExit(1)


if __name__=='__main__':
    main()
