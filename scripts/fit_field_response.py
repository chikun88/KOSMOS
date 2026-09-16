#!/usr/bin/env python3
"""Fit wire-command to wheel response using publication/acquisition clocks.

Read-only offline analysis. Does not modify calibration or drive the robot.
Legacy records use the v4 packet's CRC-validated transmit timestamp, never
headerless /cmd_vel_safe callback time (which can lag by seconds).
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'ros2_ws/src/omni_autonomy_next'))
from omni_autonomy_next.motor_udp_protocol import decode_v4_command


def unwrap_tx_us(tx, received_ns):
    """Resolve the 32-bit microsecond clock against same-host UNIX receipt."""
    received_us = received_ns // 1000
    age_us = (received_us-int(tx)) & 0xffffffff
    if age_us > 10_000_000:
        raise ValueError('v4 transmit timestamp is future, stale, or on another clock')
    return (received_us-age_us)*1000


def read_run(directory):
    directory = Path(directory)
    manifest = json.loads((directory/'manifest.json').read_text())
    if manifest['settings']['operation_mode'] != 'hardware':
        raise ValueError('response fitting requires a hardware run')
    summary = json.loads((directory/'recording-summary.json').read_text())
    if not summary.get('closed') or summary.get('error') or summary.get('pending') or summary.get('dropped'):
        raise ValueError('recording is incomplete; inspect it before fitting')
    origin = manifest['start_ros_ns']
    commands, wheel, pose = [], [], []
    ages = {'wheel_callback': [], 'command_callback': []}
    hashes = {}
    for path in sorted(directory.glob('samples-*.jsonl')):
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for raw in stream:
                digest.update(raw)
                row = json.loads(raw)
                topic, value = row['topic'], row['value']
                if topic not in ('/motor/network_status', '/wheel/odometry', '/localization/pose'):
                    continue
                # Header acquisition uses ROS time, v4 uses UNIX time. Reject
                # simulated/jumped clocks instead of fitting across domains.
                if abs(row['received_ros_ns']-row['received_unix_ns']) > 20_000_000:
                    raise ValueError('ROS and UNIX clocks disagree')
                if topic == '/motor/network_status':
                    state = json.loads(value['data'])
                    if state.get('state') != 'JETSON_PACKET_SENT' or state.get('payload_format') != 'v4_uart':
                        continue
                    _, velocity, _, _, _, tx = decode_v4_command(bytes(state['packet_bytes']))
                    stamp = unwrap_tx_us(tx, row['received_unix_ns'])
                    commands.append([(stamp-origin)*1e-9, *[v*.001 for v in velocity]])
                    ages['command_callback'].append((row['received_unix_ns']-stamp)*1e-9)
                elif topic == '/wheel/odometry':
                    v = value['twist']['twist']
                    wheel.append([(row['source_ros_ns']-origin)*1e-9,
                                  v['linear']['x'], v['linear']['y'], v['angular']['z']])
                    ages['wheel_callback'].append((row['received_ros_ns']-row['source_ros_ns'])*1e-9)
                else:
                    p = value['pose']['pose']; q = p['orientation']
                    yaw = math.atan2(2*(q['w']*q['z']+q['x']*q['y']), 1-2*(q['y']**2+q['z']**2))
                    pose.append([(row['source_ros_ns']-origin)*1e-9, p['position']['x'], p['position']['y'], yaw])
        hashes[path.name] = digest.hexdigest()
    arrays = []
    for rows in (commands, wheel, pose):
        a = np.asarray(rows, dtype=float)
        if len(a) < 20 or not np.isfinite(a).all():
            raise ValueError('insufficient or invalid recorded data')
        arrays.append(a[np.argsort(a[:, 0], kind='stable')])
    return arrays, dict(session_id=manifest['session_id'], raw_sha256=hashes,
                       recording=summary, receipt_age_s={k: dict(zip(('median','p90','p99'),
                           np.percentile(v, [50,90,99]).tolist())) for k,v in ages.items()})


def pairs(command, wheel, axis, lag):
    indices = np.searchsorted(command[:, 0], wheel[:, 0]-lag, side='right')-1
    age = wheel[:, 0]-lag-command[indices, 0]
    valid = (indices >= 0) & (age >= 0.) & (age <= .15) & (abs(command[indices, axis+1]) > .08)
    return command[indices[valid], axis+1], wheel[valid, axis+1]


def fit_axes(command, wheel):
    results = []
    for axis in range(3):
        candidates = []
        for lag in np.arange(0., .501, .01):
            x, y = pairs(command, wheel, axis, lag)
            if len(x) < 50:
                continue
            gain = float(np.dot(x,y)/np.dot(x,x))
            error = float(np.sqrt(np.mean((y-gain*x)**2)))
            candidates.append((error, float(lag), gain, len(x)))
        if not candidates:
            raise ValueError(f'insufficient excitation on axis {axis}')
        rmse, lag, gain, count = min(candidates)
        if not .5 < gain < 5. or lag >= .5:
            raise ValueError('fit outside validated search range')
        x, y = pairs(command, wheel, axis, lag)
        signs = {}
        for label, mask in [('positive',x>0), ('negative',x<0)]:
            signs[label] = dict(samples=int(mask.sum()), gain=(float(np.dot(x[mask],y[mask])/np.dot(x[mask],x[mask]))
                                                              if mask.sum() >= 20 else None))
        results.append(dict(axis=('vx','vy','wz')[axis], gain=gain, effective_lag_s=lag,
                            residual_rmse=rmse, samples=count, directions=signs))
    return results


def pose_agreement(wheel, pose):
    """Cross-check fused pose displacement; this is not independent ground truth."""
    pose = pose.copy(); pose[:,3] = np.unwrap(pose[:,3])
    times = np.arange(pose[0,0]+.5, pose[-1,0]-.5, .1)
    xs, ys = [], []
    for t in times:
        section = wheel[(wheel[:,0] > t-.4) & (wheel[:,0] <= t),1:]
        if len(section) < 3:
            continue
        a = np.array([np.interp(t,pose[:,0],pose[:,k]) for k in (1,2,3)])
        b = np.array([np.interp(t-.4,pose[:,0],pose[:,k]) for k in (1,2,3)])
        v = (a-b)/.4; yaw = (a[2]+b[2])*.5
        body = [v[0]*math.cos(yaw)+v[1]*math.sin(yaw),
                -v[0]*math.sin(yaw)+v[1]*math.cos(yaw),v[2]]
        avg = section.mean(axis=0)
        if np.linalg.norm(avg[:2]) > .2:
            xs.append(avg); ys.append(body)
    x,y=np.array(xs),np.array(ys)
    return dict(samples=len(x), gain=(np.sum(x*y,axis=0)/np.sum(x*x,axis=0)).tolist(),
                rmse=np.sqrt(np.mean((x-y)**2,axis=0)).tolist(),
                caveat='Localization is fused with wheel odometry, not independent ground truth.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('training_run',type=Path)
    parser.add_argument('validation_run',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.training_run.resolve() == args.validation_run.resolve():
        raise ValueError('validation must be a separate run')
    runs=[]
    for directory in (args.training_run,args.validation_run):
        (command,wheel,pose),report=read_run(directory)
        report['fit']=fit_axes(command,wheel)
        report['pose_wheel_agreement']=pose_agreement(wheel,pose)
        runs.append(report)
    train=runs[0]['fit']; scales=[math.floor(100/max(train[0]['gain'],train[1]['gain']))/100,
                                math.floor(100/train[2]['gain'])/100]
    report=dict(kind='measured response fit, not a post-change hardware result',
                clock='v4 packet UNIX transmit time / wheel ROS acquisition time (same-host clocks checked)',
                alignment='past wire command, <=150 ms command gap; |axis command| >0.08',
                model='per-axis linear gain and effective lag; no friction or dynamic plant identification',
                runs=runs, suggested_scales=dict(linear=scales[0],angular=scales[1]),
                validation_predicted_effective_gains=[r['gain']*s for r,s in zip(runs[1]['fit'],[scales[0],scales[0],scales[1]])])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(output=str(args.output),suggested_scales=report['suggested_scales'],
                         validation_predicted_effective_gains=report['validation_predicted_effective_gains']),indent=2))


if __name__=='__main__':
    main()
