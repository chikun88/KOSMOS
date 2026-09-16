#!/usr/bin/env python3
"""Read-only speed-target audit of real recordings and deployed UART limits.

Never writes calibration or publishes commands. Incomplete recordings and
low-speed extrapolations are diagnostic evidence, not acceptance of a target.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import yaml

from fit_field_response import ROOT, fit_axes, unwrap_tx_us
from omni_autonomy_next.motor_udp_protocol import decode_v4_command
from omni_autonomy_next.robomas_uart import AUTO_UNITS_PER_MPS, AUTO_WHEEL_LIMIT, mix_velocity


def displacement_peak(wheel, window=.5, max_gap=.15):
    """Peak odometry displacement / elapsed time in contiguous windows.

    This is a wheel-based estimate, not independent ground truth. Reject gaps
    and duplicate/backwards acquisition stamps instead of bridging them.
    """
    best = None
    for i in range(len(wheel)):
        j = int(np.searchsorted(wheel[:, 0], wheel[i, 0]-window))
        if j >= i:
            continue
        elapsed = wheel[i, 0]-wheel[j, 0]
        gaps = np.diff(wheel[j:i+1, 0])
        if elapsed < .9*window or np.any(gaps <= 0.) or np.any(gaps > max_gap):
            continue
        speed = float(np.linalg.norm(wheel[i, 4:6]-wheel[j, 4:6])/elapsed)
        if best is None or speed > best['speed_m_s']:
            best = dict(speed_m_s=speed, start_sec=float(wheel[j, 0]),
                        end_sec=float(wheel[i, 0]), samples=i-j+1)
    return best


def target_budget(target, drive, acceleration, deceleration, reaction):
    if not math.isfinite(target) or target <= 0.:
        raise ValueError('target must be positive and finite')
    wire = target*drive['linear_command_scale']
    wheels, saturation = mix_velocity(wire, 0., 0.)
    return dict(target_m_s=target,
        requested_axis_uart_units=wire*AUTO_UNITS_PER_MPS,
        transmitted_axis_uart_peak=max(abs(v) for v in wheels),
        uart_limit=AUTO_WHEEL_LIMIT, uart_scale=saturation,
        unsaturated_axis_request_limit_m_s=AUTO_WHEEL_LIMIT/AUTO_UNITS_PER_MPS/drive['linear_command_scale'],
        required_wire_to_speed_gain=target*AUTO_UNITS_PER_MPS/AUTO_WHEEL_LIMIT,
        acceleration_m_s2=acceleration, braking_deceleration_m_s2=deceleration,
        braking_reaction_sec=reaction,
        modeled_braking_distance_m=target*reaction+target**2/(2*deceleration),
        modeled_accel_and_brake_distance_m=target**2/(2*acceleration)+target*reaction+target**2/(2*deceleration),
        caveat='Distances assume the configured constant acceleration and braking model; no measured stopping distance, jerk, obstacle or clearance allowance.')


def profile_budgets(target, robot, nav, runtime):
    """Use the same per-profile acceleration and separate braking as tracker."""
    drive = robot['drivetrain']
    smoother = nav['velocity_smoother']['ros__parameters']
    profiles = json.loads(runtime['runtime_guard']['ros__parameters']['profiles_json'])
    reaction = robot['calibrated_tracking']['feedback_delay_sec'] + .06
    result = {}
    for name, profile in profiles.items():
        acceleration = min(drive.get('profile_linear_accelerations', {}).get(
            name, smoother['max_accel'][0]), profile['linear_accel'])
        budget = target_budget(target, drive, acceleration,
                               .75*abs(smoother['max_decel'][0]), reaction)
        budget['profile_linear_limit_m_s'] = profile['linear']
        budget['profile_lateral_limit_m_s'] = profile['lateral']
        directions = []
        for degrees in range(0, 360, 45):
            angle = math.radians(degrees)
            wheels, scale = mix_velocity(target*drive['linear_command_scale']*math.cos(angle),
                                         target*drive['linear_command_scale']*math.sin(angle), 0.)
            directions.append(dict(body_direction_deg=degrees, uart_scale=scale,
                transmitted_wheel_peak=max(map(abs, wheels)),
                command_equivalent_speed_m_s=target*scale))
        budget['directions_without_yaw'] = directions
        budget['direction_caveat'] = ('Command equivalents, not measured speeds; concurrent yaw '
                                     'consumes additional wheel budget. Profile, curvature and clearance limits still apply.')
        result[name] = budget
    return result


def inspect(directory, target):
    directory = Path(directory)
    manifest_raw = (directory/'manifest.json').read_bytes()
    manifest = json.loads(manifest_raw)
    if manifest['settings']['operation_mode'] != 'hardware':
        raise ValueError(f'{directory}: hardware recording required')
    summary_raw = (directory/'recording-summary.json').read_bytes()
    summary = json.loads(summary_raw)
    origin = manifest['start_ros_ns']
    data = defaultdict(list)
    counts = Counter()
    planned_peaks = []
    tracker_policy = defaultdict(Counter)
    sources = {}
    snapshots = [(p, p.stat().st_size) for p in sorted(directory.glob('samples-*.jsonl'))]
    for path, size in snapshots:
        digest = hashlib.sha256()
        consumed = 0
        with path.open('rb') as stream:
            while consumed < size:
                raw = stream.readline(size-consumed)
                consumed += len(raw)
                digest.update(raw)
                if not raw.endswith(b'\n'):
                    counts['partial_final_rows'] += 1
                    break
                row = json.loads(raw)
                topic, value = row['topic'], row['value']
                if topic.startswith('/cmd_vel') or topic in ('/motor/network_status', '/wheel/odometry'):
                    if abs(row['received_ros_ns']-row['received_unix_ns']) > 20_000_000:
                        counts['clock_mismatch_rows_excluded'] += 1
                        continue
                if topic == '/wheel/odometry':
                    if row.get('source_ros_ns') is None:
                        counts['missing_wheel_acquisition_time'] += 1
                        continue
                    v = value['twist']['twist']
                    p = value['pose']['pose']['position']
                    data['wheel'].append([(row['source_ros_ns']-origin)*1e-9,
                        v['linear']['x'], v['linear']['y'], v['angular']['z'], p['x'], p['y']])
                elif topic == '/motor/network_status':
                    state = json.loads(value['data'])
                    if state.get('state') != 'JETSON_PACKET_SENT' or state.get('payload_format') != 'v4_uart':
                        continue
                    try:
                        _, velocity, _, _, _, tx = decode_v4_command(bytes(state['packet_bytes']))
                        stamp = unwrap_tx_us(tx, row['received_unix_ns'])
                    except ValueError:
                        counts['invalid_packet_or_tx_clock'] += 1
                        continue
                    data['wire'].append([(stamp-origin)*1e-9, *[v*.001 for v in velocity]])
                    if state.get('wheel_commands') is not None:
                        data['uart'].append([(stamp-origin)*1e-9, max(abs(v) for v in state['wheel_commands'])])
                elif topic.startswith('/cmd_vel'):
                    stamp = row.get('published_unix_ns')
                    if not stamp:
                        counts['missing_command_publication_time'] += 1
                        continue
                    data[topic].append([(stamp-origin)*1e-9,
                        value['linear']['x'], value['linear']['y'], value['angular']['z']])
                elif topic == '/collision_monitor/state':
                    counts['collision_action:'+str(value.get('action_type'))+':'+str(value.get('polygon_name'))] += 1
                elif topic in ('/system/profile', '/system/speed_scale', '/system/safety_state', '/trajectory_tracker/status'):
                    state = value['data']
                    if topic in ('/system/safety_state', '/trajectory_tracker/status'):
                        state = json.loads(state)
                        if topic == '/trajectory_tracker/status' and 'planned_peak_speed' in state:
                            planned_peaks.append(float(state['planned_peak_speed']))
                            for key in ('profile', 'speed_scale', 'acceleration_limit_m_s2',
                                        'braking_limit_m_s2', 'wheel_speed_budget_rad_s'):
                                if key in state:
                                    tracker_policy[key][str(state[key])] += 1
                        state = state.get('reason', state.get('state', 'unknown'))
                    counts[topic+':'+str(state)] += 1
        sources[path.name] = dict(prefix_bytes=consumed, sha256=digest.hexdigest())
    arrays = {}
    for key, rows in data.items():
        a = np.asarray(rows, dtype=float)
        finite = np.isfinite(a).all(axis=1)
        counts['nonfinite_rows_excluded'] += int((~finite).sum())
        a = a[finite]
        if len(a):
            arrays[key] = a[np.argsort(a[:, 0], kind='stable')]
    metrics = {}
    for key, a in arrays.items():
        speed = a[:, 1] if key == 'uart' else np.linalg.norm(a[:, 1:3], axis=1)
        metrics[key] = dict(samples=len(a), units='UART command units' if key == 'uart' else 'm/s',
            p99=float(np.percentile(speed, 99)), maximum=float(speed.max()))
    fit = None
    fit_error = None
    try:
        fit = fit_axes(arrays['wire'], arrays['wheel'][:, :4])
        for axis in fit[:2]:
            axis['extrapolated_axis_speed_at_uart_limit_m_s'] = axis['gain']*AUTO_WHEEL_LIMIT/AUTO_UNITS_PER_MPS
            axis['extrapolated_uart_units_for_target'] = target/axis['gain']*AUTO_UNITS_PER_MPS
    except (ValueError, KeyError) as error:
        fit_error = str(error)
    configs = {}
    for c in manifest['configs']:
        if Path(c['path']).name in ('robot.yaml', 'runtime.yaml'):
            configs[Path(c['path']).name] = dict(sha256=c['sha256'])
            config = yaml.safe_load(c['text'])
            if Path(c['path']).name == 'runtime.yaml':
                configs['runtime.yaml']['profiles'] = json.loads(config['runtime_guard']['ros__parameters']['profiles_json'])
    return dict(session=manifest['session_id'], settings=manifest['settings'],
        recording_at_snapshot=summary,
        metadata_sha256=dict(manifest=hashlib.sha256(manifest_raw).hexdigest(),
                             summary=hashlib.sha256(summary_raw).hexdigest()),
        raw_prefixes=sources, recorded_configs=configs, message_counts=dict(counts), metrics=metrics,
        reported_planned_peak_m_s=max(planned_peaks) if planned_peaks else None,
        observed_tracker_policy={key: dict(values) for key, values in tracker_policy.items()},
        peak_half_second_displacement=(displacement_peak(arrays['wheel']) if 'wheel' in arrays else None),
        exploratory_fit=fit, fit_error=fit_error,
        physical_target_verified=False,
        caveat='Message counts are not durations. Raw velocity maxima can include spikes. Wheel displacement is not independent ground truth. Dropped or open recordings cannot certify calibration. Linear fits extrapolate beyond observed commands and do not identify motor limits.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directories', type=Path, nargs='+')
    parser.add_argument('--target', type=float, default=3.)
    parser.add_argument('--profile', default='sprint', help='Profile for the top-level current budget')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    config = ROOT/'ros2_ws/src/omni_autonomy_next/config'
    paths = [config/'robot.yaml', config/'nav2_next.yaml', config/'runtime.yaml',
             Path(__file__), ROOT/'scripts/fit_field_response.py',
             ROOT/'ros2_ws/src/omni_autonomy_next/omni_autonomy_next/trajectory_tracker_node.py',
             ROOT/'ros2_ws/src/omni_autonomy_next/omni_autonomy_next/robomas_uart.py']
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    robot = yaml.safe_load(paths[0].read_text())['robot']
    nav = yaml.safe_load(paths[1].read_text())
    budgets = profile_budgets(args.target, robot, nav, yaml.safe_load(paths[2].read_text()))
    if args.profile not in budgets:
        parser.error(f'unknown profile: {args.profile}')
    report = dict(kind='Read-only physical speed feasibility audit; no calibration changes',
        sources=hashes, budget_profile=args.profile, budget=budgets[args.profile],
        current_profile_budgets=budgets,
        runs=[inspect(d, args.target) for d in args.directories])
    for p in paths:
        if hashlib.sha256(p.read_bytes()).hexdigest() != hashes[str(p.relative_to(ROOT))]:
            raise RuntimeError(f'Source changed during audit: {p}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report['budget'], indent=2))
    for run in report['runs']:
        print(run['session'], run['peak_half_second_displacement'])


if __name__ == '__main__':
    main()
