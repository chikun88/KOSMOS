#!/usr/bin/env python3
"""Inspect a bounded prefix of detailed hardware logs without touching the robot.

An open recording is diagnostic evidence only. Fits are exploratory and never
write calibration. Times are v4 transmit time and odometry acquisition time.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from fit_field_response import fit_axes, unwrap_tx_us
from omni_autonomy_next.motor_udp_protocol import decode_v4_command


def inspect(directory):
    manifest = json.loads((directory/'manifest.json').read_text())
    if manifest['settings']['operation_mode'] != 'hardware':
        raise ValueError('hardware recording required')
    summary = json.loads((directory/'recording-summary.json').read_text())
    snapshots = [(p, p.stat().st_size) for p in sorted(directory.glob('samples-*.jsonl'))]
    origin = manifest['start_ros_ns']
    commands, wheels, ages = [], [], []
    statuses, scales = Counter(), Counter()
    sources = {}
    malformed = 0
    for path, size in snapshots:
        digest = hashlib.sha256()
        consumed = 0
        with path.open('rb') as stream:
            while consumed < size:
                raw = stream.readline(size-consumed)
                consumed += len(raw)
                if not raw:
                    break
                digest.update(raw)
                try:
                    row = json.loads(raw)
                except json.JSONDecodeError:
                    malformed += 1
                    continue
                topic, value = row['topic'], row['value']
                if topic in ('/motor/network_status', '/wheel/odometry'):
                    if abs(row['received_ros_ns']-row['received_unix_ns']) > 20_000_000:
                        raise ValueError('ROS and UNIX clocks disagree')
                if topic == '/motor/network_status':
                    state = json.loads(value['data'])
                    if state.get('state') == 'JETSON_PACKET_SENT' and state.get('payload_format') == 'v4_uart':
                        _, velocity, _, _, _, tx = decode_v4_command(bytes(state['packet_bytes']))
                        stamp = unwrap_tx_us(tx, row['received_unix_ns'])
                        commands.append([(stamp-origin)/1e9, *[v*.001 for v in velocity]])
                elif topic == '/wheel/odometry':
                    v = value['twist']['twist']
                    wheels.append([(row['source_ros_ns']-origin)/1e9,
                                   v['linear']['x'], v['linear']['y'], v['angular']['z']])
                    ages.append((row['received_ros_ns']-row['source_ros_ns'])/1e9)
                elif topic == '/system/speed_scale':
                    scales[str(value['data'])] += 1
                elif topic in ('/trajectory_tracker/status', '/system/safety_state', '/navigation/goal_status'):
                    try:
                        state = json.loads(value['data'])
                    except (ValueError, KeyError):
                        continue
                    statuses[topic+':'+str(state.get('state', state.get('reason', 'unknown')))] += 1
        sources[path.name] = dict(prefix_bytes=consumed, sha256=digest.hexdigest())
    command, wheel = np.asarray(commands), np.asarray(wheels)
    if len(command) < 50 or len(wheel) < 50 or not np.isfinite(command).all() or not np.isfinite(wheel).all():
        raise ValueError('insufficient or nonfinite samples')
    fitted = fit_axes(command[np.argsort(command[:, 0])], wheel[np.argsort(wheel[:, 0])])
    return dict(session=manifest['session_id'], recording_at_snapshot=summary,
                caveat='Bounded log prefixes; open/incomplete recordings cannot certify calibration or success rates. Status counts are messages, not independent goals. Wheel peaks may include outliers.',
                exploratory_fit=fitted, raw_prefixes=sources, malformed_rows=malformed,
                status_counts=dict(statuses), scale_events=dict(scales),
                wheel_speed_percentiles=dict(zip(('p50', 'p90', 'p99', 'max'),
                    np.percentile(np.linalg.norm(wheel[:, 1:3], axis=1), [50, 90, 99, 100]).tolist())),
                wheel_receipt_age_percentiles=dict(zip(('p50', 'p90', 'p99'),
                    np.percentile(ages, [50, 90, 99]).tolist())))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = inspect(args.directory)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report['exploratory_fit'], indent=2))


if __name__ == '__main__':
    main()
