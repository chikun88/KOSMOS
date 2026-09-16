#!/usr/bin/env python3
"""Summarize actual wheel commands from a closed run; never infer motor RPM."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np


def inspect(directory):
    summary = json.loads((directory / 'recording-summary.json').read_text())
    if not summary['closed'] or summary['dropped'] or summary['pending'] or summary['error']:
        raise ValueError('A closed, complete recording is required')
    wheels, clock_offsets = [], []
    profiles, calibrations, states = Counter(), Counter(), Counter()
    sources = {}
    for path in sorted(directory.glob('samples-*.jsonl')):
        digest = hashlib.sha256()
        count = 0
        with path.open('rb') as stream:
            for raw in stream:
                digest.update(raw)
                count += 1
                row = json.loads(raw)
                topic, value = row['topic'], row['value']
                if topic == '/system/profile':
                    profiles[value['data']] += 1
                if topic in ('/wheel/odometry', '/motor/network_status'):
                    clock_offsets.append(abs(row['received_ros_ns'] - row['received_unix_ns']) / 1e9)
                if topic == '/motor/network_status':
                    state = json.loads(value['data'])
                    if state.get('state') == 'JETSON_PACKET_SENT' and state.get('wheel_commands') is not None:
                        wheels.append(max(abs(v) for v in state['wheel_commands']))
                        calibrations[json.dumps(state.get('command_calibration'), sort_keys=True)] += 1
                if topic == '/system/safety_state':
                    states[json.loads(value['data'])['reason']] += 1
        sources[path.name] = dict(sha256=digest.hexdigest(), rows=count)
    nonzero = np.asarray([v for v in wheels if v > 0])
    if not len(nonzero):
        raise ValueError('No moving wheel commands')
    return dict(
        session=directory.name, recording=summary, sources=sources,
        sent_packets=len(wheels), nonzero_packets=len(nonzero),
        nonzero_peak_command_percentiles=dict(zip(('p50', 'p90', 'p99', 'max'),
            np.percentile(nonzero, [50, 90, 99, 100]).tolist())),
        profile_events=dict(profiles), calibration_messages=dict(calibrations),
        safety_message_counts=dict(states),
        clock_checks=len(clock_offsets), clock_difference_over_20ms=sum(v > .02 for v in clock_offsets),
        caveats=[
            'Command units are not measured motor RPM; no gear ratio or motor tachometer is established.',
            'Counts describe sampled messages, not motion duration or independent goals.',
            'This summary does not fit response gains or align command and odometry receipt times.',
            'Software command headroom does not establish mechanical, thermal or electrical headroom.',
        ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = inspect(args.directory)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['nonzero_peak_command_percentiles']))


if __name__ == '__main__':
    main()
