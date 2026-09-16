#!/usr/bin/env python3
"""Export command/measurement pairs without ROS or loading the run into memory."""
import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path


COMMANDS = ('/cmd_vel_nav', '/cmd_vel_nav_smoothed', '/cmd_vel_rl',
            '/cmd_vel_collision_safe', '/cmd_vel_safe')


def samples(directory):
    for path in sorted(directory.glob('samples-*.jsonl')):
        with path.open(encoding='utf-8') as stream:
            for number, line in enumerate(stream, 1):
                # A process killed during a write may leave one partial final row.
                if not line.endswith('\n'):
                    yield {'topic': 'recorder/truncated_line', 'file': path.name, 'line': number}
                    continue
                try:
                    row = json.loads(line)
                except ValueError as error:
                    raise ValueError(f'{path}:{number}: {error}') from error
                if row.get('schema') != 2:
                    raise ValueError(f'{path}:{number}: expected schema 2')
                yield row


def twist(value):
    return [value['linear']['x'], value['linear']['y'], value['angular']['z']]


def analyze(directory, max_age=.25):
    directory = Path(directory)
    if not math.isfinite(max_age) or max_age <= 0:
        raise ValueError('max_age must be finite and positive')
    manifest = json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    if not list(directory.glob('samples-*.jsonl')):
        raise ValueError('no samples found')
    columns = ['monotonic_ns', 'source_ros_ns', 'received_ros_ns', 'elapsed_s',
               'odom_vx', 'odom_vy', 'odom_wz']
    for topic in COMMANDS:
        label = topic.removeprefix('/cmd_vel_')
        columns += [label+'_age_s'] + [label+'_'+axis for axis in ('vx', 'vy', 'wz')]
    columns += ['error_vx', 'error_vy', 'error_wz']
    counts, latest, previous, max_gaps = Counter(), {}, {}, {}
    source_ages = {}
    pairs, squares, first, last, missing = 0, [0., 0., 0.], None, None, 0
    with (directory/'control.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in samples(directory):
            topic = row['topic']
            counts[topic] += 1
            if 'monotonic_ns' not in row:
                continue
            if row.get('published_unix_ns', 0) > 0 and 'received_unix_ns' in row:
                age = (row['received_unix_ns']-row['published_unix_ns'])*1e-9
                source_ages[topic] = max(source_ages.get(topic, 0.), age)
            elif 'source_ros_ns' in row and 'received_ros_ns' in row:
                age = (row['received_ros_ns']-row['source_ros_ns'])*1e-9
                source_ages[topic] = max(source_ages.get(topic, 0.), age)
            stamp = row['monotonic_ns']
            if topic in previous:
                max_gaps[topic] = max(max_gaps.get(topic, 0.), (stamp-previous[topic])*1e-9)
            previous[topic] = stamp
            first = stamp if first is None else min(first, stamp)
            last = stamp if last is None else max(last, stamp)
            if topic in COMMANDS:
                latest[topic] = row
            if topic != '/wheel/odometry':
                continue
            measured = twist(row['value']['twist']['twist'])
            result = dict(monotonic_ns=stamp, source_ros_ns=row.get('source_ros_ns', ''),
                          received_ros_ns=row.get('received_ros_ns', ''),
                          elapsed_s=(stamp-first)*1e-9,
                          **dict(zip(('odom_vx', 'odom_vy', 'odom_wz'), measured)))
            safe = None
            for command_topic in COMMANDS:
                command = latest.get(command_topic)
                if command is None:
                    continue
                age = (stamp-command['monotonic_ns'])*1e-9
                label = command_topic.removeprefix('/cmd_vel_')
                result[label+'_age_s'] = age
                if not 0. <= age <= max_age:
                    continue
                values = twist(command['value'])
                if not all(isinstance(v, (float, int)) and math.isfinite(v) for v in values):
                    continue
                result.update({label+'_'+axis: value for axis, value in zip(('vx','vy','wz'), values)})
                if command_topic == '/cmd_vel_safe':
                    safe = values
            if safe is not None and all(isinstance(v, (float, int)) and math.isfinite(v) for v in measured):
                errors = [m-c for m, c in zip(measured, safe)]
                result.update(dict(zip(('error_vx', 'error_vy', 'error_wz'), errors)))
                squares = [s+e*e for s, e in zip(squares, errors)]
                pairs += 1
            else:
                missing += 1
            writer.writerow(result)
    summary_path = directory/'recording-summary.json'
    summary = dict(schema=1, session_id=manifest['session_id'],
                   operation_mode=manifest['settings']['operation_mode'],
                   duration_s=(last-first)*1e-9 if first is not None else 0.,
                   topics=dict(counts), max_receipt_gap_s=max_gaps,
                   max_source_to_callback_age_s=source_ages,
                   receipt_alignment_warning=(
                       'Source/callback delay exceeds 250 ms. Do not infer drive calibration from this CSV; '
                       'use publication/acquisition timestamps (fit_field_response.py for recorded v4 hardware).'
                       if max((source_ages.get(t, 0.) for t in
                           (*COMMANDS, '/wheel/odometry', '/localization/pose')), default=0.) > .25 else None),
                   alignment='latest previously received command; no delay compensation',
                   max_command_age_s=max_age, paired_odometry_samples=pairs,
                   unpaired_odometry_samples=missing,
                   velocity_rmse=dict(zip(('vx_m_s', 'vy_m_s', 'wz_rad_s'),
                       [math.sqrt(s/pairs) if pairs else None for s in squares])),
                   recording=json.loads(summary_path.read_text()) if summary_path.exists() else None)
    (directory/'analysis.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--max-command-age', type=float, default=.25)
    args = parser.parse_args()
    summary = analyze(args.directory, args.max_command_age)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f'CSV: {args.directory / "control.csv"}')


if __name__ == '__main__':
    main()
