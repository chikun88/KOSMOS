#!/usr/bin/env python3
"""Locate short, abrupt stops using publisher clocks, without ROS or robot I/O.

Raw scans are evidence, not the monitor's transformed input. Do not classify a
short detection as false or suppress it merely because adjacent scans differ.
"""
import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path


COMMANDS = ('/cmd_vel_nav_smoothed', '/cmd_vel_rl',
            '/cmd_vel_collision_safe', '/cmd_vel_safe')
STATES = ('/trajectory_tracker/status', '/system/safety_state',
          '/navigation/goal_status')


def twist(value):
    return [value['linear']['x'], value['linear']['y'], value['angular']['z']]


def inspect(directory):
    manifest = json.loads((directory / 'manifest.json').read_text())
    summary = json.loads((directory / 'recording-summary.json').read_text())
    origin = manifest['start_ros_ns']
    series = defaultdict(list)
    states, sources = Counter(), {}
    missing_clock = Counter()
    clock_disagreements = []
    malformed = 0
    snapshots = [(p, p.stat().st_size) for p in sorted(directory.glob('samples-*.jsonl'))]
    for path, size in snapshots:
        digest = hashlib.sha256()
        consumed = 0
        with path.open('rb') as stream:
            while consumed < size:
                line = stream.readline(size - consumed)
                if not line:
                    break
                consumed += len(line)
                digest.update(line)
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    malformed += 1
                    continue
                topic, value = row['topic'], row['value']
                if topic not in (*COMMANDS, *STATES, '/wheel/odometry', '/rosout'):
                    continue
                if 'received_ros_ns' in row and 'received_unix_ns' in row:
                    delta = row['received_ros_ns'] - row['received_unix_ns']
                    if abs(delta) > 20_000_000:
                        clock_disagreements.append(dict(sequence=row.get('sequence'),
                                                       topic=topic, difference_ns=delta))
                stamp = row.get('published_unix_ns')
                if topic == '/wheel/odometry':
                    stamp = row.get('source_ros_ns')
                    value = twist(value['twist']['twist'])
                elif topic == '/rosout':
                    stamp = value['stamp']['sec'] * 10**9 + value['stamp']['nanosec']
                    if value['name'] != 'collision_monitor':
                        continue
                elif topic in COMMANDS:
                    value = twist(value)
                else:
                    try:
                        value = json.loads(value['data'])
                    except (ValueError, KeyError):
                        continue
                    states[topic + ':' + str(value.get('state', value.get('reason')))] += 1
                if not stamp or stamp <= 0:
                    missing_clock[topic] += 1
                    continue
                series[topic].append(((stamp - origin) / 1e9, value))
        sources[path.name] = {'prefix_bytes': consumed, 'sha256': digest.hexdigest()}
    for rows in series.values():
        rows.sort(key=lambda r: r[0])
    times = {topic: [r[0] for r in rows] for topic, rows in series.items()}

    def preceding(topic, stamp, max_age=.25):
        i = bisect_right(times.get(topic, []), stamp) - 1
        if i < 0:
            return None
        at, value = series[topic][i]
        return {'age_sec': stamp - at, 'value': value} if stamp-at <= max_age else None

    events = []
    for topic in COMMANDS:
        rows = series[topic]
        for i in range(1, len(rows)):
            at, command = rows[i]
            before_at, before = rows[i-1]
            if not (0 < at-before_at <= .25 and math.hypot(*before[:2]) > .1
                    and max(map(abs, command)) < 1e-6):
                continue
            recovery = next((t for t, v in rows[i+1:]
                             if t-at > .5 or max(map(abs, v)) > 1e-6), None)
            # Final arrivals and idle STALE_COMMAND messages are not pulses.
            if recovery is None or recovery-at > .5:
                continue
            wheel = series['/wheel/odometry']
            pre = [math.hypot(*v[:2]) for t, v in wheel if at-.2 <= t <= at]
            post = [math.hypot(*v[:2]) for t, v in wheel if at+.05 <= t <= at+.35]
            events.append(dict(
                topic=topic, at_sec=at, zero_duration_sec=recovery-at,
                previous_command=before, interval_sec=at-before_at,
                state_before={t: preceding(t, at) for t in STATES},
                commands_before={t: preceding(t, at) for t in COMMANDS},
                collision_messages=[{'at_sec': t, 'message': v['msg']}
                                    for t, v in series['/rosout'] if abs(t-at) <= .12],
                wheel_speed_before_max=max(pre) if pre else None,
                wheel_speed_after_min=min(post) if post else None))
    return dict(
        session=manifest['session_id'], operation_mode=manifest['settings']['operation_mode'],
        recording_summary=summary, raw_prefixes=sources, malformed_rows=malformed,
        missing_publisher_clock=dict(missing_clock), state_message_counts=dict(states),
        ros_unix_clock_disagreements=clock_disagreements,
        abrupt_stop_pulses=events,
        limitations=[
            'Counts are recorded messages/pulses, not independent goals or failure rates.',
            'Uses DDS publisher UNIX time, wheel ROS acquisition time and rosout source time; '
            'requires a shared ROS/UNIX clock. Receipt time is never substituted.',
            'No drive latency correction; wheel extrema can contain noise or manual motion.',
            'Raw LiDAR alone cannot distinguish self reflection from a real obstacle. '
            'A replay of recorded observations does not predict motion after a code change.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directories', type=Path, nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    reports = [inspect(p) for p in args.directories]
    args.output.write_text(json.dumps(reports, indent=2, ensure_ascii=False) + '\n')
    for report in reports:
        print(report['session'], report['operation_mode'],
              dict(Counter(e['topic'] for e in report['abrupt_stop_pulses'])))


if __name__ == '__main__':
    main()
