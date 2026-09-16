#!/usr/bin/env python3
"""Inspect raw scans near recorded collision-stop pulses; never filter live data.

Run with the workspace ROS environment sourced. The radius/time windows select
near-body diagnostic candidates, not a rule for rejecting obstacle detections.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from omni_autonomy_next.scan_footprint_filter_node import ray_exit_distances


def inspect(report, runs):
    directory = runs / report['session']
    manifest = json.loads((directory / 'manifest.json').read_text())
    robot = next(yaml.safe_load(c['text'])['robot'] for c in manifest['configs']
                 if c['path'].endswith('/robot.yaml'))
    origin_ns = manifest['start_ros_ns']
    footprint = np.asarray(robot['footprint'])
    lidars = {lidar['topic']: lidar for lidar in robot['lidars']}
    events = [e['at_sec'] for e in report['abrupt_stop_pulses']
              if e['topic'] == '/cmd_vel_collision_safe']
    scans = {topic: [] for topic in lidars}
    # Read exactly the prefixes already hashed by inspect_abrupt_stops.py.
    for name, source in report['raw_prefixes'].items():
        remaining = source['prefix_bytes']
        digest = hashlib.sha256()
        with (directory / name).open('rb') as stream:
            while remaining:
                line = stream.readline(remaining)
                if not line:
                    break
                remaining -= len(line)
                digest.update(line)
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                topic = row['topic']
                if topic not in scans or not row.get('published_unix_ns'):
                    continue
                pub = (row['published_unix_ns'] - origin_ns) / 1e9
                if not any(t-.6 <= pub <= t+.3 for t in events):
                    continue
                msg = row['value']
                ranges = np.asarray(msg['ranges'], dtype=float)
                angles = msg['angle_min'] + np.arange(len(ranges)) * msg['angle_increment']
                mount = lidars[topic]['pose']
                sensor = np.array([mount['x'], mount['y']])
                directions = np.c_[np.cos(angles+mount['yaw']), np.sin(angles+mount['yaw'])]
                exits = ray_exit_distances(sensor, directions, footprint)
                # Reproduce the existing 50 mm radial self-mask; do not widen it.
                valid = (np.isfinite(ranges) & (ranges >= msg['range_min'])
                         & (ranges <= msg['range_max'])
                         & (ranges > np.where(exits > 0., exits+.05, 0.)))
                indices = np.flatnonzero(valid)
                points = sensor + ranges[indices, None]*directions[indices]
                indices = indices[np.linalg.norm(points, axis=1) < .85]
                groups = np.split(indices, np.flatnonzero(np.diff(indices) > 1)+1)
                clusters = []
                for group in groups:
                    if not len(group):
                        continue
                    points = sensor + ranges[group, None]*directions[group]
                    clusters.append(dict(
                        beams=[int(group[0]), int(group[-1])], count=len(group),
                        angle_deg=np.rad2deg(angles[group[[0, -1]]]).tolist(),
                        range_m=[float(min(ranges[group])), float(max(ranges[group]))],
                        body_centroid=np.mean(points, axis=0).tolist()))
                scans[topic].append(dict(
                    sequence=row['sequence'], file=name, published_sec=pub,
                    source_sec=(row['source_ros_ns']-origin_ns)/1e9,
                    near_body_clusters=clusters))
        if remaining or digest.hexdigest() != source['sha256']:
            raise ValueError(f'recorded prefix changed: {directory / name}')
    result = []
    for event in events:
        windows = {topic: sorted([row for row in rows
                                  if event-.4 <= row['published_sec'] <= event+.2],
                                 key=lambda r: r['published_sec'])
                   for topic, rows in scans.items()}
        result.append(dict(at_sec=event, scans=windows))
    return dict(session=report['session'], events=result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--runs', type=Path, default=Path.home()/'.ros/omni_autonomy_next/runs')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    reports = json.loads(args.evidence.read_text())
    result = [inspect(report, args.runs) for report in reports
              if report['operation_mode'] == 'hardware']
    args.output.write_text(json.dumps(dict(
        method='Existing radial 50mm self-mask, raw points within 0.85m of base origin; '
               'scan publication window -0.4..+0.2s around a recorded collision-output zero. '
               'Candidate association only: filtered scan delivery/monitor TF are not recorded. '
               'Ranges outside these diagnostic bounds are not judged invalid.',
        runs=result), indent=2)+'\n')
    for run in result:
        for event in run['events']:
            candidates = {topic: [r['sequence'] for r in rows
                                  if r['near_body_clusters'] and r['published_sec'] <= event['at_sec']]
                          for topic, rows in event['scans'].items()}
            print(run['session'], round(event['at_sec'], 3), candidates)


if __name__ == '__main__':
    main()
