#!/usr/bin/env python3
"""Replay recorded scans through installed Nav2 geometry without a ROS graph.

Source /opt/ros/jazzy/setup.bash and ros2_ws/install/setup.bash first.
This tests the instantaneous scan geometry, not the robot's subsequent motion
or the monitor's historical TF/scan transport. Original scans remain intact.
"""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import yaml

from omni_autonomy_next.config import load_robot
from omni_autonomy_next.scan_self_reflections import self_reflection_mask

ROOT = Path(__file__).resolve().parents[1]


def scan_points(scan, lidar, filtered):
    ranges = np.asarray(scan['ranges'], float)
    keep = np.isfinite(ranges) & (ranges >= scan['range_min']) & (ranges <= scan['range_max'])
    if filtered:
        keep &= ~self_reflection_mask(ranges, scan['angle_min'], scan['angle_increment'],
                                     lidar['self_reflection_windows'])
    angles = scan['angle_min'] + np.arange(len(ranges))*scan['angle_increment'] + lidar['pose']['yaw']
    return np.column_stack((lidar['pose']['x']+ranges[keep]*np.cos(angles[keep]),
                            lidar['pose']['y']+ranges[keep]*np.sin(angles[keep])))


def replay():
    config = ROOT/'ros2_ws/src/omni_autonomy_next/config'
    robot = load_robot(str(config/'robot.yaml'))
    monitor = yaml.safe_load((config/'nav2_next.yaml').read_text())['collision_monitor']['ros__parameters']
    approach = monitor['FootprintApproach']
    fixture = json.loads((ROOT/'docs/slowdown_scan_fixture_20260915.json').read_text())
    # The fixture's mount and body geometry must still describe this setup.
    np.testing.assert_allclose(robot['footprint'], fixture['footprint'])
    for current, recorded in zip(robot['lidars'], fixture['lidars']):
        assert current['pose'] == recorded['pose']
    cases = []
    for sample in fixture['samples']:
        cmd = sample['/cmd_vel_rl']['value']
        velocity = [cmd['linear']['x'], cmd['linear']['y'], cmd['angular']['z']]
        for filtered in (False, True):
            points = np.vstack([scan_points(sample[lidar['topic']+'_filtered']['value'], lidar, filtered)
                                for lidar in robot['lidars']])
            cases.append((f"{sample['received_sec']}s_{'after' if filtered else 'before'}", velocity, points))
    # An obstacle on the same rear bearing outside the surveyed range must
    # remain visible and stop/brake the very same reverse command.
    rear = robot['lidars'][1]
    for distance in (.32, .50):
        scan = dict(angle_min=np.deg2rad(26.), angle_increment=0., ranges=[distance],
                    range_min=.15, range_max=12.)
        points = scan_points(scan, rear, True)
        assert len(points) == 1
        cases.append((f'obstacle_{distance}m', [-.8, 0., 0.], points))
    cases.append(('inside_body', [-.8, 0., 0.], np.array([[-.42, .1]])))
    cases.append(('clear', [-.8, 0., 0.], np.empty((0, 2))))
    header = [approach['time_before_collision'], approach['simulation_time_step'],
              approach['min_points'], len(robot['footprint']), *robot['footprint'].ravel()]
    lines = [' '.join(map(str, header))]
    slow_zone = monitor['SlowZone']
    slow_points = np.asarray(json.loads(slow_zone['points']), float)
    lines.append(' '.join(map(str, [approach['time_before_collision'],
                                   approach['simulation_time_step'], slow_zone['min_points'],
                                   len(slow_points), *slow_points.ravel()])))
    for _, velocity, points in cases:
        lines.append(' '.join(map(str, [*velocity, len(points), *points.ravel()])))
    with tempfile.TemporaryDirectory(prefix='slowdown-replay-') as name:
        directory = Path(name)
        (directory/'CMakeLists.txt').write_text(
            'cmake_minimum_required(VERSION 3.16)\nproject(slowdown_probe)\n'
            'find_package(ament_cmake REQUIRED)\nfind_package(nav2_collision_monitor REQUIRED)\n'
            f'add_executable(probe "{ROOT / "scripts/slowdown_collision_probe.cpp"}")\n'
            'ament_target_dependencies(probe nav2_collision_monitor)\n')
        subprocess.run(['cmake', '-S', str(directory), '-B', str(directory/'build')],
                       check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        subprocess.run(['cmake', '--build', str(directory/'build'), '-j2'],
                       check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        run = subprocess.run([str(directory/'build/probe')], input='\n'.join(lines)+'\n',
                             text=True, capture_output=True, check=True)
    times = [list(map(float, line.split())) for line in run.stdout.splitlines()]
    assert len(times) == len(cases), run.stdout
    results = {}
    for (name, velocity, points), (collision_time, slow_hits) in zip(cases, times):
        ratio = 1. if collision_time < 0. else collision_time/approach['time_before_collision']
        monitor_ratio = min(ratio, slow_zone['slowdown_ratio']
                            if slow_hits >= slow_zone['min_points'] else 1.)
        results[name] = dict(points=len(points), collision_time=collision_time,
                             approach_ratio=ratio,
                             monitor_ratio=monitor_ratio,
                             monitor_linear_speed=float(np.linalg.norm(velocity[:2])*monitor_ratio))
    for at in (160, 175, 200):
        assert results[f'{at}s_before']['approach_ratio'] <= .1
        assert results[f'{at}s_after']['approach_ratio'] > .5
    for key in ('obstacle_0.32m', 'obstacle_0.5m', 'inside_body'):
        assert 0. <= results[key]['approach_ratio'] < .5
    assert results['inside_body']['approach_ratio'] == 0.
    assert results['clear']['approach_ratio'] == 1.
    return dict(session=fixture['session'], results=results,
                limitation='Native instantaneous geometry replay; does not reproduce historical '
                'TF/transport or predict subsequent physical motion. Includes unchanged SlowZone.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = replay()
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
