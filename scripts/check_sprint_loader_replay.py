#!/usr/bin/env python3
"""Offline loader comparison on 8 m and 60 m empty straight paths.

Source the ROS environment before running. Never creates ROS nodes or publishes
commands. Results do not measure physical speed or current route performance.
"""
import json
import math
from unittest.mock import patch

import yaml

import compare_field_response as plant
from omni_autonomy_next.config import calibrated_tracking_parameters


def main():
    config = plant.ROOT/'ros2_ws/src/omni_autonomy_next/config/robot.yaml'
    robot = yaml.safe_load(config.read_text())['robot']
    tuning = calibrated_tracking_parameters(robot, hardware=True, motion_mode='simultaneous')
    audit = json.loads((plant.ROOT/'docs/four_mps_audit_20260916.json').read_text())
    original_loader = plant.load_robot

    def legacy_loader(path):
        data = original_loader(path)
        data['drivetrain'].pop('profile_max_wheel_speeds', None)
        return data

    results = []
    for run in audit['runs']:
        for distance in (8., 60.):
            for direction in (0., math.pi/2):
                kwargs = dict(
                    gains=[axis['gain'] for axis in run['exploratory_fit']],
                    calibration=[robot['drivetrain']['linear_command_scale'],
                                 robot['drivetrain']['angular_command_scale']],
                    delay=.2, tau=.12, direction=direction, goal_yaw=0., scale=1.,
                    overrides=tuning, distance=distance, profile='sprint',
                    duration_sec=60. if distance == 60. else 30.)
                with patch.object(plant, 'load_robot', legacy_loader):
                    before = plant.replay(**kwargs)
                after = plant.replay(**kwargs)
                results.append(dict(session=run['session'], distance_m=distance,
                                    direction_rad=direction, before=before, after=after))
                print(run['session'], distance, direction, before['peak_speed_m_s'],
                      after['peak_speed_m_s'], flush=True)
    report = dict(
        physical_target_verified=False,
        limitations='Empty hypothetical straight paths, not recorded route results. '
        'Production tracker/guard/UART; low-speed fitted gain extrapolation; delay=.2 tau=.12. '
        'Baseline removes profile_max_wheel_speeds from the loader only.',
        cases=results)
    output = plant.ROOT/'docs/sprint_loader_straight_replay_20260916.json'
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
