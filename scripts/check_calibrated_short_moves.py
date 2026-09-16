#!/usr/bin/env python3
"""Check close goals and in-place turns with the production calibrated tuning."""
import json
import math

import numpy as np
import yaml
from compare_field_response import replay, ROOT
from omni_autonomy_next.config import calibrated_tracking_parameters


def main():
    robot = yaml.safe_load((ROOT/'ros2_ws/src/omni_autonomy_next/config/robot.yaml').read_text())['robot']
    tuning = calibrated_tracking_parameters(robot, hardware=True, motion_mode='simultaneous')
    latest = json.loads((ROOT/'docs/speed_run_diagnostics_20260914.json').read_text())
    gains = np.array([axis['gain'] for axis in latest['exploratory_fit']])
    cases = []
    for distance in [0., .1, .3]:
        for yaw in [0., math.pi/2, -math.pi]:
            for delay, tau in [(.1, .08), (.3, .15)]:
                result = replay(gains, [.38, .34], delay, tau, math.pi/4, yaw, 1.,
                                overrides=tuning, distance=distance)
                cases.append(dict(distance=distance, yaw=yaw, delay=delay, tau=tau, result=result))
                print(distance, yaw, delay, result['arrival_sec'], flush=True)
    accepted = all(c['result']['arrival_sec'] is not None
                   and c['result']['tail_position_error_m'] < .015
                   and c['result']['tail_yaw_error_rad'] < .015
                   and c['result']['physical_acceleration_peak_m_s2'] < 1.3
                   for c in cases)
    (ROOT/'docs/calibrated_short_moves_20260914.json').write_text(json.dumps(
        dict(kind='Offline short-goal checks, exploratory latest gain', candidate=tuning,
             cases=cases, accepted=accepted), indent=2)+'\n')
    print('accepted', accepted)
    if not accepted:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
