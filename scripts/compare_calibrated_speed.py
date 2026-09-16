#!/usr/bin/env python3
"""Paired offline speed tuning against measured gains; no ROS publishers."""
import json
import math

import numpy as np
from compare_field_response import replay, ROOT


def main():
    fitted = json.loads((ROOT / 'docs/field_response_fit_20260914.json').read_text())
    import yaml
    from omni_autonomy_next.config import calibrated_tracking_parameters
    robot = yaml.safe_load((ROOT/'ros2_ws/src/omni_autonomy_next/config/robot.yaml').read_text())['robot']
    candidate = calibrated_tracking_parameters(robot, hardware=True, motion_mode='simultaneous')
    latest = json.loads((ROOT/'docs/speed_run_diagnostics_20260914.json').read_text())
    fitted['runs'].append(dict(session_id=latest['session'] + ' (exploratory incomplete prefix)', fit=latest['exploratory_fit']))
    cases = []
    for run in fitted['runs']:
        gains = np.array([axis['gain'] for axis in run['fit']])
        for delay, tau in [(.10, .08), (.20, .12), (.30, .15)]:
            for direction, yaw in [(0., 0.), (math.pi/4, math.pi/2), (math.pi/2, -math.pi)]:
                for scale in [.78, 1.]:
                    args = (gains, [.38, .34], delay, tau, direction, yaw, scale)
                    before = replay(*args, overrides={'position_gain': .8, 'yaw_gain': 1., 'feedback_delay_sec': .2})
                    after = replay(*args, overrides=candidate)
                    cases.append(dict(session=run['session_id'], delay=delay, tau=tau,
                                      direction=direction, yaw=yaw, scale=scale,
                                      before=before, after=after))
                    print(len(cases), before['arrival_sec'], after['arrival_sec'], flush=True)
    accepted = all(c['after']['arrival_sec'] is not None
                   and c['after']['tail_position_error_m'] < .015
                   and c['after']['tail_yaw_error_rad'] < .015
                   and c['after']['cross_track_peak_m'] < .06
                   and c['after']['physical_acceleration_peak_m_s2'] < 1.3
                   for c in cases)
    improvements = [(1-c['after']['arrival_sec']/c['before']['arrival_sec'])*100
                    for c in cases if c['after']['arrival_sec'] and c['before']['arrival_sec']]
    accepted = accepted and float(np.median(improvements)) > 5.
    report = dict(kind='Offline uncertainty sweep, not a physical speed measurement',
                  limitations='No CAD obstacles, slip, localization failures or real downstream collision monitor.',
                  baseline={'position_gain': .8, 'yaw_gain': 1., 'feedback_delay_sec': .2},
                  candidate=candidate, cases=cases, accepted=accepted,
                  improvement_percent=dict(min=min(improvements), median=float(np.median(improvements)), max=max(improvements)))
    (ROOT/'docs/calibrated_speed_replay_20260914.json').write_text(json.dumps(report, indent=2)+'\n')
    print(report['improvement_percent'], 'accepted', accepted)
    if not accepted:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
