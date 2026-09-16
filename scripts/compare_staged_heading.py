#!/usr/bin/env python3
"""Paired delayed-plant replay of the actual tracker tick, without motors.

Uses the repository's ROS test fixtures. The live-map gate is supplied as free
space: this measures control stability, not full-stack collision acceptance.
"""
import json
import math
from pathlib import Path
import sys
from collections import deque
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'ros2_ws/src/omni_autonomy_next/test'),
                str(ROOT/'ros2_ws/src/omni_autonomy_next')]
from test_staged_heading import staged_node
from test_smooth_arrival import make_node
from omni_autonomy_next import trajectory_tracker_node as tracker


def replay(mode, gain, delay, tau, angle):
    goal = np.array([2., .5, angle])
    node = staged_node(goal=goal) if mode == 'staged_heading' else make_node(goal=goal)
    truth, actual = node.pose.copy(), np.zeros(3)
    pipe = deque([np.zeros(3) for _ in range(round(delay/.01))])
    tail = []
    settled_since, arrived = None, None
    max_speed = 0.
    with pytest.MonkeyPatch.context() as patch:
        for step in range(3000):
            now = step*.01
            patch.setattr(tracker.time,'monotonic',lambda: now)
            node.pose = truth.copy()
            node.pose_stamp = node.velocity_stamp = now
            node.velocity += (1-math.exp(-.01/.06))*(actual-node.velocity)
            if step % 100 == 0 or node.plan_event.is_set():
                node.plan_event.clear()
                node.last_plan = (np.array([truth[:2],goal[:2]]),angle)
                tracker.TrajectoryTracker._build_trajectory(node,*node.last_plan)
            if step % 5 == 0:
                tracker.TrajectoryTracker._tick(node)
            pipe.append(node.command.copy())
            actual += (pipe.popleft()*gain-actual)*(.01/tau)
            c,s = math.cos(truth[2]),math.sin(truth[2])
            truth += .01*np.array([actual[0]*c-actual[1]*s,actual[1]*c+actual[0]*s,actual[2]])
            error = float(np.linalg.norm(truth[:2]-goal[:2]))
            yaw_error = abs(tracker.wrap(truth[2]-angle))
            stopped = np.linalg.norm(actual[:2]) < .03 and abs(actual[2]) < .025
            if error < .04 and yaw_error < .035 and stopped:
                if settled_since is None:
                    settled_since = now
                if now-settled_since >= .5 and arrived is None:
                    arrived = settled_since
            else:
                settled_since = None
            max_speed = max(max_speed,float(np.linalg.norm(actual[:2])))
            if now >= 27.:
                tail.append((error,yaw_error))
    tail = np.array(tail)
    return dict(mode=mode, gain=gain, delay_s=delay, tau_s=tau,
                turn_deg=round(math.degrees(angle)), arrived_s=arrived,
                tail_position_mm=round(float(tail[:,0].max())*1000,3),
                tail_yaw_deg=round(math.degrees(float(tail[:,1].max())),4),
                max_actual_speed_mps=round(max_speed,3), final_status=node.statuses[-1])


if __name__ == '__main__':
    records = [replay(mode,gain,delay,tau,angle)
               for gain,delay,tau in [(1.,.12,.08),(2.8,.2,.12),(2.8,.3,.15)]
               for angle in [math.pi/2,-math.pi]
               for mode in ['simultaneous','staged_heading']]
    path = ROOT/'docs/staged_heading_comparison.json'
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(dict(model='open space delayed plant, actual tracker tick; not hardware',
                                   records=records),indent=2)+'\n')
    print(json.dumps(records,indent=2))
