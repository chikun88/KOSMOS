#!/usr/bin/env python3
"""Offline closed-loop comparison using measured gains, never motor publishers.

Uses the production tracker tick, guard, and motor-bridge velocity conversion.
The delay and first-order response are uncertainty scenarios, not identified
hardware dynamics. No CAD obstacles, slippage or independent localization.
"""
import argparse
from collections import deque
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'ros2_ws/src/omni_autonomy_next'),
              str(ROOT/'ros2_ws/src/omni_autonomy_next/test')]
from compare_field_capture import make_guard
from test_smooth_arrival import make_node, TRACKER_DEFAULTS
from omni_autonomy_next import trajectory_tracker_node as tracker
from omni_autonomy_next.runtime_guard import RuntimeGuard, GuardHealth
from omni_autonomy_next.motor_udp_bridge_node import MotorUdpBridge
from geometry_msgs.msg import Twist
from omni_autonomy_next.runtime_guard import MotionLimits
from omni_autonomy_next.robomas_uart import mix_velocity
from omni_autonomy_next.config import load_robot
from omni_autonomy_next.motion_metrics import simultaneous_motion_metrics


def transmitted_response(wire, gains):
    """Apply the UART mixer's proportional saturation to the fitted plant.

    Gains were identified from unsaturated wire commands. Above the UART
    budget all three axes are scaled together, including concurrent yaw.
    Quantization and unmeasured high-speed motor nonlinearities are not modeled.
    """
    wheels, saturation = mix_velocity(*wire)
    return wheels, np.asarray(wire) * saturation * np.asarray(gains), saturation


def replay(gains, calibration, delay, tau, direction, goal_yaw, scale,
           overrides=None, distance=2., profile_limits=None, profile='balanced',
           bridge_limits=None, wheel_limit=None, duration_sec=20.):
    goal=np.array([distance*math.cos(direction),distance*math.sin(direction),goal_yaw])
    node=make_node(goal=goal)
    if overrides:
        get_parameter = node.get_parameter
        node.get_parameter = lambda name: (SimpleNamespace(value=overrides[name])
            if name in overrides else get_parameter(name))
    node.speed_scale=scale
    bridge=SimpleNamespace(latest_twist=Twist(),linear_x_sign=1.,linear_y_sign=1.,angular_z_sign=1.,
        max_linear_speed=1.,max_angular_speed=1.8,linear_command_scale=calibration[0],angular_command_scale=calibration[1])
    guard=make_guard(RuntimeGuard)
    # Exercise the same normalized configuration and profile application as
    # the live tracker. Raw-YAML overrides concealed the lost sprint budget.
    drive = load_robot(str(ROOT/'ros2_ws/src/omni_autonomy_next/config/robot.yaml'))['drivetrain']
    node.profiles = {name: dict(linear=p.linear, lateral=p.lateral, angular=p.angular, linear_accel=p.linear_accel)
                     for name, p in guard.profiles.items()}
    node.default_max_wheel_speed = drive['max_wheel_speed']
    node.profile_max_wheel_speeds = drive.get('profile_max_wheel_speeds', {})
    node.default_acceleration = node.acceleration
    node.deceleration = node.acceleration
    node.profile_linear_accelerations = drive.get('profile_linear_accelerations', {})
    tracker.TrajectoryTracker._apply_profile(node, profile)
    if wheel_limit is not None:
        guard.max_wheel_speed = float(wheel_limit)
        guard.profile_max_wheel_speeds[profile] = float(wheel_limit)
        node.envelope.max_wheel = float(wheel_limit)
    # Use the deployed bridge envelope. A fixed 1 m/s here concealed clipping
    # when evaluating a higher profile. Paired historical runs can override it.
    bridge.max_linear_speed, bridge.max_angular_speed = (
        bridge_limits if bridge_limits is not None else
        (max(guard.hard_limits.linear, guard.hard_limits.lateral), guard.hard_limits.angular))
    limits = guard.profiles[profile]
    node.profile_name = profile
    node.speed_limit, node.lateral_limit, node.yaw_limit = (
        limits.linear, limits.lateral, limits.angular)
    if profile_limits is not None:
        node.speed_limit, node.lateral_limit, node.yaw_limit = profile_limits
        guard.profiles[profile] = MotionLimits(
            *profile_limits, limits.linear_accel, limits.angular_accel,
            limits.linear_jerk, limits.angular_jerk)
    actual=np.zeros(3);truth=np.zeros(3)
    pipeline=deque(np.zeros(3) for _ in range(round(delay/.01)))
    rows=[];commands=[];wheel_commands=[];saturations=[]
    rng=np.random.default_rng(13)
    for step in range(round(duration_sec/.01)):
        now=step*.01
        with patch.object(tracker.time,'monotonic',return_value=now):
            node.pose=truth.copy();node.pose[2]+=rng.normal(0.,.0025)
            node.velocity_stamp=node.pose_stamp=now
            if step%2==0:
                sample=actual+rng.normal(0.,[.02,.02,.04])
                node.velocity+=(1.-math.exp(-.02/TRACKER_DEFAULTS['velocity_filter_sec']))*(sample-node.velocity)
            if step%100==0:
                tracker.TrajectoryTracker._build_trajectory(node,np.array([node.pose[:2],goal[:2]]),goal[2])
            if step%5==0:
                tracker.TrajectoryTracker._tick(node)
        safe=guard.step(node.command,now_sec=now,command_age_sec=0.,
            health=GuardHealth(True,False,True,True,True),profile=profile,user_scale=scale,red_zone=False)
        target=np.array(safe.velocity)
        bridge.latest_twist.linear.x=float(target[0]);bridge.latest_twist.linear.y=float(target[1])
        bridge.latest_twist.angular.z=float(target[2])
        wire=np.array(MotorUdpBridge._velocity_from_latest_twist(bridge))
        wheels, response, saturation = transmitted_response(wire, gains)
        wheel_commands.append(wheels)
        saturations.append(saturation)
        pipeline.append(response)
        actual+=(pipeline.popleft()-actual)*(.01/tau)
        c,s=math.cos(truth[2]),math.sin(truth[2])
        truth+=.01*np.array([actual[0]*c-actual[1]*s,actual[0]*s+actual[1]*c,actual[2]])
        commands.append(target)
        rows.append([now,*truth,*actual])
    data=np.array(rows);command=np.array(commands)
    error=np.linalg.norm(data[:,1:3]-goal[:2],axis=1)
    yaw_error=np.abs([tracker.wrap(v-goal_yaw) for v in data[:,3]])
    speed=np.linalg.norm(data[:,4:6],axis=1)
    progress=data[:,1]*math.cos(direction)+data[:,2]*math.sin(direction)
    cruise=speed[(progress >= .3*distance)&(progress <= .7*distance)] if distance >= 1. else np.array([])
    settled=(error<=.015)&(yaw_error<=.015)&(speed<=.025)&(abs(data[:,6])<=.025)
    arrival=None
    for i in range(len(settled)-50):
        if np.all(settled[i:i+50]):arrival=round(float(data[i,0]),3);break
    moving=(np.linalg.norm(command[:,:2],axis=1)>.05)|(abs(command[:,2])>.05)
    metric=dict(arrival_sec=arrival,peak_speed_m_s=float(speed.max()),
        middle_speed_p10_m_s=float(np.percentile(cruise,10)) if len(cruise) else None,
        middle_speed_median_m_s=float(np.median(cruise)) if len(cruise) else None,
        wheel_command_peak=int(np.max(np.abs(wheel_commands))),
        uart_saturation_min=float(min(saturations)),
        uart_saturated_steps=sum(s < 1.-1e-12 for s in saturations),
        velocity_tracking_rmse=np.sqrt(np.mean((data[moving,4:]-command[moving])**2,axis=0)).tolist(),
        cross_track_peak_m=float(np.max(np.abs(data[:,1]*math.sin(direction)-data[:,2]*math.cos(direction)))),
        tail_position_error_m=float(error[-300:].max()),tail_yaw_error_rad=float(yaw_error[-300:].max()),
        physical_acceleration_peak_m_s2=float(np.linalg.norm(np.diff(data[:,4:6],axis=0)/.01,axis=1).max()))
    metric['simultaneous_motion'] = simultaneous_motion_metrics(data[:,0], data[:,4:7])
    return metric


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fit',type=Path,default=ROOT/'docs/field_response_fit_20260914.json')
    parser.add_argument('--output',type=Path,default=ROOT/'docs/field_response_replay_20260914.json')
    args=parser.parse_args()
    fitted=json.loads(args.fit.read_text())
    config=yaml.safe_load((ROOT/'ros2_ws/src/omni_autonomy_next/config/robot.yaml').read_text())['robot']['drivetrain']
    calibration=[config['linear_command_scale'],config['angular_command_scale']]
    cases=[]
    for run in fitted['runs']:
        gains=np.array([r['gain'] for r in run['fit']])
        for delay,tau in [(.10,.08),(.20,.12)]:
            for direction,yaw in [(0.,0.),(math.pi/4,math.pi/2),(math.pi/2,-math.pi)]:
                before=replay(gains,[1.,1.],delay,tau,direction,yaw,.78)
                after=replay(gains,calibration,delay,tau,direction,yaw,.78)
                cases.append(dict(fit_run=run['session_id'],gain=gains.tolist(),delay_s=delay,tau_s=tau,
                                  direction_rad=direction,goal_yaw_rad=yaw,before=before,after=after))
                print('case',len(cases),'arrival',before['arrival_sec'],after['arrival_sec'],flush=True)
    report=dict(kind='offline fitted-gain comparison, not post-change physical validation',
        calibration=calibration,cases=cases,
        accepted=all(c['after']['arrival_sec'] is not None
            and c['after']['tail_position_error_m']<.04 and c['after']['tail_yaw_error_rad']<.035
            and np.linalg.norm(c['after']['velocity_tracking_rmse'])<np.linalg.norm(c['before']['velocity_tracking_rmse'])
            and c['after']['physical_acceleration_peak_m_s2']<c['before']['physical_acceleration_peak_m_s2']
            for c in cases))
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print('accepted:',report['accepted'])
    if not report['accepted']:raise SystemExit(1)


if __name__=='__main__':main()
