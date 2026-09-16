#!/usr/bin/env python3
"""Isolated ROS synthetic integration check; never enables motor output."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--initial-pose-id', default='1')
    parser.add_argument('--goal-id', default='4')
    parser.add_argument('--remembered-name', default=None)
    parser.add_argument('--next-remembered-name', default=None)
    parser.add_argument('--switch-after-sec', type=float, default=15.)
    parser.add_argument('--remembered-poses-file', type=Path,
                        default=Path.home()/'.ros/omni_autonomy_next/remembered_poses.json')
    parser.add_argument('--mode', choices=['simultaneous', 'staged_heading'], default='staged_heading')
    parser.add_argument('--delay', type=float, default=.12)
    parser.add_argument('--tau', type=float, default=.08)
    parser.add_argument('--seconds', type=float, default=85.)
    parser.add_argument('--output', default='/tmp/staged-heading-demo.json')
    args = parser.parse_args()
    if args.next_remembered_name and not args.remembered_name:
        parser.error('--next-remembered-name requires --remembered-name')
    os.environ['ROS_DOMAIN_ID'] = '91'
    os.environ['ROS_LOCALHOST_ONLY'] = '1'
    import rclpy
    from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
    from std_msgs.msg import String
    from nav_msgs.msg import Path as NavPath, Odometry
    from geometry_msgs.msg import PoseWithCovarianceStamped
    from ament_index_python.packages import get_package_share_directory
    from omni_autonomy_next.configured_goals import load_configured_poses, load_remembered_defaults
    from omni_autonomy_next.remembered_poses import load_remembered_poses, resolve_remembered_pose
    config = Path(get_package_share_directory('omni_autonomy_next'))/'config'
    poses = load_configured_poses(config/'field_poses.yaml')
    if args.remembered_name:
        saved = {**load_remembered_defaults(config/'field_poses.yaml', poses),
                 **load_remembered_poses(args.remembered_poses_file)}
        target_name = args.next_remembered_name or args.remembered_name
        target = resolve_remembered_pose(saved[target_name], 'left')
    else:
        target = poses[str(args.goal_id)]
    output = Path(args.output)
    records = []
    geometry = {'pose_trace': [], 'wheel_trace': []}
    started = time.monotonic()
    rclpy.init()
    node = rclpy.create_node('staged_heading_demo_check')
    remembered_publisher = node.create_publisher(String, '/navigation/remembered_goal_request', 5)
    remembered_sent = False
    replacement_sent = False
    qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                     reliability=ReliabilityPolicy.RELIABLE)
    def record(topic, message):
        data = json.loads(message.data)
        records.append(dict(time=round(time.monotonic()-started,3),topic=topic,**data))
    def succeeded(record):
        return (record.get('state') == 'SUCCEEDED'
                and (not args.remembered_name or record.get('remembered_pose') == target_name))
    for topic in ('/trajectory_tracker/status','/navigation/goal_status'):
        node.create_subscription(String,topic,lambda m,t=topic:record(t,m),qos)
    def plan_cb(message):
        geometry['plan'] = [[p.pose.position.x,p.pose.position.y] for p in message.poses]
        if 'first_plan' not in geometry:
            geometry['first_plan'] = geometry['plan']
    def pose_cb(message):
        p=message.pose.pose
        geometry['pose'] = [p.position.x,p.position.y,
            2*__import__('math').atan2(p.orientation.z,p.orientation.w)]
        geometry['pose_trace'].append([time.monotonic()-started,*geometry['pose']])
    node.create_subscription(NavPath,'/plan',plan_cb,5)
    node.create_subscription(PoseWithCovarianceStamped,'/localization/pose',pose_cb,5)
    def wheel_cb(message):
        p, v = message.pose.pose, message.twist.twist
        geometry['wheel_trace'].append([time.monotonic()-started,p.position.x,p.position.y,
            2*__import__('math').atan2(p.orientation.z,p.orientation.w),
            v.linear.x,v.linear.y,v.angular.z])
    node.create_subscription(Odometry,'/wheel/odometry',wheel_cb,20)
    command = ['ros2','launch','omni_autonomy_next','system.launch.py',
        'demo:=true','motors:=false','lidars:=false','wheels:=false',
        'gui:=false','rviz:=false',f'motion_mode:={args.mode}',
        # This check already captures pose/odometry/status traces. A second
        # full-rate recorder competes with the control loop on small hosts.
        'record_runs:=false',
        f'sim_command_delay:={args.delay}',f'sim_velocity_tau:={args.tau}',
        f'initial_pose_id:={args.initial_pose_id}',
        f'remembered_poses_file:={args.remembered_poses_file}']
    if not args.remembered_name:
        command.append(f'goal_id:={args.goal_id}')
    with output.with_suffix('.log').open('w') as log:
        process = subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while time.monotonic()-started < args.seconds and process.poll() is None:
                rclpy.spin_once(node,timeout_sec=.1)
                if (args.remembered_name and not remembered_sent
                        and any(r['topic'] == '/navigation/goal_status' for r in records)
                        and 'pose' in geometry and remembered_publisher.get_subscription_count()):
                    remembered_publisher.publish(String(data=args.remembered_name))
                    remembered_sent = True
                first_send = next((r['time'] for r in records if r.get('state') == 'SENDING'), None)
                if (args.next_remembered_name and not replacement_sent and first_send is not None
                        and time.monotonic()-started-first_send >= args.switch_after_sec):
                    remembered_publisher.publish(String(data=args.next_remembered_name))
                    replacement_sent = True
                if any(succeeded(r) for r in records):
                    break
        finally:
            try:
                os.killpg(process.pid,signal.SIGINT)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=15.)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGTERM)
                process.wait(timeout=10.)
            node.destroy_node()
            rclpy.shutdown()
    success = any(succeeded(r) for r in records)
    from ament_index_python.packages import get_package_share_directory
    from omni_autonomy_next.rl_residual import CadClearanceModel
    import numpy as np
    config = Path(get_package_share_directory('omni_autonomy_next'))/'config'
    model = CadClearanceModel.from_yaml(config/'field_planning.yaml',config/'competition_footprints.yaml')
    send_time = min((r['time'] for r in records if r.get('state')=='SENDING'),default=float('inf'))
    trace = [p for p in geometry['pose_trace'] if p[0]>=send_time]
    minimum = None
    if trace:
        margins = [model.body_clearance(p[1:3],p[3]) for p in trace]
        minimum = min(margins)
        for a,b,ca,cb in zip(trace[:-1],trace[1:],margins[:-1],margins[1:]):
            turn = (b[3]-a[3]+np.pi)%(2*np.pi)-np.pi
            swept_bound = .5*(np.linalg.norm(np.array(b[1:3])-a[1:3])+model.radius*abs(turn))
            minimum = min(minimum,min(ca,cb)-swept_bound)
    minimum = None if minimum is None else float(minimum)
    success = bool(success and minimum is not None and minimum > 0.)
    arrival = next((r['time'] for r in records if succeeded(r)), None)
    performance = {'arrival_sec': None if arrival is None else arrival-send_time}
    if len(trace) > 1:
        samples = np.asarray(trace)
        dt = np.diff(samples[:,0])
        delta = np.diff(samples[:,1:3], axis=0)
        valid = dt > .001
        performance.update(path_length_m=float(np.sum(np.linalg.norm(delta,axis=1))))
        axis = samples[-1,1:3]-samples[0,1:3]
        axis /= max(np.linalg.norm(axis),1.e-9)
        performance['reverse_along_goal_axis_m'] = float(np.sum(np.maximum(0.,-delta@axis)))
        from omni_autonomy_next.configured_goals import load_configured_poses
        performance['arrival_position_error_m'] = float(np.linalg.norm(samples[-1,1:3]-[target['x'],target['y']]))
        performance['arrival_yaw_error_rad'] = float(abs((samples[-1,3]-target['yaw']+np.pi)%(2*np.pi)-np.pi))
    # The demo's wheel pose integrates exactly the synthetic plant's motion.
    # Unlike localization pose differences it has no scan-correction steps or
    # repeated pose samples masquerading as stops or very high velocity.
    from omni_autonomy_next.configured_goals import load_configured_poses
    poses = load_configured_poses(config/'field_poses.yaml')
    initial = poses[str(args.initial_pose_id)]
    wheel = np.asarray([p for p in geometry['wheel_trace'] if p[0]>=send_time])
    truth_minimum = None
    if len(wheel)>1:
        c,s = np.cos(initial['yaw']),np.sin(initial['yaw'])
        truth = np.column_stack((wheel[:,0], initial['x']+c*wheel[:,1]-s*wheel[:,2],
            initial['y']+s*wheel[:,1]+c*wheel[:,2],initial['yaw']+wheel[:,3]))
        geometry['synthetic_truth_trace'] = truth.tolist()
        margins = np.array([model.body_clearance(p[1:3],p[3]) for p in truth])
        turns = (np.diff(truth[:,3])+np.pi)%(2*np.pi)-np.pi
        bounds = .5*(np.linalg.norm(np.diff(truth[:,1:3],axis=0),axis=1)+model.radius*np.abs(turns))
        truth_minimum = float(min(np.min(margins),np.min(np.minimum(margins[:-1],margins[1:])-bounds)))
        dt = np.diff(wheel[:,0])
        speed = np.linalg.norm(wheel[:,4:6],axis=1)
        performance['synthetic_speed_p95_m_s'] = float(np.quantile(speed,.95))
        performance['synthetic_translation_stationary_sec'] = float(np.sum(dt[speed[:-1]<.02]))
        performance['synthetic_stationary_sec'] = float(np.sum(dt[
            (speed[:-1]<.02) & (np.abs(wheel[:-1,6])<.025)]))
        # Count sustained interior pauses separately from initial acceleration
        # and final settling, which are intentional stops.
        moving = np.flatnonzero(speed >= .10)
        pauses = []
        pause = 0.
        if len(moving) > 1:
            for i in range(int(moving[0]), int(moving[-1])):
                if speed[i] < .07:
                    pause += float(dt[i])
                else:
                    if pause >= .25:
                        pauses.append(pause)
                    pause = 0.
            if pause >= .25:
                pauses.append(pause)
        performance['interior_pause_count'] = len(pauses)
        performance['interior_pause_sec'] = sum(pauses)
        performance['synthetic_arrival_position_error_m'] = float(np.linalg.norm(truth[-1,1:3]-[target['x'],target['y']]))
        success = bool(success and truth_minimum>0.)
    report = dict(success=success, initial_pose_id=args.initial_pose_id, goal_id=args.goal_id,
                  remembered_name=args.remembered_name, target=target,
                  next_remembered_name=args.next_remembered_name,
                  motion_mode=args.mode, delay_sec=args.delay, velocity_tau_sec=args.tau,
                  performance=performance,
                  minimum_synthetic_truth_clearance_m=truth_minimum,
                  motors_enabled=False, ros_domain_id=91, records=records, geometry=geometry,
                  minimum_localized_cad_clearance_m=minimum)
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(dict(success=success,states=sorted({r.get('state','') for r in records}),
                         reasons=sorted({r.get('reason','') for r in records if r.get('reason')}),
                         minimum_localized_cad_clearance_m=minimum,report=str(output))))
    return 0 if success else 1


if __name__ == '__main__':
    raise SystemExit(main())
