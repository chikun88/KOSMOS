"""MU3 bytes -> real C++ gateway -> real motor bridge -> Nav2 -> tracker.
Synthetic sensors/plant, isolated ROS and loopback UDP; never opens motor UART.
"""
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import threading
import time

os.environ['ROS_DOMAIN_ID'] = '94'
os.environ['ROS_LOCALHOST_ONLY'] = '1'
import rclpy
from rclpy.qos import QoSProfile, DurabilityPolicy
from std_msgs.msg import String, Bool
from nav_msgs.msg import Path as NavPath
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from lifecycle_msgs.srv import GetState

def pose_matches(pose, target, xy_tolerance=.02, yaw_tolerance=.02):
    """Use the configured Nav2 tolerance, including wrapped heading error."""
    return (pose is not None and len(pose) == 3 and
            all(math.isfinite(value) for value in pose) and
            math.hypot(pose[0]-target['x'], pose[1]-target['y']) <= xy_tolerance and
            abs((pose[2]-target['yaw']+math.pi) % (2*math.pi)-math.pi) <= yaw_tolerance)

def current_goal_succeeded(status, name, previous_request_id):
    """A retained success from an earlier request cannot complete this goal."""
    return (status.get('remembered_pose') == name and status.get('field_side') == 'left'
            and status.get('state') == 'SUCCEEDED' and bool(status.get('request_id'))
            and status['request_id'] != previous_request_id)

def arrival_evidence(status, name, previous_request_id, start_pose, target,
                     finish_pose, nonzero_uart, plan_points, wheel_commands):
    assert current_goal_succeeded(status, name, previous_request_id), status
    assert pose_matches(finish_pose, target), ('unsettled pose', finish_pose, target)
    assert len(wheel_commands) == 4 and not any(wheel_commands), ('unsettled wheels', wheel_commands)
    # A deliberately drives to its approach gate and reverses into the bay,
    # even when its final docking pose equals the current pose.
    already_settled = name != 'A' and pose_matches(start_pose, target)
    if not already_settled:
        assert nonzero_uart and plan_points > 1, ('missing displacement', name, plan_points, nonzero_uart)
    elif pose_matches(start_pose, target, 1.e-6, 1.e-6):
        assert not nonzero_uart, ('unexpected exact-pose movement', name)
    return {'already_settled': already_settled, 'settled_zero_uart': True}

def expected_right_active_goal(name, left_target):
    """Independent field reflection; A's documented docking gate is 0.25 m ahead."""
    target = dict(left_target)
    target['x'] = -left_target['x']
    target['yaw'] = (math.pi-left_target['yaw']+math.pi) % (2*math.pi)-math.pi
    if name == 'A':
        target['x'] += .25*math.cos(target['yaw'])
        target['y'] += .25*math.sin(target['yaw'])
    return target

def mirror_goal_is_current(status, name, previous_request_id, started,
                           status_received_at, goal):
    """Join the active PoseStamped to this request using its exact ROS stamp."""
    return (status.get('remembered_pose') == name and status.get('field_side') == 'right'
            and bool(status.get('request_id')) and status['request_id'] != previous_request_id
            and status_received_at >= started and goal['received_at'] >= started
            and bool(status.get('goal_stamp')) and status['goal_stamp'] == goal['stamp']
            and any(goal['stamp']))

def mirror_evidence(status, name, previous_request_id, started, status_received_at,
                    goal, left_target):
    assert mirror_goal_is_current(status, name, previous_request_id, started,
                                  status_received_at, goal), ('unrelated active goal', status, goal)
    expected = expected_right_active_goal(name, left_target)
    assert goal['frame_id'] == expected['frame_id'], ('mirror frame', goal, expected)
    assert goal['pose'][0] > .30, ('mirror inside divider', goal)
    assert pose_matches(goal['pose'], expected, 1.e-6, 1.e-6), ('incorrect mirror', goal, expected)
    return {'field_side': 'right', 'request_id': status['request_id'],
            'goal_pose': goal['pose'], 'goal_stamp': goal['stamp'],
            'expected_pose': [expected['x'], expected['y'], expected['yaw']]}

output = Path('/tmp/navigation-full-chain.json')
processes = []
logs = []
records = []
states = {}
received_at = {}
radio_token = 0xc0
radio_enabled = True
finished = False
began = time.monotonic()

def launch(label, cmd):
    stream = open('/tmp/navigation-full-' + label + '.log', 'w')
    logs.append(stream)
    p = subprocess.Popen(cmd, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    processes.append(p)
    return p

def radio_loop():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        while not finished:
            if radio_enabled:
                sock.sendto(bytes([128,128,128,128,0,0,radio_token]), ('127.0.0.1',39401))
            time.sleep(.02)

launch('gateway', ['/tmp/navigation_loopback_gateway'])
launch('stack', ['ros2','launch','omni_autonomy_next','system.launch.py',
    'demo:=true','motors:=false','lidars:=false','wheels:=false',
    # This observer already records the chain. A second full-rate recorder
    # competes with the gateway/bridge and creates artificial watchdog faults.
    'gui:=false','rviz:=false','record_runs:=false',
    'motion_mode:=simultaneous','initial_pose_id:=1'])
launch('motor', ['ros2','run','omni_autonomy_next','motor_udp_bridge','--ros-args',
    '-p','local_ip:=127.0.0.1','-p','local_port:=39402',
    '-p','remote_ip:=127.0.0.1','-p','remote_port:=39400',
    '-p','payload_format:=v4_uart','-p','require_healthy_telemetry:=true',
    '-p','cmd_vel_topic:=/cmd_vel_safe','-p','enable_topic:=/system/armed',
    '-p','estop_topic:=/system/emergency_stop','-p','send_rate_hz:=100.0'])
launch('mu3', ['ros2','run','omni_autonomy_next','mu3_navigation'])
thread = threading.Thread(target=radio_loop, daemon=True)
thread.start()
rclpy.init()
node = rclpy.create_node('navigation_full_chain_observer')
qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)

def record(topic, message):
    try: data=json.loads(message.data)
    except ValueError: data=message.data
    states[topic] = data
    received_at[topic] = time.monotonic()
    if topic != '/motor/telemetry':
        if not records or records[-1].get('data') != data:
            records.append({'t':round(time.monotonic()-began,3),'topic':topic,'data':data})
    elif any(data.get('pi',{}).get('wheel_commands') or []):
        states['nonzero_uart'] = True

for topic in ['/mu3/navigation_status','/navigation/goal_status','/system/safety_state','/motor/telemetry',
              '/navigation/remembered_poses']:
    node.create_subscription(String,topic,lambda m,t=topic:record(t,m),qos)
plans=[]
node.create_subscription(NavPath,'/plan',lambda m:plans.append((time.monotonic(),len(m.poses))),5)
poses=[]
def pose_record(message):
    pose = message.pose.pose
    q = pose.orientation
    poses.append((pose.position.x, pose.position.y,
                  math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))))
    received_at['/localization/pose'] = time.monotonic()
node.create_subscription(PoseWithCovarianceStamped,'/localization/pose',pose_record,5)
# Preserve the full active pose and its request-correlated ROS timestamp.
goals=[]
def active_goal_record(message):
    pose, q = message.pose, message.pose.orientation
    goal = {'pose': [pose.position.x, pose.position.y,
                    math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))],
            'stamp': [message.header.stamp.sec, message.header.stamp.nanosec],
            'frame_id': message.header.frame_id, 'received_at': time.monotonic()}
    goals.append(goal)
    records.append({'t': round(goal['received_at']-began, 3),
                    'topic': '/navigation/active_goal', 'data': goal})
node.create_subscription(PoseStamped,'/navigation/active_goal',active_goal_record,qos)
arm = node.create_publisher(Bool,'/system/armed',qos)
nav2_clients = {name: node.create_client(GetState, '/' + name + '/get_state')
                for name in ('bt_navigator', 'collision_monitor', 'controller_server',
                             'planner_server', 'velocity_smoother')}

def spin(seconds):
    end=time.monotonic()+seconds
    while time.monotonic()<end:
        if any(p.poll() is not None for p in processes):
            raise RuntimeError('A process stopped: see /tmp/navigation-full-*.log')
        rclpy.spin_once(node,timeout_sec=.03)

def wait_ready(timeout=40):
    """Wait for measured health and lifecycle activation, with a hard deadline."""
    deadline = time.monotonic() + timeout
    futures = {}
    last_poll = 0.0
    ready_since = None
    while time.monotonic() < deadline:
        now = time.monotonic()
        if now - last_poll >= .5:
            for name, client in nav2_clients.items():
                if client.service_is_ready() and (name not in futures or futures[name].done()):
                    futures[name] = client.call_async(GetState.Request())
            last_poll = now
        nav2_active = len(futures) == len(nav2_clients) and all(
            future.done() and future.exception() is None and
            future.result() is not None and future.result().current_state.id == 3
            for future in futures.values())
        pi = states.get('/motor/telemetry', {}).get('pi', {})
        health = states.get('/system/safety_state', {})
        ready = nav2_active and health.get('health_reason') == 'ACTIVE' and all(
            pi.get(flag) for flag in ('link_alive', 'uart_open', 'mu3_alive')) and not pi.get('estop_active') and all(
            now - received_at.get(topic, 0) < .25
            for topic in ('/motor/telemetry', '/system/safety_state'))
        if ready:
            ready_since = now if ready_since is None else ready_since
            if now - ready_since >= .3:
                return
        else:
            ready_since = None
        spin(.05)
    raise AssertionError(('chain readiness deadline', states))

results=[]
try:
    wait_ready()
    arm.publish(Bool(data=False))
    spin(.6)
    destinations = [] if os.environ.get('NAV_TEST_SAFETY_ONLY') else [
        'A','BAKETU2','BAKETU3','旗上側','旗下側','退避位置','装填位置']
    left_targets = {}
    for slot,name in enumerate(destinations,1):
        radio_token=0xc0
        spin(.6)
        wait_ready(timeout=20)
        assert poses and time.monotonic()-received_at.get('/localization/pose', 0) < .25
        saved = states.get('/navigation/remembered_poses', {})
        assert saved.get('field_side') == 'left', saved
        target = saved['poses'][name]
        left_targets[name] = dict(target)
        start_pose = poses[-1]
        previous_request_id = states.get('/navigation/goal_status', {}).get('request_id')
        start=time.monotonic()
        states['nonzero_uart']=False
        radio_token=0xc0 | (slot<<1)
        deadline=start+120
        success=False
        while time.monotonic()<deadline:
            spin(.1)
            status=states.get('/navigation/goal_status',{})
            if (received_at.get('/navigation/goal_status', 0) >= start
                    and current_goal_succeeded(status, name, previous_request_id)):
                success=True
                break
            remote=states.get('/mu3/navigation_status','')
            if any(reason in remote for reason in ['TIMEOUT','NOT_READY','HEALTH_LOST','FAILED']):
                raise RuntimeError(remote)
        points=max([n for t,n in plans if t>=start] or [0])
        result=dict(name=name,success=success,plan_points=points,
                    nonzero_uart=states.get('nonzero_uart'),seconds=round(time.monotonic()-start,2))
        result['field_side']=states.get('/navigation/goal_status',{}).get('field_side')
        assert success, result
        spin(.3)
        assert time.monotonic()-received_at.get('/motor/telemetry', 0) < .25
        assert time.monotonic()-received_at.get('/localization/pose', 0) < .25
        result.update(arrival_evidence(status, name, previous_request_id, start_pose,
            target, poses[-1], result['nonzero_uart'], points,
            states['/motor/telemetry']['pi']['wheel_commands']))
        results.append(result)
        print(json.dumps(result,ensure_ascii=False),flush=True)
    # The right field is the same seven points reflected about the divider.
    # The robot starts on the left, so this checks the command path and the
    # mirrored target rather than arrival: driving across the divider is not
    # something the planner should ever be asked to do.
    for slot,name in enumerate(destinations,1):
        radio_token=0xc0
        spin(.6)
        goals.clear()
        previous_request_id = states.get('/navigation/goal_status', {}).get('request_id')
        started = time.monotonic()
        radio_token=0xc0 | ((slot+8)<<1)
        deadline=started+30
        mirrored=None
        while time.monotonic()<deadline:
            spin(.1)
            status=states.get('/navigation/goal_status',{})
            mirrored = next((goal for goal in reversed(goals) if mirror_goal_is_current(
                status, name, previous_request_id, started,
                received_at.get('/navigation/goal_status', 0), goal)), None)
            if mirrored is not None:
                break
        radio_token=0xc0
        assert mirrored is not None, ('no current mirrored active goal', name, status, goals)
        result=dict(name=name,slot=slot+8, **mirror_evidence(status, name,
            previous_request_id, started, received_at.get('/navigation/goal_status', 0),
            mirrored, left_targets[name]))
        spin(.6)
        results.append(result)
        print(json.dumps(result,ensure_ascii=False),flush=True)
    # Slot 8 is point 0 on the right and must be refused, not rounded to a point.
    radio_token=0xc0
    spin(.6)
    radio_token=0xc0 | (8<<1)
    deadline=time.monotonic()+5
    while time.monotonic()<deadline and 'UNKNOWN_SAVED_POSE' not in str(
            states.get('/mu3/navigation_status','')):
        spin(.1)
    assert 'UNKNOWN_SAVED_POSE' in str(states.get('/mu3/navigation_status','')), states
    results.append({'scenario':'slot_8_rejected','refused':True})
    print(json.dumps(results[-1]),flush=True)
    radio_token=0xc0
    spin(.6)
    # Stop during motion, then restart and lose radio during motion.
    for fault in ['stop_button', 'radio_loss']:
        radio_enabled=True
        radio_token=0xc0
        spin(.7)
        wait_ready(timeout=20)
        states['nonzero_uart']=False
        radio_token=0xc4  # BAKETU2, away from the initial/final loading bay.
        deadline=time.monotonic()+15
        while not states.get('nonzero_uart') and time.monotonic()<deadline:
            spin(.1)
        assert states.get('nonzero_uart'), ('did not start',fault,states)
        if fault == 'stop_button': radio_token=0xc0
        else: radio_enabled=False
        spin(.8)
        assert not states['/motor/telemetry']['pi']['auto_engaged'], fault
        assert not any(states['/motor/telemetry']['pi']['wheel_commands']), fault
        spin(2.5)
        assert not states['/motor/telemetry']['pi']['auto_engaged'], ('unexpected rearm',fault)
        results.append({'scenario':fault,'stopped':True,'no_idle_rearm':True})
        print(json.dumps(results[-1]),flush=True)
    # Link recovery with the old held command must not restart navigation.
    radio_enabled=True
    spin(.8)
    assert not states['/motor/telemetry']['pi']['auto_engaged']
    results.append({'scenario':'radio_recovery_held_button','no_restart':True})
    radio_token=0xc0
    spin(.7)
    assert not states['/motor/telemetry']['pi']['auto_engaged']
    radio_enabled=False
    spin(.5)
    assert not states['/motor/telemetry']['pi']['auto_engaged']
    print('FULL_CHAIN_PASS',flush=True)
finally:
    finished=True
    arm.publish(Bool(data=False))
    for p in processes:
        if p.poll() is None: os.killpg(p.pid,signal.SIGINT)
    for p in processes:
        try: p.wait(timeout=12)
        except subprocess.TimeoutExpired: os.killpg(p.pid,signal.SIGKILL)
    for log in logs: log.close()
    output.write_text(json.dumps(dict(results=results,records=records,plans=plans,poses=poses,
        physical_motors=False,ros_domain=94),ensure_ascii=False))
    node.destroy_node()
    rclpy.shutdown()
