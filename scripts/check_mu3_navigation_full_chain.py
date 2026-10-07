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
import sys
import time

os.environ['ROS_DOMAIN_ID'] = '94'
os.environ['ROS_LOCALHOST_ONLY'] = '1'
import rclpy
from rclpy.qos import QoSProfile, DurabilityPolicy, qos_profile_sensor_data
from std_msgs.msg import String, Bool
from nav_msgs.msg import Path as NavPath, Odometry
from sensor_msgs.msg import LaserScan
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

def navigation_failure(status):
    """Every terminal refusal or stop must fail the outstanding arrival promptly."""
    reason = str(status).split(':', 1)[0]
    return str(status) if reason in {
        'DISARMED', 'DISARM_TIMEOUT', 'ARM_TIMEOUT', 'GOAL_ACK_TIMEOUT',
        'TELEMETRY_LOST', 'MU3_LOST', 'HEALTH_LOST', 'NOT_READY', 'FAILED',
        'REJECTED', 'ABORTED', 'CANCELED', 'UNKNOWN_REMEMBERED_POSE',
        'INVALID_REMEMBERED_POSE', 'UNKNOWN_SAVED_POSE', 'RELEASE_REQUIRED',
        'RELEASED', 'SHUTDOWN',
    } else None

def post_fault_pi_evidence(telemetry, receipt, failed_at, now, baseline_count,
                           failed_realtime_ns, now_realtime_ns):
    """Require an actual post-fault kernel arrival, never a retained zero."""
    bridge, pi = telemetry.get('bridge', {}), telemetry.get('pi', {})
    count, age_ms = bridge.get('telemetry_count'), bridge.get('telemetry_age_ms')
    arrival_ns = telemetry.get('kernel_arrival_realtime_ns')
    count_is_new = (type(count) is int and type(baseline_count) is int
                    and count > baseline_count >= 0)
    age_is_valid = (type(age_ms) in (int, float) and math.isfinite(age_ms)
                    and 0 <= age_ms <= 250)
    arrival_is_fresh = (type(arrival_ns) is int and
                        failed_realtime_ns < arrival_ns <= now_realtime_ns and
                        now_realtime_ns-arrival_ns <= 250_000_000)
    fresh = (receipt > failed_at and 0 <= now-receipt <= .25 and
             count_is_new and age_is_valid and arrival_is_fresh)
    wheels = pi.get('wheel_commands')
    valid_wheels = (isinstance(wheels, list) and len(wheels) == 4 and all(
        type(value) in (int, float) and math.isfinite(value) for value in wheels))
    zero_wheels = valid_wheels and not any(wheels)
    return {'fresh': fresh, 'stopped': fresh and pi.get('auto_engaged') is False and zero_wheels,
            'moving': fresh and pi.get('auto_engaged') is True and valid_wheels and any(wheels),
            'telemetry_count': count, 'telemetry_age_ms': age_ms,
            'kernel_arrival_realtime_ns': arrival_ns,
            'receipt_after_failure_sec': receipt-failed_at,
            'auto_engaged': pi.get('auto_engaged'), 'wheel_commands': wheels}

def observe_terminal_failure(failure, scenario, timeout=.8):
    """Keep the original failure; observe without enabling or changing MU3 input."""
    failed_at = time.monotonic()
    failed_realtime_ns = time.time_ns()
    baseline = states.get('/motor/telemetry', {}).get('bridge', {}).get('telemetry_count')
    evidence = {'failure': failure, 'scenario': scenario,
                'started_t_sec': round(failed_at-began, 6),
                'failure_realtime_ns': failed_realtime_ns,
                'window_limit_sec': timeout, 'baseline_telemetry_count': baseline,
                'fresh_samples_observed': 0, 'fresh_gateway_stop_observed': False,
                'last_sample': None}
    try:
        while time.monotonic() < failed_at+timeout:
            spin(min(.03, max(0., failed_at+timeout-time.monotonic())))
            now = time.monotonic()
            if now > failed_at+timeout:
                evidence['deadline_overrun_sec'] = round(now-failed_at-timeout, 6)
                break
            sample = post_fault_pi_evidence(states.get('/motor/telemetry', {}),
                received_at.get('/motor/telemetry', 0), failed_at,
                now, baseline, failed_realtime_ns, time.time_ns())
            evidence['last_sample'] = sample
            if type(baseline) is not int and type(sample['telemetry_count']) is int:
                baseline = sample['telemetry_count']
            if sample['fresh']:
                # Count distinct gateway updates rather than repeated loop reads.
                baseline = sample['telemetry_count']
                evidence['fresh_samples_observed'] += 1
                if sample['stopped']:
                    evidence['fresh_gateway_stop_observed'] = True
                    break
    except (Exception, KeyboardInterrupt) as error:
        evidence['observation_error'] = f'{type(error).__name__}: {error}'
    evidence['elapsed_sec'] = round(time.monotonic()-failed_at, 6)
    print(json.dumps({'terminal_failure_evidence': evidence}, ensure_ascii=False), flush=True)
    return evidence

def telemetry_checkpoint():
    return {'failed_at': time.monotonic(), 'failed_realtime_ns': time.time_ns(),
            'baseline_count': states.get('/motor/telemetry', {}).get('bridge', {}).get('telemetry_count')}

def pi_evidence_after_checkpoint(checkpoint):
    return post_fault_pi_evidence(states.get('/motor/telemetry', {}),
        received_at.get('/motor/telemetry', 0), now=time.monotonic(),
        now_realtime_ns=time.time_ns(), **checkpoint)

def assert_fresh_gateway_stop(checkpoint, scenario):
    sample = pi_evidence_after_checkpoint(checkpoint)
    assert sample['stopped'], ('fresh gateway stop unproven', scenario, sample)
    return sample

def collect_no_restart_evidence(window, telemetry, receipt, realtime_ns):
    checkpoint = dict(window['checkpoint'], baseline_count=window['last_count'])
    sample = post_fault_pi_evidence(telemetry, receipt, now=receipt,
        now_realtime_ns=realtime_ns, **checkpoint)
    if sample['fresh']:
        window['last_count'] = sample['telemetry_count']
        window['fresh_samples_observed'] += 1
        if not sample['stopped'] and window['first_violation'] is None:
            window['first_violation'] = sample

def assert_no_restart_window(seconds, scenario, checkpoint=None):
    """Latch every fresh telemetry violation, including a transient rearm."""
    global stop_window
    checkpoint = telemetry_checkpoint() if checkpoint is None else checkpoint
    window = {'checkpoint': checkpoint, 'last_count': checkpoint['baseline_count'],
              'fresh_samples_observed': 0, 'first_violation': None}
    assert stop_window is None
    stop_window = window
    try:
        spin(seconds)
    finally:
        stop_window = None
    endpoint = assert_fresh_gateway_stop(checkpoint, scenario)
    assert window['first_violation'] is None, ('transient gateway restart', scenario, window)
    return {'fresh_samples_observed': window['fresh_samples_observed'],
            'first_violation': window['first_violation'], 'endpoint': endpoint}

def arrival_slot_selection(value):
    if value is None:
        return tuple(range(1, 8))
    slots = tuple(int(token) for token in value.split(','))
    if not slots or len(set(slots)) != len(slots) or any(not 1 <= slot <= 7 for slot in slots):
        raise ValueError('NAV_TEST_ARRIVAL_SLOTS must be unique slots 1..7')
    return slots

def cleanup_processes(processes, logs, disarm):
    # rclpy's SIGINT handler can invalidate the context before finally runs.
    # A failed best-effort disarm must not leave test process groups alive.
    try:
        disarm()
    except Exception as error:
        print(f'Cleanup disarm unavailable: {error}', flush=True)
    for process in processes:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
    for process in processes:
        try:
            process.wait(timeout=12)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=12)
    for log in logs:
        log.close()

output = Path('/tmp/navigation-full-chain.json')
processes = []
logs = []
records = []
states = {}
received_at = {}
radio_token = 0xc0
radio_enabled = True
began = time.monotonic()
arrival_slots = arrival_slot_selection(os.environ.get('NAV_TEST_ARRIVAL_SLOTS'))
arrival_only = os.environ.get('NAV_TEST_ARRIVAL_ONLY') == '1'
timing_diagnostics = os.environ.get('NAV_TEST_TIMING_DIAGNOSTICS') == '1'
timing_records = []
stop_window = None
radio_timing_path = Path('/tmp/navigation-radio-timing.json')
radio_timing_path.unlink(missing_ok=True)
radio_control, radio_worker_control = socket.socketpair(socket.AF_UNIX, socket.SOCK_DGRAM)
radio_control.setblocking(False)

def launch(label, cmd, **kwargs):
    stream = open('/tmp/navigation-full-' + label + '.log', 'w')
    logs.append(stream)
    p = subprocess.Popen(cmd, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True, **kwargs)
    processes.append(p)
    return p

def set_radio(token=None, enabled=None):
    """One atomic state keeps a held token unchanged through radio recovery."""
    global radio_token, radio_enabled
    if token is not None:
        radio_token = token
    if enabled is not None:
        radio_enabled = enabled
    radio_control.send(bytes([int(radio_enabled), radio_token]))

launch('radio', [sys.executable, str(Path(__file__).with_name('mu3_radio_fixture.py')),
                '--control-fd', str(radio_worker_control.fileno()),
                '--output', str(radio_timing_path), '--started', str(began)],
       pass_fds=(radio_worker_control.fileno(),))
radio_worker_control.close()
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
    else:
        if any(data.get('pi',{}).get('wheel_commands') or []):
            states['nonzero_uart'] = True
        if stop_window is not None:
            collect_no_restart_evidence(stop_window, data, received_at[topic], time.time_ns())

for topic in ['/mu3/navigation_status','/navigation/goal_status','/system/safety_state','/motor/telemetry',
              '/motor/network_status',
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
if timing_diagnostics:
    def header_timing(topic, message):
        stamp = message.header.stamp
        receipt_ns = node.get_clock().now().nanoseconds
        acquisition_ns = stamp.sec*1_000_000_000+stamp.nanosec
        timing_records.append({'t': round(time.monotonic()-began, 6), 'topic': topic,
                               'acquisition_ns': acquisition_ns, 'receipt_ns': receipt_ns,
                               'age_sec': (receipt_ns-acquisition_ns)*1.e-9})
    for topic in ('/scan_front', '/scan_rear', '/scan_front_filtered', '/scan_rear_filtered'):
        node.create_subscription(LaserScan, topic,
            lambda message, name=topic: header_timing(name, message), qos_profile_sensor_data)
    node.create_subscription(Odometry, '/wheel/odometry',
        lambda message: header_timing('/wheel/odometry', message), qos_profile_sensor_data)
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
        rclpy.spin_once(node,timeout_sec=min(.03, max(0., end-time.monotonic())))

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
terminal_failure_evidence = None
try:
    wait_ready()
    arm.publish(Bool(data=False))
    spin(.6)
    names = ['A','BAKETU2','BAKETU3','旗上側','旗下側','退避位置','装填位置']
    destinations = [] if os.environ.get('NAV_TEST_SAFETY_ONLY') else [
        (slot, names[slot-1]) for slot in arrival_slots]
    left_targets = {}
    for slot,name in destinations:
        set_radio(token=0xc0)
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
        set_radio(token=0xc0 | (slot<<1))
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
            if (received_at.get('/mu3/navigation_status', 0) >= start
                    and navigation_failure(remote)):
                terminal_failure_evidence = observe_terminal_failure(remote, f'arrival:{name}:left')
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
    if arrival_only:
        print('PARTIAL_ARRIVAL_PASS', flush=True)
        raise SystemExit(0)
    # The right field is the same seven points reflected about the divider.
    # The robot starts on the left, so this checks the command path and the
    # mirrored target rather than arrival: driving across the divider is not
    # something the planner should ever be asked to do.
    for slot,name in destinations:
        set_radio(token=0xc0)
        spin(.6)
        goals.clear()
        previous_request_id = states.get('/navigation/goal_status', {}).get('request_id')
        started = time.monotonic()
        set_radio(token=0xc0 | ((slot+8)<<1))
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
            remote = states.get('/mu3/navigation_status', '')
            if (received_at.get('/mu3/navigation_status', 0) >= started
                    and navigation_failure(remote)):
                terminal_failure_evidence = observe_terminal_failure(remote, f'mirror:{name}:right')
                raise RuntimeError(('right mirror unavailable', name, remote, status))
        set_radio(token=0xc0)
        assert mirrored is not None, ('no current mirrored active goal', name, status, goals)
        result=dict(name=name,slot=slot+8, **mirror_evidence(status, name,
            previous_request_id, started, received_at.get('/navigation/goal_status', 0),
            mirrored, left_targets[name]))
        spin(.6)
        results.append(result)
        print(json.dumps(result,ensure_ascii=False),flush=True)
    # Slot 8 is point 0 on the right and must be refused, not rounded to a point.
    set_radio(token=0xc0)
    spin(.6)
    set_radio(token=0xc0 | (8<<1))
    deadline=time.monotonic()+5
    while time.monotonic()<deadline and 'UNKNOWN_SAVED_POSE' not in str(
            states.get('/mu3/navigation_status','')):
        spin(.1)
    assert 'UNKNOWN_SAVED_POSE' in str(states.get('/mu3/navigation_status','')), states
    results.append({'scenario':'slot_8_rejected','refused':True})
    print(json.dumps(results[-1]),flush=True)
    set_radio(token=0xc0)
    spin(.6)
    # Stop during motion, then restart and lose radio during motion.
    for fault in ['stop_button', 'radio_loss']:
        set_radio(token=0xc0, enabled=True)
        spin(.7)
        wait_ready(timeout=20)
        states['nonzero_uart']=False
        motion_checkpoint = telemetry_checkpoint()
        set_radio(token=0xc4)  # BAKETU2, away from the initial/final loading bay.
        deadline=time.monotonic()+15
        motion_evidence = pi_evidence_after_checkpoint(motion_checkpoint)
        while not motion_evidence['moving'] and time.monotonic()<deadline:
            spin(.1)
            motion_evidence = pi_evidence_after_checkpoint(motion_checkpoint)
        assert motion_evidence['moving'], ('no fresh motion at fault injection',fault,motion_evidence)
        fault_checkpoint = telemetry_checkpoint()
        if fault == 'stop_button': set_radio(token=0xc0)
        else: set_radio(enabled=False)
        spin(.8)
        stop_evidence = assert_fresh_gateway_stop(fault_checkpoint, fault)
        idle_evidence = assert_no_restart_window(2.5, ('idle', fault))
        results.append({'scenario':fault,'stopped':True,'no_idle_rearm':True,
                        'motion_at_injection': motion_evidence,
                        'gateway_stop_evidence': stop_evidence, 'idle_stop_evidence': idle_evidence})
        print(json.dumps(results[-1]),flush=True)
    # Link recovery with the old held command must not restart navigation.
    recovery_checkpoint = telemetry_checkpoint()
    set_radio(enabled=True)
    recovery_evidence = assert_no_restart_window(.8, 'radio_recovery_held_button', recovery_checkpoint)
    results.append({'scenario':'radio_recovery_held_button','no_restart':True,
                    'gateway_stop_evidence': recovery_evidence})
    release_checkpoint = telemetry_checkpoint()
    set_radio(token=0xc0)
    assert_no_restart_window(.7, 'radio_recovery_release', release_checkpoint)
    final_checkpoint = telemetry_checkpoint()
    set_radio(enabled=False)
    assert_no_restart_window(.5, 'radio_final_loss', final_checkpoint)
    print('FULL_CHAIN_PASS',flush=True)
finally:
    cleanup_processes(processes, logs, lambda: arm.publish(Bool(data=False)))
    radio_control.close()
    radio_timing = (json.loads(radio_timing_path.read_text()) if radio_timing_path.exists()
                    else {'error': 'radio subprocess did not save timing'})
    output.write_text(json.dumps(dict(results=results,records=records,plans=plans,poses=poses,
        final_states=states, timing_records=timing_records, arrival_slots=arrival_slots,
        arrival_only=arrival_only, timing_diagnostics=timing_diagnostics,
        radio_timing=radio_timing,
        terminal_failure_evidence=terminal_failure_evidence,
        physical_motors=False,ros_domain=94),ensure_ascii=False))
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
