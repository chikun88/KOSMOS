"""MU3 bytes -> real C++ gateway -> real motor bridge -> Nav2 -> tracker.
Synthetic sensors/plant, isolated ROS and loopback UDP; never opens motor UART.
"""
import json
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

output = Path('/tmp/navigation-full-chain.json')
processes = []
logs = []
records = []
states = {}
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
    'gui:=false','rviz:=false','motion_mode:=simultaneous','initial_pose_id:=1'])
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
    if topic != '/motor/telemetry':
        if not records or records[-1].get('data') != data:
            records.append({'t':round(time.monotonic()-began,3),'topic':topic,'data':data})
    elif any(data.get('pi',{}).get('wheel_commands') or []):
        states['nonzero_uart'] = True

for topic in ['/mu3/navigation_status','/navigation/goal_status','/system/safety_state','/motor/telemetry']:
    node.create_subscription(String,topic,lambda m,t=topic:record(t,m),qos)
plans=[]
node.create_subscription(NavPath,'/plan',lambda m:plans.append((time.monotonic(),len(m.poses))),5)
poses=[]
node.create_subscription(PoseWithCovarianceStamped,'/localization/pose',
    lambda m:poses.append((m.pose.pose.position.x,m.pose.pose.position.y)),5)
# x of each goal actually sent to Nav2, to see the right field's mirror.
goals=[]
node.create_subscription(PoseStamped,'/navigation/active_goal',
    lambda m:goals.append(round(m.pose.position.x,3)),qos)
arm = node.create_publisher(Bool,'/system/armed',qos)

def spin(seconds):
    end=time.monotonic()+seconds
    while time.monotonic()<end:
        if any(p.poll() is not None for p in processes):
            raise RuntimeError('A process stopped: see /tmp/navigation-full-*.log')
        rclpy.spin_once(node,timeout_sec=.03)

results=[]
try:
    spin(22)
    arm.publish(Bool(data=False))
    spin(.6)
    destinations = [] if os.environ.get('NAV_TEST_SAFETY_ONLY') else [
        'A','BAKETU2','BAKETU3','旗上側','旗下側','退避位置','装填位置']
    for slot,name in enumerate(destinations,1):
        radio_token=0xc0
        spin(.6)
        start=time.monotonic()
        states['nonzero_uart']=False
        radio_token=0xc0 | (slot<<1)
        deadline=start+120
        success=False
        while time.monotonic()<deadline:
            spin(.1)
            status=states.get('/navigation/goal_status',{})
            if status.get('remembered_pose')==name and status.get('state')=='SUCCEEDED':
                success=True
                break
            remote=states.get('/mu3/navigation_status','')
            if any(reason in remote for reason in ['TIMEOUT','NOT_READY','HEALTH_LOST','FAILED']):
                raise RuntimeError(remote)
        points=max([n for t,n in plans if t>=start] or [0])
        result=dict(name=name,success=success,plan_points=points,
                    nonzero_uart=states.get('nonzero_uart'),seconds=round(time.monotonic()-start,2))
        result['field_side']=states.get('/navigation/goal_status',{}).get('field_side')
        results.append(result)
        print(json.dumps(result,ensure_ascii=False),flush=True)
        assert success and points>1 and result['nonzero_uart'], result
        assert result['field_side']=='left', result
        spin(.3)
    # The right field is the same seven points reflected about the divider.
    # The robot starts on the left, so this checks the command path and the
    # mirrored target rather than arrival: driving across the divider is not
    # something the planner should ever be asked to do.
    for slot,name in enumerate(destinations,1):
        radio_token=0xc0
        spin(.6)
        goals.clear()
        radio_token=0xc0 | ((slot+8)<<1)
        deadline=time.monotonic()+30
        mirrored=None
        while time.monotonic()<deadline:
            spin(.1)
            status=states.get('/navigation/goal_status',{})
            if status.get('remembered_pose')==name and goals:
                mirrored=goals[-1]
                break
        radio_token=0xc0
        spin(.6)
        result=dict(name=name,slot=slot+8,
                    field_side=status.get('field_side'),goal_x=mirrored)
        results.append(result)
        print(json.dumps(result,ensure_ascii=False),flush=True)
        assert result['field_side']=='right', result
        assert mirrored is not None and mirrored > 0.30, result
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
