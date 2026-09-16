"""Regression: real RuntimeGuard QoS, fake motor peer, isolated ROS domain."""
import json
import os
import time
import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import QoSProfile, DurabilityPolicy
from std_msgs.msg import Bool, String
from geometry_msgs.msg import Twist
from omni_autonomy_next.mu3_navigation_node import Mu3NavigationNode
from omni_autonomy_next.runtime_guard_node import RuntimeGuardNode

if os.environ.get('ROS_DOMAIN_ID') != '93' or os.environ.get('ROS_LOCALHOST_ONLY') != '1':
    raise SystemExit('Run with ROS_DOMAIN_ID=93 ROS_LOCALHOST_ONLY=1; hardware domain is forbidden')
rclpy.init()
remote = Mu3NavigationNode()
guard = RuntimeGuardNode()
peer = Node('fake_motor_peer_no_hardware')
executor = SingleThreadedExecutor()
for node in (remote, guard, peer): executor.add_node(node)
qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
poses_pub = peer.create_publisher(String, '/navigation/remembered_poses', qos)
input_pub = peer.create_publisher(String, '/mu3/navigation_input', 10)
status_pub = peer.create_publisher(String, '/navigation/goal_status', qos)
tracking = peer.create_publisher(Bool, '/localization/tracking_ok', 5)
link = peer.create_publisher(Bool, '/motor/link_ok', 5)
engaged = peer.create_publisher(Bool, '/motor/auto_engaged', 5)
rl = peer.create_publisher(Bool, '/rl/healthy', qos)
command = peer.create_publisher(Twist, '/cmd_vel_collision_safe', 5)
state = {'arm': False, 'goals': [], 'cancels': 0}
peer.create_subscription(Bool, '/system/armed', lambda m: state.update(arm=m.data), 5)
peer.create_subscription(Bool, '/navigation/cancel_request',
                         lambda m: state.update(cancels=state['cancels'] + 1), 5)
peer.create_subscription(String, '/navigation/remembered_goal_request',
                         lambda m: state['goals'].append(m.data), 5)
seq = 0

def run(slot, generation, duration, alive=True):
    global seq
    end = time.monotonic() + duration
    while time.monotonic() < end:
        seq += 1
        tracking.publish(Bool(data=True))
        link.publish(Bool(data=True))
        engaged.publish(Bool(data=state['arm']))
        rl.publish(Bool(data=True))
        command.publish(Twist())  # Zero only, no motor device or socket exists.
        poses_pub.publish(String(data=json.dumps({'poses': {'A': {}, 'BAKETU2': {}}})))
        input_pub.publish(String(data=json.dumps(dict(
            slot=slot, sequence=generation, pi_seq=seq, mu3_alive=alive,
            auto_engaged=state['arm'], link_alive=True, uart_open=True,
            estop_active=False, fault_latched=False))))
        for _ in range(30): executor.spin_once(timeout_sec=.001)

try:
    run(0, 0, 1.0)
    run(1, 1, 1.0)
    assert guard.armed, 'ARM did not reach the actual RuntimeGuard'
    assert state['goals'] == ['A'], state
    run(1, 1, .3)
    assert state['goals'] == ['A'], state
    status_pub.publish(String(data=json.dumps({'remembered_pose': 'A', 'state': 'SUCCEEDED'})))
    run(1, 1, .2)
    assert not guard.armed and not state['arm']
    run(0, 2, .3)
    run(2, 3, .8)
    assert state['goals'] == ['A', 'BAKETU2'], state
    run(2, 3, .3, alive=False)
    assert not guard.armed and not state['arm']
    run(2, 3, .5)
    assert state['goals'] == ['A', 'BAKETU2']
    print('PASS: actual RuntimeGuard ARM/DISARM, saved goal, completion, radio loss, no replay')
finally:
    remote.control.stop('TEST_FINISHED')
    executor.shutdown()
    for node in (remote, guard, peer): node.destroy_node()
    rclpy.shutdown()
