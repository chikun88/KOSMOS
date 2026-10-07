"""Connect the CRC-checked Pi MU3 commands to the existing saved-pose bridge."""
import json
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from std_msgs.msg import Bool, String

from .mu3_navigation import RemoteNavigation, MAX_SAVED_POSES


class Mu3NavigationNode(Node):
    def __init__(self):
        super().__init__('mu3_navigation')
        self.declare_parameter('saved_pose_names', [
            'A', 'BAKETU2', 'BAKETU3', '旗上側', '旗下側', '退避位置', '装填位置',
            '机保護1', '机保護2', 'バケツ後2', 'バケツ後3', 'バケツ2B', 'バケツ3B',
        ])
        names = list(self.get_parameter('saved_pose_names').value)
        # The second bank adds six points without renumbering legacy slots.
        if not 1 <= len(names) <= MAX_SAVED_POSES or len(set(names)) != len(names):
            raise ValueError(
                f'saved_pose_names must contain 1..{MAX_SAVED_POSES} unique names'
            )
        transient = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                               durability=DurabilityPolicy.TRANSIENT_LOCAL)
        # RuntimeGuard requests TRANSIENT_LOCAL; VOLATILE offers cannot match it.
        # This also remains compatible with motor_udp_bridge's volatile subscriber.
        self.arm = self.create_publisher(Bool, '/system/armed', transient)
        self.cancel = self.create_publisher(Bool, '/navigation/cancel_request', 5)
        self.goal = self.create_publisher(String, '/navigation/remembered_goal_request', 5)
        self.status = self.create_publisher(String, '/mu3/navigation_status', transient)
        self.control = RemoteNavigation(names, self._emit)
        self.create_subscription(String, '/mu3/navigation_input', self._input, 10)
        self.create_subscription(String, '/system/safety_state', self._safety, transient)
        self.create_subscription(String, '/navigation/remembered_poses', self._poses, transient)
        self.create_subscription(String, '/navigation/goal_status', self._navigation, transient)
        self.create_timer(.05, lambda: self.control.tick(time.monotonic()))
        self._emit('status', 'READY: release required before navigation')

    def _emit(self, kind, value):
        if kind == 'arm':
            self.arm.publish(Bool(data=value))
        elif kind == 'cancel':
            self.cancel.publish(Bool(data=value))
        elif kind == 'goal':
            # Name and field travel together; goal_bridge applies the mirror
            # for this one request instead of relying on topic ordering.
            name, side = value
            self.goal.publish(String(data=json.dumps(
                {'name': name, 'side': side, 'request_id': self.control.request_id},
                ensure_ascii=False)))
        else:
            self.status.publish(String(data=str(value)))
            self.get_logger().info(str(value))

    def _input(self, msg):
        try:
            data = json.loads(msg.data)
            if not isinstance(data, dict):
                return
            self.control.receive(data, time.monotonic(),
                                 self.goal.get_subscription_count() > 0
                                 and self.arm.get_subscription_count() >= 2
                                 and self.cancel.get_subscription_count() > 0)
        except (ValueError, KeyError, TypeError):
            return

    def _safety(self, msg):
        try:
            data = json.loads(msg.data)
            if isinstance(data, dict):
                self.control.safety_update(data, time.monotonic())
        except (ValueError, TypeError):
            pass

    def _poses(self, msg):
        try:
            data = json.loads(msg.data).get('poses', {})
            if isinstance(data, dict):
                self.control.poses = data
        except (ValueError, TypeError, AttributeError):
            pass

    def _navigation(self, msg):
        try:
            data = json.loads(msg.data)
            if isinstance(data, dict):
                self.control.navigation_update(data)
        except (ValueError, TypeError):
            pass


def main(args=None):
    rclpy.init(args=args)
    node = Mu3NavigationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.control.stop('SHUTDOWN')
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
