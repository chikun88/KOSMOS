"""ROS 2 bridge which applies a promoted RL residual before collision gating."""

import json
import math
import time

from ament_index_python.packages import get_package_share_directory
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
from nav_msgs.msg import Path
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, Float32, String

from .config import load_collision_monitor_horizon
from .rl_policy import CompactRLPolicy
from .rl_residual import (
    CadClearanceModel,
    apply_clearance_residual,
    limit_yaw_rate,
    make_observation,
    select_path_target,
)


def quaternion_yaw(orientation):
    return math.atan2(
        2.0 * (orientation.w * orientation.z + orientation.x * orientation.y),
        1.0 - 2.0 * (orientation.y * orientation.y + orientation.z * orientation.z),
    )


class RLPolicyNode(Node):
    """Fail-closed runtime adapter for the offline-learned greedy Q policy."""

    def __init__(self):
        super().__init__('rl_policy')
        share = get_package_share_directory('omni_autonomy_next')
        defaults = {
            'policy_file': share + '/config/rl_policy.yaml',
            'field_config_file': share + '/config/field_planning.yaml',
            'footprint_config_file': share + '/config/competition_footprints.yaml',
            'footprint_profile': 'NORMAL',
            'input_topic': '/cmd_vel_nav_smoothed',
            'output_topic': '/cmd_vel_rl',
            'plan_topic': '/plan',
            'pose_topic': '/localization/pose',
            'scale_topic': '/rl/speed_scale',
            'health_topic': '/rl/healthy',
            'state_topic': '/rl/state',
            'pose_timeout_sec': 0.50,
            'lookahead_m': 0.55,
            # Must match simulation/dynamics.SimProfile, which is what the
            # policy was trained against.
            'repulsion_edge_m': 0.06,
            'repulsion_authority': 0.30,
            # MPPI has its own footprint obstacle critic, but the feed-forward
            # tracker does not. system.launch.py enables this only in tracker
            # mode so both command producers have the same baseline response.
            'apply_baseline_repulsion': False,
            'reference_speed_mps': 0.78,
            'heartbeat_rate_hz': 10.0,
            # Collision Monitor's own projection horizon, read from the
            # deployed Nav2 parameters so the two cannot drift.  This node is
            # the last stage before the monitor on both command paths, so it
            # is where a rotation the monitor is certain to veto gets trimmed;
            # see rl_residual.limit_yaw_rate.
            'nav2_config_file': share + '/config/nav2_next.yaml',
            'limit_yaw_rate_to_clearance': True,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

        self.apply_baseline_repulsion = bool(
            self.get_parameter('apply_baseline_repulsion').value
        )
        self.monitor_horizon = None
        if bool(self.get_parameter('limit_yaw_rate_to_clearance').value):
            self.monitor_horizon = load_collision_monitor_horizon(
                str(self.get_parameter('nav2_config_file').value))

        self.policy = CompactRLPolicy.from_yaml(
            self.get_parameter('policy_file').value
        )
        self.field = CadClearanceModel.from_yaml(
            self.get_parameter('field_config_file').value,
            self.get_parameter('footprint_config_file').value,
            self.get_parameter('footprint_profile').value,
        )
        self.position = None
        self.yaw = 0.0
        self.pose_time = -math.inf
        self.path = None
        self.goal_yaw = None
        self.goal_clearance = None
        self.current_scale = 1.0
        self.current_healthy = False
        self.last_reason = 'WAITING_FOR_LOCALIZATION'
        self.last_decision = None
        self.command_count = 0
        self.learned_decision_count = 0
        self.last_logged_decision = None
        self.heartbeat_count = 0

        control_qos = QoSProfile(depth=1)
        control_qos.reliability = ReliabilityPolicy.RELIABLE
        control_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.output_pub = self.create_publisher(
            Twist, self.get_parameter('output_topic').value, 5
        )
        self.scale_pub = self.create_publisher(
            Float32, self.get_parameter('scale_topic').value, control_qos
        )
        self.health_pub = self.create_publisher(
            Bool, self.get_parameter('health_topic').value, control_qos
        )
        self.state_pub = self.create_publisher(
            String, self.get_parameter('state_topic').value, control_qos
        )
        self.diag_pub = self.create_publisher(DiagnosticArray, '/diagnostics', 10)
        self.create_subscription(
            Twist, self.get_parameter('input_topic').value, self._command_cb, 5
        )
        self.create_subscription(
            Path, self.get_parameter('plan_topic').value, self._plan_cb, 5
        )
        self.cad_override_stamp = -math.inf
        self.create_subscription(String, '/trajectory_tracker/status',
                                 self._tracker_status_cb, control_qos)
        self.create_subscription(
            PoseWithCovarianceStamped,
            self.get_parameter('pose_topic').value,
            self._pose_cb,
            5,
        )
        rate = max(2.0, float(self.get_parameter('heartbeat_rate_hz').value))
        self.create_timer(1.0 / rate, self._heartbeat)
        self.get_logger().info(
            f'Promoted RL policy active with {len(self.policy.overrides)} '
            f'learned states, baseline wall repulsion='
            f'{self.apply_baseline_repulsion}'
        )

    def _tracker_status_cb(self, message):
        self.cad_override_stamp = -math.inf
        try:
            data = json.loads(message.data)
            if data.get('state') == 'REVERSING' and data.get('cad_override') is True:
                self.cad_override_stamp = time.monotonic()
        except (ValueError, AttributeError):
            pass

    def _pose_cb(self, message):
        pose = message.pose.pose
        values = (pose.position.x, pose.position.y, quaternion_yaw(pose.orientation))
        if not all(math.isfinite(value) for value in values):
            self.current_healthy = False
            self.last_reason = 'INVALID_LOCALIZATION'
            return
        self.position = np.asarray(values[:2], dtype=float)
        self.yaw = values[2]
        self.pose_time = time.monotonic()

    def _plan_cb(self, message):
        points = np.asarray([
            [pose.pose.position.x, pose.pose.position.y]
            for pose in message.poses
        ], dtype=float)
        if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] != 2:
            self.path = None
            self.goal_yaw = None
            self.goal_clearance = None
            return
        if not np.all(np.isfinite(points)):
            self.path = None
            self.goal_yaw = None
            self.goal_clearance = None
            return
        self.path = points
        # How tightly the goal pose itself fits is what separates a firing pose
        # beside the fixed bucket from an open one, and it is fixed for the
        # whole plan, so it is computed once here rather than every command.
        goal_yaw = quaternion_yaw(message.poses[-1].pose.orientation)
        if math.isfinite(goal_yaw):
            self.goal_yaw = goal_yaw
            self.goal_clearance = self.field.body_clearance(points[-1], goal_yaw)
        else:
            self.goal_yaw = None
            self.goal_clearance = None
        self.policy.reset()

    @staticmethod
    def _zero_message():
        return Twist()

    def _ready(self, now):
        if self.position is None or now - self.pose_time > float(
            self.get_parameter('pose_timeout_sec').value
        ):
            return False, 'STALE_LOCALIZATION'
        if self.path is None:
            return False, 'NO_GLOBAL_PLAN'
        if self.goal_yaw is None or self.goal_clearance is None:
            return False, 'NO_GOAL_ORIENTATION'
        return True, 'ACTIVE'

    def _command_cb(self, message):
        now = time.monotonic()
        command = np.asarray([
            message.linear.x, message.linear.y, message.angular.z
        ], dtype=float)
        self.command_count += 1
        if command.shape != (3,) or not np.all(np.isfinite(command)):
            self.current_healthy = False
            self.last_reason = 'INVALID_COMMAND'
            self.output_pub.publish(self._zero_message())
            return
        if float(np.linalg.norm(command)) < 1.0e-9:
            self.current_scale = 1.0
            self.current_healthy = self.position is not None
            self.last_reason = 'IDLE'
            self.policy.reset()
            self.output_pub.publish(self._zero_message())
            return

        ready, reason = self._ready(now)
        if not ready:
            self.current_scale = 1.0
            self.current_healthy = False
            self.last_reason = reason
            self.output_pub.publish(self._zero_message())
            return

        if (0. <= now-getattr(self, 'cad_override_stamp', -math.inf) <= .2
                and -.06 <= command[0] <= .02
                and abs(command[1]) <= .02 and abs(command[2]) <= .08):
            # Only fresh, low-speed docking commands bypass the CAD residual.
            # Collision Monitor and RuntimeGuard still inspect this output.
            self.output_pub.publish(message)
            self.current_scale = 1.
            self.current_healthy = True
            self.last_reason = 'DOCKING_CAD_OVERRIDE'
            self.last_decision = None
            self.policy.reset()
            return

        try:
            target, goal = select_path_target(
                self.path,
                self.position,
                float(self.get_parameter('lookahead_m').value),
            )
            _, gradient = self.field.clearance_and_gradient(self.position)
            body_clearance = self.field.body_clearance(self.position, self.yaw)
            observation = make_observation(
                position=self.position,
                yaw=self.yaw,
                body_velocity=command[:2],
                target=target,
                goal=goal,
                body_clearance=body_clearance,
                goal_clearance=self.goal_clearance,
                goal_yaw=self.goal_yaw,
                reference_speed=float(
                    self.get_parameter('reference_speed_mps').value
                ),
            )
            decision = self.policy.decide(observation, now)
            adjusted = apply_clearance_residual(
                command[:2],
                yaw=self.yaw,
                body_clearance=body_clearance,
                gradient=gradient,
                clearance_push=decision.action.clearance_push,
                repulsion_edge=float(
                    self.get_parameter('repulsion_edge_m').value
                ),
                repulsion_authority=float(
                    self.get_parameter('repulsion_authority').value
                ),
                include_baseline=self.apply_baseline_repulsion,
            )
            yaw_rate = float(command[2])
            if self.monitor_horizon is not None:
                yaw_rate = limit_yaw_rate(
                    yaw_rate,
                    position=self.position,
                    yaw=self.yaw,
                    clearance_model=self.field,
                    horizon=self.monitor_horizon,
                )
        except (TypeError, ValueError, FloatingPointError) as error:
            self.current_scale = 1.0
            self.current_healthy = False
            self.last_reason = f'POLICY_ERROR:{error}'
            self.output_pub.publish(self._zero_message())
            return

        output = Twist()
        output.linear.x = float(adjusted[0])
        output.linear.y = float(adjusted[1])
        output.angular.z = float(yaw_rate)
        self.output_pub.publish(output)
        self.current_scale = decision.action.speed_scale
        self.current_healthy = True
        self.last_reason = 'ACTIVE'
        if decision.learned_override:
            self.learned_decision_count += 1
        decision_key = (decision.state, decision.action_index)
        if decision_key != self.last_logged_decision:
            self.get_logger().info(
                f'RL decision state={decision.state}, action={decision.action_index}, '
                f'scale={decision.action.speed_scale:.2f}, '
                f'clearance_push={decision.action.clearance_push:.2f}, '
                f'learned={decision.learned_override}'
            )
            self.last_logged_decision = decision_key
        self.last_decision = {
            'state': list(decision.state),
            'action_index': decision.action_index,
            'learned_override': decision.learned_override,
            'speed_scale': decision.action.speed_scale,
            'clearance_push': decision.action.clearance_push,
            'remaining_distance_m': observation.remaining_distance,
            'body_clearance_m': observation.clearance_margin,
            'goal_clearance_m': observation.goal_clearance_margin,
            'yaw_error_rad': observation.yaw_error,
            'requested_yaw_rate': float(command[2]),
            'yaw_rate': float(yaw_rate),
        }

    def _heartbeat(self):
        now = time.monotonic()
        pose_fresh = self.position is not None and now - self.pose_time <= float(
            self.get_parameter('pose_timeout_sec').value
        )
        healthy = bool(self.current_healthy and pose_fresh)
        self.health_pub.publish(Bool(data=healthy))
        self.scale_pub.publish(Float32(data=float(self.current_scale)))
        state = {
            'healthy': healthy,
            'reason': self.last_reason,
            'speed_scale': self.current_scale,
            'has_plan': self.path is not None,
            'pose_fresh': pose_fresh,
            'command_count': self.command_count,
            'learned_decision_count': self.learned_decision_count,
            'baseline_wall_repulsion': self.apply_baseline_repulsion,
            'decision': self.last_decision,
        }
        self.state_pub.publish(String(data=json.dumps(state, separators=(',', ':'))))
        self.heartbeat_count += 1
        if self.heartbeat_count % 10 == 0:
            status = DiagnosticStatus(
                level=DiagnosticStatus.OK if healthy else DiagnosticStatus.WARN,
                name='omni_autonomy_next/rl_policy',
                message=self.last_reason,
                hardware_id='MU3',
                values=[KeyValue(key=key, value=str(value)) for key, value in state.items()],
            )
            diagnostics = DiagnosticArray()
            diagnostics.header.stamp = self.get_clock().now().to_msg()
            diagnostics.status = [status]
            self.diag_pub.publish(diagnostics)


def main(args=None):
    rclpy.init(args=args)
    node = RLPolicyNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
