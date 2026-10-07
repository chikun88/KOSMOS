import json
import math
import time

from ament_index_python.packages import get_package_share_directory
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
from nav2_msgs.msg import SpeedLimit
import rclpy
from rclpy._rclpy_pybind11 import RCLError
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, Float32, String
import yaml

from .runtime_guard import GuardHealth, MotionLimits, RuntimeGuard


def _limits(data):
    return MotionLimits(
        linear=float(data['linear']), lateral=float(data['lateral']),
        angular=float(data['angular']), linear_accel=float(data['linear_accel']),
        angular_accel=float(data['angular_accel']),
        linear_jerk=float(data['linear_jerk']),
        angular_jerk=float(data['angular_jerk']),
    )


class RuntimeGuardNode(Node):
    def __init__(self):
        super().__init__('runtime_guard')
        defaults = {
            'input_topic': '/cmd_vel_collision_safe',
            'output_topic': '/cmd_vel_safe',
            'command_timeout_sec': 0.25,
            'control_rate_hz': 100.0,
            'require_armed': True,
            'operation_mode': 'unknown',
            'require_tracking': True,
            'tracking_rejection_grace_sec': 0.35,
            'tracking_heartbeat_timeout_sec': 0.50,
            'require_motor_link': True,
            'motor_heartbeat_timeout_sec': 1.0,
            'require_auto_engaged': True,
            'require_rl_policy': True,
            'rl_scale_topic': '/rl/speed_scale',
            'rl_health_topic': '/rl/healthy',
            'rl_heartbeat_timeout_sec': 0.40,
            'default_profile': 'balanced',
            'default_speed_scale': 1.0,
            'speed_limit_topic': '/speed_limit',
            'speed_limit_publish_rate_hz': 2.0,
            'planner_reference_speeds': [0.78, 0.702, 1.30],
            'hard_max_linear_speed': 1.0,
            'hard_max_lateral_speed': 0.85,
            'hard_max_angular_speed': 1.8,
            'hard_max_linear_acceleration': 1.6,
            'hard_max_angular_acceleration': 3.0,
            'hard_max_linear_jerk': 4.0,
            'hard_max_angular_jerk': 8.0,
            'red_zone_radius': 0.95,
            'red_zone_speed_scale': 0.70,
            'red_zone_pose_ids': [0, 2, 3, 4, 5, 6, 7],
            'field_poses_file': '',
            'robot_config_file': '',
            'profiles_json': '{}',
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

        share = get_package_share_directory('omni_autonomy_next')
        robot_file = self.get_parameter('robot_config_file').value or (
            share + '/config/robot.yaml'
        )
        poses_file = self.get_parameter('field_poses_file').value or (
            share + '/config/field_poses.yaml'
        )
        with open(robot_file, encoding='utf-8') as stream:
            robot = yaml.safe_load(stream)['robot']
        with open(poses_file, encoding='utf-8') as stream:
            pose_config = yaml.safe_load(stream)

        drivetrain = robot['drivetrain']
        hard = MotionLimits(
            linear=self.get_parameter('hard_max_linear_speed').value,
            lateral=self.get_parameter('hard_max_lateral_speed').value,
            angular=self.get_parameter('hard_max_angular_speed').value,
            linear_accel=self.get_parameter('hard_max_linear_acceleration').value,
            angular_accel=self.get_parameter('hard_max_angular_acceleration').value,
            linear_jerk=self.get_parameter('hard_max_linear_jerk').value,
            angular_jerk=self.get_parameter('hard_max_angular_jerk').value,
        )
        profiles_data = json.loads(self.get_parameter('profiles_json').value)
        profiles = {name: _limits(value) for name, value in profiles_data.items()}
        if not profiles:
            profiles = {'balanced': hard}
        self.guard = RuntimeGuard(
            profiles=profiles, hard_limits=hard,
            default_profile=self.get_parameter('default_profile').value,
            command_timeout_sec=self.get_parameter('command_timeout_sec').value,
            red_zone_speed_scale=self.get_parameter('red_zone_speed_scale').value,
            wheel_radius=drivetrain['wheel_radius'],
            wheel_positions=drivetrain['wheel_positions'],
            wheel_drive_angles_rad=[
                math.radians(v) for v in drivetrain['wheel_drive_angles_deg']
            ],
            wheel_signs=drivetrain['wheel_signs'],
            max_wheel_speed=drivetrain['max_wheel_speed'],
            profile_max_wheel_speeds=drivetrain.get('profile_max_wheel_speeds', {}),
            translation_budget_share=float(
                drivetrain.get('translation_budget_share', 0.45)
            ),
        )
        ids = {str(v) for v in self.get_parameter('red_zone_pose_ids').value}
        self.red_zone_points = [
            (float(value['x']), float(value['y']))
            for key, value in pose_config.get('poses', {}).items()
            if str(key) in ids and value.get('configured', False)
        ]
        self.red_zone_radius = float(self.get_parameter('red_zone_radius').value)

        self.command = (0.0, 0.0, 0.0)
        self.command_time = -math.inf
        self.tracking_ok = False
        self.tracking_message_time = -math.inf
        self.tracking_last_ok_time = -math.inf
        self.motor_link_ok = False
        self.motor_link_message_time = -math.inf
        self.auto_engaged = False
        self.auto_message_time = -math.inf
        self.armed = False
        self.estop = False
        self.rl_scale = 1.0
        self.rl_healthy = False
        self.rl_message_time = -math.inf
        self.rl_scale_time = -math.inf
        self.rl_scale_valid = False
        self.profile = self.get_parameter('default_profile').value
        self.user_scale = float(self.get_parameter('default_speed_scale').value)
        self.red_zone = False
        self.last_reason = None
        self.tick = 0
        self.planner_reference_speeds = [
            float(value)
            for value in self.get_parameter('planner_reference_speeds').value
        ]
        if len(self.planner_reference_speeds) != 3:
            raise ValueError(
                'planner_reference_speeds must contain [vx, vy, wz] maxima'
            )
        speed_limit_rate = max(
            0.0, float(self.get_parameter('speed_limit_publish_rate_hz').value)
        )
        self.speed_limit_period_sec = (
            1.0 / speed_limit_rate if speed_limit_rate > 0.0 else None
        )
        self.last_speed_limit = None
        self.last_speed_limit_time = -math.inf

        control_qos = QoSProfile(depth=1)
        control_qos.reliability = ReliabilityPolicy.RELIABLE
        control_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        sensor_qos = QoSProfile(depth=5)
        self.output_pub = self.create_publisher(
            Twist, self.get_parameter('output_topic').value, 5
        )
        self.state_pub = self.create_publisher(String, '/system/safety_state', control_qos)
        self.diag_pub = self.create_publisher(DiagnosticArray, '/diagnostics', 10)
        # controller_server latches the newest SpeedLimit, so TRANSIENT_LOCAL
        # also covers a Nav2 restart or the sequenced navigation_delay start.
        self.speed_limit_pub = self.create_publisher(
            SpeedLimit,
            str(self.get_parameter('speed_limit_topic').value),
            control_qos,
        )
        self.create_subscription(
            Twist, self.get_parameter('input_topic').value, self._command_cb, sensor_qos
        )
        self.create_subscription(Bool, '/localization/tracking_ok', self._tracking_cb, 5)
        self.create_subscription(Bool, '/motor/link_ok', self._link_cb, 5)
        self.create_subscription(Bool, '/motor/auto_engaged', self._auto_cb, 5)
        self.create_subscription(Bool, '/system/armed', self._armed_cb, control_qos)
        self.create_subscription(Bool, '/system/emergency_stop', self._estop_cb, control_qos)
        self.create_subscription(Float32, '/system/speed_scale', self._scale_cb, control_qos)
        self.create_subscription(String, '/system/profile', self._profile_cb, control_qos)
        self.create_subscription(
            Float32,
            str(self.get_parameter('rl_scale_topic').value),
            self._rl_scale_cb,
            control_qos,
        )
        self.create_subscription(
            Bool,
            str(self.get_parameter('rl_health_topic').value),
            self._rl_health_cb,
            control_qos,
        )
        self.create_subscription(
            PoseWithCovarianceStamped,
            '/localization/pose',
            self._pose_cb,
            5,
        )
        rate = max(20.0, float(self.get_parameter('control_rate_hz').value))
        self.create_timer(1.0 / rate, self._timer_cb)

    def _command_cb(self, message):
        self.command = (
            float(message.linear.x), float(message.linear.y), float(message.angular.z)
        )
        self.command_time = time.monotonic()

    def _tracking_cb(self, message):
        self.tracking_ok = bool(message.data)
        now = time.monotonic()
        self.tracking_message_time = now
        if self.tracking_ok:
            self.tracking_last_ok_time = now

    def _link_cb(self, message):
        self.motor_link_ok = bool(message.data)
        self.motor_link_message_time = time.monotonic()

    def _auto_cb(self, message):
        self.auto_engaged = bool(message.data)
        self.auto_message_time = time.monotonic()

    def _armed_cb(self, message):
        self.armed = bool(message.data)

    def _estop_cb(self, message):
        self.estop = bool(message.data)

    def _scale_cb(self, message):
        self.user_scale = float(message.data)

    def _profile_cb(self, message):
        self.profile = str(message.data)

    def _rl_scale_cb(self, message):
        value = float(message.data)
        self.rl_scale_valid = math.isfinite(value) and 0.0 < value <= 1.0
        self.rl_scale = value if self.rl_scale_valid else 0.0
        self.rl_scale_time = time.monotonic()

    def _rl_health_cb(self, message):
        self.rl_healthy = bool(message.data)
        self.rl_message_time = time.monotonic()

    def _pose_cb(self, message):
        x = float(message.pose.pose.position.x)
        y = float(message.pose.pose.position.y)
        self.red_zone = any(
            math.hypot(x - px, y - py) <= self.red_zone_radius
            for px, py in self.red_zone_points
        )

    def _publish_speed_limit(self, now):
        """Tell the controller the envelope this guard will pass unclipped."""
        if self.speed_limit_period_sec is None:
            return
        percentage = self.guard.planner_speed_percentage(
            profile=self.profile,
            user_scale=self.user_scale,
            red_zone=self.red_zone,
            reference=self.planner_reference_speeds,
            rl_scale=self.rl_scale,
        )
        changed = (
            self.last_speed_limit is None
            or abs(percentage - self.last_speed_limit) > 0.05
        )
        if not changed and now - self.last_speed_limit_time < (
            self.speed_limit_period_sec
        ):
            return
        message = SpeedLimit()
        message.header.stamp = self.get_clock().now().to_msg()
        message.percentage = True
        message.speed_limit = float(percentage)
        self.speed_limit_pub.publish(message)
        self.last_speed_limit = percentage
        self.last_speed_limit_time = now

    def _timer_cb(self):
        now = time.monotonic()
        require_tracking = bool(self.get_parameter('require_tracking').value)
        require_armed = bool(self.get_parameter('require_armed').value)
        require_link = bool(self.get_parameter('require_motor_link').value)
        require_auto = bool(self.get_parameter('require_auto_engaged').value)
        require_rl = bool(self.get_parameter('require_rl_policy').value)
        tracking_age = now - self.tracking_message_time
        tracking_fresh = 0.0 <= tracking_age <= float(
            self.get_parameter('tracking_heartbeat_timeout_sec').value
        )
        tracking_effective = tracking_fresh and (
            self.tracking_ok
            or now - self.tracking_last_ok_time <= float(
                self.get_parameter('tracking_rejection_grace_sec').value
            )
        )
        rl_age = now - self.rl_message_time
        rl_timeout = float(self.get_parameter('rl_heartbeat_timeout_sec').value)
        rl_effective = (
            self.rl_healthy and self.rl_scale_valid
            and 0.0 <= rl_age <= rl_timeout
            and 0.0 <= now - self.rl_scale_time <= rl_timeout
        )
        motor_timeout = float(self.get_parameter('motor_heartbeat_timeout_sec').value)
        motor_link_effective = (self.motor_link_ok
            and 0.0 <= now - self.motor_link_message_time <= motor_timeout)
        auto_effective = (self.auto_engaged
            and 0.0 <= now - self.auto_message_time <= motor_timeout)
        health = GuardHealth(
            armed=self.armed or not require_armed,
            emergency_stop=self.estop,
            tracking_ok=tracking_effective or not require_tracking,
            motor_link_ok=motor_link_effective or not require_link,
            auto_engaged=auto_effective or not require_auto,
            rl_policy_ok=rl_effective or not require_rl,
        )
        result = self.guard.step(
            self.command, now_sec=now, command_age_sec=now - self.command_time,
            health=health,
            profile=self.profile,
            user_scale=self.user_scale,
            red_zone=self.red_zone,
            rl_scale=self.rl_scale,
        )
        message = Twist()
        message.linear.x, message.linear.y, message.angular.z = result.velocity
        self.output_pub.publish(message)
        self._publish_speed_limit(now)
        if result.reason != self.last_reason:
            # rclpy keys a log call by source location and rejects a changing
            # severity at that location. Keep INFO and WARN on distinct lines.
            if result.allowed:
                self.get_logger().info(f'RuntimeGuard state: {result.reason}')
            else:
                self.get_logger().warning(
                    f'RuntimeGuard state: {result.reason}; '
                    f'command_age={now - self.command_time:.3f}s; '
                    f'health={self.guard._health_reason(health)}; '
                    f'user_scale={self.user_scale:.3f}')
            self.last_reason = result.reason
        self.tick += 1
        if self.tick % 10 == 0:
            state = {
                'allowed': result.allowed, 'reason': result.reason,
                'operation_mode': self.get_parameter('operation_mode').value,
                'require_motor_link': require_link,
                'profile': result.profile, 'requested_scale': self.user_scale,
                'rl_scale': self.rl_scale,
                'applied_scale': result.applied_scale, 'red_zone': result.red_zone,
                'reference_scale': self.guard.reference_scale(
                    health, self.user_scale, self.red_zone, self.rl_scale),
                'health_reason': self.guard._health_reason(health),
                'command_age_sec': (round(now - self.command_time, 3)
                                    if math.isfinite(self.command_time) else None),
                'planner_speed_limit_pct': self.last_speed_limit,
                'tracking_ok': self.tracking_ok,
                'tracking_effective': tracking_effective,
                'tracking_age_sec': (
                    round(tracking_age, 3) if math.isfinite(tracking_age) else None
                ),
                'motor_link_ok': self.motor_link_ok,
                'motor_link_effective': motor_link_effective,
                'auto_effective': auto_effective,
                'auto_engaged': self.auto_engaged,
                'rl_healthy': self.rl_healthy,
                'rl_effective': rl_effective,
                'rl_age_sec': round(rl_age, 3) if math.isfinite(rl_age) else None,
                'require_rl_policy': require_rl,
                'armed': self.armed,
                'require_armed': require_armed,
                'emergency_stop': self.estop, 'velocity': list(result.velocity),
            }
            state_message = String(data=json.dumps(state, separators=(',', ':')))
            self.state_pub.publish(state_message)
            status = DiagnosticStatus(
                level=DiagnosticStatus.OK if result.allowed else DiagnosticStatus.WARN,
                name='omni_autonomy_next/runtime_guard',
                message=result.reason,
                hardware_id='MU3',
                values=[KeyValue(key=k, value=str(v)) for k, v in state.items()],
            )
            diagnostics = DiagnosticArray()
            diagnostics.header.stamp = self.get_clock().now().to_msg()
            diagnostics.status = [status]
            self.diag_pub.publish(diagnostics)


def main(args=None):
    rclpy.init(args=args)
    node = RuntimeGuardNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except RCLError:
        if rclpy.ok():
            raise
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
