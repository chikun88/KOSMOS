from collections import deque
import math
from typing import Deque, Optional

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped, Twist
from nav_msgs.msg import Odometry
from rclpy._rclpy_pybind11 import RCLError
from rclpy.exceptions import ParameterUninitializedException
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from tf2_ros import TransformBroadcaster

from .config import ConfigError, load_field, load_robot
from .geometry import compose_pose, raycast_segments


NANOSECONDS_PER_SECOND = 1_000_000_000


def completed_scan_start_nanoseconds(
    publish_nanoseconds: int, scan_time: float
) -> int:
    """Return the first-beam time for a scan completed at publication."""
    duration = float(scan_time)
    if not math.isfinite(duration) or duration <= 0.0:
        raise ValueError('scan_time must be finite and positive')
    return max(
        0,
        int(publish_nanoseconds)
        - int(round(duration * NANOSECONDS_PER_SECOND)),
    )


def set_yaw(quaternion, yaw: float) -> None:
    quaternion.x = 0.0
    quaternion.y = 0.0
    quaternion.z = math.sin(0.5 * yaw)
    quaternion.w = math.cos(0.5 * yaw)


def raycast_circles(
    origin: np.ndarray,
    directions: np.ndarray,
    circles: np.ndarray,
) -> np.ndarray:
    """Nearest positive ray/circle intersection distances (circles: N x [x, y, r])."""
    if len(circles) == 0:
        return np.full(len(directions), np.inf)
    to_center = circles[None, :, :2] - origin[None, None, :]
    along = np.sum(directions[:, None, :] * to_center, axis=2)
    closest_sq = np.sum(to_center * to_center, axis=2) - along * along
    radii_sq = circles[None, :, 2] * circles[None, :, 2]
    half_chord = np.sqrt(np.maximum(radii_sq - closest_sq, 0.0))
    near = along - half_chord
    far = along + half_chord
    distance = np.where(near > 0.0, near, np.where(far > 0.0, far, np.inf))
    distance = np.where(closest_sq <= radii_sq, distance, np.inf)
    return np.min(distance, axis=1)


class SyntheticScanNode(Node):
    """Dual-LiDAR raycast simulator with an optional closed-loop robot model.

    With simulate_motion enabled the node integrates /cmd_vel into the true
    pose and can publish measurement-wheel-style odometry, so the complete
    localization -> planning -> following -> command chain can be exercised
    end-to-end without hardware.
    """

    def __init__(self) -> None:
        super().__init__('synthetic_scans')
        self.declare_parameter('field_config_file', '')
        self.declare_parameter('robot_config_file', '')
        self.declare_parameter('true_pose', [1.35, 1.20, 0.18])
        self.declare_parameter('publish_rate_hz', 10.0)
        self.declare_parameter('samples', 720)
        self.declare_parameter('range_min', 0.15)
        self.declare_parameter('range_max', 12.0)
        self.declare_parameter('noise_stddev', 0.004)
        self.declare_parameter('seed', 7)
        self.declare_parameter('simulate_motion', True)
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('cmd_timeout_sec', 0.4)
        self.declare_parameter('motion_rate_hz', 50.0)
        # Drivebase model between /cmd_vel_safe and actual motion.  Both
        # default to 0.0, which is the ideal integrator this node has always
        # been, so existing demo evidence keeps its meaning.  Setting them
        # reproduces what the real chain adds beyond the ROS graph:
        #   command_delay_sec        - UDP hop, the Pi's 200 Hz loop, the
        #                              115200 baud UART frame and the dev
        #                              board's own command period: dead time.
        #   velocity_time_constant_sec - C620 current loop, gearbox and wheel
        #                              inertia: first-order velocity lag.
        # Dead time is what destabilizes a cross-track loop, so these are the
        # knobs that turn this demo into a stability-margin measurement rather
        # than a kinematic replay.  See docs/ACCEPTANCE.md step 6.
        self.declare_parameter('command_delay_sec', 0.0)
        self.declare_parameter('velocity_time_constant_sec', 0.0)
        # '' disables the simulated wheel odometry publisher.
        self.declare_parameter('wheel_odom_topic', '/wheel/odometry')
        self.declare_parameter('odom_frame_id', 'odom')
        # Mirror the real measurement_wheel node, which broadcasts the
        # odom->base_link transform; Nav2's local costmap needs it.
        self.declare_parameter('publish_wheel_tf', True)
        # Nav2 controller_server (DWB) reads current velocity from the fixed
        # /odom topic; mirror measurement_wheel and publish there too.
        self.declare_parameter('standard_odom_topic', '/odom')
        # 動的な未知障害物(人・相手ロボット役)の円柱: [x, y, r, x, y, r, ...]
        # (map座標)。実行中に ros2 param set で置く/消すことで、障害物
        # 回避・脱出のE2E検証に使う。
        self.declare_parameter(
            'extra_obstacles', Parameter.Type.DOUBLE_ARRAY
        )

        field_path = self.get_parameter('field_config_file').value
        robot_path = self.get_parameter('robot_config_file').value
        if not field_path or not robot_path:
            raise ConfigError(
                'field_config_file and robot_config_file are required'
            )
        self.field = load_field(field_path)
        self.robot = load_robot(robot_path)
        self.true_pose = np.asarray(
            self.get_parameter('true_pose').value, dtype=float
        )
        self.samples = int(self.get_parameter('samples').value)
        self.range_min = float(self.get_parameter('range_min').value)
        self.range_max = float(self.get_parameter('range_max').value)
        self.noise = float(self.get_parameter('noise_stddev').value)
        self.random = np.random.default_rng(
            int(self.get_parameter('seed').value)
        )
        self.scan_publishers = {
            lidar['name']: self.create_publisher(
                LaserScan, lidar['topic'], qos_profile_sensor_data
            )
            for lidar in self.robot['lidars']
        }
        rate = float(self.get_parameter('publish_rate_hz').value)
        if not math.isfinite(rate) or rate <= 0.0:
            raise ConfigError('publish_rate_hz must be finite and positive')
        self.scan_period = 1.0 / rate
        # Raycasting must not starve command reception or wheel odometry.
        # Both motion callbacks remain serialized in the default group.
        self.scan_callback_group = MutuallyExclusiveCallbackGroup()
        self.timer = self.create_timer(
            self.scan_period, self._publish,
            callback_group=self.scan_callback_group)

        self.latest_command = np.zeros(3)
        self.latest_command_time = None
        self.command_delay_sec = max(
            0.0, float(self.get_parameter('command_delay_sec').value)
        )
        self.velocity_time_constant = max(
            0.0, float(self.get_parameter('velocity_time_constant_sec').value)
        )
        self.delayed_commands: Deque = deque()
        self.actual_velocity = np.zeros(3)
        self.wheel_pose = np.zeros(3)
        self.odom_publisher = None
        self.standard_odom_publisher = None
        self.tf_broadcaster = None
        if bool(self.get_parameter('simulate_motion').value):
            self.create_subscription(
                Twist,
                str(self.get_parameter('cmd_vel_topic').value),
                self._cmd_callback,
                10,
            )
            wheel_topic = str(self.get_parameter('wheel_odom_topic').value)
            if wheel_topic:
                self.odom_publisher = self.create_publisher(
                    Odometry, wheel_topic, 20
                )
                standard_topic = str(
                    self.get_parameter('standard_odom_topic').value
                )
                if standard_topic and standard_topic != wheel_topic:
                    self.standard_odom_publisher = self.create_publisher(
                        Odometry, standard_topic, 20
                    )
                if bool(self.get_parameter('publish_wheel_tf').value):
                    self.tf_broadcaster = TransformBroadcaster(self)
            motion_rate = max(
                5.0, float(self.get_parameter('motion_rate_hz').value)
            )
            self.motion_dt = 1.0 / motion_rate
            self.create_timer(self.motion_dt, self._motion_step)
        self.get_logger().info(
            f'Publishing synthetic dual-LiDAR scans at true pose '
            f'{self.true_pose.tolist()} '
            f'(simulate_motion={bool(self.get_parameter("simulate_motion").value)})'
        )

    def _cmd_callback(self, message: Twist) -> None:
        command = np.array([
            message.linear.x,
            message.linear.y,
            message.angular.z,
        ])
        now = self.get_clock().now()
        self.latest_command_time = now
        if self.command_delay_sec > 0.0:
            self.delayed_commands.append((
                now.nanoseconds * 1.0e-9 + self.command_delay_sec, command
            ))
        else:
            self.latest_command = command

    def _motion_step(self) -> None:
        now = self.get_clock().now()
        now_sec = now.nanoseconds * 1.0e-9
        # Release every command whose transport dead time has elapsed. Keeping
        # only the newest matches the Pi, which drains its socket each loop and
        # applies the last datagram.
        while self.delayed_commands and self.delayed_commands[0][0] <= now_sec:
            self.latest_command = self.delayed_commands.popleft()[1]
        command = self.latest_command
        if self.latest_command_time is None:
            command = np.zeros(3)
        else:
            age = (now - self.latest_command_time).nanoseconds * 1.0e-9
            if age > float(self.get_parameter('cmd_timeout_sec').value):
                command = np.zeros(3)
                self.delayed_commands.clear()
        if self.velocity_time_constant > 0.0:
            blend = 1.0 - math.exp(
                -self.motion_dt / self.velocity_time_constant
            )
            self.actual_velocity = (
                self.actual_velocity
                + (command - self.actual_velocity) * blend
            )
        else:
            self.actual_velocity = command
        delta = self.actual_velocity * self.motion_dt
        self.true_pose = compose_pose(self.true_pose, delta)
        self.wheel_pose = compose_pose(self.wheel_pose, delta)
        if self.odom_publisher is not None:
            message = Odometry()
            message.header.stamp = self.get_clock().now().to_msg()
            message.header.frame_id = str(
                self.get_parameter('odom_frame_id').value
            )
            message.child_frame_id = self.robot['base_frame_id']
            message.pose.pose.position.x = float(self.wheel_pose[0])
            message.pose.pose.position.y = float(self.wheel_pose[1])
            set_yaw(message.pose.pose.orientation, float(self.wheel_pose[2]))
            # Report the velocity the base is actually moving at, not the one
            # that was asked for. The measurement wheels see the wheel, so a
            # simulated actuator lag has to be visible to everything that
            # closes a loop on /wheel/odometry.
            message.twist.twist.linear.x = float(self.actual_velocity[0])
            message.twist.twist.linear.y = float(self.actual_velocity[1])
            message.twist.twist.angular.z = float(self.actual_velocity[2])
            self.odom_publisher.publish(message)
            if self.standard_odom_publisher is not None:
                self.standard_odom_publisher.publish(message)
            if self.tf_broadcaster is not None:
                transform = TransformStamped()
                transform.header.stamp = message.header.stamp
                transform.header.frame_id = message.header.frame_id
                transform.child_frame_id = message.child_frame_id
                transform.transform.translation.x = float(self.wheel_pose[0])
                transform.transform.translation.y = float(self.wheel_pose[1])
                set_yaw(
                    transform.transform.rotation, float(self.wheel_pose[2])
                )
                self.tf_broadcaster.sendTransform(transform)

    def _extra_circles(self) -> np.ndarray:
        try:
            values = self.get_parameter('extra_obstacles').value
        except ParameterUninitializedException:
            # 宣言のみ(型だけ)で未設定の間は障害物なし。
            return np.zeros((0, 3))
        if not values or len(values) < 3:
            return np.zeros((0, 3))
        count = len(values) // 3
        return np.asarray(values[: count * 3], dtype=float).reshape(-1, 3)

    def _publish(self) -> None:
        # _motion_step replaces this array rather than modifying it in place.
        # Snapshot once so both LiDARs see the same pose while motion proceeds.
        scan_pose = self.true_pose.copy()
        angle_min = -math.pi
        angle_increment = 2.0 * math.pi / self.samples
        local_angles = angle_min + np.arange(self.samples) * angle_increment
        # LaserScan.header.stamp is the acquisition time of the first ray.
        # This callback publishes a completed revolution, so dating its first
        # ray at publication time placed the final rays ~one scan period in
        # the future. The orchestrator correctly rejected those rays because
        # no future odometry exists, which disabled every high-speed sprint
        # in demo mode while slower Nav2 remained usable.
        publish_time = self.get_clock().now()
        stamp = Time(
            nanoseconds=completed_scan_start_nanoseconds(
                publish_time.nanoseconds, self.scan_period
            ),
            clock_type=publish_time.clock_type,
        ).to_msg()
        circles = self._extra_circles()

        for lidar in self.robot['lidars']:
            lidar_config = lidar['pose']
            base_to_lidar = np.array([
                lidar_config['x'],
                lidar_config['y'],
                lidar_config['yaw'],
            ])
            map_to_lidar = compose_pose(scan_pose, base_to_lidar)
            map_angles = local_angles + map_to_lidar[2]
            directions = np.column_stack((
                np.cos(map_angles),
                np.sin(map_angles),
            ))
            ranges = raycast_segments(
                map_to_lidar[:2], directions, self.field['walls']
            )
            if len(circles):
                ranges = np.minimum(
                    ranges,
                    raycast_circles(map_to_lidar[:2], directions, circles),
                )
            finite = np.isfinite(ranges)
            ranges[finite] += self.random.normal(0.0, self.noise, np.count_nonzero(finite))
            valid = finite & (ranges >= self.range_min) & (ranges <= self.range_max)
            ranges[~valid] = np.inf

            message = LaserScan()
            message.header.stamp = stamp
            message.header.frame_id = lidar['frame_id']
            message.angle_min = angle_min
            message.angle_max = angle_min + (self.samples - 1) * angle_increment
            message.angle_increment = angle_increment
            message.scan_time = self.scan_period
            message.time_increment = self.scan_period / self.samples
            message.range_min = self.range_min
            message.range_max = self.range_max
            message.ranges = ranges.astype(np.float32).tolist()
            self.scan_publishers[lidar['name']].publish(message)


def main(args: Optional[list] = None) -> None:
    rclpy.init(args=args)
    node = SyntheticScanNode()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except RCLError:
        if rclpy.ok():
            raise
    except RuntimeError:
        # rclpy may surface a pybind take-message conversion error when SIGINT
        # invalidates the subscription context between wait and take. Treat it
        # as normal only after shutdown; runtime conversion failures still
        # propagate.
        if rclpy.ok():
            raise
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
