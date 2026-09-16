import json
import math
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rcl_interfaces.msg import SetParametersResult
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.parameter import Parameter as RclpyParameter
from std_msgs.msg import Float64MultiArray, String
from std_srvs.srv import Trigger
from tf2_ros import TransformBroadcaster

from .config import ConfigError, load_robot
from .contec_cnt import ContecCounter, ContecCounterError
from .measurement_wheel_odometry import MeasurementWheelKinematics
from .measurement_wheel_power import (
    MeasurementWheelPowerSwitch,
    PowerControlConfig,
    PowerControlError,
)


COUNTER_SIGNAL_TYPES = {
    'isolate': ContecCounter.CNT_SIGTYPE_ISOLATE,
    'isolated': ContecCounter.CNT_SIGTYPE_ISOLATE,
    'ttl': ContecCounter.CNT_SIGTYPE_TTL,
    'line_receiver': ContecCounter.CNT_SIGTYPE_LINERECEIVER,
    'linereceiver': ContecCounter.CNT_SIGTYPE_LINERECEIVER,
}
COUNTER_DIRECTIONS = {
    'up': ContecCounter.CNT_DIR_UP,
    'down': ContecCounter.CNT_DIR_DOWN,
}
COUNTER_PHASES = {
    '1phase': ContecCounter.CNT_MODE_1PHASE,
    '1_phase': ContecCounter.CNT_MODE_1PHASE,
    '2phase': ContecCounter.CNT_MODE_2PHASE,
    '2_phase': ContecCounter.CNT_MODE_2PHASE,
    'gate': ContecCounter.CNT_MODE_GATECONTROL,
    'gate_control': ContecCounter.CNT_MODE_GATECONTROL,
}
COUNTER_MULTIPLIERS = {
    'x1': ContecCounter.CNT_MUL_X1,
    '1': ContecCounter.CNT_MUL_X1,
    'x2': ContecCounter.CNT_MUL_X2,
    '2': ContecCounter.CNT_MUL_X2,
    'x4': ContecCounter.CNT_MUL_X4,
    '4': ContecCounter.CNT_MUL_X4,
}
COUNTER_CLEAR_MODES = {
    'async': ContecCounter.CNT_CLR_ASYNC,
    'asynchronous': ContecCounter.CNT_CLR_ASYNC,
    'sync': ContecCounter.CNT_CLR_SYNC,
    'synchronous': ContecCounter.CNT_CLR_SYNC,
}
COUNTER_Z_PHASES = {
    'not_use': ContecCounter.CNT_ZPHASE_NOT_USE,
    'unused': ContecCounter.CNT_ZPHASE_NOT_USE,
    'next_one': ContecCounter.CNT_ZPHASE_NEXT_ONE,
    'every_time': ContecCounter.CNT_ZPHASE_EVERY_TIME,
}
COUNTER_Z_LOGIC = {
    'positive': ContecCounter.CNT_ZLOGIC_POSITIVE,
    'negative': ContecCounter.CNT_ZLOGIC_NEGATIVE,
}


def set_yaw(quaternion, yaw: float) -> None:
    quaternion.x = 0.0
    quaternion.y = 0.0
    quaternion.z = math.sin(0.5 * yaw)
    quaternion.w = math.cos(0.5 * yaw)


class MeasurementWheelNode(Node):
    """Read a Contec CNT-3204IN-USB and publish wheel odometry."""

    def __init__(self) -> None:
        super().__init__('measurement_wheel')
        self._declare_parameters()

        settings = self._load_settings()
        self.device_name = settings['device_name']
        self.channels = settings['channels']
        self.odom_frame = settings['odom_frame_id']
        self.base_frame = settings['base_frame_id']
        self.publish_tf = settings['publish_tf']
        self.max_update_gap_sec = settings['max_update_gap_sec']
        self.max_delta_translation = settings['max_delta_translation']
        self.max_delta_rotation = settings['max_delta_rotation']
        self.power_switch = MeasurementWheelPowerSwitch(
            settings['power_control']
        )

        self.kinematics = MeasurementWheelKinematics(
            wheel_positions=settings['wheel_positions'],
            wheel_drive_angles=np.radians(settings['wheel_drive_angles_deg']),
            count_signs=settings['count_signs'],
            counter_bits=settings['counter_bits'],
            meters_per_count=settings['meters_per_count'],
            counts_per_revolution=settings['counts_per_revolution'],
            wheel_radius=settings['wheel_radius'],
            odometry_scale=settings['odometry_scale'],
            calibration_matrix=settings['calibration_matrix'],
        )
        if self.kinematics.calibration_matrix is not None:
            self.get_logger().info(
                'Using measurement_wheels.calibration_matrix; wheel geometry, '
                'count_signs and odometry_scale are bypassed for odometry.'
            )
        self._sync_odometry_scale_parameters(settings['odometry_scale'])
        self.add_on_set_parameters_callback(self._parameters_callback)

        self.counter: Optional[ContecCounter] = None
        try:
            self.power_switch.turn_on()
            self.counter = ContecCounter(
                device_name=self.device_name,
                channels=self.channels,
                library_path=settings['cnt_library'],
            )
            self.counter.open()
            self.counter.configure_channels(**settings['counter_mode'])
            self.counter.start()
        except Exception:
            self.power_switch.close()
            raise

        self.pose = np.zeros(3)
        self.previous_counts: Optional[list] = None
        self.latest_counts: Optional[list] = None
        self.last_time = None
        self.last_status: Dict[str, object] = {
            'state': 'STARTING',
            'device_name': self.device_name,
            'channels': self.channels,
            'odometry_scale': [
                float(value) for value in self.kinematics.odometry_scale
            ],
        }

        self.odom_publisher = self.create_publisher(
            Odometry,
            str(self.get_parameter('odom_topic').value),
            20,
        )
        # Nav2 controller_server (DWB) reads the current velocity from the
        # fixed topic /odom (not remappable via params in Humble). Without it
        # DWB assumes v=0 forever and can only command ~1.5 cm/s per cycle.
        self.standard_odom_publisher = None
        standard_topic = str(
            self.get_parameter('standard_odom_topic').value
        )
        if standard_topic and standard_topic != str(
            self.get_parameter('odom_topic').value
        ):
            self.standard_odom_publisher = self.create_publisher(
                Odometry, standard_topic, 20
            )
        self.count_publisher = self.create_publisher(
            Float64MultiArray,
            str(self.get_parameter('count_topic').value),
            10,
        )
        self.delta_publisher = self.create_publisher(
            Float64MultiArray,
            str(self.get_parameter('delta_count_topic').value),
            10,
        )
        self.status_publisher = self.create_publisher(
            String,
            str(self.get_parameter('status_topic').value),
            10,
        )
        self._last_status_state: Optional[str] = None
        self._last_status_publish_ns = 0
        self.create_service(
            Trigger,
            str(self.get_parameter('reset_odometry_service').value),
            self._reset_odometry_callback,
        )
        self.create_service(
            Trigger,
            str(self.get_parameter('save_odometry_scale_service').value),
            self._save_odometry_scale_callback,
        )
        self.tf_broadcaster = TransformBroadcaster(self)

        poll_rate = max(1.0, float(self.get_parameter('poll_rate_hz').value))
        self.create_timer(1.0 / poll_rate, self._timer_callback)
        self.get_logger().info(
            f'CNT measurement wheel ready: device={self.device_name}, '
            f'channels={self.channels}, odom_topic='
            f'{self.get_parameter("odom_topic").value}'
        )

    def _declare_parameters(self) -> None:
        defaults = {
            'robot_config_file': '',
            'use_robot_config_measurement_wheels': True,
            'device_name': 'CNT000',
            'cnt_library': '',
            'channels': [0, 1, 2, 3],
            'counter_bits': 32,
            'count_signs': [1.0, 1.0, 1.0, 1.0],
            'counts_per_revolution': [2048.0, 2048.0, 2048.0, 2048.0],
            'meters_per_count': [0.0, 0.0, 0.0, 0.0],
            'wheel_radius': 0.0508,
            'wheel_positions': [
                0.333072, 0.333072,
                0.333072, -0.333072,
                -0.333072, 0.333072,
                -0.333072, -0.333072,
            ],
            'wheel_drive_angles_deg': [-45.0, 45.0, 45.0, -45.0],
            'odometry_scale_x': 1.0,
            'odometry_scale_y': 1.0,
            'odometry_scale_yaw': 1.0,
            'odom_topic': '/wheel/odometry',
            'standard_odom_topic': '/odom',
            'count_topic': '/wheel/counts',
            'delta_count_topic': '/wheel/delta_counts',
            'status_topic': '/wheel/status',
            'reset_odometry_service': '/wheel/reset_odometry',
            'save_odometry_scale_service': '/wheel/save_odometry_scale',
            'odom_frame_id': 'odom',
            'base_frame_id': 'base_link',
            'publish_tf': True,
            'poll_rate_hz': 50.0,
            'max_update_gap_sec': 0.25,
            'max_delta_translation': 0.50,
            'max_delta_rotation': 1.0,
            'pose_covariance_xy': 0.02,
            'pose_covariance_yaw': 0.05,
            'twist_covariance_linear': 0.05,
            'twist_covariance_angular': 0.10,
            'power_control_enabled': False,
            'power_gpio_pin': -1,
            'power_gpio_mode': 'BOARD',
            'power_gpio_backend': 'auto',
            'power_active_high': True,
            'power_settle_sec': 0.2,
            'power_off_on_shutdown': True,
            'cnt_signal_type': 'isolate',
            'cnt_count_direction': 'up',
            'cnt_operation_phase': '2phase',
            'cnt_multiplier': 'x1',
            'cnt_sync_clear': 'async',
            'cnt_z_phase': 'not_use',
            'cnt_z_logic': 'positive',
            'cnt_digital_filter': 0,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

    def _load_settings(self) -> Dict[str, object]:
        settings: Dict[str, object] = {
            'device_name': str(self.get_parameter('device_name').value),
            'cnt_library': str(self.get_parameter('cnt_library').value),
            'channels': [
                int(value) for value in self.get_parameter('channels').value
            ],
            'counter_bits': int(self.get_parameter('counter_bits').value),
            'count_signs': np.asarray(
                self.get_parameter('count_signs').value, dtype=float
            ),
            'counts_per_revolution': np.asarray(
                self.get_parameter('counts_per_revolution').value, dtype=float
            ),
            'wheel_radius': float(self.get_parameter('wheel_radius').value),
            'wheel_positions': np.asarray(
                self.get_parameter('wheel_positions').value, dtype=float
            ).reshape(-1, 2),
            'wheel_drive_angles_deg': np.asarray(
                self.get_parameter('wheel_drive_angles_deg').value, dtype=float
            ),
            'odometry_scale': np.asarray(
                [
                    self.get_parameter('odometry_scale_x').value,
                    self.get_parameter('odometry_scale_y').value,
                    self.get_parameter('odometry_scale_yaw').value,
                ],
                dtype=float,
            ),
            'odom_frame_id': str(self.get_parameter('odom_frame_id').value),
            'base_frame_id': str(self.get_parameter('base_frame_id').value),
            'publish_tf': bool(self.get_parameter('publish_tf').value),
            'max_update_gap_sec': float(
                self.get_parameter('max_update_gap_sec').value
            ),
            'max_delta_translation': float(
                self.get_parameter('max_delta_translation').value
            ),
            'max_delta_rotation': float(
                self.get_parameter('max_delta_rotation').value
            ),
            'power_control': PowerControlConfig(
                enabled=bool(
                    self.get_parameter('power_control_enabled').value
                ),
                gpio_pin=int(self.get_parameter('power_gpio_pin').value),
                gpio_mode=str(self.get_parameter('power_gpio_mode').value),
                gpio_backend=str(
                    self.get_parameter('power_gpio_backend').value
                ),
                active_high=bool(self.get_parameter('power_active_high').value),
                settle_sec=float(self.get_parameter('power_settle_sec').value),
                off_on_shutdown=bool(
                    self.get_parameter('power_off_on_shutdown').value
                ),
            ),
            'counter_mode': self._counter_mode_from_parameters(),
            'calibration_matrix': None,
        }
        meters_per_count = np.asarray(
            self.get_parameter('meters_per_count').value, dtype=float
        )
        settings['meters_per_count'] = (
            meters_per_count
            if meters_per_count.size > 0 and np.all(meters_per_count > 0.0)
            else None
        )

        robot_path = str(self.get_parameter('robot_config_file').value)
        if (
            robot_path
            and bool(self.get_parameter('use_robot_config_measurement_wheels').value)
        ):
            robot = load_robot(robot_path)
            settings['base_frame_id'] = robot['base_frame_id']
            wheel_config = robot['measurement_wheels']
            power_config = wheel_config['power_control']
            settings.update({
                'device_name': wheel_config['device_name'],
                'channels': wheel_config['channels'],
                'counter_bits': wheel_config['counter_bits'],
                'count_signs': wheel_config['count_signs'],
                'counts_per_revolution': wheel_config['counts_per_revolution'],
                'meters_per_count': wheel_config['meters_per_count'],
                'wheel_radius': wheel_config['wheel_radius'],
                'wheel_positions': wheel_config['wheel_positions'],
                'wheel_drive_angles_deg': wheel_config['wheel_drive_angles_deg'],
                'odometry_scale': wheel_config['odometry_scale'],
                'calibration_matrix': wheel_config['calibration_matrix'],
                'odom_frame_id': wheel_config['odom_frame_id'],
                'publish_tf': wheel_config['publish_tf'],
                'counter_mode': self._normalize_counter_mode(
                    wheel_config['counter_mode']
                ),
                'power_control': PowerControlConfig(
                    enabled=power_config['enabled'],
                    gpio_pin=power_config['gpio_pin'],
                    gpio_mode=power_config['gpio_mode'],
                    gpio_backend=power_config['gpio_backend'],
                    active_high=power_config['active_high'],
                    settle_sec=power_config['settle_sec'],
                    off_on_shutdown=power_config['off_on_shutdown'],
                ),
            })
        return settings

    def _sync_odometry_scale_parameters(self, scale) -> None:
        values = np.asarray(scale, dtype=float)
        self.set_parameters([
            RclpyParameter('odometry_scale_x', value=float(values[0])),
            RclpyParameter('odometry_scale_y', value=float(values[1])),
            RclpyParameter('odometry_scale_yaw', value=float(values[2])),
        ])

    def _parameters_callback(self, parameters) -> SetParametersResult:
        index_by_name = {
            'odometry_scale_x': 0,
            'odometry_scale_y': 1,
            'odometry_scale_yaw': 2,
        }
        scale = self.kinematics.odometry_scale.copy()
        changed = False
        for parameter in parameters:
            index = index_by_name.get(parameter.name)
            if index is None:
                continue
            try:
                scale[index] = float(parameter.value)
            except (TypeError, ValueError):
                return SetParametersResult(
                    successful=False,
                    reason=f'{parameter.name} must be a number',
                )
            changed = True
        if not changed:
            return SetParametersResult(successful=True)
        try:
            self.kinematics.set_odometry_scale(scale)
        except ValueError as error:
            return SetParametersResult(successful=False, reason=str(error))
        self.get_logger().info(
            'measurement wheel odometry_scale updated: '
            f'x={scale[0]:.3f}, y={scale[1]:.3f}, yaw={scale[2]:.3f}'
        )
        return SetParametersResult(successful=True)

    def _counter_mode_from_parameters(self) -> Dict[str, int]:
        return self._normalize_counter_mode({
            'signal_type': self.get_parameter('cnt_signal_type').value,
            'count_direction': self.get_parameter('cnt_count_direction').value,
            'operation_phase': self.get_parameter('cnt_operation_phase').value,
            'multiplier': self.get_parameter('cnt_multiplier').value,
            'sync_clear': self.get_parameter('cnt_sync_clear').value,
            'z_phase': self.get_parameter('cnt_z_phase').value,
            'z_logic': self.get_parameter('cnt_z_logic').value,
            'digital_filter': self.get_parameter('cnt_digital_filter').value,
        })

    def _normalize_counter_mode(self, mode: Dict[str, object]) -> Dict[str, int]:
        return {
            'signal_type': self._enum_value(
                mode['signal_type'], COUNTER_SIGNAL_TYPES, 'cnt_signal_type'
            ),
            'count_direction': self._enum_value(
                mode['count_direction'],
                COUNTER_DIRECTIONS,
                'cnt_count_direction',
            ),
            'operation_phase': self._enum_value(
                mode['operation_phase'],
                COUNTER_PHASES,
                'cnt_operation_phase',
            ),
            'multiplier': self._enum_value(
                mode['multiplier'], COUNTER_MULTIPLIERS, 'cnt_multiplier'
            ),
            'sync_clear': self._enum_value(
                mode['sync_clear'], COUNTER_CLEAR_MODES, 'cnt_sync_clear'
            ),
            'z_phase': self._enum_value(
                mode['z_phase'], COUNTER_Z_PHASES, 'cnt_z_phase'
            ),
            'z_logic': self._enum_value(
                mode['z_logic'], COUNTER_Z_LOGIC, 'cnt_z_logic'
            ),
            'digital_filter': int(mode['digital_filter']),
        }

    @staticmethod
    def _enum_value(value: object, mapping: Dict[str, int], name: str) -> int:
        if isinstance(value, int):
            return value
        key = str(value).strip().lower().replace('-', '_')
        if key in mapping:
            return mapping[key]
        choices = ', '.join(sorted(mapping))
        raise ValueError(f'{name} must be one of: {choices}')

    def _timer_callback(self) -> None:
        now = self.get_clock().now()
        try:
            raw_counts = self.counter.read()
        except ContecCounterError as error:
            self.get_logger().error(str(error))
            self.last_status = {'state': 'READ_ERROR', 'error': str(error)}
            self._publish_status()
            return

        self.latest_counts = list(raw_counts)
        self._publish_array(self.count_publisher, raw_counts)
        if self.previous_counts is None:
            self.previous_counts = raw_counts
            self.last_time = now
            self.last_status = {
                'state': 'PRIMED',
                'raw_counts': raw_counts,
            }
            self._publish_status()
            return

        dt = (now - self.last_time).nanoseconds * 1.0e-9
        if dt <= 0.0:
            return
        if dt > self.max_update_gap_sec:
            self.previous_counts = raw_counts
            self.last_time = now
            self.last_status = {
                'state': 'SKIPPED_LONG_GAP',
                'dt': round(dt, 6),
                'raw_counts': raw_counts,
            }
            self._publish_status()
            return

        delta_counts, wheel_displacements, body_delta = self.kinematics.step(
            raw_counts, self.previous_counts
        )
        translation = float(np.linalg.norm(body_delta[:2]))
        rotation = abs(float(body_delta[2]))
        if (
            translation > self.max_delta_translation
            or rotation > self.max_delta_rotation
        ):
            self.previous_counts = raw_counts
            self.last_time = now
            self.last_status = {
                'state': 'SKIPPED_LARGE_DELTA',
                'translation': translation,
                'rotation': rotation,
                'delta_counts': [float(value) for value in delta_counts],
            }
            self._publish_status()
            return

        self.pose = self.kinematics.integrate_pose(self.pose, body_delta)
        twist = body_delta / dt
        stamp = now.to_msg()
        self._publish_array(self.delta_publisher, delta_counts)
        self._publish_odometry(stamp, twist)
        if self.publish_tf:
            self._publish_tf(stamp)

        self.previous_counts = raw_counts
        self.last_time = now
        self.last_status = {
            'state': 'ACTIVE',
            'dt': round(dt, 6),
            'raw_counts': raw_counts,
            'delta_counts': [float(value) for value in delta_counts],
            'wheel_m': [round(float(value), 6) for value in wheel_displacements],
            'body_delta': [round(float(value), 6) for value in body_delta],
            'pose': [round(float(value), 6) for value in self.pose],
            'odometry_scale': [
                round(float(value), 6)
                for value in self.kinematics.odometry_scale
            ],
        }
        self._publish_status()

    def _reset_odometry_callback(self, request, response):
        del request
        now = self.get_clock().now()
        self.pose = np.zeros(3)
        if self.latest_counts is not None:
            self.previous_counts = list(self.latest_counts)
        else:
            self.previous_counts = None
        self.last_time = now
        self.last_status = {
            'state': 'ODOMETRY_RESET',
            'raw_counts': self.latest_counts or [],
            'pose': [0.0, 0.0, 0.0],
            'odometry_scale': [
                round(float(value), 6)
                for value in self.kinematics.odometry_scale
            ],
        }
        self._publish_status()
        self._publish_odometry(now.to_msg(), np.zeros(3))
        if self.publish_tf:
            self._publish_tf(now.to_msg())
        response.success = True
        response.message = 'Measurement wheel odometry reset'
        return response

    def _save_odometry_scale_callback(self, request, response):
        del request
        robot_path = str(self.get_parameter('robot_config_file').value)
        if not robot_path:
            response.success = False
            response.message = 'robot_config_file is not set'
            return response
        try:
            path = Path(robot_path).expanduser().resolve()
            self._write_odometry_scale(path, self.kinematics.odometry_scale)
        except OSError as error:
            response.success = False
            response.message = f'Failed to save odometry scale: {error}'
            return response
        except ValueError as error:
            response.success = False
            response.message = str(error)
            return response

        self.last_status = {
            'state': 'ODOMETRY_SCALE_SAVED',
            'robot_config_file': str(path),
            'odometry_scale': [
                round(float(value), 6)
                for value in self.kinematics.odometry_scale
            ],
        }
        self._publish_status()
        response.success = True
        response.message = f'Saved odometry_scale to {path}'
        return response

    @staticmethod
    def _write_odometry_scale(path: Path, scale: np.ndarray) -> None:
        values = [float(value) for value in np.asarray(scale, dtype=float)]
        if len(values) != 3 or not np.all(np.isfinite(values)):
            raise ValueError('odometry_scale must be three finite numbers')

        text = path.read_text(encoding='utf-8')
        lines = text.splitlines(keepends=True)
        replacement = (
            '[' + ', '.join(f'{value:.8g}' for value in values) + ']'
        )
        in_measurement_wheels = False
        measurement_indent = 0
        replaced = False

        for index, line in enumerate(lines):
            line_without_newline = line.rstrip('\r\n')
            newline = line[len(line_without_newline):]
            stripped = line_without_newline.lstrip(' ')
            indent = len(line_without_newline) - len(stripped)
            if stripped.startswith('measurement_wheels:'):
                in_measurement_wheels = True
                measurement_indent = indent
                continue
            if (
                in_measurement_wheels
                and stripped
                and not stripped.startswith('#')
                and indent <= measurement_indent
            ):
                break
            if in_measurement_wheels and stripped.startswith('odometry_scale:'):
                comment = ''
                if '#' in line_without_newline:
                    comment = '  ' + line_without_newline.split('#', 1)[1].rstrip()
                    comment = comment.replace('  ', '  #', 1)
                lines[index] = (
                    ' ' * indent
                    + f'odometry_scale: {replacement}'
                    + comment
                    + newline
                )
                replaced = True
                break

        if not replaced:
            raise ValueError(
                f'Could not find measurement_wheels.odometry_scale in {path}'
            )
        temporary = path.with_name(path.name + '.tmp')
        temporary.write_text(''.join(lines), encoding='utf-8')
        temporary.replace(path)

    @staticmethod
    def _publish_array(publisher, values) -> None:
        message = Float64MultiArray()
        message.data = [float(value) for value in values]
        publisher.publish(message)

    def _publish_odometry(self, stamp, twist: np.ndarray) -> None:
        message = Odometry()
        message.header.stamp = stamp
        message.header.frame_id = self.odom_frame
        message.child_frame_id = self.base_frame
        message.pose.pose.position.x = float(self.pose[0])
        message.pose.pose.position.y = float(self.pose[1])
        set_yaw(message.pose.pose.orientation, float(self.pose[2]))
        message.twist.twist.linear.x = float(twist[0])
        message.twist.twist.linear.y = float(twist[1])
        message.twist.twist.angular.z = float(twist[2])
        message.pose.covariance = self._pose_covariance()
        message.twist.covariance = self._twist_covariance()
        self.odom_publisher.publish(message)
        if self.standard_odom_publisher is not None:
            self.standard_odom_publisher.publish(message)

    def _publish_tf(self, stamp) -> None:
        transform = TransformStamped()
        transform.header.stamp = stamp
        transform.header.frame_id = self.odom_frame
        transform.child_frame_id = self.base_frame
        transform.transform.translation.x = float(self.pose[0])
        transform.transform.translation.y = float(self.pose[1])
        set_yaw(transform.transform.rotation, float(self.pose[2]))
        self.tf_broadcaster.sendTransform(transform)

    def _pose_covariance(self):
        covariance = np.zeros((6, 6))
        covariance[0, 0] = float(self.get_parameter('pose_covariance_xy').value)
        covariance[1, 1] = float(self.get_parameter('pose_covariance_xy').value)
        covariance[2, 2] = 1.0e3
        covariance[3, 3] = 1.0e3
        covariance[4, 4] = 1.0e3
        covariance[5, 5] = float(self.get_parameter('pose_covariance_yaw').value)
        return covariance.reshape(-1).tolist()

    def _twist_covariance(self):
        covariance = np.zeros((6, 6))
        covariance[0, 0] = float(
            self.get_parameter('twist_covariance_linear').value
        )
        covariance[1, 1] = float(
            self.get_parameter('twist_covariance_linear').value
        )
        covariance[2, 2] = 1.0e3
        covariance[3, 3] = 1.0e3
        covariance[4, 4] = 1.0e3
        covariance[5, 5] = float(
            self.get_parameter('twist_covariance_angular').value
        )
        return covariance.reshape(-1).tolist()

    def _publish_status(self) -> None:
        # 100 HzポーリングごとのJSON整形+送信はOrin NanoのPythonノードでは
        # 無視できないCPUを使い、オドメトリ配信のジッタ源になる。状態遷移は
        # 即時に、同一状態の連続更新は10 Hzへ間引く(GUI表示には十分)。
        now_ns = self.get_clock().now().nanoseconds
        state = str(self.last_status.get('state', ''))
        if (
            state == self._last_status_state
            and now_ns - self._last_status_publish_ns < 100_000_000
        ):
            return
        self._last_status_state = state
        self._last_status_publish_ns = now_ns
        message = String()
        message.data = json.dumps(self.last_status)
        self.status_publisher.publish(message)

    def close(self) -> None:
        if self.counter is not None:
            self.counter.close()
        self.power_switch.close()


def main(args=None) -> None:
    rclpy.init(args=args)
    node: Optional[MeasurementWheelNode] = None
    try:
        node = MeasurementWheelNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except (
        ConfigError,
        ContecCounterError,
        PowerControlError,
        ValueError,
    ) as error:
        rclpy.logging.get_logger('measurement_wheel').fatal(str(error))
        raise
    finally:
        if node is not None:
            node.close()
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
