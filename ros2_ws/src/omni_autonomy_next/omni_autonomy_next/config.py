import hashlib
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import yaml

from .scan_self_reflections import reflection_windows


class ConfigError(ValueError):
    """Raised when a field or robot configuration is invalid."""


def _require_finite(name: str, values) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ConfigError(f'{name} must contain only finite values')
    return array


def _require_positive(name: str, value) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ConfigError(f'{name} must be finite and positive')
    return result


def _require_bool(name: str, value) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f'{name} must be a YAML boolean')
    return value


def _read_yaml(path: str) -> Dict[str, Any]:
    config_path = Path(path).expanduser()
    if not config_path.is_file():
        raise ConfigError(f'Configuration file does not exist: {config_path}')
    try:
        with config_path.open('r', encoding='utf-8') as stream:
            data = yaml.safe_load(stream)
    except (OSError, yaml.YAMLError) as error:
        raise ConfigError(
            f'Cannot read configuration {config_path}: {error}'
        ) from error
    if not isinstance(data, dict):
        raise ConfigError(f'Configuration must be a YAML mapping: {config_path}')
    return data


def load_field(path: str) -> Dict[str, Any]:
    data = _read_yaml(path)
    field = data.get('field')
    if not isinstance(field, dict):
        raise ConfigError('field.yaml must contain a "field" mapping')

    raw_walls = field.get('walls', [])
    if not isinstance(raw_walls, list) or len(raw_walls) < 2:
        raise ConfigError('At least two wall segments are required')

    walls: List[List[float]] = []
    names: List[str] = []
    for index, wall in enumerate(raw_walls):
        if not isinstance(wall, dict):
            raise ConfigError(f'Wall {index} must be a mapping')
        start = wall.get('start')
        end = wall.get('end')
        if (
            not isinstance(start, list)
            or not isinstance(end, list)
            or len(start) != 2
            or len(end) != 2
        ):
            raise ConfigError(f'Wall {index} needs two-element start and end values')
        segment = [float(start[0]), float(start[1]), float(end[0]), float(end[1])]
        _require_finite(f'Wall {index}', segment)
        if np.hypot(segment[2] - segment[0], segment[3] - segment[1]) < 1.0e-6:
            raise ConfigError(f'Wall {index} has zero length')
        walls.append(segment)
        names.append(str(wall.get('name', f'wall_{index}')))

    source_stl = str(field.get('source_stl', ''))
    expected_digest = str(field.get('source_stl_sha256', ''))
    if bool(source_stl) != bool(expected_digest):
        raise ConfigError(
            'field.source_stl and source_stl_sha256 must be provided together'
        )
    if source_stl and expected_digest:
        source_path = Path(source_stl).expanduser()
        if not source_path.is_file():
            raise ConfigError(f'Field source STL does not exist: {source_path}')
        actual_digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
        if actual_digest != expected_digest:
            raise ConfigError(
                f'Field CAD changed: expected SHA-256 {expected_digest}, '
                f'got {actual_digest}. Regenerate field_cad.yaml.'
            )

    source_layout = str(field.get('source_layout', ''))
    expected_layout_digest = str(field.get('source_layout_sha256', ''))
    if bool(source_layout) != bool(expected_layout_digest):
        raise ConfigError(
            'field.source_layout and source_layout_sha256 must be provided together'
        )
    if source_layout and expected_layout_digest:
        layout_path = Path(source_layout).expanduser()
        if not layout_path.is_file():
            raise ConfigError(f'Field layout correction does not exist: {layout_path}')
        actual_layout_digest = hashlib.sha256(layout_path.read_bytes()).hexdigest()
        if actual_layout_digest != expected_layout_digest:
            raise ConfigError(
                f'Field layout correction changed: expected SHA-256 '
                f'{expected_layout_digest}, got {actual_layout_digest}. '
                'Regenerate the field YAML files.'
            )

    frame_id = str(field.get('frame_id', 'map')).strip()
    if not frame_id:
        raise ConfigError('field.frame_id must not be empty')
    source_slice_z_mm = float(field.get('source_slice_z_mm', 0.0))
    if not np.isfinite(source_slice_z_mm) or source_slice_z_mm < 0.0:
        raise ConfigError('field.source_slice_z_mm must be finite and non-negative')

    return {
        'frame_id': frame_id,
        'walls': np.asarray(walls, dtype=float),
        'wall_names': names,
        'source_stl': source_stl,
        'source_stl_sha256': expected_digest,
        'source_layout': source_layout,
        'source_layout_sha256': expected_layout_digest,
        'source_slice_z_mm': source_slice_z_mm,
    }


def load_robot(path: str) -> Dict[str, Any]:
    data = _read_yaml(path)
    robot = data.get('robot')
    if not isinstance(robot, dict):
        raise ConfigError('robot.yaml must contain a "robot" mapping')

    footprint = np.asarray(robot.get('footprint', []), dtype=float)
    if footprint.ndim != 2 or footprint.shape[0] < 3 or footprint.shape[1] != 2:
        raise ConfigError('robot.footprint must contain at least three [x, y] points')
    _require_finite('robot.footprint', footprint)
    footprint_twice_area = abs(float(sum(
        footprint[index, 0] * footprint[(index + 1) % len(footprint), 1]
        - footprint[(index + 1) % len(footprint), 0] * footprint[index, 1]
        for index in range(len(footprint))
    )))
    if footprint_twice_area <= 1.0e-9:
        raise ConfigError('robot.footprint must have non-zero area')

    raw_lidars = robot.get('lidars', [])
    if not isinstance(raw_lidars, list) or len(raw_lidars) != 2:
        raise ConfigError('Exactly two entries are required in robot.lidars')

    lidars = []
    used_names = set()
    used_topics = set()
    used_frames = set()
    used_serial_ports = set()
    used_device_serials = set()
    for index, lidar in enumerate(raw_lidars):
        if not isinstance(lidar, dict):
            raise ConfigError(f'LiDAR {index} must be a mapping')
        name = str(lidar.get('name', f'lidar_{index}')).strip()
        if not name:
            raise ConfigError(f'LiDAR {index} name must not be empty')
        if name in used_names:
            raise ConfigError(f'Duplicate LiDAR name: {name}')
        used_names.add(name)

        pose = lidar.get('pose', {})
        if not isinstance(pose, dict):
            raise ConfigError(f'LiDAR {name} pose must be a mapping')
        topic = str(lidar.get('topic', f'/scan_{name}')).strip()
        frame_id = str(lidar.get('frame_id', name)).strip()
        serial_port = str(lidar.get('serial_port', f'/dev/{name}')).strip()
        for value, used, label in (
            (topic, used_topics, 'topic'),
            (frame_id, used_frames, 'frame_id'),
            (serial_port, used_serial_ports, 'serial_port'),
        ):
            if not value:
                raise ConfigError(f'LiDAR {name} {label} must not be empty')
            if value in used:
                raise ConfigError(f'Duplicate LiDAR {label}: {value}')
            used.add(value)
        pose_values = {
            key: float(pose.get(key, 0.0))
            for key in ('x', 'y', 'z', 'roll', 'pitch', 'yaw')
        }
        _require_finite(f'LiDAR {name} pose', tuple(pose_values.values()))
        if pose_values['z'] < 0.0:
            raise ConfigError(f'LiDAR {name} pose.z must be non-negative')
        # The two CP2102 bridges share the USB serial string 0001, so the LiDAR's
        # own GET_DEVICE_INFO serial number is the only stable discriminator.
        device_serial = str(lidar.get('device_serial', '')).strip().upper()
        if device_serial:
            if not all(character in '0123456789ABCDEF' for character in device_serial):
                raise ConfigError(
                    f'LiDAR {name} device_serial must be hexadecimal: {device_serial}'
                )
            if device_serial in used_device_serials:
                raise ConfigError(f'Duplicate LiDAR device_serial: {device_serial}')
            used_device_serials.add(device_serial)
        serial_baudrate = int(lidar.get('serial_baudrate', 115200))
        if serial_baudrate <= 0:
            raise ConfigError(f'LiDAR {name} serial_baudrate must be positive')
        try:
            self_reflections = reflection_windows(lidar.get('self_reflection_windows', []))
        except ValueError as error:
            raise ConfigError(f'LiDAR {name}: {error}') from error
        lidars.append({
            'name': name,
            'topic': topic,
            'frame_id': frame_id,
            'serial_port': serial_port,
            'device_serial': device_serial,
            'serial_baudrate': serial_baudrate,
            'scan_mode': str(lidar.get('scan_mode', 'Boost')),
            'inverted': _require_bool(
                f'LiDAR {name} inverted', lidar.get('inverted', False)
            ),
            'angle_compensate': _require_bool(
                f'LiDAR {name} angle_compensate',
                lidar.get('angle_compensate', True),
            ),
            'pose': pose_values,
            'self_reflection_windows': self_reflections,
        })

    cad_model_file = str(robot.get('cad_model_file', ''))
    cad_model_sha256 = str(robot.get('cad_model_sha256', ''))
    if bool(cad_model_file) != bool(cad_model_sha256):
        raise ConfigError(
            'robot.cad_model_file and cad_model_sha256 must be provided together'
        )
    if cad_model_file and cad_model_sha256:
        cad_path = Path(cad_model_file).expanduser()
        if not cad_path.is_file():
            raise ConfigError(f'Robot CAD STL does not exist: {cad_path}')
        actual_digest = hashlib.sha256(cad_path.read_bytes()).hexdigest()
        if actual_digest != cad_model_sha256:
            raise ConfigError(
                f'Robot CAD changed: expected SHA-256 {cad_model_sha256}, '
                f'got {actual_digest}. Update robot.yaml.'
            )

    drivetrain = robot.get('drivetrain', {})
    if not isinstance(drivetrain, dict):
        raise ConfigError('robot.drivetrain must be a mapping')
    drivetrain_type = str(drivetrain.get('type', 'omni4'))
    wheel_order = [str(value) for value in drivetrain.get(
        'wheel_order',
        ['front_left', 'front_right', 'rear_left', 'rear_right'],
    )]
    wheel_positions = np.asarray(
        drivetrain.get('wheel_positions', []), dtype=float
    )
    wheel_angles = np.asarray(
        drivetrain.get('wheel_drive_angles_deg', []), dtype=float
    )
    wheel_signs = np.asarray(
        drivetrain.get('wheel_signs', []), dtype=float
    )
    if drivetrain_type == 'omni4':
        if wheel_positions.shape != (4, 2):
            raise ConfigError(
                'robot.drivetrain.wheel_positions must contain four [x, y] values'
            )
        if len(wheel_order) != 4 or wheel_angles.shape != (4,):
            raise ConfigError(
                'omni4 wheel_order and wheel_drive_angles_deg need four values'
            )
        if wheel_signs.shape != (4,):
            raise ConfigError('omni4 wheel_signs needs four values')
        if len(set(wheel_order)) != 4 or any(not value for value in wheel_order):
            raise ConfigError('omni4 wheel_order values must be non-empty and unique')
        _require_finite('robot.drivetrain.wheel_positions', wheel_positions)
        _require_finite('robot.drivetrain.wheel_drive_angles_deg', wheel_angles)
        _require_finite('robot.drivetrain.wheel_signs', wheel_signs)
        if not np.all(np.isin(wheel_signs, (-1.0, 1.0))):
            raise ConfigError('robot.drivetrain.wheel_signs values must be -1 or 1')

    drivetrain_wheel_radius = _require_positive(
        'robot.drivetrain.wheel_radius', drivetrain.get('wheel_radius', 0.05)
    )
    drivetrain_wheel_width = _require_positive(
        'robot.drivetrain.wheel_width', drivetrain.get('wheel_width', 0.05)
    )
    drivetrain_max_wheel_speed = _require_positive(
        'robot.drivetrain.max_wheel_speed',
        drivetrain.get('max_wheel_speed', 25.0),
    )
    # The tracker consumes this normalized configuration, whereas the guard
    # reads the YAML directly. Dropping profile overrides here silently kept
    # sprint planning on the balanced wheel budget despite its 4 m/s profile.
    raw_profile_wheel_speeds = drivetrain.get('profile_max_wheel_speeds', {})
    if not isinstance(raw_profile_wheel_speeds, dict):
        raise ConfigError('robot.drivetrain.profile_max_wheel_speeds must be a mapping')
    profile_wheel_speeds = {}
    for name, value in raw_profile_wheel_speeds.items():
        if not isinstance(name, str) or not name.strip():
            raise ConfigError('wheel speed profile names must be nonempty strings')
        profile_wheel_speeds[name] = _require_positive(
            'robot.drivetrain.profile_max_wheel_speeds.' + name, value)
    raw_profile_accelerations = drivetrain.get('profile_linear_accelerations', {})
    if not isinstance(raw_profile_accelerations, dict):
        raise ConfigError('robot.drivetrain.profile_linear_accelerations must be a mapping')
    profile_accelerations = {}
    for name, value in raw_profile_accelerations.items():
        if not isinstance(name, str) or not name.strip():
            raise ConfigError('acceleration profile names must be nonempty strings')
        profile_accelerations[name] = _require_positive(
            'robot.drivetrain.profile_linear_accelerations.' + name, value)

    measurement = robot.get('measurement_wheels', {})
    if measurement is None:
        measurement = {}
    if not isinstance(measurement, dict):
        raise ConfigError('robot.measurement_wheels must be a mapping')
    measurement_channels = [
        int(value)
        for value in measurement.get('channels', [0, 1, 2, 3])
    ]
    if len(measurement_channels) < 3:
        raise ConfigError(
            'robot.measurement_wheels.channels needs at least three values'
        )
    if len(set(measurement_channels)) != len(measurement_channels):
        raise ConfigError('robot.measurement_wheels.channels must be unique')
    if any(channel < 0 for channel in measurement_channels):
        raise ConfigError('robot.measurement_wheels.channels must be non-negative')
    measurement_positions = np.asarray(
        measurement.get('wheel_positions', wheel_positions), dtype=float
    )
    measurement_angles = np.asarray(
        measurement.get('wheel_drive_angles_deg', wheel_angles), dtype=float
    )
    if measurement_positions.shape != (len(measurement_channels), 2):
        raise ConfigError(
            'robot.measurement_wheels.wheel_positions must match channel count'
        )
    if measurement_angles.shape != (len(measurement_channels),):
        raise ConfigError(
            'robot.measurement_wheels.wheel_drive_angles_deg must match channel count'
        )
    _require_finite(
        'robot.measurement_wheels.wheel_positions', measurement_positions
    )
    _require_finite(
        'robot.measurement_wheels.wheel_drive_angles_deg', measurement_angles
    )
    measurement_count_signs = np.asarray(
        measurement.get('count_signs', [1.0] * len(measurement_channels)),
        dtype=float,
    )
    if measurement_count_signs.shape != (len(measurement_channels),):
        raise ConfigError(
            'robot.measurement_wheels.count_signs must match channel count'
        )
    _require_finite(
        'robot.measurement_wheels.count_signs', measurement_count_signs
    )
    if not np.all(np.isin(measurement_count_signs, (-1.0, 1.0))):
        raise ConfigError(
            'robot.measurement_wheels.count_signs values must be -1 or 1'
        )
    measurement_counts_per_revolution = np.asarray(
        measurement.get(
            'counts_per_revolution', [2048.0] * len(measurement_channels)
        ),
        dtype=float,
    )
    if measurement_counts_per_revolution.shape != (len(measurement_channels),):
        raise ConfigError(
            'robot.measurement_wheels.counts_per_revolution must match channel count'
        )
    if (
        not np.all(np.isfinite(measurement_counts_per_revolution))
        or np.any(measurement_counts_per_revolution <= 0.0)
    ):
        raise ConfigError(
            'robot.measurement_wheels.counts_per_revolution must be positive'
        )
    raw_meters_per_count = np.asarray(
        measurement.get('meters_per_count', []), dtype=float
    )
    if raw_meters_per_count.size == 0:
        measurement_meters_per_count = None
    else:
        if raw_meters_per_count.shape != (len(measurement_channels),):
            raise ConfigError(
                'robot.measurement_wheels.meters_per_count must match channel count'
            )
        if (
            not np.all(np.isfinite(raw_meters_per_count))
            or np.any(raw_meters_per_count <= 0.0)
        ):
            raise ConfigError(
                'robot.measurement_wheels.meters_per_count values must be positive'
            )
        measurement_meters_per_count = raw_meters_per_count
    raw_odometry_scale = measurement.get('odometry_scale', [1.0, 1.0, 1.0])
    if isinstance(raw_odometry_scale, dict):
        measurement_odometry_scale = np.asarray(
            [
                raw_odometry_scale.get('x', 1.0),
                raw_odometry_scale.get('y', 1.0),
                raw_odometry_scale.get('yaw', 1.0),
            ],
            dtype=float,
        )
    else:
        measurement_odometry_scale = np.asarray(
            raw_odometry_scale, dtype=float
        )
    if measurement_odometry_scale.shape != (3,):
        raise ConfigError(
            'robot.measurement_wheels.odometry_scale must contain x, y, yaw'
        )
    if not np.all(np.isfinite(measurement_odometry_scale)):
        raise ConfigError(
            'robot.measurement_wheels.odometry_scale values must be finite'
        )
    if np.any(measurement_odometry_scale <= 0.0):
        raise ConfigError(
            'robot.measurement_wheels.odometry_scale values must be positive'
        )
    raw_calibration = measurement.get('calibration_matrix')
    if raw_calibration is None:
        measurement_calibration = None
    else:
        measurement_calibration = np.asarray(raw_calibration, dtype=float)
        if measurement_calibration.shape != (3, len(measurement_channels)):
            raise ConfigError(
                'robot.measurement_wheels.calibration_matrix must have three '
                f'rows of {len(measurement_channels)} values '
                '([dx, dy, dyaw] per wheel channel)'
            )
        if not np.all(np.isfinite(measurement_calibration)):
            raise ConfigError(
                'robot.measurement_wheels.calibration_matrix values must be finite'
            )
    power_control = measurement.get('power_control', {})
    if power_control is None:
        power_control = {}
    if not isinstance(power_control, dict):
        raise ConfigError('robot.measurement_wheels.power_control must be a mapping')
    counter_mode = measurement.get('counter_mode', {})
    if counter_mode is None:
        counter_mode = {}
    if not isinstance(counter_mode, dict):
        raise ConfigError('robot.measurement_wheels.counter_mode must be a mapping')

    base_frame_id = str(robot.get('base_frame_id', 'base_link')).strip()
    if not base_frame_id:
        raise ConfigError('robot.base_frame_id must not be empty')
    cad_origin_mm = np.asarray(
        robot.get('cad_origin_mm', [0.0, 0.0, 0.0]), dtype=float
    )
    if cad_origin_mm.shape != (3,):
        raise ConfigError('robot.cad_origin_mm must contain x, y, z')
    _require_finite('robot.cad_origin_mm', cad_origin_mm)
    cad_to_base_yaw = float(robot.get('cad_to_base_yaw', 0.0))
    if not np.isfinite(cad_to_base_yaw):
        raise ConfigError('robot.cad_to_base_yaw must be finite')
    cad_scale = _require_positive('robot.cad_scale', robot.get('cad_scale', 0.001))
    measurement_counter_bits = int(measurement.get('counter_bits', 32))
    if not 2 <= measurement_counter_bits <= 64:
        raise ConfigError('robot.measurement_wheels.counter_bits must be 2..64')
    measurement_wheel_radius = _require_positive(
        'robot.measurement_wheels.wheel_radius',
        measurement.get('wheel_radius', drivetrain_wheel_radius),
    )
    measurement_odom_frame_id = str(
        measurement.get('odom_frame_id', 'odom')
    ).strip()
    if not measurement_odom_frame_id:
        raise ConfigError('robot.measurement_wheels.odom_frame_id must not be empty')
    power_settle_sec = float(power_control.get('settle_sec', 0.2))
    if not np.isfinite(power_settle_sec) or power_settle_sec < 0.0:
        raise ConfigError(
            'robot.measurement_wheels.power_control.settle_sec must be '
            'finite and non-negative'
        )
    power_enabled = _require_bool(
        'robot.measurement_wheels.power_control.enabled',
        power_control.get('enabled', False),
    )
    power_gpio_pin = int(power_control.get('gpio_pin', -1))
    if power_enabled and power_gpio_pin < 0:
        raise ConfigError(
            'robot.measurement_wheels.power_control.gpio_pin must be '
            'non-negative when power control is enabled'
        )
    counter_digital_filter = int(counter_mode.get('digital_filter', 0))
    if counter_digital_filter < 0:
        raise ConfigError(
            'robot.measurement_wheels.counter_mode.digital_filter must be '
            'non-negative'
        )

    return {
        'base_frame_id': base_frame_id,
        'footprint': footprint,
        'lidars': lidars,
        'cad_model_file': cad_model_file,
        'cad_model_sha256': cad_model_sha256,
        'cad_origin_mm': cad_origin_mm,
        'cad_to_base_yaw': cad_to_base_yaw,
        'cad_scale': cad_scale,
        'drivetrain': {
            'type': drivetrain_type,
            'wheel_order': wheel_order,
            'wheel_radius': drivetrain_wheel_radius,
            'wheel_width': drivetrain_wheel_width,
            'wheel_positions': wheel_positions,
            'wheel_drive_angles_deg': wheel_angles,
            'wheel_signs': wheel_signs,
            'max_wheel_speed': drivetrain_max_wheel_speed,
            'profile_max_wheel_speeds': profile_wheel_speeds,
            'profile_linear_accelerations': profile_accelerations,
        },
        'measurement_wheels': {
            'device_name': str(measurement.get('device_name', 'CNT000')),
            'channels': measurement_channels,
            'counter_bits': measurement_counter_bits,
            'count_signs': measurement_count_signs,
            'counts_per_revolution': measurement_counts_per_revolution,
            'meters_per_count': measurement_meters_per_count,
            'wheel_radius': measurement_wheel_radius,
            'wheel_positions': measurement_positions,
            'wheel_drive_angles_deg': measurement_angles,
            'odometry_scale': measurement_odometry_scale,
            'calibration_matrix': measurement_calibration,
            'odom_frame_id': measurement_odom_frame_id,
            'publish_tf': _require_bool(
                'robot.measurement_wheels.publish_tf',
                measurement.get('publish_tf', True),
            ),
            'counter_mode': {
                'signal_type': counter_mode.get('signal_type', 'isolate'),
                'count_direction': counter_mode.get('count_direction', 'up'),
                'operation_phase': counter_mode.get(
                    'operation_phase', '2phase'
                ),
                'multiplier': counter_mode.get('multiplier', 'x1'),
                'sync_clear': counter_mode.get('sync_clear', 'async'),
                'z_phase': counter_mode.get('z_phase', 'not_use'),
                'z_logic': counter_mode.get('z_logic', 'positive'),
                'digital_filter': counter_digital_filter,
            },
            'power_control': {
                'enabled': power_enabled,
                'gpio_pin': power_gpio_pin,
                'gpio_mode': str(power_control.get('gpio_mode', 'BOARD')),
                'gpio_backend': str(power_control.get('gpio_backend', 'auto')),
                'active_high': _require_bool(
                    'robot.measurement_wheels.power_control.active_high',
                    power_control.get('active_high', True),
                ),
                'settle_sec': power_settle_sec,
                'off_on_shutdown': _require_bool(
                    'robot.measurement_wheels.power_control.off_on_shutdown',
                    power_control.get('off_on_shutdown', True),
                ),
            },
        },
    }


def calibrated_tracking_parameters(robot: dict, *, hardware: bool,
                                   motion_mode: str) -> dict:
    """Enable the fitted-loop tuning only with its matching actuator calibration."""
    if not hardware or motion_mode != 'simultaneous':
        return {}
    tuning = robot.get('calibrated_tracking', {})
    drive = robot.get('drivetrain', {})
    if not tuning.get('enabled', False):
        return {}
    for axis in ('linear', 'angular'):
        key = axis + '_command_scale'
        if drive.get(key, 1.0) != tuning.get(key):
            return {}
    result = {key: _require_positive('calibrated_tracking.' + key, tuning[key])
            for key in ('position_gain', 'yaw_gain', 'feedback_delay_sec')}
    for key in ('predictive_sprint', 'sprint_turn_everywhere'):
        if key in tuning:
            result[key] = _require_bool('calibrated_tracking.' + key, tuning[key])
    return result


def load_collision_monitor_horizon(path: str) -> float:
    """``FootprintApproach``'s ``time_before_collision`` from the Nav2 params.

    Every stage that produces a twist has to know this: the monitor projects
    the commanded velocity forward over exactly this long, holding it
    constant, and scales the whole twist by ``contact_time / horizon`` when
    the projection touches an obstacle.  A producer that asks for more
    rotation than the footprint can sweep in this time is throttled on its
    translation as well.  Reading the deployed value rather than repeating it
    keeps the two from drifting apart.
    """
    data = _read_yaml(path)
    monitor = data.get('collision_monitor', {})
    parameters = monitor.get('ros__parameters', {}) if isinstance(
        monitor, dict) else {}
    approach = parameters.get('FootprintApproach', {})
    if not isinstance(approach, dict) or 'time_before_collision' not in approach:
        raise ConfigError(
            f'collision_monitor.FootprintApproach.time_before_collision is '
            f'missing from {path}'
        )
    return _require_positive(
        'collision_monitor.FootprintApproach.time_before_collision',
        approach['time_before_collision'],
    )
