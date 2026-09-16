import math
from pathlib import Path

import numpy as np
import yaml

from omni_autonomy_next.measurement_wheel_odometry import (
    MeasurementWheelKinematics,
)
from omni_autonomy_next.motor_kinematics import twist_to_wheel_speeds


CONFIG = Path(__file__).resolve().parents[1] / 'config'


def test_measurement_wheel_calibration_is_preserved_exactly():
    robot = yaml.safe_load((CONFIG / 'robot.yaml').read_text(encoding='utf-8'))['robot']
    wheels = robot['measurement_wheels']
    assert wheels['channels'] == [0, 1, 2, 3]
    assert wheels['counts_per_revolution'] == [2048.0] * 4
    # 50.8 mm is the wheel's outside diameter, so the radius is 25.4 mm.
    # test_geometric_odometry_matches_the_on_robot_calibration below is what
    # holds this value to the hardware rather than to a copied number.
    assert wheels['wheel_radius'] == 0.0254
    assert wheels['odometry_scale'] == [1, 1, 1]
    np.testing.assert_allclose(wheels['wheel_positions'], [
        [0.2154, 0.0], [0.0, -0.2154], [-0.2154, 0.0], [0.0, 0.2154]
    ], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(wheels['calibration_matrix'], [
        [-3.58081271e-05, 7.774156689e-05, -3.51855336e-05, 7.877766435e-07],
        [0.0002772848508, -0.0002360042453, 0.0002038839814, -0.0002399216352],
        [0.0001076063798, 6.82663499e-05, 0.0001073162177, 6.804585874e-05],
    ], rtol=0.0, atol=0.0)


def test_geometric_odometry_matches_the_on_robot_calibration():
    """The declared wheel geometry must reproduce the calibrated matrix.

    ``calibration_matrix`` was identified on the robot and takes precedence at
    runtime, which is exactly why an inconsistent geometry block can hide for a
    long time: it is only used when the matrix is absent or being re-fitted.
    A ``wheel_radius`` holding the 50.8 mm outside diameter instead of the
    25.4 mm radius made the fallback path report 2.02x the real translation,
    with ``odometry_scale`` masking it on the yaw axis alone.
    """
    robot = yaml.safe_load((CONFIG / 'robot.yaml').read_text(encoding='utf-8'))['robot']
    wheels = robot['measurement_wheels']
    geometric = MeasurementWheelKinematics(
        wheel_positions=wheels['wheel_positions'],
        wheel_drive_angles=[
            math.radians(value) for value in wheels['wheel_drive_angles_deg']
        ],
        count_signs=wheels['count_signs'],
        counter_bits=wheels['counter_bits'],
        counts_per_revolution=wheels['counts_per_revolution'],
        wheel_radius=wheels['wheel_radius'],
        odometry_scale=wheels['odometry_scale'],
    )
    calibrated = np.asarray(wheels['calibration_matrix'], dtype=float)

    # Canonical excitations: pure +x, pure +y, pure +yaw in raw counts.
    for counts, axis in (
        ([0, 1000, 0, -1000], 0),
        ([1000, 0, -1000, 0], 1),
        ([1000, 1000, 1000, 1000], 2),
    ):
        counts = np.asarray(counts, dtype=float)
        _, _, geometric_delta = geometric.step(counts, np.zeros(4, dtype=int))
        calibrated_delta = calibrated @ counts
        assert geometric_delta[axis] == 0.0 or abs(
            calibrated_delta[axis] / geometric_delta[axis] - 1.0
        ) < 0.08, (
            f'axis {axis}: calibrated {calibrated_delta[axis]:.6e} vs '
            f'geometric {geometric_delta[axis]:.6e}'
        )


def test_lidar_extrinsics_and_persistent_ports_are_preserved():
    robot = yaml.safe_load((CONFIG / 'robot.yaml').read_text(encoding='utf-8'))['robot']
    front, rear = robot['lidars']
    assert front['serial_port'].endswith('usb-0:2.1:1.0-port0')
    assert rear['serial_port'].endswith('usb-0:2.3:1.0-port0')
    np.testing.assert_allclose(
        [front['pose'][key] for key in ('x', 'y', 'z', 'yaw')],
        [0.4001377285808883, -0.37442462280962147, 0.13, 0.799360797413403],
        rtol=0.0, atol=0.0,
    )
    np.testing.assert_allclose(
        [rear['pose'][key] for key in ('x', 'y', 'z', 'yaw')],
        [-0.4001377285808883, 0.37442462280962147, 0.13, -2.32128790515246],
        rtol=0.0, atol=0.0,
    )


def test_drive_geometry_and_full_footprint_are_preserved():
    robot = yaml.safe_load((CONFIG / 'robot.yaml').read_text(encoding='utf-8'))['robot']
    drive = robot['drivetrain']
    assert drive['type'] == 'omni4'
    assert drive['wheel_radius'] == 0.05
    # The nominal budget now permits 2 m/s axes with measured wire scales.
    # Its calibrated request must still fit the unchanged 8000-unit limit.
    assert drive['max_wheel_speed'] * max(
        drive['linear_command_scale'], drive['angular_command_scale']) <= 15.709120382
    # Pin the kinematic structure, not the literal list.  The inherited
    # [-45, 45, 45, -45] made a pure +vx drive all four wheels the same way and
    # a pure +wz drive them alternating, the exact opposite of the deployed Pi
    # mixer in bacon_gateway/src/move.cpp (+vx -> two each way, +wz -> all the
    # same way).  No motor order or polarity can reconcile those, so the value
    # was wrong rather than merely a different convention.
    angles = np.radians(drive['wheel_drive_angles_deg'])
    positions = np.asarray(drive['wheel_positions'], dtype=float)
    signs = np.asarray(drive['wheel_signs'], dtype=float)
    def wheels(vx, vy, wz):
        return twist_to_wheel_speeds(
            vx, vy, wz, drive_model='omni4',
            wheel_radius=drive['wheel_radius'],
            track_width=0.0, wheelbase=0.0,
            wheel_positions=positions, wheel_drive_angles=angles,
            wheel_signs=signs,
        )
    yaw_signs = np.sign(np.round(wheels(0.0, 0.0, 1.0), 9))
    assert abs(yaw_signs.sum()) == 4, 'pure yaw must drive every wheel the same way'
    for translation in ((1.0, 0.0), (0.0, 1.0)):
        translation_signs = np.sign(np.round(wheels(translation[0], translation[1], 0.0), 9))
        assert translation_signs.sum() == 0, (
            'pure translation must drive two wheels each way'
        )
    # Translation must never be starved to zero by a saturating yaw request.
    assert 0.0 < drive['translation_budget_share'] < 1.0
    assert len(robot['footprint']) == 10


def test_every_image_critical_pose_is_configured():
    poses = yaml.safe_load((CONFIG / 'field_poses.yaml').read_text(encoding='utf-8'))['poses']
    for pose_id in (0, 2, 3, 4, 5, 6, 7):
        value = poses.get(pose_id, poses.get(str(pose_id)))
        assert value is not None
        assert value['configured'] is True

