"""Measured acquisition-time coverage for high-speed scan compensation.

These tests execute the actual localizer methods without importing ROS. Their
synthetic wall and wheel measurements have known geometry and acquisition times.
"""
import ast
from bisect import bisect_left
import math
from pathlib import Path
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from omni_autonomy_next.geometry import (
    compose_pose, interpolate_pose, interpolate_poses_at, inverse_pose,
    relative_pose, transform_points_from_poses,
)
from omni_autonomy_next.scan_freshness import scan_generations_changed, timestamp_is_fresh
from omni_autonomy_next.source_freshness import message_stamp_nanoseconds


def localizer_methods():
    path = Path(__file__).resolve().parents[1] / 'omni_autonomy_next' / 'wall_localizer_node.py'
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == 'WallLocalizer')
    names = {'_wheel_pose_at', '_motion_compensated_points', '_latest_scan_point_stamp',
             '_recent_points', '_wheel_stationary', '_wheel_odom_callback'}
    cls.bases = []
    cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef)
                and node.name in names]
    scope = dict(globals(), yaw_from_quaternion=lambda q: 0.)
    exec(compile('from __future__ import annotations\n' + ast.unparse(cls), str(path), 'exec'), scope)
    return scope['WallLocalizer']


def stamp(seconds):
    return round(seconds * 1.e9)


def node_with_history(times=(10., 10.05, 10.1), speed=4.):
    node = localizer_methods()()
    node._pose_lock = threading.RLock()
    node.wheel_history_stamps = [stamp(t) for t in times]
    node.wheel_history = [(stamp(t), np.array([speed * (t - 10.), 0., 0.])) for t in times]
    node.latest_wheel_stamp_ns, node.latest_wheel_pose = node.wheel_history[-1]
    node.latest_wheel_received_ns = node.latest_wheel_stamp_ns
    node.pose = node.latest_wheel_pose + [2., 3., 0.]
    node.wheel_generation = 7
    node.pose_reset_generation = 0
    node.wheel_stamp_tolerance_ns = 250000000
    node.params = {'min_correspondences': 3, 'wheel_odom_freshness_sec': .3,
                   'settled_wheel_linear_speed': .04, 'settled_wheel_angular_speed': .1}
    node.latest_wheel_linear_speed = speed
    node.latest_wheel_angular_speed = 0.
    node.get_clock = lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=stamp(10.12)))
    node.get_logger = lambda: SimpleNamespace(warning=lambda *a, **k: None)
    node.use_wheel_odometry = node.motion_compensate_scans = True
    node.robot = {'lidars': [{'name': 'front'}]}
    node.scan_timeout = SimpleNamespace(nanoseconds=300000000)
    node.min_active_lidars = 1
    node.last_processed_scan_generations = None
    return node


def wall_scan(times, speed=4.):
    # A fixed wall at odom x=10, seen by a base travelling at constant speed.
    return {'name': 'front', 'generation': 1, 'stamp_ns': stamp(min(times)),
            'received': SimpleNamespace(nanoseconds=stamp(10.12)),
            'point_stamp_ns': np.array([stamp(t) for t in times]),
            'points': np.array([[10. - speed * (t - 10.), i * .1] for i, t in enumerate(times)])}


def test_fast_scan_uses_only_measured_wheel_coverage_instead_of_clamping():
    node = node_with_history()
    scan = wall_scan([9.99, 10., 10.05, 10.1, 10.35])
    result = node._motion_compensated_points([scan])
    assert result is not None
    assert len(result['points']) == 3
    # Covered rays all land on the same wall, referenced at the last wheel
    # sample. The old 250 ms endpoint clamp accepted a ray displaced by 1 m.
    np.testing.assert_allclose(result['points'][:, 0], 9.6, atol=1.e-12)
    np.testing.assert_allclose(result['initial_pose'], [2.4, 3., 0.])
    assert result['reference_stamp_ns'] == stamp(10.1)


def test_normal_async_scan_keeps_covered_beams_when_newest_beam_precedes_next_wheel():
    node = node_with_history(times=(10., 10.04, 10.09))
    scan = wall_scan([10., 10.02, 10.05, 10.08, 10.095])
    result = node._motion_compensated_points([scan])
    assert result is not None
    assert len(result['points']) == 4
    np.testing.assert_allclose(result['points'][:, 0], 9.64, atol=1.e-12)


def test_instantaneous_scan_waits_for_next_wheel_and_preserves_generation():
    node = node_with_history(times=(10., 10.09))
    node.latest_scans = {'front': wall_scan([10.095] * 3)}
    result, state = node._recent_points()
    assert result is None and state == 'WAITING_FOR_WHEEL_ODOMETRY'
    assert node.last_processed_scan_generations is None
    node.wheel_history.append((stamp(10.1), np.array([.4, 0., 0.])))
    node.wheel_history_stamps.append(stamp(10.1))
    node.latest_wheel_stamp_ns, node.latest_wheel_pose = node.wheel_history[-1]
    node.pose = np.array([2.4, 3., 0.])
    result, state = node._recent_points()
    assert state == 'READY'
    assert result['motion_compensated']
    assert result['scan_generations'] == (('front', 1),)
    np.testing.assert_allclose(result['points'][:, 0], 9.62, atol=1.e-12)


def test_currently_stopped_base_does_not_use_uncovered_older_moving_scan():
    node = node_with_history(times=(10.1,), speed=0.)
    node.latest_scans = {'front': wall_scan([10., 10.02, 10.05])}
    assert node._wheel_stationary()
    assert node._recent_points() == (None, 'WAITING_FOR_WHEEL_ODOMETRY')


@pytest.mark.parametrize('empty', [False, True])
def test_fresh_lidar_floor_is_separate_from_covered_point_observability(empty):
    node = node_with_history()
    node.robot['lidars'].append({'name': 'rear'})
    node.min_active_lidars = 2
    front = wall_scan([10., 10.05, 10.1])
    rear = wall_scan([10.115] * 3)
    rear['name'] = 'rear'
    if empty:
        # Existing all-inf scan contract: the sensor is alive despite having
        # no wall returns. Other LiDARs must provide min_correspondences.
        rear['points'] = np.empty((0, 2))
        rear['point_stamp_ns'] = np.empty(0, dtype=np.int64)
    node.latest_scans = {'front': front, 'rear': rear}
    result, state = node._recent_points()
    assert state == 'READY'
    assert node.active_lidar_count == 2
    assert len(result['points']) == 3
    np.testing.assert_allclose(result['points'][:, 0], 9.6, atol=1.e-12)


def test_stationary_requires_fresh_acquisition_and_receipt_times():
    node = node_with_history(speed=0.)
    assert node._wheel_stationary()
    node.latest_wheel_stamp_ns = stamp(9.7)
    assert not node._wheel_stationary()


@pytest.mark.parametrize('speed', [0., 4., math.nan])
def test_first_wheel_velocity_controls_stationary_gate_and_retries_pending_coverage(speed):
    node = node_with_history(speed=0.)
    node.odom_frame = 'odom'
    node.base_frame = 'base_link'
    node.previous_wheel_pose = node.previous_wheel_stamp_ns = None
    node.latest_wheel_stamp_ns = None
    node.latest_wheel_pose = None
    node.latest_wheel_received_ns = None
    node._append_wheel_history = lambda *a: None
    node._waiting_for_wheel_coverage = True
    retries = []
    node._start_requested_solve = lambda: retries.append(True)
    message = SimpleNamespace(header=SimpleNamespace(frame_id='odom',
        stamp=SimpleNamespace(sec=10, nanosec=100000000)), child_frame_id='base_link',
        pose=SimpleNamespace(pose=SimpleNamespace(position=SimpleNamespace(x=0., y=0.),
            orientation=None)), twist=SimpleNamespace(twist=SimpleNamespace(
                linear=SimpleNamespace(x=speed, y=0.), angular=SimpleNamespace(z=0.))))
    node._wheel_odom_callback(message)
    assert node._wheel_stationary() == (speed == 0.)
    assert bool(retries) == math.isfinite(speed)
    if math.isfinite(speed):
        assert node.latest_wheel_linear_speed == speed
