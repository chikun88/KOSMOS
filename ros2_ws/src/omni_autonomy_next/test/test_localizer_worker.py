"""Real ROS executor/worker contracts; existing geometry tests cover ICP math."""
import math
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

rclpy = pytest.importorskip('rclpy')
from geometry_msgs.msg import Point32, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from rcl_interfaces.msg import Parameter as ParameterMessage, ParameterValue, SetParametersResult
from rcl_interfaces.srv import SetParametersAtomically
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool
from omni_autonomy_interfaces.srv import SetMotionContext

from omni_autonomy_next import wall_localizer_node as localization
from omni_autonomy_next.competition_footprint import footprint_digest
from omni_autonomy_next.geometry import OptimizationCancelled, OptimizationResult, optimize_pose

CONFIG = Path(__file__).resolve().parents[1] / 'config'


def wait_for(predicate, timeout=2.):
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        threading.Event().wait(.005)
    assert predicate()


def fit(before):
    return OptimizationResult(np.asarray(before) + [.02, 0., 0.], np.eye(3) * .01,
                              True, 100, .01, 1, 0., 0.)


def wheel(node, x=0.):
    msg = Odometry()
    msg.header.stamp = node.get_clock().now().to_msg()
    msg.header.frame_id, msg.child_frame_id = node.odom_frame, node.base_frame
    msg.pose.pose.position.x = x
    msg.pose.pose.orientation.w = 1.
    return msg


def scan(node, lidar):
    msg = LaserScan()
    msg.header.stamp = node.get_clock().now().to_msg()
    msg.header.frame_id = lidar['frame_id']
    msg.angle_min, msg.angle_increment = -math.pi, 2. * math.pi / 72
    msg.angle_max = msg.angle_min + 71 * msg.angle_increment
    msg.range_min, msg.range_max = .15, 12.
    msg.ranges = [2.] * 72
    return msg


def seed(node, x=0.):
    node._wheel_odom_callback(wheel(node, x))
    for lidar in node.robot['lidars']:
        node._scan_callback(scan(node, lidar), lidar)


@pytest.fixture
def harness(monkeypatch):
    # All solves in these scheduling tests are deliberately controlled. Avoid
    # spending time building a lookup whose queries the injected solve never uses.
    monkeypatch.setattr(localization, 'WallLookupGrid',
                        lambda *a, **k: SimpleNamespace(cells_x=0, cells_y=0))
    rclpy.init(args=['--ros-args',
        '-p', f'field_config_file:={CONFIG / "field_planning.yaml"}',
        '-p', f'robot_config_file:={CONFIG / "robot.yaml"}',
        '-p', 'use_wheel_odometry:=true'])
    node = localization.WallLocalizer()
    source = Node('localizer_worker_test')
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    executor.add_node(source)
    thread = None
    errors = []
    def start():
        nonlocal thread
        def spin():
            try:
                executor.spin()
            except ExternalShutdownException:
                pass
            except Exception as error:
                if rclpy.ok():
                    errors.append(error)
        thread = threading.Thread(target=spin)
        thread.start()
    yield node, source, start, errors
    executor.shutdown()
    if thread is not None:
        thread.join(2.)
        assert not thread.is_alive()
    node.destroy_node()
    source.destroy_node()
    rclpy.try_shutdown()


def test_slow_icp_keeps_actual_ros_wheel_scan_and_pose_callbacks_responsive(harness, monkeypatch):
    node, source, start, errors = harness
    entered, release = threading.Event(), threading.Event()
    def slow(self, points, before):
        assert not points.flags.writeable and not before.flags.writeable
        with pytest.raises(TypeError):
            self.params['max_iterations'] = 100
        entered.set()
        assert release.wait(2.)
        return fit(before)
    monkeypatch.setattr(localization.SnapshotAligner, '_optimize', slow)
    pub = source.create_publisher(Odometry, '/wheel/odometry', qos_profile_sensor_data)
    scan_pubs = {x['name']: source.create_publisher(LaserScan, x['topic'], qos_profile_sensor_data)
                 for x in node.robot['lidars']}
    poses = []
    source.create_subscription(PoseWithCovarianceStamped, '/localization/pose',
                              lambda msg: poses.append(time.monotonic()), 10)
    seed(node)
    start()
    assert entered.wait(2.)
    first_wheel, first_scans = node.wheel_generation, dict(node.scan_generations)
    try:
        began = time.monotonic()
        while time.monotonic() - began < .18:
            pub.publish(wheel(node, .01))
            for lidar in node.robot['lidars']:
                scan_pubs[lidar['name']].publish(scan(node, lidar))
            time.sleep(.02)
        assert node.wheel_generation >= first_wheel + 5
        assert all(node.scan_generations[name] >= value + 3
                   for name, value in first_scans.items())
        assert len([t for t in poses if t >= began]) >= 5
        assert not errors
    finally:
        release.set()


def test_initialpose_during_solve_rejects_old_pose_and_health(harness, monkeypatch):
    node, source, start, errors = harness
    entered, release, next_entered, next_release = [threading.Event() for _ in range(4)]
    calls = []
    def slow(self, points, before):
        calls.append(before.copy())
        if len(calls) == 1:
            entered.set(); assert release.wait(2.)
        else:
            next_entered.set(); assert next_release.wait(2.)
        return fit(before)
    monkeypatch.setattr(localization.SnapshotAligner, '_optimize', slow)
    seed(node); start(); assert entered.wait(2.)
    reset = PoseWithCovarianceStamped()
    reset.header.frame_id = node.map_frame
    reset.pose.pose.position.x, reset.pose.pose.position.y = 3., 4.
    reset.pose.pose.orientation.w = 1.
    reset.pose.covariance = np.eye(6).reshape(-1).tolist()
    pub = source.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)
    try:
        until = time.monotonic() + 1.
        while node.pose_reset_generation == 0 and time.monotonic() < until:
            pub.publish(reset); time.sleep(.02)
        assert node.pose_reset_generation > 0
        release.set(); assert next_entered.wait(2.)
        np.testing.assert_allclose(node.pose, [3., 4., 0.])
        assert node.last_result['accepted'] is False
        assert node.last_tracking_accept_time == -math.inf
        np.testing.assert_allclose(calls[1], [3., 4., 0.])
        assert not errors
    finally:
        release.set(); next_release.set()


def test_atomic_parameter_commit_during_solve_uses_frozen_configuration(harness, monkeypatch):
    node, source, start, errors = harness
    entered, release, next_entered, next_release = [threading.Event() for _ in range(4)]
    configured = []
    def slow(self, points, before):
        configured.append(self.params['max_accepted_rmse'])
        if len(configured) == 1:
            entered.set(); assert release.wait(2.)
            assert self.params['max_accepted_rmse'] == .18
        else:
            next_entered.set(); assert next_release.wait(2.)
        return fit(before)
    monkeypatch.setattr(localization.SnapshotAligner, '_optimize', slow)
    seed(node); start(); assert entered.wait(2.)
    client = source.create_client(SetParametersAtomically, '/wall_localizer/set_parameters_atomically')
    assert client.wait_for_service(timeout_sec=1.)
    request = SetParametersAtomically.Request(parameters=[ParameterMessage(
        name='max_accepted_rmse', value=ParameterValue(type=3, double_value=.001))])
    try:
        future = client.call_async(request)
        wait_for(future.done)
        assert future.result().result.successful
        assert node.params['max_accepted_rmse'] == .001
        release.set(); assert next_entered.wait(2.)
        assert configured == [.18, .001]
        assert node.last_result['accepted'] is False
        assert node.last_tracking_accept_time == -math.inf
        assert not errors
    finally:
        release.set(); next_release.set()


def test_later_parameter_validator_rejection_preserves_cache_and_generation(harness):
    node, _, _, _ = harness
    before, generation = node.params.copy(), node._solution_generation
    node.add_on_set_parameters_callback(lambda parameters: SetParametersResult(
        successful=False, reason='another validator rejected the transaction'))
    from rclpy.parameter import Parameter
    result = node.set_parameters_atomically([Parameter('max_iterations', value=24)])
    assert not result.successful
    assert node.params == before and node._solution_generation == generation


def test_motion_context_during_solve_discards_old_mask_result(harness, monkeypatch):
    node, source, start, errors = harness
    entered, release = threading.Event(), threading.Event()
    def slow(self, points, before):
        entered.set(); assert release.wait(2.)
        return fit(before)
    monkeypatch.setattr(localization.SnapshotAligner, '_optimize', slow)
    seed(node); original = node.pose.copy(); start(); assert entered.wait(2.)
    client = source.create_client(SetMotionContext, '/wall_localizer/set_motion_context')
    assert client.wait_for_service(timeout_sec=1.)
    request = SetMotionContext.Request(revision=1, execution_id=1, footprint_profile='test')
    request.footprint.header.frame_id = node.base_frame
    # Exactly representable Point32 coordinates keep the wire digest identical.
    polygon = [[-.5, -.5], [.5, -.5], [.5, .5], [-.5, .5]]
    request.footprint.polygon.points = [Point32(x=x, y=y) for x, y in polygon]
    request.footprint_digest = footprint_digest(polygon)
    try:
        future = client.call_async(request); wait_for(future.done)
        assert future.result().success
        release.set(); wait_for(lambda: not node._solve_busy)
        np.testing.assert_allclose(node.pose, original)
        assert node.latest_scans == {}
        assert node.last_result['accepted'] is False
        assert node.last_tracking_accept_time == -math.inf
        assert not errors
    finally:
        release.set()


def test_expired_scan_completion_never_refreshes_tracking_health(harness, monkeypatch):
    node, _, _, _ = harness
    entered, release = threading.Event(), threading.Event()
    stamp = [node.get_clock().now().nanoseconds]
    monkeypatch.setattr(node, 'get_clock', lambda: SimpleNamespace(
        now=lambda: Time(nanoseconds=stamp[0])))
    def slow(self, points, before):
        entered.set(); assert release.wait(2.)
        return fit(before)
    monkeypatch.setattr(localization.SnapshotAligner, '_optimize', slow)
    seed(node); original = node.pose.copy(); node._timer_callback(); assert entered.wait(2.)
    stamp[0] += node.scan_timeout.nanoseconds + 1
    # New arrivals do not make the old points used by this solve fresh.
    seed(node)
    release.set(); wait_for(lambda: node._worker_completion is not None)
    node._commit_solve_result()
    np.testing.assert_allclose(node.pose, original)
    assert node.last_result['accepted'] is False
    assert node.last_tracking_accept_time == -math.inf
    assert node._scans_are_fresh(stamp[0])


def test_reset_during_input_preparation_cannot_relabel_old_points_as_new_epoch(harness, monkeypatch):
    node, _, _, _ = harness
    seed(node)
    recent_points = node._recent_points
    def reset_during_preparation():
        prepared = recent_points()
        reset = PoseWithCovarianceStamped()
        reset.header.frame_id = node.map_frame
        reset.pose.pose.position.x = 3.
        reset.pose.pose.orientation.w = 1.
        reset.pose.covariance = np.eye(6).reshape(-1).tolist()
        node._initial_pose_callback(reset)
        return prepared
    monkeypatch.setattr(node, '_recent_points', reset_during_preparation)
    node._timer_callback()
    assert node._worker_snapshot is None and not node._solve_busy
    assert node._solve_requested
    np.testing.assert_allclose(node.pose, [3., 0., 0.])
    assert node.last_tracking_accept_time == -math.inf


def test_worker_failure_propagates_on_control_callback(harness, monkeypatch):
    node, _, _, _ = harness
    def broken(*args):
        raise ValueError('injected ICP failure')
    monkeypatch.setattr(localization.SnapshotAligner, '_optimize', broken)
    seed(node); node._timer_callback(); node._solve_worker.join(2.)
    with pytest.raises(RuntimeError, match='ICP worker failed') as caught:
        node._commit_solve_result()
    assert isinstance(caught.value.__cause__, ValueError)


def test_motion_during_recovery_rejects_stationary_snapshot_relock(harness, monkeypatch):
    node, _, _, _ = harness
    entered, release = threading.Event(), threading.Event()
    calls = []
    def recovered(self, points, before):
        calls.append(before.copy())
        result = fit(before)
        if len(calls) == 1:
            result.rmse = 1.  # Force the existing multi-start recovery path.
        elif len(calls) == 2:
            assert self.stationary
            entered.set(); assert release.wait(2.)
        return result
    monkeypatch.setattr(localization.SnapshotAligner, '_optimize', recovered)
    seed(node)
    node.rejected_streak = int(node.params['recovery_after_rejections'])
    original_covariance = node.covariance.copy()
    node._timer_callback(); assert entered.wait(2.)
    try:
        # Localizer velocity is derived from acquisition-stamped wheel poses,
        # rather than trusting the publisher's optional twist estimate.
        moving = wheel(node, .01)
        node._wheel_odom_callback(moving)
        assert not node._wheel_stationary()
        wheel_propagated_pose = node.pose.copy()
        release.set(); wait_for(lambda: node._worker_completion is not None)
        assert node._worker_completion[1]['recovered']
        node._commit_solve_result()
        np.testing.assert_allclose(node.pose, wheel_propagated_pose)
        np.testing.assert_allclose(node.covariance, original_covariance)
        assert node.last_result['state'] == 'REJECTED'
        assert node.last_result['accepted'] is False
        assert node.last_tracking_accept_time == -math.inf
    finally:
        release.set()


def test_busy_timer_requests_coalesce_and_acquire_latest_state_after_commit(harness, monkeypatch):
    node, _, _, _ = harness
    entered, release, next_entered, next_release = [threading.Event() for _ in range(4)]
    starts = []
    def slow(self, points, before):
        starts.append(before.copy())
        if len(starts) == 1:
            entered.set(); assert release.wait(2.)
        else:
            next_entered.set(); assert next_release.wait(2.)
        return fit(before)
    monkeypatch.setattr(localization.SnapshotAligner, '_optimize', slow)
    seed(node); node._timer_callback(); assert entered.wait(2.)
    try:
        for index in range(20):
            seed(node, (index + 1) * .01)
            node._timer_callback()
        assert len(starts) == 1
        newest = tuple((name, node.scan_generations[name]) for name in node.scan_generations)
        release.set(); wait_for(lambda: node._worker_completion is not None)
        node._commit_solve_result(); assert next_entered.wait(2.)
        assert len(starts) == 2
        # Wheel movement received during the first solve must survive commit.
        assert node.pose[0] > 1.19
        np.testing.assert_allclose(starts[1], node.pose)
        next_release.set(); wait_for(lambda: node._worker_completion is not None)
        node._commit_solve_result()
        assert node.last_processed_scan_generations == newest
        assert not node._solve_busy and len(starts) == 2
    finally:
        release.set(); next_release.set()


def test_fast_health_expires_sources_even_while_worker_is_busy(harness, monkeypatch):
    node, _, _, _ = harness
    seed(node)
    now = node.get_clock().now().nanoseconds + node.scan_timeout.nanoseconds + 1
    monkeypatch.setattr(node, 'get_clock', lambda: SimpleNamespace(
        now=lambda: Time(nanoseconds=now)))
    node._solve_busy = True
    node.last_result = {'state': 'TRACKING'}
    node.last_tracking_accept_time = time.monotonic()
    published = []
    node.tracking_publisher = SimpleNamespace(publish=lambda message: published.append(message.data))
    node._fast_publish_callback()
    assert published == [False]


def test_context_shutdown_during_solve_does_not_raise_in_worker(harness, monkeypatch):
    node, _, _, _ = harness
    entered, release = threading.Event(), threading.Event()
    def slow(self, points, before):
        entered.set(); assert release.wait(2.)
        return fit(before)
    monkeypatch.setattr(localization.SnapshotAligner, '_optimize', slow)
    seed(node); node._timer_callback(); assert entered.wait(2.)
    rclpy.shutdown()
    release.set(); wait_for(lambda: node._worker_completion is not None)
    node.destroy_node()
    assert not node._solve_worker.is_alive()
    assert node._worker_error is None


def test_destroy_cancels_and_joins_worker_before_destroying_ros_handles(harness, monkeypatch):
    node, _, _, _ = harness
    entered = threading.Event()
    def cancellable(self, points, before):
        entered.set()
        while not self.cancelled():
            threading.Event().wait(.005)
        raise OptimizationCancelled()
    monkeypatch.setattr(localization.SnapshotAligner, '_optimize', cancellable)
    seed(node); node._timer_callback(); assert entered.wait(2.)
    began = time.monotonic(); node.destroy_node()
    assert time.monotonic() - began < .5
    assert not node._solve_worker.is_alive()
    assert node._worker_error is None


def test_geometry_cancellation_is_checked_before_iteration():
    with pytest.raises(OptimizationCancelled):
        optimize_pose(np.ones((4, 2)), np.array([[0., 0., 2., 0.]]), np.zeros(3),
                      min_correspondences=3, cancelled=lambda: True)
