"""Real ROS regressions for initialized parameters and synthetic scan timing."""
import copy
import math
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

rclpy = pytest.importorskip('rclpy')
from rcl_interfaces.srv import GetParameters
from rclpy.parameter import Parameter
from rclpy.time import Time
from geometry_msgs.msg import Twist

from omni_autonomy_next import synthetic_scan_node as simulation
from omni_autonomy_next.wall_localizer_node import WallLocalizer

CONFIG = Path(__file__).resolve().parents[1] / 'config'


@pytest.fixture
def make_simulator():
    nodes = []
    def make(*overrides):
        rclpy.init(args=['--ros-args',
            '-p', f'field_config_file:={CONFIG / "field_planning.yaml"}',
            '-p', f'robot_config_file:={CONFIG / "robot.yaml"}',
            '-p', 'simulate_motion:=false', *overrides])
        node = simulation.SyntheticScanNode()
        nodes.append(node)
        return node
    yield make
    for node in nodes:
        node.destroy_node()
    rclpy.try_shutdown()


def test_empty_optional_obstacle_array_is_readable_by_ros_parameter_clients(make_simulator):
    node = make_simulator()
    parameter = node.get_parameter('extra_obstacles')
    assert parameter.type_ == Parameter.Type.DOUBLE_ARRAY
    assert parameter.value == []
    client = node.create_client(GetParameters, '/synthetic_scans/get_parameters')
    assert client.wait_for_service(timeout_sec=1.)
    future = client.call_async(GetParameters.Request(names=['extra_obstacles']))
    rclpy.spin_until_future_complete(node, future, timeout_sec=2.)
    result = future.result()
    assert result is not None
    assert result.values[0].type == Parameter.Type.DOUBLE_ARRAY.value
    assert list(result.values[0].double_array_value) == []
    assert node._extra_circles().shape == (0, 3)


def test_obstacle_override_survives_default_initialization_and_can_be_cleared(make_simulator):
    node = make_simulator('-p', 'extra_obstacles:=[1.0, 2.0, 0.3]')
    np.testing.assert_array_equal(node._extra_circles(), [[1., 2., .3]])
    result = node.set_parameters([Parameter('extra_obstacles', Parameter.Type.DOUBLE_ARRAY, [])])
    assert result[0].successful
    assert node._extra_circles().shape == (0, 3)


@pytest.mark.parametrize('values', [[1., 2.], [1., 2., -1.], [math.nan, 2., .3]])
def test_incomplete_or_invalid_obstacle_updates_are_rejected(make_simulator, values):
    node = make_simulator()
    result = node.set_parameters([Parameter('extra_obstacles', Parameter.Type.DOUBLE_ARRAY, values)])
    assert not result[0].successful
    assert node.get_parameter('extra_obstacles').value == []


def test_slow_front_scan_does_not_backdate_rear_or_invent_future_beams(make_simulator, monkeypatch):
    node = make_simulator()
    timestamp = [10_000_000_000]
    monkeypatch.setattr(node, 'get_clock', lambda: SimpleNamespace(
        now=lambda: Time(nanoseconds=timestamp[0])))
    poses = []
    output = []
    original_pose = node.true_pose.copy()
    def raycast(origin, directions, walls):
        poses.append(origin.copy())
        if len(poses) == 1:
            # Simulate a 400 ms front calculation while the motion callback
            # independently advances the robot. Rear acquisition occurs later.
            timestamp[0] += 400_000_000
            node.true_pose = original_pose + np.array([.4, 0., 0.])
        return np.ones(len(directions))
    monkeypatch.setattr(simulation, 'raycast_segments', raycast)
    node.scan_publishers = {lidar['name']: SimpleNamespace(
        publish=lambda message: output.append(copy.deepcopy(message)))
        for lidar in node.robot['lidars']}
    node._publish()
    stamps = [message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec
              for message in output]
    assert stamps == [10_000_000_000, 10_400_000_000]
    for message, stamp in zip(output, stamps):
        assert message.time_increment == 0.
        assert message.scan_time == pytest.approx(node.scan_period)
        point_stamps = WallLocalizer._scan_point_stamps(
            message, np.array([0, node.samples - 1]), stamp, node.samples)
        np.testing.assert_array_equal(point_stamps, [stamp, stamp])
    # Acquiring the rear at the new pose is essential: merely replacing the old
    # front stamp at publication would claim old geometry was fresh.
    yaw = float(original_pose[2])
    rotation = np.array([[math.cos(yaw), -math.sin(yaw)],
                         [math.sin(yaw), math.cos(yaw)]])
    extrinsic_offset = rotation @ (
        node.base_to_lidar['rear'][:2] - node.base_to_lidar['front'][:2])
    np.testing.assert_allclose(poses[1] - poses[0], [.4, 0.] + extrinsic_offset,
                               atol=1.e-12)


def test_nonfinite_commands_do_not_change_simulated_motion(make_simulator):
    node = make_simulator()
    command = Twist()
    command.linear.x = math.nan
    node._cmd_callback(command)
    assert node.latest_command_time is None
    np.testing.assert_array_equal(node.latest_command, np.zeros(3))


def test_moving_scan_stamp_matches_authoritative_published_odometry_and_tf(
    make_simulator, monkeypatch,
):
    node = make_simulator('-p', 'simulate_motion:=true')
    timestamp = [10_000_000_000]
    monkeypatch.setattr(node, 'get_clock', lambda: SimpleNamespace(
        now=lambda: Time(nanoseconds=timestamp[0])))
    odometry, transforms, scans = [], [], []
    node.odom_publisher = SimpleNamespace(publish=lambda message: odometry.append(copy.deepcopy(message)))
    node.standard_odom_publisher = None
    node.tf_broadcaster = SimpleNamespace(sendTransform=lambda transform: transforms.append(copy.deepcopy(transform)))
    node.latest_command = np.array([.4, 0., 0.])
    node.latest_command_time = Time(nanoseconds=timestamp[0])
    node._motion_step()
    node.scan_publishers = {lidar['name']: SimpleNamespace(
        publish=lambda message: scans.append(copy.deepcopy(message)))
        for lidar in node.robot['lidars']}
    monkeypatch.setattr(simulation, 'raycast_segments',
                        lambda origin, directions, walls: np.ones(len(directions)))
    # Capturing 30 ms later does not create a new model pose. Both scans must
    # use the already-published state, whose TF is available at that stamp.
    timestamp[0] += 30_000_000
    node._publish()
    assert len(scans) == 2
    assert all(message.header.stamp == odometry[0].header.stamp for message in scans)
    assert transforms[0].header.stamp == odometry[0].header.stamp


def test_moving_scan_startup_waits_for_first_published_motion_state(make_simulator, monkeypatch):
    node = make_simulator('-p', 'simulate_motion:=true')
    monkeypatch.setattr(simulation, 'raycast_segments',
        lambda *args: pytest.fail('must not raycast before an odometry state exists'))
    assert node._published_motion_snapshot is None
    node._publish()


def test_worker_coalesces_missed_periods_without_concurrent_raycast(make_simulator, monkeypatch):
    node = make_simulator()
    entered, release, completed = threading.Event(), threading.Event(), threading.Event()
    calls = []
    def slow_raycast(origin, directions, walls):
        calls.append(origin.copy())
        if len(calls) == 1:
            entered.set()
            assert release.wait(2.)
        if len(calls) == 4:
            completed.set()
        return np.ones(len(directions))
    monkeypatch.setattr(simulation, 'raycast_segments', slow_raycast)
    node._request_scan()
    assert entered.wait(2.)
    try:
        for _ in range(20):
            node._request_scan()
        # No second worker can acquire the rear while the front is blocked.
        assert len(calls) == 1
    finally:
        release.set()
    assert completed.wait(2.)
    # One in-flight scan plus one coalesced request; no 20-scan backlog.
    time.sleep(.05)
    assert len(calls) == 4


def test_scan_worker_fault_is_propagated_to_control_executor(make_simulator, monkeypatch):
    node = make_simulator()
    def broken_raycast(*args):
        raise ValueError('injected raycast failure')
    monkeypatch.setattr(simulation, 'raycast_segments', broken_raycast)
    node._request_scan()
    node._scan_worker.join(2.)
    assert not node._scan_worker.is_alive()
    with pytest.raises(RuntimeError, match='Synthetic scan worker failed') as caught:
        node._request_scan()
    assert isinstance(caught.value.__cause__, ValueError)


def test_destroy_joins_inflight_worker_before_destroying_publishers(make_simulator, monkeypatch):
    node = make_simulator()
    entered, release, destroyed = threading.Event(), threading.Event(), threading.Event()
    def slow_raycast(origin, directions, walls):
        entered.set()
        assert release.wait(2.)
        return np.ones(len(directions))
    monkeypatch.setattr(simulation, 'raycast_segments', slow_raycast)
    node._request_scan()
    assert entered.wait(2.)
    thread = threading.Thread(target=lambda: (node.destroy_node(), destroyed.set()))
    thread.start()
    try:
        assert not destroyed.wait(.05)
        # A stopped worker cannot queue another acquisition during teardown.
        node._request_scan()
    finally:
        release.set()
        thread.join(2.)
    assert destroyed.is_set()
    assert not node._scan_worker.is_alive()
