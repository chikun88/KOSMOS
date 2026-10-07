"""Exercise the actual ROS residual adapter with malformed/freshness inputs."""
import math
from types import SimpleNamespace

import numpy as np
import pytest
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Quaternion, Twist
from nav_msgs.msg import Path

from omni_autonomy_next import rl_policy_node as module
from omni_autonomy_next.rl_policy_node import RLPolicyNode, quaternion_yaw
from omni_autonomy_next.rl_policy import CompactRLPolicy
from omni_autonomy_next.rl_residual import CadClearanceModel
from pathlib import Path as FilePath


def _node():
    outputs = []
    node = SimpleNamespace(position=np.zeros(2), yaw=0., pose_time=1.,
                           pose_source_stamp_ns=1_000_000_000, pose_clock_ns=1_000_000_000,
                           clock_ns=1_100_000_000,
                           current_healthy=True, command_count=0,
                           path=np.zeros((2, 2)), goal_yaw=0., goal_clearance=.1,
                           policy=SimpleNamespace(reset=lambda: None),
                           _zero_message=RLPolicyNode._zero_message,
                           output_pub=SimpleNamespace(publish=outputs.append),
                           get_parameter=lambda _name: SimpleNamespace(value=.5))
    node.get_clock = lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=node.clock_ns))
    node._ready = lambda now: RLPolicyNode._ready(node, now)
    node._policy_context_ready = lambda now: RLPolicyNode._policy_context_ready(node, now)
    node._pose_is_fresh = lambda now: RLPolicyNode._pose_is_fresh(node, now)
    return node, outputs


@pytest.mark.parametrize('invalid', ['position', 'quaternion'])
def test_invalid_localization_cannot_become_healthy_using_previous_fresh_pose(monkeypatch, invalid):
    node, outputs = _node()
    monkeypatch.setattr(module.time, 'monotonic', lambda: 1.1)
    pose = PoseWithCovarianceStamped()
    pose.header.frame_id = 'map'
    pose.header.stamp.sec = 1
    pose.header.stamp.nanosec = 100_000_000
    pose.pose.pose.orientation.w = 1.
    if invalid == 'position':
        pose.pose.pose.position.x = math.nan
    else:
        pose.pose.pose.orientation.w = 0.
    RLPolicyNode._pose_cb(node, pose)
    assert node.position is None
    assert node.pose_time == -math.inf
    command = Twist()
    command.linear.x = .1
    RLPolicyNode._command_cb(node, command)
    assert not node.current_healthy
    assert outputs[-1].linear.x == 0.


def test_idle_command_cannot_advertise_health_with_stale_localization(monkeypatch):
    node, _outputs = _node()
    monkeypatch.setattr(module.time, 'monotonic', lambda: 2.)
    RLPolicyNode._command_cb(node, Twist())
    assert not node.current_healthy
    assert node.last_reason == 'STALE_LOCALIZATION'


def test_invalid_plan_clears_previously_healthy_adapter():
    node, _outputs = _node()
    RLPolicyNode._plan_cb(node, Path())
    assert node.path is None
    assert not node.current_healthy
    assert node.last_reason == 'INVALID_GLOBAL_PLAN'


def test_quaternion_yaw_normalizes_finite_scaled_orientation():
    orientation = Quaternion(z=2. * math.sin(.3), w=2. * math.cos(.3))
    assert quaternion_yaw(orientation) == pytest.approx(.6)
    assert math.isnan(quaternion_yaw(Quaternion(w=0.)))


def test_nonempty_policy_needs_fresh_matching_guard_profile():
    node, _outputs = _node()
    node.policy.overrides = {'state': 1}
    node.policy.observation_context = {'reference_profile': 'sprint'}
    node.guard_profile_time = 1.
    node.guard_profile = 'balanced'
    assert not node._ready(1.1)[0]
    node.guard_profile = 'sprint'
    assert node._ready(1.1)[0]
    node.pose_time = 2.
    assert not node._ready(2.1)[0]


def _pose(sec=1, nanosec=100_000_000):
    message = PoseWithCovarianceStamped()
    message.header.frame_id = 'map'
    message.header.stamp.sec = sec
    message.header.stamp.nanosec = nanosec
    return message


@pytest.mark.parametrize('mutate', [
    lambda pose: setattr(pose.header, 'frame_id', 'odom'),
    lambda pose: setattr(pose.header.stamp, 'sec', 0) or setattr(pose.header.stamp, 'nanosec', 0),
    lambda pose: setattr(pose.header.stamp, 'sec', -1),
    lambda pose: setattr(pose.header.stamp, 'nanosec', 0),  # Duplicate last accepted acquisition.
    lambda pose: setattr(pose.header.stamp, 'sec', 0) or setattr(pose.header.stamp, 'nanosec', 900_000_000),
    lambda pose: setattr(pose.header.stamp, 'sec', 0) or setattr(pose.header.stamp, 'nanosec', 500_000_000),
    lambda pose: setattr(pose.header.stamp, 'sec', 2),
    lambda pose: setattr(pose.header.stamp, 'nanosec', 1_000_000_000),
])
def test_invalid_pose_sources_cannot_renew_localization_or_enable_commands(monkeypatch, mutate):
    node, outputs = _node()
    monkeypatch.setattr(module.time, 'monotonic', lambda: 1.1)
    message = _pose()
    mutate(message)
    RLPolicyNode._pose_cb(node, message)
    assert node.position is None
    assert node.pose_time == -math.inf
    assert node.pose_source_stamp_ns == 1_000_000_000
    command = Twist()
    command.linear.x = .1
    RLPolicyNode._command_cb(node, command)
    assert outputs[-1].linear.x == 0.
    assert not node.current_healthy


def test_delayed_valid_pose_uses_remaining_source_age_lease(monkeypatch):
    node, _outputs = _node()
    node.clock_ns = 1_500_000_000
    monkeypatch.setattr(module.time, 'monotonic', lambda: 5.)
    RLPolicyNode._pose_cb(node, _pose(nanosec=100_000_000))
    assert node.pose_time == pytest.approx(4.6)
    assert node._pose_is_fresh(5.05)
    assert not node._pose_is_fresh(5.11)
    node.clock_ns = 2_000_000_000
    assert not node._pose_is_fresh(5.05)  # ROS clock jump also expires the pose.


def test_source_order_can_restart_only_after_real_ros_clock_rollback(monkeypatch):
    node, _outputs = _node()
    node.clock_ns = 500_000_000
    monkeypatch.setattr(module.time, 'monotonic', lambda: 1.1)
    RLPolicyNode._pose_cb(node, _pose(sec=0, nanosec=490_000_000))
    assert node.pose_source_stamp_ns == 490_000_000
    assert node._pose_is_fresh(1.1)


def _plan():
    path = Path()
    path.header.frame_id = 'map'
    path.header.stamp.sec = 1
    pose = PoseStamped()
    pose.header.frame_id = 'map'
    pose.header.stamp.sec = 1
    pose.pose.position.x = 1.
    path.poses = [pose]
    return path


@pytest.mark.parametrize('mutate', [
    lambda path: setattr(path.header, 'frame_id', 'odom'),
    lambda path: setattr(path.header.stamp, 'sec', 0),
    lambda path: setattr(path.poses[0].header, 'frame_id', 'base_link'),
    lambda path: setattr(path.poses[0].header.stamp, 'sec', 0),
    lambda path: setattr(path.poses[0].pose.orientation, 'w', 0.),
    lambda path: setattr(path.poses[0].pose.position, 'x', math.nan),
])
def test_plan_geometry_rejects_mixed_frames_missing_stamps_and_invalid_yaw(mutate):
    node, _outputs = _node()
    path = _plan()
    mutate(path)
    RLPolicyNode._plan_cb(node, path)
    assert node.path is None
    assert not node.current_healthy


def test_map_plan_uses_finite_normalized_goal_yaw():
    node, _outputs = _node()
    node.field = SimpleNamespace(body_clearance=lambda _point, _yaw: .1)
    path = _plan()
    path.poses[0].pose.orientation.z = 2. * math.sin(.3)
    path.poses[0].pose.orientation.w = 2. * math.cos(.3)
    RLPolicyNode._plan_cb(node, path)
    assert node.goal_yaw == pytest.approx(.6)
    assert node.goal_clearance == .1


def _active_residual_node(monkeypatch):
    node, outputs = _node()
    config = FilePath(__file__).resolve().parents[1] / 'config'
    node.field = CadClearanceModel.from_yaml(
        config / 'field_planning.yaml', config / 'competition_footprints.yaml')
    node.position = np.array([-.825, .825])
    node.path = np.array([node.position, [-.8, 1.34]])
    node.policy = CompactRLPolicy.from_yaml(config / 'rl_policy.yaml')
    node.apply_baseline_repulsion = True
    node.monitor_horizon = None
    node.last_logged_decision = None
    node.get_logger = lambda: SimpleNamespace(info=lambda _message: None)
    parameters = {'pose_timeout_sec': .5, 'lookahead_m': .55,
                  'reference_speed_mps': .78, 'repulsion_edge_m': .06,
                  'repulsion_authority': .3}
    node.get_parameter = lambda name: SimpleNamespace(value=parameters[name])
    monkeypatch.setattr(module.time, 'monotonic', lambda: 1.1)
    return node, outputs


@pytest.mark.parametrize('command_x,expected', [(.1, .1), (-.1, -.081779)])
def test_actual_adapter_uses_footprint_obstacle_for_lane_correction(monkeypatch, command_x, expected):
    node, outputs = _active_residual_node(monkeypatch)
    command = Twist()
    command.linear.x = command_x
    RLPolicyNode._command_cb(node, command)
    assert outputs[-1].linear.x == pytest.approx(expected)
    assert outputs[-1].linear.y == pytest.approx(0.)
    assert node.current_healthy
    assert node.last_decision['body_clearance_m'] == pytest.approx(.023558)


def test_invalid_footprint_direction_stops_actual_adapter(monkeypatch):
    node, outputs = _active_residual_node(monkeypatch)
    node.field = SimpleNamespace(
        body_clearance_and_gradient=lambda _point, _yaw: (.03, [math.nan, 0.]))
    command = Twist()
    command.linear.x = .1
    RLPolicyNode._command_cb(node, command)
    assert outputs[-1].linear.x == 0.
    assert not node.current_healthy
    assert node.last_reason.startswith('POLICY_ERROR:')
