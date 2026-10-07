"""Planned gate stops decelerate; faults still stop in the same tick."""
import math
from types import SimpleNamespace

import numpy as np
import pytest

from omni_autonomy_next.staged_heading import HeadingStage
from test_staged_heading import staged_node


def approaching_node(direction=0.):
    node = staged_node()
    node.heading_stage = HeadingStage(
        0., math.pi / 2, gate=np.zeros(2), phase='APPROACH')
    node.command = np.array([.07 * math.cos(direction), .07 * math.sin(direction), 0.])
    node.velocity = node.command.copy()
    return node


@pytest.mark.parametrize('direction', [0., math.pi / 4, math.pi / 2, math.pi])
def test_gate_capture_decelerates_before_zero(direction):
    node = approaching_node(direction)
    previous = node.command.copy()
    for tick in range(4):
        node._stage_tick(tick * node.period, node.pose, node.velocity, 1.)
        assert node.heading_stage.phase == 'SETTLE'
        assert np.linalg.norm(node.command[:2] - previous[:2]) <= node.acceleration * node.period + 1.e-9
        assert np.dot(node.command[:2], previous[:2]) >= -1.e-12
        assert node.command[2] == 0.
        previous = node.command.copy()
    assert node.command == pytest.approx(np.zeros(3))


def test_turn_waits_for_command_to_stop_despite_delayed_odometry():
    node = approaching_node()
    node.heading_stage.phase = 'SETTLE'
    node.heading_stage.settled_since = 0.
    node.heading_stage.braking_started = 0.
    node._stage_tick(.3, node.pose, np.zeros(3), 1.)
    assert node.heading_stage.phase == 'SETTLE'
    assert node.heading_stage.settled_since is None
    assert node.command[0] > 0.


@pytest.mark.parametrize('fault', ['clearance', 'drift', 'blocked'])
def test_planned_braking_does_not_delay_safety_stop(fault):
    node = approaching_node()
    node.heading_stage.phase = 'SETTLE'
    if fault == 'clearance':
        node._stage_live_turn_clear = lambda *args: False
    elif fault == 'drift':
        node.pose[0] = .2
    else:
        node.stage_blocked = 'FIXED_HEADING_PATH_BLOCKED'
    node._stage_tick(.05, node.pose, node.velocity, 1.)
    assert node.command == pytest.approx(np.zeros(3))


def test_transport_drains_after_last_braking_command():
    node = approaching_node()
    # Exercise the calibrated 320 ms transport lag explicitly. The generic
    # tracker fixture uses 200 ms, which has already drained by the .30 tick.
    parameter = node.get_parameter
    node.get_parameter = lambda name: (SimpleNamespace(value=.32)
        if name == 'feedback_delay_sec' else parameter(name))
    node.heading_stage.phase = 'SETTLE'
    for now in (0., .05, .10, .20):
        node._stage_tick(now, node.pose, np.zeros(3), 1.)
        assert node.heading_stage.phase == 'SETTLE'
    node._stage_tick(.30, node.pose, np.zeros(3), 1.)
    assert node.heading_stage.phase == 'SETTLE'
    node._stage_tick(.45, node.pose, np.zeros(3), 1.)
    assert node.heading_stage.phase == 'ROTATE'
    assert node.command[:2] == pytest.approx([0., 0.])


def test_unrepairable_fixed_heading_path_is_not_executed(monkeypatch):
    from omni_autonomy_next import trajectory_tracker_node as tracker
    from omni_autonomy_next.rl_residual import CadClearanceModel
    from test_smooth_arrival import make_node
    from test_staged_heading import open_model

    node = make_node(goal=(2., 0., 0.))
    node.clearance = CadClearanceModel(
        [[[1., -2.], [1., 2.]]], open_model().footprint)
    # Include a previous reference: failed repair must not leave it executing.
    monkeypatch.setattr(tracker.time, 'monotonic', lambda: 0.)
    node.trajectory = object()
    tracker.TrajectoryTracker._build_trajectory(
        node, np.array([[0., 0.], [2., 0.]]), 0.)
    assert node.trajectory is None


def test_new_goal_resets_half_turn_direction():
    from geometry_msgs.msg import PoseStamped
    from omni_autonomy_next import trajectory_tracker_node as tracker
    from test_smooth_arrival import make_node

    node = make_node()
    node.previous_yaw_error = math.pi
    message = PoseStamped()
    message.header.frame_id = 'map'
    message.header.stamp.sec = 1
    message.pose.orientation.w = 1.
    tracker.TrajectoryTracker._on_goal(node, message)
    assert node.previous_yaw_error is None
