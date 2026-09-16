"""Regression coverage for backwards gate correction and handoff latency."""
import math

import numpy as np
import pytest

from omni_autonomy_next import trajectory_tracker_node as tracker
from omni_autonomy_next import staged_heading_node as staged
from omni_autonomy_next.staged_heading import HeadingStage, prepare_stage
from test_staged_heading import staged_node, open_model


def test_replan_projection_never_adds_a_backwards_vertex():
    # The closest sampled point is behind us. Use the segment projection.
    pose = np.array([.009, 0., 0.])
    stage, path, _ = prepare_stage(open_model(), [[0., 0.], [2., 0.]],
                                  pose, HeadingStage(0., 0.))
    assert np.min(path[:, 0]) >= pose[0]-1.e-12
    assert np.all(np.diff(path[:, 0]) >= -1.e-12)


def test_passing_gate_by_two_centimetres_does_not_command_a_return(monkeypatch):
    node = staged_node(pose=(1.02, 0., 0.))
    node.heading_stage = HeadingStage(0., math.pi/2,
                                     gate=np.array([1., 0.]), phase='APPROACH')
    monkeypatch.setattr(tracker.time, 'monotonic', lambda: 0.)
    tracker.TrajectoryTracker._build_trajectory(node,
        np.array([[1.02, 0.], [2., 0.]]), math.pi/2)
    tracker.TrajectoryTracker._tick(node)
    assert node.heading_stage.phase == 'SETTLE'
    assert node.commands == [(0., 0., 0.)]
    node._stage_tick(.05, node.pose, np.zeros(3), 1.)
    assert node.heading_stage.phase == 'SETTLE'  # drain transport first
    node._stage_tick(.2, node.pose, np.zeros(3), 1.)
    assert node.heading_stage.phase == 'SETTLE'
    node._stage_tick(.35, node.pose, np.zeros(3), 1.)
    assert node.heading_stage.phase == 'ROTATE'
    assert node.heading_stage.gate == pytest.approx([1.02, 0.])
    assert node.commands[-1][2] > 0.  # first torque in transition tick
    tracker.TrajectoryTracker._build_trajectory(node,
        np.array([[1., 0.], [2., 0.]]), math.pi/2)
    assert np.min(node.stage_continuation[0].points[:, 0]) >= 1.02-1.e-12


@pytest.mark.parametrize('speed,offset', [(.3, .0), (.08, .01), (.02, .11)])
def test_capture_rejects_excess_momentum_or_position_error(speed, offset):
    node = staged_node(pose=(offset, 0., 0.))
    node.heading_stage = HeadingStage(0., math.pi/2, gate=np.zeros(2), phase='APPROACH')
    assert not node._stage_tick(0., node.pose, np.array([speed, 0., 0.]), 1.)
    assert node.heading_stage.phase == 'APPROACH'


def test_departure_starts_in_handoff_tick_without_old_approach_command(monkeypatch):
    node = staged_node(pose=(0., 0., math.pi/2))
    node.heading_stage = HeadingStage(0., math.pi/2, gate=np.zeros(2),
        phase='ROTATE', rotation_started=0., settled_since=0.)
    node.tracked_endpoint = np.zeros(2)
    node.best_distance = .005
    # The stale approach would drive backwards if the tick kept its snapshot.
    node.trajectory = tracker.Trajectory(np.array([[0., 0.], [-1., 0.]]),
        np.zeros(2), np.ones(2), acceleration=.85,
        lateral_acceleration=1.2, entry_speed=0.)
    monkeypatch.setattr(tracker.time, 'monotonic', lambda: .05)
    tracker.TrajectoryTracker._build_trajectory(node, np.array([[0., 0.], [2., 0.]]), math.pi/2)
    departure = node.stage_continuation[0]
    tracker.TrajectoryTracker._tick(node)
    assert node.heading_stage.phase == 'TRANSLATE'
    assert node.trajectory is departure
    assert len(node.commands) == 1
    assert node.commands[0][1] < 0.  # +world-x at +90 degree heading
    assert node.commands[0][2] == pytest.approx(0.)
    assert node.reference_time > 0.
    assert node.tracked_endpoint == pytest.approx([2., 0.])
    assert node.best_distance > 1.9


def test_obstacle_still_blocks_same_tick_rotation_handoff():
    node = staged_node()
    node.heading_stage = HeadingStage(0., math.pi/2, gate=np.zeros(2),
        phase='SETTLE', settled_since=0., braking_started=0.)
    node._stage_live_turn_clear = lambda *args: False
    node._stage_tick(.35, node.pose, np.zeros(3), 1.)
    assert node.commands[-1] == (0., 0., 0.)
    assert node.statuses[-1] == 'STAGED_WAITING_CLEARANCE'


def test_stop_confirmation_resets_after_measured_motion():
    node = staged_node()
    node.heading_stage = HeadingStage(0., math.pi/2, gate=np.zeros(2),
        phase='SETTLE', settled_since=0., braking_started=0.)
    node._stage_tick(.2, node.pose, np.array([.025, 0., 0.]), 1.)
    assert node.heading_stage.settled_since is None
    node._stage_tick(.25, node.pose, np.zeros(3), 1.)
    assert node.heading_stage.phase == 'SETTLE'
    node._stage_tick(.3, node.pose, np.zeros(3), 1.)
    assert node.heading_stage.phase == 'SETTLE'
    node._stage_tick(.35, node.pose, np.zeros(3), 1.)
    assert node.heading_stage.phase == 'ROTATE'


def test_early_capture_requires_clearance_at_actual_stop_not_only_nominal_gate(monkeypatch):
    node = staged_node(pose=(-.02, 0., 0.))
    node.heading_stage = HeadingStage(0., math.pi/2, gate=np.zeros(2), phase='APPROACH')
    monkeypatch.setattr(staged, 'rotation_clearance', lambda *args: .095)
    assert not node._stage_tick(0., node.pose, np.zeros(3), 1.)
    assert node.heading_stage.phase == 'APPROACH'
