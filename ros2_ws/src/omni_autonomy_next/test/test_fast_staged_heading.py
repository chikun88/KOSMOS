"""Geometry equivalence and fast staged-mode handoff regression tests."""
import math
from pathlib import Path

import numpy as np
import pytest
from std_msgs.msg import Bool

from omni_autonomy_next import trajectory_tracker_node as tracker
from omni_autonomy_next.rl_residual import CadClearanceModel
from omni_autonomy_next.staged_heading import HeadingStage, prepare_stage
from test_staged_heading import staged_node, open_model


def test_batch_polygon_clearance_matches_scalar_on_field_and_invalid_poses():
    config = Path(tracker.__file__).resolve().parents[1] / 'config'
    model = CadClearanceModel.from_yaml(config/'field_planning.yaml',
                                        config/'competition_footprints.yaml')
    rng = np.random.default_rng(6026)
    points = rng.uniform([-3., -2.], [3., 4.], (257, 2))
    yaws = rng.uniform(-math.pi, math.pi, len(points))
    points[4, 0] = np.nan
    yaws[25] = np.inf
    for cap in (.06, .35, 2.):
        scalar = [model.body_clearance(p, y, cap) for p, y in zip(points, yaws)]
        assert model.clearance_over_poses(points, yaws, cap) == pytest.approx(scalar, abs=1.e-12)


def test_batch_rejects_crossing_and_swallowed_wall_segments():
    for wall in ([[[-2., 0.], [2., 0.]]], [[[.01, .01], [.02, .02]]]):
        model = CadClearanceModel(wall, open_model().footprint)
        assert model.clearance_over_poses([[0.,0.]], [.3])[0] == 0.


def test_replan_can_keep_gate_off_new_centerline_with_checked_connectors():
    gate = np.array([1., .3])
    stage = HeadingStage(0., math.pi/2, gate=gate, phase='APPROACH')
    updated, path, yaw = prepare_stage(open_model(), [[0.,0.],[2.,0.]],
        np.zeros(3), stage)
    assert path[-1] == pytest.approx(gate)
    assert updated.gate == pytest.approx(gate)
    # A connector through an obstacle must still fail even with clear endpoints.
    blocked = CadClearanceModel([[[.8, .2], [1.2, .2]]], open_model().footprint)
    with pytest.raises(ValueError, match='ROTATION_GATE_BLOCKED'):
        prepare_stage(blocked, [[0.,0.],[2.,0.]], np.zeros(3), stage)


def test_parallel_walls_do_not_artificially_limit_forward_cruise(monkeypatch):
    node = staged_node(goal=(4., 0., 0.))
    node.clearance = CadClearanceModel([
        [[-1., .48], [6., .48]], [[-1., -.48], [6., -.48]]], open_model().footprint)
    monkeypatch.setattr(tracker.time, 'monotonic', lambda: 0.)
    tracker.TrajectoryTracker._build_trajectory(node, np.array([[0.,0.],[4.,0.]]), 0.)
    assert node.stage_blocked is None
    assert max(node.trajectory.speed) == pytest.approx(node.speed_limit)
    assert node._stage_safe_command(.7, 0., 0.) == pytest.approx((.7, 0., 0.))
    # The same speed toward a wall still has to brake.
    assert node._stage_safe_command(0., .7, 0.)[1] < .1


def test_departure_is_built_during_turn_and_cannot_survive_cancel(monkeypatch):
    node = staged_node()
    node.heading_stage = HeadingStage(0., math.pi/2, gate=np.zeros(2),
        phase='ROTATE', rotation_started=0.)
    original = object()
    node.trajectory = original
    monkeypatch.setattr(tracker.time, 'monotonic', lambda: 0.)
    tracker.TrajectoryTracker._build_trajectory(node, np.array([[0.,0.],[2.,0.]]), math.pi/2)
    assert node.trajectory is original
    assert node.stage_continuation[0].yaws == pytest.approx(math.pi/2)
    assert node.stage_continuation[0].speed[0] == 0.
    node._stage_cancel(Bool(data=True))
    assert node.stage_continuation is None and node.trajectory is None


def test_final_gate_centimetre_has_a_real_acceleration_segment(monkeypatch):
    node = staged_node()
    node.heading_stage = HeadingStage(0., math.pi/2, gate=np.array([.008, 0.]),
        phase='APPROACH')
    monkeypatch.setattr(tracker.time, 'monotonic', lambda: 0.)
    tracker.TrajectoryTracker._build_trajectory(node, np.array([[0.,0.],[2.,0.]]), math.pi/2)
    assert node.trajectory.duration < .5
    assert max(node.trajectory.speed) > .01


def test_unchanged_gate_approach_preserves_its_acceleration_clock(monkeypatch):
    node = staged_node()
    node.heading_stage = HeadingStage(0., math.pi/2, gate=np.array([1., 0.]),
        phase='APPROACH')
    monkeypatch.setattr(tracker.time, 'monotonic', lambda: 0.)
    points = np.array([[0.,0.],[2.,0.]])
    tracker.TrajectoryTracker._build_trajectory(node, points, math.pi/2)
    first = node.trajectory
    node.reference_time = .6
    tracker.TrajectoryTracker._build_trajectory(node, points, math.pi/2)
    assert node.trajectory is first
    assert node.reference_time == .6


@pytest.mark.parametrize('invalidate', ['none', 'profile', 'revision', 'drift'])
def test_handoff_uses_only_matching_prebuilt_departure(monkeypatch, invalidate):
    node = staged_node(pose=(0.,0.,math.pi/2))
    node.heading_stage = HeadingStage(0., math.pi/2, gate=np.zeros(2),
        phase='ROTATE', rotation_started=0., settled_since=0.)
    monkeypatch.setattr(tracker.time, 'monotonic', lambda: 0.)
    tracker.TrajectoryTracker._build_trajectory(node, np.array([[0.,0.],[2.,0.]]), math.pi/2)
    departure = node.stage_continuation[0]
    if invalidate == 'profile':
        node.speed_limit *= .5
    if invalidate == 'revision':
        node.stage_revision += 1
    if invalidate == 'drift':
        node.pose[0] += .02
    node._stage_tick(.2, node.pose, np.zeros(3), 1.)
    assert node.heading_stage.phase == 'TRANSLATE'
    assert (node.trajectory is departure) == (invalidate == 'none')


def test_in_flight_tick_cannot_overwrite_replacement_clock_or_publish(monkeypatch):
    node = staged_node(goal=(4.,0.,0.))
    monkeypatch.setattr(tracker.time, 'monotonic', lambda: 0.)
    tracker.TrajectoryTracker._build_trajectory(node, np.array([[0.,0.],[4.,0.]]), 0.)
    old = node.trajectory
    sample = old.sample
    replacement = object()
    def replace_during_sample(t):
        node.trajectory = replacement
        node.reference_time = 12.3
        return sample(t)
    old.sample = replace_during_sample
    tracker.TrajectoryTracker._tick(node)
    assert node.trajectory is replacement
    assert node.reference_time == 12.3
    assert not node.commands


def test_timer_jitter_uses_elapsed_time_without_pause_catchup(monkeypatch):
    node = staged_node(goal=(4.,0.,0.))
    now = 0.
    monkeypatch.setattr(tracker.time, 'monotonic', lambda: now)
    tracker.TrajectoryTracker._build_trajectory(node, np.array([[0.,0.],[4.,0.]]), 0.)
    tracker.TrajectoryTracker._tick(node)
    first = node.reference_time
    now = .075
    node.pose_stamp = node.velocity_stamp = node.plan_stamp = now
    tracker.TrajectoryTracker._tick(node)
    assert node.reference_time-first == pytest.approx(.075)
    assert node.control_dt == .075
    node.speed_scale = 0.
    now = 9.
    node.pose_stamp = node.velocity_stamp = node.plan_stamp = now
    tracker.TrajectoryTracker._tick(node)
    assert node.control_dt <= 2*node.period
    paused = node.reference_time
    now += .05
    node.speed_scale = 1.
    node.pose_stamp = node.velocity_stamp = node.plan_stamp = now
    tracker.TrajectoryTracker._tick(node)
    assert node.reference_time-paused <= .05+1.e-9
