"""Full-body checks for simultaneous paths whose heading changes while moving."""
import math
import json
from pathlib import Path
from types import SimpleNamespace
import threading

import numpy as np
import pytest
import rclpy
from rclpy.node import Node

from omni_autonomy_next import trajectory_tracker_node as tracker
from omni_autonomy_next.rl_residual import CadClearanceModel
from test_smooth_arrival import make_node


CONFIG = Path(tracker.__file__).resolve().parents[1] / 'config'


def field():
    return CadClearanceModel.from_yaml(
        CONFIG / 'field_planning.yaml', CONFIG / 'competition_footprints.yaml')


# Downsampled first Nav2 plan from the frozen fddd421 6->3 synthetic run.
# The original trace's true minimum was 4.827 mm beside the upper pole corner.
RECORDED_PATH = np.array([
    [-.83, .22], [-.99, .245640], [-1.15, .279282], [-1.31, .279996],
    [-1.47, .28], [-1.63, .28], [-1.79, .28], [-1.95, .28],
    [-2.11, .280119], [-2.27, .293299], [-2.43, .392145], [-2.59, .472401],
    [-2.75, .515988], [-2.91, .534322], [-3.07, .539931], [-3.23, .538746],
    [-3.39, .521306], [-3.55, .492417], [-3.709999, .428838],
    [-3.869739, .315565], [-4.012062, .159963], [-4.068023, 0.],
    [-4.069924, -.16], [-4.061498, -.32], [-3.949175, -.48], [-3.89, -.54],
])


def cad_node():
    node = make_node(pose=(-.81, .23, -.2), goal=(-3.88, -.55, 0.))
    node.clearance = field()
    return node


def test_recorded_changing_heading_path_is_repaired_and_certified(monkeypatch):
    monkeypatch.setattr(tracker.time, 'monotonic', lambda: 0.)
    old = cad_node()
    # Reproduce the prior missing validation while keeping the actual builder's
    # smoothing, endpoint snap and linear yaw schedule unchanged.
    with monkeypatch.context() as bypass:
        bypass.setattr(tracker, 'pose_path_clearance', lambda *a: math.inf)
        tracker.TrajectoryTracker._build_trajectory(old, RECORDED_PATH.copy(), 0.)
    assert old.trajectory is not None
    assert tracker.pose_path_clearance(
        old.clearance, old.trajectory.points, old.trajectory.yaws) < .025

    node = cad_node()
    tracker.TrajectoryTracker._build_trajectory(node, RECORDED_PATH.copy(), 0.)
    assert node.trajectory is not None, getattr(node, 'planning_blocked', None)
    assert tracker.pose_path_clearance(
        node.clearance, node.trajectory.points, node.trajectory.yaws) >= .025
    assert node.trajectory.pose_path_clearance_m >= .025
    assert node.trajectory.points[0] == pytest.approx(node.pose[:2])
    assert node.trajectory.points[-1] == pytest.approx(node.active_goal[:2])
    # Every repaired point remains within the prior allowed corridor.
    start, delta = old.trajectory.points[:-1], np.diff(old.trajectory.points, axis=0)
    offsets = node.trajectory.points[:, None, :] - start
    fraction = np.clip(np.einsum('ijk,jk->ij', offsets, delta)
                       / np.maximum(np.sum(delta*delta, axis=1), 1.e-12), 0., 1.)
    distances = np.linalg.norm(offsets-fraction[:, :, None]*delta, axis=2)
    assert np.max(np.min(distances, axis=1)) <= .120000001


def test_rotation_between_clear_endpoints_is_not_missed():
    # A long narrow outline fits horizontally and after 180 degrees, while
    # its intermediate vertical heading crosses the horizontal wall.
    model = CadClearanceModel([[[ -2., .30], [2., .30]]],
                              [[-.5, -.1], [.5, -.1], [.5, .1], [-.5, .1]])
    points = np.zeros((2, 2))
    assert model.body_clearance(points[0], 0.) > .025
    assert model.body_clearance(points[1], math.pi) > .025
    assert tracker.pose_path_clearance(model, points, [0., math.pi]) < .025


def test_certification_follows_raw_trajectory_yaw_interpolation():
    points, yaws = tracker.dense_pose_path(
        np.zeros((2, 2)), [math.pi-.1, -math.pi+.1], radius=.5)
    # np.interp in Trajectory.sample traverses zero for these stored values.
    # A shortest-angle checker would certify the other, brief rotation.
    assert np.min(np.abs(yaws)) < .01
    travel = np.linalg.norm(np.diff(points, axis=0), axis=1)+.5*np.abs(np.diff(yaws))
    assert travel.max() <= .010000001


@pytest.mark.parametrize('failure', ['below_margin', 'nan', 'repair_error'])
def test_uncertifiable_new_path_clears_old_motion_and_reports_zero(monkeypatch, failure):
    node = cad_node()
    monkeypatch.setattr(tracker.time, 'monotonic', lambda: 0.)
    with monkeypatch.context() as bypass:
        bypass.setattr(tracker, 'pose_path_clearance', lambda *a: .04)
        tracker.TrajectoryTracker._build_trajectory(node, RECORDED_PATH.copy(), 0.)
    assert node.trajectory is not None
    node.stage_continuation = object()
    # Force a materially different incoming path, rather than cache reuse.
    replacement = RECORDED_PATH.copy()
    replacement[5:10, 1] += .15
    if failure == 'nan':
        monkeypatch.setattr(node.clearance, 'clearance_over_poses',
                            lambda points, yaws, **kw: np.full(len(points), np.nan))
    else:
        monkeypatch.setattr(tracker, 'pose_path_clearance', lambda *a: .024)
        if failure == 'repair_error':
            def reject(*args, **kwargs):
                raise ValueError('bad repair geometry')
            monkeypatch.setattr(tracker, 'repair_pose_path', reject)
    tracker.TrajectoryTracker._build_trajectory(node, replacement, 0.)
    assert node.trajectory is None
    assert node.stage_continuation is None
    assert node.planning_blocked
    node.velocity_stamp = node.pose_stamp = .1
    node._relay_behavior = lambda *a: pytest.fail('blocked path relayed behavior')
    monkeypatch.setattr(tracker.time, 'monotonic', lambda: .1)
    tracker.TrajectoryTracker._tick(node)
    assert node.commands[-1] == pytest.approx((0., 0., 0.))
    assert node.statuses[-1] == 'PATH_CLEARANCE_BLOCKED'


def test_failed_old_worker_cannot_clear_new_revision(monkeypatch):
    node = cad_node()
    node.stage_revision = 0
    monkeypatch.setattr(tracker.time, 'monotonic', lambda: 0.)
    original = object()
    node.trajectory = None
    def superseded(*args):
        node.stage_revision += 1
        node.trajectory = original
        return .0
    monkeypatch.setattr(tracker, 'pose_path_clearance', superseded)
    monkeypatch.setattr(tracker, 'repair_pose_path', lambda *a, **kw: a[1])
    tracker.TrajectoryTracker._build_trajectory(node, RECORDED_PATH.copy(), 0.)
    assert node.trajectory is original
    assert not getattr(node, 'planning_blocked', None)


def test_mixed_path_endpoints_cannot_be_moved_to_invent_clearance():
    model = field()
    points = np.array([[-3.476190, .497266], [-3.6, .60], [-3.88, -.55]])
    yaws = np.array([-.060167, -.03, 0.])
    repaired = tracker.repair_pose_path(model, points, yaws)
    assert repaired[0] == pytest.approx(points[0])
    assert repaired[-1] == pytest.approx(points[-1])
    # The recorded near-contact start remains blocked; a local repair may not
    # relocate the physical base to conceal its insufficient margin.
    linear_yaws = np.linspace(yaws[0], yaws[-1], len(repaired))
    assert tracker.pose_path_clearance(model, repaired, linear_yaws) < .025


@pytest.mark.parametrize('superseded', [False, True])
def test_unexpected_worker_error_clears_only_its_own_plan(superseded):
    previous, replacement = object(), object()
    node = SimpleNamespace(stopping=False, pending_plan=(RECORDED_PATH, 0.),
                           trajectory=previous, stage_revision=1,
                           stage_continuation=object(), lock=threading.Lock(),
                           get_logger=lambda: SimpleNamespace(warning=lambda *a: None))
    node.plan_event = SimpleNamespace(wait=lambda **kw: True, clear=lambda: None)
    def failed_build(*args):
        node.stopping = True
        if superseded:
            node.stage_revision += 1
            node.trajectory = replacement
        raise RuntimeError('unexpected geometry failure')
    node._build_trajectory = failed_build
    tracker.TrajectoryTracker._plan_worker(node)
    if superseded:
        assert node.trajectory is replacement
        assert not getattr(node, 'planning_blocked', None)
    else:
        assert node.trajectory is None and node.stage_continuation is None
        assert node.planning_blocked == 'TRAJECTORY_BUILD_ERROR:unexpected geometry failure'


@pytest.mark.parametrize('failure', ['missing_file', 'invalid_geometry', 'missing_polygon'])
def test_configured_cad_failure_rejects_constructor_before_worker(monkeypatch, failure):
    def load(*args, **kwargs):
        if failure == 'missing_file':
            raise OSError('CAD file not found')
        if failure == 'invalid_geometry':
            raise ValueError('invalid CAD wall')
        return CadClearanceModel([[[0., 0.], [1., 0.]]])
    monkeypatch.setattr(tracker.CadClearanceModel, 'from_yaml', load)
    rclpy.init(args=['--ros-args', '-p', 'field_config_file:=/configured/field.yaml',
                     '-p', f'robot_config_file:={CONFIG / "robot.yaml"}',
                     '-p', f'runtime_config_file:={CONFIG / "runtime.yaml"}',
                     '-p', f'nav2_config_file:={CONFIG / "nav2_next.yaml"}'])
    node = tracker.TrajectoryTracker.__new__(tracker.TrajectoryTracker)
    try:
        with pytest.raises(ValueError, match='configured footprint clearance'):
            tracker.TrajectoryTracker.__init__(node)
        assert not hasattr(node, 'worker')
    finally:
        Node.destroy_node(node)
        rclpy.try_shutdown()


def test_recorded_bucket_to_flag_uses_a_cad_derived_heading_hold(monkeypatch):
    monkeypatch.setattr(tracker.time, 'monotonic', lambda: 0.)
    recorded = json.loads((Path(__file__).parent / 'data/mixed_heading_4_7_plan.json').read_text())
    points = np.asarray(recorded['points'])
    node = make_node(pose=(-.81, 1.34, 0.), goal=(-.85, -1.19, .18))
    node.clearance = field()
    tracker.TrajectoryTracker._build_trajectory(node, points, .18)
    assert node.trajectory is not None, getattr(node, 'planning_blocked', None)
    trajectory = node.trajectory
    assert trajectory.pose_path_clearance_m >= .025
    assert trajectory.heading_hold_m is not None
    assert trajectory.points[0] == pytest.approx(node.pose[:2])
    assert trajectory.points[-1] == pytest.approx(node.active_goal[:2])
    assert trajectory.yaws[0] == pytest.approx(node.pose[2])
    assert trajectory.yaws[-1] == pytest.approx(node.active_goal[2])
    cutoff = max(trajectory.length-tracker.terminal_zone(node), .5*trajectory.length)
    assert np.allclose(trajectory.yaws[trajectory.arclength >= cutoff], .18)
    original = tracker.smooth_path(tracker.resample(
        tracker.remaining_path(np.vstack((points, node.active_goal[:2])), node.pose[:2]), .05), .05, .12)
    start, delta = original[:-1], np.diff(original, axis=0)
    offsets = trajectory.points[:, None, :] - start
    fraction = np.clip(np.einsum('ijk,jk->ij', offsets, delta)
                      / np.maximum(np.sum(delta*delta, axis=1), 1.e-12), 0., 1.)
    assert np.max(np.min(np.linalg.norm(offsets-fraction[:, :, None]*delta, axis=2), axis=1)) <= .120000001
