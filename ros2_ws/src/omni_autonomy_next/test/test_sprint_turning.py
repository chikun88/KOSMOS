"""Exercise coupled sprint turns, checked corridors and conservative fallback."""
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from omni_autonomy_next import trajectory_tracker_node as tracker
from omni_autonomy_next.config import load_robot
from omni_autonomy_next.rl_residual import CadClearanceModel
from test_smooth_arrival import make_node

CONFIG = Path(__file__).resolve().parents[1]/'config'


def corridor(wall_y):
    footprint = yaml.safe_load((CONFIG/'robot.yaml').read_text())['robot']['footprint']
    return CadClearanceModel([[[-200., wall_y], [200., wall_y]]], footprint)


@pytest.mark.parametrize('direction', np.arange(8)*math.pi/4)
@pytest.mark.parametrize('yaw', [-.3, .3])
def test_open_sprint_rotates_at_high_translation_speed(monkeypatch, direction, yaw):
    monkeypatch.setattr(tracker.time, 'monotonic', lambda: 0.)
    goal = [60.*math.cos(direction), 60.*math.sin(direction), yaw]
    node = make_node(goal=goal)
    node.profile_name = 'sprint'
    node.speed_limit = node.lateral_limit = 4.
    node.envelope.max_wheel = load_robot(str(CONFIG/'robot.yaml'))['drivetrain']['profile_max_wheel_speeds']['sprint']
    node.acceleration, node.deceleration = 3., .85
    node.clearance = corridor(-100.)
    tracker.TrajectoryTracker._build_trajectory(node, np.array([[0., 0.], goal[:2]]), yaw)
    path = node.trajectory
    turning = abs(path.yaw_per_metre) > 1.e-6
    assert np.max(path.speed[turning]) > 2.5
    assert np.all(path.motion_limits[turning, :2] == 4.)
    assert path.speed[-1] == 0.
    assert path.yaws[-1] == pytest.approx(yaw)
    for tangent, heading, rate, speed in zip(tracker.path_tangents(path.points),
                                            path.yaws, path.yaw_per_metre, path.speed):
        body = tracker.to_body(tangent*speed, heading)
        assert node.envelope.wheel_cost(*body, rate*speed) <= node.envelope.max_wheel+1.e-7


@pytest.mark.parametrize('model', [None, corridor(-.54)])
def test_missing_or_narrow_corridor_keeps_turn_cap(monkeypatch, model):
    monkeypatch.setattr(tracker.time, 'monotonic', lambda: 0.)
    node = make_node(goal=[8., 0., .3])
    node.profile_name = 'sprint'
    node.speed_limit = node.lateral_limit = 4.
    node.clearance = model
    tracker.TrajectoryTracker._build_trajectory(node, np.array([[0., 0.], [8., 0.]]), .3)
    turning = abs(node.trajectory.yaw_per_metre) > 1.e-6
    assert np.all(node.trajectory.motion_limits[turning, 0] == 1.05)


def test_between_sample_sweep_cannot_skip_wall_or_rotation():
    model = corridor(0.)
    points = np.array([[0., -1.], [0., 1.]])
    assert not tracker.sprint_turn_clearance(points, np.zeros(2), model).any()
    points = np.array([[0., 1.], [.01, 1.]])
    assert tracker.sprint_turn_clearance(points, np.zeros(2), model).all()
    assert not tracker.sprint_turn_clearance(points, np.array([0., math.pi]), model).any()


def test_nonfinite_geometry_cannot_grant_high_speed():
    assert not tracker.sprint_turn_clearance(
        np.array([[0., 0.], [float('nan'), 0.]]), np.zeros(2), corridor(-100.)).any()


def test_narrow_end_keeps_budget_for_entire_remaining_route():
    points = np.column_stack((np.zeros(40), np.linspace(1., .51, 40)))
    assert not tracker.sprint_turn_clearance(points, np.zeros(40), corridor(0.)).any()


@pytest.mark.parametrize('profile,expected', [('sprint', 1.3), ('balanced', 1.05)])
def test_checked_corridor_gets_intermediate_sprint_budget(monkeypatch, profile, expected):
    monkeypatch.setattr(tracker.time, 'monotonic', lambda: 0.)
    node = make_node(goal=[8., 0., .3])
    node.profile_name = profile
    node.speed_limit = node.lateral_limit = 4.
    node.envelope.max_wheel = 56.568542494
    node.clearance = corridor(-.8)
    tracker.TrajectoryTracker._build_trajectory(node, np.array([[0., 0.], [8., 0.]]), .3)
    path = node.trajectory
    turning = abs(path.yaw_per_metre) > 1.e-6
    assert np.all(path.motion_limits[turning, 0] == expected)
    assert .98*expected <= np.max(path.speed[turning]) <= expected+1.e-9
    for tangent, heading, rate, speed, budget in zip(tracker.path_tangents(path.points),
            path.yaws, path.yaw_per_metre, path.speed, path.motion_limits):
        body = tracker.to_body(tangent*speed, heading)
        assert node.envelope.wheel_cost(*body, rate*speed) <= budget[2]+1.e-7
    assert path.speed[-1] == 0.


@pytest.mark.parametrize('reserve', [0., -.1, float('nan'), float('inf')])
def test_invalid_reserve_never_releases_turn_budget(reserve):
    assert not tracker.sprint_turn_clearance(np.array([[0., 0.], [.05, 0.]]),
        np.zeros(2), corridor(-100.), reserve_m=reserve).any()


def test_replan_cannot_promote_rejected_cruise_but_new_goal_can(monkeypatch):
    monkeypatch.setattr(tracker.time, 'monotonic', lambda: 0.)
    node = make_node(goal=[8., 0., .3])
    node.profile_name = 'sprint'
    node.speed_limit = node.lateral_limit = 4.
    node.envelope.max_wheel = 56.568542494
    node.clearance = None
    tracker.TrajectoryTracker._build_trajectory(node, np.array([[0., 0.], [8., 0.]]), .3)
    assert not node.trajectory.sprint_cruise_allowed
    node.clearance = corridor(-.8)
    tracker.TrajectoryTracker._build_trajectory(node, np.array([[0., 0.], [4., .4], [8., 0.]]), .3)
    assert not node.trajectory.sprint_cruise_allowed
    turning = abs(node.trajectory.yaw_per_metre) > 1.e-6
    assert np.all(node.trajectory.motion_limits[turning, 0] == 1.05)
    node.active_goal = np.array([9., 0., .3])
    tracker.TrajectoryTracker._build_trajectory(node, np.array([[0., 0.], [9., 0.]]), .3)
    assert node.trajectory.sprint_cruise_allowed
    # Remain footprint-clear while removing the optional open-sprint reserve.
    node.clearance = corridor(-.54)
    tracker.TrajectoryTracker._build_trajectory(node, np.array([[0., 0.], [4., .4], [9., 0.]]), .3)
    assert not node.trajectory.sprint_cruise_allowed


def test_predictive_corridor_uses_four_mps_without_changing_balanced(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(tracker.time, 'monotonic', lambda: 0.)
    for profile, expected in [('sprint', 4.), ('balanced', 1.05)]:
        node = make_node(goal=[8., 0., .3])
        node.profile_name = profile
        node.speed_limit = node.lateral_limit = 4.
        node.envelope.max_wheel = 56.568542494
        node.clearance = corridor(-.8)
        get = node.get_parameter
        node.get_parameter = lambda name: SimpleNamespace(value=True) if name=='predictive_sprint' else get(name)
        tracker.TrajectoryTracker._build_trajectory(node,np.array([[0.,0.],[8.,0.]]),.3)
        turning = abs(node.trajectory.yaw_per_metre) > 1.e-6
        assert np.all(node.trajectory.motion_limits[turning,0] == expected)


def test_predictor_is_not_used_when_corridor_is_ineligible(monkeypatch):
    from types import SimpleNamespace
    node = make_node(goal=[8.,0.,.3])
    node.profile_name = 'sprint'
    node.speed_limit = node.lateral_limit = 4.
    get = node.get_parameter
    node.get_parameter = lambda name: SimpleNamespace(value=True) if name=='predictive_sprint' else get(name)
    monkeypatch.setattr(tracker.time,'monotonic',lambda:0.)
    tracker.TrajectoryTracker._build_trajectory(node,np.array([[0.,0.],[8.,0.]]),.3)
    tracker.TrajectoryTracker._tick(node)
    assert node.prediction_frame is None
    assert not hasattr(node,'motion_predictor')


@pytest.mark.parametrize('direction,wall,expected', [
    (0., -.8, 4.), (math.pi, -.8, 4.),
    (math.pi/2, -.8, 3.), (math.pi/4, -.8, 3.),
    (0., -.62, 3.),
])
def test_faster_turn_requires_straight_departure_and_extra_clearance(monkeypatch, direction, wall, expected):
    from types import SimpleNamespace
    monkeypatch.setattr(tracker.time, 'monotonic', lambda: 0.)
    axis = np.array([math.cos(direction), math.sin(direction)])
    normal = np.array([-axis[1], axis[0]])
    node = make_node(goal=[*(8*axis), .3])
    node.profile_name = 'sprint'
    node.speed_limit = node.lateral_limit = 4.
    node.envelope.max_wheel = 56.568542494
    model = corridor(wall)
    node.clearance = CadClearanceModel(
        [[(-200*axis+wall*normal).tolist(), (200*axis+wall*normal).tolist()]], model.footprint)
    get = node.get_parameter
    node.get_parameter = lambda name: SimpleNamespace(value=True) if name == 'predictive_sprint' else get(name)
    tracker.TrajectoryTracker._build_trajectory(node, np.array([[0., 0.], 8*axis]), .3)
    turning = abs(node.trajectory.yaw_per_metre) > 1.e-6
    assert np.all(node.trajectory.motion_limits[turning, 0] == expected)


def test_fast_turn_replan_keeps_departure_frame_and_cannot_promote(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(tracker.time, 'monotonic', lambda: 0.)
    node = make_node(goal=[8., 0., .3])
    node.profile_name = 'sprint'
    node.speed_limit = node.lateral_limit = 4.
    node.envelope.max_wheel = 56.568542494
    node.clearance = corridor(-.8)
    get = node.get_parameter
    node.get_parameter = lambda name: SimpleNamespace(value=True) if name == 'predictive_sprint' else get(name)
    path = np.array([[0., 0.], [8., 0.]])
    tracker.TrajectoryTracker._build_trajectory(node, path, .3)
    assert node.trajectory.sprint_fast_turn_allowed
    node.pose[2] = .15
    path = np.array([[0., 0.], [4., -.05], [8., 0.]])
    tracker.TrajectoryTracker._build_trajectory(node, path, .3)
    assert node.trajectory.sprint_fast_turn_allowed
    assert node.trajectory.turn_start_yaw == 0.
    node.clearance = corridor(-.62)
    path = np.array([[0., 0.], [4., -.1], [8., 0.]])
    tracker.TrajectoryTracker._build_trajectory(node, path, .3)
    assert not node.trajectory.sprint_fast_turn_allowed
    node.clearance = corridor(-.8)
    path = np.array([[0., 0.], [8., 0.]])
    tracker.TrajectoryTracker._build_trajectory(node, path, .3)
    assert not node.trajectory.sprint_fast_turn_allowed


@pytest.mark.parametrize('direction', np.arange(8)*math.pi/4)
@pytest.mark.parametrize('model', [None, corridor(-.51), corridor(-.8)])
def test_everywhere_sprint_preserves_footprint_gate_and_coupled_budget(monkeypatch, direction, model):
    from types import SimpleNamespace
    monkeypatch.setattr(tracker.time, 'monotonic', lambda: 0.)
    goal = [8.*math.cos(direction), 8.*math.sin(direction), .3]
    node = make_node(goal=goal)
    node.profile_name = 'sprint'
    node.speed_limit = node.lateral_limit = 4.
    node.clearance = model
    get = node.get_parameter
    node.get_parameter = lambda name: SimpleNamespace(value=True) if name in (
        'predictive_sprint', 'sprint_turn_everywhere') else get(name)
    for endpoint in (goal[:2], [goal[0]*.9, goal[1]*.9]):
        previous = node.trajectory
        tracker.TrajectoryTracker._build_trajectory(node, np.array([[0., 0.], endpoint]), .3)
        path = node.trajectory
        if np.linalg.norm(np.asarray(endpoint)-goal[:2]) > get('goal_snap_distance_m').value:
            # This short replan cannot reach the active goal and therefore
            # retains the first accepted path (or its blocked outcome).
            assert path is previous
        if path is None:
            # The synthetic narrow wall fixture can be crossed by this
            # direction or lack the reserved margin for its rotated outline.
            # Sprint eligibility
            # must not grant permission to execute that uncertified route.
            assert model is not None and node.planning_blocked
            raw = tracker.resample(np.array([[0., 0.], goal[:2]]), .05)
            _, _, headings = tracker.path_heading_schedule(node, raw, node.pose, .3)
            assert tracker.pose_path_clearance(model, raw, headings) < .025
            continue
        if model is not None:
            assert path.pose_path_clearance_m >= .025
        assert path.sprint_cruise_allowed
        assert path.sprint_fast_turn_allowed
        assert np.all(path.motion_limits[:, :2] == 4.)
        assert path.speed[-1] == 0.
        for tangent, heading, rate, speed in zip(tracker.path_tangents(path.points),
                path.yaws, path.yaw_per_metre, path.speed):
            body = tracker.to_body(tangent*speed, heading)
            assert node.envelope.wheel_cost(*body, rate*speed) <= node.envelope.max_wheel+1.e-7


@pytest.mark.parametrize('profile,predictive', [('balanced', True), ('sprint', False)])
def test_everywhere_requires_predictive_sprint(monkeypatch, profile, predictive):
    from types import SimpleNamespace
    monkeypatch.setattr(tracker.time, 'monotonic', lambda: 0.)
    node = make_node(goal=[8., 0., .3])
    node.profile_name = profile
    node.speed_limit = node.lateral_limit = 4.
    get = node.get_parameter
    values = {'sprint_turn_everywhere': True, 'predictive_sprint': predictive}
    node.get_parameter = lambda name: SimpleNamespace(value=values[name]) if name in values else get(name)
    tracker.TrajectoryTracker._build_trajectory(node, np.array([[0., 0.], [8., 0.]]), .3)
    turning = abs(node.trajectory.yaw_per_metre) > 1.e-6
    assert np.all(node.trajectory.motion_limits[turning, 0] == 1.05)
