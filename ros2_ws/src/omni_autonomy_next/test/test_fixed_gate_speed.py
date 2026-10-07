"""Fixed-gate passage speed must be planned before a fast rounded corner."""
import ast
from pathlib import Path

import numpy as np
import pytest

from omni_autonomy_next.fixed_gate_speed import FixedGateSpeed
from omni_autonomy_next import trajectory_tracker_node as tracker
from test_smooth_arrival import make_node
from test_staged_heading import staged_node

PACKAGE = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('sign', [-1., 1.])
def test_start_exit_override_is_local_and_survives_replan(sign):
    limiter = FixedGateSpeed.from_yaml(PACKAGE/'config/routes.yaml')
    for gate, expected in [([sign*1.8, 4.1], 3.5), ([sign*.8, 2.15], 2.)]:
        points = np.array([gate, gate, gate])+[[0., -.2], [0., 0.], [0., .1]]
        limits, turns = limiter.plan_limits(points, .85,
            curvature=[0., 3., 0.], arclength=[0., .2, .3])
        assert limits[1] == pytest.approx(expected)
        limits, _ = limiter.plan_limits(points[1:], .85,
            curvature=[0., 0.], arclength=[0., .1], retained_turns=turns)
        assert limits[0] == pytest.approx(expected)


@pytest.mark.parametrize('value', [0., -1., float('nan'), float('inf')])
def test_invalid_local_speed_is_rejected(tmp_path, value):
    import yaml
    data = yaml.safe_load((PACKAGE/'config/routes.yaml').read_text())
    data['fixed_departures'][1]['turn_speed_m_s'] = value
    path = tmp_path/'routes.yaml'
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match='turn_speed_m_s'):
        FixedGateSpeed.from_yaml(path)


def test_straight_cruise_is_not_capped_on_either_field():
    limit = FixedGateSpeed.from_yaml(PACKAGE/'config/routes.yaml')
    points = np.array([[1.55, .15], [1.55, -1.25], [.8, -1.55], [.8, .75], [1.8, 4.1]])
    for sign in (-1., 1.):
        assert np.isinf(limit.limits(points*[sign, 1.], .85,
            curvature=np.zeros(len(points)), arclength=np.arange(len(points)))).all()


def test_braking_preview_has_no_boundary_step_and_includes_response_delay():
    limit = FixedGateSpeed([[3., 0.]])
    distances = np.linspace(0., 3., 1001)
    curvature = np.zeros(len(distances)); curvature[-1] = 5.
    speed = limit.limits(np.column_stack((distances, np.zeros_like(distances))), .85,
                         curvature=curvature, arclength=distances)
    travel = (speed-.2)*.3+(speed**2-.2**2)/(2*.85)
    assert travel == pytest.approx(np.maximum(0., 2.75-distances))
    assert np.max(np.abs(np.diff(speed))) < .01
    assert speed[-1] == pytest.approx(.2)
    # Open-space cruising is still available outside the braking distance.
    assert speed[0] > 1.8


@pytest.mark.parametrize('staged', [False, True])
def test_real_builder_keeps_straight_gate_cruise_and_lower_profile_limits(monkeypatch, staged):
    monkeypatch.setattr(tracker.time, 'monotonic', lambda: 0.)
    node = (staged_node if staged else make_node)(goal=(4., 0., 0.))
    node.fixed_gate_speed = FixedGateSpeed([[2., 0.]])
    node.speed_limit = node.lateral_limit = 2.
    tracker.TrajectoryTracker._build_trajectory(node, np.array([[0., 0.], [4., 0.]]), 0.)
    trajectory = node.trajectory
    assert trajectory is not None
    near = np.linalg.norm(trajectory.points-[2., 0.], axis=1) <= .35
    assert max(trajectory.speed[near]) > 1.
    assert trajectory.points[-1] == pytest.approx([4., 0.])
    node.speed_limit = node.lateral_limit = .15
    tracker.TrajectoryTracker._build_trajectory(node, np.array([[0., 0.], [4., 0.]]), 0.)
    assert max(node.trajectory.speed) <= .15+1.e-9


def test_tight_turn_brakes_but_releases_the_straight_exit(monkeypatch):
    monkeypatch.setattr(tracker.time, 'monotonic', lambda: 0.)
    node = make_node(goal=(2., 2., 0.))
    node.fixed_gate_speed = FixedGateSpeed([[2., 0.]])
    node.speed_limit = node.lateral_limit = 2.
    tracker.TrajectoryTracker._build_trajectory(node,
        np.array([[0., 0.], [2., 0.], [2., 2.]]), 0.)
    t = node.trajectory
    near = np.linalg.norm(t.points-[2., 0.], axis=1) < .10
    assert min(t.speed[near]) <= .21
    assert max(t.speed[t.points[:, 1] > .5]) > .6


def test_replan_inside_turn_retains_braking_until_the_exit_is_clear():
    limiter = FixedGateSpeed([[2., 0.]])
    # The new plan's straight prefix has lost the old bend. Keep the turn
    # captured by the preceding trajectory, without reducing a fresh straight
    # route that never had a turn at this gate.
    points = np.array([[2., .02], [2., .1], [2., .2], [2., .6]])
    arc = points[:, 1]-points[0, 1]
    limits, _ = limiter.plan_limits(points, .85, curvature=np.zeros(4), arclength=arc,
                                    retained_turns=((2., 0., .2),))
    assert limits[:2] == pytest.approx([.2, .2])
    assert np.isinf(limits[-1])
    fresh = limiter.limits(points, .85, curvature=np.zeros(4), arclength=arc)
    assert np.isinf(fresh).all()
    beyond = np.array([[2., .4], [2., .7]])
    limits, _ = limiter.plan_limits(beyond, .85, curvature=np.zeros(2), arclength=[0., .3],
                                    retained_turns=((2., 0., .2),))
    assert np.isinf(limits).all()


def test_feedback_cannot_override_turn_speed_but_open_cruise_is_unlimited(monkeypatch):
    monkeypatch.setattr(tracker.time, 'monotonic', lambda: 0.)
    node = make_node(goal=(3., 0., 0.))
    # Observe the complete requested command; acceleration limiting has its
    # own tests and would otherwise hide the cap behind the first ramp tick.
    node._rate_limit = lambda *command: command
    tracker.TrajectoryTracker._build_trajectory(node, np.array([[0., 0.], [3., 0.]]), 0.)
    node.reference_time = .8
    node.trajectory.gate_turn_limits = np.full(len(node.trajectory.points), .2)
    tracker.TrajectoryTracker._tick(node)
    assert 0. < np.linalg.norm(node.command[:2]) <= .2
    node.trajectory.gate_turn_limits[:] = np.inf
    node.reference_time = .8
    tracker.TrajectoryTracker._tick(node)
    assert np.linalg.norm(node.command[:2]) > .2


def test_launch_passes_route_geometry_to_tracker():
    tree = ast.parse((PACKAGE/'launch/system.launch.py').read_text())
    nodes = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and any(k.arg == 'executable' and isinstance(k.value, ast.Constant)
                     and k.value.value == 'trajectory_tracker' for k in n.keywords)]
    assert len(nodes) == 1
    assert 'routes_config_file' in ast.unparse(nodes[0])


def test_invalid_map_frame_is_rejected(tmp_path):
    path = tmp_path/'routes.yaml'
    path.write_text('frame_id: odom\n')
    with pytest.raises(ValueError, match='map frame'):
        FixedGateSpeed.from_yaml(path)


def test_launcher_rebuilds_when_native_route_plugin_is_missing(tmp_path, monkeypatch):
    import importlib.util
    spec = importlib.util.spec_from_file_location('bucket_launcher', PACKAGE.parents[2]/'run.py')
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    setup = tmp_path/'install/setup.bash'; setup.parent.mkdir(); setup.touch()
    share = tmp_path/'share'; share.mkdir()
    monkeypatch.setattr(launcher, 'WORKSPACE', tmp_path)
    monkeypatch.setattr(launcher, 'WORKSPACE_SETUP', setup)
    monkeypatch.setattr(launcher, 'PACKAGE_SHARE', share)
    assert not launcher.workspace_is_built()
    library = tmp_path/'install/omni_route_bt/lib/libomni_remove_passed_bucket_goals_bt_node.so'
    library.parent.mkdir(parents=True); library.touch()
    source = tmp_path/'src'; source.mkdir()
    launcher.remember_built_sources(source, tmp_path/'install', launcher.source_digest(source))
    assert launcher.workspace_is_built()
