from copy import deepcopy
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from omni_autonomy_next.config import load_collision_monitor_horizon
from omni_autonomy_next.configured_goals import load_configured_poses
from omni_autonomy_next.rl_policy import STATE_FIELDS, CompactRLPolicy, RLObservation
from omni_autonomy_next.rl_residual import (
    CadClearanceModel,
    apply_clearance_residual,
    limit_yaw_rate,
    load_footprint,
    make_observation,
)


CONFIG = Path(__file__).resolve().parents[1] / 'config'


def deployed_policy():
    return CompactRLPolicy.from_yaml(CONFIG / 'rl_policy.yaml')


def observation(**overrides):
    values = {
        'remaining_distance': 9.0,
        'clearance_margin': 9.0,
        'turn_error': 3.0,
        'speed_fraction': 2.0,
        'goal_clearance_margin': 9.0,
        'yaw_error': 3.0,
    }
    values.update(overrides)
    return RLObservation(**values)


def test_deployed_policy_is_certified_and_falls_back_to_baseline():
    policy = deployed_policy()
    decision = policy.decide(observation(), now_sec=0.0)
    assert decision.learned_override is False
    assert decision.action.speed_scale == 1.0
    assert decision.action.clearance_push == 1.0


def test_deployed_policy_holds_a_decision_for_its_period():
    policy = deployed_policy()
    first = policy.decide(observation(remaining_distance=2.0), now_sec=0.0)
    held = policy.decide(observation(), now_sec=0.10)
    assert held == first
    assert policy.decision_period_sec < 1.0
    later = policy.decide(observation(), now_sec=1.01)
    assert later.state == policy.state(observation())
    assert later.state != first.state


def test_goal_convergence_shield_restores_baseline_action():
    """Learning may not slow or redirect the final arrival."""
    policy = deployed_policy()
    decision = policy.decide(
        observation(
            remaining_distance=0.10, clearance_margin=0.05,
            turn_error=0.20, speed_fraction=0.10, goal_clearance_margin=0.05,
            yaw_error=0.01,
        ),
        now_sec=0.0,
    )
    assert decision.learned_override is False
    assert decision.action.speed_scale == 1.0
    assert decision.action.clearance_push == 1.0


def test_runtime_policy_rejects_unsafe_or_uncertified_models():
    data = yaml.safe_load((CONFIG / 'rl_policy.yaml').read_text(encoding='utf-8'))
    unsafe = deepcopy(data)
    unsafe['actions'][0]['speed_scale'] = 1.01
    with pytest.raises(ValueError, match='speed_scale'):
        CompactRLPolicy(unsafe)
    regressing = deepcopy(data)
    regressing['certification']['paired_regression_passed'] = False
    with pytest.raises(ValueError, match='paired non-regression'):
        CompactRLPolicy(regressing)
    collided = deepcopy(data)
    collided['certification']['evaluation_collisions'] = 1
    with pytest.raises(ValueError, match='collided'):
        CompactRLPolicy(collided)
    stale = deepcopy(data)
    stale['format_version'] = 1
    with pytest.raises(ValueError, match='format'):
        CompactRLPolicy(stale)


def test_runtime_state_layout_matches_the_policy_table():
    data = yaml.safe_load((CONFIG / 'rl_policy.yaml').read_text(encoding='utf-8'))
    assert tuple(data['bins']) == STATE_FIELDS
    width = len(STATE_FIELDS)
    for encoded in data.get('overrides', {}):
        assert len(str(encoded).split(',')) == width


def test_clearance_residual_redirects_without_increasing_speed():
    original = np.asarray([0.6, 0.0])
    adjusted = apply_clearance_residual(
        original,
        yaw=0.0,
        body_clearance=0.02,
        gradient=[0.0, 1.0],
        clearance_push=1.60,
    )
    assert adjusted[1] > 0.0
    assert np.linalg.norm(adjusted) <= np.linalg.norm(original) + 1.0e-12


def test_clearance_residual_is_inert_without_a_learned_push():
    original = np.asarray([0.6, 0.1])
    unchanged = apply_clearance_residual(
        original, yaw=0.3, body_clearance=0.02, gradient=[0.0, 1.0],
        clearance_push=1.0,
    )
    assert unchanged == pytest.approx(original)
    far = apply_clearance_residual(
        original, yaw=0.3, body_clearance=0.50, gradient=[0.0, 1.0],
        clearance_push=1.60,
    )
    assert far == pytest.approx(original)


def test_tracker_baseline_preserves_wall_parallel_progress():
    """Tracker mode supplies the baseline obstacle turn that MPPI already has.

    A small wall-facing component used to make Collision Monitor's approach
    action scale the complete holonomic twist, including the much larger safe
    component parallel to the wall.
    """
    original = np.asarray([-0.08, 0.60])
    adjusted = apply_clearance_residual(
        original,
        yaw=0.0,
        body_clearance=0.02,
        gradient=[1.0, 0.0],
        clearance_push=1.0,
        repulsion_edge=0.06,
        repulsion_authority=0.30,
        include_baseline=True,
    )
    assert adjusted[0] > original[0]
    assert adjusted[1] > 0.5
    assert np.linalg.norm(adjusted) <= np.linalg.norm(original)


def test_tracker_baseline_is_inert_outside_the_repulsion_edge():
    original = np.asarray([-0.08, 0.60])
    adjusted = apply_clearance_residual(
        original,
        yaw=0.0,
        body_clearance=0.061,
        gradient=[1.0, 0.0],
        clearance_push=1.0,
        repulsion_edge=0.06,
        repulsion_authority=0.30,
        include_baseline=True,
    )
    assert adjusted == pytest.approx(original)


def test_tracker_launch_enables_only_the_missing_baseline_response():
    launch = (
        Path(__file__).resolve().parents[1] / 'launch' / 'system.launch.py'
    ).read_text(encoding='utf-8')
    node = (
        Path(__file__).resolve().parents[1]
        / 'omni_autonomy_next' / 'rl_policy_node.py'
    ).read_text(encoding='utf-8')
    assert "'apply_baseline_repulsion': ParameterValue(" in launch
    assert "LaunchConfiguration('tracker'), value_type=bool" in launch
    assert "self.get_parameter('apply_baseline_repulsion').value" in node
    assert "'baseline_wall_repulsion': self.apply_baseline_repulsion" in node


def test_cad_clearance_gradient_points_away_from_nearest_wall():
    field = CadClearanceModel([[[0.0, -1.0], [0.0, 1.0]]])
    clearance, gradient = field.clearance_and_gradient([0.4, 0.2])
    assert clearance == pytest.approx(0.4)
    assert gradient == pytest.approx([1.0, 0.0])


def test_body_clearance_is_yaw_dependent_and_reaches_zero_on_contact():
    """base_link distance says nothing about whether the outline fits.

    The deployed footprint reaches 0.588 m at the corners and 0.510 m at its
    nearest vertex, so a wall 0.45 m from base_link is already inside it.
    """
    footprint = load_footprint(CONFIG / 'competition_footprints.yaml')
    field = CadClearanceModel([[[0.0, -3.0], [0.0, 3.0]]], footprint)
    centre, _ = field.clearance_and_gradient([-0.50, 0.0])
    assert centre == pytest.approx(0.50)
    # +x extent of the outline is 0.451 m, so a 0.50 m base_link distance
    # leaves only about 49 mm of body clearance at yaw 0.
    assert field.body_clearance([-0.50, 0.0], 0.0) == pytest.approx(0.049, abs=2e-3)
    # Rotated by 45 degrees the corner reaches further and touches the wall.
    assert field.body_clearance([-0.50, 0.0], math.radians(45.0)) == 0.0
    assert field.body_clearance([-2.00, 0.0], 0.0) == pytest.approx(0.35)


def test_runtime_and_offline_observations_agree_on_the_tight_lane():
    """The deployed observation must be the one the policy was trained on."""
    footprint = load_footprint(CONFIG / 'competition_footprints.yaml')
    field = CadClearanceModel.from_yaml(
        CONFIG / 'field_planning.yaml', CONFIG / 'competition_footprints.yaml'
    )
    assert field.footprint == pytest.approx(footprint)
    poses = load_configured_poses(CONFIG / 'field_poses.yaml')
    goal = poses['4']
    position = np.asarray([goal['x'], goal['y']])
    goal_clearance = field.body_clearance(position, goal['yaw'])
    assert 0.0 < goal_clearance < 0.060
    built = make_observation(
        position=position,
        yaw=goal['yaw'],
        body_velocity=np.asarray([0.2, 0.0]),
        target=position + np.asarray([0.3, 0.0]),
        goal=position,
        body_clearance=goal_clearance,
        goal_clearance=goal_clearance,
        goal_yaw=goal['yaw'],
        reference_speed=0.78,
    )
    assert built.clearance_margin == pytest.approx(goal_clearance)
    assert built.goal_clearance_margin == pytest.approx(goal_clearance)
    assert built.yaw_error == pytest.approx(0.0)
    assert built.remaining_distance == pytest.approx(0.0)
    assert set(STATE_FIELDS) <= set(vars(built))


def test_all_configured_goal_ids_are_finite_map_poses():
    poses = load_configured_poses(CONFIG / 'field_poses.yaml')
    assert set(poses) == {str(value) for value in range(8)}
    for configured in poses.values():
        assert configured['frame_id'] == 'map'
        assert all(math.isfinite(configured[key]) for key in ('x', 'y', 'yaw'))


def deployed_field():
    return CadClearanceModel.from_yaml(
        CONFIG / 'field_planning.yaml', CONFIG / 'competition_footprints.yaml'
    )


def test_clearance_over_rotation_matches_the_per_yaw_measurement():
    """The swept-rotation pass may not be a different measurement.

    It shares one point-to-wall pass across every candidate yaw, which is what
    makes it affordable inside the control loop, so the equivalence with
    :meth:`body_clearance` is the whole of its correctness.
    """
    field = deployed_field()
    rng = np.random.default_rng(20260904)
    points = np.column_stack((
        rng.uniform(-4.6, -0.4, 60), rng.uniform(-3.2, 4.9, 60),
    ))
    for point in points:
        yaws = rng.uniform(-math.pi, math.pi, 9)
        for cap in (1.0e-3, 0.02, 0.35, 0.95):
            batched = field.clearance_over_rotation(point, yaws, cap=cap)
            per_yaw = [field.body_clearance(point, yaw, cap=cap) for yaw in yaws]
            assert batched == pytest.approx(per_yaw, abs=1.0e-12)
            assert list(batched <= 0.0) == [value <= 0.0 for value in per_yaw]


def test_yaw_rate_limit_is_inert_where_no_wall_is_reachable():
    """A rate the monitor would pass unscaled must come back untouched."""
    field = CadClearanceModel([[[0.0, -3.0], [0.0, 3.0]]], load_footprint(
        CONFIG / 'competition_footprints.yaml'))
    far = [-6.0, 0.0]
    for rate in (1.30, 0.30, -1.30, -0.05):
        assert limit_yaw_rate(
            rate, position=far, yaw=0.0, clearance_model=field, horizon=1.2,
        ) == pytest.approx(rate)
    # No model, no horizon and a zero request are all pass-through.
    assert limit_yaw_rate(
        1.3, position=far, yaw=0.0, clearance_model=None, horizon=1.2) == 1.3
    assert limit_yaw_rate(
        1.3, position=far, yaw=0.0, clearance_model=field, horizon=0.0) == 1.3
    assert limit_yaw_rate(
        0.0, position=far, yaw=0.0, clearance_model=field, horizon=1.2) == 0.0


def test_yaw_rate_limit_only_ever_reduces_and_keeps_its_sign():
    field = deployed_field()
    poses = load_configured_poses(CONFIG / 'field_poses.yaml')
    for configured in poses.values():
        position = np.asarray([configured['x'], configured['y']])
        for request in (1.30, 0.45, -1.30, -0.45):
            limited = limit_yaw_rate(
                request, position=position, yaw=configured['yaw'],
                clearance_model=field, horizon=1.2,
            )
            assert abs(limited) <= abs(request) + 1.0e-12
            assert limited == 0.0 or math.copysign(1.0, limited) == math.copysign(
                1.0, request)


def test_yaw_rate_limit_yields_a_sweep_collision_monitor_passes_unscaled():
    """The point of the limit: the projection the monitor runs stays clear.

    ``FootprintApproach`` holds the commanded angular velocity constant for
    ``time_before_collision`` and scales the whole twist, translation with it,
    by ``contact_time / horizon`` on any predicted contact.  A limited rate
    whose own sweep still collided would buy nothing.
    """
    horizon = load_collision_monitor_horizon(CONFIG / 'nav2_next.yaml')
    field = deployed_field()
    poses = load_configured_poses(CONFIG / 'field_poses.yaml')
    tight = []
    for key, configured in sorted(poses.items()):
        position = np.asarray([configured['x'], configured['y']])
        yaw = configured['yaw']
        if field.body_clearance(position, yaw) <= 0.0:
            continue
        for request in (1.30, -1.30):
            limited = limit_yaw_rate(
                request, position=position, yaw=yaw,
                clearance_model=field, horizon=horizon)
            # A rotation is always available: a configured pose that fits has
            # some headroom, so the base is never left unable to turn at all.
            assert abs(limited) > 0.0
            if abs(limited) < abs(request):
                tight.append(key)
            swept = yaw + limited * np.linspace(
                horizon / 24.0, horizon, 24)
            assert np.all(field.clearance_over_rotation(
                position, swept, cap=1.0e-3) > 0.0)
    # The tight firing poses are exactly where this binds; if it stopped
    # binding there the crawl it removes would be back.
    assert {'1', '4', '5', '7'} <= set(tight)


def test_collision_monitor_horizon_comes_from_the_deployed_parameters():
    horizon = load_collision_monitor_horizon(CONFIG / 'nav2_next.yaml')
    monitor = yaml.safe_load(
        (CONFIG / 'nav2_next.yaml').read_text(encoding='utf-8')
    )['collision_monitor']['ros__parameters']
    assert horizon == pytest.approx(
        monitor['FootprintApproach']['time_before_collision'])
    assert monitor['FootprintApproach']['action_type'] == 'approach'
