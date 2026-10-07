import json
import math
from pathlib import Path

import numpy as np
import pytest

from simulation.body_model import BodyClearanceModel
from simulation.dynamics import (
    ControlAdjustment,
    ControlObservation,
    EpisodeResult,
    SimProfile,
    simulate_episode,
)
from simulation.reinforcement_learning import (
    CompactDeploymentPolicy,
    SafeQLearningPolicy,
    export_compact_policy,
    paired_regression_report,
    performance_non_regression_passed,
)


CONFIG = (
    Path(__file__).resolve().parents[1]
    / 'ros2_ws' / 'src' / 'omni_autonomy_next' / 'config'
)


def observation(**overrides):
    values = {
        'remaining_distance': 1.0,
        'clearance_margin': 0.08,
        'turn_error': 0.2,
        'speed_fraction': 0.5,
        'goal_clearance_margin': 0.25,
        'yaw_error': 0.10,
    }
    values.update(overrides)
    return ControlObservation(**values)


def test_unseen_state_uses_fast_action_without_exceeding_limits():
    policy = SafeQLearningPolicy()
    action = policy.select_adjustment(observation())
    assert action.speed_scale == 1.0
    assert 1.0 <= action.clearance_push <= 2.0


def test_state_covers_every_binned_feature():
    """A feature that is binned but not read would be silently ignored.

    The tight-lane features were added for exactly this policy, so a state
    vector that quietly drops them would train against the wrong observation.
    """
    policy = SafeQLearningPolicy()
    assert len(policy._state(observation())) == len(policy.bins)
    tight = policy._state(observation(goal_clearance_margin=0.05))
    open_goal = policy._state(observation(goal_clearance_margin=0.30))
    assert tight != open_goal


def test_terminal_q_update_uses_learning_rate():
    policy = SafeQLearningPolicy(training=True, epsilon=0.0, learning_rate=0.25)
    policy.begin_episode()
    state = policy._state(observation())
    policy.select_adjustment(observation())
    chosen = len(policy.actions) - 1
    policy.observe_transition(8.0, observation(), terminal=True)
    assert policy.q_values[(*state, chosen)] == pytest.approx(2.0)
    assert policy.visit_counts[(*state, chosen)] == 1


def test_model_round_trip_preserves_policy(tmp_path):
    policy = SafeQLearningPolicy(training=True, epsilon=0.0)
    policy.begin_episode()
    policy.select_adjustment(observation())
    policy.observe_transition(3.0, observation(), terminal=True)
    path = tmp_path / 'policy.json'
    policy.save(path, {'purpose': 'test'})

    restored = SafeQLearningPolicy.load(path)
    assert np.array_equal(restored.q_values, policy.q_values)
    assert np.array_equal(restored.visit_counts, policy.visit_counts)
    assert restored.select_adjustment(observation()) == policy.select_adjustment(observation())


def test_loading_policy_rejects_action_above_safety_envelope(tmp_path):
    data = SafeQLearningPolicy().to_dict()
    data['actions'][0]['speed_scale'] = 1.01
    path = tmp_path / 'unsafe.json'
    path.write_text(json.dumps(data), encoding='utf-8')
    with pytest.raises(ValueError, match='may not raise'):
        SafeQLearningPolicy.load(path)


def _passing_metadata():
    return {
        'paired_regression_passed': True,
        'evaluation_collisions': 0,
        'safety_campaign_passed': False,
        'baseline_failures': 6,
        'learned_failures': 2,
        'held_out_seed': 2,
        'evaluation_episodes': 10,
        'training': {'seed': 1, 'episodes': 10, 'critical_pose_ids': ['4']},
    }


def test_only_a_non_regressing_model_can_be_exported_for_ros(tmp_path):
    policy = SafeQLearningPolicy()
    model = tmp_path / 'model.json'
    deployed = tmp_path / 'deployed.yaml'
    policy.save(model, {'paired_regression_passed': False})
    with pytest.raises(ValueError, match='regresses a scenario'):
        export_compact_policy(model, deployed)

    policy.save(model, {**_passing_metadata(), 'evaluation_collisions': 1})
    with pytest.raises(ValueError, match='collided'):
        export_compact_policy(model, deployed)

    policy.save(model, _passing_metadata())
    compact = export_compact_policy(model, deployed)
    assert deployed.is_file()
    assert compact['certification']['paired_regression_passed'] is True
    assert compact['certification']['evaluation_collisions'] == 0
    fallback = compact['actions'][compact['default_action']]
    assert fallback == {'speed_scale': 1.0, 'clearance_push': 1.0}


def test_nonempty_model_actual_velocity_table_cannot_export_as_runtime_command_table(tmp_path):
    policy = SafeQLearningPolicy()
    state = policy._state(observation())
    policy.q_values[(*state, 0)] = 10.
    policy.visit_counts[(*state, 0)] = policy.minimum_eval_visits
    model = tmp_path / 'model.json'
    deployed = tmp_path / 'deployed.yaml'
    policy.save(model, {**_passing_metadata(), 'observation_context': {
        'velocity_source': 'model_actual_velocity', 'reference_speed_mps': 4.,
        'reference_profile': 'sprint'}})
    with pytest.raises(ValueError, match='observation context'):
        export_compact_policy(model, deployed)
    assert not deployed.exists()


def _result(success, elapsed):
    return EpisodeResult(
        success=success, collision=False, timeout=not success, aborted=False,
        final_position_error=0.02, final_yaw_error=0.01, elapsed=elapsed,
        path_length=4.0, minimum_clearance=0.04, braked_fraction=0.1,
        command_reversals=2,
    )


def test_promotion_allows_trading_speed_for_arrival_but_not_regression():
    """The deterministic controller does not arrive on the bucket-lane transits.

    An aggregate mean-time rule rejected exactly the slower-but-arriving
    behaviour that fixes them, so the gate is paired per scenario instead.
    """
    baseline = [_result(True, 8.0), _result(False, 30.0)]
    rescuing = [_result(True, 8.2), _result(True, 26.0)]
    report = paired_regression_report(baseline, rescuing)
    assert report['passed']
    assert report['rescued_scenarios'] == [1]
    assert report['regressed_scenarios'] == []

    # A solved scenario may not become much slower ...
    assert not performance_non_regression_passed(
        baseline, [_result(True, 40.0), _result(True, 26.0)]
    )
    # ... nor may it stop arriving, however much else improves.
    assert not performance_non_regression_passed(
        baseline, [_result(False, 8.0), _result(True, 26.0)]
    )


def test_paired_gate_requires_matching_scenario_counts():
    with pytest.raises(ValueError, match='identical scenario counts'):
        paired_regression_report([_result(True, 8.0)], [])


def test_compact_deployment_policy_enforces_goal_convergence_shield():
    policy = CompactDeploymentPolicy.load(CONFIG / 'rl_policy.yaml')
    policy.begin_episode()
    action = policy.select_adjustment(observation(
        remaining_distance=0.10, clearance_margin=0.05, speed_fraction=0.10,
        goal_clearance_margin=0.05,
    ))
    assert action == ControlAdjustment(1.0, 1.0)


def test_deployed_policy_state_layout_matches_the_offline_learner():
    """The runtime table is indexed by the offline state vector, in order.

    A mismatch would silently apply learned actions to unrelated situations.
    The ROS package is not importable from here, so the declaration is read
    from source.
    """
    import ast

    from simulation.reinforcement_learning import DEFAULT_BINS

    source = (
        CONFIG.parent / 'omni_autonomy_next' / 'rl_policy.py'
    ).read_text(encoding='utf-8')
    runtime_fields = None
    for node in ast.parse(source).body:
        targets = getattr(node, 'targets', [])
        if targets and getattr(targets[0], 'id', None) == 'STATE_FIELDS':
            runtime_fields = ast.literal_eval(node.value)
    assert runtime_fields == tuple(DEFAULT_BINS)


def test_tight_firing_poses_have_less_body_clearance_than_the_goal_tolerance():
    """The measurement that makes this whole failure mode inevitable.

    Every configured firing pose beside the fixed bucket has under 60 mm of
    footprint clearance, while nav2's own goal checker accepts a 40 mm position
    error, so an in-tolerance arrival can already be touching the field.
    """
    import yaml

    body = BodyClearanceModel.from_yaml(
        CONFIG / 'field_planning.yaml', CONFIG / 'competition_footprints.yaml'
    )
    poses = yaml.safe_load(
        (CONFIG / 'field_poses.yaml').read_text(encoding='utf-8')
    )['poses']
    for goal_id in (4, 5, 6, 7):
        pose = poses[goal_id]
        clearance = body.clearance([pose['x'], pose['y']], pose['yaw'])
        assert 0.0 < clearance < 0.060, (goal_id, clearance)
    assert body.radius > 0.58
    assert body.inscribed_radius > 0.41


class OpenField:
    """Minimal stand-in with no wall anywhere near the robot."""

    planning_margin = 0.035

    class _Body:
        radius = 0.588

    body = _Body()

    def body_clearance(self, _point, _yaw):
        return 10.0

    def body_clearance_batch(self, points, _yaws, cap=None):
        return np.full(len(points), 10.0)

    def clearance_and_gradient(self, _point):
        return 10.0, np.zeros(2)

    def planning_clearance(self, _point, _yaw):
        return 10.0

    def segment_safe(self, _start, _end, _yaw, margin=None):
        return True


class UnsafeController:
    def begin_episode(self):
        pass

    def select_adjustment(self, _observation):
        return ControlAdjustment(speed_scale=1.1, clearance_push=1.0)

    def observe_transition(self, _reward, _next_observation, _terminal):
        pass


def test_simulator_enforces_policy_speed_envelope():
    with pytest.raises(ValueError, match='speed_scale'):
        simulate_episode(
            OpenField(),
            np.asarray([[0.0, 0.0], [0.5, 0.0]]),
            0.0,
            0.0,
            SimProfile(),
            np.random.default_rng(1),
            controller=UnsafeController(),
        )
