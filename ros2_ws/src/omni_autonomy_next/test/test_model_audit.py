"""Regression cases for continuous geometry and delayed-control evidence."""
import math

import numpy as np
import pytest

from omni_autonomy_next.motion_prediction import DelayedMotionPredictor
from omni_autonomy_next.rl_policy import CompactRLPolicy, RLObservation, STATE_FIELDS
from omni_autonomy_next.rl_residual import CadClearanceModel


SQUARE = np.array([[-.5, -.5], [.5, -.5], [.5, .5], [-.5, .5]])


@pytest.mark.parametrize('direction', [1., -1.])
def test_rotation_headroom_certifies_contacts_between_clear_samples(direction):
    wall_x = math.sqrt(.5) - 1.e-6
    field = CadClearanceModel([[[wall_x, -2.], [wall_x, 2.]]], SQUARE)
    span = direction * 1.56
    # All 24 poses fit, but the intervening corner at 45 degrees crosses.
    sampled = span * np.arange(1, 25) / 24
    assert np.all(field.clearance_over_rotation([0., 0.], sampled) > 0.)
    assert field.body_clearance([0., 0.], direction * math.pi / 4) == 0.
    limited = field.rotation_headroom([0., 0.], 0., span, samples=24)
    assert 0. < abs(limited) < math.pi / 4
    assert np.all(field.clearance_over_rotation(
        [0., 0.], np.linspace(0., limited, 1001)) > 0.)


def test_rotation_headroom_cannot_certify_a_pose_already_in_contact():
    field = CadClearanceModel([[[.49, .49], [.491, .49]]], SQUARE)
    assert field.body_clearance([0., 0.], 0.) == 0.
    assert field.rotation_headroom([0., 0.], 0., .7) == 0.


@pytest.mark.parametrize('polygon', [SQUARE, SQUARE[::-1]])
def test_enclosed_wall_contact_is_independent_of_polygon_winding(polygon):
    field = CadClearanceModel([[[-.05, 0.], [.05, 0.]]], polygon)
    assert field.body_clearance([0., 0.], 0.) == 0.
    assert field.clearance_over_poses([[0., 0.]], [0.])[0] == 0.


@pytest.mark.parametrize('polygon', [
    [[0., 0.], [0., 0.], [1., 1.]],
    [[0., 0.], [1., 0.], [2., 0.]],
    [[0., 0.], [1., 0.], [.2, .2], [0., 1.]],
])
def test_degenerate_and_concave_footprints_cannot_certify_clearance(polygon):
    with pytest.raises(ValueError, match='footprint'):
        CadClearanceModel([[[0., -2.], [0., 2.]]], polygon)


@pytest.mark.parametrize('cap', [math.nan, math.inf, 0., -1.])
def test_invalid_caps_cannot_produce_nan_clearance_that_evades_contact_checks(cap):
    field = CadClearanceModel([[[0., -2.], [0., 2.]]], SQUARE)
    with pytest.raises(ValueError, match='cap'):
        field.body_clearance([2., 0.], 0., cap=cap)
    with pytest.raises(ValueError, match='cap'):
        field.clearance_over_poses([[2., 0.]], [0.], cap=cap)


def test_partial_delayed_history_preserves_measured_motion_before_first_command():
    model = DelayedMotionPredictor()
    model.record(1., [3., 0., 0.])
    pose, velocity = model.predict(np.zeros(3), [3., 0., 0.], 1.)
    assert pose == pytest.approx([3. * model.horizon, 0., 0.])
    assert velocity == pytest.approx([3., 0., 0.])


def _policy_data():
    return {
        'format_version': 2, 'algorithm': 'tabular_q_learning_greedy',
        'certification': {'paired_regression_passed': True,
                          'evaluation_collisions': 0},
        'bins': {name: [.5] for name in STATE_FIELDS},
        'actions': [{'speed_scale': 1., 'clearance_push': 1.},
                    {'speed_scale': .5, 'clearance_push': 1.5}],
        'default_action': 0,
        'overrides': {'1,0,0,0,0,0': 1},
        'observation_context': {'velocity_source': 'smoothed_command',
                                'reference_speed_mps': 1., 'reference_profile': 'test'},
    }


def _policy():
    data = _policy_data()
    return CompactRLPolicy(data, observation_context=data['observation_context'])


def test_nonempty_policy_cannot_load_without_matched_runtime_observations():
    data = _policy_data()
    with pytest.raises(ValueError, match='observation context'):
        CompactRLPolicy(data)
    expected = dict(data['observation_context'])
    data['observation_context']['velocity_source'] = 'model_actual_velocity'
    with pytest.raises(ValueError, match='observation context'):
        CompactRLPolicy(data, observation_context=expected)


def _observation(**overrides):
    values = {name: .01 for name in STATE_FIELDS}
    values['remaining_distance'] = .8
    values.update(overrides)
    return RLObservation(**values)


def test_goal_shield_restores_baseline_even_inside_held_decision_period():
    policy = _policy()
    assert policy.decide(_observation(), 0.).learned_override
    final = policy.decide(_observation(remaining_distance=.1), .1)
    assert not final.learned_override
    assert final.action.speed_scale == 1.
    assert final.action.clearance_push == 1.


def test_invalid_observation_is_rejected_even_inside_held_decision_period():
    policy = _policy()
    policy.decide(_observation(), 0.)
    with pytest.raises(ValueError, match='finite'):
        policy.decide(_observation(yaw_error=math.nan), .1)


@pytest.mark.parametrize('now', [math.nan, math.inf, -math.inf])
def test_policy_cannot_hold_or_schedule_an_invalid_decision_clock(now):
    with pytest.raises(ValueError, match='time'):
        _policy().decide(_observation(), now)


def test_clock_rollback_cannot_hold_a_decision_from_the_future():
    policy = _policy()
    assert policy.decide(_observation(), 10.).learned_override
    assert not policy.decide(_observation(speed_fraction=.9), 9.).learned_override
