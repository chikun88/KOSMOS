import math
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest
import yaml

from simulation.body_model import BodyClearanceModel
from simulation.dynamics import SimProfile, _approach_scale, _yaw_rate_gate, _project_constant_twist
from simulation.run_campaign import campaign_passed, prepare_campaign
from simulation.test_run_campaign import passing_report
from simulation.reinforcement_learning import CompactDeploymentPolicy
from simulation.dynamics import ControlObservation


SQUARE = np.array([[-.5, -.5], [.5, -.5], [.5, .5], [-.5, .5]])


@pytest.mark.parametrize('polygon', [SQUARE, SQUARE[::-1]])
def test_offline_enclosed_wall_contact_is_independent_of_winding(polygon):
    body = BodyClearanceModel([[[-.05, 0.], [.05, 0.]]], polygon)
    assert body.clearance([0., 0.], 0.) == 0.
    assert body.clearance_batch([[0., 0.]], [0.])[0] == 0.


def test_offline_projection_rejects_intervening_corner_contact():
    wall_x = math.sqrt(.5) - 1.e-6
    body = BodyClearanceModel([[[wall_x, -2.], [wall_x, 2.]]], SQUARE)
    field = SimpleNamespace(body=body, body_clearance_batch=body.clearance_batch)
    profile = SimProfile()
    rate = 1.56 / profile.approach_horizon_sec
    clearance = body.clearance([0., 0.], 0.)
    limited = _yaw_rate_gate(field, np.zeros(2), 0., rate, profile, clearance)
    assert 0. < limited * profile.approach_horizon_sec < math.pi / 4
    assert _approach_scale(
        field, np.zeros(2), 0., np.zeros(2), rate, profile, clearance) < 1.


def test_offline_swept_translation_cannot_skip_a_thin_wall():
    # Both ends of the horizon are clear; the narrow body crosses mid-step.
    tiny = SQUARE * .001
    body = BodyClearanceModel([[[.05, -2.], [.05, 2.]]], tiny)
    field = SimpleNamespace(body=body, body_clearance_batch=body.clearance_batch)
    profile = SimProfile(approach_horizon_sec=.1, approach_step_sec=.1)
    scale = _approach_scale(field, np.zeros(2), 0., np.array([1., 0.]),
                            0., profile, body.clearance([0., 0.], 0.))
    assert 0. < scale < .5


def test_collision_projection_follows_held_body_twist_arc():
    points = _project_constant_twist([0., 0.], [0., 1.], 1., [math.pi / 2])
    assert points[0] == pytest.approx([-1., 1.])
    straight = _project_constant_twist([0., 0.], [0., 1.], 0., [2.])
    assert straight[0] == pytest.approx([0., 2.])
    body = BodyClearanceModel([[[-.5, -5.], [-.5, 5.]]], SQUARE * .001)
    field = SimpleNamespace(body=body, body_clearance_batch=body.clearance_batch)
    profile = SimProfile(approach_horizon_sec=2.)
    scale = _approach_scale(field, np.zeros(2), 0., np.array([0., 1.]),
                            1., profile, body.clearance([0., 0.], 0.))
    assert scale < 1.


@pytest.mark.parametrize('polygon', [[], [[0., 0.]] * 3,
                                    [[0., 0.], [1., 0.], [2., 0.]]])
def test_offline_clearance_rejects_degenerate_footprints(polygon):
    with pytest.raises(ValueError, match='footprint'):
        BodyClearanceModel([[[0., -2.], [0., 2.]]], polygon)


@pytest.mark.parametrize('metric', ['p95_position_error_m', 'p95_yaw_error_deg',
                                  'minimum_clearance_m'])
def test_campaign_nan_metrics_cannot_pass(metric):
    report = passing_report()
    report['summary'][metric] = math.nan
    assert not campaign_passed(report)


def test_campaign_cannot_pass_without_any_tested_scenarios():
    report = passing_report()
    for summary in report.values():
        summary['episodes'] = summary['successes'] = 0
    assert not campaign_passed(report)


def test_legacy_report_without_progress_abort_evidence_cannot_pass():
    report = passing_report()
    del report['summary']['progress_aborts']
    assert not campaign_passed(report)


def test_campaign_uses_profile_arrival_tolerance():
    report = passing_report()
    report['profile'] = {'xy_goal_tolerance': .02, 'yaw_goal_tolerance': .02}
    assert not campaign_passed(report)


@pytest.mark.parametrize('episodes,random', [(0, 0), (-1, 0), (1, 2), (1, -1)])
def test_invalid_campaign_counts_are_rejected_before_source_access(episodes, random):
    with pytest.raises(ValueError):
        prepare_campaign('/missing-field', '/missing-poses', episodes, random, 0)


def test_random_campaign_rejects_start_that_only_fits_at_goal_yaw(monkeypatch, tmp_path):
    from simulation import run_campaign as module
    body = BodyClearanceModel([[[.6, -2.], [.6, 2.]]], SQUARE)
    calls = [0]

    def random_point(*_args, **_kwargs):
        calls[0] += 1
        return np.array([0., 0.]) if calls[0] % 2 else np.array([-3., 0.])

    field = SimpleNamespace(
        body=body, planning_margin=.035, body_clearance=body.clearance,
        random_free_point=random_point,
        plan=lambda start, goal, yaw: np.array([start, goal]))
    angles = iter([0., math.pi / 4, 0., 0.])
    rng = SimpleNamespace(uniform=lambda *_args: next(angles),
                          integers=lambda *_args: 123)
    monkeypatch.setattr(module.np.random, 'default_rng', lambda _seed: rng)
    monkeypatch.setattr(module.GridField, 'from_yaml', lambda *_args: field)
    monkeypatch.setattr(module, '_load_poses', lambda _path: {'1': np.array([-1., 0., 0.])})
    paths = [tmp_path / name for name in ('field.yaml', 'poses.yaml', 'footprint.yaml')]
    for path in paths:
        path.write_text('test source')
    _field, scenarios, _metadata = prepare_campaign(paths[0], paths[1], 1, 1, 0,
                                                    footprint_file=paths[2])
    path, start_yaw, _goal_yaw, _seed = scenarios[0]
    assert start_yaw == 0.
    assert body.clearance(path[0], start_yaw) >= field.planning_margin
    assert calls[0] == 4  # The first candidate had an initially colliding yaw.


def _compact_data():
    path = (Path(__file__).resolve().parents[1] / 'ros2_ws' / 'src'
            / 'omni_autonomy_next' / 'config' / 'rl_policy.yaml')
    return yaml.safe_load(path.read_text())


def _observation(remaining=1.):
    return ControlObservation(remaining, .08, .2, .5, .25, .1)


def _learned_adapter(period=.25):
    data = _compact_data()
    obs = _observation()
    encoded = ','.join(str(int(np.digitize(getattr(obs, name), edges)))
                       for name, edges in data['bins'].items())
    data['overrides'] = {encoded: 0}
    data['decision_period_sec'] = period
    data['observation_context'] = {'velocity_source': 'model_actual_velocity',
                                   'reference_speed_mps': 4., 'reference_profile': 'sprint'}
    return CompactDeploymentPolicy(data)


def test_compact_simulator_hold_period_matches_runtime_deadline_ticks():
    policy = _learned_adapter(.12)
    action = policy.select_adjustment(_observation())
    assert action.speed_scale < 1.
    for _ in range(2):
        policy.observe_transition(0., _observation(), False)
        assert policy.select_adjustment(_observation(remaining=3.)) == action
    policy.observe_transition(0., _observation(), False)
    assert policy.select_adjustment(_observation(remaining=3.)).speed_scale == 1.


def test_compact_simulator_goal_shield_matches_runtime_held_override():
    policy = _learned_adapter()
    assert policy.select_adjustment(_observation()).speed_scale < 1.
    assert policy.select_adjustment(_observation(remaining=.1)).speed_scale == 1.


@pytest.mark.parametrize('mutate', [
    lambda data: data.update(default_action=-1),
    lambda data: data.update(overrides={'0,0': 0}),
    lambda data: data.update(overrides={'0,0,0,0,0,0': 999}),
    lambda data: data['bins']['clearance_margin'].append(math.nan),
    lambda data: data.update(decision_period_sec=math.nan),
])
def test_compact_simulator_rejects_policies_runtime_cannot_deploy(mutate):
    data = _compact_data()
    mutate(data)
    with pytest.raises(ValueError):
        CompactDeploymentPolicy(data)
