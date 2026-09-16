"""Prevent command saturation or sparse odometry from proving a speed target."""
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT/'scripts'))
from audit_speed_target import displacement_peak, inspect, target_budget, profile_budgets


def test_profile_budget_separates_acceleration_and_braking():
    import json
    robot = dict(drivetrain=dict(linear_command_scale=.38,
                 profile_linear_accelerations={'sprint': 3.}),
                 calibrated_tracking=dict(feedback_delay_sec=.32))
    nav = {'velocity_smoother': {'ros__parameters': {
        'max_accel': [.85, .85, 1.], 'max_decel': [-.85, -.85, -1.]}}}
    runtime = {'runtime_guard': {'ros__parameters': {'profiles_json': json.dumps({
        'sprint': dict(linear=4., lateral=4., linear_accel=3.3),
        'balanced': dict(linear=2., lateral=2., linear_accel=2.1)})}}}
    budgets = profile_budgets(4., robot, nav, runtime)
    sprint = budgets['sprint']
    assert sprint['acceleration_m_s2'] == 3.
    assert budgets['balanced']['acceleration_m_s2'] == .85
    assert sprint['braking_deceleration_m_s2'] == pytest.approx(.6375)
    assert sprint['modeled_accel_and_brake_distance_m'] == pytest.approx(16.7356862745)
    axis, diagonal = sprint['directions_without_yaw'][:2]
    assert diagonal['command_equivalent_speed_m_s'] == pytest.approx(
        axis['command_equivalent_speed_m_s']/np.sqrt(2))
    assert all(d['transmitted_wheel_peak'] <= 10000 for d in sprint['directions_without_yaw'])
    robot['drivetrain']['profile_linear_accelerations']['sprint'] = 10.
    assert profile_budgets(4., robot, nav, runtime)['sprint']['acceleration_m_s2'] == 3.3


def test_six_mps_request_exceeds_current_uart_budget():
    result = target_budget(6., {'linear_command_scale': .38}, .85, .6375, .38)
    assert result['requested_axis_uart_units'] == pytest.approx(16420.56)
    assert result['uart_scale'] < 1.
    assert result['transmitted_axis_uart_peak'] == 10000
    assert result['modeled_accel_and_brake_distance_m'] == pytest.approx(51.6917647059)
    assert result['required_wire_to_speed_gain'] == pytest.approx(4.3212)


def test_contiguous_displacement_ignores_raw_twist_spike():
    t = np.arange(0., 1.01, .02)
    wheel = np.zeros((len(t), 6))
    wheel[:, 0] = t
    wheel[:, 4] = t
    wheel[10, 1] = 30.  # An instantaneous twist spike is not a 30 m/s run.
    assert displacement_peak(wheel)['speed_m_s'] == pytest.approx(1.)
    assert displacement_peak(wheel[[0, 25, 50]]) is None


def test_demo_cannot_be_used_as_hardware_evidence(tmp_path):
    (tmp_path/'manifest.json').write_text('{"settings":{"operation_mode":"demo"}}')
    with pytest.raises(ValueError, match='hardware recording required'):
        inspect(tmp_path, 3.)


def test_saturated_replay_scales_translation_and_yaw_together():
    from compare_field_response import transmitted_response
    wheels, velocity, saturation = transmitted_response(np.array([3., 2., 1.]), [2.6, 2.5, 2.8])
    assert max(abs(v) for v in wheels) == 10000
    assert 0. < saturation < 1.
    assert velocity == pytest.approx(np.array([7.8, 5., 2.8])*saturation)
    # The old plant ignored saturation and claimed this impossible response.
    assert velocity[0] < 3.


def test_unsaturated_replay_retains_fitted_response():
    from compare_field_response import transmitted_response
    _, velocity, saturation = transmitted_response(np.array([.2, -.1, .05]), [2.6, 2.5, 2.8])
    assert saturation == 1.
    assert velocity == pytest.approx([.52, -.25, .14])


@pytest.mark.parametrize('target', [float('nan'), float('inf'), 0., -1.])
def test_invalid_speed_targets_rejected(target):
    with pytest.raises(ValueError):
        target_budget(target, {'linear_command_scale': .38}, .85, .6375, .38)
