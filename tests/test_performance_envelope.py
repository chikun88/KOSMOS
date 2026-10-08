"""Keep operator-facing speed estimates aligned with the deployed command path."""
import importlib.util
import math
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    'performance_envelope', Path(__file__).parents[1] / 'scripts/performance_envelope.py')
envelope = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(envelope)


def test_trapezoid_uses_independent_acceleration_and_braking():
    # 2 m acceleration + 4 m cruise + 4 m braking = 10 m.
    # Their durations are 2 + 2 + 4 = 8 s.
    assert envelope.trapezoid_time(10., 2., 1., .5) == pytest.approx(8.)
    # Half the distance needed to reach 2 m/s: peak sqrt(2), then brake.
    assert envelope.trapezoid_time(3., 2., 1., .5) == pytest.approx(3*math.sqrt(2))
    assert envelope.trapezoid_time(0., 2., 1., .5) == 0.


@pytest.mark.parametrize('values', [(-1., 2., 1., .5), (1., 0., 1., .5),
    (1., 2., 0., .5), (1., 2., 1., 0.), (1., 2., 1., math.nan)])
def test_invalid_kinematic_estimates_are_rejected(values):
    with pytest.raises(ValueError):
        envelope.trapezoid_time(*values)


def test_report_uses_sprint_budget_tracker_ramps_and_actual_uart_cap(capsys):
    drive = envelope.load('robot.yaml')['robot']['drivetrain']
    runtime = envelope.load('runtime.yaml')['runtime_guard']['ros__parameters']
    smoother = envelope.load('nav2_next.yaml')['velocity_smoother']['ros__parameters']
    limits = envelope.profile_envelopes(drive, runtime, smoother)
    assert limits['precision']['acceleration'] == .85
    assert limits['balanced']['acceleration'] == 1.8
    assert limits['balanced']['axis_speed'] == pytest.approx(2.)
    sprint = limits['sprint']
    assert sprint['axis_speed'] == pytest.approx(3.5)
    assert sprint['wheel_limit'] > limits['balanced']['wheel_limit']
    assert sprint['acceleration'] == 3.3
    assert sprint['deceleration'] == .85
    assert sprint['uart_peak'] == 9579
    assert sprint['uart_scale'] == 1.
    assert envelope.main() == 0
    report = capsys.readouterr().out
    assert 'sprint 前進: 3.500 m/s' in report
    assert '9579 / 10000' in report
    assert '7.206 m' in report
    assert '9.062 m' in report
