"""Hypothetical caps must reach the real limiter and never leak between trials."""
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT/'scripts'))
import check_speed_target_replay as sweep
from omni_autonomy_next.runtime_guard import RuntimeGuard, GuardHealth
from omni_autonomy_next import robomas_uart


def case(cap):
    return dict(target=6., uart_limit=cap, gains=[2.7, 2.6, 2.8],
                calibration=[.38, .34], delay=.2, tau=.12, direction=0.,
                tuning={}, distance=60., duration=60., wheel_radius=.05)


@pytest.mark.parametrize('cap', [8000, 17000])
def test_trial_reaches_candidate_guard_and_uart_then_restores(monkeypatch, cap):
    original_factory = sweep.plant.make_guard
    original_cap = robomas_uart.AUTO_WHEEL_LIMIT

    def probe(*args, **kwargs):
        guard = sweep.plant.make_guard(RuntimeGuard)
        guard.max_wheel_speed = kwargs['wheel_limit']
        for i in range(800):
            result = guard.step((6., 0., 0.), now_sec=i*.01, command_age_sec=0.,
                health=GuardHealth(True, False, True, True, True),
                profile='balanced', user_scale=1., red_zone=False)
        assert result.velocity[0] == pytest.approx(6.)
        assert kwargs['bridge_limits'][0] == 6.
        wheels, scale = robomas_uart.mix_velocity(result.velocity[0]*.38, 0., 0.)
        assert max(map(abs, wheels)) == min(cap, 16421)
        assert (scale < 1.) == (cap < 16420.56)
        return {'probe': True}

    monkeypatch.setattr(sweep.plant, 'replay', probe)
    assert sweep.evaluate(case(cap))['result']['probe']
    assert sweep.plant.make_guard is original_factory
    assert robomas_uart.AUTO_WHEEL_LIMIT == original_cap


def test_failed_trial_restores_existing_uart_cap(monkeypatch):
    original_factory = sweep.plant.make_guard
    original_cap = robomas_uart.AUTO_WHEEL_LIMIT
    def fail(*args, **kwargs):
        assert robomas_uart.AUTO_WHEEL_LIMIT == 17000
        raise RuntimeError('trial failed')
    monkeypatch.setattr(sweep.plant, 'replay', fail)
    with pytest.raises(RuntimeError, match='trial failed'):
        sweep.evaluate(case(17000))
    assert robomas_uart.AUTO_WHEEL_LIMIT == original_cap
    assert sweep.plant.make_guard is original_factory
