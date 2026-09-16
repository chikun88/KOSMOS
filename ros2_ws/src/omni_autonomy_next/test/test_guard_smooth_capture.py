"""Normal speed transitions release acceleration before crossing the target."""
import math
import numpy as np
import pytest
from test_runtime_guard import make_guard, ACTIVE


@pytest.mark.parametrize('direction', [0., math.pi/4, math.pi/2, math.pi])
@pytest.mark.parametrize('dt', [.005, .01, .02])
def test_cruise_capture_has_bounded_jerk_including_last_tick(direction, dt):
    guard = make_guard()
    target = np.array([.4*math.cos(direction), .4*math.sin(direction), .5])
    accelerations = [np.zeros(3)]
    previous = np.zeros(3)
    for i in range(350):
        result = guard.step(target, now_sec=i*dt, command_age_sec=0., health=ACTIVE,
            profile='balanced', user_scale=1., red_zone=False)
        velocity = np.array(result.velocity)
        elapsed = .01 if i == 0 else dt
        acceleration = (velocity-previous)/elapsed
        jerk = (acceleration-accelerations[-1])/elapsed
        assert np.linalg.norm(jerk[:2]) <= 2.8+1.e-7
        assert abs(jerk[2]) <= 5.5+1.e-7
        assert np.dot(velocity[:2], target[:2]) <= np.dot(target[:2],target[:2])+1.e-9
        accelerations.append(acceleration)
        previous = velocity
    assert previous == pytest.approx(target, abs=1.e-8)


def test_emergency_zero_still_bypasses_smoothing():
    guard = make_guard()
    for i in range(100):
        guard.step((.4, .1, .2), now_sec=i*.01, command_age_sec=0., health=ACTIVE,
            profile='balanced', user_scale=1., red_zone=False)
    result = guard.step((0., 0., 0.), now_sec=1., command_age_sec=0., health=ACTIVE,
        profile='balanced', user_scale=1., red_zone=False)
    assert result.velocity == result.acceleration == (0., 0., 0.)
