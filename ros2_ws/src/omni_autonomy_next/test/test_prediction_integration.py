"""Independent integration oracles for fast, delayed body-frame motion."""
import math

import numpy as np
import pytest

from omni_autonomy_next.motion_prediction import DelayedMotionPredictor


@pytest.mark.parametrize('twist', [(3.5, 0., 1.6), (0., 3.5, -1.6),
                                  (2., -1., .8), (3.5, 0., 0.)])
def test_constant_twist_prediction_matches_exact_se2_motion(twist):
    model = DelayedMotionPredictor()
    pose = np.array([1., -2., .7])
    predicted, velocity = model.predict(pose, twist, 10.)
    vx, vy, wz = twist
    angle = wz*model.horizon
    a = math.sin(angle)/wz if wz else model.horizon
    b = (1.-math.cos(angle))/wz if wz else 0.
    x, y = a*vx-b*vy, b*vx+a*vy
    c, s = math.cos(pose[2]), math.sin(pose[2])
    expected = pose + [c*x-s*y, s*x+c*y, angle]
    np.testing.assert_allclose(predicted, expected, atol=1.e-11, rtol=0.)
    np.testing.assert_allclose(velocity, twist, atol=1.e-12)


@pytest.mark.parametrize('change_at', [.0001, .009, .017, .051, .137, .2999])
@pytest.mark.parametrize('response', [.0005, .001, .05, .15, .3])
def test_stop_uses_exact_delayed_transition_time(change_at, response):
    model = DelayedMotionPredictor(response_sec=response)
    now, speed = 1., 3.5
    # History is fresh and describes cruise, then a stop which takes effect
    # change_at seconds after the measured state. Neither stamp is quantized.
    model.record(now-model.delay_sec, [speed, 0., 0.])
    model.record(now-model.delay_sec+change_at, [0., 0., 0.])
    # Keep a fresh heartbeat without altering the chosen command transition.
    model.record(now, [0., 0., 0.])
    # record deliberately resets on >200 ms gaps: provide the valid ordered
    # history directly to isolate integration from that separate watchdog.
    model.history.clear()
    model.history.extend([(now-model.delay_sec, np.array([speed, 0., 0.])),
                          (now-model.delay_sec+change_at, np.zeros(3)),
                          (now, np.zeros(3))])
    predicted, velocity = model.predict(np.zeros(3), [speed, 0., 0.], now)
    braking_time = model.horizon-change_at
    distance = speed*(change_at+response*(1.-math.exp(-braking_time/response)))
    assert predicted == pytest.approx([distance, 0., 0.], abs=2.e-9)
    assert velocity == pytest.approx([speed*math.exp(-braking_time/response), 0., 0.], abs=1.e-12)


def test_changing_body_twist_matches_independent_dense_quadrature():
    model = DelayedMotionPredictor()
    initial = np.array([3.5, -.5, 1.2])
    command = np.array([1., .8, -.7])
    model.record(1., command)
    pose = np.array([-.5, .2, .8])
    predicted, velocity = model.predict(pose, initial, 1.)
    # With the timestamp above, measured motion holds until delay_sec, then
    # each body axis follows its analytic first-order response.
    dt = model.horizon/100000
    times = (np.arange(100000)+.5)*dt
    active = np.maximum(0., times-model.delay_sec)
    v = command+(initial-command)*np.exp(-active[:, None]/model.response_sec)
    yaw = (pose[2]+initial[2]*np.minimum(times, model.delay_sec)
           +command[2]*active+(initial[2]-command[2])*model.response_sec
           *(-np.expm1(-active/model.response_sec)))
    expected = pose.copy()
    expected[0] += dt*np.sum(np.cos(yaw)*v[:, 0]-np.sin(yaw)*v[:, 1])
    expected[1] += dt*np.sum(np.sin(yaw)*v[:, 0]+np.cos(yaw)*v[:, 1])
    active_end = model.horizon-model.delay_sec
    expected[2] += (initial[2]*model.delay_sec+command[2]*active_end
        +(initial[2]-command[2])*model.response_sec*(1.-math.exp(-active_end/model.response_sec)))
    np.testing.assert_allclose(predicted, expected, atol=1.e-9, rtol=0.)
    np.testing.assert_allclose(velocity, command+(initial-command)
        *math.exp(-active_end/model.response_sec), atol=1.e-12)


@pytest.mark.parametrize('response', [.0005, .001])
@pytest.mark.parametrize('command', [(0., 0., 0.), (1., .8, -.7)])
def test_fast_turning_response_resolves_transient_before_long_tail(response, command):
    model = DelayedMotionPredictor(response_sec=response)
    initial = np.array([3.5, -.5, 1.6])
    command = np.asarray(command)
    pose = np.array([-.5, .2, .8])
    # A regular fresh command history means the response starts at t=0.
    for stamp in np.linspace(1.-model.delay_sec, 1., 7):
        model.record(stamp, command)
    predicted, velocity = model.predict(pose, initial, 1.)
    # Independent composite midpoint quadrature: resolve the first 30 time
    # constants on a fine grid, then cover the smooth tail on a coarse grid.
    transient_end = 30.*response
    edges = np.r_[np.linspace(0., transient_end, 30001),
                  np.linspace(transient_end, model.horizon, 20001)[1:]]
    times = .5*(edges[1:]+edges[:-1])
    widths = np.diff(edges)
    delta = initial-command
    v = command+delta*np.exp(-times[:, None]/response)
    yaw = pose[2]+command[2]*times-delta[2]*response*np.expm1(-times/response)
    expected = pose.copy()
    expected[0] += np.sum(widths*(np.cos(yaw)*v[:, 0]-np.sin(yaw)*v[:, 1]))
    expected[1] += np.sum(widths*(np.sin(yaw)*v[:, 0]+np.cos(yaw)*v[:, 1]))
    expected[2] += (command[2]*model.horizon
                   - delta[2]*response*math.expm1(-model.horizon/response))
    np.testing.assert_allclose(predicted, expected, atol=1.e-9, rtol=0.)
    np.testing.assert_allclose(velocity, command+delta
        *math.exp(-model.horizon/response), atol=1.e-12)


def test_tiny_turning_response_has_bounded_integration_work(monkeypatch):
    from omni_autonomy_next import motion_prediction
    model = DelayedMotionPredictor(response_sec=1.e-12)
    for stamp in np.linspace(.7, 1., 7):
        model.record(stamp, [0., 0., 0.])
    original_exp = motion_prediction.math.exp
    calls = []
    def count_exp(value):
        calls.append(value)
        return original_exp(value)
    monkeypatch.setattr(motion_prediction.math, 'exp', count_exp)
    predicted, velocity = model.predict(np.zeros(3), [3.5, 0., 1.6], 1.)
    assert len(calls) < 400
    assert np.linalg.norm(predicted) < 1.e-10
    np.testing.assert_array_equal(velocity, np.zeros(3))
