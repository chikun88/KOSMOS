"""Whole-segment yaw/translation feasibility, beyond just sampled vertices."""
import math

import numpy as np
import pytest

from omni_autonomy_next.trajectory_tracker_node import Trajectory, resample


@pytest.mark.parametrize('seed', range(12))
def test_shared_acceleration_budget_holds_inside_every_segment(seed):
    rng = np.random.default_rng(seed)
    ds = rng.uniform(.005, .12, 40)
    s = np.r_[0., np.cumsum(ds)]
    points = np.column_stack((s, .07*np.sin(2.*s)))
    yaws = .3*np.sin(1.7*s) + .1*np.cos(4.*s)
    angular_speed, angular_acceleration = .9, 1.5
    plan = Trajectory(points, yaws, rng.uniform(.2, 1., len(s)),
        acceleration=.85, lateral_acceleration=1.2, entry_speed=.2,
        angular_speed=angular_speed, angular_acceleration=angular_acceleration)
    assert np.all(np.isfinite(plan.time)) and np.all(np.diff(plan.time) > 0.)
    assert plan.speed[-1] == 0.
    # q and v^2 are linear in arc length inside a segment; check their
    # physical product, including opposing signs of q' and acceleration.
    for i, length in enumerate(np.diff(plan.arclength)):
        q0, q1 = plan.yaw_per_metre[i:i+2]
        u0, u1 = plan.speed[i:i+2]**2
        a = (u1-u0)/(2.*length)
        assert abs(a) <= .85+1.e-10
        for fraction in np.linspace(0., 1., 101):
            q = q0 + fraction*(q1-q0)
            u = u0 + fraction*(u1-u0)
            assert abs(q*math.sqrt(max(0., u))) <= angular_speed+1.e-9
            assert abs((q1-q0)/length*u + q*a) <= angular_acceleration+1.e-9


def test_rotation_dominated_short_move_keeps_its_settling_reserve():
    points = resample(np.array([[0., 0.], [.3, 0.]]), .05)
    s = points[:, 0]
    yaws = .5*math.pi*np.minimum(s/.2, 1.)
    plan = Trajectory(points, yaws, np.full(len(s), .78),
        acceleration=.85, lateral_acceleration=1.2, entry_speed=0.,
        angular_speed=1.3, angular_acceleration=2.)
    # Full allocation failed strict settling with 300 ms delay and high
    # drive gain. Match the baseline constructor on this exact geometry.
    assert plan.duration == pytest.approx(3.065958658631477)
    samples = np.linspace(0., plan.duration, 3001)
    rates = np.array([plan.sample(t)[3] for t in samples])
    assert np.max(np.abs(rates)) <= 1.3+1.e-9
    assert np.max(np.abs(np.diff(rates)/np.diff(samples))) <= 2.+1.e-6


def test_translation_dominated_turn_uses_the_available_budget():
    points = resample(np.array([[0., 0.], [2., 0.]]), .05)
    s = points[:, 0]
    plan = Trajectory(points, .5*math.pi*np.minimum(s/1.9, 1.), np.full(len(s), .5),
        acceleration=.85, lateral_acceleration=1.2, entry_speed=0.,
        angular_speed=1.3, angular_acceleration=2.)
    # The taper in q can use over half alpha without exceeding the full
    # segment constraint, eliminating the old artificial cruise-speed dip.
    acceleration = np.diff(plan.speed**2)/(2.*np.diff(plan.arclength))
    bend = np.abs(np.diff(plan.yaw_per_metre)/np.diff(plan.arclength))
    assert np.max(bend*np.maximum(plan.speed[:-1], plan.speed[1:])**2) > 1.05
    assert np.max(np.abs(acceleration)) <= .85+1.e-9


@pytest.mark.parametrize('angle', [0., .4, -.4])
def test_straight_and_constant_yaw_slope_keep_the_full_linear_budget(angle):
    points = resample(np.array([[0., 0.], [2., 0.]]), .05)
    plan = Trajectory(points, angle*points[:, 0], np.full(len(points), .5),
        acceleration=.85, lateral_acceleration=1.2, entry_speed=0.,
        angular_speed=1.3, angular_acceleration=2.)
    assert np.max(plan.speed) == pytest.approx(.5)
    assert plan.speed[1] == pytest.approx(math.sqrt(2.*.85*.05))
