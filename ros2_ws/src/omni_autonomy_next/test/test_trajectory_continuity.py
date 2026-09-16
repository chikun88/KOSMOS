"""Regression coverage for short moves and continuous trajectory progress."""
import math
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from omni_autonomy_next.trajectory_tracker_node import (
    Trajectory, TrajectoryTracker, resample, terminal_yaw_reference,
)


def trajectory(length=2.0):
    points = resample(np.array([[0.0, 0.0], [length, 0.0]]), 0.05)
    return Trajectory(
        points, np.zeros(len(points)), np.full(len(points), 0.78),
        acceleration=0.85, lateral_acceleration=1.2, entry_speed=0.0)


@pytest.mark.parametrize('length', [0.005, 0.02, 0.04, 0.06])
def test_short_move_from_rest_has_a_real_acceleration_phase(length):
    plan = trajectory(length)
    # A triangular acceleration/deceleration profile, not length / 0.0001.
    assert plan.duration == pytest.approx(2 * math.sqrt(length / 0.85))
    assert np.linalg.norm(plan.sample(plan.duration / 2)[1]) > 0.01


@pytest.mark.parametrize('fraction', [0.03, 0.12, 0.88, 0.97])
def test_feedforward_equals_the_derivative_of_the_reference(fraction):
    plan = trajectory()
    t = fraction * plan.duration
    dt = 1.0e-6
    derivative = (plan.sample(t + dt)[0] - plan.sample(t - dt)[0]) / (2 * dt)
    assert derivative == pytest.approx(plan.sample(t)[1], abs=1.0e-6)


def test_progress_is_not_quantized_to_path_vertices():
    plan = trajectory()
    positions = np.linspace(0.12, 0.18, 121)
    progress = [plan.project(np.array([x, 0.01])) for x in positions]
    assert progress == pytest.approx(positions, abs=1.0e-10)


def test_projection_handles_duplicate_vertices_and_clamps_to_endpoints():
    points = np.array([[0., 0.], [0., 0.], [1., 0.], [1., 1.]])
    plan = Trajectory(points, np.zeros(4), np.full(4, 0.5),
                      acceleration=0.85, lateral_acceleration=1.2, entry_speed=0.)
    assert plan.project(np.array([0.32, 0.1])) == pytest.approx(0.32)
    assert plan.project(np.array([-0.2, 0.])) == 0.
    assert plan.project(np.array([1., 1.2])) == 2.


@pytest.mark.parametrize('length', [.02, 2.0])
def test_replanning_and_lead_clamp_use_the_inverse_trajectory_clock(length):
    plan = trajectory(length)
    for t in np.linspace(0., plan.duration, 51):
        s = plan.sample(t)[4]
        assert plan.time_at_arclength(s) == pytest.approx(t, abs=1.0e-7)


def test_terminal_rotation_targets_the_goal_even_if_the_reference_is_behind():
    # A slowed reference clock or a localization correction can leave the
    # robot at the goal while the time-based yaw reference is still behind.
    # Exercise the real control tick: a +0.2 rad pose must turn toward zero,
    # not back toward the +1.0 rad start of the trajectory.
    plan = trajectory()
    plan.yaws = np.linspace(1., 0., len(plan.points))
    plan.yaw_per_metre = np.full(len(plan.points), -1. / plan.length)
    params = dict(pose_timeout_sec=.3, velocity_timeout_sec=.2, plan_timeout_sec=2., feedback_delay_sec=.2,
                  max_predicted_yaw_rad=.35, max_reference_lead_m=.35,
                  no_progress_speed_m_s=.05, no_progress_epsilon_m=.02,
                  no_progress_timeout_sec=4., terminal_approach_m=.10,
                  terminal_timeout_sec=4., terminal_hold_sec=3.,
                  position_gain=2.4, position_damping=.25, yaw_gain=1.6,
                  terminal_speed=.3)
    now = time.monotonic()
    commands = []
    node = SimpleNamespace(
        trajectory=plan, pose=np.array([1.98, 0., .2]), pose_stamp=now,
        reference_time=0., plan_stamp=now, active_goal=np.array([2., 0., 0.]),
        velocity_stamp=now, odometry_paused=False,
        lock=threading.Lock(), velocity=np.zeros(3), speed_scale=.1, period=.05,
        best_distance=math.inf, best_distance_at=now, terminal_since=None,
        finished_at=None, speed_limit=.78, lateral_limit=.702, yaw_limit=1.3,
        envelope=SimpleNamespace(max_wheel=100., wheel_cost=lambda *args: 0.),
        plan_build_ms=0., profile_name='balanced', snap_offset=0., snapped=True,
        get_parameter=lambda name: SimpleNamespace(value=params[name]),
        _relay_behavior=lambda *args: False, _status=lambda *args, **kwargs: None,
        _monitor_safe_yaw_rate=lambda rate, *args: rate,
        _rate_limit=lambda *args: args, _track_flow=lambda *args: None,
        _publish=lambda *args: commands.append(args),
        _flow_gap=lambda: 0., _flow_gain=lambda: 1.)
    TrajectoryTracker._tick(node)
    assert commands[-1][2] < 0.
    assert commands[-1][2] == pytest.approx(-.32)


def test_terminal_yaw_handoff_is_continuous_and_takes_the_short_turn():
    for boundary in (.1, .2):
        before = terminal_yaw_reference(3.1, .4, -3.1, boundary + 1.e-6, .1)
        after = terminal_yaw_reference(3.1, .4, -3.1, boundary - 1.e-6, .1)
        assert after == pytest.approx(before, abs=1.e-7)
    middle, rate = terminal_yaw_reference(3.1, .4, -3.1, .15, .1)
    assert abs(abs(middle) - math.pi) < 1.e-8
    assert rate == pytest.approx(.2)
    assert terminal_yaw_reference(.3, .4, 0., .08, .1) == pytest.approx((0., 0.))


def test_stale_motor_output_does_not_latch_the_reference_clock_at_zero():
    import json
    node = SimpleNamespace(speed_scale=0., profiles={'balanced': {}}, profile_name='balanced')
    TrajectoryTracker._on_safety_state(node, SimpleNamespace(data=json.dumps(dict(
        reason='STALE_COMMAND', applied_scale=0., reference_scale=.7,
        profile='balanced'))))
    assert node.speed_scale == .7
    plan = trajectory()
    # This fresh nonzero feed-forward gets evaluated by Collision Monitor.
    # With applied_scale=0 both position and velocity stayed zero forever.
    reference = plan.sample(.05 * node.speed_scale)
    assert reference[0][0] > 0.
    assert reference[1][0] > 0.
    TrajectoryTracker._on_safety_state(node, SimpleNamespace(data=json.dumps(dict(
        reason='EMERGENCY_STOP', applied_scale=0., reference_scale=0.))))
    assert node.speed_scale == 0.
