"""Production trajectory numerics, runnable without ROS transport packages."""
import ast
import math
from pathlib import Path
import sys
from typing import Optional

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / 'ros2_ws/src/omni_autonomy_next'
sys.path.insert(0, str(PACKAGE))
from omni_autonomy_next import omni_yaw

SOURCE = PACKAGE / 'omni_autonomy_next/trajectory_tracker_node.py'
NAMES = {'resample', 'path_tangents', 'menger_curvature', 'Trajectory', 'tracking_projection'}
TREE = ast.parse(SOURCE.read_text())
NUMERICS = ast.Module(body=[node for node in TREE.body
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in NAMES], type_ignores=[])
NAMESPACE = dict(vars(omni_yaw), Optional=Optional)
exec(compile(NUMERICS, str(SOURCE), 'exec'), NAMESPACE)
Trajectory = NAMESPACE['Trajectory']
resample = NAMESPACE['resample']
tracking_projection = NAMESPACE['tracking_projection']


def make_trajectory(points, speed=3.5, **kwargs):
    points = np.asarray(points, float)
    settings = dict(acceleration=3.3, deceleration=.85, lateral_acceleration=.85, entry_speed=0.)
    settings.update(kwargs)
    return Trajectory(points, np.zeros(len(points)), np.full(len(points), speed), **settings)


def test_slow_reference_clock_reaches_the_endpoint_without_an_artificial_speed_floor():
    plan = make_trajectory([[0., 0.], [.5, 0.], [1., 0.]], speed=1.e-5)
    assert plan.duration == pytest.approx(200000.)
    assert plan.sample(plan.duration)[0] == pytest.approx([1., 0.], abs=1.e-10)
    assert plan.sample(plan.duration)[1] == pytest.approx([0., 0.], abs=1.e-10)
    for at in np.linspace(0., plan.duration, 21):
        assert plan.time_at_arclength(plan.sample(at)[4]) == pytest.approx(at, abs=1.e-5)


@pytest.mark.parametrize('points', [[[0., 0.], [1., 0.]], [[0., 0.], [.5, 0.], [1., 0.]]])
def test_stationary_nonzero_interval_is_rejected_instead_of_inventing_motion(points):
    with pytest.raises(ValueError, match='no reachable speed'):
        make_trajectory(points, speed=0.)


def test_two_stationary_endpoints_require_an_acceleration_midpoint():
    with pytest.raises(ValueError, match='no reachable speed'):
        make_trajectory([[0., 0.], [.02, 0.]])
    plan = make_trajectory(resample(np.array([[0., 0.], [.02, 0.]]), .05))
    assert plan.duration < 1.
    assert plan.sample(plan.duration)[0] == pytest.approx([.02, 0.])


@pytest.mark.parametrize('overrides', [dict(acceleration=float('nan')), dict(deceleration=0.),
    dict(lateral_acceleration=-1.), dict(entry_speed=float('inf')), dict(angular_speed=float('nan')),
    dict(angular_acceleration=-2.)])
def test_nonphysical_motion_limits_fail_before_publishing_a_trajectory(overrides):
    with pytest.raises(ValueError):
        make_trajectory([[0., 0.], [.5, 0.], [1., 0.]], **overrides)


def test_duplicate_points_and_one_pose_holds_remain_finite():
    points = [[0., 0.], [0., 0.], [.5, 0.], [1., 0.], [1., 0.]]
    plan = make_trajectory(points)
    assert plan.project(np.array([.23, .1])) == pytest.approx(.23)
    assert plan.sample(plan.duration)[0] == pytest.approx([1., 0.])
    hold = make_trajectory([[1., 2.]], speed=0.)
    assert hold.duration == 0.
    assert hold.project(np.array([5., 6.])) == 0.
    assert hold.sample(3.)[0] == pytest.approx([1., 2.])


def test_projection_window_clips_inside_segments_without_vertex_quantization():
    plan = make_trajectory([[0., 0.], [.5, 0.], [1., 0.]])
    for at in np.linspace(.12, .18, 61):
        assert plan.project(np.array([at, .03]), minimum=.11, maximum=.19) == pytest.approx(at)
    assert plan.project(np.array([.9, 0.]), minimum=.13, maximum=.37) == pytest.approx(.37)
    assert plan.project(np.array([0., 0.]), minimum=.13, maximum=.37) == pytest.approx(.13)
    assert plan.project(np.array([.9, 0.]), minimum=.3, maximum=.3) == pytest.approx(.3)
    assert plan.project(np.array([5., 0.]), minimum=2., maximum=3.) == pytest.approx(1.)


def test_ordered_progress_does_not_skip_a_loop_when_return_lane_is_closer():
    points = resample(np.array([[-2., 0.], [2., 0.], [2., 2.],
                               [-2., 2.], [-2., .04], [2., .04]]), .05)
    plan = make_trajectory(points)
    previous = None
    skipped = []
    # At 3.5 m/s and 20 Hz, subsequent noisy samples are closer to the future
    # return lane. The global nearest projection would skip its 12 m loop.
    for step, x in enumerate(np.arange(-1.5, 1.51, 3.5*.05)):
        observed = np.array([x, 0. if step == 0 else .026])
        predicted = observed + [.35, 0.]
        observed_s, predicted_s, previous = tracking_projection(plan, observed, predicted, previous)
        assert observed_s == pytest.approx(x+2., abs=1.e-8)
        assert predicted_s == pytest.approx(x+2.35, abs=1.e-8)
        skipped.append(plan.project(observed)-observed_s)
    assert max(skipped) > 11.


def test_ordered_progress_can_finish_the_loop_and_reset_for_a_replacement():
    points = resample(np.array([[0., 0.], [2., 0.], [2., 2.], [0., 2.], [0., 0.]]), .05)
    plan = make_trajectory(points)
    previous = None
    for expected_s, point in zip(plan.arclength, plan.points):
        observed_s, predicted_s, previous = tracking_projection(plan, point, point, previous)
        assert observed_s == pytest.approx(expected_s, abs=1.e-8)
        assert predicted_s == pytest.approx(expected_s, abs=1.e-8)
    replacement = make_trajectory([[0., 0.], [.5, 0.], [1., 0.]])
    observed_s, _, state = tracking_projection(replacement, np.array([.8, 0.]), np.array([.8, 0.]), previous)
    assert observed_s == pytest.approx(.8)
    assert state[0] is replacement


def test_curve_profile_shares_tangential_and_lateral_acceleration_budget():
    theta = np.linspace(0., math.pi/2., 151)
    points = np.c_[3.*np.sin(theta), 3.*(1.-np.cos(theta))]
    plan = make_trajectory(points)
    curvature = NAMESPACE['menger_curvature'](plan.points)
    curvature = np.maximum(curvature[:-1], curvature[1:])
    acceleration = np.diff(plan.speed**2)/(2.*np.diff(plan.arclength))
    forward_budget = np.where(acceleration >= 0., 3.3, .85)
    lateral = curvature*np.maximum(plan.speed[:-1], plan.speed[1:])**2
    demand = np.hypot(acceleration/forward_budget, lateral/.85)
    assert np.max(demand) <= 1.+1.e-9
    assert np.max(plan.speed) <= math.sqrt(.85*3.)+1.e-8
    assert np.max(plan.speed) > 1.5


def test_straight_path_retains_requested_high_speed_with_zero_lateral_demand():
    plan = make_trajectory(resample(np.array([[0., 0.], [12., 0.]]), .05))
    assert np.max(plan.speed) == pytest.approx(3.5)
    assert np.isinf(plan.curvature_speed_limits).all()


def test_full_command_envelope_brakes_before_a_curve_and_can_leave_rest():
    # A long straight feeding a quarter circle. The execution cap must be
    # below cruise BEFORE the bend; local curvature alone cannot brake in time.
    straight = np.c_[np.linspace(-6., 0., 121), np.zeros(121)]
    theta = np.linspace(0., math.pi/2., 61)
    bend = np.c_[2.*np.sin(theta), 2.*(1.-np.cos(theta))]
    plan = make_trajectory(np.vstack((straight[:-1], bend)))
    limits = plan.command_speed_limits
    assert limits[0] > 3.
    assert plan.speed[0] == 0.
    assert limits[-1] > 0.  # the final position servo must still correct overshoot
    before = np.searchsorted(plan.points[:, 0], -1.)
    assert np.isinf(plan.curvature_speed_limits[before])
    assert limits[before] < 2.
    curvature = NAMESPACE['menger_curvature'](plan.points)
    deceleration = np.maximum(0., -np.diff(limits**2)/(2.*np.diff(plan.arclength)))
    lateral = np.maximum(curvature[:-1], curvature[1:])*np.maximum(limits[:-1], limits[1:])**2
    assert np.max(np.hypot(deceleration/.85, lateral/.85)) <= 1.+1.e-9
