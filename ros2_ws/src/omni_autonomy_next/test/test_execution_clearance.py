"""Independent numerical checks of the delayed braking clearance envelope."""
import math
from pathlib import Path

import numpy as np
import pytest

from omni_autonomy_next.execution_clearance import (
    additional_delay_clearance_error, braking_pose_path,
    delayed_braking_certificate, delayed_braking_clearance,
)
from omni_autonomy_next import execution_clearance
from omni_autonomy_next.rl_residual import CadClearanceModel, load_footprint


def dense_oracle(pose, command, delay, linear, angular, radius, count=100001):
    """Numerically integrate velocity on a dense time grid, without path math.

    Trapezoidal time integration does not use the helper's travel inversion,
    weighted heading formula, interval error bound, or displacement formula.
    """
    pose, command = np.asarray(pose), np.asarray(command)
    speed = float(np.linalg.norm(command[:2]))
    linear_stop = delay + speed / linear
    angular_stop = delay + abs(command[2]) / angular
    times = np.unique(np.r_[np.linspace(0., max(linear_stop, angular_stop), count),
                            delay, linear_stop, angular_stop])
    dt = np.diff(times)
    speeds = np.maximum(0., speed - linear*np.maximum(0., times-delay))
    rates = np.sign(command[2])*np.maximum(
        0., abs(command[2])-angular*np.maximum(0., times-delay))
    headings = pose[2] + np.r_[0., np.cumsum(.5*(rates[:-1]+rates[1:])*dt)]
    body_direction = command[:2]/speed if speed else np.zeros(2)
    c, s = np.cos(headings), np.sin(headings)
    velocities = speeds[:, None]*np.column_stack((
        c*body_direction[0]-s*body_direction[1],
        s*body_direction[0]+c*body_direction[1]))
    points = pose[:2] + np.vstack((np.zeros(2), np.cumsum(
        .5*(velocities[:-1]+velocities[1:])*dt[:, None], axis=0)))
    travel_rates = speeds + radius*np.abs(rates)
    travel = np.r_[0., np.cumsum(.5*(travel_rates[:-1]+travel_rates[1:])*dt)]
    return times, points, headings, travel


@pytest.mark.parametrize('command,delay,linear,angular', [
    ((1.2, .4, .9), .12, .85, .6),
    ((-.7, .9, -1.8), 0., 1.3, 3.),
    ((.8, -.2, 2.4), .31, 1.5, 1.7),
    ((0., 0., -1.2), .12, .85, 1.1),
    ((1., 0., 0.), .2, .9, 1.),
])
def test_path_error_and_phase_samples_against_dense_time_integration(
        command, delay, linear, angular):
    pose, radius = np.array([.31, -.7, .83]), .588
    points, yaws, error, intervals = braking_pose_path(
        pose, command, delay, linear, angular, radius)
    times, dense_points, dense_yaws, dense_travel = dense_oracle(
        pose, command, delay, linear, angular, radius)
    sample_travel = np.r_[0., np.cumsum(intervals)]
    oracle_points = np.column_stack([
        np.interp(sample_travel, dense_travel, dense_points[:, axis])
        for axis in range(2)])
    oracle_yaws = np.interp(sample_travel, dense_travel, dense_yaws)
    # The allowance is for the independent numerical oracle's discretization.
    assert np.all(np.linalg.norm(points-oracle_points, axis=1) <= error+2.e-8)
    assert yaws == pytest.approx(oracle_yaws, abs=2.e-8)
    assert np.all(intervals >= 0.) and np.max(intervals) <= .01+1.e-11
    assert np.all(np.diff(error) >= 0.) and error[0] == 0.
    assert sample_travel[-1] == pytest.approx(dense_travel[-1], abs=2.e-10)
    stops = (delay, delay+np.linalg.norm(command[:2])/linear,
             delay+abs(command[2])/angular)
    for boundary in stops:
        arc = np.interp(boundary, times, dense_travel)
        assert np.min(abs(sample_travel-arc)) <= 2.e-10
    delay_arc = np.interp(delay, times, dense_travel)
    assert np.all(error[sample_travel <= delay_arc+1.e-10] == 0.)


def rectangle(half_x, half_y):
    return np.array([[-half_x, -half_y], [half_x, -half_y],
                     [half_x, half_y], [-half_x, half_y]])


def test_pure_rotation_finds_contact_between_clear_endpoint_poses():
    model = CadClearanceModel([[[.7, .7], [.7, .71]]], rectangle(1., .1))
    pose, command = [0., 0., 0.], [0., 0., 2.]
    angular = 4./math.pi  # End at pi/2; the long corner crosses the short wall.
    _, points, yaws, _ = dense_oracle(pose, command, 0., 1., angular, model.radius,
                                    count=12001)
    assert model.body_clearance(points[0], yaws[0]) > .2
    assert model.body_clearance(points[-1], yaws[-1]) > .2
    assert np.min(model.clearance_over_poses(points, yaws)) == 0.
    assert delayed_braking_clearance(model, pose, command, 0., 1., angular) < 0.


def test_combined_curved_motion_finds_contact_between_clear_endpoint_poses():
    pose, command = [0., 0., -.3], [.6, .15, 2.]
    delay, linear, angular = .6, .9, 2.7
    footprint = rectangle(.08, .045)
    radius = float(np.max(np.linalg.norm(footprint, axis=1)))
    times, points, yaws, _ = dense_oracle(
        pose, command, delay, linear, angular, radius, count=12001)
    crossing = np.array([np.interp(.35, times, points[:, axis]) for axis in range(2)])
    model = CadClearanceModel([[crossing-[0., .02], crossing+[0., .02]]], footprint)
    assert model.body_clearance(points[0], yaws[0]) > .1
    assert model.body_clearance(points[-1], yaws[-1]) > .1
    assert np.min(model.clearance_over_poses(points, yaws)) == 0.
    bound = delayed_braking_clearance(model, pose, command, delay, linear, angular)
    assert bound < 0.


def test_half_travel_reserve_covers_contact_between_consecutive_clear_samples():
    # A deliberately tiny outline and wall make every sampled pose clear while
    # the true continuous translation crosses the wall between the first pair.
    model = CadClearanceModel([[[.005, -.001], [.005, .001]]], rectangle(.0001, .0001))
    pose, command = [0., 0., 0.], [1., 0., 0.]
    points, yaws, _, _ = braking_pose_path(pose, command, 0., 1., 1., model.radius)
    sampled = model.clearance_over_poses(points, yaws)
    assert np.min(sampled) > .0048
    _, dense_points, dense_yaws, _ = dense_oracle(
        pose, command, 0., 1., 1., model.radius, count=12001)
    assert np.min(model.clearance_over_poses(dense_points, dense_yaws)) == 0.
    assert delayed_braking_clearance(model, pose, command, 0., 1., 1.) < 0.


@pytest.mark.parametrize('command', [(.4, .15, 1.4), (-.25, .3, -1.1)])
def test_clearance_bound_never_exceeds_dense_full_footprint_minimum(command):
    model = CadClearanceModel(
        [[[.55, -.5], [.55, 1.]], [[-1., -.4], [1., -.4]]], rectangle(.12, .07))
    pose, delay, linear, angular = [.02, .02, .6], .12, .8, 1.3
    _, points, yaws, _ = dense_oracle(pose, command, delay, linear, angular,
                                    model.radius, count=12001)
    dense_minimum = float(np.min(model.clearance_over_poses(points, yaws)))
    bound = delayed_braking_clearance(model, pose, command, delay, linear, angular)
    assert bound <= dense_minimum-model.radius*.02 + 1.e-8


def test_measured_momentum_can_be_unsafe_when_nominal_request_is_safe():
    model = CadClearanceModel([[[.3, -.5], [.3, .5]]], rectangle(.1, .1))
    pose = [0., 0., 0.]
    measured_bound = delayed_braking_clearance(model, pose, [.5, 0., 0.], .2, 1., 1.)
    request_bound = delayed_braking_clearance(model, pose, [-.2, 0., 0.], .2, 1., 1.)
    assert measured_bound < 0.
    assert request_bound >= .025
    # Both envelopes must be considered independently; this does not certify
    # an arbitrary transition from the measured twist to the request.


def test_stationary_envelope_keeps_exact_reserves_without_extra_margin_subtraction():
    model = CadClearanceModel([[[.3, -.5], [.3, .5]]], rectangle(.1, .1))
    pose = [0., 0., 0.]
    points, yaws, error, travel = braking_pose_path(pose, [0., 0., 0.], .2, 1., 1., model.radius)
    assert points.tolist() == [[0., 0.]] and yaws.tolist() == [0.]
    assert error.tolist() == [0.] and travel.size == 0
    bound = delayed_braking_clearance(model, pose, [0., 0., 0.], .2, 1., 1., margin=.3)
    assert bound == pytest.approx(.2-model.radius*.02-.005)


@pytest.mark.parametrize('overrides', [
    {'pose': [math.nan, 0., 0.]}, {'pose': [0., 0., math.inf]},
    {'pose': [0., 0.]}, {'command': [0., math.nan, 0.]},
    {'command': [0., 0., -math.inf]}, {'command': [0., 0.]},
    {'delay_sec': math.inf}, {'delay_sec': -.1},
    {'linear_deceleration': 0.}, {'linear_deceleration': math.nan},
    {'angular_deceleration': -1.}, {'angular_deceleration': math.inf},
    {'radius': math.nan}, {'radius': -.1},
    {'spacing': 0.}, {'spacing': .02}, {'spacing': math.nan}, {'spacing': 5.e-324},
])
def test_invalid_path_inputs_fail_closed(overrides):
    kwargs = dict(pose=[0., 0., 0.], command=[.2, .1, .3], delay_sec=.12,
                  linear_deceleration=.8, angular_deceleration=1., radius=.5)
    kwargs.update(overrides)
    with pytest.raises(ValueError):
        braking_pose_path(**kwargs)


@pytest.mark.parametrize('margin', [math.nan, math.inf, -.01])
def test_invalid_comparison_margin_is_rejected(margin):
    model = CadClearanceModel([[[.3, -.5], [.3, .5]]], rectangle(.1, .1))
    with pytest.raises(ValueError):
        delayed_braking_clearance(model, [0., 0., 0.], [.1, 0., 0.], .1, 1., 1., margin)


@pytest.mark.parametrize('values', [[math.nan], [-.1], [[.1, .1]]])
def test_invalid_model_distance_data_is_rejected(values):
    class InvalidModel:
        radius = .5

        def clearance_over_poses(self, _points, _yaws, cap):
            return values

    with pytest.raises(ValueError):
        delayed_braking_clearance(InvalidModel(), [0., 0., 0.], [0., 0., 0.], .1, 1., 1.)


class ObservedModel(CadClearanceModel):
    def __init__(self, segments, footprint):
        super().__init__(segments, footprint)
        self.batch_sizes = []

    def clearance_over_poses(self, points, yaws, cap=.35):
        self.batch_sizes.append(len(yaws))
        return super().clearance_over_poses(points, yaws, cap)


@pytest.mark.parametrize('command', [(.2, .08, .3), (-.15, .06, -.4)])
def test_fast_accept_is_a_complete_dense_oracle_bound_without_path_queries(command):
    model = ObservedModel([[[1., -2.], [1., 2.]]], rectangle(.1, .07))
    pose, delay, linear, angular = [0., 0., .4], .12, .8, 1.3
    safe, bound, rejected = delayed_braking_certificate(
        model, pose, command, delay, linear, angular)
    assert safe is True and bound >= .025 and rejected is None
    assert model.batch_sizes == []
    _, points, yaws, _ = dense_oracle(pose, command, delay, linear, angular,
                                    model.radius, count=12001)
    dense_minimum = np.min(CadClearanceModel.clearance_over_poses(model, points, yaws))
    assert bound <= dense_minimum-model.radius*.02 + 1.e-8


def test_early_rejection_skips_later_chunks_and_never_claims_a_complete_minimum():
    model = ObservedModel([[[.3, -.5], [.3, .5]]], rectangle(.1, .1))
    pose, command = [0., 0., 0.], [.7, 0., 0.]
    safe, complete, rejected = delayed_braking_certificate(model, pose, command, .12, .8, 1.)
    assert safe is False and complete is None and rejected < .025
    assert model.batch_sizes == [24]
    points, _, _, _ = braking_pose_path(pose, command, .12, .8, 1., model.radius)
    assert len(points) > 24
    # The rejected sample precedes contact and is a higher value than the
    # actual complete proof's minimum; it must not be labelled a full bound.
    full_bound = delayed_braking_clearance(model, pose, command, .12, .8, 1.)
    assert full_bound < rejected


@pytest.mark.parametrize('command', [
    (0., 0., 0.), (.15, .04, .2), (.4, .15, 1.4),
    (.8, 0., 0.), (-.25, .3, -1.1), (0., 0., 2.),
])
def test_certificate_boolean_matches_full_envelope_decision(command):
    model = ObservedModel(
        [[[.55, -.5], [.55, 1.]], [[-1., -.4], [1., -.4]]], rectangle(.12, .07))
    args = (model, [.02, .02, .6], command, .12, .8, 1.3)
    full_bound = delayed_braking_clearance(*args)
    model.batch_sizes.clear()
    safe, complete, rejected = delayed_braking_certificate(*args)
    assert safe == (full_bound >= .025)
    if safe:
        assert complete >= .025 and rejected is None
        if model.batch_sizes:
            assert complete == pytest.approx(full_bound, abs=1.e-12)
    else:
        assert complete is None and rejected < .025
    assert all(size <= 24 for size in model.batch_sizes)


@pytest.mark.parametrize('overrides', [
    {'pose': [math.nan, 0., 0.]}, {'pose': [0., 0.]},
    {'command': [math.inf, 0., 0.]}, {'command': [0., 0., math.nan]},
    {'command': [1.e308, 0., 0.]},
    {'delay_sec': math.inf}, {'delay_sec': -.1},
    {'linear_deceleration': 0.}, {'angular_deceleration': math.nan},
    {'margin': math.inf}, {'margin': -.1},
    {'spacing': .02}, {'spacing': 0.}, {'spacing': 5.e-324},
])
def test_certificate_validates_inputs_before_cheap_acceptance(overrides):
    model = ObservedModel([[[1., -2.], [1., 2.]]], rectangle(.1, .07))
    kwargs = dict(model=model, pose=[0., 0., 0.], command=[.2, .1, .3], delay_sec=.12,
                  linear_deceleration=.8, angular_deceleration=1.)
    kwargs.update(overrides)
    with pytest.raises(ValueError):
        delayed_braking_certificate(**kwargs)
    assert model.batch_sizes == []


def test_runtime_work_budget_rejects_large_finite_raw_motion_before_geometry_or_path(monkeypatch):
    class ForbiddenGeometry:
        radius = .5

        def body_clearance(self, *args, **kwargs):
            pytest.fail('oversized runtime envelope queried current geometry')

        def clearance_over_poses(self, *args, **kwargs):
            pytest.fail('oversized runtime envelope queried path geometry')

    def forbidden_path(*args, **kwargs):
        pytest.fail('oversized runtime envelope allocated a braking path')

    monkeypatch.setattr(execution_clearance, 'braking_pose_path', forbidden_path)
    with pytest.raises(ValueError, match='sample budget'):
        delayed_braking_certificate(ForbiddenGeometry(), [0., 0., 0.], [8., 0., 1.],
                                    .2, .8, 1.)


def test_research_path_retains_larger_budget_and_runtime_path_stays_at_2049_poses():
    large = braking_pose_path([0., 0., 0.], [8., 0., 1.], .2, .8, 1., .5)
    assert len(large[0]) > 2049
    configured = braking_pose_path([0., 0., 0.], [4., 0., 1.], .32, .8, 1., .588,
                                   max_intervals=2048)
    points, yaws, error, intervals = configured
    assert len(intervals) <= 2048
    assert len(points) == len(yaws) == len(error) == len(intervals)+1 <= 2049


def test_phase_rounding_cannot_allocate_more_than_the_generator_budget(monkeypatch):
    # Total travel needs only 2.7 sample intervals. Its two phases separately
    # round up to 1+3 intervals; reject the second phase before allocating it.
    allocations = []
    linspace = execution_clearance.np.linspace

    def observe_allocation(start, stop, count):
        allocations.append(count)
        return linspace(start, stop, count)

    monkeypatch.setattr(execution_clearance.np, 'linspace', observe_allocation)
    with pytest.raises(ValueError, match='sample budget'):
        braking_pose_path([0., 0., 0.], [.2, 0., 0.], .015,
                          .2**2/(2.*.024), 1., .1, max_intervals=3)
    assert allocations == [2]


def test_runtime_passes_its_exact_budget_to_fallback_generator(monkeypatch):
    model = CadClearanceModel([[[.3, -.5], [.3, .5]]], rectangle(.1, .1))
    path = execution_clearance.braking_pose_path
    received = []

    def observe_budget(*args, **kwargs):
        received.append(kwargs.get('max_intervals'))
        return path(*args, **kwargs)

    monkeypatch.setattr(execution_clearance, 'braking_pose_path', observe_budget)
    delayed_braking_certificate(model, [0., 0., 0.], [.7, 0., 0.], .12, .8, 1.)
    assert received == [2048]


def body_points(points, yaws, footprint):
    c, s = np.cos(yaws)[:, None], np.sin(yaws)[:, None]
    return points[:, None, :] + np.stack((
        c*footprint[None, :, 0] - s*footprint[None, :, 1],
        s*footprint[None, :, 0] + c*footprint[None, :, 1]), axis=-1)


@pytest.mark.parametrize('command,elapsed', [
    ((.7, .25, 0.), .136),
    ((.7, -.2, 1.4), .09),
    ((-.3, .65, -1.8), .31),
    ((.3, .2, 3.), 1.1),  # Rotation factor reaches the global chord cap of 2.
    ((0., 0., -1.5), .2),
    ((.5, -.1, .8), 0.),
])
def test_extra_delay_bound_covers_dense_prefix_and_matching_braking_progress(command, elapsed):
    # The outline is asymmetric about base_link, so centroid-based radii or a
    # yaw-only center comparison would miss some body-point displacement.
    footprint = np.array([[-.1, -.1], [.37, -.1], [.37, .2], [-.1, .2]])
    radius = float(np.max(np.linalg.norm(footprint, axis=1)))
    pose, delay, linear, angular = [.4, -.7, .61], .12, .85, 1.3
    times, points, yaws, _ = dense_oracle(
        pose, command, delay, linear, angular, radius, count=24001)
    later_times, later_points, later_yaws, _ = dense_oracle(
        pose, command, delay+elapsed, linear, angular, radius, count=24001)
    matching_points = np.column_stack([
        np.interp(times+elapsed, later_times, later_points[:, axis])
        for axis in range(2)])
    matching_yaws = np.interp(times+elapsed, later_times, later_yaws)
    suffix_errors = np.linalg.norm(
        body_points(matching_points, matching_yaws, footprint)
        - body_points(points, yaws, footprint), axis=2)
    prefix = later_times <= elapsed
    prefix_errors = np.linalg.norm(
        body_points(later_points[prefix], later_yaws[prefix], footprint)
        - body_points(np.asarray([pose[:2]]), np.asarray([pose[2]]), footprint), axis=2)
    bound = additional_delay_clearance_error(command, delay, linear, radius, elapsed)
    assert np.max(suffix_errors) <= bound+2.e-7
    assert np.max(prefix_errors) <= bound+2.e-7


def test_extra_delay_rejects_straight_nominal_false_pass_with_deployed_outline():
    config = Path(__file__).resolve().parents[1] / 'config'
    footprint = load_footprint(config / 'competition_footprints.yaml')
    model = CadClearanceModel([[[0., -3.], [0., 3.]]], footprint)
    pose = [.3-float(np.min(footprint[:, 0])), 0., 0.]
    command, delay, linear, angular, elapsed = [-.4, 0., 0.], .33, .85, 1.2, .136
    nominal = delayed_braking_clearance(model, pose, command, delay, linear, angular)
    assert nominal >= .025
    error = additional_delay_clearance_error(command, delay, linear, model.radius, elapsed)
    assert error == pytest.approx(.0544)
    _, points, yaws, _ = dense_oracle(
        pose, command, delay+elapsed, linear, angular, model.radius, count=12001)
    actual_minimum = float(np.min(model.clearance_over_poses(points, yaws)))
    assert actual_minimum == pytest.approx(.019482352941176, abs=2.e-7)
    assert actual_minimum < .025 and nominal-error < .025


@pytest.mark.parametrize('overrides', [
    {'command': [0., 0.]}, {'command': [math.nan, 0., 0.]},
    {'command': [0., 0., math.inf]}, {'command': [1.e308, 0., 0.]},
    {'command': [0., 0., 1.e308], 'elapsed_sec': 10.},
    {'delay_sec': -.1}, {'delay_sec': math.inf},
    {'linear_deceleration': 0.}, {'linear_deceleration': math.nan},
    {'radius': -.1}, {'radius': math.inf},
    {'elapsed_sec': -.1}, {'elapsed_sec': math.nan}, {'elapsed_sec': math.inf},
])
def test_additional_delay_error_rejects_invalid_or_overflowing_data(overrides):
    kwargs = dict(command=[.5, .1, .3], delay_sec=.12,
                  linear_deceleration=.85, radius=.588, elapsed_sec=.1)
    kwargs.update(overrides)
    with pytest.raises(ValueError):
        additional_delay_clearance_error(**kwargs)


@pytest.mark.parametrize('command', [(.2, .08, .3), (-.15, .06, -.4)])
def test_query_headroom_retains_fast_certificate_clearance_without_changing_safety(command):
    model = ObservedModel([[[1., -2.], [1., 2.]]], rectangle(.1, .07))
    pose, delay, linear, angular = [0., 0., .4], .12, .8, 1.3
    args = (model, pose, command, delay, linear, angular)
    original = delayed_braking_certificate(*args)
    expanded = delayed_braking_certificate(*args, query_headroom_m=.1)
    assert original[0] is expanded[0] is True
    assert original[2] is expanded[2] is None
    assert expanded[1] == pytest.approx(original[1]+.1, abs=1.e-12)
    assert model.batch_sizes == []
    _, points, yaws, _ = dense_oracle(
        pose, command, delay, linear, angular, model.radius, count=12001)
    dense_minimum = float(np.min(CadClearanceModel.clearance_over_poses(
        model, points, yaws, cap=2.)))
    assert expanded[1] <= dense_minimum-model.radius*.02-.005+1.e-8


def test_query_headroom_retains_dense_certificate_clearance_without_changing_samples():
    # Travel exceeds the initial clearance, so the cheap proof fails. Moving
    # away from the wall is nevertheless safe over the entire braking sweep.
    model = ObservedModel([[[.3, -2.], [.3, 2.]]], rectangle(.1, .1))
    pose, command, delay, linear, angular = [0., 0., 0.], [-.7, 0., 0.], .12, .8, 1.
    args = (model, pose, command, delay, linear, angular)
    original = delayed_braking_certificate(*args)
    original_batches = list(model.batch_sizes)
    model.batch_sizes.clear()
    expanded = delayed_braking_certificate(*args, query_headroom_m=.1)
    assert original[0] is expanded[0] is True
    assert original[2] is expanded[2] is None
    assert expanded[1] == pytest.approx(original[1]+.1, abs=1.e-12)
    assert original_batches and model.batch_sizes == original_batches
    _, points, yaws, _ = dense_oracle(
        pose, command, delay, linear, angular, model.radius, count=12001)
    dense_minimum = float(np.min(CadClearanceModel.clearance_over_poses(
        model, points, yaws, cap=2.)))
    assert expanded[1] <= dense_minimum-model.radius*.02-.005+1.e-8


@pytest.mark.parametrize('headroom', [.1, 1., 100.])
def test_query_headroom_does_not_accept_an_unsafe_sweep_or_change_early_rejection(headroom):
    model = ObservedModel([[[.3, -.5], [.3, .5]]], rectangle(.1, .1))
    args = (model, [0., 0., 0.], [.7, 0., 0.], .12, .8, 1.)
    original = delayed_braking_certificate(*args)
    model.batch_sizes.clear()
    expanded = delayed_braking_certificate(*args, query_headroom_m=headroom)
    assert original[0] is expanded[0] is False
    assert original[1] is expanded[1] is None
    assert expanded[2] == pytest.approx(original[2], abs=1.e-12)
    assert model.batch_sizes == [24]


@pytest.mark.parametrize('headroom,margin', [
    (math.nan, .025), (math.inf, .025), (-math.inf, .025), (-.1, .025),
    (1.e308, 1.e308),
])
def test_query_headroom_rejects_invalid_or_overflowing_caps_before_geometry(headroom, margin):
    class ForbiddenGeometry:
        radius = .5

        def body_clearance(self, *args, **kwargs):
            pytest.fail('invalid headroom queried current geometry')

        def clearance_over_poses(self, *args, **kwargs):
            pytest.fail('invalid headroom queried path geometry')

    with pytest.raises(ValueError):
        delayed_braking_certificate(ForbiddenGeometry(), [0., 0., 0.], [.2, .1, .3],
                                    .12, .8, 1., margin=margin,
                                    query_headroom_m=headroom)


def test_query_headroom_does_not_expand_runtime_work_budget(monkeypatch):
    class ForbiddenGeometry:
        radius = .5

        def body_clearance(self, *args, **kwargs):
            pytest.fail('headroom bypassed the runtime work budget')

    def forbidden_path(*args, **kwargs):
        pytest.fail('headroom allocated an oversized braking path')

    monkeypatch.setattr(execution_clearance, 'braking_pose_path', forbidden_path)
    with pytest.raises(ValueError, match='sample budget'):
        delayed_braking_certificate(ForbiddenGeometry(), [0., 0., 0.], [8., 0., 1.],
                                    .2, .8, 1., query_headroom_m=100.)


class CapObservedModel(ObservedModel):
    def __init__(self, segments, footprint):
        super().__init__(segments, footprint)
        self.current_caps = []
        self.current_values = []
        self.path_caps = []

    def body_clearance(self, point, yaw, cap=.35):
        self.current_caps.append(cap)
        value = super().body_clearance(point, yaw, cap)
        self.current_values.append(value)
        return value

    def clearance_over_poses(self, points, yaws, cap=.35):
        self.path_caps.append(cap)
        return super().clearance_over_poses(points, yaws, cap)


@pytest.mark.parametrize('command', [(-.7, 0., 0.), (-.7, .1, .6)])
def test_exact_initial_clearance_limits_query_cap_without_changing_complete_minimum(command):
    model = CapObservedModel([[[.3, -2.], [.3, 2.]]], rectangle(.1, .1))
    pose, delay, linear, angular = [0., 0., 0.], .12, .8, 1.
    safe, bound, rejected = delayed_braking_certificate(
        model, pose, command, delay, linear, angular, query_headroom_m=2.)
    assert safe is True and rejected is None
    current = model.current_values[0]
    assert current < model.current_caps[0]
    points, yaws, error, travel = braking_pose_path(
        pose, command, delay, linear, angular, model.radius)
    reserve = model.radius*.02+max(.005, .5*float(np.max(travel, initial=0.)))
    full_values = CadClearanceModel.clearance_over_poses(model, points, yaws, cap=3.)
    assert bound == pytest.approx(float(np.min(full_values-error))-reserve, abs=1.e-12)
    assert model.path_caps and all(
        cap == pytest.approx(current+error[-1], abs=1.e-12) for cap in model.path_caps)
    _, dense_points, dense_yaws, _ = dense_oracle(
        pose, command, delay, linear, angular, model.radius, count=12001)
    dense_minimum = float(np.min(CadClearanceModel.clearance_over_poses(
        model, dense_points, dense_yaws, cap=3.)))
    assert bound <= dense_minimum-model.radius*.02-.005+1.e-8


def test_exact_initial_cap_limiter_preserves_unsafe_partial_rejection():
    model = CapObservedModel([[[.3, -.5], [.3, .5]]], rectangle(.1, .1))
    args = (model, [0., 0., 0.], [.7, 0., 0.], .12, .8, 1.)
    original = delayed_braking_certificate(*args)
    model.batch_sizes.clear()
    model.path_caps.clear()
    expanded = delayed_braking_certificate(*args, query_headroom_m=2.)
    assert original[0] is expanded[0] is False
    assert original[1] is expanded[1] is None
    assert expanded[2] == pytest.approx(original[2], abs=1.e-12)
    assert model.path_caps == pytest.approx([.2])
    assert model.batch_sizes == [24]


def test_saturated_current_query_keeps_its_whole_path_cheap_proof():
    model = CapObservedModel([[[10., -2.], [10., 2.]]], rectangle(.1, .1))
    command, delay, linear, angular = [.2, .05, .3], .12, .8, 1.
    safe, bound, rejected = delayed_braking_certificate(
        model, [0., 0., 0.], command, delay, linear, angular, query_headroom_m=1.)
    speed, rate = math.hypot(*command[:2]), abs(command[2])
    travel = (delay*(speed+model.radius*rate)
              + speed**2/(2.*linear)+model.radius*rate**2/(2.*angular))
    assert safe is True and rejected is None
    assert model.current_values == model.current_caps
    assert bound == pytest.approx(model.current_caps[0]-travel-model.radius*.02-.005)
    assert model.path_caps == []


def test_exact_initial_contact_retains_positive_query_cap_and_collision_rejection():
    model = CapObservedModel([[[.1, -2.], [.1, 2.]]], rectangle(.1, .1))
    safe, complete, rejected = delayed_braking_certificate(
        model, [0., 0., 0.], [-.7, 0., 0.], .12, .8, 1., query_headroom_m=2.)
    assert model.current_values == [0.]
    assert safe is False and complete is None and rejected < .025
    assert model.path_caps and all(cap > 0. for cap in model.path_caps)
    assert model.batch_sizes == [24]


def test_loose_cheap_bound_uses_sampled_away_wall_proof_to_retain_headroom():
    # The full body moves away from a straight wall. Subtracting all travel
    # from its exact initial clearance gives a valid but unnecessarily loose
    # cheap proof. A work-age reserve can erase that proof even though the
    # independently integrated whole envelope remains much farther away.
    model = CapObservedModel([[[.3, -2.], [.3, 2.]]], rectangle(.1, .1))
    pose, command = [0., 0., 0.], [-.4, 0., 0.]
    delay, linear, angular, margin, headroom = .12, .8, 1., .025, .08
    context = {}
    safe, complete, rejected = delayed_braking_certificate(
        model, pose, command, delay, linear, angular, margin=margin,
        query_headroom_m=headroom, diagnostics=context)
    initial = model.current_values[0]
    travel = .4*delay+.4**2/(2.*linear)
    cheap_bound = initial-travel-model.radius*.02-.005
    age_error = additional_delay_clearance_error(
        command, delay, linear, model.radius, .05)
    assert margin <= cheap_bound < margin+headroom
    assert cheap_bound-age_error < margin
    assert safe is True and rejected is None and complete-age_error >= margin
    assert context['proof_kind'] == 'sampled' and model.batch_sizes
    assert complete > cheap_bound
    _, points, yaws, _ = dense_oracle(
        pose, command, delay, linear, angular, model.radius, count=12001)
    dense_minimum = float(np.min(CadClearanceModel.clearance_over_poses(
        model, points, yaws, cap=2.)))
    assert dense_minimum == pytest.approx(initial, abs=1.e-12)
    assert complete <= dense_minimum-model.radius*.02-.005+1.e-8


def test_sampled_acceptance_keeps_original_margin_when_full_proof_lacks_headroom():
    model = CapObservedModel([[[.16, -2.], [.16, 2.]]], rectangle(.1, .1))
    context = {}
    safe, complete, rejected = delayed_braking_certificate(
        model, [0., 0., 0.], [-.2, 0., 0.], .12, .8, 1.,
        query_headroom_m=.1, diagnostics=context)
    assert safe is True and rejected is None
    assert .025 <= complete < .025+.1
    assert context['proof_kind'] == 'sampled'
    assert complete == pytest.approx(.06-model.radius*.02-.005, abs=1.e-12)


@pytest.mark.parametrize('wall_x,command,headroom,kind,exact', [
    (10., [.2, .05, .3], 1., 'cheap', False),
    (.3, [-.4, 0., 0.], .08, 'sampled', True),
    (.3, [.7, 0., 0.], .1, 'sampled_rejected', True),
])
def test_scalar_diagnostics_describe_actual_proof_without_extra_geometry(
        wall_x, command, headroom, kind, exact):
    args = ([0., 0., 0.], command, .12, .8, 1.)
    segments, footprint = [[[wall_x, -2.], [wall_x, 2.]]], rectangle(.1, .1)
    model = CapObservedModel(segments, footprint)
    baseline = CapObservedModel(segments, footprint)
    expected = delayed_braking_certificate(
        baseline, *args, query_headroom_m=headroom)
    context = {'stale': 42.}
    result = delayed_braking_certificate(
        model, *args, query_headroom_m=headroom, diagnostics=context)
    assert result == expected
    assert model.current_caps == baseline.current_caps
    assert model.path_caps == baseline.path_caps
    assert model.batch_sizes == baseline.batch_sizes
    assert context == {
        'proof_kind': kind,
        'current_clearance_m': model.current_values[0],
        'current_clearance_is_exact': exact,
        'total_combined_travel_m': pytest.approx(
            .12*(math.hypot(*command[:2])+model.radius*abs(command[2]))
            +math.hypot(*command[:2])**2/(2.*.8)
            +model.radius*abs(command[2])**2/2.),
        'margin_m': .025,
        'query_headroom_m': headroom,
        'current_query_cap_m': model.current_caps[0],
        'sampled_query_cap_m': model.path_caps[0] if model.path_caps else None,
        'complete_bound_m': result[1],
        'rejected_sample_bound_m': result[2],
    }
    assert all(value is None or isinstance(value, (str, bool))
               or math.isfinite(value) for value in context.values())
    if kind == 'sampled_rejected':
        assert context['complete_bound_m'] is None
        assert context['rejected_sample_bound_m'] < context['margin_m']
    else:
        assert context['complete_bound_m'] >= context['margin_m']
        assert context['rejected_sample_bound_m'] is None


@pytest.mark.parametrize('overrides', [
    {'pose': [math.nan, 0., 0.]},
    {'query_headroom_m': math.nan},
    {'command': [8., 0., 1.]},
])
def test_validation_failure_clears_previous_complete_proof_before_geometry(overrides):
    model = CapObservedModel([[[10., -2.], [10., 2.]]], rectangle(.1, .1))
    context = {'proof_kind': 'cheap', 'complete_bound_m': .5,
               'rejected_sample_bound_m': None}
    kwargs = dict(model=model, pose=[0., 0., 0.], command=[.2, 0., 0.],
                  delay_sec=.12, linear_deceleration=.8,
                  angular_deceleration=1., query_headroom_m=.1,
                  diagnostics=context)
    kwargs.update(overrides)
    with pytest.raises(ValueError):
        delayed_braking_certificate(**kwargs)
    assert context == {}
    assert model.current_caps == model.path_caps == []


def test_invalid_diagnostics_type_is_rejected_before_geometry():
    model = CapObservedModel([[[10., -2.], [10., 2.]]], rectangle(.1, .1))
    with pytest.raises(ValueError, match='diagnostics must be a dict'):
        delayed_braking_certificate(
            model, [0., 0., 0.], [.2, 0., 0.], .12, .8, 1., diagnostics=[])
    assert model.current_caps == model.path_caps == []
