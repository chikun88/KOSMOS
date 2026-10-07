"""Body-clearance steering must use the obstacle nearest the entire outline."""
import math
from pathlib import Path

import numpy as np
import pytest

from omni_autonomy_next.footprint_gradient import footprint_outward_direction
from omni_autonomy_next.rl_residual import CadClearanceModel, apply_clearance_residual
from simulation.body_model import BodyClearanceModel, load_footprint, load_wall_segments


CONFIG = Path(__file__).resolve().parents[1] / 'config'
SQUARE = np.array([[-.5, -.5], [.5, -.5], [.5, .5], [-.5, .5]])


def paired_models(walls, polygon=SQUARE):
    return CadClearanceModel(walls, polygon), BodyClearanceModel(walls, polygon)


def direction(model, point, yaw):
    if isinstance(model, CadClearanceModel):
        return model.body_clearance_and_gradient(point, yaw)
    return model.clearance_and_gradient(point, yaw)


def residual(command, clearance, gradient, yaw=0., baseline=True):
    return apply_clearance_residual(
        command, yaw=yaw, body_clearance=clearance, gradient=gradient,
        clearance_push=1. if baseline else 1.6, repulsion_edge=.06,
        repulsion_authority=.3, include_baseline=baseline)


def test_recorded_bucket_lane_pose_uses_body_obstacle_not_nearest_centre_wall():
    walls = load_wall_segments(CONFIG / 'field_planning.yaml')
    footprint = load_footprint(CONFIG / 'competition_footprints.yaml')
    point = np.array([-.825, .825])
    runtime, simulation = paired_models(walls, footprint)
    # The centre is nearer the dividing wall on the right, while the footprint
    # is only 23.558 mm from the bucket on the left. Their normals are opposite.
    assert runtime.clearance_and_gradient(point)[1] == pytest.approx([-1., 0.])
    for model in (runtime, simulation):
        clearance, outward = direction(model, point, 0.)
        assert clearance == pytest.approx(.023558)
        assert outward == pytest.approx([1., 0.])
        assert residual([.1, 0.], clearance, outward) == pytest.approx([.1, 0.])
        assert residual([-.1, 0.], clearance, outward) == pytest.approx([-.081779, 0.])


@pytest.mark.parametrize('yaw', [0., math.pi / 2, -.47])
def test_rotated_asymmetric_outline_direction_is_in_map_frame(yaw):
    polygon = np.array([[-.2, -.3], [.7, -.3], [.7, .2], [-.2, .2]])
    c, s = math.cos(yaw), math.sin(yaw)
    rotation = np.array([[c, -s], [s, c]])
    point = np.array([1.1, -.4])
    wall = (np.array([[.74, -2.], [.74, 2.]]) @ rotation.T) + point
    expected = rotation @ [-1., 0.]
    for model in paired_models([wall], polygon):
        clearance, outward = direction(model, point, yaw)
        assert clearance == pytest.approx(.04)
        assert outward == pytest.approx(expected)
        # The correction rotates into map coordinates and back exactly once.
        assert residual([-.1, .07], clearance, outward, yaw) == pytest.approx([-.1, .07])
        assert residual([.1, .07], clearance, outward, yaw) == pytest.approx([.09, .07])


def test_wall_endpoint_against_polygon_edge_uses_polygon_minus_wall_normal():
    # The nearest pair is the wall's first endpoint and the middle of the top
    # polygon edge; no polygon vertex is closest to this short wall.
    for model in paired_models([[[0., .54], [0., .7]]]):
        clearance, outward = direction(model, [0., 0.], 0.)
        assert clearance == pytest.approx(.04)
        assert outward == pytest.approx([0., -1.])


@pytest.mark.parametrize('mirror', [1., -1.])
def test_corner_clearance_direction_is_covariant_under_reflection(mirror):
    wall = np.array([[.53, .54], [.6, .6]]) * [mirror, 1.]
    for model in paired_models([wall]):
        clearance, outward = direction(model, [0., 0.], 0.)
        assert clearance == pytest.approx(.05)
        assert outward == pytest.approx([-.6 * mirror, -.8])


def test_equal_perpendicular_features_and_duplicate_facets_do_not_bias_direction():
    walls = [[[.54, -2.], [.54, 2.]], [[-2., .54], [2., .54]]]
    expected = np.array([-1., -1.]) / math.sqrt(2.)
    for order in (walls, walls[::-1], [walls[0]] * 7 + [walls[1]]):
        for model in paired_models(order):
            clearance, outward = direction(model, [0., 0.], 0.)
            assert clearance == pytest.approx(.04)
            assert outward == pytest.approx(expected)


def test_opposing_tied_features_have_no_arbitrary_outward_direction():
    walls = [[[.54, -2.], [.54, 2.]], [[-.54, -2.], [-.54, 2.]]]
    for model in paired_models(walls):
        clearance, outward = direction(model, [0., 0.], 0.)
        assert clearance == pytest.approx(.04)
        assert outward == pytest.approx([0., 0.])
        assert residual([.1, .02], clearance, outward) == pytest.approx([.1, .02])


def test_three_way_tie_rejects_a_mean_that_approaches_one_active_feature():
    # Three long walls each lie .04 m beyond a support feature. The average
    # of their normals approaches the +X-normal wall, so it must be rejected.
    normals = np.array([[1., 0.], [-.8, .6], [-.8, -.6]])
    walls = []
    for normal in normals:
        tangent = np.array([-normal[1], normal[0]])
        support = float(np.min(SQUARE @ normal))
        point = normal * (support - .04)
        walls.append([point - 3. * tangent, point + 3. * tangent])
    for model in paired_models(walls):
        clearance, outward = direction(model, [0., 0.], 0.)
        assert clearance == pytest.approx(.04)
        assert outward == pytest.approx([0., 0.])


@pytest.mark.parametrize('walls', [
    [[[.49, -2.], [.49, 2.]]],
    [[[-.01, 0.], [.01, 0.]]],
])
def test_intersection_or_swallowed_wall_has_zero_clearance_and_no_direction(walls):
    for model in paired_models(walls):
        clearance, outward = direction(model, [0., 0.], 0.)
        assert clearance == 0.
        assert outward == pytest.approx([0., 0.])


def test_saturated_open_clearance_has_no_unrelated_wall_direction():
    for model in paired_models([[[2., -2.], [2., 2.]]]):
        clearance, outward = direction(model, [0., 0.], 0.)
        assert clearance == pytest.approx(.35)
        assert outward == pytest.approx([0., 0.])


@pytest.mark.parametrize('point,yaw', [([math.nan, 0.], 0.), ([0., 0.], math.inf)])
def test_invalid_pose_returns_contact_without_a_direction(point, yaw):
    for model in paired_models([[[.54, -2.], [.54, 2.]]]):
        clearance, outward = direction(model, point, yaw)
        assert clearance == 0.
        assert outward == pytest.approx([0., 0.])


@pytest.mark.parametrize('change', [
    {'body_velocity': [math.nan, 0.]}, {'gradient': [math.inf, 0.]},
    {'gradient': [1., 0., 0.]}, {'yaw': math.inf},
    {'body_clearance': math.nan}, {'body_clearance': -.01},
    {'clearance_push': math.inf}, {'repulsion_authority': math.nan},
    {'repulsion_edge': math.nan},
])
def test_invalid_residual_inputs_raise_instead_of_publishing_nan(change):
    kwargs = dict(body_velocity=[.1, .1], yaw=0., body_clearance=.03,
                  gradient=[1., 0.], clearance_push=1., repulsion_edge=.06,
                  repulsion_authority=.3, include_baseline=True)
    kwargs.update(change)
    with pytest.raises(ValueError, match='finite'):
        apply_clearance_residual(**kwargs)


@pytest.mark.parametrize('baseline', [True, False])
@pytest.mark.parametrize('yaw', [0., .71, -2.])
def test_true_feature_residual_keeps_command_norm_bound(baseline, yaw):
    for command in ([.1, .2], [-.2, .1], [0., 0.]):
        adjusted = residual(command, .03, [1., 0.], yaw, baseline)
        assert np.isfinite(adjusted).all()
        assert np.linalg.norm(adjusted) <= np.linalg.norm(command) + 1.e-12


def test_helper_rejects_nonfinite_geometry_before_normalizing():
    with pytest.raises(ValueError, match='finite'):
        footprint_outward_direction(SQUARE, [[math.nan, 0.]], [[1., 0.]], .04)
