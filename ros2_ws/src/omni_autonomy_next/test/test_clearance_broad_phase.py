"""Exact CAD broad-phase filtering preserves the all-wall distance queries."""
from pathlib import Path

import numpy as np
import pytest

from omni_autonomy_next.rl_residual import CadClearanceModel


class AllWallModel(CadClearanceModel):
    """Reference path bypasses only the new bounding-box candidate filter."""

    def _nearby_segments(self, points, reach):
        return np.arange(len(self.starts))


@pytest.mark.parametrize('cap', [.01, .06, .35, 2.])
def test_field_distances_match_all_wall_reference(cap):
    config = Path(__file__).resolve().parents[1] / 'config'
    model, reference = [cls.from_yaml(
        config / 'field_planning.yaml', config / 'competition_footprints.yaml')
        for cls in (CadClearanceModel, AllWallModel)]
    rng = np.random.default_rng(261008)
    points = rng.uniform([-5., -4.], [5., 5.], (513, 2))
    yaws = rng.uniform(-np.pi, np.pi, len(points))
    points[3, 0], points[31, 1], yaws[200] = np.nan, np.inf, np.nan
    assert np.array_equal(model.clearance_over_poses(points, yaws, cap),
                          reference.clearance_over_poses(points, yaws, cap))
    for point, yaw in zip(points[::7], yaws[::7]):
        assert model.body_clearance(point, yaw, cap) == reference.body_clearance(
            point, yaw, cap)


@pytest.mark.parametrize('wall', [
    [[-50., 0.], [50., 0.]],  # Both endpoints outside; wall crosses the body.
    [[-.02, 0.], [.02, 0.]],  # Both endpoints swallowed by the body.
    [[-.5, .23], [.5, .23]],  # A horizontal segment close to a long edge.
    [[.2, .2], [3., 3.]],    # A corner touches the outline.
    [[0., .4], [0., .4]],    # Degenerate wall remains a point obstacle.
])
def test_long_swallowed_touching_and_degenerate_segments_are_retained(wall):
    footprint = [[-.2, -.2], [.2, -.2], [.2, .2], [-.2, .2]]
    far = [[[100., 100.], [101., 100.]]]
    model = CadClearanceModel([wall, *far], footprint)
    reference = AllWallModel([wall, *far], footprint)
    points, yaws = np.zeros((3, 2)), np.array([0., .1, -.3])
    assert model.clearance_over_poses(points, yaws) == pytest.approx(
        reference.clearance_over_poses(points, yaws), abs=1.e-15)
    assert model.body_clearance(points[0], 0.) == reference.body_clearance(points[0], 0.)


def test_broad_phase_removes_distant_facets_without_dropping_boundary():
    footprint = [[-.2, -.2], [.2, -.2], [.2, .2], [-.2, .2]]
    model = CadClearanceModel(
        [[[.5, -10.], [.5, 10.]], [[10., 10.], [11., 10.]]], footprint)
    assert model._nearby_segments(np.zeros((1, 2)), .5).tolist() == [0]
    # Invalid members cannot move the finite batch's bounding box to origin.
    points = np.array([[np.nan, 0.], [10., 10.]])
    assert model.clearance_over_poses(points, np.zeros(2)).tolist() == [0., 0.]
