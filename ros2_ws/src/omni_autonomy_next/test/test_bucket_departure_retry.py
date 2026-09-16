"""Fixed-bucket exits survive retries/replacement goals inside the throat."""
from pathlib import Path

import pytest

from omni_autonomy_next.configured_goals import load_configured_poses
from omni_autonomy_next.route_approaches import (
    load_fixed_goal_approaches, select_fixed_goal_departure,
)

CONFIG = Path(__file__).resolve().parents[1] / 'config'


@pytest.mark.parametrize('origin_id', ['4', '5'])
@pytest.mark.parametrize('side,direction', [('upper', 1.), ('lower', -1.)])
@pytest.mark.parametrize('offset', [.25, .55])
def test_retry_keeps_unpassed_exit_gates(origin_id, side, direction, offset):
    poses = load_configured_poses(CONFIG / 'field_poses.yaml')
    routes = load_fixed_goal_approaches(CONFIG / 'routes.yaml')
    origin = poses[origin_id]
    position = (-.80, origin['y'] + direction*offset)
    destination = (-3., origin['y'] + direction*3.)
    route_id, gates = select_fixed_goal_departure(
        routes, poses, position, destination, .20)
    assert route_id == routes[origin_id][side]['departure_id']
    assert gates
    assert gates[-1] == routes[origin_id][side]['waypoints'][0]
    assert all(direction*(p['y']-position[1]) > .01 for p in gates)


@pytest.mark.parametrize('position', [(-1.1, -1.8), (-.8, -1.50), (-.8, -3.25)])
def test_off_lane_and_cleared_bucket_do_not_reenter_exit(position):
    poses = load_configured_poses(CONFIG / 'field_poses.yaml')
    routes = load_fixed_goal_approaches(CONFIG / 'routes.yaml')
    assert select_fixed_goal_departure(
        routes, poses, position, (-3., 3.), .20) == (None, [])


def test_reversing_destination_inside_lower_lane_keeps_upper_exit():
    poses = load_configured_poses(CONFIG / 'field_poses.yaml')
    routes = load_fixed_goal_approaches(CONFIG / 'routes.yaml')
    route_id, gates = select_fixed_goal_departure(
        routes, poses, (-.8, -2.9), (-1.8, 4.75), .20,
        destination_goal_id='1')
    assert route_id == 'fixed_bucket_3_to_upper'
    assert [p['y'] for p in gates] == [-1.95, -1.55]
