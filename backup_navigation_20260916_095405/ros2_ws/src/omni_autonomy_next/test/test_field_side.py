import math
from pathlib import Path

import pytest
import yaml

from omni_autonomy_next.configured_goals import (
    load_configured_poses,
    load_remembered_defaults,
)
from omni_autonomy_next.field_side import (
    LEFT,
    RIGHT,
    apply_pose,
    mirror_goal_approaches,
    mirror_pose,
    mirror_poses,
    mirror_waypoint_routes,
    normalize_side,
    store_pose,
)
from omni_autonomy_next.route_approaches import (
    load_fixed_departures,
    load_fixed_goal_approaches,
    select_fixed_goal_approach,
)


CONFIG = Path(__file__).resolve().parents[1] / 'config'


def test_mirror_reflects_about_the_divider_at_x_zero():
    pose = {'frame_id': 'map', 'x': -0.81, 'y': 1.34, 'yaw': 0.25}
    mirrored = mirror_pose(pose)
    assert mirrored['x'] == pytest.approx(0.81)
    assert mirrored['y'] == pytest.approx(1.34)
    assert mirrored['yaw'] == pytest.approx(math.pi - 0.25)
    assert mirrored['frame_id'] == 'map'


@pytest.mark.parametrize('yaw', [0.0, 0.25, -0.25, math.pi / 2, -math.pi / 2,
                                 math.pi - 1e-9, -3.0, 3.0])
def test_mirror_is_its_own_inverse_and_stays_wrapped(yaw):
    pose = {'frame_id': 'map', 'x': -1.8, 'y': 4.75, 'yaw': yaw}
    once = mirror_pose(pose)
    # Same half-open convention as staged_heading.wrap: yaw=0 mirrors to -pi.
    assert -math.pi <= once['yaw'] < math.pi
    twice = mirror_pose(once)
    assert twice['x'] == pytest.approx(pose['x'])
    assert twice['y'] == pytest.approx(pose['y'])
    assert math.cos(twice['yaw']) == pytest.approx(math.cos(yaw), abs=1e-12)
    assert math.sin(twice['yaw']) == pytest.approx(math.sin(yaw), abs=1e-12)


def test_recording_on_the_right_field_stores_the_left_coordinate():
    """An adjustment made on either field must apply to both."""
    measured = {'frame_id': 'map', 'x': 0.79, 'y': 1.30, 'yaw': math.pi - 0.02}
    stored = store_pose(measured, RIGHT)
    assert stored['x'] == pytest.approx(-0.79)
    assert apply_pose(stored, RIGHT)['x'] == pytest.approx(measured['x'])
    assert apply_pose(stored, RIGHT)['yaw'] == pytest.approx(measured['yaw'])
    # A copy, never the record still held in the persistent store.
    left = apply_pose(stored, LEFT)
    assert left == stored and left is not stored


@pytest.mark.parametrize('value', ['', 'both', 'LEFTT', None, 0])
def test_unknown_field_side_is_rejected(value):
    with pytest.raises(ValueError):
        normalize_side(value)


@pytest.mark.parametrize('value,expected', [
    ('left', LEFT), ('RIGHT', RIGHT), (' Left ', LEFT),
])
def test_field_side_spelling_is_normalized(value, expected):
    assert normalize_side(value) == expected


def test_every_configured_pose_mirrors_into_the_opposite_field():
    poses = load_configured_poses(CONFIG / 'field_poses.yaml')
    mirrored = mirror_poses(poses)
    assert set(mirrored) == set(poses)
    for goal_id, pose in poses.items():
        # The deployed field poses are all on the left of the divider.
        assert pose['x'] < 0.0
        assert mirrored[goal_id]['x'] == pytest.approx(-pose['x'])
        assert mirrored[goal_id]['y'] == pytest.approx(pose['y'])
        assert mirrored[goal_id]['name'] == pose['name']


def test_mirrored_lanes_keep_their_upper_lower_roles():
    """The reflection preserves y, so split_y and the variants stay valid."""
    approaches = load_fixed_goal_approaches(CONFIG / 'routes.yaml')
    mirrored = mirror_goal_approaches(approaches)
    for goal_id, entry in approaches.items():
        assert mirrored[goal_id]['split_y'] == entry['split_y']
        for variant in ('upper', 'lower'):
            lane = entry[variant]
            mirrored_lane = mirrored[goal_id][variant]
            assert mirrored_lane['id'] == lane['id']
            assert mirrored_lane['departure_id'] == lane['departure_id']
            assert [p['y'] for p in mirrored_lane['waypoints']] == [
                p['y'] for p in lane['waypoints']
            ]
            assert [p['x'] for p in mirrored_lane['waypoints']] == [
                pytest.approx(-p['x']) for p in lane['waypoints']
            ]
        # Selecting from the same y on the mirrored field yields the same lane.
        for probe_y in (5.0, -5.0):
            left_id, _ = select_fixed_goal_approach(
                approaches, goal_id, (entry['upper']['waypoints'][0]['x'], probe_y)
            )
            right_id, _ = select_fixed_goal_approach(
                mirrored, goal_id,
                (-entry['upper']['waypoints'][0]['x'], probe_y),
            )
            assert left_id == right_id


def test_departure_lanes_mirror_without_losing_their_pose_ids():
    departures = load_fixed_departures(CONFIG / 'routes.yaml')
    mirrored = mirror_waypoint_routes(departures)
    assert set(mirrored) == set(departures)
    for pose_id, entry in departures.items():
        assert mirrored[pose_id]['id'] == entry['id']
        assert [p['x'] for p in mirrored[pose_id]['waypoints']] == [
            pytest.approx(-p['x']) for p in entry['waypoints']
        ]


def test_every_mu3_slot_has_a_seed_pose():
    """A slot with no pose would only fail when tapped during a match."""
    field_poses_file = CONFIG / 'field_poses.yaml'
    configured = load_configured_poses(field_poses_file)
    defaults = load_remembered_defaults(field_poses_file, configured)
    saved_names = yaml.safe_load(
        (CONFIG / 'mu3_navigation.yaml').read_text(encoding='utf-8')
    )['mu3_navigation']['ros__parameters']['saved_pose_names']

    assert len(saved_names) == 7
    assert len(set(saved_names)) == len(saved_names)
    for name in saved_names:
        assert name in defaults, name
        assert set(defaults[name]) == {'frame_id', 'x', 'y', 'yaw'}


def test_seed_pointing_at_an_unknown_goal_id_is_a_startup_error(tmp_path):
    path = tmp_path / 'field_poses.yaml'
    path.write_text(yaml.safe_dump({
        'frame_id': 'map',
        'poses': {0: {'name': 'a', 'configured': True,
                      'x': -1.0, 'y': 0.0, 'yaw': 0.0}},
        'remembered_pose_defaults': {'ghost': 42},
    }), encoding='utf-8')
    configured = load_configured_poses(path)
    with pytest.raises(ValueError, match='ghost'):
        load_remembered_defaults(path, configured)
