import json
import math
from pathlib import Path

import pytest

from omni_autonomy_next.remembered_poses import (
    load_remembered_poses,
    normalize_pose_name,
    save_remembered_poses,
    remember_field_pose,
    resolve_remembered_pose,
)


def test_remembered_pose_round_trip_is_persistent_and_unicode_safe(tmp_path):
    path = tmp_path / 'state' / 'remembered.json'
    poses = {
        ' 作業台前 ': {
            'frame_id': 'map',
            'x': 1.25,
            'y': -2.5,
            'yaw': math.pi / 3.0,
        },
    }

    save_remembered_poses(path, poses)

    assert load_remembered_poses(path) == {
        '作業台前': {
            'frame_id': 'map',
            'x': 1.25,
            'y': -2.5,
            'yaw': math.pi / 3.0,
        },
    }
    assert json.loads(path.read_text(encoding='utf-8'))['version'] == 1
    assert not list(path.parent.glob('*.tmp'))


def test_missing_remembered_pose_file_starts_empty(tmp_path):
    assert load_remembered_poses(tmp_path / 'missing.json') == {}


def test_loading_calibration_is_independent_and_survives_restart(tmp_path):
    legacy = dict(frame_id='map', x=-1.8, y=4.5, yaw=3.1)
    left = dict(frame_id='map', x=-1.9, y=4.4, yaw=3.0)
    right = dict(frame_id='map', x=1.85, y=4.6, yaw=.03)
    original_right = resolve_remembered_pose(legacy, 'right')
    pose = remember_field_pose(legacy, left, 'left')
    assert resolve_remembered_pose(pose, 'right') == original_right
    pose = remember_field_pose(pose, right, 'right')
    assert resolve_remembered_pose(pose, 'left') == left
    assert resolve_remembered_pose(pose, 'right') == right
    assert 'field_poses' not in legacy
    path = tmp_path/'poses.json'
    save_remembered_poses(path, {'A': pose})
    loaded = load_remembered_poses(path)
    new_left = {**left, 'y': 4.42}
    loaded['A'] = remember_field_pose(loaded['A'], new_left, 'left')
    save_remembered_poses(path, loaded)
    restored = load_remembered_poses(path)['A']
    assert resolve_remembered_pose(restored, 'left') == new_left
    assert resolve_remembered_pose(restored, 'right') == right


@pytest.mark.parametrize('fields', [
    {'right': dict(frame_id='map', x=math.nan, y=0., yaw=0.)},
    {'right': dict(frame_id='odom', x=1., y=0., yaw=0.)},
    {'unknown': dict(frame_id='map', x=1., y=0., yaw=0.)},
    [],
])
def test_invalid_field_calibration_does_not_overwrite_file(tmp_path, fields):
    path = tmp_path/'poses.json'
    pose = dict(frame_id='map', x=-1., y=0., yaw=0.)
    save_remembered_poses(path, {'A': pose})
    before = path.read_bytes()
    with pytest.raises(ValueError):
        save_remembered_poses(path, {'A': {**pose, 'field_poses': fields}})
    assert path.read_bytes() == before


@pytest.mark.parametrize('name', ['', '   ', 'bad\nname', 'x' * 65])
def test_invalid_remembered_pose_names_are_rejected(name):
    with pytest.raises(ValueError):
        normalize_pose_name(name)


@pytest.mark.parametrize('value', [math.nan, math.inf, -math.inf])
def test_non_finite_remembered_pose_coordinates_are_rejected(tmp_path, value):
    with pytest.raises(ValueError):
        save_remembered_poses(tmp_path / 'poses.json', {
            'bad': {'frame_id': 'map', 'x': value, 'y': 0.0, 'yaw': 0.0},
        })


def test_corrupt_or_wrong_version_file_is_rejected(tmp_path):
    path = tmp_path / 'poses.json'
    path.write_text('{"version": 99, "poses": {}}', encoding='utf-8')
    with pytest.raises(ValueError, match='unsupported'):
        load_remembered_poses(path)


def test_goal_bridge_and_gui_expose_remember_and_recall_topics():
    package_root = Path(__file__).resolve().parents[1]
    goal_source = (
        package_root / 'omni_autonomy_next' / 'goal_bridge_node.py'
    ).read_text(encoding='utf-8')
    gui_source = (
        package_root / 'omni_autonomy_next' / 'speed_gui_node.py'
    ).read_text(encoding='utf-8')
    launch_source = (package_root / 'launch' / 'system.launch.py').read_text(
        encoding='utf-8'
    )

    for topic in (
        '/navigation/remember_pose_request',
        '/navigation/remembered_goal_request',
        '/navigation/remembered_poses',
        # The panel must be able to select the field it is recording on, or a
        # pose measured on the right field is stored as a left-field pose.
        '/navigation/field_side',
    ):
        assert topic in goal_source
        assert topic in gui_source
    assert 'self.current_pose_received_at' in goal_source
    assert "'remembered_poses_file': remembered_poses_file" in launch_source
