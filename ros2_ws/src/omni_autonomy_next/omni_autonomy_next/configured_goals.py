"""Pure loader and validation for named navigation goals."""

import math
from pathlib import Path

import yaml


def load_configured_poses(path):
    with Path(path).open(encoding='utf-8') as stream:
        data = yaml.safe_load(stream)
    frame_id = str(data.get('frame_id', 'map'))
    if not frame_id:
        raise ValueError('field pose frame_id must not be empty')
    configured = {}
    for key, value in data.get('poses', {}).items():
        if not value.get('configured', False):
            continue
        values = (float(value['x']), float(value['y']), float(value['yaw']))
        if not all(math.isfinite(item) for item in values):
            raise ValueError(f'configured pose {key} contains a non-finite value')
        configured[str(key)] = {
            'name': str(value.get('name', key)),
            'frame_id': frame_id,
            'x': values[0],
            'y': values[1],
            'yaw': values[2],
        }
    if not configured:
        raise ValueError('field pose file contains no configured poses')
    return configured


def load_remembered_defaults(path, configured):
    """Map saved-pose names to their seed pose, taken from configured goals.

    An unknown goal id is a startup error rather than a skipped entry: a slot
    that silently has no pose would only be discovered when the operator taps
    it during a match.
    """
    with Path(path).open(encoding='utf-8') as stream:
        data = yaml.safe_load(stream) or {}
    defaults = {}
    for name, raw_goal_id in (data.get('remembered_pose_defaults') or {}).items():
        goal_id = str(raw_goal_id).strip()
        pose = configured.get(goal_id)
        if pose is None:
            raise ValueError(
                f'remembered pose default {name!r} refers to unknown goal id '
                f'{goal_id!r}'
            )
        defaults[str(name).strip()] = {
            key: pose[key] for key in ('frame_id', 'x', 'y', 'yaw')
        }
    return defaults
