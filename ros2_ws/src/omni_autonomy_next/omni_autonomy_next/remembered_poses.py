"""Persistent, validated storage for operator-recorded navigation poses."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import tempfile
import unicodedata
from .field_side import apply_pose, normalize_side


FORMAT_VERSION = 1
MAX_POSE_NAME_LENGTH = 64


def normalize_pose_name(name: str) -> str:
    """Return a safe display/key name, or raise for unusable input."""
    if not isinstance(name, str):
        raise ValueError('pose name must be a string')
    normalized = name.strip()
    if not normalized:
        raise ValueError('pose name must not be empty')
    if len(normalized) > MAX_POSE_NAME_LENGTH:
        raise ValueError(
            f'pose name must be at most {MAX_POSE_NAME_LENGTH} characters'
        )
    if any(unicodedata.category(character) == 'Cc' for character in normalized):
        raise ValueError('pose name must not contain control characters')
    return normalized


def validate_remembered_pose(pose, *, name='pose', allow_fields=True) -> dict:
    """Normalize one JSON pose record and reject unsafe numeric values."""
    if not isinstance(pose, dict):
        raise ValueError(f'remembered pose {name!r} must be an object')
    raw_frame = pose.get('frame_id', '')
    if not isinstance(raw_frame, str):
        raise ValueError(f'remembered pose {name!r} frame_id must be a string')
    frame_id = raw_frame.strip()
    if not frame_id:
        raise ValueError(f'remembered pose {name!r} has no frame_id')
    try:
        if any(isinstance(pose.get(key), bool) for key in ('x', 'y', 'yaw')):
            raise ValueError('boolean pose coordinate')
        values = tuple(float(pose[key]) for key in ('x', 'y', 'yaw'))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f'remembered pose {name!r} must contain numeric x, y, and yaw'
        ) from error
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f'remembered pose {name!r} contains a non-finite value')
    normalized = {
        'frame_id': frame_id,
        'x': values[0],
        'y': values[1],
        'yaw': values[2],
    }
    if 'field_poses' in pose:
        fields = pose['field_poses']
        if not allow_fields or not isinstance(fields, dict):
            raise ValueError('invalid field-specific poses')
        normalized['field_poses'] = {}
        for side, value in fields.items():
            if side not in ('left', 'right'):
                raise ValueError('invalid field-specific pose side')
            field_pose = validate_remembered_pose(value, name=name, allow_fields=False)
            if field_pose['frame_id'] != frame_id:
                raise ValueError('field-specific pose frame mismatch')
            normalized['field_poses'][side] = field_pose
    return normalized


def resolve_remembered_pose(pose, side):
    """Resolve an independently measured map pose, falling back to reflection."""
    side = normalize_side(side)
    override = pose.get('field_poses', {}).get(side)
    if override is not None:
        return dict(override)
    base = {key: pose[key] for key in ('frame_id', 'x', 'y', 'yaw')}
    return apply_pose(base, side)


def remember_field_pose(previous, measured, side):
    """Update only one field; keep the legacy seed and the opposite side."""
    result = validate_remembered_pose(previous)
    fields = result.setdefault('field_poses', {})
    fields[normalize_side(side)] = validate_remembered_pose(measured, allow_fields=False)
    return validate_remembered_pose(result)


def load_remembered_poses(path) -> dict[str, dict]:
    """Load a pose collection. A missing file means no poses have been saved."""
    storage_path = Path(path).expanduser()
    if not storage_path.exists():
        return {}
    with storage_path.open(encoding='utf-8') as stream:
        data = json.load(stream)
    if not isinstance(data, dict) or data.get('version') != FORMAT_VERSION:
        raise ValueError('unsupported remembered-pose file format')
    raw_poses = data.get('poses', {})
    if not isinstance(raw_poses, dict):
        raise ValueError('remembered-pose collection must be an object')
    return _normalize_poses(raw_poses)


def _normalize_poses(poses):
    if not isinstance(poses, dict):
        raise ValueError('remembered-pose collection must be an object')
    normalized = {}
    for name, pose in poses.items():
        key = normalize_pose_name(name)
        if key in normalized:
            raise ValueError(f'duplicate remembered pose name after normalization: {key!r}')
        normalized[key] = validate_remembered_pose(pose, name=name)
    return normalized


def save_remembered_poses(path, poses) -> None:
    """Atomically replace the persistent pose collection."""
    storage_path = Path(path).expanduser()
    normalized = _normalize_poses(poses)
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {'version': FORMAT_VERSION, 'poses': normalized}
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode='w', encoding='utf-8', dir=storage_path.parent,
            prefix=f'.{storage_path.name}.', suffix='.tmp', delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, storage_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
