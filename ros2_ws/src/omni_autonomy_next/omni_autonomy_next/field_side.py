"""Mirror saved geometry between the left and right competition fields.

Configured poses, fixed routes and legacy operator-recorded poses use the
LEFT-field convention. Independently calibrated loading poses are stored in
their measured field's map coordinates and resolved in remembered_poses.
The divider occupies
x=[-0.300, +0.300] in the map frame (see config/field_layout.yaml), so the two
fields are related by a reflection about x=0: the CAD map already covers both
of them and no second map is needed.

Reflecting (x, y) -> (-x, y) turns the heading vector (cos yaw, sin yaw) into
(-cos yaw, sin yaw) = (cos(pi - yaw), sin(pi - yaw)), so yaw -> pi - yaw.  The
reflection is its own inverse, which is why the same function converts a stored
pose to the right field and a right-field measurement back to storage.
"""

import math


LEFT = 'left'
RIGHT = 'right'
SIDES = (LEFT, RIGHT)


def normalize_side(value):
    """Return 'left' or 'right'; anything else is a caller error."""
    side = str(value).strip().lower()
    if side not in SIDES:
        raise ValueError(f'field side must be one of {SIDES}, got {value!r}')
    return side


def mirror_xy(x, y):
    return -float(x), float(y)


def mirror_yaw(yaw):
    """Reflected heading, wrapped to [-pi, pi) like staged_heading.wrap."""
    return (math.pi - float(yaw) + math.pi) % (2 * math.pi) - math.pi


def mirror_pose(pose):
    """Reflect one {x, y, yaw, ...} record, preserving every other key."""
    mirrored = dict(pose)
    mirrored['x'], mirrored['y'] = mirror_xy(pose['x'], pose['y'])
    mirrored['yaw'] = mirror_yaw(pose['yaw'])
    return mirrored


def apply_pose(pose, side):
    """Convert a stored left-field pose into the requested field.

    Always a copy: the left-field case would otherwise hand the caller the
    record still held in the persistent pose store.
    """
    return dict(pose) if normalize_side(side) == LEFT else mirror_pose(pose)


def store_pose(pose, side):
    """Convert a pose measured on `side` back into left-field storage."""
    return apply_pose(pose, side)


def mirror_poses(poses):
    return {key: mirror_pose(pose) for key, pose in poses.items()}


def mirror_waypoint_routes(routes):
    """Reflect {id: {'waypoints': [...]}} entries such as fixed departures."""
    return {
        key: {**entry, 'waypoints': [mirror_pose(p) for p in entry['waypoints']]}
        for key, entry in routes.items()
    }


def mirror_goal_approaches(approaches):
    """Reflect the two-variant approach lanes.

    ``split_y`` and the upper/lower roles are unchanged: the reflection keeps y,
    so a lane that was above the split stays above it.
    """
    mirrored = {}
    for goal_id, entry in approaches.items():
        record = {'split_y': entry['split_y']}
        for variant in ('upper', 'lower'):
            lane = entry[variant]
            record[variant] = {
                **lane,
                'waypoints': [mirror_pose(p) for p in lane['waypoints']],
            }
        mirrored[goal_id] = record
    return mirrored
