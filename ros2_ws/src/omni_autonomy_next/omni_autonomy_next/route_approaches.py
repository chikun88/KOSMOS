"""Validated, deterministic approach/departure lanes for configured goals."""

import math
from pathlib import Path

import yaml


def _load_waypoints(raw_waypoints, frame_id, label):
    if not isinstance(raw_waypoints, list) or not raw_waypoints:
        raise ValueError(f'{label} requires non-empty waypoints')
    waypoints = []
    for index, raw_pose in enumerate(raw_waypoints):
        try:
            pose = {
                'frame_id': frame_id,
                'x': float(raw_pose['x']),
                'y': float(raw_pose['y']),
                'yaw': float(raw_pose['yaw']),
            }
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f'{label} waypoint {index} is invalid') from error
        if not all(math.isfinite(pose[key]) for key in ('x', 'y', 'yaw')):
            raise ValueError(f'{label} waypoint {index} is not finite')
        waypoints.append(pose)
    return waypoints


def load_fixed_departures(path):
    """Load pose-specific lanes traversed before free planning."""
    with Path(path).open(encoding='utf-8') as stream:
        document = yaml.safe_load(stream) or {}
    frame_id = str(document.get('frame_id', 'map')).strip()
    if not frame_id:
        raise ValueError('routes frame_id must not be empty')
    loaded = {}
    for raw_pose_id, raw_entry in (
        document.get('fixed_departures', {}) or {}
    ).items():
        pose_id = str(raw_pose_id).strip()
        if not pose_id or not isinstance(raw_entry, dict):
            raise ValueError(f'fixed departure {pose_id!r} must be a mapping')
        route_id = str(raw_entry.get('id', '')).strip()
        if not route_id:
            raise ValueError(f'fixed departure {pose_id!r} requires an id')
        loaded[pose_id] = {
            'id': route_id,
            'waypoints': _load_waypoints(
                raw_entry.get('waypoints'), frame_id,
                f'fixed departure {pose_id!r}',
            ),
        }
    return loaded


def load_fixed_goal_approaches(path):
    """Load CAD-validated bidirectional lanes for configured goals.

    Invalid geometry is a startup error. Silently falling back to an arbitrary
    path beside a fixed bucket would defeat the route contract the operator
    selected.
    """
    with Path(path).open(encoding='utf-8') as stream:
        document = yaml.safe_load(stream) or {}
    frame_id = str(document.get('frame_id', 'map')).strip()
    if not frame_id:
        raise ValueError('routes frame_id must not be empty')
    loaded = {}
    for raw_goal_id, raw_entry in (
        document.get('fixed_goal_approaches', {}) or {}
    ).items():
        goal_id = str(raw_goal_id).strip()
        if not isinstance(raw_entry, dict):
            raise ValueError(f'fixed approach {goal_id!r} must be a mapping')
        split_y = float(raw_entry['split_y'])
        if not math.isfinite(split_y):
            raise ValueError(f'fixed approach {goal_id!r} has invalid split_y')
        variants = {}
        for side in ('upper', 'lower'):
            raw_variant = raw_entry.get(side)
            if not isinstance(raw_variant, dict):
                raise ValueError(
                    f'fixed approach {goal_id!r} requires {side!r}'
                )
            route_id = str(raw_variant.get('id', '')).strip()
            departure_id = str(
                raw_variant.get('departure_id', '')
            ).strip()
            if (
                not route_id
                or not departure_id
            ):
                raise ValueError(
                    f'fixed approach {goal_id!r}/{side} is incomplete'
                )
            waypoints = _load_waypoints(
                raw_variant.get('waypoints'), frame_id,
                f'fixed approach {goal_id!r}/{side}',
            )
            variants[side] = {
                'id': route_id,
                'departure_id': departure_id,
                'waypoints': waypoints,
            }
        loaded[goal_id] = {'split_y': split_y, **variants}
    return loaded


def select_fixed_pose_departure(
    departures,
    configured_poses,
    current_position,
    trigger_distance,
    destination_goal_id=None,
):
    """Select an exit lane when starting beside its configured pose."""
    if current_position is None:
        return None, []
    trigger_distance = max(0.0, float(trigger_distance))
    current_x, current_y = map(float, current_position[:2])
    if not all(math.isfinite(value) for value in (
        current_x, current_y, trigger_distance
    )):
        return None, []
    candidates = []
    for origin_pose_id, departure in departures.items():
        origin = configured_poses.get(str(origin_pose_id))
        if origin is None:
            continue
        origin_xy = (float(origin['x']), float(origin['y']))
        gate = departure['waypoints'][0]
        gate_xy = (float(gate['x']), float(gate['y']))
        dx, dy = gate_xy[0] - origin_xy[0], gate_xy[1] - origin_xy[1]
        length2 = dx * dx + dy * dy
        if length2 <= 1.0e-12:
            continue
        projection = (
            (current_x - origin_xy[0]) * dx
            + (current_y - origin_xy[1]) * dy
        ) / length2
        clamped = min(1.0, max(0.0, projection))
        nearest = (
            origin_xy[0] + clamped * dx,
            origin_xy[1] + clamped * dy,
        )
        lateral = math.hypot(current_x - nearest[0], current_y - nearest[1])
        distance_to_origin = math.hypot(
            current_x - origin_xy[0], current_y - origin_xy[1]
        )
        distance_to_gate = math.hypot(
            current_x - gate_xy[0], current_y - gate_xy[1]
        )
        # Keep the exit contract across a retry after the robot has already
        # moved more than trigger_distance from the configured pose. Dropping
        # the gate halfway down the lane would let the replacement plan start
        # rotating in the same slot this route exists to avoid.
        on_departure_lane = (
            0.0 <= projection < 1.0
            and lateral <= trigger_distance
            and distance_to_gate > trigger_distance
        )
        if distance_to_origin <= trigger_distance or on_departure_lane:
            candidates.append((lateral, str(origin_pose_id), departure))
    if not candidates:
        return None, []
    _, origin_pose_id, departure = min(candidates, key=lambda item: item[0])
    if (
        destination_goal_id is not None
        and str(destination_goal_id) == origin_pose_id
    ):
        return None, []
    return departure['id'], list(departure['waypoints'])


def match_fixed_goal_approach(approaches, configured_poses, destination,
                            tolerance=.30):
    """Recognize calibrated/saved/RViz targets beside a fixed-bucket goal.

    A saved pose has no numbered goal id. Match its location, not its name,
    so renamed presets retain the same lane and unrelated saved poses do not
    inherit a bucket route. This never replaces the requested destination.
    """
    x, y = map(float, destination[:2])
    candidates = []
    for goal_id in approaches:
        goal = configured_poses.get(str(goal_id))
        if goal is None:
            continue
        distance = math.hypot(x-float(goal['x']), y-float(goal['y']))
        if distance <= tolerance:
            candidates.append((distance, str(goal_id)))
    return min(candidates)[1] if candidates else None


def select_fixed_goal_approach(approaches, goal_id, current_position,
                               destination_position=None):
    """Select one of the two predetermined lanes from the robot's actual side."""
    entry = approaches.get(str(goal_id))
    if entry is None:
        return None, []
    # Before the first localization sample, use the upper variant. The normal
    # competition sequence approaches goals 4/5 from positioning_wait (point
    # 2), which is above both. As soon as localization is present the choice
    # is deterministic from the actual side of the goal.
    current_y = None if current_position is None else float(current_position[1])
    split_y = (float(entry['split_y']) if destination_position is None
               else float(destination_position[1]))
    side = (
        'upper'
        if current_y is None or current_y >= split_y
        else 'lower'
    )
    selected = entry[side]
    waypoints = list(selected['waypoints'])
    direction = -1. if side == 'upper' else 1.
    if destination_position is not None:
        # A calibrated goal may be before the last nominal gate. Do not drive
        # past the saved target and reverse inside the narrow bucket throat.
        waypoints = [p for p in waypoints
                     if direction*(split_y-float(p['y'])) >= -1.e-6]
    if current_position is not None:
        waypoints = remaining_lane_gates(waypoints, current_position,
                                        direction)
    return selected['id'], waypoints


def remaining_lane_gates(waypoints, position, direction):
    """Do not return to a gate already passed inside the same straight lane.

    Off-lane starts retain every gate: skipping them could cut a bucket corner.
    This only handles the configured constant-x, monotonic-y approach lanes.
    """
    if not waypoints:
        return waypoints
    x, y = map(float, position[:2])
    lane_x = float(waypoints[0]['x'])
    if (abs(x-lane_x) > .04
            or any(abs(float(p['x'])-lane_x) > 1.e-6 for p in waypoints)
            or any(direction*(b['y']-a['y']) < 0.
                   for a, b in zip(waypoints, waypoints[1:]))):
        return waypoints
    return [p for p in waypoints if direction*(float(p['y'])-y) > .01]


def select_fixed_goal_departure(
    approaches,
    configured_poses,
    current_position,
    destination_position,
    trigger_distance,
    destination_goal_id=None,
):
    """Select the predetermined lane for leaving fixed-bucket goal 4 or 5.

    The destination side selects the exit. Approach waypoints are stored from
    the outside toward the goal, so departure traverses them in reverse order.
    """
    if current_position is None or destination_position is None:
        return None, []
    trigger_distance = max(0.0, float(trigger_distance))
    current_x, current_y = map(float, current_position[:2])
    destination_y = float(destination_position[1])
    if not all(math.isfinite(value) for value in (
        current_x, current_y, destination_y, trigger_distance
    )):
        return None, []

    candidates = []
    for origin_goal_id, entry in approaches.items():
        origin = configured_poses.get(str(origin_goal_id))
        if origin is None:
            continue
        distance = math.hypot(
            current_x - float(origin['x']),
            current_y - float(origin['y']),
        )
        # A retry or replacement goal can arrive after leaving the 20 cm
        # origin disk but before clearing the bucket. Retain the exit contract
        # throughout either straight lane; otherwise free planning can cut
        # sideways through the bucket or request a turn inside its throat.
        # Use the same narrow lateral tolerance as remaining_lane_gates, not
        # the origin disk radius, so unrelated off-lane starts are excluded.
        on_lane = False
        for side in ('upper', 'lower'):
            gates = entry[side]['waypoints']
            lane_x = float(gates[0]['x'])
            gate_y = float(gates[0]['y'])
            origin_y = float(origin['y'])
            direction = 1. if side == 'upper' else -1.
            if (abs(current_x-lane_x) <= .04
                    and all(abs(float(p['x'])-lane_x) <= 1.e-6 for p in gates)
                    and direction*(current_y-origin_y) >= 0.
                    and direction*(gate_y-current_y) > .01):
                on_lane = True
                break
        if distance <= trigger_distance or on_lane:
            candidates.append((distance, str(origin_goal_id), entry))
    if not candidates:
        return None, []

    _, origin_goal_id, entry = min(candidates, key=lambda item: item[0])
    # Selecting the same numbered goal while already on it must not make the
    # robot leave through a gate and return.
    if (
        destination_goal_id is not None
        and str(destination_goal_id) == origin_goal_id
    ):
        return None, []
    side = (
        'upper'
        if destination_y >= float(entry['split_y'])
        else 'lower'
    )
    selected = entry[side]
    return (
        selected['departure_id'],
        remaining_lane_gates(list(reversed(selected['waypoints'])),
                             current_position, 1. if side == 'upper' else -1.),
    )
