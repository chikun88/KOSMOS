"""Geometry and bounded command composition for the runtime RL residual."""

import math
from pathlib import Path

import numpy as np
import yaml

from .rl_policy import RLObservation
from .footprint_gradient import footprint_outward_direction


def wrap_angle(value):
    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


def _point_segment_distance(point, start, delta, length2):
    t = np.clip(
        np.einsum('...i,...i->...', point - start, delta) / length2, 0.0, 1.0
    )
    offset = point - (start + t[..., None] * delta)
    return np.sqrt(np.einsum('...i,...i->...', offset, offset))


class CadClearanceModel:
    """base_link and footprint clearance against the CAD wall segments.

    The learned policy is keyed on the clearance of the *footprint*, not of
    base_link.  Every vertex of the deployed ten-vertex outline is at least
    0.510 m from base_link and the corners reach 0.588 m, so a base_link
    distance says nothing about whether the body fits: the tight firing poses
    have 47-59 mm of body clearance at their commanded yaw and none at all a
    few degrees away.  The offline model this policy was trained against uses
    exactly the same measure.
    """

    def __init__(self, segments, footprint=None):
        array = np.asarray(segments, dtype=float)
        if array.ndim != 3 or array.shape[1:] != (2, 2) or len(array) == 0:
            raise ValueError('field segments must have shape (N, 2, 2)')
        if not np.all(np.isfinite(array)):
            raise ValueError('field segments must be finite')
        self.starts = array[:, 0, :]
        self.ends = array[:, 1, :]
        self.deltas = self.ends - self.starts
        self.length2 = np.einsum('ij,ij->i', self.deltas, self.deltas)
        self.length2[self.length2 < 1.0e-12] = 1.0e-12
        self.footprint = None
        self.radius = 0.0
        if footprint is not None:
            polygon = np.asarray(footprint, dtype=float)
            if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3:
                raise ValueError('footprint must be a polygon of [x, y] points')
            if not np.all(np.isfinite(polygon)):
                raise ValueError('footprint must be finite')
            edge = np.roll(polygon, -1, axis=0) - polygon
            if np.any(np.linalg.norm(edge, axis=1) <= 1.0e-9):
                raise ValueError('footprint must have distinct adjacent vertices')
            relative = polygon[None, :, :] - polygon[:, None, :]
            cross = edge[:, None, 0] * relative[:, :, 1] - (
                edge[:, None, 1] * relative[:, :, 0])
            area2 = np.sum(polygon[:, 0] * np.roll(polygon, -1, axis=0)[:, 1]
                           - polygon[:, 1] * np.roll(polygon, -1, axis=0)[:, 0])
            if abs(area2) <= 1.0e-12 or not (
                    np.all(cross >= -1.0e-12) or np.all(cross <= 1.0e-12)):
                raise ValueError('footprint must be a nondegenerate convex polygon')
            self.footprint = polygon
            self.radius = float(np.max(np.linalg.norm(polygon, axis=1)))

    @classmethod
    def from_yaml(cls, path, footprint_file=None, profile='NORMAL'):
        with Path(path).open(encoding='utf-8') as stream:
            data = yaml.safe_load(stream)
        segments = [
            (wall['start'], wall['end']) for wall in data['field']['walls']
        ]
        footprint = None
        if footprint_file is not None:
            footprint = load_footprint(footprint_file, profile)
        return cls(segments, footprint)

    def clearance_and_gradient(self, point):
        point = np.asarray(point, dtype=float)
        relative = point[None, :] - self.starts
        projection = np.einsum('ij,ij->i', relative, self.deltas) / self.length2
        nearest = self.starts + np.clip(projection, 0.0, 1.0)[:, None] * self.deltas
        offsets = point[None, :] - nearest
        distances = np.linalg.norm(offsets, axis=1)
        index = int(np.argmin(distances))
        clearance = float(distances[index])
        gradient = np.zeros(2, dtype=float)
        if clearance > 1.0e-9:
            gradient = offsets[index] / clearance
        return clearance, gradient

    def rotated_footprint(self, point, yaw):
        if self.footprint is None:
            raise ValueError('no footprint is configured')
        c, s = math.cos(float(yaw)), math.sin(float(yaw))
        return np.asarray([
            [c, s], [-s, c],
        ]).T.dot(self.footprint.T).T + np.asarray(point, dtype=float)

    def body_clearance(self, point, yaw, cap=0.35):
        """Exact footprint-to-wall distance, saturated at ``cap``.

        Zero means the outline touches a wall.
        """
        if self.footprint is None:
            raise ValueError('no footprint is configured')
        point = np.asarray(point, dtype=float)
        cap = float(cap)
        if not math.isfinite(cap) or cap <= 0.0:
            raise ValueError('clearance cap must be finite and positive')
        if not np.all(np.isfinite(point)) or not math.isfinite(float(yaw)):
            return 0.0
        centre = _point_segment_distance(
            point[None, :], self.starts, self.deltas, self.length2
        )
        near = np.flatnonzero(centre <= self.radius + cap)
        if len(near) == 0:
            return cap
        if float(centre[near].min()) - self.radius >= cap:
            return cap
        polygon = self.rotated_footprint(point, yaw)
        edge_delta = np.roll(polygon, -1, axis=0) - polygon
        edge_length2 = np.einsum('ij,ij->i', edge_delta, edge_delta)
        starts = self.starts[near]
        ends = self.ends[near]
        # The closest pair between a convex polygon and a segment always uses a
        # vertex of one and a feature of the other, unless the pair crosses.
        vertex_to_wall = _point_segment_distance(
            polygon[:, None, :], starts[None, :, :],
            self.deltas[near][None, :, :], self.length2[near][None, :],
        )
        endpoint_to_edge = np.minimum(
            _point_segment_distance(
                starts[None, :, :], polygon[:, None, :],
                edge_delta[:, None, :], edge_length2[:, None],
            ),
            _point_segment_distance(
                ends[None, :, :], polygon[:, None, :],
                edge_delta[:, None, :], edge_length2[:, None],
            ),
        )
        distance = float(min(vertex_to_wall.min(), endpoint_to_edge.min()))
        if distance <= 0.0:
            return 0.0
        overlap = near[centre[near] < self.radius]
        if len(overlap) and (
            _polygon_crosses(
                polygon, edge_delta, self.starts[overlap], self.ends[overlap]
            )
            or _polygon_contains_any(polygon, self.starts[overlap])
            or _polygon_contains_any(polygon, self.ends[overlap])
        ):
            # A short wall facet swallowed whole by the outline crosses no edge.
            return 0.0
        return min(distance, cap)

    def body_clearance_and_gradient(self, point, yaw, cap=0.35):
        """Footprint distance and its outward translation direction in map."""
        point = np.asarray(point, dtype=float)
        cap = float(cap)
        if point.shape != (2,):
            raise ValueError('footprint gradient point must be [x, y]')
        clearance = self.body_clearance(point, yaw, cap)
        if clearance <= 0. or clearance >= cap:
            return clearance, np.zeros(2)
        centre = _point_segment_distance(
            point[None, :], self.starts, self.deltas, self.length2)
        near = np.flatnonzero(centre <= self.radius + cap)
        direction = footprint_outward_direction(
            self.rotated_footprint(point, yaw), self.starts[near],
            self.ends[near], clearance)
        return clearance, direction

    def clearance_over_rotation(self, point, yaws, cap=0.35):
        """Exact polygon clearance at one position for multiple headings."""
        yaws = np.atleast_1d(np.asarray(yaws, dtype=float))
        return self.clearance_over_poses(
            np.broadcast_to(point, (len(yaws), 2)), yaws, cap)

    def clearance_over_poses(self, points, yaws, cap=0.35):
        """Batch the scalar polygon test, with bounded temporary memory.

        Preserve intersections, enclosed wall endpoints and nonfinite rejection.
        Chunking limits candidate wall unions for long routes.
        """
        points = np.asarray(points, dtype=float)
        yaws = np.atleast_1d(np.asarray(yaws, dtype=float))
        if yaws.ndim != 1 or points.shape != (len(yaws), 2):
            raise ValueError('points/yaws shape mismatch')
        if not len(yaws):
            return np.empty(0)
        return np.concatenate([
            self._clearance_pose_chunk(points[i:i+24], yaws[i:i+24], cap)
            for i in range(0, len(yaws), 24)])

    def _clearance_pose_chunk(self, points, yaws, cap):
        if self.footprint is None:
            raise ValueError('no footprint is configured')
        points = np.asarray(points, dtype=float)
        yaws = np.atleast_1d(np.asarray(yaws, dtype=float))
        if yaws.ndim != 1:
            raise ValueError('yaws must be one-dimensional')
        cap = float(cap)
        if not math.isfinite(cap) or cap <= 0.0:
            raise ValueError('clearance cap must be finite and positive')
        result = np.full(len(yaws), cap, dtype=float)
        if points.shape != (len(yaws), 2):
            raise ValueError('points must have shape (len(yaws), 2)')
        finite = np.isfinite(yaws) & np.isfinite(points).all(axis=1)
        result[~finite] = 0.0
        if not np.any(finite):
            return result

        points = np.where(finite[:, None], points, 0.)
        yaws = np.where(finite, yaws, 0.)

        centre = _point_segment_distance(
            points[:, None, :], self.starts[None, :, :],
            self.deltas[None, :, :], self.length2[None, :]
        )
        near = np.flatnonzero(np.any(centre[finite] <= self.radius + cap, axis=0))
        if len(near) == 0 or float(centre[finite][:, near].min()) - self.radius >= cap:
            return result
        starts = self.starts[near]
        ends = self.ends[near]

        cosines, sines = np.cos(yaws), np.sin(yaws)
        # (S, V, 2): the outline rotated by every candidate yaw at once.
        polygons = np.stack((
            cosines[:, None] * self.footprint[None, :, 0]
            - sines[:, None] * self.footprint[None, :, 1],
            sines[:, None] * self.footprint[None, :, 0]
            + cosines[:, None] * self.footprint[None, :, 1],
        ), axis=-1) + points[:, None, :]
        edge_delta = np.roll(polygons, -1, axis=1) - polygons
        edge_length2 = np.einsum('svi,svi->sv', edge_delta, edge_delta)

        vertex_to_wall = _point_segment_distance(
            polygons[:, :, None, :], starts[None, None, :, :],
            self.deltas[near][None, None, :, :],
            self.length2[near][None, None, :],
        )
        endpoint_to_edge = np.minimum(
            _point_segment_distance(
                starts[None, None, :, :], polygons[:, :, None, :],
                edge_delta[:, :, None, :], edge_length2[:, :, None],
            ),
            _point_segment_distance(
                ends[None, None, :, :], polygons[:, :, None, :],
                edge_delta[:, :, None, :], edge_length2[:, :, None],
            ),
        )
        distance = np.minimum(
            vertex_to_wall.min(axis=(1, 2)), endpoint_to_edge.min(axis=(1, 2))
        )
        np.minimum(result, distance, out=result, where=finite)
        result[~finite] = 0.0

        overlap = near[np.any(centre[finite][:, near] < self.radius, axis=0)]
        if len(overlap) == 0:
            return result
        swallowed = (
            _polygon_crosses(
                polygons, edge_delta,
                self.starts[overlap], self.ends[overlap],
            )
            | _polygon_contains_any(polygons, self.starts[overlap])
            | _polygon_contains_any(polygons, self.ends[overlap])
        )
        result[finite & (result > 0.0) & swallowed] = 0.0
        return result

    def rotation_headroom(self, point, yaw, span, samples=12):
        """How far the footprint can rotate from ``yaw`` before it touches.

        Returns an angle in ``[0, |span|]`` with the sign of ``span``, and
        never more than the outline really has: the answer is a lower bound,
        so a caller may rotate by it but is not promised the largest such
        angle.

        Two measurements, both bounded, because this runs inside a 20 Hz
        control loop on a Jetson that is already the reason the loop is not
        faster:

        * rotating by ``a`` moves no body point further than ``radius * a``,
          so the present clearance divided by the circumradius is an angle
          that is safe without looking at a single wall again.  In the open
          field that alone answers the question and nothing else runs.
        * beyond it, endpoint clearances certify every intervening angle.
          Footprint clearance is Lipschitz with constant ``radius`` in yaw:
          two endpoint clearances whose sum exceeds ``radius * angle_step``
          cover the complete interval. Clear samples alone cannot establish
          this: a corner may touch a wall between them.

        The first keeps the second from ever returning zero at a pose that
        does fit, which matters: a zero would leave the base unable to turn at
        a firing pose, which is the failure this whole limit exists to avoid.
        """
        span = float(span)
        samples = max(1, int(samples))
        if not math.isfinite(span) or span == 0.0:
            return 0.0
        direction = math.copysign(1.0, span)
        point = np.asarray(point, dtype=float)
        reach = abs(span)
        # Capped exactly at the clearance that would make the whole span safe,
        # so this call stays as cheap as the question allows.
        initial_clearance = self.body_clearance(
            point, yaw, cap=self.radius * reach)
        if initial_clearance <= 0.0:
            return 0.0
        guaranteed = initial_clearance / self.radius
        if guaranteed >= reach:
            return span
        offsets = direction * reach * (
            np.arange(1, samples + 1, dtype=float) / samples)
        values = self.clearance_over_rotation(
            point, float(yaw) + offsets,
            # Magnitude is required to certify the gaps between sample poses.
            cap=self.radius * reach / samples,
        )
        step = reach / samples
        previous_clearance = initial_clearance
        for index, clearance in enumerate(values):
            if previous_clearance + clearance <= self.radius * step:
                # Only the initial side of this interval is certified. Use a
                # strict bound so the returned endpoint also remains clear.
                safe = np.nextafter(previous_clearance / self.radius, 0.0)
                return direction * min(reach, index * step + safe)
            previous_clearance = float(clearance)
        return direction * reach


# Both predicates take a polygon of shape ``(..., V, 2)``.  A single outline
# gives a scalar answer and a stack of outlines gives one answer per outline,
# so the swept-rotation pass never has to loop over its candidate yaws in
# Python -- which is where its cost was, not in the array work.
def _polygon_crosses(polygon, edge_delta, starts, ends):
    p1 = polygon[..., :, None, :]
    d1 = edge_delta[..., :, None, :]
    d2 = ends - starts
    denom = d1[..., 0] * d2[..., 1] - d1[..., 1] * d2[..., 0]
    parallel = np.abs(denom) < 1.0e-12
    safe = np.where(parallel, 1.0, denom)
    diff = starts - p1
    t = (diff[..., 0] * d2[..., 1] - diff[..., 1] * d2[..., 0]) / safe
    u = (diff[..., 0] * d1[..., 1] - diff[..., 1] * d1[..., 0]) / safe
    return np.any(
        ~parallel & (t >= 0.0) & (t <= 1.0) & (u >= 0.0) & (u <= 1.0),
        axis=(-2, -1),
    )


def _polygon_contains_any(polygon, points):
    edge = np.roll(polygon, -1, axis=-2) - polygon
    relative = points - polygon[..., :, None, :]
    cross = (
        edge[..., :, None, 0] * relative[..., 1]
        - edge[..., :, None, 1] * relative[..., 0]
    )
    return np.any(
        np.all(cross >= -1.0e-12, axis=-2)
        | np.all(cross <= 1.0e-12, axis=-2), axis=-1)


def load_footprint(path, profile='NORMAL'):
    """Read a configured competition footprint polygon from its YAML source."""
    with Path(path).open(encoding='utf-8') as stream:
        data = yaml.safe_load(stream)
    entry = data['profiles'][profile]
    if not entry.get('configured', False) or entry.get('footprint') is None:
        raise ValueError(f'footprint profile {profile!r} is not configured')
    polygon = np.asarray(entry['footprint'], dtype=float)
    if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3:
        raise ValueError(f'footprint profile {profile!r} is not a polygon')
    if not np.all(np.isfinite(polygon)):
        raise ValueError(f'footprint profile {profile!r} has non-finite vertices')
    return polygon


def select_path_target(path, position, lookahead):
    points = np.asarray(path, dtype=float)
    position = np.asarray(position, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) == 0:
        raise ValueError('path must contain map-frame XY points')
    closest = int(np.argmin(np.linalg.norm(points - position[None, :], axis=1)))
    target = closest
    distance = 0.0
    while target + 1 < len(points) and distance < float(lookahead):
        distance += float(np.linalg.norm(points[target + 1] - points[target]))
        target += 1
    return points[target], points[-1]


def make_observation(
    *, position, yaw, body_velocity, target, goal, body_clearance,
    goal_clearance, goal_yaw, reference_speed,
):
    position = np.asarray(position, dtype=float)
    target = np.asarray(target, dtype=float)
    body_velocity = np.asarray(body_velocity, dtype=float)
    c, s = math.cos(float(yaw)), math.sin(float(yaw))
    map_velocity = np.asarray([
        c * body_velocity[0] - s * body_velocity[1],
        s * body_velocity[0] + c * body_velocity[1],
    ])
    direction = target - position
    speed = float(np.linalg.norm(map_velocity))
    turn_error = 0.0
    if float(np.linalg.norm(direction)) > 1.0e-9 and speed > 1.0e-3:
        turn_error = abs(wrap_angle(
            math.atan2(direction[1], direction[0])
            - math.atan2(map_velocity[1], map_velocity[0])
        ))
    return RLObservation(
        remaining_distance=float(np.linalg.norm(np.asarray(goal) - position)),
        clearance_margin=max(0.0, float(body_clearance)),
        turn_error=turn_error,
        speed_fraction=speed / max(float(reference_speed), 1.0e-6),
        goal_clearance_margin=max(0.0, float(goal_clearance)),
        yaw_error=abs(wrap_angle(float(goal_yaw) - float(yaw))),
    )


def limit_yaw_rate(
    yaw_rate, *, position, yaw, clearance_model, horizon, samples=24,
):
    """Cap a yaw rate to what Collision Monitor will let through unscaled.

    ``FootprintApproach`` projects the *commanded* twist forward for
    ``time_before_collision`` holding the angular velocity constant, and when
    that projection touches an obstacle it scales the whole twist -- the
    translation with it -- by ``contact_time / time_before_collision``.  Held
    for 1.2 s, the ``balanced`` profile's 1.30 rad/s sweeps 89 degrees, and
    this field's firing poses have 10-24 degrees of rotational headroom
    (measured from the CAD walls against the deployed ten-vertex outline at
    poses 1, 4, 5, 6 and 7).  So every ordinary yaw rate beside a wall is
    predicted to collide.

    That is what the reported crawl is.  The executed yaw rate settles at
    ``headroom / horizon`` -- about 8 deg/s -- whatever is asked, because the
    ratio falls as fast as the request rises.  The tracker's yaw error
    therefore grows, its P term raises the commanded yaw rate, the ratio
    shrinks again, and the translation is throttled along with it: at the yaw
    limit the whole twist passes at about 0.11, so 0.7 m/s of planned
    translation leaves as 0.08 m/s and the base stops covering ground.

    Asking for that rotation buys nothing, since the geometry, not the gate,
    is what bounds it.  Keeping the request inside the headroom the footprint
    actually has means the monitor passes the twist unscaled, so the
    translation keeps its planned speed while the rotation runs at the fastest
    rate the walls allow.  Collision Monitor is unchanged and keeps every stop
    right downstream; this only stops handing it a command it is certain to
    veto.
    """
    yaw_rate = float(yaw_rate)
    horizon = float(horizon)
    if (
        clearance_model is None
        or getattr(clearance_model, 'footprint', None) is None
        or not math.isfinite(yaw_rate)
        or not math.isfinite(horizon)
        or horizon <= 0.0
        or yaw_rate == 0.0
    ):
        return yaw_rate
    if not math.isfinite(float(yaw)) or not np.all(np.isfinite(
        np.asarray(position, dtype=float)
    )):
        return yaw_rate
    headroom = clearance_model.rotation_headroom(
        position, yaw, yaw_rate * horizon, samples=samples)
    return math.copysign(
        min(abs(yaw_rate), abs(headroom) / horizon), yaw_rate)


def apply_clearance_residual(
    body_velocity, *, yaw, body_clearance, gradient, clearance_push,
    repulsion_edge=0.10, repulsion_authority=0.45,
    include_baseline=False,
):
    """Redirect translation away from a wall without increasing its norm.

    MPPI already supplies the deterministic wall-repulsion turn represented in
    the offline model.  Its residual therefore applies only the *extra* turn
    requested by ``clearance_push`` and a value of 1.0 is inert.

    The trajectory tracker uses ``include_baseline=True`` to attenuate only
    wall-facing motion. This matters for a holonomic base beside a wall: Nav2's
    ``approach`` action scales the whole twist when even a small component aims
    into the wall, which otherwise removes the safe tangential component and
    can leave the robot crawling in place. Direction is changed but speed is
    never increased, and Collision Monitor remains the final sensor-based stop.
    """
    body_velocity = np.asarray(body_velocity, dtype=float)
    gradient = np.asarray(gradient, dtype=float)
    if (body_velocity.shape != (2,) or gradient.shape != (2,)
            or not np.isfinite(body_velocity).all() or not np.isfinite(gradient).all()
            or not all(math.isfinite(float(value)) for value in (
                yaw, body_clearance, clearance_push, repulsion_edge, repulsion_authority))
            or float(body_clearance) < 0.):
        raise ValueError('clearance residual requires finite planar motion/geometry')
    original_norm = float(np.linalg.norm(body_velocity))
    edge = float(repulsion_edge)
    if (
        original_norm <= 1.0e-9
        or (not include_baseline and float(clearance_push) <= 1.0)
        or float(body_clearance) >= edge
        or edge <= 0.0
    ):
        return body_velocity.copy()
    gradient = np.asarray(gradient, dtype=float)
    gradient_norm = float(np.linalg.norm(gradient))
    if gradient_norm <= 1.0e-9:
        return body_velocity.copy()

    ramp = max(0.0, (edge - float(body_clearance)) / edge)
    authority = float(repulsion_authority)
    baseline_weight = min(1.0, ramp * authority)
    pushed_weight = min(1.0, ramp * authority * float(clearance_push))
    weight = pushed_weight if include_baseline else pushed_weight - baseline_weight
    if weight <= 0.0:
        return body_velocity.copy()

    c, s = math.cos(float(yaw)), math.sin(float(yaw))
    map_velocity = np.asarray([
        c * body_velocity[0] - s * body_velocity[1],
        s * body_velocity[0] + c * body_velocity[1],
    ])
    # A wall-parallel request is already safe in this direction. Adding an
    # outward velocity biases the path follower off its checked corridor and
    # makes it steer back on every cycle. Attenuate only the inward component;
    # preserve tangential and outward motion, without restoring the old norm.
    if include_baseline:
        outward = gradient / gradient_norm
        inward = min(0.0, float(map_velocity @ outward))
        map_velocity -= weight * inward * outward
    else:
        # Preserve the trained extra steering residual for the MPPI path.
        blended = ((1.-weight)*map_velocity/original_norm
                   + weight*gradient/gradient_norm)
        norm = float(np.linalg.norm(blended))
        if norm <= 1.e-9:
            return body_velocity.copy()
        map_velocity = blended/norm*original_norm
    return np.asarray([
        c * map_velocity[0] + s * map_velocity[1],
        -s * map_velocity[0] + c * map_velocity[1],
    ])
