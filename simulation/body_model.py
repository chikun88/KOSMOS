"""Yaw-dependent clearance of the real robot outline to the CAD field walls.

The offline model previously treated the robot as a disc of radius 0.39 m,
which is close to the *inscribed* radius of the deployed ten-vertex footprint.
Every vertex of that footprint is at least 0.510 m from base_link and the
corners reach 0.588 m, so a disc model cannot see the corner graze that
actually stops the robot in the narrow bucket lane.  It also cannot see that
the tight firing poses only fit at one specific yaw.

Clearance here is the exact minimum distance between the rotated footprint
polygon and the CAD wall segments.  Zero means contact.  Values are saturated
at a caller-supplied cap because only the near field is ever gated on, and the
cap lets an open-field pose skip the exact test entirely.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import yaml


DEFAULT_CAP = 0.35
_BUCKET = 0.5


def _point_segment_distance(point, start, delta, length2):
    t = np.clip(
        np.einsum('...i,...i->...', point - start, delta) / length2, 0.0, 1.0
    )
    offset = point - (start + t[..., None] * delta)
    return np.sqrt(np.einsum('...i,...i->...', offset, offset))


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


def load_wall_segments(path):
    with Path(path).open(encoding='utf-8') as stream:
        data = yaml.safe_load(stream)
    return np.asarray(
        [[wall['start'], wall['end']] for wall in data['field']['walls']],
        dtype=float,
    )


class BodyClearanceModel:
    """Exact footprint-to-wall clearance behind a precomputed broad phase."""

    def __init__(self, segments, footprint, cap=DEFAULT_CAP):
        segments = np.asarray(segments, dtype=float)
        if segments.ndim != 3 or segments.shape[1:] != (2, 2) or len(segments) == 0:
            raise ValueError('wall segments must have shape (N, 2, 2)')
        if not np.all(np.isfinite(segments)):
            raise ValueError('wall segments must be finite')
        self.cap = float(cap)
        if not 0.0 < self.cap <= 2.0:
            raise ValueError('clearance cap must be within (0, 2]')

        self.footprint = np.asarray(footprint, dtype=float)
        if (self.footprint.ndim != 2 or self.footprint.shape[1] != 2
                or len(self.footprint) < 3 or not np.isfinite(self.footprint).all()):
            raise ValueError('footprint must be a finite polygon of [x, y] points')
        self._edge_delta = np.roll(self.footprint, -1, axis=0) - self.footprint
        if np.any(np.linalg.norm(self._edge_delta, axis=1) <= 1.0e-9):
            raise ValueError('footprint must have distinct adjacent vertices')
        relative = self.footprint[None, :, :] - self.footprint[:, None, :]
        cross = self._edge_delta[:, None, 0] * relative[:, :, 1] - (
            self._edge_delta[:, None, 1] * relative[:, :, 0])
        next_vertex = np.roll(self.footprint, -1, axis=0)
        area2 = np.sum(self.footprint[:, 0] * next_vertex[:, 1]
                       - self.footprint[:, 1] * next_vertex[:, 0])
        if abs(area2) <= 1.0e-12 or not (
                np.all(cross >= -1.0e-12) or np.all(cross <= 1.0e-12)):
            raise ValueError('footprint must be a nondegenerate convex polygon')
        self.radius = float(np.max(np.linalg.norm(self.footprint, axis=1)))
        self.inscribed_radius = float(np.min(_point_segment_distance(
            np.zeros((len(self.footprint), 2)),
            self.footprint,
            self._edge_delta,
            np.einsum('ij,ij->i', self._edge_delta, self._edge_delta),
        )))

        self.starts = segments[:, 0, :]
        self.ends = segments[:, 1, :]
        self.deltas = self.ends - self.starts
        length2 = np.einsum('ij,ij->i', self.deltas, self.deltas)
        self._length2 = np.where(length2 < 1.0e-12, 1.0e-12, length2)
        self.minimum = segments.reshape(-1, 2).min(axis=0)
        self.maximum = segments.reshape(-1, 2).max(axis=0)

        self._reach = self.radius + self.cap
        self._buckets = self._build_buckets()

    @classmethod
    def from_yaml(cls, field_file, footprint_file, profile='NORMAL', cap=DEFAULT_CAP):
        return cls(
            load_wall_segments(field_file),
            load_footprint(footprint_file, profile),
            cap=cap,
        )

    def _build_buckets(self):
        """Per-bucket wall subsets, so a query is one dictionary lookup.

        A wall registered for a bucket is within ``self._reach`` of some point
        in that bucket, so no wall that could touch the footprint is missed.
        """
        low = np.floor(self.minimum / _BUCKET).astype(int) - 1
        high = np.ceil(self.maximum / _BUCKET).astype(int) + 1
        buckets = {}
        for ix in range(low[0], high[0] + 1):
            for iy in range(low[1], high[1] + 1):
                corner = np.asarray([ix * _BUCKET, iy * _BUCKET])
                # Largest distance from any point of the bucket square to a
                # wall is bounded by the corner distance plus the diagonal.
                centre = corner + 0.5 * _BUCKET
                distance = _point_segment_distance(
                    centre[None, :], self.starts, self.deltas, self._length2
                )
                keep = np.flatnonzero(
                    distance <= self._reach + _BUCKET * math.sqrt(2.0)
                )
                if len(keep):
                    buckets[(ix, iy)] = (
                        self.starts[keep], self.ends[keep], self.deltas[keep],
                        self._length2[keep],
                    )
        return buckets

    def _bucket(self, point):
        return (
            int(math.floor(point[0] / _BUCKET)),
            int(math.floor(point[1] / _BUCKET)),
        )

    def centre_clearance_and_gradient(self, point):
        """Distance from base_link to the nearest wall, and its unit normal."""
        point = np.asarray(point, dtype=float)
        local = self._buckets.get(self._bucket(point))
        if local is None:
            starts, ends, deltas, length2 = (
                self.starts, self.ends, self.deltas, self._length2
            )
        else:
            starts, ends, deltas, length2 = local
        t = np.clip(
            np.einsum('ij,ij->i', point[None, :] - starts, deltas) / length2,
            0.0, 1.0,
        )
        offsets = point[None, :] - (starts + t[:, None] * deltas)
        distances = np.sqrt(np.einsum('ij,ij->i', offsets, offsets))
        index = int(np.argmin(distances))
        clearance = float(distances[index])
        gradient = np.zeros(2, dtype=float)
        if clearance > 1.0e-9:
            gradient = offsets[index] / clearance
        return clearance, gradient

    def rotated_footprint(self, point, yaw):
        c, s = math.cos(float(yaw)), math.sin(float(yaw))
        return np.asarray([
            [c, s], [-s, c],
        ]).T.dot(self.footprint.T).T + np.asarray(point, dtype=float)

    def clearance(self, point, yaw):
        """Exact footprint-to-wall distance, saturated at ``self.cap``."""
        point = np.asarray(point, dtype=float)
        if not np.all(np.isfinite(point)) or not math.isfinite(float(yaw)):
            return 0.0
        local = self._buckets.get(self._bucket(point))
        if local is None:
            return self.cap
        starts, ends, deltas, length2 = local
        # Valid early-out: every footprint point lies within self.radius of
        # base_link, so the body can never be closer than this.
        centre = _point_segment_distance(point[None, :], starts, deltas, length2)
        near = np.flatnonzero(centre <= self._reach)
        if len(near) == 0:
            return self.cap
        if float(centre[near].min()) - self.radius >= self.cap:
            return self.cap
        polygon = self.rotated_footprint(point, yaw)
        # Only a wall closer to base_link than the circumscribed radius can be
        # crossed or swallowed by the footprint, so the overlap tests are
        # usually skipped outright.
        overlap = near[centre[near] < self.radius]
        value = self._polygon_distance(
            polygon, starts[near], ends[near], deltas[near], length2[near],
            starts[overlap], ends[overlap],
        )
        return min(value, self.cap)

    def clearance_batch(self, points, yaws, cap=None):
        """Exact clearance for a pose sequence in one vectorised pass.

        The Collision Monitor projection needs two dozen poses per control
        step.  Evaluating them together costs little more than a single pose
        because the work is dominated by array-call overhead, not by size.

        A caller that only needs the sign of the clearance can pass a small
        ``cap``, which shrinks the candidate wall set and so the whole pass.
        """
        points = np.asarray(points, dtype=float)
        yaws = np.asarray(yaws, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError('points must have shape (S, 2)')
        if yaws.shape != (len(points),):
            raise ValueError('yaws must have shape (S,)')
        cap = self.cap if cap is None else float(cap)
        if not math.isfinite(cap) or cap <= 0.0:
            raise ValueError('clearance cap must be finite and positive')
        cap = min(cap, self.cap)
        reach = self.radius + cap
        result = np.full(len(points), cap, dtype=float)
        finite = np.all(np.isfinite(points), axis=1) & np.isfinite(yaws)
        result[~finite] = 0.0
        if not np.any(finite):
            return result

        centre = _point_segment_distance(
            points[:, None, :], self.starts[None, :, :],
            self.deltas[None, :, :], self._length2[None, :],
        )
        near = np.flatnonzero(np.any(centre <= reach, axis=0))
        if len(near) == 0:
            return result
        starts = self.starts[near]
        ends = self.ends[near]
        deltas = self.deltas[near]
        length2 = self._length2[near]

        cosines = np.cos(yaws)
        sines = np.sin(yaws)
        # (S, V, 2) rotated footprints.
        polygons = np.stack((
            cosines[:, None] * self.footprint[None, :, 0]
            - sines[:, None] * self.footprint[None, :, 1],
            sines[:, None] * self.footprint[None, :, 0]
            + cosines[:, None] * self.footprint[None, :, 1],
        ), axis=-1) + points[:, None, :]
        rotated_delta = np.roll(polygons, -1, axis=1) - polygons

        vertex_to_wall = _point_segment_distance(
            polygons[:, :, None, :], starts[None, None, :, :],
            deltas[None, None, :, :], length2[None, None, :],
        )
        edge_length2 = np.einsum('svi,svi->sv', rotated_delta, rotated_delta)
        endpoint_to_edge = np.minimum(
            _point_segment_distance(
                starts[None, None, :, :], polygons[:, :, None, :],
                rotated_delta[:, :, None, :], edge_length2[:, :, None],
            ),
            _point_segment_distance(
                ends[None, None, :, :], polygons[:, :, None, :],
                rotated_delta[:, :, None, :], edge_length2[:, :, None],
            ),
        )
        distance = np.minimum(
            vertex_to_wall.min(axis=(1, 2)), endpoint_to_edge.min(axis=(1, 2))
        )
        np.minimum(result, distance, out=result, where=finite)
        result[~finite] = 0.0

        # Overlap only matters where a wall is inside the circumscribed circle.
        overlap_rows = np.flatnonzero(
            finite & (result > 0.0) & np.any(centre[:, near] < self.radius, axis=1)
        )
        for row in overlap_rows:
            local = near[centre[row, near] < self.radius]
            if self._crosses(
                polygons[row], rotated_delta[row],
                self.starts[local], self.ends[local],
            ) or self._contains_any(
                polygons[row], self.starts[local]
            ) or self._contains_any(polygons[row], self.ends[local]):
                result[row] = 0.0
        return result

    def _polygon_distance(
        self, polygon, starts, ends, deltas, length2,
        overlap_starts, overlap_ends,
    ):
        """Exact convex-polygon to segment-set distance."""
        # The closest pair between a convex polygon and a segment always uses a
        # vertex of one and a feature of the other, unless the pair crosses.
        # These edge vectors must come from the rotated polygon, not from the
        # body-frame footprint.
        rotated_delta = np.roll(polygon, -1, axis=0) - polygon
        edge_start = polygon[:, None, :]
        edge_delta = rotated_delta[:, None, :]
        edge_length2 = np.einsum(
            'ij,ij->i', rotated_delta, rotated_delta
        )[:, None]
        vertex_to_wall = _point_segment_distance(
            polygon[:, None, :], starts[None, :, :], deltas[None, :, :],
            length2[None, :],
        )
        endpoint_to_edge = np.minimum(
            _point_segment_distance(
                starts[None, :, :], edge_start, edge_delta, edge_length2
            ),
            _point_segment_distance(
                ends[None, :, :], edge_start, edge_delta, edge_length2
            ),
        )
        distance = float(min(vertex_to_wall.min(), endpoint_to_edge.min()))
        if distance <= 0.0:
            return 0.0
        if len(overlap_starts) == 0:
            return distance
        if self._crosses(polygon, rotated_delta, overlap_starts, overlap_ends):
            return 0.0
        if (
            self._contains_any(polygon, overlap_starts)
            or self._contains_any(polygon, overlap_ends)
        ):
            # A short wall segment swallowed whole by the footprint crosses no
            # edge, so the edge tests alone would report clearance.
            return 0.0
        return distance

    @staticmethod
    def _crosses(polygon, rotated_delta, starts, ends):
        p1 = polygon[:, None, :]
        d1 = rotated_delta[:, None, :]
        q1 = starts[None, :, :]
        d2 = (ends - starts)[None, :, :]
        denom = d1[..., 0] * d2[..., 1] - d1[..., 1] * d2[..., 0]
        parallel = np.abs(denom) < 1.0e-12
        safe = np.where(parallel, 1.0, denom)
        diff = q1 - p1
        t = (diff[..., 0] * d2[..., 1] - diff[..., 1] * d2[..., 0]) / safe
        u = (diff[..., 0] * d1[..., 1] - diff[..., 1] * d1[..., 0]) / safe
        return bool(np.any(
            ~parallel & (t >= 0.0) & (t <= 1.0) & (u >= 0.0) & (u <= 1.0)
        ))

    @staticmethod
    def _contains_any(polygon, points):
        edge = np.roll(polygon, -1, axis=0) - polygon
        relative = points[:, None, :] - polygon[None, :, :]
        cross = (
            edge[None, :, 0] * relative[:, :, 1]
            - edge[None, :, 1] * relative[:, :, 0]
        )
        return bool(np.any(
            np.all(cross >= -1.0e-12, axis=1)
            | np.all(cross <= 1.0e-12, axis=1)))

    def in_bounds(self, point, yaw):
        polygon = self.rotated_footprint(point, yaw)
        return bool(
            np.all(polygon >= self.minimum[None, :])
            and np.all(polygon <= self.maximum[None, :])
        )

    def support(self, yaw, direction):
        """Footprint extent along a world-frame unit direction."""
        c, s = math.cos(float(yaw)), math.sin(float(yaw))
        world = np.asarray([[c, s], [-s, c]]).T.dot(self.footprint.T).T
        return float(np.max(world @ np.asarray(direction, dtype=float)))
