"""CAD-checked connectors between the fixed bucket approach/departure lanes."""

import heapq
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml

from .field_side import LEFT, RIGHT, apply_pose, normalize_side
from .route_approaches import _load_waypoints
from .staged_heading import dense_path


class FixedBucketTransit:
    """Select intermediate gates without changing the narrow terminal lanes.

    Nav2 still plans each leg against its live costmap. These gates prevent
    the free connector from changing sides of an intervening fixed bucket.
    The footprint is held at the lane heading; the tracker remains responsible
    for checking its actual heading schedule and any in-place rotation.
    """

    def __init__(self, model, waypoints):
        self.model = model
        self.waypoints = {
            side: [apply_pose(p, side) for p in waypoints]
            for side in (LEFT, RIGHT)
        }
        self.edges = {}
        for side, points in self.waypoints.items():
            edges = [[] for _ in points]
            for i, a in enumerate(points):
                for j in range(i):
                    b = points[j]
                    if self._clear((a['x'], a['y']), (b['x'], b['y']), a['yaw']):
                        distance = math.hypot(a['x']-b['x'], a['y']-b['y'])
                        edges[i].append((j, distance))
                        edges[j].append((i, distance))
            self.edges[side] = edges

    @classmethod
    def from_yaml(cls, model, path):
        with Path(path).open(encoding='utf-8') as stream:
            data = yaml.safe_load(stream) or {}
        entry = data.get('fixed_bucket_transit')
        points = [] if entry is None else _load_waypoints(
            entry.get('waypoints'), data.get('frame_id', 'map'),
            'fixed bucket transit',
        )
        return cls(model, points)

    def _clear(self, start, end, yaw):
        # Fixed gates recur while Nav2 starts and on goal retries. Avoid
        # blocking lifecycle replies by recomputing their polygon sweeps.
        start, end = sorted((tuple(start), tuple(end)))
        return self._segment_clear(start, end, yaw)

    @lru_cache(maxsize=256)
    def _segment_clear(self, start, end, yaw):
        points = dense_path([start, end], .01)
        # Reject blocked connectors using a sparse subset first. Every sample
        # of an accepted connector is still checked at the original 1 cm step.
        reserve = .005 + self.model.radius*.02
        endpoint_values = self.model.clearance_over_poses(
            points[[0, -1]], np.full(2, yaw), cap=.15)
        if not np.isfinite(endpoint_values).all():
            return False
        endpoint_margin = np.minimum(
            endpoint_values[0] - reserve - .015 + .2*np.linalg.norm(points-points[0], axis=1),
            endpoint_values[-1] - reserve - .015 + .2*np.linalg.norm(points-points[-1], axis=1),
        )
        required = np.maximum(.025, np.minimum(.10, endpoint_margin))
        if np.any(endpoint_values-reserve < required[[0, -1]]-1.e-9):
            return False
        probe = np.arange(1, len(points)-1, 20)
        remaining = np.setdiff1d(np.arange(1, len(points)-1), probe)
        for indices in (probe, remaining):
            if not len(indices):
                continue
            values = self.model.clearance_over_poses(
                points[indices], np.full(len(indices), yaw), cap=.15)
            if not np.isfinite(values).all() or np.any(values-reserve < required[indices]-1.e-9):
                return False
        return True

    def select(self, start, end, side=LEFT):
        """Return the shortest checked connector's intermediate waypoints.

        An arbitrary off-network pose may have no visible connection. Leave
        that case to Nav2's obstacle-aware planner, preserving all existing
        approach and departure gates, rather than inventing an unchecked leg.
        """
        side = normalize_side(side)
        gates = self.waypoints[side]
        if start is None or not gates:
            return []
        start, end = np.asarray(start[:2], dtype=float), np.asarray(end[:2], dtype=float)
        if not np.isfinite([start, end]).all():
            return []
        yaw = gates[0]['yaw']
        if self._clear(start, end, yaw):
            return []
        count = len(gates)
        edges = [list(row) for row in self.edges[side]] + [[], []]
        for endpoint_id, endpoint in ((count, start), (count+1, end)):
            for i, gate in enumerate(gates):
                xy = np.array([gate['x'], gate['y']])
                if self._clear(endpoint, xy, yaw):
                    distance = float(np.linalg.norm(endpoint-xy))
                    edges[endpoint_id].append((i, distance))
                    edges[i].append((endpoint_id, distance))
        queue = [(0., count, [])]
        visited = set()
        while queue:
            distance, index, route = heapq.heappop(queue)
            if index in visited:
                continue
            if index == count+1:
                return [dict(gates[i]) for i in route if i < count
                        and math.dist(start, (gates[i]['x'], gates[i]['y'])) > .01
                        and math.dist(end, (gates[i]['x'], gates[i]['y'])) > .01]
            visited.add(index)
            for neighbor, length in edges[index]:
                if neighbor not in visited:
                    heapq.heappush(queue, (distance+length, neighbor, route+[neighbor]))
        return []
