from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
import heapq
import math
from pathlib import Path

import numpy as np
import yaml

try:
    from .body_model import BodyClearanceModel, load_footprint, load_wall_segments
except ImportError:
    from body_model import BodyClearanceModel, load_footprint, load_wall_segments


# Nav2 forbids a footprint-colliding pose and merely adds inflation cost
# elsewhere, so planning feasibility is a small positive body margin rather
# than the old disc allowance.  The tight firing poses have only 47-59 mm of
# body clearance, so this margin has to stay below that: 35 mm is the largest
# value for which every configured goal is still reachable.
PLANNING_MARGIN = 0.035


def _point_segment_distance(point, start, end):
    point = np.asarray(point, dtype=float)
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    delta = end - start
    length2 = float(delta @ delta)
    if length2 <= 1.0e-15:
        return float(np.linalg.norm(point - start))
    t = float(np.clip(((point - start) @ delta) / length2, 0.0, 1.0))
    return float(np.linalg.norm(point - (start + t * delta)))


@dataclass
class GridField:
    """CAD field with a yaw-dependent robot-body clearance model.

    ``clearance`` remains the base_link distance to the nearest wall.  Every
    feasibility decision uses ``body_clearance``, the distance from the rotated
    ten-vertex footprint to the walls, because the deployed robot is neither a
    disc nor yaw-invariant.
    """

    segments: np.ndarray
    footprint: np.ndarray
    resolution: float = 0.10
    planning_margin: float = PLANNING_MARGIN
    clearance_cap: float = 0.35

    def __post_init__(self):
        points = self.segments.reshape(-1, 2)
        self.minimum = points.min(axis=0)
        self.maximum = points.max(axis=0)
        self.body = BodyClearanceModel(
            self.segments, self.footprint, cap=self.clearance_cap
        )
        self.width = int(math.floor((self.maximum[0] - self.minimum[0]) / self.resolution)) + 1
        self.height = int(math.floor((self.maximum[1] - self.minimum[1]) / self.resolution)) + 1
        self._distance_resolution = 0.05
        self._distance_x = np.arange(
            self.minimum[0], self.maximum[0] + 0.5 * self._distance_resolution,
            self._distance_resolution,
        )
        self._distance_y = np.arange(
            self.minimum[1], self.maximum[1] + 0.5 * self._distance_resolution,
            self._distance_resolution,
        )
        self._distance_grid = self._build_distance_grid()
        self._body_grids = {}
        self._boundary_offsets = self._build_boundary_offsets()

    def _build_distance_grid(self):
        starts = self.segments[:, 0, :]
        deltas = self.segments[:, 1, :] - starts
        length2 = np.einsum('ij,ij->i', deltas, deltas)
        length2[length2 < 1.0e-15] = 1.0e-15
        grid = np.empty(
            (len(self._distance_y), len(self._distance_x)), dtype=np.float32
        )
        # Exact base_link distance at every node; runtime lookup is bilinear.
        for row, y_value in enumerate(self._distance_y):
            query = np.column_stack((
                self._distance_x,
                np.full(len(self._distance_x), y_value, dtype=float),
            ))
            relative = query[:, None, :] - starts[None, :, :]
            projection = np.clip(
                np.einsum('qsi,si->qs', relative, deltas) / length2[None, :],
                0.0, 1.0,
            )
            nearest = starts[None, :, :] + projection[:, :, None] * deltas[None, :, :]
            grid[row, :] = np.linalg.norm(
                nearest - query[:, None, :], axis=2
            ).min(axis=1)
        return grid

    def _build_boundary_offsets(self):
        """Body-frame samples of the footprint outline, spaced under half a cell."""
        polygon = self.body.footprint
        spacing = 0.5 * self._distance_resolution
        samples = []
        for index, start in enumerate(polygon):
            end = polygon[(index + 1) % len(polygon)]
            count = max(2, int(math.ceil(float(np.linalg.norm(end - start)) / spacing)) + 1)
            for t in np.linspace(0.0, 1.0, count, endpoint=False):
                samples.append(start + t * (end - start))
        return np.asarray(samples, dtype=float)

    @classmethod
    def from_yaml(cls, path, footprint_file=None, profile='NORMAL', **kwargs):
        segments = load_wall_segments(path)
        if footprint_file is None:
            footprint_file = str(Path(path).with_name('competition_footprints.yaml'))
        return cls(
            segments=segments,
            footprint=load_footprint(footprint_file, profile),
            **kwargs,
        )

    # ------------------------------------------------------------------
    # base_link clearance
    # ------------------------------------------------------------------

    def clearance(self, point):
        """Bilinear base_link distance to the nearest wall."""
        point_array = np.asarray(point, dtype=float)
        gx = (point_array[0] - self.minimum[0]) / self._distance_resolution
        gy = (point_array[1] - self.minimum[1]) / self._distance_resolution
        if gx < 0.0 or gy < 0.0 or gx > len(self._distance_x) - 1 or gy > len(self._distance_y) - 1:
            return 0.0
        x0 = min(int(math.floor(gx)), len(self._distance_x) - 2)
        y0 = min(int(math.floor(gy)), len(self._distance_y) - 2)
        tx = gx - x0
        ty = gy - y0
        grid = self._distance_grid
        value = (
            (1.0 - tx) * (1.0 - ty) * grid[y0, x0]
            + tx * (1.0 - ty) * grid[y0, x0 + 1]
            + (1.0 - tx) * ty * grid[y0 + 1, x0]
            + tx * ty * grid[y0 + 1, x0 + 1]
        )
        # Half-centimetre conservative allowance for grid interpolation.
        return max(0.0, float(value) - 0.005)

    def clearance_and_gradient(self, point):
        return self.body.centre_clearance_and_gradient(point)

    # ------------------------------------------------------------------
    # robot-body clearance
    # ------------------------------------------------------------------

    def body_clearance(self, point, yaw):
        """Exact footprint clearance, saturated at ``clearance_cap``."""
        return self.body.clearance(point, yaw)

    def body_clearance_batch(self, points, yaws, cap=None):
        """Exact footprint clearance for a pose sequence, in one pass."""
        return self.body.clearance_batch(points, yaws, cap=cap)

    def _body_grid(self, yaw):
        """Eroded clearance field for one body yaw, for planning only.

        Erosion is a minimum of the exact node-distance grid sampled at the
        footprint outline.  Its bilinear interpolation makes it slightly
        pessimistic between nodes, which is the safe direction for a planner.
        """
        key = int(round(math.degrees(float(yaw)))) % 360
        cached = self._body_grids.get(key)
        if cached is not None:
            return cached
        angle = math.radians(key)
        c, s = math.cos(angle), math.sin(angle)
        offsets = self._boundary_offsets @ np.asarray([[c, s], [-s, c]])
        resolution = self._distance_resolution
        height, width = self._distance_grid.shape
        pad = int(math.ceil(self.body.radius / resolution)) + 2
        padded = np.pad(
            self._distance_grid, pad, mode='constant', constant_values=0.0
        )
        eroded = np.full((height, width), np.inf, dtype=np.float32)
        for offset_x, offset_y in offsets:
            fx = offset_x / resolution
            fy = offset_y / resolution
            ix = int(math.floor(fx))
            iy = int(math.floor(fy))
            tx = fx - ix
            ty = fy - iy
            row = pad + iy
            column = pad + ix
            top = padded[row:row + height, column:column + width]
            top_right = padded[row:row + height, column + 1:column + 1 + width]
            bottom = padded[row + 1:row + 1 + height, column:column + width]
            bottom_right = padded[row + 1:row + 1 + height, column + 1:column + 1 + width]
            sample = (
                (1.0 - ty) * ((1.0 - tx) * top + tx * top_right)
                + ty * ((1.0 - tx) * bottom + tx * bottom_right)
            )
            np.minimum(eroded, sample, out=eroded)
        self._body_grids[key] = eroded
        return eroded

    def planning_clearance(self, point, yaw):
        """Fast approximate body clearance from the eroded planning grid."""
        grid = self._body_grid(yaw)
        gx = (float(point[0]) - self.minimum[0]) / self._distance_resolution
        gy = (float(point[1]) - self.minimum[1]) / self._distance_resolution
        if gx < 0.0 or gy < 0.0 or gx > grid.shape[1] - 1 or gy > grid.shape[0] - 1:
            return 0.0
        x0 = min(int(math.floor(gx)), grid.shape[1] - 2)
        y0 = min(int(math.floor(gy)), grid.shape[0] - 2)
        tx = gx - x0
        ty = gy - y0
        value = (
            (1.0 - tx) * (1.0 - ty) * grid[y0, x0]
            + tx * (1.0 - ty) * grid[y0, x0 + 1]
            + (1.0 - tx) * ty * grid[y0 + 1, x0]
            + tx * ty * grid[y0 + 1, x0 + 1]
        )
        return max(0.0, float(value))

    # ------------------------------------------------------------------
    # grid helpers and planning
    # ------------------------------------------------------------------

    def world_to_cell(self, point):
        return (
            int(round((float(point[0]) - self.minimum[0]) / self.resolution)),
            int(round((float(point[1]) - self.minimum[1]) / self.resolution)),
        )

    def cell_to_world(self, cell):
        return np.asarray([
            self.minimum[0] + cell[0] * self.resolution,
            self.minimum[1] + cell[1] * self.resolution,
        ])

    def in_bounds(self, cell):
        return 0 <= cell[0] < self.width and 0 <= cell[1] < self.height

    def is_free(self, cell, yaw, margin=None):
        if not self.in_bounds(cell):
            return False
        margin = self.planning_margin if margin is None else float(margin)
        return self.planning_clearance(self.cell_to_world(cell), yaw) >= margin

    def nearest_free(self, point, yaw, max_radius=1.0):
        origin = self.world_to_cell(point)
        cells = int(math.ceil(max_radius / self.resolution))
        candidates = []
        for dx in range(-cells, cells + 1):
            for dy in range(-cells, cells + 1):
                cell = (origin[0] + dx, origin[1] + dy)
                if self.is_free(cell, yaw):
                    distance = float(np.linalg.norm(self.cell_to_world(cell) - point))
                    if distance <= max_radius:
                        candidates.append((distance, cell))
        return min(candidates)[1] if candidates else None

    def segment_safe(self, start, end, yaw, margin=None):
        start = np.asarray(start, dtype=float)
        end = np.asarray(end, dtype=float)
        distance = float(np.linalg.norm(end - start))
        # A quarter cell.  Half-cell sampling let a string-pulled segment skip
        # a thin violation next to the bucket.
        samples = max(2, int(math.ceil(distance / (0.25 * self.resolution))) + 1)
        margin = self.planning_margin if margin is None else float(margin)
        for t in np.linspace(0.0, 1.0, samples):
            point = start + t * (end - start)
            if self.planning_clearance(point, yaw) < margin:
                return False
        return True

    def plan(self, start, goal, yaw):
        """A* route for the footprint held at ``yaw``, then string-pulled."""
        start_point = np.asarray(start, dtype=float)
        goal_point = np.asarray(goal, dtype=float)
        start_cell = self.nearest_free(start_point, yaw)
        goal_cell = self.nearest_free(goal_point, yaw)
        if start_cell is None:
            # The robot is already standing here.  A tight pose need not fit at
            # the travel yaw it is about to rotate into; the yaw-rate gate in
            # the dynamics is what holds that rotation back until it does.
            start_cell = self.world_to_cell(start_point)
        if goal_cell is None:
            return None
        moves = [
            (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
            (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)),
            (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2)),
        ]
        queue = [(0.0, start_cell)]
        cost = {start_cell: 0.0}
        parent = {}
        while queue:
            _, current = heapq.heappop(queue)
            if current == goal_cell:
                break
            current_cost = cost[current]
            for dx, dy, step in moves:
                neighbor = (current[0] + dx, current[1] + dy)
                if not self.is_free(neighbor, yaw):
                    continue
                clearance = self.planning_clearance(self.cell_to_world(neighbor), yaw)
                # Mirrors the global inflation layer: proximity is expensive
                # but only a footprint collision is forbidden.
                soft = 0.10 / max(0.02, clearance)
                candidate = current_cost + step * self.resolution * (1.0 + soft)
                if candidate >= cost.get(neighbor, math.inf):
                    continue
                cost[neighbor] = candidate
                parent[neighbor] = current
                heuristic = float(np.linalg.norm(
                    self.cell_to_world(neighbor) - self.cell_to_world(goal_cell)
                ))
                heapq.heappush(queue, (candidate + heuristic, neighbor))
        if goal_cell not in cost:
            return None
        cells = [goal_cell]
        while cells[-1] != start_cell:
            cells.append(parent[cells[-1]])
        raw = [self.cell_to_world(cell) for cell in reversed(cells)]
        raw[0] = start_point
        raw[-1] = goal_point
        return self._string_pull(raw, yaw)

    def _string_pull(self, points, yaw):
        if len(points) <= 2:
            return np.asarray(points)
        output = [np.asarray(points[0])]
        index = 0
        while index < len(points) - 1:
            furthest = index + 1
            for candidate in range(index + 2, len(points)):
                if self.segment_safe(points[index], points[candidate], yaw):
                    furthest = candidate
                else:
                    break
            output.append(np.asarray(points[furthest]))
            index = furthest
        return np.asarray(output)

    def random_free_point(self, rng, yaw, side_sign=0):
        margin = self.planning_margin
        for _ in range(10000):
            point = rng.uniform(
                self.minimum + self.body.radius,
                self.maximum - self.body.radius,
            )
            if side_sign and side_sign * point[0] < self.body.radius:
                continue
            if self.planning_clearance(point, yaw) >= margin:
                return point
        raise RuntimeError('unable to sample a free field point')
