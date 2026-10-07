from dataclasses import dataclass
import math
from typing import Callable, Optional, Tuple

import numpy as np


class OptimizationCancelled(Exception):
    """A caller stopped an otherwise unchanged iterative pose solve."""


@dataclass
class OptimizationResult:
    pose: np.ndarray
    covariance: np.ndarray
    converged: bool
    correspondences: int
    rmse: float
    iterations: int
    final_translation_step: float
    final_rotation_step: float


def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def compose_pose(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Return T(first) * T(second) for planar [x, y, yaw] poses."""
    c = math.cos(float(first[2]))
    s = math.sin(float(first[2]))
    return np.array([
        first[0] + c * second[0] - s * second[1],
        first[1] + s * second[0] + c * second[1],
        normalize_angle(float(first[2] + second[2])),
    ])


def inverse_pose(pose: np.ndarray) -> np.ndarray:
    c = math.cos(float(pose[2]))
    s = math.sin(float(pose[2]))
    return np.array([
        -c * pose[0] - s * pose[1],
        s * pose[0] - c * pose[1],
        normalize_angle(float(-pose[2])),
    ])


def relative_pose(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return compose_pose(inverse_pose(first), second)


def interpolate_pose(
    first: np.ndarray,
    second: np.ndarray,
    fraction: float,
) -> np.ndarray:
    """Interpolate along the shortest planar motion from first to second."""
    motion = relative_pose(
        np.asarray(first, dtype=float),
        np.asarray(second, dtype=float),
    )
    ratio = float(fraction)
    return compose_pose(
        np.asarray(first, dtype=float),
        np.array([
            motion[0] * ratio,
            motion[1] * ratio,
            motion[2] * ratio,
        ]),
    )


def interpolate_poses_at(
    stamps: np.ndarray,
    poses: np.ndarray,
    query_stamps: np.ndarray,
) -> np.ndarray:
    """Vectorized linear pose interpolation over a sorted stamp history.

    Query stamps outside the history are clamped to the first/last pose. Yaw
    is interpolated along the shortest arc, which matches interpolate_pose to
    first order for the small inter-sample motions of a 100 Hz odometry feed.
    """
    stamps = np.asarray(stamps, dtype=np.int64)
    poses = np.asarray(poses, dtype=float)
    query = np.asarray(query_stamps, dtype=np.int64)
    if len(stamps) == 1:
        return np.repeat(poses, len(query), axis=0)

    upper = np.clip(np.searchsorted(stamps, query), 1, len(stamps) - 1)
    lower = upper - 1
    span = (stamps[upper] - stamps[lower]).astype(float)
    span[span <= 0.0] = 1.0
    fraction = np.clip((query - stamps[lower]).astype(float) / span, 0.0, 1.0)

    before = poses[lower]
    after = poses[upper]
    delta = after - before
    delta[:, 2] = np.arctan2(np.sin(delta[:, 2]), np.cos(delta[:, 2]))
    result = before + fraction[:, None] * delta
    result[:, 2] = np.arctan2(np.sin(result[:, 2]), np.cos(result[:, 2]))
    return result


def transform_points(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    c = math.cos(float(pose[2]))
    s = math.sin(float(pose[2]))
    rotation = np.array([[c, -s], [s, c]])
    return points @ rotation.T + pose[:2]


def transform_points_between_poses(
    points: np.ndarray,
    source_pose: np.ndarray,
    target_pose: np.ndarray,
) -> np.ndarray:
    """Express source-frame points in the target pose frame."""
    source_to_target = relative_pose(
        np.asarray(target_pose, dtype=float),
        np.asarray(source_pose, dtype=float),
    )
    return transform_points(np.asarray(points, dtype=float), source_to_target)


def transform_points_from_poses(
    points: np.ndarray,
    source_poses: np.ndarray,
    target_pose: np.ndarray,
) -> np.ndarray:
    """Express each point, taken in its own source pose frame, in the target frame.

    Vectorized equivalent of calling transform_points_between_poses per point
    with a per-point source pose (used for scan motion compensation).
    """
    points = np.asarray(points, dtype=float)
    source_poses = np.asarray(source_poses, dtype=float)
    target = np.asarray(target_pose, dtype=float)

    ct = math.cos(float(target[2]))
    st = math.sin(float(target[2]))
    offsets = source_poses[:, :2] - target[:2]
    translation = np.column_stack((
        ct * offsets[:, 0] + st * offsets[:, 1],
        -st * offsets[:, 0] + ct * offsets[:, 1],
    ))
    relative_yaw = source_poses[:, 2] - float(target[2])
    c = np.cos(relative_yaw)
    s = np.sin(relative_yaw)
    return np.column_stack((
        c * points[:, 0] - s * points[:, 1] + translation[:, 0],
        s * points[:, 0] + c * points[:, 1] + translation[:, 1],
    ))


def points_in_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Vectorized ray-casting test. Points on the boundary are treated as inside."""
    if len(points) == 0:
        return np.zeros(0, dtype=bool)
    x = points[:, 0]
    y = points[:, 1]
    inside = np.zeros(len(points), dtype=bool)
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        intersects = ((yi > y) != (yj > y)) & (
            x <= (xj - xi) * (y - yi) / (yj - yi + 1.0e-15) + xi
        )
        inside ^= intersects
        j = i
    return inside


def closest_wall_points(
    points: np.ndarray,
    segments: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find the closest point and segment index for every 2-D point."""
    closest, distances, indices, _ = closest_wall_correspondences(
        points, segments
    )
    return closest, distances, indices


def closest_wall_correspondences(
    points: np.ndarray,
    segments: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Brute-force closest point, distance, segment index and edge fraction."""
    starts = segments[:, :2]
    vectors = segments[:, 2:] - starts
    lengths_squared = np.maximum(np.sum(vectors * vectors, axis=1), 1.0e-12)

    offsets = points[:, None, :] - starts[None, :, :]
    fractions = np.sum(offsets * vectors[None, :, :], axis=2) / lengths_squared[None, :]
    fractions = np.clip(fractions, 0.0, 1.0)
    candidates = starts[None, :, :] + fractions[:, :, None] * vectors[None, :, :]
    errors = points[:, None, :] - candidates
    distances_squared = np.sum(errors * errors, axis=2)
    indices = np.argmin(distances_squared, axis=1)
    rows = np.arange(len(points))
    return (
        candidates[rows, indices],
        np.sqrt(distances_squared[rows, indices]),
        indices,
        fractions[rows, indices],
    )


class WallLookupGrid:
    """Precomputed nearest-wall-segment index over a regular grid.

    Replaces the O(points x segments) brute-force correspondence search in the
    ICP loop with an O(points) table lookup followed by an exact projection on
    the selected segment. Near Voronoi boundaries the cell-center vote can pick
    the second-closest segment; the resulting distance error is bounded by the
    cell diagonal and is negligible for wall alignment.
    """

    def __init__(
        self,
        segments: np.ndarray,
        resolution: float = 0.04,
        margin: float = 1.0,
        chunk_size: int = 2048,
    ) -> None:
        segments = np.asarray(segments, dtype=float)
        if segments.ndim != 2 or segments.shape[1] != 4 or len(segments) == 0:
            raise ValueError('segments must have shape (N, 4)')
        self.segments = segments
        self.starts = segments[:, :2].copy()
        self.vectors = segments[:, 2:] - self.starts
        self.lengths_squared = np.maximum(
            np.sum(self.vectors * self.vectors, axis=1), 1.0e-12
        )
        directions = self.vectors / np.sqrt(self.lengths_squared)[:, None]
        self.normals = np.column_stack((-directions[:, 1], directions[:, 0]))

        low = np.minimum(segments[:, :2], segments[:, 2:]).min(axis=0) - margin
        high = np.maximum(segments[:, :2], segments[:, 2:]).max(axis=0) + margin
        self.resolution = float(resolution)
        self.origin = low
        self.cells_x = max(1, int(math.ceil((high[0] - low[0]) / self.resolution)))
        self.cells_y = max(1, int(math.ceil((high[1] - low[1]) / self.resolution)))

        xs = low[0] + (np.arange(self.cells_x) + 0.5) * self.resolution
        ys = low[1] + (np.arange(self.cells_y) + 0.5) * self.resolution
        grid_x, grid_y = np.meshgrid(xs, ys)
        centers = np.column_stack((grid_x.reshape(-1), grid_y.reshape(-1)))

        # Peak memory is O(chunk_size * wall_count).  The previous 16k block
        # transiently used hundreds of MB per lookup on the competition map,
        # and several nodes build one concurrently at launch.  Smaller blocks
        # also fit cache better while producing the exact same table.
        nearest = np.empty(len(centers), dtype=np.int32)
        for begin in range(0, len(centers), chunk_size):
            block = centers[begin:begin + chunk_size]
            offsets = block[:, None, :] - self.starts[None, :, :]
            fractions = (
                np.sum(offsets * self.vectors[None, :, :], axis=2)
                / self.lengths_squared[None, :]
            )
            np.clip(fractions, 0.0, 1.0, out=fractions)
            candidates = (
                self.starts[None, :, :]
                + fractions[:, :, None] * self.vectors[None, :, :]
            )
            errors = block[:, None, :] - candidates
            nearest[begin:begin + chunk_size] = np.argmin(
                np.sum(errors * errors, axis=2), axis=1
            ).astype(np.int32)
        self.nearest = nearest.reshape(self.cells_y, self.cells_x)

    def query(
        self,
        points: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return closest point, distance, segment index and edge fraction."""
        points = np.asarray(points, dtype=float)
        if len(points) == 0:
            empty = np.zeros(0)
            return points.copy(), empty, empty.astype(np.int32), empty
        cell_x = np.clip(
            ((points[:, 0] - self.origin[0]) / self.resolution).astype(np.int64),
            0,
            self.cells_x - 1,
        )
        cell_y = np.clip(
            ((points[:, 1] - self.origin[1]) / self.resolution).astype(np.int64),
            0,
            self.cells_y - 1,
        )
        indices = self.nearest[cell_y, cell_x]
        starts = self.starts[indices]
        vectors = self.vectors[indices]
        fractions = np.clip(
            np.sum((points - starts) * vectors, axis=1)
            / self.lengths_squared[indices],
            0.0,
            1.0,
        )
        closest = starts + fractions[:, None] * vectors
        distances = np.linalg.norm(points - closest, axis=1)
        return closest, distances, indices, fractions


def raycast_segments(
    origin: np.ndarray,
    directions: np.ndarray,
    segments: np.ndarray,
) -> np.ndarray:
    """Return nearest positive ray/segment intersection distances."""
    starts = segments[:, :2]
    wall_vectors = segments[:, 2:] - starts
    offset = starts[None, :, :] - origin[None, None, :]
    direction_grid = directions[:, None, :]
    wall_grid = wall_vectors[None, :, :]

    cross_direction_wall = (
        direction_grid[:, :, 0] * wall_grid[:, :, 1]
        - direction_grid[:, :, 1] * wall_grid[:, :, 0]
    )
    safe = np.abs(cross_direction_wall) > 1.0e-10
    numerator_ray = (
        offset[:, :, 0] * wall_grid[:, :, 1]
        - offset[:, :, 1] * wall_grid[:, :, 0]
    )
    numerator_wall = (
        offset[:, :, 0] * direction_grid[:, :, 1]
        - offset[:, :, 1] * direction_grid[:, :, 0]
    )
    ray_distance = np.divide(
        numerator_ray,
        cross_direction_wall,
        out=np.full(cross_direction_wall.shape, np.inf),
        where=safe,
    )
    wall_fraction = np.divide(
        numerator_wall,
        cross_direction_wall,
        out=np.full(cross_direction_wall.shape, np.inf),
        where=safe,
    )
    valid = (
        safe
        & (ray_distance >= 0.0)
        & (wall_fraction >= 0.0)
        & (wall_fraction <= 1.0)
    )
    ray_distance[~valid] = np.inf
    return np.min(ray_distance, axis=1)


def optimize_pose(
    base_points: np.ndarray,
    wall_segments: np.ndarray,
    initial_pose: np.ndarray,
    *,
    max_iterations: int = 12,
    max_correspondence_distance: float = 0.35,
    huber_delta: float = 0.08,
    min_correspondences: int = 40,
    max_translation_step: float = 0.20,
    max_rotation_step: float = 0.20,
    convergence_translation: float = 1.0e-4,
    convergence_rotation: float = 1.0e-4,
    lookup: Optional[WallLookupGrid] = None,
    trim_ratio: float = 0.0,
    point_to_line: bool = True,
    cancelled: Optional[Callable[[], bool]] = None,
) -> OptimizationResult:
    """Locally align base-frame scan points to known wall segments.

    Interior correspondences use a point-to-line residual (distance along the
    wall normal) so points may slide along their wall; correspondences that
    project onto a segment endpoint keep the full 2-D point-to-point residual,
    which anchors the estimate at corners. trim_ratio additionally drops the
    worst-matching fraction of correspondences each iteration, rejecting
    off-map returns (other robots, hands, game objects) that survive the
    distance gate.
    """
    pose = np.asarray(initial_pose, dtype=float).copy()
    points = np.asarray(base_points, dtype=float)
    segments = np.asarray(wall_segments, dtype=float)

    empty_covariance = np.diag([1.0, 1.0, math.radians(30.0) ** 2])
    if len(points) < min_correspondences:
        return OptimizationResult(
            pose, empty_covariance, False, 0, math.inf, 0, math.inf, math.inf
        )

    if point_to_line:
        if lookup is not None:
            segment_normals = lookup.normals
        else:
            seg_vectors = segments[:, 2:] - segments[:, :2]
            seg_lengths = np.maximum(
                np.linalg.norm(seg_vectors, axis=1), 1.0e-12
            )
            seg_directions = seg_vectors / seg_lengths[:, None]
            segment_normals = np.column_stack(
                (-seg_directions[:, 1], seg_directions[:, 0])
            )
    trim_ratio = float(np.clip(trim_ratio, 0.0, 0.5))

    converged = False
    last_hessian: Optional[np.ndarray] = None
    last_squared_error = math.inf
    last_count = 0
    iterations_done = 0
    final_translation_step = math.inf
    final_rotation_step = math.inf

    for iteration in range(max_iterations):
        if cancelled is not None and cancelled():
            raise OptimizationCancelled()
        iterations_done = iteration + 1
        map_points = transform_points(points, pose)
        if lookup is not None:
            wall_points, distances, indices, fractions = lookup.query(map_points)
        else:
            wall_points, distances, indices, fractions = (
                closest_wall_correspondences(map_points, segments)
            )
        mask = distances <= max_correspondence_distance
        if trim_ratio > 0.0:
            matched = int(np.count_nonzero(mask))
            keep = max(min_correspondences, int(matched * (1.0 - trim_ratio)))
            if matched > keep:
                threshold = np.partition(distances[mask], keep - 1)[keep - 1]
                mask &= distances <= threshold
        count = int(np.count_nonzero(mask))
        if count < min_correspondences:
            return OptimizationResult(
                pose, empty_covariance, False, count, math.inf, iterations_done,
                final_translation_step, final_rotation_step
            )

        selected_base = points[mask]
        selected_map = map_points[mask]
        selected_wall = wall_points[mask]
        residuals = selected_map - selected_wall

        c = math.cos(float(pose[2]))
        s = math.sin(float(pose[2]))
        d_x_d_yaw = -s * selected_base[:, 0] - c * selected_base[:, 1]
        d_y_d_yaw = c * selected_base[:, 0] - s * selected_base[:, 1]

        if point_to_line:
            selected_fractions = fractions[mask]
            interior = (selected_fractions > 1.0e-9) & (
                selected_fractions < 1.0 - 1.0e-9
            )
            normals = segment_normals[indices[mask]]
            # Scalar residual along the wall normal for interior matches; the
            # projected distance equals the point-segment distance there.
            errors = np.where(
                interior,
                np.abs(np.sum(residuals * normals, axis=1)),
                np.linalg.norm(residuals, axis=1),
            )
        else:
            interior = np.zeros(count, dtype=bool)
            errors = np.linalg.norm(residuals, axis=1)

        weights = np.ones(count)
        robust_mask = errors > huber_delta
        weights[robust_mask] = huber_delta / np.maximum(
            errors[robust_mask], 1.0e-12
        )

        interior_count = int(np.count_nonzero(interior))
        endpoint = ~interior
        endpoint_count = count - interior_count
        rows = interior_count + 2 * endpoint_count
        jacobian = np.zeros((rows, 3))
        residual_vector = np.zeros(rows)
        row_weights = np.zeros(rows)

        if interior_count:
            normal_rows = normals[interior]
            jacobian[:interior_count, 0] = normal_rows[:, 0]
            jacobian[:interior_count, 1] = normal_rows[:, 1]
            jacobian[:interior_count, 2] = (
                normal_rows[:, 0] * d_x_d_yaw[interior]
                + normal_rows[:, 1] * d_y_d_yaw[interior]
            )
            residual_vector[:interior_count] = np.sum(
                residuals[interior] * normal_rows, axis=1
            )
            row_weights[:interior_count] = weights[interior]
        if endpoint_count:
            offset = interior_count
            jacobian[offset::2, 0][:endpoint_count] = 1.0
            jacobian[offset + 1::2, 1][:endpoint_count] = 1.0
            jacobian[offset::2, 2][:endpoint_count] = d_x_d_yaw[endpoint]
            jacobian[offset + 1::2, 2][:endpoint_count] = d_y_d_yaw[endpoint]
            residual_vector[offset::2][:endpoint_count] = residuals[endpoint, 0]
            residual_vector[offset + 1::2][:endpoint_count] = (
                residuals[endpoint, 1]
            )
            row_weights[offset::2][:endpoint_count] = weights[endpoint]
            row_weights[offset + 1::2][:endpoint_count] = weights[endpoint]

        sqrt_weights = np.sqrt(row_weights)
        weighted_jacobian = jacobian * sqrt_weights[:, None]
        weighted_residual = residual_vector * sqrt_weights
        hessian = weighted_jacobian.T @ weighted_jacobian
        gradient = weighted_jacobian.T @ weighted_residual
        # Tiny Tikhonov damping keeps the solve stable when nearly all
        # correspondences share one wall direction; it is negligible otherwise.
        hessian += np.eye(3) * (1.0e-9 * max(float(np.trace(hessian)), 1.0))

        if np.linalg.cond(hessian) > 1.0e10:
            return OptimizationResult(
                pose, empty_covariance, False, count,
                float(np.sqrt(np.mean(errors ** 2))), iterations_done,
                final_translation_step, final_rotation_step
            )

        try:
            delta = -np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            return OptimizationResult(
                pose, empty_covariance, False, count,
                float(np.sqrt(np.mean(errors ** 2))), iterations_done,
                final_translation_step, final_rotation_step
            )

        translation_norm = float(np.linalg.norm(delta[:2]))
        if translation_norm > max_translation_step:
            delta[:2] *= max_translation_step / translation_norm
        delta[2] = float(np.clip(delta[2], -max_rotation_step, max_rotation_step))
        final_translation_step = float(np.linalg.norm(delta[:2]))
        final_rotation_step = abs(float(delta[2]))

        pose += delta
        pose[2] = normalize_angle(float(pose[2]))
        last_hessian = hessian
        last_squared_error = float(np.sum(weights * errors ** 2))
        last_count = count

        if (
            np.linalg.norm(delta[:2]) < convergence_translation
            and abs(delta[2]) < convergence_rotation
        ):
            converged = True
            break

    if last_hessian is None or last_count == 0:
        covariance = empty_covariance
        rmse = math.inf
    else:
        degrees_of_freedom = max(1, 2 * last_count - 3)
        variance = max(last_squared_error / degrees_of_freedom, 1.0e-8)
        try:
            covariance = np.linalg.inv(last_hessian) * variance
        except np.linalg.LinAlgError:
            covariance = empty_covariance
        rmse = math.sqrt(last_squared_error / last_count)

    return OptimizationResult(
        pose=pose,
        covariance=covariance,
        converged=converged,
        correspondences=last_count,
        rmse=rmse,
        iterations=iterations_done,
        final_translation_step=final_translation_step,
        final_rotation_step=final_rotation_step,
    )
