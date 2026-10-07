import math
from typing import Optional, Sequence, Tuple

import numpy as np

from .geometry import compose_pose


def build_wheel_matrix(
    wheel_positions: Sequence[Sequence[float]],
    wheel_drive_angles: Sequence[float],
) -> np.ndarray:
    """Build the linear map from base motion to wheel travel."""
    positions = np.asarray(wheel_positions, dtype=float)
    angles = np.asarray(wheel_drive_angles, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError('wheel_positions must be an Nx2 array')
    if positions.shape[0] < 3:
        raise ValueError('at least three measurement wheels are required')
    if angles.shape != (positions.shape[0],):
        raise ValueError(
            'wheel_drive_angles must contain one angle per measurement wheel'
        )
    if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(angles)):
        raise ValueError('measurement wheel positions and angles must be finite')

    directions = np.column_stack((np.cos(angles), np.sin(angles)))
    rotation_terms = (
        -positions[:, 1] * directions[:, 0]
        + positions[:, 0] * directions[:, 1]
    )
    matrix = np.column_stack((directions, rotation_terms))
    if np.linalg.matrix_rank(matrix) < 3:
        raise ValueError('measurement wheel geometry cannot observe x, y, and yaw')
    return matrix


def counter_delta(
    current_counts: Sequence[int],
    previous_counts: Sequence[int],
    *,
    counter_bits: int = 32,
) -> np.ndarray:
    """Return shortest signed deltas for wrapping hardware counters."""
    try:
        valid_bits = not isinstance(counter_bits, bool) and int(counter_bits) == counter_bits
    except (TypeError, ValueError, OverflowError):
        valid_bits = False
    if not valid_bits:
        raise ValueError('counter_bits must be an integer')
    if counter_bits > 64:
        raise ValueError('counter_bits cannot exceed 64')
    def integer_counts(values):
        result = []
        for value in values:
            try:
                integer = int(value)
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError('counter values must be finite integers') from error
            if integer != value:
                raise ValueError('counter values must be finite integers')
            result.append(integer)
        return result

    current = integer_counts(current_counts)
    previous = integer_counts(previous_counts)
    if len(current) != len(previous):
        raise ValueError('current_counts and previous_counts must have same length')

    if counter_bits <= 0:
        return np.asarray(
            [current_value - previous_value
             for current_value, previous_value in zip(current, previous)],
            dtype=float,
        )

    modulus = 1 << int(counter_bits)
    half_modulus = modulus // 2
    return np.asarray(
        [
            ((current_value - previous_value + half_modulus) % modulus)
            - half_modulus
            for current_value, previous_value in zip(current, previous)
        ],
        dtype=float,
    )


def signed_count_delta(
    current_counts: Sequence[int],
    previous_counts: Sequence[int],
    count_signs: Sequence[float],
    *,
    counter_bits: int = 32,
) -> np.ndarray:
    raw_delta = counter_delta(
        current_counts, previous_counts, counter_bits=counter_bits
    )
    signs = np.asarray(count_signs, dtype=float)
    if signs.shape != raw_delta.shape:
        raise ValueError('count_signs must contain one sign per channel')
    if not np.all(np.isfinite(signs)) or np.any(np.abs(signs) != 1.0):
        raise ValueError('count_signs values must be +1 or -1')
    return raw_delta * signs


def meters_per_count_from_radius(
    counts_per_revolution: Sequence[float],
    wheel_radius: float,
) -> np.ndarray:
    counts = np.asarray(counts_per_revolution, dtype=float)
    if counts.ndim != 1 or not counts.size or not np.all(np.isfinite(counts)) or np.any(counts <= 0.0):
        raise ValueError('counts_per_revolution must contain finite positive values')
    radius = float(wheel_radius)
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError('wheel_radius must be finite and positive')
    return (2.0 * math.pi * radius) / counts


def counts_to_wheel_displacements(
    delta_counts: Sequence[float],
    *,
    meters_per_count: Optional[Sequence[float]] = None,
    counts_per_revolution: Optional[Sequence[float]] = None,
    wheel_radius: Optional[float] = None,
) -> np.ndarray:
    deltas = np.asarray(delta_counts, dtype=float)
    if deltas.ndim != 1 or not np.all(np.isfinite(deltas)):
        raise ValueError('delta_counts must contain finite values')
    if meters_per_count is not None:
        scale = np.asarray(meters_per_count, dtype=float)
    else:
        if counts_per_revolution is None or wheel_radius is None:
            raise ValueError(
                'meters_per_count or counts_per_revolution + wheel_radius is required'
            )
        scale = meters_per_count_from_radius(counts_per_revolution, wheel_radius)
    if scale.shape != deltas.shape:
        raise ValueError('distance scale must contain one value per channel')
    if not np.all(np.isfinite(scale)) or np.any(scale <= 0.0):
        raise ValueError('distance scale must contain finite positive values')
    with np.errstate(over='ignore', invalid='ignore'):
        displacements = deltas * scale
    if not np.all(np.isfinite(displacements)):
        raise ValueError('wheel displacements overflowed')
    return displacements


def body_delta_from_wheel_displacements(
    wheel_displacements: Sequence[float],
    wheel_matrix: np.ndarray,
) -> np.ndarray:
    displacements = np.asarray(wheel_displacements, dtype=float)
    wheel_matrix = np.asarray(wheel_matrix, dtype=float)
    if displacements.ndim != 1 or wheel_matrix.ndim != 2 or wheel_matrix.shape[1] != 3:
        raise ValueError('wheel_matrix must be Nx3 and wheel_displacements a vector')
    if wheel_matrix.shape[0] != displacements.shape[0]:
        raise ValueError('wheel_displacements length must match wheel_matrix rows')
    if not np.all(np.isfinite(displacements)) or not np.all(np.isfinite(wheel_matrix)):
        raise ValueError('wheel displacements and geometry must be finite')
    if np.linalg.matrix_rank(wheel_matrix) < 3:
        raise ValueError('measurement wheel geometry cannot observe x, y, and yaw')
    delta, _, _, _ = np.linalg.lstsq(wheel_matrix, displacements, rcond=None)
    return delta


def validate_odometry_scale(odometry_scale: Sequence[float]) -> np.ndarray:
    scale = np.asarray(odometry_scale, dtype=float)
    if scale.shape != (3,):
        raise ValueError('odometry_scale must contain [x, y, yaw] values')
    if not np.all(np.isfinite(scale)):
        raise ValueError('odometry_scale values must be finite')
    if np.any(scale <= 0.0):
        raise ValueError('odometry_scale values must be positive')
    return scale


def validate_calibration_matrix(
    calibration_matrix: Sequence[Sequence[float]],
    channel_count: int,
) -> np.ndarray:
    matrix = np.asarray(calibration_matrix, dtype=float)
    if matrix.shape != (3, channel_count):
        raise ValueError(
            f'calibration_matrix must be 3x{channel_count} '
            '[dx, dy, dyaw] rows by wheel channel'
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError('calibration_matrix values must be finite')
    if np.linalg.matrix_rank(matrix) < 3:
        raise ValueError('calibration_matrix cannot observe x, y, and yaw')
    return matrix


def solve_calibration_matrix(
    count_deltas: np.ndarray,
    body_deltas: np.ndarray,
    *,
    ridge: float = 1.0e-9,
) -> Tuple[np.ndarray, dict]:
    """Fit C so that body_delta ~= C @ raw_count_delta by ridge least squares.

    count_deltas: (K, N) raw wrapped counter deltas per window.
    body_deltas:  (K, 3) reference body-frame [dx, dy, dyaw] per window
                  (e.g. from accepted LiDAR wall localization poses).

    The single matrix absorbs wheel radius, counts-per-revolution, encoder
    multiplier, count direction and mounting geometry at once, which is why it
    calibrates systems where hand-tuning wheel_radius/odometry_scale fails.
    Returns (C, diagnostics).
    """
    counts = np.asarray(count_deltas, dtype=float)
    bodies = np.asarray(body_deltas, dtype=float)
    if counts.ndim != 2 or bodies.ndim != 2 or bodies.shape[1] != 3:
        raise ValueError('count_deltas must be (K, N) and body_deltas (K, 3)')
    if counts.shape[0] != bodies.shape[0]:
        raise ValueError('count_deltas and body_deltas need matching rows')
    if counts.shape[1] < 3:
        raise ValueError('at least three wheel channels are required')
    if not np.all(np.isfinite(counts)) or not np.all(np.isfinite(bodies)):
        raise ValueError('calibration samples must be finite')
    if not math.isfinite(float(ridge)) or ridge < 0.0:
        raise ValueError('ridge must be finite and nonnegative')
    if counts.shape[0] < max(12, 3 * counts.shape[1]):
        raise ValueError(
            f'Not enough samples ({counts.shape[0]}) to solve a '
            f'3x{counts.shape[1]} calibration; collect more motion'
        )
    if np.linalg.matrix_rank(counts) < 3 or np.linalg.matrix_rank(bodies) < 3:
        raise ValueError('calibration needs independent x, y, and yaw motion')

    normal = counts.T @ counts
    normal += np.eye(counts.shape[1]) * (
        ridge * max(float(np.trace(normal)), 1.0)
    )
    solution = np.linalg.solve(normal, counts.T @ bodies)
    matrix = validate_calibration_matrix(solution.T, counts.shape[1])

    predicted = counts @ matrix.T
    residuals = predicted - bodies
    rms = np.sqrt(np.mean(residuals ** 2, axis=0))
    singular_values = np.linalg.svd(counts, compute_uv=False)
    diagnostics = {
        'samples': int(counts.shape[0]),
        'residual_rms_xy_m': [float(rms[0]), float(rms[1])],
        'residual_rms_yaw_rad': float(rms[2]),
        'excitation_singular_values': [
            float(value) for value in singular_values
        ],
        'excitation_condition': float(
            singular_values[0] / max(singular_values[-1], 1.0e-12)
        ),
    }
    return matrix, diagnostics


def body_twist_delta_to_pose_delta(body_delta: Sequence[float]) -> np.ndarray:
    delta = np.asarray(body_delta, dtype=float)
    if delta.shape != (3,):
        raise ValueError('body_delta must contain [dx, dy, dyaw]')
    if not np.all(np.isfinite(delta)):
        raise ValueError('body_delta values must be finite')
    theta = float(delta[2])
    if abs(theta) < 1.0e-9:
        a = 1.0 - theta * theta / 6.0
        b = 0.5 * theta - theta * theta * theta / 24.0
    else:
        a = math.sin(theta) / theta
        b = (1.0 - math.cos(theta)) / theta
    return np.array([
        a * delta[0] - b * delta[1],
        b * delta[0] + a * delta[1],
        theta,
    ])


class MeasurementWheelKinematics:
    def __init__(
        self,
        *,
        wheel_positions: Sequence[Sequence[float]],
        wheel_drive_angles: Sequence[float],
        count_signs: Sequence[float],
        counter_bits: int,
        meters_per_count: Optional[Sequence[float]] = None,
        counts_per_revolution: Optional[Sequence[float]] = None,
        wheel_radius: Optional[float] = None,
        odometry_scale: Sequence[float] = (1.0, 1.0, 1.0),
        calibration_matrix: Optional[Sequence[Sequence[float]]] = None,
    ) -> None:
        self.wheel_matrix = build_wheel_matrix(
            wheel_positions, wheel_drive_angles
        )
        # The wheel geometry is constant for the lifetime of the node.  Solving
        # the same tiny least-squares system with an SVD at every 100 Hz sample
        # wastes CPU and adds avoidable odometry jitter, so factor it once and
        # reduce the hot path to a matrix-vector product.
        self.wheel_solver = np.linalg.pinv(self.wheel_matrix)
        self.count_signs = np.asarray(count_signs, dtype=float)
        if self.count_signs.shape != (self.wheel_matrix.shape[0],):
            raise ValueError('count_signs must contain one sign per channel')
        # Validate calibration inputs before opening or powering any hardware.
        signed_count_delta([0] * len(self.count_signs), [0] * len(self.count_signs),
                           self.count_signs, counter_bits=counter_bits)
        self.counter_bits = int(counter_bits)
        self.meters_per_count = (
            None
            if meters_per_count is None
            else np.asarray(meters_per_count, dtype=float)
        )
        self.counts_per_revolution = (
            None
            if counts_per_revolution is None
            else np.asarray(counts_per_revolution, dtype=float)
        )
        self.wheel_radius = None if wheel_radius is None else float(wheel_radius)
        counts_to_wheel_displacements(
            np.zeros(len(self.count_signs)),
            meters_per_count=self.meters_per_count,
            counts_per_revolution=self.counts_per_revolution,
            wheel_radius=self.wheel_radius,
        )
        self.odometry_scale = validate_odometry_scale(odometry_scale)
        self.calibration_matrix = (
            None
            if calibration_matrix is None
            else validate_calibration_matrix(
                calibration_matrix, self.wheel_matrix.shape[0]
            )
        )

    def set_odometry_scale(self, odometry_scale: Sequence[float]) -> None:
        self.odometry_scale = validate_odometry_scale(odometry_scale)

    def step(
        self,
        current_counts: Sequence[int],
        previous_counts: Sequence[int],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        delta_counts = signed_count_delta(
            current_counts,
            previous_counts,
            self.count_signs,
            counter_bits=self.counter_bits,
        )
        wheel_displacements = counts_to_wheel_displacements(
            delta_counts,
            meters_per_count=self.meters_per_count,
            counts_per_revolution=self.counts_per_revolution,
            wheel_radius=self.wheel_radius,
        )
        if self.calibration_matrix is not None:
            # The calibrated map is defined on raw wrapped deltas and already
            # contains sign, scale and geometry, so it replaces the geometric
            # solution and odometry_scale entirely.
            raw_delta = counter_delta(
                current_counts,
                previous_counts,
                counter_bits=self.counter_bits,
            )
            body_delta = self.calibration_matrix @ raw_delta
            return delta_counts, wheel_displacements, body_delta

        body_delta = self.wheel_solver @ wheel_displacements
        body_delta = body_delta * self.odometry_scale
        return delta_counts, wheel_displacements, body_delta

    @staticmethod
    def integrate_pose(pose: np.ndarray, body_delta: Sequence[float]) -> np.ndarray:
        pose = np.asarray(pose, dtype=float)
        if pose.shape != (3,) or not np.all(np.isfinite(pose)):
            raise ValueError('pose must contain finite [x, y, yaw] values')
        return compose_pose(
            pose,
            body_twist_delta_to_pose_delta(body_delta),
        )
