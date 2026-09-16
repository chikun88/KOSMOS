import math

import numpy as np


def twist_to_wheel_speeds(
    linear_x: float,
    linear_y: float,
    angular_z: float,
    *,
    drive_model: str,
    wheel_radius: float,
    track_width: float,
    wheelbase: float,
    wheel_positions: np.ndarray = None,
    wheel_drive_angles: np.ndarray = None,
    wheel_signs: np.ndarray = None,
) -> np.ndarray:
    """Convert base velocity to wheel angular velocities in rad/s."""
    if wheel_radius <= 0.0:
        raise ValueError('wheel_radius must be positive')
    if drive_model == 'differential':
        half_track = 0.5 * track_width
        return np.array([
            (linear_x - half_track * angular_z) / wheel_radius,
            (linear_x + half_track * angular_z) / wheel_radius,
        ])
    if drive_model == 'mecanum':
        rotation_radius = 0.5 * (track_width + wheelbase)
        return np.array([
            (linear_x - linear_y - rotation_radius * angular_z) / wheel_radius,
            (linear_x + linear_y + rotation_radius * angular_z) / wheel_radius,
            (linear_x + linear_y - rotation_radius * angular_z) / wheel_radius,
            (linear_x - linear_y + rotation_radius * angular_z) / wheel_radius,
        ])
    if drive_model == 'omni4':
        if wheel_positions is None or wheel_drive_angles is None:
            raise ValueError(
                'omni4 requires wheel_positions and wheel_drive_angles'
            )
        positions = np.asarray(wheel_positions, dtype=float)
        angles = np.asarray(wheel_drive_angles, dtype=float)
        if positions.shape != (4, 2) or angles.shape != (4,):
            raise ValueError(
                'omni4 wheel_positions must be 4x2 and angles must contain 4 values'
            )
        signs = (
            np.ones(4)
            if wheel_signs is None
            else np.asarray(wheel_signs, dtype=float)
        )
        if signs.shape != (4,):
            raise ValueError('omni4 wheel_signs must contain 4 values')
        directions = np.column_stack((np.cos(angles), np.sin(angles)))
        rotation_terms = (
            -positions[:, 1] * directions[:, 0]
            + positions[:, 0] * directions[:, 1]
        )
        speeds = (
            directions[:, 0] * linear_x
            + directions[:, 1] * linear_y
            + rotation_terms * angular_z
        ) / wheel_radius
        return speeds * signs
    raise ValueError(f'Unsupported drive_model: {drive_model}')


def limit_wheel_speeds(speeds: np.ndarray, maximum: float) -> np.ndarray:
    speeds = np.asarray(speeds, dtype=float)
    if maximum <= 0.0:
        raise ValueError('maximum wheel speed must be positive')
    peak = float(np.max(np.abs(speeds))) if len(speeds) else 0.0
    if peak <= maximum:
        return speeds.copy()
    return speeds * (maximum / peak)


def allocate_omni4_wheel_budget(
    linear_x: float,
    linear_y: float,
    angular_z: float,
    *,
    wheel_radius: float,
    wheel_positions: np.ndarray,
    wheel_drive_angles: np.ndarray,
    wheel_signs: np.ndarray,
    maximum: float,
    translation_budget_share: float = 0.45,
) -> tuple[float, float, float]:
    """Fit an omni twist into the wheel-speed budget without changing its shape.

    The whole twist is scaled by one factor, so the executed motion is the
    commanded motion at a lower speed.  Direction of travel, the ratio of
    translation to yaw, and therefore the instantaneous path curvature are all
    preserved exactly.

    This matters because the controller is a closed loop.  A uniform slowdown
    is a pure time rescaling of the planned trajectory: MPPI replans from the
    measured state every cycle and simply arrives later.  Scaling translation
    and yaw by *different* factors instead changes the curvature of the motion
    the base actually performs, so the base leaves the planned path, the
    controller corrects, the allocator distorts the correction in turn, and the
    loop hunts.  Measured on this robot: a `balanced` request of vx 0.78 m/s
    with wz 1.30 rad/s needs 11.03 + 12.25 = 23.28 of the 15.709 rad/s wheel
    limit.  The previous split allocation executed (0.500, 0.917), a 45%
    curvature error against the plan; uniform scaling executes (0.527, 0.877)
    at exactly the planned curvature, and is also faster.

    It also makes this allocator agree with the deployed Pi mixer, which
    already scales all four wheel commands by one factor
    (``OMNI::mix_velocity`` in ``bacon_gateway/src/move.cpp``).  Two
    disagreeing allocators in one command path cannot both be right.

    The earlier "spins in place and never advances" report is not a reason to
    split the budget: that was yaw *absolute priority*, which left only
    0.245 m/s of forward speed.  Uniform scaling leaves 0.527 m/s for the same
    request, so it dominates both the original behaviour and the split
    allocation it replaced.  ``translation_budget_share`` is retained for
    configuration compatibility and is validated, but a uniform scale needs no
    reservation and the value is not used.
    """
    if maximum <= 0.0 or not math.isfinite(float(maximum)):
        raise ValueError('maximum wheel speed must be finite and positive')
    share = float(translation_budget_share)
    if not math.isfinite(share) or not 0.0 <= share < 1.0:
        raise ValueError('translation_budget_share must be in [0.0, 1.0)')
    speeds = twist_to_wheel_speeds(
        linear_x,
        linear_y,
        angular_z,
        drive_model='omni4',
        wheel_radius=wheel_radius,
        track_width=0.0,
        wheelbase=0.0,
        wheel_positions=wheel_positions,
        wheel_drive_angles=wheel_drive_angles,
        wheel_signs=wheel_signs,
    )
    peak = float(np.max(np.abs(speeds))) if len(speeds) else 0.0
    scale = min(1.0, float(maximum) / peak) if peak > 1.0e-12 else 1.0
    return (
        float(linear_x) * scale,
        float(linear_y) * scale,
        float(angular_z) * scale,
    )


def radians_per_second_to_rpm(speeds: np.ndarray) -> np.ndarray:
    return np.asarray(speeds, dtype=float) * 60.0 / (2.0 * math.pi)
