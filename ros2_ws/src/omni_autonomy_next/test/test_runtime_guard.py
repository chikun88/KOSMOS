import math

import numpy as np
import pytest

from omni_autonomy_next.motor_kinematics import twist_to_wheel_speeds
from omni_autonomy_next.runtime_guard import GuardHealth, MotionLimits, RuntimeGuard


POSITIONS = np.asarray([
    [0.333072, 0.333072], [0.333072, -0.333072],
    [-0.333072, 0.333072], [-0.333072, -0.333072],
])
ANGLES = np.radians([-45.0, 45.0, 45.0, -45.0])
SIGNS = np.ones(4)


def make_guard():
    hard = MotionLimits(1.0, 0.85, 1.8, 1.6, 3.0, 4.0, 8.0)
    return RuntimeGuard(
        profiles={
            'balanced': MotionLimits(0.72, 0.65, 1.3, 1.05, 2.0, 2.8, 5.5),
            'precision': MotionLimits(0.35, 0.32, 0.75, 0.55, 1.1, 1.2, 3.0),
        },
        hard_limits=hard, default_profile='balanced', command_timeout_sec=0.25,
        red_zone_speed_scale=0.55, wheel_radius=0.05,
        wheel_positions=POSITIONS, wheel_drive_angles_rad=ANGLES,
        wheel_signs=SIGNS, max_wheel_speed=15.709120382,
    )


ACTIVE = GuardHealth(True, False, True, True, True)


def test_all_faults_fail_closed_immediately():
    guard = make_guard()
    now = 0.0
    for _ in range(40):
        guard.step((0.5, 0.2, 0.4), now_sec=now, command_age_sec=0.0,
                   health=ACTIVE, profile='balanced', user_scale=1.0, red_zone=False)
        now += 0.01
    for health, reason in [
        (GuardHealth(False, False, True, True, True), 'DISARMED'),
        (GuardHealth(True, True, True, True, True), 'EMERGENCY_STOP'),
        (GuardHealth(True, False, False, True, True), 'LOCALIZATION_UNHEALTHY'),
        (GuardHealth(True, False, True, True, True, False), 'RL_POLICY_UNHEALTHY'),
        (GuardHealth(True, False, True, False, True), 'MOTOR_LINK_UNHEALTHY'),
        (GuardHealth(True, False, True, True, False), 'AUTO_NOT_ENGAGED'),
    ]:
        result = guard.step((0.5, 0.2, 0.4), now_sec=now, command_age_sec=0.0,
                            health=health, profile='balanced', user_scale=1.0,
                            red_zone=False)
        assert result.allowed is False
        assert result.reason == reason
        assert result.velocity == (0.0, 0.0, 0.0)


def test_stale_and_nonfinite_commands_are_rejected():
    guard = make_guard()
    stale = guard.step((0.3, 0.0, 0.0), now_sec=1.0, command_age_sec=0.26,
                       health=ACTIVE, profile='balanced', user_scale=1.0,
                       red_zone=False)
    assert stale.reason == 'STALE_COMMAND'
    invalid = guard.step((math.nan, 0.0, 0.0), now_sec=1.1, command_age_sec=0.0,
                         health=ACTIVE, profile='balanced', user_scale=1.0,
                         red_zone=False)
    assert invalid.reason == 'INVALID_COMMAND'
    assert invalid.velocity == (0.0, 0.0, 0.0)


def test_stale_output_can_be_replaced_without_opening_the_motor_gate():
    guard = make_guard()
    stale = guard.step((.3, 0., 0.), now_sec=1., command_age_sec=.5,
                       health=ACTIVE, profile='balanced', user_scale=.8,
                       red_zone=False, rl_scale=.6)
    assert stale.velocity == (0., 0., 0.)
    assert not stale.allowed
    assert stale.applied_scale == 0.
    assert guard.reference_scale(ACTIVE, .8, False, .6) == pytest.approx(.48)
    assert guard.reference_scale(ACTIVE, 0., False) == 0.
    assert guard.reference_scale(ACTIVE, 1., True) == pytest.approx(.55)


@pytest.mark.parametrize('field,value', [
    ('armed', False), ('emergency_stop', True), ('tracking_ok', False),
    ('motor_link_ok', False), ('auto_engaged', False), ('rl_policy_ok', False),
])
def test_reference_clock_still_freezes_for_every_health_interlock(field, value):
    from dataclasses import replace
    health = replace(ACTIVE, **{field: value})
    assert make_guard().reference_scale(health, 1., False) == 0.


def test_red_zone_and_user_scale_apply_the_stricter_limit():
    guard = make_guard()
    result = None
    for index in range(400):
        result = guard.step((10.0, 10.0, 10.0), now_sec=index * 0.01,
                            command_age_sec=0.0, health=ACTIVE,
                            profile='balanced', user_scale=0.90, red_zone=True)
    assert result.applied_scale == 0.55
    x, y, w = result.velocity
    assert math.hypot(x / (0.72 * 0.55), y / (0.65 * 0.55)) <= 1.0 + 1.0e-10
    assert abs(w) <= 1.3 * 0.55 + 1.0e-10


def test_rl_scale_is_composed_into_planner_and_guard_limits():
    guard = make_guard()
    percentage = guard.planner_speed_percentage(
        profile='balanced', user_scale=0.8, rl_scale=0.6,
        red_zone=False, reference=(0.72, 0.65, 1.3),
    )
    assert percentage == pytest.approx(48.0)
    result = None
    for index in range(400):
        result = guard.step(
            (10.0, 0.0, 0.0), now_sec=index * 0.01,
            command_age_sec=0.0, health=ACTIVE, profile='balanced',
            user_scale=0.8, rl_scale=0.6, red_zone=False,
        )
    assert result.applied_scale == pytest.approx(0.48)
    assert result.velocity[0] <= 0.72 * 0.48 + 1.0e-10
    assert guard.effective_scale(1.0, False, math.nan) == 0.0


def test_combined_twist_never_exceeds_physical_wheel_limit():
    guard = make_guard()
    result = None
    for index in range(500):
        result = guard.step((1.0, -1.0, 1.8), now_sec=index * 0.01,
                            command_age_sec=0.0, health=ACTIVE,
                            profile='balanced', user_scale=1.0, red_zone=False)
    wheel = twist_to_wheel_speeds(
        *result.velocity, drive_model='omni4', wheel_radius=0.05,
        track_width=0.0, wheelbase=0.0, wheel_positions=POSITIONS,
        wheel_drive_angles=ANGLES, wheel_signs=SIGNS,
    )
    assert np.max(np.abs(wheel)) <= 15.709120382 + 1.0e-9


def test_guard_brakes_no_slower_than_a_constant_deceleration_profile():
    """Measure the stopping distance the guard can actually deliver.

    MPPI scores its rollouts with its own ``ax_min``, so if that bound is more
    optimistic than this measured distance the optimizer will accept gaps the
    robot cannot brake inside.  ``test_nav2_architecture`` pins the configured
    bound to the profile value; this pins the profile value to real behaviour.
    """
    guard = make_guard()
    profile_accel = 1.05
    dt = 0.01
    now = 0.0
    # Reach cruise speed first.
    for _ in range(300):
        result = guard.step((0.72, 0.0, 0.0), now_sec=now, command_age_sec=0.0,
                            health=ACTIVE, profile='balanced', user_scale=1.0,
                            red_zone=False)
        now += dt
    entry_speed = result.velocity[0]
    assert entry_speed > 0.70

    # A non-zero command is required: an exact zero is deliberately immediate.
    distance = 0.0
    while result.velocity[0] > 1.0e-3 and now < 20.0:
        result = guard.step((1.0e-6, 0.0, 0.0), now_sec=now,
                            command_age_sec=0.0, health=ACTIVE,
                            profile='balanced', user_scale=1.0, red_zone=False)
        distance += result.velocity[0] * dt
        now += dt

    ideal = entry_speed ** 2 / (2.0 * profile_accel)
    # Jerk limiting only adds a bounded ramp-in to the deceleration.
    assert ideal <= distance <= ideal * 1.6, (distance, ideal)


def test_planner_speed_percentage_never_exceeds_what_the_guard_executes():
    """The published SpeedLimit must keep every axis inside the guard.

    One percentage rescales all three controller maxima together, so only the
    smallest per-axis ratio guarantees nothing is planned above what this
    guard will pass through unclipped.
    """
    guard = make_guard()
    reference = (0.72, 0.65, 1.3)

    assert guard.planner_speed_percentage(
        profile='balanced', user_scale=1.0, red_zone=False, reference=reference,
    ) == 100.0

    for profile, scale, red_zone in (
        ('balanced', 1.0, False), ('balanced', 0.4, False),
        ('precision', 1.0, False), ('precision', 0.7, True),
        ('balanced', 1.0, True), ('unknown-profile', 0.55, False),
    ):
        percentage = guard.planner_speed_percentage(
            profile=profile, user_scale=scale, red_zone=red_zone,
            reference=reference,
        )
        limits = guard.profiles[guard.resolve_profile(profile)]
        applied = guard.effective_scale(scale, red_zone)
        planned = [percentage / 100.0 * value for value in reference]
        executed = [
            limits.linear * applied,
            limits.lateral * applied,
            limits.angular * applied,
        ]
        for planned_axis, executed_axis in zip(planned, executed):
            assert planned_axis <= executed_axis + 1.0e-9


def test_planner_speed_percentage_never_reads_as_no_limit():
    # Nav2 treats a zero speed limit as "no limit"; a closed gate is enforced
    # by step(), so this must stay strictly positive.
    guard = make_guard()
    percentage = guard.planner_speed_percentage(
        profile='balanced', user_scale=0.0, red_zone=True,
        reference=(0.72, 0.65, 1.3),
    )
    assert 0.0 < percentage <= 100.0


def test_upstream_zero_is_not_delayed_by_smoothing():
    guard = make_guard()
    for index in range(100):
        result = guard.step((0.6, 0.0, 0.0), now_sec=index * 0.01,
                            command_age_sec=0.0, health=ACTIVE,
                            profile='balanced', user_scale=1.0, red_zone=False)
    assert result.velocity[0] > 0.0
    stopped = guard.step((0.0, 0.0, 0.0), now_sec=1.01, command_age_sec=0.0,
                         health=ACTIVE, profile='balanced', user_scale=1.0,
                         red_zone=False)
    assert stopped.velocity == (0.0, 0.0, 0.0)


# The deployed drivetrain geometry: yaw drives all four wheels the same way and
# translation drives two each way, which is the structure the Pi mixer in
# bacon_gateway/src/move.cpp implements.
DEPLOYED_ANGLES = np.radians([135.0, 45.0, 225.0, -45.0])
MAX_WHEEL = 15.709120382


def _allocate(vx, vy, wz):
    from omni_autonomy_next.motor_kinematics import allocate_omni4_wheel_budget
    return allocate_omni4_wheel_budget(
        vx, vy, wz, wheel_radius=0.05, wheel_positions=POSITIONS,
        wheel_drive_angles=DEPLOYED_ANGLES, wheel_signs=SIGNS,
        maximum=MAX_WHEEL,
    )


def test_wheel_budget_scales_the_whole_twist_and_preserves_curvature():
    """Fitting a twist into the wheel budget must not change its shape.

    A uniform scale is a pure time rescaling of the planned trajectory, which
    the controller absorbs by replanning from the measured state each cycle.
    Scaling translation and yaw by different factors instead changes the
    curvature the base actually performs, so it leaves the planned path, the
    controller corrects, the allocator distorts the correction in turn and the
    loop hunts -- the reported left/right weave.  It also has to agree with the
    deployed Pi mixer, which already scales all four wheel commands by one
    common factor.
    """
    requests = [
        (0.78, 0.0, 1.30),     # saturating: 11.03 + 12.25 of 15.709 rad/s
        (0.55, 0.48, 1.10),    # saturating diagonal translation plus yaw
        (0.30, 0.0, 0.40),     # feasible: must pass through untouched
        (0.0, 0.0, 1.30),      # pure in-place rotation
        (-0.60, 0.35, -0.90),  # reverse, mixed signs
    ]
    for vx, vy, wz in requests:
        result = _allocate(vx, vy, wz)
        speeds = twist_to_wheel_speeds(
            *result, drive_model='omni4', wheel_radius=0.05,
            track_width=0.0, wheelbase=0.0, wheel_positions=POSITIONS,
            wheel_drive_angles=DEPLOYED_ANGLES, wheel_signs=SIGNS,
        )
        assert np.max(np.abs(speeds)) <= MAX_WHEEL + 1.0e-9, (vx, vy, wz)
        scales = [
            component / request
            for component, request in zip(result, (vx, vy, wz))
            if abs(request) > 1.0e-12
        ]
        # One factor for every axis: same travel direction, same curvature.
        assert max(scales) - min(scales) < 1.0e-9, (vx, vy, wz)
        assert 0.0 < scales[0] <= 1.0 + 1.0e-12, (vx, vy, wz)


def test_feasible_twist_is_not_reduced_at_all():
    assert _allocate(0.30, 0.0, 0.40) == pytest.approx((0.30, 0.0, 0.40))
    assert _allocate(0.0, 0.0, 1.30) == pytest.approx((0.0, 0.0, 1.30))


def test_saturating_forward_plus_yaw_beats_both_previous_allocations():
    """Uniform scaling dominates the two behaviours it replaced.

    Yaw absolute priority left 0.245 m/s of forward speed for this request and
    read as "it only rotates and never gets there".  The split allocation with
    translation_budget_share 0.45 left 0.500 m/s but executed 45% more yaw per
    metre than planned.
    """
    forward, lateral, yaw = _allocate(0.78, 0.0, 1.30)
    assert forward > 0.52
    assert lateral == 0.0
    # Curvature identical to the request.
    assert yaw / forward == pytest.approx(1.30 / 0.78)
