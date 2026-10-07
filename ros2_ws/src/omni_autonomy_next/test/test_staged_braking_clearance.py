"""Exact staged braking sweeps must reject the old midpoint false pass."""
import math
import json
from types import SimpleNamespace as NS

import numpy as np
import pytest

from omni_autonomy_next import staged_heading_node as staged
from omni_autonomy_next import trajectory_tracker_node as tracker
from omni_autonomy_next.rl_residual import CadClearanceModel
from omni_autonomy_next.staged_heading import HeadingStage
from test_execution_clearance import dense_oracle
from test_mixed_heading_clearance import field
from test_staged_heading import open_model, staged_node


def ready_node(monkeypatch, pose=(0., 0., 0.)):
    monkeypatch.setattr(staged.time, 'monotonic', lambda: 0.)
    node = staged_node(pose=pose)
    node.heading_stage = HeadingStage(0., 0., phase='TRANSLATE')
    node.raw_velocity = np.zeros(3)
    node.deceleration = .85
    node.yaw_acceleration = 1.2
    parameter = node.get_parameter
    node.get_parameter = lambda name: (type(parameter(name))(value=.32)
        if name == 'feedback_delay_sec' else parameter(name))
    return node


def test_whole_horizon_midpoint_false_pass_against_independent_integration(monkeypatch):
    node = ready_node(monkeypatch)
    node.clearance = CadClearanceModel(
        [[[-2., .517373012801603], [3., .517373012801603]]], field().footprint)
    request = np.array([.78, 0., .3])
    # Reconstruct the historical midpoint approximation to demonstrate the
    # exact false acceptance; the true trajectory below uses a time oracle.
    speed, rate, delay = .78, .3, .32
    horizon = delay + max(speed/.85, rate/1.2)
    travel = (speed+node.clearance.radius*rate)*horizon
    count = max(2, math.ceil(travel/.005))
    times = np.linspace(0., horizon, count+1)
    braking = np.clip(times-delay, 0., speed/.85)
    distance_factor = np.minimum(times, delay)+braking-.5*.85*braking**2/speed
    yaw_braking = np.clip(times-delay, 0., rate/1.2)
    yaw_factor = np.minimum(times, delay)+yaw_braking-.5*1.2*yaw_braking**2/rate
    midpoints = speed*distance_factor[:, None]*np.column_stack((
        np.cos(.5*rate*yaw_factor), np.sin(.5*rate*yaw_factor)))
    old_bound = np.min(node.clearance.clearance_over_poses(
        midpoints, rate*yaw_factor))-.5*travel/count
    assert old_bound >= .035
    _, truth, yaws, _ = dense_oracle(
        node.pose, request, delay, .85, 1.2, node.clearance.radius)
    assert np.min(node.clearance.clearance_over_poses(truth, yaws)) < .025

    applied = np.array(node._stage_safe_command(*request))
    assert np.linalg.norm(applied) < np.linalg.norm(request)
    _, truth, yaws, _ = dense_oracle(
        node.pose, applied, delay, .85, 1.2, node.clearance.radius)
    assert np.min(node.clearance.clearance_over_poses(truth, yaws)) >= .035


def test_low_positive_deceleration_is_not_inflated(monkeypatch):
    node = ready_node(monkeypatch, pose=(.75, 0., 0.))
    node.clearance = CadClearanceModel([[[0., -3.], [0., 3.]]], open_model().footprint)
    node.deceleration = .05
    vx, vy, wz = node._stage_safe_command(-.2, 0., 0.)
    assert -.2 < vx < 0. and vy == wz == 0.
    assert .75-.45-abs(vx)*.32-vx*vx/(2*.05) >= .035


@pytest.mark.parametrize('deceleration', [0., -.05, math.nan])
def test_invalid_deceleration_refuses_safely(monkeypatch, deceleration):
    node = ready_node(monkeypatch)
    node.deceleration = deceleration
    assert node._stage_safe_command(.05, 0., 0.) == (0., 0., 0.)
    assert node.stage_blocked.startswith('INVALID_BRAKING_SWEEP:')


def test_raw_inward_momentum_cannot_be_hidden_by_filtered_velocity(monkeypatch):
    node = ready_node(monkeypatch, pose=(.7, 0., 0.))
    node.clearance = CadClearanceModel([[[0., -3.], [0., 3.]]], open_model().footprint)
    node.raw_velocity = np.array([-.8, 0., 0.])
    assert np.array_equal(node.velocity, np.zeros(3))
    assert node._stage_safe_command(.05, 0., 0.) == (0., 0., 0.)
    assert node.stage_blocked == 'MEASURED_BRAKING_SWEEP_BLOCKED'


def test_reverse_retains_28mm_clearance_floor_exception(monkeypatch):
    node = ready_node(monkeypatch, pose=(.478, 0., 0.))
    node.clearance = CadClearanceModel([[[0., -3.], [0., 3.]]], open_model().footprint)
    node.reverse_goal = np.array([.6, 0., 0.])
    assert node.clearance.body_clearance(node.pose[:2], node.pose[2]) == pytest.approx(.028)
    assert node._stage_safe_command(.01, 0., 0.) == (.01, 0., 0.)


def test_oversized_finite_raw_envelope_refuses_before_path_allocation(monkeypatch):
    node = ready_node(monkeypatch)
    node.raw_velocity = np.array([1.e5, 0., 0.])
    monkeypatch.setattr(staged, 'braking_pose_path',
                        lambda *a, **kw: pytest.fail('oversized envelope allocated'))
    assert node._stage_safe_command(.05, 0., 0.) == (0., 0., 0.)
    assert 'work budget' in node.stage_blocked


def test_new_samples_cannot_renew_an_expired_staged_certificate(monkeypatch):
    node = ready_node(monkeypatch)
    clock = [0.]
    monkeypatch.setattr(staged.time, 'monotonic', lambda: clock[0])
    original = node.clearance.body_clearance
    def renewing(*args, **kwargs):
        result = original(*args, **kwargs)
        clock[0] = .31
        node.pose_stamp = node.velocity_stamp = clock[0]
        return result
    monkeypatch.setattr(node.clearance, 'body_clearance', renewing)
    assert node._stage_safe_command(.05, 0., 0.) == (0., 0., 0.)
    assert 'expired during sweep' in node.stage_blocked
    node.pose = None
    assert node._stage_safe_command(0., 0., 0.) == (0., 0., 0.)


@pytest.mark.parametrize('state', ['TRACKING', 'REVERSING'])
def test_post_publish_raw_rejection_is_visible_and_cannot_report_arrival(monkeypatch, state):
    node = ready_node(monkeypatch)
    node.active_goal = node.pose.copy()
    node.active_goal_stamp = [10, 2]
    node.raw_velocity = np.array([1.e5, 0., 0.])
    if state == 'REVERSING':
        node.reverse_goal = node.pose.copy()
    node._publish(.05, 0., 0.)
    assert np.array_equal(node.command, np.zeros(3))
    assert 'work budget' in node.stage_blocked
    node._publish(0., 0., 0.)
    assert 'work budget' in node.stage_blocked
    messages = []
    node.last_status = None
    node.status_pub = NS(publish=messages.append)
    node.get_logger = lambda: NS(info=lambda *args: None)
    for at in (0., .1, .35):
        monkeypatch.setattr(staged.time, 'monotonic', lambda: at)
        node.pose_stamp = node.velocity_stamp = at
        tracker.TrajectoryTracker._status(node, state)
        status = json.loads(messages[-1].data)
        assert status['stage_blocked_reason'] == node.stage_blocked
        assert status['arrival']['ready'] is False


@pytest.mark.parametrize('raw', [[.006, 0., 0.], [0., 0., .011], None])
def test_reverse_settlement_requires_the_same_strict_raw_stop_thresholds(monkeypatch, raw):
    node = ready_node(monkeypatch)
    node.reverse_goal = node.active_goal = node.pose.copy()
    node.raw_velocity = raw
    node.command = np.zeros(3)
    node.last_status = None
    messages = []
    node.status_pub = NS(publish=messages.append)
    node.get_logger = lambda: NS(info=lambda *args: None)
    for at in (0., .1, .2, .35):
        monkeypatch.setattr(staged.time, 'monotonic', lambda: at)
        node.pose_stamp = node.velocity_stamp = at
        tracker.TrajectoryTracker._status(node, 'REVERSING')
        assert json.loads(messages[-1].data)['arrival']['ready'] is False


def test_work_age_is_included_even_when_original_sources_remain_fresh(monkeypatch):
    node = ready_node(monkeypatch)
    clock = [0.]
    monkeypatch.setattr(staged.time, 'monotonic', lambda: clock[0])
    node.clearance = CadClearanceModel([[[0., -3.], [0., 3.]]], field().footprint)
    node.pose = np.array([.3-float(np.min(node.clearance.footprint[:, 0])), 0., 0.])
    node.pose_stamp = -.01
    node.raw_velocity = np.array([-.4, 0., 0.])
    original = node.clearance.body_clearance
    def slow_scalar(*args, **kwargs):
        result = original(*args, **kwargs)
        clock[0] = .136
        return result
    monkeypatch.setattr(node.clearance, 'body_clearance', slow_scalar)
    assert node._stage_safe_command(.05, 0., 0.) == (0., 0., 0.)
    assert node.stage_blocked == 'MEASURED_BRAKING_SWEEP_BLOCKED'
    assert clock[0]-node.pose_stamp < .3 and clock[0]-node.velocity_stamp < .2


def test_fresh_successful_reverse_recertification_clears_only_transient_failure(monkeypatch):
    node = ready_node(monkeypatch, pose=(.7, 0., 0.))
    node.clearance = CadClearanceModel([[[0., -3.], [0., 3.]]], open_model().footprint)
    node.reverse_goal = np.array([.8, 0., 0.])
    node.stage_blocked = 'MEASURED_BRAKING_SWEEP_BLOCKED'
    assert node._stage_safe_command(.05, 0., 0.) == (.05, 0., 0.)
    assert node.stage_blocked is None
    node.stage_blocked = 'ROTATION_TIMEOUT'
    assert node._stage_safe_command(.05, 0., 0.) == (0., 0., 0.)
    assert node.stage_blocked == 'ROTATION_TIMEOUT'
    node.reverse_ignore_cad = True
    assert node._stage_safe_command(.05, 0., 0.) == (0., 0., 0.)
    assert node.stage_blocked == 'ROTATION_TIMEOUT'
    node.stage_blocked = None
    assert node._stage_safe_command(.05, 0., 0.) == (.05, 0., 0.)


@pytest.mark.parametrize('fault', ['BRAKING_SWEEP_BLOCKED', 'MEASURED_BRAKING_SWEEP_BLOCKED'])
def test_stationary_reverse_explicitly_recertifies_zero_before_clearing_fault(monkeypatch, fault):
    node = ready_node(monkeypatch, pose=(.478, 0., 0.))
    node.clearance = CadClearanceModel([[[0., -3.], [0., 3.]]], open_model().footprint)
    node.reverse_goal = node.active_goal = node.pose.copy()
    node.reverse_started = node.reverse_heartbeat = 0.
    node.stage_blocked = fault
    node._publish(0., 0., 0.)
    assert node.stage_blocked == fault
    node._reverse_tick(0., node.pose, 0.)
    assert node.stage_blocked is None
    assert node.stage_measured_clearance_bound >= .025
    assert node.stage_execution_clearance_bound >= .025
    assert node.statuses[-1] == 'REVERSING'


@pytest.mark.parametrize('failure', ['moving_raw', 'stale_raw', 'permanent_fault'])
def test_stationary_reverse_cannot_clear_unhealthy_or_permanent_evidence(monkeypatch, failure):
    node = ready_node(monkeypatch, pose=(.478, 0., 0.))
    node.clearance = CadClearanceModel([[[0., -3.], [0., 3.]]], open_model().footprint)
    node.reverse_goal = node.active_goal = node.pose.copy()
    node.reverse_started = node.reverse_heartbeat = 0.
    node.stage_blocked = 'MEASURED_BRAKING_SWEEP_BLOCKED'
    if failure == 'moving_raw':
        node.raw_velocity = np.array([-1., 0., 0.])
    elif failure == 'stale_raw':
        node.velocity_stamp = -.21
    else:
        node.stage_blocked = 'ROTATION_TIMEOUT'
    node._reverse_tick(0., node.pose, 0.)
    assert node.stage_blocked is not None
    assert np.array_equal(node.command, np.zeros(3))


@pytest.mark.parametrize('safe', [False, True])
def test_translate_tick_and_bare_zero_cannot_erase_a_failed_proof(monkeypatch, safe):
    node = ready_node(monkeypatch, pose=(.7 if safe else .478, 0., 0.))
    node.clearance = CadClearanceModel(
        [[[0., -3.], [0., 3.]]], open_model().footprint)
    node.active_goal = node.pose.copy()
    node.active_goal_stamp = [10, 2]
    node.raw_velocity = np.array([-.02, 0., 0.])
    # These speeds satisfy existing general arrival thresholds. Only a fresh
    # complete CAD stopping proof can retire the preceding braking fault.
    node.velocity = np.zeros(3)
    node.stage_blocked = 'MEASURED_BRAKING_SWEEP_BLOCKED'
    node.command = np.zeros(3)
    assert not node._stage_tick(0., node.pose, node.velocity, 1.)
    node._publish(0., 0., 0.)
    messages = []
    node.last_status = None
    node.status_pub = NS(publish=messages.append)
    node.get_logger = lambda: NS(info=lambda *args: None)
    tracker.TrajectoryTracker._status(node, 'TERMINAL')
    status = json.loads(messages[-1].data)
    if safe:
        assert node.stage_blocked is None
        assert node.stage_measured_clearance_bound >= .035
        assert node.stage_execution_clearance_bound >= .035
        assert status['arrival']['ready'] is True
    else:
        assert node.stage_blocked in staged._TRANSIENT_BRAKING_REASONS
        assert status['stage_blocked_reason'] == node.stage_blocked
        assert status['arrival']['ready'] is False


def test_staged_measured_full_envelope_retains_future_work_headroom(monkeypatch):
    node = ready_node(monkeypatch, pose=(.55, 0., 0.))
    node.clearance = CadClearanceModel(
        [[[0., -3.], [0., 3.]]], open_model().footprint)
    node.raw_velocity = np.array([.4, 0., 0.])
    clock = [0.]
    monkeypatch.setattr(staged.time, 'monotonic', lambda: clock[0])
    original = node.clearance.clearance_over_poses
    def work_during_first_sampled_proof(*args, **kwargs):
        result = original(*args, **kwargs)
        clock[0] = .05
        return result
    monkeypatch.setattr(node.clearance, 'clearance_over_poses',
                        work_during_first_sampled_proof)
    # Initial100mm clearance is less than222mm total stopping travel, so the
    # raw envelope needs sampled proof. It moves AWAY from the wall throughout;
    # clipping that whole proof to40mm would lose real clearance before final E.
    applied = node._stage_safe_command(.05, 0., 0.)
    assert applied == pytest.approx((.05, 0., 0.))
    assert node.stage_blocked is None
    assert node.stage_measured_age_error == pytest.approx(.02)
    assert node.stage_measured_clearance_bound >= .035
    _, positions, yaws, _ = dense_oracle(
        node.pose, node.raw_velocity, .32+.05, .85, 1.2,
        node.clearance.radius, count=12001)
    modeled_minimum = np.min(original(positions, yaws, cap=.35))
    assert modeled_minimum == pytest.approx(.1)
    assert node.stage_measured_clearance_bound <= modeled_minimum
    assert clock[0]-node.pose_stamp < .3
    assert clock[0]-node.velocity_stamp < .2


def test_loose_staged_cheap_proof_falls_back_to_full_headroom(monkeypatch):
    node = ready_node(monkeypatch, pose=(.71, 0., 0.))
    node.clearance = CadClearanceModel(
        [[[0., -3.], [0., 3.]]], open_model().footprint)
    node.raw_velocity = np.array([.4, 0., 0.])
    clock = [0.]
    monkeypatch.setattr(staged.time, 'monotonic', lambda: clock[0])
    original = staged.StagedHeadingMixin._stage_checked_command
    def work_after_nominal_proofs(*args, **kwargs):
        result = original(*args, **kwargs)
        clock[0] = .05
        return result
    monkeypatch.setattr(staged.StagedHeadingMixin, '_stage_checked_command',
                        work_after_nominal_proofs)
    # The old Lipschitz35.38mm cheap bound passed its initial35mm floor,
    # then failed when20mm work-age reserve was subtracted. Actual motion
    # remains outward with260mm full-envelope clearance throughout.
    assert node._stage_safe_command(.05, 0., 0.) == pytest.approx((.05, 0., 0.))
    assert node.stage_blocked is None
    assert node.stage_measured_clearance_bound >= .035
    context = node.stage_certificate_context
    assert context['measured_certificate']['proof_kind'] == 'sampled'
    assert context['raw_twist'] == [.4, 0., 0.]
    assert context['selected_twist'] == [.05, 0., 0.]
    assert context['selected_certificate']['proof_kind'] == 'cheap'
    assert context['selected_certificate']['complete_bound_m'] > .035
    json.dumps(context, allow_nan=False)
    _, points, yaws, _ = dense_oracle(
        node.pose, node.raw_velocity, .32+.05, .85, 1.2,
        node.clearance.radius, count=12001)
    assert np.min(node.clearance.clearance_over_poses(points, yaws)) == pytest.approx(.26)
