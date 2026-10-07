"""The deployed publisher must enforce fresh, measured stopping envelopes."""
import json
import threading
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped

from omni_autonomy_next import trajectory_tracker_node as tracker
from omni_autonomy_next.execution_clearance import (
    additional_delay_clearance_error, delayed_braking_certificate,
)
from omni_autonomy_next.rl_residual import CadClearanceModel


def node():
    root = Path(tracker.__file__).resolve().parents[1] / 'config'
    model = CadClearanceModel.from_yaml(
        root / 'field_planning.yaml', root / 'competition_footprints.yaml')
    params = dict(pose_timeout_sec=.3, velocity_timeout_sec=.2,
                  feedback_delay_sec=.32, velocity_filter_sec=.3)
    return NS(pose=np.array([-.81, 1.34, 0.]), pose_stamp=10.,
              raw_velocity=np.zeros(3), velocity=np.zeros(3), velocity_stamp=10.,
              clearance=model, motion_mode='simultaneous', reverse_goal=None,
              acceleration=.85, deceleration=.85, yaw_acceleration=1.2,
              get_parameter=lambda key: NS(value=params[key]), stopping=False)


def test_scaled_command_has_its_own_complete_certificate(monkeypatch):
    monkeypatch.setattr(tracker.time, 'monotonic', lambda: 10.)
    base = node()
    request = np.array([0., -.8, .1])
    applied = np.array(tracker.constrain_simultaneous_command(base, request))
    assert 0. < base.execution_clearance_scale < 1.
    assert applied == pytest.approx(request*base.execution_clearance_scale)
    safe, bound, rejected = delayed_braking_certificate(
        base.clearance, base.pose, applied, .32, .85, 1.2)
    assert safe and bound >= .025 and rejected is None
    assert base.execution_clearance_bound >= .025
    assert base.execution_clearance_reason is None
    # A rejected sample is explicitly separate from a complete envelope bound.
    assert base.execution_rejected_sample_bound < .025


def test_raw_momentum_can_veto_an_independently_safe_request(monkeypatch):
    monkeypatch.setattr(tracker.time, 'monotonic', lambda: 10.)
    base = node()
    base.raw_velocity = np.array([0., -.8, 0.])
    base.velocity = np.zeros(3)  # A filtered value must not hide momentum.
    requested = np.array([0., .05, 0.])
    assert delayed_braking_certificate(
        base.clearance, base.pose, requested, .32, .85, 1.2)[0]
    assert tracker.constrain_simultaneous_command(base, requested) == (0., 0., 0.)
    assert base.execution_clearance_reason == 'MEASURED_BRAKING_CLEARANCE_BLOCKED'
    assert base.measured_braking_clearance_bound is None
    assert base.execution_rejected_sample_bound < .025


@pytest.mark.parametrize('stamp', ['pose_stamp', 'velocity_stamp'])
def test_new_callbacks_cannot_renew_a_certificate_of_expired_inputs(monkeypatch, stamp):
    clock = [10.]
    monkeypatch.setattr(tracker.time, 'monotonic', lambda: clock[0])
    base = node()
    def updating_certificate(*args, **kwargs):
        clock[0] = 10.31
        setattr(base, stamp, clock[0])
        return True, .1, None
    monkeypatch.setattr(tracker, 'delayed_braking_certificate', updating_certificate)
    assert tracker.constrain_simultaneous_command(base, (.05, 0., 0.)) == (0., 0., 0.)
    assert 'expired during certificate' in base.execution_clearance_reason


def test_pose_acquisition_age_is_part_of_the_modeled_delay(monkeypatch):
    monkeypatch.setattr(tracker.time, 'monotonic', lambda: 10.)
    base = node()
    base.pose_stamp = 9.9
    delays = []
    def certificate(model, pose, command, delay, *args, **kwargs):
        delays.append(delay)
        return True, .1, None
    monkeypatch.setattr(tracker, 'delayed_braking_certificate', certificate)
    assert tracker.constrain_simultaneous_command(base, (.05, 0., 0.)) == (.05, 0., 0.)
    assert delays == pytest.approx([.42, .42])


def test_accepted_raw_velocity_is_retained_before_filtering_replay_rejected(monkeypatch):
    monkeypatch.setattr(tracker.time, 'monotonic', lambda: 10.)
    base = node()
    base.lock = threading.Lock()
    base.velocity_stamp = 9.9
    base.velocity_source_stamp = 19.9
    base.get_clock = lambda: NS(now=lambda: NS(nanoseconds=20_000_000_000))
    message = Odometry()
    message.header.frame_id = 'odom'
    message.child_frame_id = 'base_link'
    message.header.stamp.sec = 20
    message.twist.twist.linear.x = 2.
    tracker.TrajectoryTracker._on_odom(base, message)
    assert base.raw_velocity == pytest.approx([2., 0., 0.])
    assert 0. < base.velocity[0] < 1.
    message.twist.twist.linear.x = 30.
    tracker.TrajectoryTracker._on_odom(base, message)
    assert base.raw_velocity == pytest.approx([2., 0., 0.])
    message.header.stamp.nanosec = 1
    message.header.frame_id = 'map'
    tracker.TrajectoryTracker._on_odom(base, message)
    assert base.raw_velocity == pytest.approx([2., 0., 0.])


def test_publisher_gates_behavior_commands_and_zero_remains_unconditional(monkeypatch):
    monkeypatch.setattr(tracker.time, 'monotonic', lambda: 10.)
    base = node()
    messages = []
    base.command_pub = NS(publish=messages.append)
    base.raw_velocity = np.array([0., -.8, 0.])
    tracker.TrajectoryTracker._publish(base, 0., .05, 0.)
    assert base.command == pytest.approx(np.zeros(3))
    assert messages[-1].linear.y == 0.
    # Stopping does not require a usable pose, velocity or CAD query.
    base.pose = None
    tracker.TrajectoryTracker._publish(base, 0., 0., 0.)
    assert base.command == pytest.approx(np.zeros(3))


@pytest.mark.parametrize('state', ['TRACKING', 'BEHAVIOR', 'TERMINAL',
                                  'EXECUTION_CLEARANCE_BLOCKED'])
def test_execution_failure_never_reports_arrival_even_in_terminal(monkeypatch, state):
    monkeypatch.setattr(tracker.time, 'monotonic', lambda: 10.)
    base = node()
    base.active_goal = base.pose.copy()
    base.active_goal_stamp = [10, 2]
    base.command = np.zeros(3)
    base.last_status = None
    base.execution_clearance_reason = 'MEASURED_BRAKING_CLEARANCE_BLOCKED'
    base.execution_clearance_scale = 0.
    base.execution_clearance_bound = base.measured_braking_clearance_bound = None
    messages = []
    base.status_pub = NS(publish=messages.append)
    base.get_logger = lambda: NS(info=lambda *args: None)
    tracker.TrajectoryTracker._status(base, state)
    status = json.loads(messages[-1].data)
    assert status['state'] == 'EXECUTION_CLEARANCE_BLOCKED'
    assert status['arrival']['ready'] is False
    assert status['execution_clearance_reason'] == base.execution_clearance_reason


def test_configured_reverse_exception_preserves_its_separate_gate(monkeypatch):
    base = node()
    base.reverse_goal = np.zeros(3)
    def forbidden(*args):
        pytest.fail('simultaneous certificate must not replace authorized reverse gate')
    monkeypatch.setattr(tracker, 'delayed_braking_certificate', forbidden)
    assert tracker.constrain_simultaneous_command(base, (-.02, 0., 0.)) == (-.02, 0., 0.)


def test_control_flow_and_status_use_the_applied_command(monkeypatch):
    from test_smooth_arrival import make_node
    monkeypatch.setattr(tracker.time, 'monotonic', lambda: 0.)
    base = make_node(goal=(1., 0., .2))
    tracker.TrajectoryTracker._build_trajectory(base, np.array([[0., 0.], [1., 0.]]), 0.)
    outputs, flows, statuses = [], [], []
    def constrained_publish(*command):
        outputs.append(np.asarray(command)*.5)
        base.command = outputs[-1]
    base._publish = constrained_publish
    base._track_flow = lambda yaw, measured, vx, vy: flows.append((vx, vy))
    base._status = lambda state, **details: statuses.append((state, details))
    tracker.TrajectoryTracker._tick(base)
    assert outputs and np.linalg.norm(outputs[-1]) > 0.
    assert flows[-1] == pytest.approx(outputs[-1][:2])
    assert statuses[-1][1]['yaw_rate'] == pytest.approx(outputs[-1][2], abs=.0005)


@pytest.mark.parametrize('raw', [None, [float('nan'), 0., 0.], [.03, 0., 0.], [0., 0., .03]])
def test_filtered_stop_cannot_hide_missing_invalid_or_moving_raw_sample(monkeypatch, raw):
    monkeypatch.setattr(tracker.time, 'monotonic', lambda: 10.)
    base = node()
    base.active_goal = base.pose.copy()
    base.raw_velocity = raw
    base.last_status = None
    messages = []
    base.status_pub = NS(publish=messages.append)
    base.get_logger = lambda: NS(info=lambda *args: None)
    tracker.TrajectoryTracker._status(base, 'TRACKING')
    assert json.loads(messages[-1].data)['arrival']['ready'] is False


def test_measured_nominal_pass_cannot_ignore_fresh_but_elapsed_work_age(monkeypatch):
    clock = [10.]
    monkeypatch.setattr(tracker.time, 'monotonic', lambda: clock[0])
    base = node()
    base.clearance = CadClearanceModel([[[0., -3.], [0., 3.]]], base.clearance.footprint)
    base.pose = np.array([.3-float(np.min(base.clearance.footprint[:, 0])), 0., 0.])
    base.pose_stamp = 9.99  # Initial delay .32+.01; work remains within .30 s.
    base.raw_velocity = np.array([-.4, 0., 0.])
    original = tracker.delayed_braking_certificate
    calls = []
    def slow_first_proof(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append(result)
        clock[0] = 10.136
        return result
    monkeypatch.setattr(tracker, 'delayed_braking_certificate', slow_first_proof)
    assert tracker.constrain_simultaneous_command(base, (.05, 0., 0.)) == (0., 0., 0.)
    assert calls[0][0] and calls[0][1] >= .025
    assert base.measured_braking_age_error == pytest.approx(.0544)
    assert base.measured_braking_clearance_bound < .025
    assert base.execution_clearance_reason == 'MEASURED_BRAKING_AGE_CLEARANCE_BLOCKED'
    assert clock[0]-base.pose_stamp < .3 and clock[0]-base.velocity_stamp < .2


def test_age_reserve_selects_a_certified_lower_candidate_during_search(monkeypatch):
    clock = [10.]
    monkeypatch.setattr(tracker.time, 'monotonic', lambda: clock[0])
    base = node()
    def finite_proof(model, pose, command, delay, *args, margin=.025,
                     query_headroom_m=0., diagnostics=None):
        if np.array_equal(command, np.zeros(3)):
            clock[0] = 10.1
            return True, .1, None
        return (True, .04, None) if margin <= .04 else (False, None, .04)
    monkeypatch.setattr(tracker, 'delayed_braking_certificate', finite_proof)
    applied = np.asarray(tracker.constrain_simultaneous_command(base, (.4, 0., 0.)))
    assert 0. < applied[0] < .4 and applied[1] == applied[2] == 0.
    assert base.execution_age_error == pytest.approx(applied[0]*.1)
    assert base.execution_clearance_bound == pytest.approx(.04-base.execution_age_error)
    assert base.execution_clearance_bound >= .025
    assert base.execution_clearance_reason is None


_MEASURED_SNAPSHOTS = json.loads((Path(__file__).parent / 'data' /
    'compute_age_measured_snapshots.json').read_text())['cases']


@pytest.mark.parametrize('snapshot', _MEASURED_SNAPSHOTS,
                         ids=lambda snapshot: snapshot['pair'])
def test_captured_measured_envelope_keeps_headroom_for_actual_work_age(
        monkeypatch, snapshot):
    from test_execution_clearance import dense_oracle
    clock = [10.]
    monkeypatch.setattr(tracker.time, 'monotonic', lambda: clock[0])
    base = node()
    base.pose = np.asarray(snapshot['pose'])
    base.pose_stamp -= snapshot['pose_age_sec']
    base.velocity_stamp -= snapshot['velocity_age_sec']
    base.raw_velocity = np.asarray(snapshot['raw_velocity'])
    delay = .32+snapshot['pose_age_sec']
    elapsed = snapshot['work_elapsed_sec']
    reserve = additional_delay_clearance_error(
        base.raw_velocity, delay, .85, base.clearance.radius, elapsed)
    original = tracker.delayed_braking_certificate
    old_safe, truncated, _ = original(
        base.clearance, base.pose, base.raw_velocity, delay, .85, 1.2)
    assert old_safe and truncated-reserve < .025
    observed_proofs = []
    def work_after_first_complete_proof(*args, **kwargs):
        result = original(*args, **kwargs)
        observed_proofs.append(result)
        clock[0] = 10.+elapsed
        return result
    monkeypatch.setattr(tracker, 'delayed_braking_certificate',
                        work_after_first_complete_proof)
    applied = tracker.constrain_simultaneous_command(base, (.05, 0., 0.))
    assert applied == pytest.approx((.05, 0., 0.))
    assert base.execution_clearance_reason is None
    assert base.measured_braking_clearance_bound >= .025
    assert base.measured_braking_clearance_bound == pytest.approx(
        observed_proofs[0][1]-reserve)
    # Independently integrate the COMPLETE longer-delay envelope. The captured
    # nearby acquisition snapshot is not claimed to be the tracker input.
    _, points, yaws, _ = dense_oracle(
        base.pose, base.raw_velocity, delay+elapsed, .85, 1.2,
        base.clearance.radius, count=12001)
    modeled_minimum = np.min(base.clearance.clearance_over_poses(
        points, yaws, cap=1.))
    assert modeled_minimum >= .025
    assert base.measured_braking_clearance_bound <= modeled_minimum
    assert clock[0]-base.pose_stamp < .3
    assert clock[0]-base.velocity_stamp < .2


def failed_stationary_node(monkeypatch):
    monkeypatch.setattr(tracker.time, 'monotonic', lambda: 10.)
    base = node()
    footprint = np.array([[-.1, -.1], [.1, -.1], [.1, .1], [-.1, .1]])
    base.clearance = CadClearanceModel([[[.139, -3.], [.139, 3.]]], footprint)
    base.pose = base.active_goal = np.zeros(3)
    base.active_goal_stamp = [10, 2]
    base.raw_velocity = np.array([.02, 0., 0.])
    base.command_pub = NS(publish=lambda message: None)
    tracker.TrajectoryTracker._publish(base, -.01, 0., 0.)
    assert base.execution_clearance_reason == 'MEASURED_BRAKING_CLEARANCE_BLOCKED'
    return base


@pytest.mark.parametrize('bypass', ['none', 'staged', 'reverse', 'no_model'])
def test_bare_zero_and_mode_bypass_preserve_failed_arrival_evidence(monkeypatch, bypass):
    base = failed_stationary_node(monkeypatch)
    if bypass == 'staged':
        base.motion_mode = 'staged_heading'
    elif bypass == 'reverse':
        base.reverse_goal = np.zeros(3)
    elif bypass == 'no_model':
        base.clearance = None
    tracker.TrajectoryTracker._publish(base, 0., 0., 0.)
    assert base.command == pytest.approx(np.zeros(3))
    assert base.execution_clearance_reason == 'MEASURED_BRAKING_CLEARANCE_BLOCKED'
    assert base.execution_clearance_scale == 0.
    base.last_status = None
    messages = []
    base.status_pub = NS(publish=messages.append)
    base.get_logger = lambda: NS(info=lambda *args: None)
    tracker.TrajectoryTracker._status(base, 'TERMINAL')
    status = json.loads(messages[-1].data)
    assert status['state'] == 'EXECUTION_CLEARANCE_BLOCKED'
    assert status['arrival']['ready'] is False


def test_successful_fresh_raw_and_requested_proofs_can_retire_failure(monkeypatch):
    base = failed_stationary_node(monkeypatch)
    base.raw_velocity = np.zeros(3)
    tracker.TrajectoryTracker._publish(base, -.01, 0., 0.)
    assert base.command == pytest.approx((-.01, 0., 0.))
    assert base.execution_clearance_reason is None
    assert base.measured_braking_clearance_bound >= .025
    assert base.execution_clearance_bound >= .025


def test_same_goal_reverse_hold_cannot_override_retained_execution_failure(monkeypatch):
    base = failed_stationary_node(monkeypatch)
    base.reverse_goal = base.active_goal.copy()
    base.raw_velocity = np.zeros(3)
    base.stage_blocked = None
    base.last_status = None
    messages = []
    base.status_pub = NS(publish=messages.append)
    base.get_logger = lambda: NS(info=lambda *args: None)
    for at in (10., 10.1, 10.2, 10.35):
        monkeypatch.setattr(tracker.time, 'monotonic', lambda: at)
        base.pose_stamp = base.velocity_stamp = at
        tracker.TrajectoryTracker._publish(base, 0., 0., 0.)
        tracker.TrajectoryTracker._status(base, 'REVERSING')
        status = json.loads(messages[-1].data)
        assert status['arrival']['goal_stamp'] == [10, 2]
        assert status['execution_clearance_reason'] == 'MEASURED_BRAKING_CLEARANCE_BLOCKED'
        assert status['arrival']['ready'] is False


def test_only_validated_newer_goal_can_reset_failed_goal_evidence(monkeypatch):
    base = failed_stationary_node(monkeypatch)
    old_stage_context = {'pose': [0., 0., 0.], 'initial_delay_sec': .32}
    base.stage_certificate_context = old_stage_context
    base.lock = threading.Lock()
    resets = []
    base._stage_new_goal = lambda: resets.append(True)
    message = PoseStamped()
    message.header.frame_id = 'map'
    message.header.stamp.sec, message.header.stamp.nanosec = 10, 2
    message.pose.orientation.w = 1.
    for stamp in ((10, 2), (9, 999)):
        message.header.stamp.sec, message.header.stamp.nanosec = stamp
        message.pose.position.x = .2  # Conflicting old selections are ignored.
        tracker.TrajectoryTracker._on_goal(base, message)
        assert base.active_goal_stamp == [10, 2]
        assert base.active_goal == pytest.approx(np.zeros(3))
        assert base.execution_clearance_reason == 'MEASURED_BRAKING_CLEARANCE_BLOCKED'
        assert base.stage_certificate_context is old_stage_context
        assert not resets
    message.header.stamp.sec, message.header.stamp.nanosec = 11, 1
    message.header.frame_id = 'odom'
    tracker.TrajectoryTracker._on_goal(base, message)
    assert not resets and base.execution_clearance_reason
    assert base.stage_certificate_context is old_stage_context
    message.header.frame_id = 'map'
    tracker.TrajectoryTracker._on_goal(base, message)
    assert resets == [True]
    assert base.active_goal_stamp == [11, 1]
    assert base.active_goal == pytest.approx((.2, 0., 0.))
    assert base.execution_clearance_reason is None
    assert base.execution_clearance_scale == 1.
    assert base.stage_certificate_context is None


def test_execution_context_tracks_applied_candidate_and_captured_inputs(monkeypatch):
    monkeypatch.setattr(tracker.time, 'monotonic', lambda: 10.)
    base = node()
    requested = np.array([0., -.8, .1])
    applied = np.asarray(tracker.constrain_simultaneous_command(base, requested))
    context = base.execution_certificate_context
    assert context['pose'] == pytest.approx(base.pose)
    assert context['raw_twist'] == [0., 0., 0.]
    assert context['requested_twist'] == pytest.approx(requested)
    assert context['selected_twist'] == pytest.approx(applied)
    assert context['initial_delay_sec'] == .32
    assert context['pose_acquisition_age_sec'] == 0.
    assert context['velocity_acquisition_age_sec'] == 0.
    assert context['selected_certificate']['complete_bound_m'] == pytest.approx(
        base.execution_clearance_bound)
    assert context['selected_certificate']['proof_kind'] in ('cheap', 'sampled')
    # The initial requested rejection is not relabeled as the selected proof.
    assert context['requested_certificate']['proof_kind'] == 'sampled_rejected'
    json.dumps(context, allow_nan=False)


@pytest.mark.parametrize('source', ['pose', 'raw_velocity'])
def test_invalid_execution_snapshot_cannot_serialize_nonfinite_context(monkeypatch, source):
    monkeypatch.setattr(tracker.time, 'monotonic', lambda: 10.)
    base = node()
    setattr(base, source, np.array([float('nan'), 0., 0.]))
    assert tracker.constrain_simultaneous_command(base, (.05, 0., 0.)) == (0., 0., 0.)
    assert base.execution_certificate_context is None
    assert base.execution_clearance_reason.startswith('INVALID_EXECUTION_CLEARANCE:')
