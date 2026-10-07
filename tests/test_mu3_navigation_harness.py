"""Exercise the actual full-chain assertions without launching ROS or UDP."""
import ast
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest


SOURCE = Path(__file__).parents[1] / 'scripts/check_mu3_navigation_full_chain.py'
FUNCTIONS = {'pose_matches', 'current_goal_succeeded', 'arrival_evidence',
             'expected_right_active_goal', 'mirror_goal_is_current', 'mirror_evidence',
             'navigation_failure', 'post_fault_pi_evidence', 'observe_terminal_failure',
             'telemetry_checkpoint', 'assert_fresh_gateway_stop',
             'pi_evidence_after_checkpoint', 'collect_no_restart_evidence', 'assert_no_restart_window',
             'cleanup_processes', 'arrival_slot_selection', 'set_radio', 'spin'}
scope = {'math': math, 'os': os, 'signal': signal, 'subprocess': subprocess,
         'time': time, 'json': __import__('json')}
body = [node for node in ast.parse(SOURCE.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS]
exec(compile(ast.Module(body=body, type_ignores=[]), str(SOURCE), 'exec'), scope)


def evidence(**changes):
    arguments = dict(
        status={'remembered_pose': 'BAKETU2', 'field_side': 'left',
                'state': 'SUCCEEDED', 'request_id': 'current'},
        name='BAKETU2', previous_request_id='earlier',
        start_pose=(0., 0., 0.), target={'x': 0., 'y': 0., 'yaw': 0.},
        finish_pose=(0., 0., 0.), nonzero_uart=False, plan_points=0,
        wheel_commands=[0, 0, 0, 0])
    arguments.update(changes)
    return scope['arrival_evidence'](**arguments)


def test_exact_current_pose_succeeds_without_artificial_motion_or_plan():
    assert evidence() == {'already_settled': True, 'settled_zero_uart': True}


def test_exact_pose_cannot_hide_unrequested_motion():
    with pytest.raises(AssertionError, match='unexpected exact-pose movement'):
        evidence(nonzero_uart=True)


def test_small_localization_correction_within_tolerance_can_settle():
    assert evidence(start_pose=(.004, 0., .003), nonzero_uart=True)['already_settled']


@pytest.mark.parametrize('changes', [
    {'nonzero_uart': False, 'plan_points': 20},
    {'nonzero_uart': True, 'plan_points': 1},
])
def test_real_displacement_requires_both_motion_and_multi_pose_plan(changes):
    with pytest.raises(AssertionError, match='missing displacement'):
        evidence(start_pose=(1., 0., 0.), **changes)


def test_loading_a_keeps_approach_and_reverse_maneuver_evidence():
    status = {'remembered_pose': 'A', 'field_side': 'left',
              'state': 'SUCCEEDED', 'request_id': 'current'}
    with pytest.raises(AssertionError, match='missing displacement'):
        evidence(status=status, name='A')
    assert not evidence(status=status, name='A', nonzero_uart=True,
                        plan_points=12)['already_settled']


@pytest.mark.parametrize('changes', [
    {'request_id': 'earlier'}, {'request_id': None},
    {'field_side': 'right'}, {'remembered_pose': 'BAKETU3'}, {'state': 'ACTIVE'},
])
def test_retained_or_other_goal_success_is_not_current_success(changes):
    status = {'remembered_pose': 'BAKETU2', 'field_side': 'left',
              'state': 'SUCCEEDED', 'request_id': 'current', **changes}
    with pytest.raises(AssertionError):
        evidence(status=status)


@pytest.mark.parametrize('pose', [(0.021, 0., 0.), (0., 0., .021),
                                  (math.nan, 0., 0.)])
def test_success_status_does_not_replace_measured_settlement(pose):
    with pytest.raises(AssertionError, match='unsettled pose'):
        evidence(finish_pose=pose)


@pytest.mark.parametrize('wheels', [[1, 0, 0, 0], [], [0, 0, 0]])
def test_success_requires_complete_stopped_gateway_wheel_telemetry(wheels):
    with pytest.raises(AssertionError, match='unsettled wheels'):
        evidence(wheel_commands=wheels)


def test_yaw_wrap_does_not_invent_required_rotation():
    assert evidence(start_pose=(0., 0., -math.pi), finish_pose=(0., 0., math.pi),
                    target={'x': 0., 'y': 0., 'yaw': math.pi})['already_settled']


def mirror(**changes):
    arguments = dict(
        status={'remembered_pose': 'BAKETU2', 'field_side': 'right',
                'state': 'SENDING', 'request_id': 'current', 'goal_stamp': [123, 456]},
        name='BAKETU2', previous_request_id='earlier', started=10., status_received_at=11.,
        goal={'pose': [.81, 1.34, -math.pi], 'frame_id': 'map',
              'stamp': [123, 456], 'received_at': 11.},
        left_target={'x': -.81, 'y': 1.34, 'yaw': 0., 'frame_id': 'map'})
    arguments.update(changes)
    return scope['mirror_evidence'](**arguments)


def test_right_goal_matches_full_independent_reflection_and_request_stamp():
    result = mirror()
    assert result['goal_pose'] == [.81, 1.34, -math.pi]
    assert result['goal_stamp'] == [123, 456]
    assert result['request_id'] == 'current'


@pytest.mark.parametrize('pose', [[.82, 1.34, -math.pi], [.81, 1.35, -math.pi],
                                  [.81, 1.34, -math.pi+.01]])
def test_positive_x_does_not_mask_wrong_mirrored_coordinate_or_heading(pose):
    with pytest.raises(AssertionError, match='incorrect mirror'):
        mirror(goal={'pose': pose, 'frame_id': 'map', 'stamp': [123, 456], 'received_at': 11.})


@pytest.mark.parametrize('changes', [
    {'request_id': 'earlier'}, {'request_id': None}, {'field_side': 'left'},
    {'remembered_pose': 'BAKETU3'}, {'goal_stamp': [123, 457]}, {'goal_stamp': [0, 0]},
])
def test_right_coordinate_cannot_pass_other_request_or_side_or_timestamp(changes):
    status = {'remembered_pose': 'BAKETU2', 'field_side': 'right', 'state': 'SENDING',
              'request_id': 'current', 'goal_stamp': [123, 456], **changes}
    with pytest.raises(AssertionError, match='unrelated active goal'):
        mirror(status=status)


@pytest.mark.parametrize('part', ['goal', 'status'])
def test_retained_goal_and_status_must_both_arrive_after_current_command(part):
    changes = ({'goal': {'pose': [.81, 1.34, -math.pi], 'frame_id': 'map',
                         'stamp': [123, 456], 'received_at': 9.}}
               if part == 'goal' else {'status_received_at': 9.})
    with pytest.raises(AssertionError, match='unrelated active goal'):
        mirror(**changes)


def test_right_pose_requires_original_coordinate_frame():
    with pytest.raises(AssertionError, match='mirror frame'):
        mirror(goal={'pose': [.81, 1.34, -math.pi], 'frame_id': 'odom',
                     'stamp': [123, 456], 'received_at': 11.})


def test_loading_a_reflects_then_checks_gate_not_final_dock():
    left = {'x': -1.8, 'y': 4.75, 'yaw': -math.pi/2, 'frame_id': 'map'}
    expected = scope['expected_right_active_goal']('A', left)
    assert expected['x'] == pytest.approx(1.8)
    assert expected['y'] == pytest.approx(4.50)
    assert expected['yaw'] == pytest.approx(-math.pi/2)
    status = {'remembered_pose': 'A', 'field_side': 'right', 'request_id': 'current',
              'goal_stamp': [123, 456]}
    assert mirror(name='A', status=status, left_target=left,
                  goal={'pose': [1.8, 4.50, -math.pi/2], 'frame_id': 'map',
                        'stamp': [123, 456], 'received_at': 11.})['field_side'] == 'right'
    with pytest.raises(AssertionError, match='incorrect mirror'):
        mirror(name='A', status=status, left_target=left,
               goal={'pose': [1.8, 4.75, -math.pi/2], 'frame_id': 'map',
                     'stamp': [123, 456], 'received_at': 11.})


def test_oblique_loading_gate_uses_reflected_heading_before_offset():
    expected = scope['expected_right_active_goal']('A',
        {'x': -2., 'y': 3., 'yaw': math.pi/6, 'frame_id': 'map'})
    assert expected['x'] == pytest.approx(2.-.25*math.sqrt(3)/2)
    assert expected['y'] == pytest.approx(3.125)
    assert expected['yaw'] == pytest.approx(5*math.pi/6)


@pytest.mark.parametrize('status', [
    'DISARMED', 'DISARM_TIMEOUT', 'ARM_TIMEOUT', 'GOAL_ACK_TIMEOUT', 'TELEMETRY_LOST',
    'MU3_LOST', 'HEALTH_LOST', 'NOT_READY:LOCALIZATION_UNHEALTHY', 'FAILED', 'REJECTED',
    'ABORTED', 'CANCELED', 'UNKNOWN_REMEMBERED_POSE', 'INVALID_REMEMBERED_POSE',
    'UNKNOWN_SAVED_POSE', 'RELEASE_REQUIRED', 'RELEASED', 'SHUTDOWN',
])
def test_every_terminal_radio_navigation_stop_is_a_prompt_harness_failure(status):
    assert scope['navigation_failure'](status) == status


@pytest.mark.parametrize('status', ['READY: release required before navigation',
                                   'PREPARING:BAKETU2@left', 'GOAL_SENT:BAKETU2@left',
                                   'SUCCEEDED'])
def test_progress_and_success_are_not_failure_statuses(status):
    assert scope['navigation_failure'](status) is None


def telemetry_evidence(**changes):
    arguments = dict(telemetry={'kernel_arrival_realtime_ns': 10_040_000_000,
                                'bridge': {'telemetry_count': 101, 'telemetry_age_ms': 0.},
                                'pi': {'auto_engaged': False, 'wheel_commands': [0, 0, 0, 0]}},
                     receipt=10.05, failed_at=10., now=10.1, baseline_count=100,
                     failed_realtime_ns=10_000_000_000, now_realtime_ns=10_100_000_000)
    arguments.update(changes)
    return scope['post_fault_pi_evidence'](**arguments)


def test_only_fresh_post_fault_disengaged_zero_gateway_telemetry_proves_stop():
    result = telemetry_evidence()
    assert result['fresh'] and result['stopped']
    assert result['telemetry_count'] == 101


@pytest.mark.parametrize('changes', [
    {'receipt': 9.99}, {'receipt': 10.}, {'receipt': 10.2}, {'now': 10.31},
    {'baseline_count': 101}, {'baseline_count': None},
    {'telemetry': {'kernel_arrival_realtime_ns': 10_040_000_000,
                   'bridge': {'telemetry_count': 102, 'telemetry_age_ms': 251},
                   'pi': {'auto_engaged': False, 'wheel_commands': [0, 0, 0, 0]}}},
    {'telemetry': {'kernel_arrival_realtime_ns': 10_040_000_000,
                   'bridge': {'telemetry_count': 102, 'telemetry_age_ms': math.nan},
                   'pi': {'auto_engaged': False, 'wheel_commands': [0, 0, 0, 0]}}},
    {'telemetry': {'kernel_arrival_realtime_ns': 10_040_000_000,
                   'bridge': {'telemetry_count': 102},
                   'pi': {'auto_engaged': False, 'wheel_commands': [0, 0, 0, 0]}}},
])
def test_cached_retained_or_age_unknown_zero_cannot_prove_post_fault_stop(changes):
    result = telemetry_evidence(**changes)
    assert not result['fresh'] and not result['stopped']


@pytest.mark.parametrize('pi', [
    {'auto_engaged': True, 'wheel_commands': [0, 0, 0, 0]},
    {'wheel_commands': [0, 0, 0, 0]},
    {'auto_engaged': False, 'wheel_commands': [0, 1, 0, 0]},
    {'auto_engaged': False, 'wheel_commands': []},
    {'auto_engaged': False, 'wheel_commands': [0, 0, 0]},
    {'auto_engaged': False, 'wheel_commands': [0, 0, 0, False]},
    {'auto_engaged': False, 'wheel_commands': [0, 0, 0, math.nan]},
])
def test_pending_or_incomplete_gateway_stop_is_not_proven(pi):
    result = telemetry_evidence(telemetry={
        'kernel_arrival_realtime_ns': 10_040_000_000,
        'bridge': {'telemetry_count': 101, 'telemetry_age_ms': 0.}, 'pi': pi})
    assert result['fresh'] and not result['stopped']


@pytest.mark.parametrize('arrival_ns', [None, True, 9_999_999_999, 10_000_000_000,
                                      10_100_000_001])
def test_delayed_ros_or_udp_processing_cannot_invent_fresh_kernel_arrival(arrival_ns):
    result = telemetry_evidence(telemetry={
        'kernel_arrival_realtime_ns': arrival_ns,
        'bridge': {'telemetry_count': 101, 'telemetry_age_ms': 0.},
        'pi': {'auto_engaged': False, 'wheel_commands': [0, 0, 0, 0]}})
    assert not result['fresh'] and not result['stopped']


def test_recent_ros_receipt_cannot_hide_old_kernel_arrival():
    result = telemetry_evidence(now=10.5, receipt=10.49, now_realtime_ns=10_500_000_000)
    assert not result['fresh'] and not result['stopped']


def test_unsupported_kernel_timestamps_cannot_prove_zero_even_with_new_ros_update():
    result = telemetry_evidence(telemetry={
        'bridge': {'telemetry_count': 101, 'telemetry_age_ms': 0.},
        'pi': {'auto_engaged': False, 'wheel_commands': [0, 0, 0, 0]}})
    assert not result['fresh'] and not result['stopped']


def failure_observer(monkeypatch, samples=(), error=None):
    clock = SimpleNamespace(now=10.)
    initial = {'bridge': {'telemetry_count': 100, 'telemetry_age_ms': 0.},
               'pi': {'auto_engaged': False, 'wheel_commands': [0, 0, 0, 0]}}
    test_states = {'/motor/telemetry': initial}
    receipts = {'/motor/telemetry': 9.99}
    updates = iter(samples)
    def fake_spin(seconds):
        clock.now += seconds
        if error:
            raise error
        sample = next(updates, None)
        if sample is not None:
            sample = dict(sample)
            sample.setdefault('kernel_arrival_realtime_ns', int(clock.now*1.e9))
            test_states['/motor/telemetry'] = sample
            receipts['/motor/telemetry'] = clock.now
    for key, value in dict(time=SimpleNamespace(monotonic=lambda: clock.now,
                                               time_ns=lambda: int(clock.now*1.e9)),
                           states=test_states, received_at=receipts, began=0., spin=fake_spin).items():
        monkeypatch.setitem(scope, key, value)
    return scope['observe_terminal_failure']('HEALTH_LOST', 'arrival:A:left')


def test_failure_observation_waits_for_new_gateway_disengagement_and_preserves_failure(monkeypatch):
    samples = [
        {'bridge': {'telemetry_count': 101, 'telemetry_age_ms': 0.},
         'pi': {'auto_engaged': True, 'wheel_commands': [1, 0, 0, 0]}},
        {'bridge': {'telemetry_count': 102, 'telemetry_age_ms': 0.},
         'pi': {'auto_engaged': False, 'wheel_commands': [0, 0, 0, 0]}},
    ]
    result = failure_observer(monkeypatch, samples)
    assert result['failure'] == 'HEALTH_LOST'
    assert result['fresh_gateway_stop_observed'] and result['fresh_samples_observed'] == 2
    assert result['elapsed_sec'] == pytest.approx(.06)


def test_failure_observation_has_deadline_and_retained_zero_stays_unproven(monkeypatch):
    result = failure_observer(monkeypatch)
    assert result['failure'] == 'HEALTH_LOST'
    assert not result['fresh_gateway_stop_observed']
    assert result['fresh_samples_observed'] == 0
    assert result['elapsed_sec'] == pytest.approx(.8)


def test_repeated_same_fresh_packet_is_counted_once_without_inventing_stop(monkeypatch):
    result = failure_observer(monkeypatch, [
        {'bridge': {'telemetry_count': 101, 'telemetry_age_ms': 0.},
         'pi': {'auto_engaged': True, 'wheel_commands': [0, 0, 0, 0]}}])
    assert result['fresh_samples_observed'] == 1
    assert not result['fresh_gateway_stop_observed']


def test_callback_overrun_cannot_prove_gateway_stop_outside_observation_window(monkeypatch):
    clock = SimpleNamespace(now=10.)
    current = {'/motor/telemetry': {'bridge': {'telemetry_count': 100}}}
    receipts = {'/motor/telemetry': 9.99}
    def overrun_spin(seconds):
        assert seconds <= .03
        clock.now = 10.9
        current['/motor/telemetry'] = {
            'kernel_arrival_realtime_ns': 10_900_000_000,
            'bridge': {'telemetry_count': 101, 'telemetry_age_ms': 0.},
            'pi': {'auto_engaged': False, 'wheel_commands': [0, 0, 0, 0]}}
        receipts['/motor/telemetry'] = clock.now
    for key, value in dict(time=SimpleNamespace(monotonic=lambda: clock.now,
                                               time_ns=lambda: int(clock.now*1.e9)),
                           states=current, received_at=receipts, began=0., spin=overrun_spin).items():
        monkeypatch.setitem(scope, key, value)
    result = scope['observe_terminal_failure']('HEALTH_LOST', 'arrival:A:left')
    assert not result['fresh_gateway_stop_observed']
    assert result['fresh_samples_observed'] == 0
    assert result['elapsed_sec'] == pytest.approx(.9)
    assert result['deadline_overrun_sec'] == pytest.approx(.1)


def test_actual_spinner_caps_ros_wait_to_observation_time_remaining(monkeypatch):
    clock, waits = SimpleNamespace(now=10.), []
    def spin_once(node, timeout_sec):
        waits.append(timeout_sec)
        clock.now += timeout_sec
    monkeypatch.setitem(scope, 'time', SimpleNamespace(monotonic=lambda: clock.now))
    monkeypatch.setitem(scope, 'rclpy', SimpleNamespace(spin_once=spin_once))
    monkeypatch.setitem(scope, 'node', object())
    monkeypatch.setitem(scope, 'processes', [])
    scope['spin'](.012)
    assert waits == [pytest.approx(.012)]
    assert clock.now == pytest.approx(10.012)


@pytest.mark.parametrize('scenario', ['stop_button', 'radio_loss', 'idle',
                                    'radio_recovery_held_button', 'radio_recovery_release'])
def test_positive_stop_and_recovery_checks_cannot_pass_on_cached_zero(monkeypatch, scenario):
    clock = SimpleNamespace(now=10.)
    cache = {'bridge': {'telemetry_count': 100, 'telemetry_age_ms': 0.},
             'kernel_arrival_realtime_ns': 9_990_000_000,
             'pi': {'auto_engaged': False, 'wheel_commands': [0, 0, 0, 0]}}
    monkeypatch.setitem(scope, 'time', SimpleNamespace(monotonic=lambda: clock.now,
                                                       time_ns=lambda: int(clock.now*1.e9)))
    monkeypatch.setitem(scope, 'states', {'/motor/telemetry': cache})
    # Even a misleading new ROS receipt cannot replace a new kernel packet.
    monkeypatch.setitem(scope, 'received_at', {'/motor/telemetry': 10.79})
    checkpoint = scope['telemetry_checkpoint']()
    clock.now = 10.8
    with pytest.raises(AssertionError, match='fresh gateway stop unproven'):
        scope['assert_fresh_gateway_stop'](checkpoint, scenario)


def test_positive_safety_result_requires_new_packet_current_disengagement_and_complete_zero(monkeypatch):
    clock = SimpleNamespace(now=10.)
    current = {'/motor/telemetry': {'bridge': {'telemetry_count': 100}}}
    receipts = {'/motor/telemetry': 9.99}
    monkeypatch.setitem(scope, 'time', SimpleNamespace(monotonic=lambda: clock.now,
                                                       time_ns=lambda: int(clock.now*1.e9)))
    monkeypatch.setitem(scope, 'states', current)
    monkeypatch.setitem(scope, 'received_at', receipts)
    checkpoint = scope['telemetry_checkpoint']()
    clock.now = 10.8
    current['/motor/telemetry'] = {
        'kernel_arrival_realtime_ns': 10_790_000_000,
        'bridge': {'telemetry_count': 150, 'telemetry_age_ms': 0.},
        'pi': {'auto_engaged': False, 'wheel_commands': [0, 0, 0, 0]}}
    receipts['/motor/telemetry'] = 10.795
    result = scope['assert_fresh_gateway_stop'](checkpoint, 'radio_loss')
    assert result['fresh'] and result['stopped'] and result['telemetry_count'] == 150


@pytest.mark.parametrize('changes', [
    {'auto_engaged': False, 'wheel_commands': [1, 0, 0, 0]},
    {'auto_engaged': True, 'wheel_commands': [0, 0, 0, 0]},
    {'auto_engaged': True, 'wheel_commands': [1]},
])
def test_latched_old_motion_or_disengaged_incomplete_motion_cannot_authorize_fault_injection(changes):
    result = telemetry_evidence(telemetry={
        'kernel_arrival_realtime_ns': 10_040_000_000,
        'bridge': {'telemetry_count': 101, 'telemetry_age_ms': 0.}, 'pi': changes})
    assert result['fresh'] and not result['moving']


def test_only_fresh_engaged_complete_nonzero_wheels_prove_motion_at_injection():
    moving = {
        'kernel_arrival_realtime_ns': 10_040_000_000,
        'bridge': {'telemetry_count': 101, 'telemetry_age_ms': 0.},
        'pi': {'auto_engaged': True, 'wheel_commands': [-1, 1, -1, 1]}}
    result = telemetry_evidence(telemetry=moving)
    assert result['moving'] and not result['stopped']
    assert not telemetry_evidence(telemetry=moving, baseline_count=101)['moving']
    assert not telemetry_evidence(telemetry=moving, receipt=9.99)['moving']


def restart_window(monkeypatch, pi_updates, error=None):
    clock = SimpleNamespace(now=10.)
    current = {'/motor/telemetry': {'bridge': {'telemetry_count': 100}}}
    receipts = {'/motor/telemetry': 9.99}
    def fake_spin(seconds):
        for count, pi in enumerate(pi_updates, 101):
            clock.now += .1
            payload = {'kernel_arrival_realtime_ns': int(clock.now*1.e9),
                       'bridge': {'telemetry_count': count, 'telemetry_age_ms': 0.}, 'pi': pi}
            current['/motor/telemetry'] = payload
            receipts['/motor/telemetry'] = clock.now
            scope['collect_no_restart_evidence'](scope['stop_window'], payload,
                                                clock.now, int(clock.now*1.e9))
        if error:
            raise error
    for key, value in dict(time=SimpleNamespace(monotonic=lambda: clock.now,
                                               time_ns=lambda: int(clock.now*1.e9)),
                           states=current, received_at=receipts, spin=fake_spin,
                           stop_window=None).items():
        monkeypatch.setitem(scope, key, value)
    return scope['assert_no_restart_window'](.8, 'radio_recovery_held_button')


@pytest.mark.parametrize('transient', [
    {'auto_engaged': True, 'wheel_commands': [0, 0, 0, 0]},
    {'auto_engaged': False, 'wheel_commands': [1, 0, 0, 0]},
])
def test_idle_or_recovery_cannot_hide_transient_restart_with_final_fresh_zero(monkeypatch, transient):
    stopped = {'auto_engaged': False, 'wheel_commands': [0, 0, 0, 0]}
    with pytest.raises(AssertionError, match='transient gateway restart'):
        restart_window(monkeypatch, [transient, stopped])
    assert scope['stop_window'] is None


def test_clean_recovery_window_records_distinct_fresh_updates_and_final_zero(monkeypatch):
    result = restart_window(monkeypatch, [
        {'auto_engaged': False, 'wheel_commands': [0, 0, 0, 0]}]*3)
    assert result['fresh_samples_observed'] == 3
    assert result['first_violation'] is None and result['endpoint']['stopped']
    assert scope['stop_window'] is None


def test_failed_window_always_clears_callback_monitor_for_outer_cleanup(monkeypatch):
    with pytest.raises(RuntimeError, match='context invalid'):
        restart_window(monkeypatch, [], error=RuntimeError('context invalid'))
    assert scope['stop_window'] is None


@pytest.mark.parametrize('error', [RuntimeError('context invalid'), KeyboardInterrupt()])
def test_interrupted_observation_preserves_failure_for_guaranteed_finally_cleanup(monkeypatch, error):
    result = failure_observer(monkeypatch, error=error)
    assert result['failure'] == 'HEALTH_LOST'
    assert not result['fresh_gateway_stop_observed']
    assert result['observation_error'].startswith(type(error).__name__)


def test_invalid_ros_context_cannot_skip_real_child_cleanup(tmp_path):
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'],
                             start_new_session=True, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
    log = (tmp_path/'cleanup.log').open('w')
    def invalid_context():
        raise RuntimeError('publisher context is invalid')
    try:
        scope['cleanup_processes']([child], [log], invalid_context)
        assert child.poll() is not None
        assert log.closed
    finally:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=5)
        log.close()


def test_unresponsive_child_is_killed_and_reaped(monkeypatch):
    events = []
    def wait(timeout):
        events.append(('wait', timeout))
        if len([event for event in events if event[0] == 'wait']) == 1:
            raise subprocess.TimeoutExpired('fixture', timeout)
    child = SimpleNamespace(pid=4321, poll=lambda: None, wait=wait)
    monkeypatch.setattr(os, 'killpg', lambda pid, sig: events.append(('kill', pid, sig)))
    scope['cleanup_processes']([child], [], lambda: None)
    assert events == [('kill', 4321, signal.SIGINT), ('wait', 12),
                      ('kill', 4321, signal.SIGKILL), ('wait', 12)]


def test_bounded_arrival_selection_preserves_radio_slot_numbers():
    assert scope['arrival_slot_selection'](None) == tuple(range(1, 8))
    assert scope['arrival_slot_selection']('1,2') == (1, 2)
    assert scope['arrival_slot_selection']('3,7') == (3, 7)


@pytest.mark.parametrize('selection', ['', '1,1', '0', '8', 'wrong', '1,'])
def test_invalid_bounded_selection_fails_before_starting_processes(selection):
    with pytest.raises(ValueError):
        scope['arrival_slot_selection'](selection)


def test_atomic_radio_control_preserves_held_token_during_loss_and_recovery():
    frames = []
    scope.update(radio_token=0xc0, radio_enabled=True,
                 radio_control=SimpleNamespace(send=frames.append))
    control = scope['set_radio']
    control(token=0xc4)
    control(enabled=False)
    control(enabled=True)
    control(token=0xc0)
    assert frames == [bytes([1, 0xc4]), bytes([0, 0xc4]),
                      bytes([1, 0xc4]), bytes([1, 0xc0])]
