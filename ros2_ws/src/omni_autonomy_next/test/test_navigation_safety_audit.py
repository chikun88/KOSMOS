"""Navigation boundary regressions runnable without a ROS installation.

Callbacks are compiled directly from the production AST, with message/clock
objects supplied here. Integration and controller replay tests still use ROS.
"""
import ast
import json
import math
from pathlib import Path
import threading
from types import MethodType, SimpleNamespace as NS

import numpy as np
import pytest

from omni_autonomy_next.field_side import normalize_side
from omni_autonomy_next.mu3_navigation import RemoteNavigation
from omni_autonomy_next.remembered_poses import (
    load_remembered_poses, normalize_pose_name, save_remembered_poses,
)
from omni_autonomy_next.staged_heading import dense_path, free_rotation_disk
from omni_autonomy_next.bucket_transit import FixedBucketTransit
from omni_autonomy_next.source_freshness import message_stamp_nanoseconds


SOURCE = Path(__file__).resolve().parents[1] / 'omni_autonomy_next'


def callbacks(filename, names, **namespace):
    source = ast.parse((SOURCE / filename).read_text(encoding='utf-8'))
    definitions = [node for node in ast.walk(source)
                   if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                   and node.name in names]
    for node in definitions:
        node.decorator_list = []
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[
        ast.alias(name='annotations')], level=0), *definitions], type_ignores=[])
    env = dict(math=math, json=json, np=np, time=NS(monotonic=lambda: 10.),
               normalize_pose_name=normalize_pose_name, normalize_side=normalize_side,
               message_stamp_nanoseconds=message_stamp_nanoseconds,
               **namespace)
    exec(compile(ast.fix_missing_locations(module), str(SOURCE / filename), 'exec'), env)
    return {name: env[name] for name in names}


def message(frame='map', x=2., yaw=0., stamp=10.):
    return NS(header=NS(frame_id=frame, stamp=NS(
        sec=int(stamp), nanosec=int(round((stamp-int(stamp))*1.e9)))),
        pose=NS(position=NS(x=x, y=0., z=0.), orientation=NS(
            x=0., y=0., z=math.sin(yaw/2), w=math.cos(yaw/2))))


def logger():
    return NS(info=lambda *a: None, warning=lambda *a: None, error=lambda *a: None)


@pytest.mark.parametrize('name,poses', [
    ('missing', {}), ('', {}), ('A', {'A': {'frame_id': 'odom'}}),
])
def test_invalid_saved_request_never_changes_the_field_or_running_action(name, poses):
    methods = callbacks('goal_bridge_node.py', ['_parse_remembered_request', '_remembered_goal_cb'])
    side_changes, queued, statuses = [], [], []
    node = NS(configured_frame='map', field_side='left',
              _parse_remembered_request=methods['_parse_remembered_request'],
              effective_remembered_poses=lambda: poses,
              _set_field_side=lambda *a, **k: side_changes.append(a),
              _queue_request=queued.append, get_logger=logger,
              _publish_status=lambda *a, **k: statuses.append((a, k)))
    methods['_remembered_goal_cb'](node, NS(data=json.dumps({'name': name, 'side': 'right'})))
    assert not side_changes and not queued
    assert statuses[-1][0] == ('INVALID_REMEMBERED_POSE',)


@pytest.mark.parametrize('frame,stamp', [('odom', 10.), ('map', 8.), ('map', 10.2)])
def test_invalid_localization_does_not_change_route_coordinates_or_pose_freshness(frame, stamp):
    method = callbacks('goal_bridge_node.py', ['_pose_cb'])['_pose_cb']
    node = NS(configured_frame='map', current_position=(1., 0.),
              current_pose={'x': 1.}, current_pose_received_at=9.9,
              get_parameter=lambda n: NS(value=1.),
              get_clock=lambda: NS(now=lambda: NS(nanoseconds=int(10.e9))))
    msg = message(frame=frame, stamp=stamp)
    msg.pose = NS(pose=msg.pose)
    method(node, msg)
    assert node.current_position == (1., 0.)
    assert node.current_pose_received_at == 9.9


def test_localization_age_and_acquisition_order_follow_the_sensor_clock():
    method = callbacks('goal_bridge_node.py', ['_pose_cb'])['_pose_cb']
    node = NS(configured_frame='map', current_position=None, current_pose=None,
              current_pose_received_at=-math.inf,
              get_parameter=lambda n: NS(value=1.),
              get_clock=lambda: NS(now=lambda: NS(nanoseconds=int(10.e9))))
    for stamp, x in [(9.8, 2.), (9.7, -100.)]:
        msg = message(stamp=stamp, x=x)
        msg.pose = NS(pose=msg.pose)
        method(node, msg)
    assert node.current_position == (2., 0.)
    assert node.current_pose_received_at == pytest.approx(9.8)


@pytest.mark.parametrize('target', [None, 1, 'abc', ['2', 0., 0.], [True, 0., 0.],
                                    [math.nan, 0., 0.], [2., 0.]])
def test_malformed_arrival_cannot_crash_or_acknowledge_a_goal(target):
    method = callbacks('goal_bridge_node.py', ['_tracker_arrived'])['_tracker_arrived']
    node = NS(tracker_arrival={'goal': target, 'ready': True},
              tracker_arrival_stamp=10., active_request=NS(pose=message()))
    assert method(node, 10.) is False


def test_quaternion_yaw_is_scale_invariant_and_rejects_invalid_rotations():
    method = callbacks('trajectory_tracker_node.py', ['yaw_of'])['yaw_of']
    q = message(yaw=math.pi/2).pose.orientation
    assert method(q) == pytest.approx(math.pi/2)
    q.z *= 3.
    q.w *= 3.
    assert method(q) == pytest.approx(math.pi/2)
    assert math.isnan(method(NS(x=0., y=0., z=0., w=0.)))
    assert math.isnan(method(NS(x=0., y=0., z=math.inf, w=1.)))


@pytest.mark.parametrize('frame,x,yaw', [('odom', 2., 0.), ('', 2., 0.),
                                        ('map', math.nan, 0.), ('map', 2., math.nan)])
def test_tracker_rejects_invalid_goals_before_resetting_the_running_controller(frame, x, yaw):
    methods = callbacks('trajectory_tracker_node.py', ['yaw_of', '_on_goal'])
    methods['_on_goal'].__globals__['yaw_of'] = methods['yaw_of']
    reset = []
    original = np.array([1., 0., 0.])
    node = NS(lock=threading.Lock(), active_goal=original, motion_mode='simultaneous',
              _stage_new_goal=lambda: reset.append(True))
    methods['_on_goal'](node, message(frame=frame, x=x, yaw=yaw))
    assert node.active_goal is original
    assert not reset


@pytest.mark.parametrize('mode', ['simultaneous', 'staged_heading'])
def test_wrong_frame_plan_invalidates_the_previous_route_in_both_modes(mode):
    method = callbacks('trajectory_tracker_node.py', ['_on_plan'])['_on_plan']
    node = NS(lock=threading.Lock(), motion_mode=mode, trajectory=object(), stage_revision=2)
    method(node, NS(header=NS(frame_id='odom', stamp=NS(sec=10, nanosec=0)), poses=[]))
    assert node.trajectory is None and node.stage_revision == 3
    assert node.stage_blocked == 'PLAN_FRAME_MISMATCH'


@pytest.mark.parametrize('mode', ['simultaneous', 'staged_heading'])
@pytest.mark.parametrize('fault', ['mixed_frame', 'empty_frame', 'zero_parent_stamp',
                                  'zero_pose_stamp', 'negative_nanoseconds',
                                  'overflow_nanoseconds', 'zero_rotation',
                                  'nonfinite_coordinate', 'empty_path'])
def test_invalid_path_cannot_keep_or_rebuild_the_previous_route(mode, fault):
    methods = callbacks('trajectory_tracker_node.py', ['yaw_of', '_on_plan'])
    methods['_on_plan'].__globals__['yaw_of'] = methods['yaw_of']
    plan = NS(header=NS(frame_id='map', stamp=NS(sec=10, nanosec=0)),
              poses=[message(x=0.), message(x=2.)])
    if fault in ('mixed_frame', 'empty_frame'):
        plan.poses[0].header.frame_id = 'odom' if fault == 'mixed_frame' else ''
    elif fault == 'zero_parent_stamp':
        plan.header.stamp.sec = 0
    elif fault == 'zero_pose_stamp':
        plan.poses[0].header.stamp.sec = 0
    elif fault == 'negative_nanoseconds':
        plan.poses[0].header.stamp.nanosec = -1
    elif fault == 'overflow_nanoseconds':
        plan.header.stamp.nanosec = 1_000_000_000
    elif fault == 'zero_rotation':
        plan.poses[0].pose.orientation.w = 0.
    elif fault == 'nonfinite_coordinate':
        plan.poses[0].pose.position.x = math.nan
    else:
        plan.poses.clear()
    node = NS(lock=threading.Lock(), motion_mode=mode, trajectory=object(),
              pending_plan=object(), last_plan=object(), stage_continuation=object(),
              stage_revision=2, plan_received_stamp=9.)
    methods['_on_plan'](node, plan)
    assert node.trajectory is None and node.pending_plan is None
    assert node.last_plan is None and node.stage_continuation is None
    assert node.stage_revision == 3 and node.plan_received_stamp == 9.


def test_fixed_route_waypoint_stamp_can_predate_the_path_publication():
    methods = callbacks('trajectory_tracker_node.py', ['yaw_of', '_on_plan'])
    methods['_on_plan'].__globals__['yaw_of'] = methods['yaw_of']
    plan = NS(header=NS(frame_id='map', stamp=NS(sec=10, nanosec=0)),
              poses=[message(x=0., stamp=1.), message(x=2., stamp=1.)])
    node = NS(lock=threading.Lock(), active_goal=None, plan_event=threading.Event())
    methods['_on_plan'](node, plan)
    assert node.pending_plan is not None and node.plan_event.is_set()
    assert node.plan_received_stamp == 10.


@pytest.mark.parametrize('parent,child', [('', 'base_link'), ('map', 'base_link'),
                                         ('odom', ''), ('odom', 'laser')])
def test_wrong_frame_wheel_feedback_never_renews_body_velocity(parent, child):
    method = callbacks('trajectory_tracker_node.py', ['_on_odom'])['_on_odom']
    node = NS(velocity=np.array([.1, .2, .3]), velocity_stamp=9.9,
              odom_frame='odom', base_frame='base_link')
    method(node, NS(header=NS(frame_id=parent), child_frame_id=child))
    assert node.velocity_stamp == 9.9
    assert node.velocity.tolist() == [.1, .2, .3]


@pytest.mark.parametrize('seconds,nanoseconds', [(0, 0), (10, -1), (9, 1_000_000_000)])
@pytest.mark.parametrize('consumer', ['bridge', 'tracker_pose', 'tracker_odom'])
def test_malformed_source_timestamp_never_refreshes_navigation_feedback(consumer, seconds, nanoseconds):
    msg = message()
    msg.header.stamp = NS(sec=seconds, nanosec=nanoseconds)
    if consumer == 'bridge':
        method = callbacks('goal_bridge_node.py', ['_pose_cb'])['_pose_cb']
        msg.pose = NS(pose=msg.pose)
        node = NS(configured_frame='map', current_position=(1., 0.),
                  current_pose_received_at=9.9)
        method(node, msg)
        assert node.current_position == (1., 0.)
        assert node.current_pose_received_at == 9.9
    elif consumer == 'tracker_pose':
        methods = callbacks('trajectory_tracker_node.py', ['yaw_of', '_on_pose'])
        methods['_on_pose'].__globals__['yaw_of'] = methods['yaw_of']
        msg.pose = NS(pose=msg.pose)
        node = NS(pose_stamp=9.9, pose=np.array([1., 0., 0.]))
        methods['_on_pose'](node, msg)
        assert node.pose_stamp == 9.9 and node.pose[0] == 1.
    else:
        method = callbacks('trajectory_tracker_node.py', ['_on_odom'])['_on_odom']
        msg.header.frame_id = 'odom'
        msg.child_frame_id = 'base_link'
        node = NS(velocity_stamp=9.9, velocity=np.array([.1, .2, .3]))
        method(node, msg)
        assert node.velocity_stamp == 9.9


@pytest.mark.parametrize('context_alive', [False, True])
def test_tracker_shutdown_handles_native_take_race_without_hiding_live_errors(context_alive):
    cleaned = []
    node = NS(stopping=False, plan_event=NS(set=lambda: cleaned.append('wake')),
              worker=NS(join=lambda **kw: cleaned.append('join')),
              destroy_node=lambda: cleaned.append('destroy'))
    def fail_spin(node):
        raise RuntimeError('native subscription invalidated')
    ros = NS(init=lambda **kw: None, spin=fail_spin, ok=lambda: context_alive,
             try_shutdown=lambda: cleaned.append('shutdown'))
    method = callbacks('trajectory_tracker_node.py', ['main'], rclpy=ros,
                       TrajectoryTracker=lambda: node,
                       ExternalShutdownException=type('ExternalShutdownException', (Exception,), {}))['main']
    if context_alive:
        with pytest.raises(RuntimeError, match='native subscription'):
            method()
    else:
        method()
    assert node.stopping and cleaned == ['wake', 'join', 'destroy', 'shutdown']


@pytest.mark.parametrize('resolution,radius,center,grid', [
    (math.nan, .3, (0., 0.), np.zeros((40, 40))),
    (.1, -.3, (0., 0.), np.zeros((40, 40))),
    (.1, math.inf, (0., 0.), np.zeros((40, 40))),
    (.1, .3, (math.nan, 0.), np.zeros((40, 40))),
    (.1, .3, (0.,), np.zeros((40, 40))),
    (.1, .3, (0., 0.), np.full((40, 40), math.nan)),
])
def test_malformed_rotation_maps_fail_closed(resolution, radius, center, grid):
    assert free_rotation_disk(grid, resolution, (-2., -2.), center, radius) is False


@pytest.mark.parametrize('spacing', [0., -1., math.nan, math.inf])
def test_sweep_densification_rejects_invalid_sampling_distance(spacing):
    with pytest.raises(ValueError, match='spacing'):
        dense_path([[0., 0.], [1., 0.]], spacing)


def test_bucket_connector_with_unknown_clearance_is_never_accepted():
    model = NS(radius=.6, clearance_over_poses=lambda points, *a, **k:
               np.full(len(points), math.nan))
    transit = FixedBucketTransit(model, [])
    assert transit._clear((0., 0.), (1., 0.), 0.) is False


@pytest.mark.parametrize('name', [None, True, 1, [], {}])
def test_saved_pose_names_require_text(name):
    with pytest.raises(ValueError):
        normalize_pose_name(name)


@pytest.mark.parametrize('operation', ['save', 'load'])
def test_normalized_duplicate_pose_names_are_rejected_without_overwriting(tmp_path, operation):
    path = tmp_path/'poses.json'
    pose = dict(frame_id='map', x=1., y=0., yaw=0.)
    save_remembered_poses(path, {'A': pose})
    before = path.read_bytes()
    poses = {'A': pose, ' A ': {**pose, 'x': 2.}}
    with pytest.raises(ValueError, match='duplicate'):
        if operation == 'save':
            save_remembered_poses(path, poses)
        else:
            other = tmp_path/'duplicates.json'
            other.write_text(json.dumps({'version': 1, 'poses': poses}))
            load_remembered_poses(other)
    assert path.read_bytes() == before


def test_remote_navigation_ignores_the_previous_fields_terminal_result():
    events = []
    nav = RemoteNavigation(['A'], lambda *event: events.append(event))
    nav.owned, nav.phase, nav.target, nav.side = True, 'NAVIGATING', 'A', 'right'
    nav.navigation_update({'remembered_pose': 'A', 'field_side': 'left', 'state': 'CANCELED'})
    assert nav.owned and not events
    nav.navigation_update({'remembered_pose': 'A', 'field_side': 'right', 'state': 'SUCCEEDED'})
    assert not nav.owned and ('status', 'SUCCEEDED') in events


def test_motor_arming_gets_its_full_timeout_after_a_slow_disarm():
    events = []
    nav = RemoteNavigation(['A'], lambda *event: events.append(event))
    nav.owned, nav.phase, nav.target, nav.started = True, 'DISARMING', 'A', 0.
    nav.received, nav.safety_received = 1.8, 1.8
    nav.input = dict(mu3_alive=True, link_alive=True, uart_open=True, auto_engaged=False)
    nav.safety = dict(tracking_effective=True, armed=False)
    nav.tick(1.8)
    assert nav.phase == 'ARMING' and nav.started == 1.8
    nav.received = nav.safety_received = 2.1
    nav.tick(2.1)
    assert nav.owned and ('status', 'ARM_TIMEOUT') not in events


def test_previous_attempt_arrival_cannot_complete_the_same_coordinates_again():
    method = callbacks('goal_bridge_node.py', ['_tracker_arrived'])['_tracker_arrived']
    node = NS(tracker_arrival={'goal': [2., 0., 0.], 'ready': True, 'goal_stamp': [9, 0]},
              tracker_arrival_stamp=10., active_request=NS(pose=message(), goal_stamp=[10, 0]))
    assert method(node, 10.) is False
    node.tracker_arrival['goal_stamp'] = [10, 0]
    assert method(node, 10.) is True


@pytest.mark.parametrize('state', ['SUCCEEDED', 'CANCELED', 'FAILED', 'FINAL_APPROACH',
                                  'REVERSE_APPROACH'])
def test_old_goal_status_cannot_disable_or_retarget_the_new_tracker_goal(state):
    method = callbacks('staged_heading_node.py', ['_stage_goal_status'])['_stage_goal_status']
    canceled = []
    node = NS(active_goal_stamp=[10, 0], stage_goal_enabled=True,
              _stage_cancel=lambda m: canceled.append(True))
    method(node, NS(data=json.dumps({'state': state, 'goal_stamp': [9, 0]})))
    assert not canceled and node.stage_goal_enabled
    assert not hasattr(node, 'reverse_goal') and not hasattr(node, 'final_approach_until')


def test_cancel_response_from_an_old_goal_cannot_clear_the_current_cancel_request():
    method = callbacks('goal_bridge_node.py', ['_cancel_response_cb'])['_cancel_response_cb']
    pending = object()
    node = NS(cancel_future=pending)
    method(node, NS(result=lambda: pytest.fail('stale response was inspected')))
    assert node.cancel_future is pending


def test_cancel_rejection_schedules_retry_and_never_claims_the_action_stopped():
    method = callbacks('goal_bridge_node.py', ['_cancel_response_cb'])['_cancel_response_cb']
    future = NS(result=lambda: NS(goals_canceling=[]))
    request, handle = object(), object()
    node = NS(cancel_future=future, cancel_accepted=False, active_request=request,
              active_goal_handle=handle, get_logger=logger)
    method(node, future)
    assert node.cancel_future is None and node.cancel_accepted is False
    assert node.cancel_retry_not_before == 10.25
    assert node.active_request is request and node.active_goal_handle is handle


def test_cancel_response_timeout_is_retried_without_sending_a_second_action():
    method = callbacks('goal_bridge_node.py', ['_try_send_pending'])['_try_send_pending']
    cancelled, retries = [], []
    future = NS(cancel=lambda: cancelled.append(True))
    request, handle = object(), object()
    node = NS(active_request=request, active_goal_handle=handle, result_future=object(),
              cancel_future=future, cancel_accepted=False, cancel_requested=False,
              cancel_requested_at=8., pending_request=object(), send_future=None,
              retry_not_before=0., get_logger=logger,
              get_parameter=lambda n: NS(value=1.),
              _request_cancel=lambda **k: retries.append(k),
              _navigation_ready=lambda *a: pytest.fail('another action was considered'))
    method(node)
    assert cancelled == [True] and retries == [{'explicit': False}]
    assert node.active_request is request and node.active_goal_handle is handle


def test_result_subscription_failure_stops_the_known_action_without_resending():
    method = callbacks('goal_bridge_node.py', ['_watch_goal_result'])['_watch_goal_result']
    canceled, statuses = [], []
    def broken_result():
        raise RuntimeError('result transport unavailable')
    handle = NS(get_result_async=broken_result)
    request = object()
    node = NS(active_request=request, active_goal_handle=handle,
              _request_cancel=lambda **k: canceled.append(k), get_logger=logger,
              _publish_status=lambda *a, **k: statuses.append((a, k)))
    assert method(node) is False
    assert node.active_request is request and node.active_goal_handle is handle
    assert node.result_monitor_cancel and node.result_retry_not_before == 10.25
    assert canceled == [{'explicit': False}]
    assert statuses[-1][0] == ('CANCELING',)


def test_late_result_cannot_clear_the_replacement_actions_state():
    method = callbacks('goal_bridge_node.py', ['_result_cb'])['_result_cb']
    request, handle = object(), object()
    node = NS(active_request=request, active_goal_handle=handle)
    method(node, NS(result=lambda: pytest.fail('late result was inspected')),
           request=object(), goal_handle=object())
    assert node.active_request is request and node.active_goal_handle is handle


def test_profile_change_invalidates_worker_results_and_prebuilt_departures_atomically():
    method = callbacks('trajectory_tracker_node.py', ['_on_safety_state'])['_on_safety_state']
    plan, notified = object(), []
    node = NS(lock=threading.Lock(), profiles={'precision': {}}, profile_name='balanced',
              stage_revision=3, stage_continuation=object(), last_plan=plan,
              plan_event=NS(set=lambda: notified.append(True)), get_logger=logger,
              speed_limit=.2, lateral_limit=.2, yaw_limit=.3)
    def apply_profile(name):
        assert node.lock.locked()
        node.profile_name = name
    node._apply_profile = apply_profile
    method(node, NS(data=json.dumps({'profile': 'precision'})))
    assert node.stage_revision == 4 and node.stage_continuation is None
    assert node.pending_plan is plan and notified == [True]


@pytest.mark.parametrize('failure', ['disappeared', 'request_exception', 'timeout'])
def test_lifecycle_failure_invalidates_cached_readiness_and_recovers_request_slots(failure):
    methods = callbacks('goal_bridge_node.py', ['_poll_lifecycle_states', '_abandon_lifecycle_request'],
                        GetState=NS(Request=lambda: object()))
    canceled, requests = [], []
    pending = NS(done=lambda: False, cancel=lambda: canceled.append(True))
    def call_async(request):
        requests.append(request)
        if failure == 'request_exception':
            raise RuntimeError('service vanished after discovery')
        return pending
    client = NS(service_is_ready=lambda: failure != 'disappeared', call_async=call_async)
    node = NS(lifecycle_clients={'controller_server': client},
              lifecycle_futures={'controller_server': pending if failure == 'timeout' else None},
              lifecycle_states={'controller_server': 3},
              lifecycle_state_times={'controller_server': 10.},
              lifecycle_request_times={'controller_server': 8.},
              lifecycle_timeouts={'controller_server': 0}, last_lifecycle_poll=-math.inf,
              get_logger=logger, get_parameter=lambda n: NS(value=1.))
    node._abandon_lifecycle_request = MethodType(methods['_abandon_lifecycle_request'], node)
    methods['_poll_lifecycle_states'](node, 10.)
    assert node.lifecycle_states['controller_server'] is None
    assert node.lifecycle_state_times['controller_server'] == -math.inf
    if failure == 'timeout':
        assert canceled == [True] and len(requests) == 1
        assert node.lifecycle_request_times['controller_server'] == 10.
    else:
        assert node.lifecycle_futures['controller_server'] is None


def test_remote_same_pose_retry_ignores_an_old_commands_terminal_status():
    events = []
    nav = RemoteNavigation(['A'], lambda *event: events.append(event))
    nav.owned, nav.phase, nav.target, nav.side = True, 'NAVIGATING', 'A', 'left'
    nav.request_id = 'new-command'
    for request_id in ['old-command', None]:
        nav.navigation_update(dict(remembered_pose='A', field_side='left',
                                   state='CANCELED', request_id=request_id))
    assert nav.owned and not nav.saw_navigation and not events
    nav.navigation_update(dict(remembered_pose='A', field_side='left',
                               state='SUCCEEDED', request_id='new-command'))
    assert not nav.owned and events[-1] == ('status', 'SUCCEEDED')


@pytest.mark.parametrize('request_id', ['', 1, [], 'x' * 129])
def test_remote_request_identifier_is_validated_before_any_field_change(request_id):
    method = callbacks('goal_bridge_node.py', ['_parse_remembered_request'])['_parse_remembered_request']
    with pytest.raises(ValueError, match='request_id'):
        method(json.dumps(dict(name='A', side='right', request_id=request_id)))


@pytest.mark.parametrize('route', [False, True])
def test_goal_bridge_does_not_poll_lifecycle_services_during_nav2_configuration(route):
    method = callbacks('goal_bridge_node.py', ['_navigation_ready'])['_navigation_ready']
    polled = []
    ready = [False]
    client = NS(server_is_ready=lambda: ready[0])
    node = NS(action_client=client, route_action_client=client,
              lifecycle_clients={}, pending_request=NS(route_poses=[object()] if route else []),
              _poll_lifecycle_states=polled.append)
    assert method(node, 10.) is False
    assert polled == []
    assert 'action server' in node.not_ready_reason
    ready[0] = True
    assert method(node, 10.) is True
    assert polled == [10.]


@pytest.mark.parametrize('status', [
    dict(state='PREEMPTING', goal_stamp=None, cancel_goal_stamp=[9, 0]),
    dict(state='PREEMPTING', goal_stamp=[9, 0]),
    dict(state='CANCELED', goal_stamp=None),
])
def test_delayed_preemption_or_unstamped_cancel_cannot_stop_a_replacement(status):
    method = callbacks('staged_heading_node.py', ['_stage_goal_status'])['_stage_goal_status']
    canceled = []
    node = NS(active_goal_stamp=[10, 0], stage_goal_enabled=True,
              _stage_cancel=lambda m: canceled.append(True))
    method(node, NS(data=json.dumps(status)))
    assert not canceled and node.stage_goal_enabled


def test_cancel_during_final_approach_preserves_the_canceled_requests_stamp():
    method = callbacks('goal_bridge_node.py', ['_request_cancel'])['_request_cancel']
    statuses = []
    request = NS(goal_stamp=[9, 0])
    node = NS(finalizing_since=1., active_request=request, pending_request=None,
              active_goal_handle=None, send_future=None,
              _publish_status=lambda *a, **k: statuses.append((a, k)))
    method(node, explicit=True)
    assert node.active_request is None and node.finalizing_since is None
    assert statuses[-1] == (('CANCELED',), {'request': request})


def test_matching_preemption_still_stops_the_old_active_goal():
    method = callbacks('staged_heading_node.py', ['_stage_goal_status'],
                       Bool=lambda **k: NS(**k))['_stage_goal_status']
    canceled = []
    node = NS(active_goal_stamp=[9, 0], _stage_cancel=lambda m: canceled.append(m.data))
    method(node, NS(data=json.dumps(dict(state='PREEMPTING', goal_stamp=None,
                                        cancel_goal_stamp=[9, 0]))))
    assert canceled == [True]
