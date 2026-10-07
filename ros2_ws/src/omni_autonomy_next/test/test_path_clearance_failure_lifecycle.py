"""A rejected footprint trajectory must stop and terminate its exact action."""
import json
from types import MethodType, SimpleNamespace as NS

import numpy as np
import pytest
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String

from omni_autonomy_next.goal_bridge_node import GoalBridgeNode, GoalRequest
from omni_autonomy_next import trajectory_tracker_node as tracker


def bridge():
    pose = PoseStamped()
    pose.pose.position.x = 2.
    pose.pose.orientation.w = 1.
    request = GoalRequest(pose, 'test', goal_stamp=[10, 2], request_id='this-request')
    calls = []
    future = NS(add_done_callback=lambda callback: None)
    handle = NS(cancel_goal_async=lambda: calls.append('cancel') or future)
    node = NS(active_request=request, pending_request=None, active_goal_handle=handle,
              cancel_future=None, send_future=None, cancel_requested=False,
              cancel_accepted=False, finalizing_since=None, statuses=[],
              tracker_arrival=None, tracker_arrival_stamp=0., verify_tracker_arrival=True,
              _try_send_pending=lambda: None,
              get_logger=lambda: NS(info=lambda *a: None, warning=lambda *a: None,
                                    error=lambda *a: None))
    node._publish_status = lambda state, **details: node.statuses.append((state, details))
    for name in ('_tracker_status_cb', '_tracker_arrived', '_request_cancel',
                 '_cancel_response_cb', '_result_cb', '_retry_or_finish', '_feedback_cb'):
        setattr(node, name, MethodType(getattr(GoalBridgeNode, name), node))
    return node, request, handle, calls


def blocked(stamp=(10, 2), target=(2., 0., 0.), state='PATH_CLEARANCE_BLOCKED'):
    return String(data=json.dumps(dict(
        state=state, reason='POSE_PATH_CLEARANCE:0.004<0.025',
        arrival=dict(goal=list(target), goal_stamp=list(stamp), ready=True))))


@pytest.mark.parametrize('stamp,target', [((9, 2), (2., 0., 0.)),
                                          ((10, 2), (3., 0., 0.))])
def test_delayed_or_wrong_target_failure_cannot_cancel_current_action(stamp, target):
    node, request, handle, calls = bridge()
    node._tracker_status_cb(blocked(stamp, target))
    assert not calls and not node.statuses
    assert node.active_request is request and node.active_goal_handle is handle
    assert request.failure_reason is None
    assert node.tracker_arrival['ready'] is False


@pytest.mark.parametrize('terminal', [GoalStatus.STATUS_SUCCEEDED,
                                     GoalStatus.STATUS_CANCELED,
                                     GoalStatus.STATUS_ABORTED])
@pytest.mark.parametrize('state', ['PATH_CLEARANCE_BLOCKED', 'EXECUTION_CLEARANCE_BLOCKED'])
def test_rejected_path_cancels_once_retains_handle_then_finishes_failed(terminal, state):
    node, request, handle, calls = bridge()
    node._tracker_status_cb(blocked(state=state))
    assert calls == ['cancel']
    assert node.active_request is request and node.active_goal_handle is handle
    assert node.cancel_requested
    assert node.statuses[-1][0] == 'FAILED'
    assert node.statuses[-1][1]['request'] is request
    assert node.statuses[-1][1]['reason'].startswith(state+':')
    assert not node._tracker_arrived(node.tracker_arrival_stamp)
    node._tracker_status_cb(blocked(state=state))
    assert calls == ['cancel']
    count = len(node.statuses)
    node._feedback_cb(NS(feedback=NS(distance_remaining=0.)), request=request)
    assert len(node.statuses) == count
    node._result_cb(NS(result=lambda: NS(status=terminal)), request=request, goal_handle=handle)
    assert node.active_request is None and node.active_goal_handle is None
    assert node.statuses[-1][0] == 'FAILED'
    assert all(state != 'SUCCEEDED' for state, _ in node.statuses)
    assert not node.cancel_requested


def test_blocked_final_positioning_cannot_be_mistaken_for_arrival():
    node, request, _, calls = bridge()
    node.active_goal_handle = None
    node.finalizing_since = 1.
    node._tracker_status_cb(blocked())
    assert node.active_request is None and node.finalizing_since is None
    assert not calls
    assert node.statuses[-1][0] == 'FAILED'
    assert all(state != 'SUCCEEDED' for state, _ in node.statuses)


def test_rejected_path_is_not_retried_after_pending_send_fails():
    node, request, _, calls = bridge()
    node.active_goal_handle = None
    node.send_future = object()
    node._tracker_status_cb(blocked())
    assert node.active_request is request and node.cancel_requested
    assert node.statuses[-1][0] == 'FAILED'
    node._retry_or_finish('REJECTED')
    assert node.active_request is None and node.pending_request is None
    assert node.statuses[-1][0] == 'FAILED'
    assert not calls


def test_delayed_block_for_old_action_preserves_queued_replacement():
    node, old, handle, calls = bridge()
    new = GoalRequest(old.pose, 'replacement', goal_stamp=[11, 2], request_id='new-request')
    sent = []
    node.pending_request = new
    node._try_send_pending = lambda: sent.append(node.pending_request)
    node._tracker_status_cb(blocked())
    assert node.pending_request is new and node.active_request is old
    assert node.active_goal_handle is handle and calls == ['cancel']
    assert not sent and new.failure_reason is None
    assert all(details['request'] is old for state, details in node.statuses if state == 'FAILED')
    node._result_cb(NS(result=lambda: NS(status=GoalStatus.STATUS_CANCELED)),
                    request=old, goal_handle=handle)
    assert sent == [new]
    assert node.statuses[-1][0] == 'FAILED' and node.statuses[-1][1]['request'] is old
    assert new.failure_reason is None


@pytest.mark.parametrize('failure', ['result_future', 'result_subscription'])
def test_failed_path_stays_failed_through_result_transport_error(failure):
    node, old, handle, _ = bridge()
    node._tracker_status_cb(blocked())
    new = GoalRequest(old.pose, 'replacement', request_id='new-request')
    node.pending_request = new
    sent = []
    node._try_send_pending = lambda: sent.append(node.pending_request)
    def broken_result():
        raise RuntimeError('broken result transport')
    if failure == 'result_future':
        node._result_cb(NS(result=broken_result), request=old, goal_handle=handle)
    else:
        handle.get_result_async = broken_result
        GoalBridgeNode._watch_goal_result(node)
    assert node.active_request is old and node.active_goal_handle is handle
    assert node.pending_request is new and not sent
    assert node.statuses[-1][0] == 'FAILED'
    assert node.statuses[-1][1]['reason'] == old.failure_reason
    node._result_cb(NS(result=lambda: NS(status=GoalStatus.STATUS_CANCELED)),
                    request=old, goal_handle=handle)
    assert sent == [new]


def test_selection_after_block_survives_cancel_response_timeout(monkeypatch):
    monkeypatch.setattr(tracker.time, 'monotonic', lambda: 10.)
    node, old, handle, calls = bridge()
    node._tracker_status_cb(blocked())
    assert node.cancel_requested
    new = GoalRequest(old.pose, 'replacement', request_id='new-request')
    node._queue_request = MethodType(GoalBridgeNode._queue_request, node)
    assert node._queue_request(new)
    assert node.pending_request is new and not node.cancel_requested
    node.cancel_future.cancel = lambda: calls.append('expired-response-canceled')
    node.cancel_requested_at = 0.
    node.result_future = object()
    node.get_parameter = lambda name: NS(value=1.)
    GoalBridgeNode._try_send_pending(node)
    assert calls == ['cancel', 'expired-response-canceled', 'cancel']
    assert node.active_request is old and node.active_goal_handle is handle
    assert node.pending_request is new and new.failure_reason is None
    sent = []
    node._try_send_pending = lambda: sent.append(node.pending_request)
    node._result_cb(NS(result=lambda: NS(status=GoalStatus.STATUS_CANCELED)),
                    request=old, goal_handle=handle)
    assert sent == [new]


@pytest.mark.parametrize('data', ['[]', 'null', '{"state":"PATH_CLEARANCE_BLOCKED"}'])
def test_malformed_blocked_status_cannot_drop_the_current_action(data):
    node, request, handle, calls = bridge()
    node._tracker_status_cb(String(data=data))
    assert node.active_request is request and node.active_goal_handle is handle
    assert not calls


def test_blocked_tracker_status_never_reports_ready_at_stopped_goal(monkeypatch):
    monkeypatch.setattr(tracker.time, 'monotonic', lambda: 1.)
    messages = []
    node = NS(last_status=None, pose=np.array([2., 0., 0.]),
              active_goal=np.array([2., 0., 0.]), active_goal_stamp=[10, 2],
              pose_stamp=1., velocity_stamp=1., velocity=np.zeros(3),
              status_pub=NS(publish=messages.append),
              get_logger=lambda: NS(info=lambda *a: None))
    tracker.TrajectoryTracker._status(node, 'PATH_CLEARANCE_BLOCKED', reason='blocked')
    assert json.loads(messages[-1].data)['arrival']['ready'] is False
