"""Nav2 cell-end success must not stop short of the actual requested pose."""
import json
import math
from types import MethodType, SimpleNamespace
import numpy as np
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from action_msgs.msg import GoalStatus
from omni_autonomy_next.goal_bridge_node import GoalBridgeNode, GoalRequest
from omni_autonomy_next import trajectory_tracker_node as tracker
from test_staged_heading import staged_node


def bridge():
    goal=PoseStamped()
    goal.pose.position.x=2.
    goal.pose.orientation.w=1.
    node=SimpleNamespace(active_request=SimpleNamespace(pose=goal,label='test'),
        pending_request=None,active_goal_handle=None,cancel_future=None,
        cancel_requested=False,send_future=None,finalizing_since=None,
        verify_tracker_arrival=True,tracker_arrival=None,tracker_arrival_stamp=-math.inf,
        statuses=[],get_logger=lambda:SimpleNamespace(info=lambda *a:None))
    node._publish_status=lambda state,**kw:node.statuses.append(state)
    for name in ('_tracker_status_cb','_tracker_arrived','_finish_tracker_arrival',
                 '_request_cancel','_result_cb'):
        setattr(node,name,MethodType(getattr(GoalBridgeNode,name),node))
    return node


def test_nav2_success_waits_for_fresh_measured_arrival_at_the_same_goal(monkeypatch):
    monkeypatch.setattr(tracker.time,'monotonic',lambda:1.)
    node=bridge()
    node._result_cb(SimpleNamespace(result=lambda:SimpleNamespace(status=GoalStatus.STATUS_SUCCEEDED)))
    assert node.statuses==['FINAL_APPROACH']
    assert node.active_request is not None
    node._tracker_status_cb(String(data=json.dumps({'arrival':{'goal':[1.,0.,0.],'ready':True}})))
    node._finish_tracker_arrival(1.)
    assert node.active_request is not None
    node._tracker_status_cb(String(data=json.dumps({'arrival':{'goal':[2.,0.,0.],'ready':True}})))
    assert not node._tracker_arrived(2.)
    node._finish_tracker_arrival(1.1)
    assert node.statuses[-1]=='SUCCEEDED'
    assert node.active_request is None


def test_final_positioning_is_bounded_and_cancelable():
    node=bridge()
    node.finalizing_since=0.
    node._finish_tracker_arrival(5.1)
    assert node.statuses[-1]=='FAILED'
    assert node.active_request is None
    node=bridge()
    node.finalizing_since=0.
    node._request_cancel(explicit=True)
    assert node.finalizing_since is None and node.active_request is None
    assert node.statuses[-1]=='CANCELED'
    node=bridge()
    node.finalizing_since=0.
    replacement=object()
    node.pending_request=replacement
    node._request_cancel(explicit=False)
    assert node.active_request is None and node.pending_request is replacement


def test_tracker_can_finish_short_cell_endpoint_without_an_unbounded_stale_plan(monkeypatch):
    node=staged_node(goal=(.04,0.,0.))
    monkeypatch.setattr(tracker.time,'monotonic',lambda:0.)
    tracker.TrajectoryTracker._build_trajectory(node,np.array([[0.,0.],[.04,0.]]),0.)
    monkeypatch.setattr(tracker.time,'monotonic',lambda:4.5)
    node.pose_stamp=node.velocity_stamp=4.5
    node._stage_goal_status(String(data='{"state":"FINAL_APPROACH"}'))
    tracker.TrajectoryTracker._tick(node)
    assert node.command[0]>0.
    monkeypatch.setattr(tracker.time,'monotonic',lambda:10.)
    node.pose_stamp=node.velocity_stamp=10.
    tracker.TrajectoryTracker._tick(node)
    assert node.statuses[-1]=='PLAN_STALE'
    assert np.linalg.norm(node.command)==0.
