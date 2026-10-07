import math
import json
from types import MethodType, SimpleNamespace
import numpy as np
import pytest
from std_msgs.msg import String, Bool
from geometry_msgs.msg import PoseStamped
from action_msgs.msg import GoalStatus
from omni_autonomy_next.reverse_approach import reverse_gate, reverse_command, reverse_target, SPEED, POSITION_TOLERANCE, YAW_TOLERANCE
from omni_autonomy_next.field_side import apply_pose
from omni_autonomy_next.goal_bridge_node import GoalBridgeNode
from omni_autonomy_next import trajectory_tracker_node as tracker
from test_staged_heading import staged_node
from test_precise_goal_result import bridge


@pytest.mark.parametrize('yaw', [0., math.pi/2, math.pi, -math.pi/2, 3.1135965885])
def test_gate_and_reverse_motion_in_saved_direction(yaw):
    goal = np.array([-1.75, 4.53, yaw])
    gate = reverse_gate(dict(x=goal[0], y=goal[1], yaw=yaw, frame_id='map'))
    pose = np.array([gate['x'], gate['y'], yaw])
    assert np.linalg.norm(pose[:2]-goal[:2]) == pytest.approx(.25)
    measured = np.zeros(3)
    for _ in range(400):
        command, state = reverse_command(pose, goal, measured, 1., .2)
        assert state == 'REVERSING'
        assert -SPEED <= command[0] <= 0. and command[1] == 0.
        measured += .2*(np.array(command)-measured)
        pose[:2] += .05*measured[0]*np.array([math.cos(yaw), math.sin(yaw)])
    assert np.linalg.norm(pose[:2]-goal[:2]) <= POSITION_TOLERANCE


@pytest.mark.parametrize('pose', [[.25,.03,0.], [.25,0.,.1], [-.03,0.,0.], [.4,0.,0.]])
def test_reverse_rejects_deviation(pose):
    command, state = reverse_command(np.array(pose), np.zeros(3), np.zeros(3), 1., .2)
    assert command == (0.,0.,0.) and state == 'REVERSE_DEVIATION'


@pytest.mark.parametrize('yaw', [0., math.pi-.005, -math.pi+.005])
@pytest.mark.parametrize('delay,gain', [(.12, .7), (.3, 1.3)])
@pytest.mark.parametrize('along', [.25, -.006])
def test_reverse_corrects_lateral_and_small_overshoot_with_actuator_delay(yaw, delay, gain, along):
    from collections import deque
    goal = np.array([0., 0., yaw])
    c, s = math.cos(yaw), math.sin(yaw)
    pose = np.array([c*along-s*.012, s*along+c*.012, yaw+.015])
    measured = np.zeros(3)
    history = deque([np.zeros(3) for _ in range(round(delay/.05))])
    for _ in range(360):
        command, state = reverse_command(pose, goal, measured, 1., delay)
        assert state == 'REVERSING'
        assert np.linalg.norm(command[:2]) <= math.hypot(SPEED, .01) + 1.e-12
        history.append(np.array(command))
        measured += .25*(gain*history.popleft()-measured)
        c, s = math.cos(pose[2]), math.sin(pose[2])
        pose[:2] += .05*np.array([c*measured[0]-s*measured[1], s*measured[0]+c*measured[1]])
        pose[2] += .05*measured[2]
    assert np.linalg.norm(pose[:2]-goal[:2]) <= POSITION_TOLERANCE
    assert abs((pose[2]-goal[2]+math.pi)%(2*math.pi)-math.pi) <= YAW_TOLERANCE
    assert np.linalg.norm(measured) < .005


def test_gate_must_settle_before_reverse_and_cancel_preempts(monkeypatch):
    monkeypatch.setattr(tracker.time, 'monotonic', lambda: 1.)
    node = bridge()
    for name in ('_begin_reverse', '_publish_reverse'):
        setattr(node, name, MethodType(getattr(GoalBridgeNode, name), node))
    final = PoseStamped()
    final.pose.position.x = 1.75
    final.pose.orientation.w = 1.
    node.active_request.reverse_final_pose = final
    node.active_request.reversing = False
    node._result_cb(SimpleNamespace(result=lambda: SimpleNamespace(status=GoalStatus.STATUS_SUCCEEDED)))
    assert node.statuses == ['FINAL_APPROACH']
    node._tracker_status_cb(String(data=json.dumps({'arrival': {'goal':[2.,0.,0.], 'ready':True}})))
    # Arrival callback switches immediately, without a bridge timer tick.
    assert node.statuses[-1] == 'REVERSE_APPROACH'
    assert node.active_request.pose is final and node.active_request.reversing
    node._finish_tracker_arrival(7.)
    assert node.active_request is not None  # reverse has a longer bounded deadline
    node._request_cancel(explicit=True)
    assert node.active_request is None and node.statuses[-1] == 'CANCELED'


def test_reverse_tracker_heartbeat_stale_pose_cancel_and_new_goal(monkeypatch):
    now = [0.]
    monkeypatch.setattr(tracker.time, 'monotonic', lambda: now[0])
    node = staged_node(pose=(.25,0.,0.), goal=(.25,0.,0.))
    message = String(data=json.dumps({'state':'REVERSE_APPROACH', 'reverse_goal':[0.,0.,0.]}))
    node._stage_goal_status(message)
    tracker.TrajectoryTracker._tick(node)
    assert -SPEED <= node.command[0] < 0. and node.command[1] == 0.
    now[0] = .7
    node.pose_stamp = node.velocity_stamp = .7
    tracker.TrajectoryTracker._tick(node)
    assert np.linalg.norm(node.command) == 0.
    node._stage_goal_status(message)
    assert node.reverse_started == 0.
    node.pose_stamp = 0.
    tracker.TrajectoryTracker._tick(node)
    assert np.linalg.norm(node.command) == 0.
    node._stage_cancel(Bool(data=True))
    assert node.reverse_goal is None and not node.stage_goal_enabled
    node._stage_new_goal()
    assert node.reverse_goal is None


def test_reverse_stops_for_obstacle(monkeypatch):
    from omni_autonomy_next.rl_residual import CadClearanceModel
    monkeypatch.setattr(tracker.time, 'monotonic', lambda: 0.)
    node = staged_node(pose=(.25,0.,0.), goal=(.25,0.,0.))
    node._stage_goal_status(String(data='{"state":"REVERSE_APPROACH","reverse_goal":[0,0,0]}'))
    node.clearance = CadClearanceModel([[[-.4,-2.],[-.4,2.]]], node.clearance.footprint)
    tracker.TrajectoryTracker._tick(node)
    assert np.linalg.norm(node.command) == 0.
    assert node.statuses[-1] == 'REVERSE_CLEARANCE_BLOCKED'


@pytest.mark.parametrize('goal', [
    [-1.7159082740346883, 4.554163889863259, -3.128787463628977],
    [1.7233270991489245, 4.529534090573496, -.008818611566363044],
])
def test_authorized_loading_reaches_exact_saved_pose_despite_cad(monkeypatch, goal):
    from pathlib import Path
    from omni_autonomy_next.rl_residual import CadClearanceModel
    goal = np.array(goal)
    gate = reverse_gate(dict(x=goal[0], y=goal[1], yaw=goal[2]))
    node = staged_node(pose=(gate['x'], gate['y'], goal[2]), goal=goal)
    config = Path(tracker.__file__).resolve().parents[1]/'config'
    node.clearance = CadClearanceModel.from_yaml(config/'field_planning.yaml', config/'competition_footprints.yaml')
    assert node.clearance.body_clearance(goal[:2], goal[2]) == 0.
    now = [0.]
    monkeypatch.setattr(tracker.time, 'monotonic', lambda: now[0])
    message = String(data=json.dumps(dict(state='REVERSE_APPROACH', remembered_pose='A',
                                         ignore_cad=True, reverse_goal=goal.tolist())))
    for i in range(300):
        now[0] = i*.05
        node.pose_stamp = node.velocity_stamp = now[0]
        node._stage_goal_status(message)
        tracker.TrajectoryTracker._tick(node)
        assert node.statuses[-1] == 'REVERSING'
        node.velocity += .2*(node.command-node.velocity)
        c, s = math.cos(node.pose[2]), math.sin(node.pose[2])
        node.pose[:2] += .05*np.array([c*node.velocity[0]-s*node.velocity[1],
                                      s*node.velocity[0]+c*node.velocity[1]])
        node.pose[2] += .05*node.velocity[2]
    assert np.linalg.norm(node.pose[:2]-goal[:2]) <= POSITION_TOLERANCE
    now[0] += .7
    node.pose_stamp = node.velocity_stamp = now[0]
    tracker.TrajectoryTracker._tick(node)
    assert node.statuses[-1] == 'REVERSE_FEEDBACK_STALE'
    assert np.linalg.norm(node.command) == 0.
    node._stage_cancel(Bool(data=True))
    assert not node.reverse_ignore_cad
    node._stage_new_goal()
    assert not node.reverse_ignore_cad


def test_rl_cad_override_requires_fresh_docking_and_localization(monkeypatch):
    from omni_autonomy_next import rl_policy_node as rl_module
    from omni_autonomy_next.rl_policy_node import RLPolicyNode
    from geometry_msgs.msg import Twist
    now = [0.]
    monkeypatch.setattr(tracker.time, 'monotonic', lambda: now[0])
    outputs = []
    node = SimpleNamespace(command_count=0, position=np.zeros(2), yaw=0.,
                           policy=SimpleNamespace(reset=lambda: None),
                           output_pub=SimpleNamespace(publish=outputs.append),
                           _zero_message=RLPolicyNode._zero_message,
                           _ready=lambda now: (True, 'ACTIVE'))
    message = Twist()
    message.linear.x = -.04
    message.angular.z = .004
    RLPolicyNode._tracker_status_cb(node, String(data='{"state":"REVERSING","cad_override":true}'))
    RLPolicyNode._command_cb(node, message)
    assert outputs[-1] is message
    assert node.last_reason == 'DOCKING_CAD_OVERRIDE'
    node._ready = lambda now: (False, 'STALE_LOCALIZATION')
    RLPolicyNode._command_cb(node, message)
    assert outputs[-1].linear.x == 0.
    assert not node.current_healthy
    node._ready = lambda now: (True, 'ACTIVE')
    node.get_parameter = lambda name: SimpleNamespace(value=.1)
    node.path = np.zeros((2, 2))
    def cad_check(*args):
        raise ValueError('CAD_CHECK_RESTORED')
    node.field = SimpleNamespace(body_clearance_and_gradient=cad_check)
    monkeypatch.setattr(rl_module, 'select_path_target', lambda *args: (np.zeros(2), np.zeros(2)))
    now[0] = .3
    RLPolicyNode._command_cb(node, message)
    assert node.last_reason == 'POLICY_ERROR:CAD_CHECK_RESTORED'
    assert outputs[-1].linear.x == 0.
    RLPolicyNode._tracker_status_cb(node, String(data='{"state":"REVERSING","cad_override":true}'))
    message.linear.x = -.2
    RLPolicyNode._command_cb(node, message)
    assert node.last_reason == 'POLICY_ERROR:CAD_CHECK_RESTORED'
    RLPolicyNode._tracker_status_cb(node, String(data='{"state":"IDLE"}'))
    assert node.cad_override_stamp == -math.inf


@pytest.mark.parametrize('side', ['left', 'right'])
def test_saved_a_real_cad_corridor_reaches_goal_with_straight_slow_commands(monkeypatch, side):
    from pathlib import Path
    from omni_autonomy_next.rl_residual import CadClearanceModel
    saved = dict(x=-1.7575643405850359, y=4.534936453947715,
                 yaw=3.113596588546968, frame_id='map')
    original = dict(saved)
    config = Path(tracker.__file__).resolve().parents[1]/'config'
    clearance = CadClearanceModel.from_yaml(config/'field_planning.yaml', config/'competition_footprints.yaml')
    if side == 'right':
        with pytest.raises(ValueError, match='REVERSE_CLEARANCE_BLOCKED'):
            reverse_target(saved, side, clearance)
        assert saved == original
        return
    target, offset = reverse_target(saved, side, clearance)
    assert saved == original
    mirrored = apply_pose(saved, side)
    assert offset == 0.
    assert target == saved
    assert target['yaw'] == mirrored['yaw']
    goal = np.array([target['x'], target['y'], target['yaw']])
    gate = reverse_gate(dict(x=goal[0], y=goal[1], yaw=goal[2]))
    node = staged_node(pose=(gate['x'], gate['y'], goal[2]), goal=goal)
    node.clearance = clearance
    now = [0.]
    monkeypatch.setattr(tracker.time, 'monotonic', lambda: now[0])
    message = String(data=json.dumps({'state':'REVERSE_APPROACH', 'reverse_goal':goal.tolist()}))
    node._stage_goal_status(message)
    for i in range(300):
        now[0] = i*.05
        node.pose_stamp = node.velocity_stamp = now[0]
        node._stage_goal_status(message)
        tracker.TrajectoryTracker._tick(node)
        assert node.statuses[-1] == 'REVERSING'
        assert -SPEED <= node.command[0] <= 0. and node.command[1] == 0.
        node.velocity += .2*(node.command-node.velocity)
        c,s = math.cos(node.pose[2]),math.sin(node.pose[2])
        node.pose[:2] += .05*np.array([c*node.velocity[0],s*node.velocity[0]])
        node.pose[2] += .05*node.velocity[2]
    assert np.linalg.norm(node.pose[:2]-goal[:2]) <= POSITION_TOLERANCE


def test_reverse_target_rejects_obstacles_beyond_bounded_mirror_correction():
    from omni_autonomy_next.rl_residual import CadClearanceModel
    model = CadClearanceModel([[[1., -2.], [1., 2.]]],
                             [[-.45,-.42],[.45,-.42],[.45,.42],[-.45,.42]])
    saved = dict(x=-1., y=0., yaw=math.pi)
    with pytest.raises(ValueError, match='REVERSE_CLEARANCE_BLOCKED'):
        reverse_target(saved, 'right', model)


@pytest.mark.parametrize('side', ['left', 'right'])
@pytest.mark.parametrize('calibrated', [False, True])
def test_saved_request_uses_exact_final_pose_and_gate(side, calibrated):
    from pathlib import Path
    from omni_autonomy_next.rl_residual import CadClearanceModel
    from builtin_interfaces.msg import Time
    config = Path(tracker.__file__).resolve().parents[1]/'config'
    saved = dict(x=-1.7575643405850359, y=4.534936453947715,
                 yaw=3.113596588546968, frame_id='map')
    if calibrated:
        measured = apply_pose(saved, side)
        measured['x'] += .05 * math.cos(measured['yaw'])
        measured['y'] += .05 * math.sin(measured['yaw'])
        saved['field_poses'] = {side: measured}
    requests = []
    node = SimpleNamespace(
        field_side=side, configured_frame='map',
        reverse_clearance=CadClearanceModel.from_yaml(
            config/'field_planning.yaml', config/'competition_footprints.yaml'),
        effective_remembered_poses=lambda: {'A': saved},
        _parse_remembered_request=GoalBridgeNode._parse_remembered_request,
        _queue_request=requests.append,
        get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(to_msg=Time)),
        get_logger=lambda: SimpleNamespace(info=lambda *args: None))
    GoalBridgeNode._remembered_goal_cb(node, String(data='A'))
    assert len(requests) == 1
    target = measured if calibrated else apply_pose(saved, side)
    gate = reverse_gate(target)
    request = requests[0]
    assert request.reverse_final_pose.pose.position.x == pytest.approx(target['x'])
    assert request.reverse_final_pose.pose.position.y == pytest.approx(target['y'])
    assert request.pose.pose.position.x == pytest.approx(gate['x'])
    assert request.pose.pose.position.y == pytest.approx(gate['y'])
    if calibrated and side == 'right':
        # CAD-overlapping saved targets remain exact and can be queued.
        saved['field_poses']['right'] = apply_pose(
            {key: saved[key] for key in ('frame_id', 'x', 'y', 'yaw')}, side)
        requests.clear()
        statuses = []
        node._publish_status = lambda state, **kw: statuses.append((state, kw))
        GoalBridgeNode._remembered_goal_cb(node, String(data='A'))
        assert len(requests) == 1
        assert not statuses
        assert requests[0].reverse_final_pose.pose.position.x == saved['field_poses']['right']['x']


@pytest.mark.parametrize('pose', [
    [-1.7159082740346883, 4.554163889863259, -3.128787463628977],
    [1.7233270991489245, 4.529534090573496, -.008818611566363044],
])
def test_current_recordings_are_not_silently_moved_away_from_cad_wall(pose):
    from pathlib import Path
    from omni_autonomy_next.rl_residual import CadClearanceModel
    config = Path(tracker.__file__).resolve().parents[1]/'config'
    clearance = CadClearanceModel.from_yaml(config/'field_planning.yaml', config/'competition_footprints.yaml')
    saved = dict(frame_id='map', x=pose[0], y=pose[1], yaw=pose[2])
    before = dict(saved)
    with pytest.raises(ValueError, match='REVERSE_CLEARANCE_BLOCKED'):
        reverse_target(saved, 'left', clearance)
    assert saved == before


@pytest.mark.parametrize('distance, state, command, ready', [
    (.014, 'REVERSING', -.008, False),
    (.007, 'REVERSING', -.003, False),
    (.007, 'REVERSE_CLEARANCE_BLOCKED', 0., False),
    (.007, 'REVERSING', 0., True),
    (.002, 'REVERSING', 0., True),
])
def test_reverse_arrival_requires_saved_target_and_completed_stop(monkeypatch, distance, state, command, ready):
    node = staged_node(pose=(distance, 0., 0.), goal=(0., 0., 0.))
    now = [0.]
    monkeypatch.setattr(tracker.time, 'monotonic', lambda: now[0])
    node.reverse_goal = np.zeros(3)
    node.command = np.array([command, 0., 0.])
    node.last_status = state
    messages = []
    node.status_pub = SimpleNamespace(publish=messages.append)
    for i in range(5):
        now[0] = i * .1
        node.pose_stamp = node.velocity_stamp = now[0]
        tracker.TrajectoryTracker._status(node, state)
        if i < 3:
            assert not json.loads(messages[-1].data)['arrival']['ready']
    assert json.loads(messages[-1].data)['arrival']['ready'] is ready
    now[0] += .4
    node.pose_stamp = node.velocity_stamp = now[0]
    tracker.TrajectoryTracker._status(node, state)
    assert not json.loads(messages[-1].data)['arrival']['ready']


def test_reverse_cruises_faster_and_slows_down_near_saved_target():
    goal = np.zeros(3)
    cruise, state = reverse_command(np.array([.25, 0., 0.]), goal, np.zeros(3), 1., .2)
    near, _ = reverse_command(np.array([.02, 0., 0.]), goal, np.zeros(3), 1., .2)
    assert state == 'REVERSING'
    assert cruise[0] == pytest.approx(-.10)
    assert abs(near[0]) < .02


@pytest.mark.parametrize('guard', ['pending', 'cancel', 'nav2_active', 'wrong_goal', 'moving'])
def test_immediate_reverse_handoff_keeps_readiness_guards(monkeypatch, guard):
    monkeypatch.setattr(tracker.time, 'monotonic', lambda: 1.)
    node = bridge()
    node.active_request.reverse_final_pose = PoseStamped()
    node.finalizing_since = .9
    node._begin_reverse = lambda now: pytest.fail('premature reverse handoff')
    if guard == 'pending':
        node.pending_request = object()
    if guard == 'cancel':
        node.cancel_requested = True
    if guard == 'nav2_active':
        node.active_goal_handle = object()
    arrival = dict(goal=[1. if guard == 'wrong_goal' else 2., 0., 0.],
                   ready=guard != 'moving')
    node._tracker_status_cb(String(data=json.dumps({'arrival': arrival})))
    assert node.active_request.reverse_final_pose is not None


@pytest.mark.parametrize('pose', [[.009, 0., 0.], [0., -.009, .014], [-.006, .004, -.014]])
def test_reverse_skips_final_fine_position_adjustment(pose):
    command, state = reverse_command(np.array(pose), np.zeros(3), np.zeros(3), 1., .2)
    assert state == 'REVERSING'
    assert command == (0., 0., 0.)


@pytest.mark.parametrize('pose', [[.011, 0., 0.], [.008, .008, 0.], [0., 0., .016]])
def test_reverse_still_corrects_outside_completion_bounds(pose):
    command, state = reverse_command(np.array(pose), np.zeros(3), np.zeros(3), 1., .2)
    assert state == 'REVERSING'
    assert np.linalg.norm(command) > 0.
