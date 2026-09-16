"""Geometric rejection and real tracker execution with uncertain actuation."""
from collections import deque
import copy
import math
from pathlib import Path
from types import MethodType, SimpleNamespace
import numpy as np
import pytest
from std_msgs.msg import Bool, String
from omni_autonomy_next.rl_residual import CadClearanceModel
from omni_autonomy_next.staged_heading import (
    HeadingStage, dense_path, free_rotation_disk, prepare_stage, rotation_clearance, repair_corridor,
)
from omni_autonomy_next.staged_heading_node import StagedHeadingMixin
from omni_autonomy_next import trajectory_tracker_node as tracker
from test_smooth_arrival import make_node


def open_model():
    return CadClearanceModel([[[20.,-20.],[20.,20.]]],
                             [[-.45,-.42],[.45,-.42],[.45,.42],[-.45,.42]])


def staged_node(pose=(0.,0.,0.), goal=(2.,0.,math.pi/2)):
    node = make_node(pose, goal)
    node.motion_mode = node.requested_motion_mode = 'staged_heading'
    node.heading_stage = None
    node.stage_revision = 0
    node.stage_goal_enabled = True
    node.stage_blocked = None
    node.clearance = open_model()
    for name, method in vars(StagedHeadingMixin).items():
        if callable(method):
            setattr(node, name, MethodType(method, node))
    node._stage_live_turn_clear = lambda *args: True
    publish = node._publish
    def checked_publish(vx, vy, wz):
        if vx or vy or wz:
            vx, vy, wz = node._stage_safe_command(vx, vy, wz)
        publish(vx, vy, wz)
    node._publish = checked_publish
    return node


def test_rotation_sweep_rejects_clear_endpoints_with_corner_collision():
    model = CadClearanceModel([[[-2.,.5],[2.,.5]]], open_model().footprint)
    assert model.body_clearance([0.,0.], 0.) > .04
    assert model.body_clearance([0.,0.], math.pi/2) > .04
    assert rotation_clearance(model, [0.,0.], 0., math.pi/2) < 0.
    with pytest.raises(ValueError, match='NO_SAFE_ROTATION_GATE'):
        prepare_stage(model, [[0.,0.],[.1,0.]], np.zeros(3), HeadingStage(0.,math.pi/2))


def test_gate_leaves_a_narrow_bay_before_rotating_and_is_stable_on_replan():
    model = CadClearanceModel([[[-1.,.5],[.6,.5]], [[-1.,-.5],[.6,-.5]]], open_model().footprint)
    state, path, yaw = prepare_stage(model, [[0.,0.],[2.,0.]], np.zeros(3), HeadingStage(0.,math.pi/2))
    assert state.gate[0] > 1.
    assert yaw == 0.
    assert np.all(path[:,1] == 0.)
    gate = state.gate.copy()
    again, path, _ = prepare_stage(model, [[.2,0.],[2.,0.]], np.array([.2,0.,0.]), copy.deepcopy(state))
    assert again.gate == pytest.approx(gate)
    assert path[-1] == pytest.approx(gate)


def test_entire_translation_is_checked_including_wall_between_endpoints():
    with pytest.raises(ValueError, match='FIXED_HEADING_PATH_BLOCKED'):
        prepare_stage(CadClearanceModel([[[1.,-2.],[1.,2.]]], open_model().footprint),
                      [[0.,0.],[2.,0.]], np.zeros(3), HeadingStage(0.,0.))


def test_turn_map_checks_unknown_edges_and_cell_corners():
    grid = np.zeros((60,60), dtype=int)
    assert free_rotation_disk(grid,.05,(-1.5,-1.5),(0.,0.),.72)
    grid[30,42] = 100
    assert not free_rotation_disk(grid,.05,(-1.5,-1.5),(0.,0.),.72)
    grid[30,42] = -1
    assert not free_rotation_disk(grid,.05,(-1.5,-1.5),(0.,0.),.72)
    grid[30,42] = 80
    assert free_rotation_disk(grid,.05,(-1.5,-1.5),(0.,0.),.72)
    assert not free_rotation_disk(grid,.05,(-1.5,-1.5),(1.,0.),.72)


def test_cancel_invalidates_pending_plans_and_mode_changes_apply_next_goal():
    node = staged_node()
    node._on_motion_mode(String(data='simultaneous'))
    assert node.motion_mode == 'staged_heading'
    node._stage_cancel(Bool(data=True))
    tracker.TrajectoryTracker._build_trajectory(node, np.array([[0.,0.],[2.,0.]]), math.pi/2)
    assert node.trajectory is None
    assert node.commands[-1] == (0.,0.,0.)
    node._stage_new_goal()
    assert node.motion_mode == 'simultaneous'


def test_dynamic_obstacle_and_stale_map_prevent_turn(monkeypatch):
    node = staged_node()
    monkeypatch.setattr(tracker.time, 'monotonic', lambda: 0.)
    tracker.TrajectoryTracker._build_trajectory(node, np.array([[0.,0.],[2.,0.]]), math.pi/2)
    node.heading_stage.phase = 'ROTATE'
    node.heading_stage.rotation_started = 0.
    node._stage_live_turn_clear = lambda *args: False
    tracker.TrajectoryTracker._tick(node)
    assert node.commands[-1] == (0.,0.,0.)
    assert node.statuses[-1] == 'STAGED_WAITING_CLEARANCE'


def test_quantized_start_cannot_block_the_final_centimetres(monkeypatch):
    node = staged_node(pose=(-.813,1.407,0.), goal=(-.81,1.34,0.))
    config = Path(tracker.__file__).resolve().parents[1]/'config'
    node.clearance = CadClearanceModel.from_yaml(config/'field_planning.yaml',config/'competition_footprints.yaml')
    node.heading_stage = HeadingStage(0.,0.,phase='TRANSLATE')
    monkeypatch.setattr(tracker.time,'monotonic',lambda:0.)
    tracker.TrajectoryTracker._build_trajectory(node,np.array([[-.83,1.40]]),0.)
    assert node.stage_blocked is None
    assert node.trajectory.points[0] == pytest.approx(node.pose[:2])
    assert node.trajectory.points[-1] == pytest.approx(node.active_goal[:2])


def test_cancel_during_geometry_build_cannot_commit_the_old_trajectory(monkeypatch):
    node = staged_node()
    original = tracker.prepare_stage
    def cancel_during_build(*args,**kwargs):
        result = original(*args,**kwargs)
        node._stage_cancel(Bool(data=True))
        return result
    monkeypatch.setattr(tracker,'prepare_stage',cancel_during_build)
    tracker.TrajectoryTracker._build_trajectory(node,np.array([[0.,0.],[2.,0.]]),math.pi/2)
    assert node.trajectory is None
    assert not node.stage_goal_enabled


def test_clearance_recovery_only_moves_outward_without_yaw(monkeypatch):
    monkeypatch.setattr(tracker.time, 'monotonic', lambda: 0.)
    node = staged_node(pose=(-.815,.84,0.),goal=(-.81,1.34,0.))
    config = Path(tracker.__file__).resolve().parents[1]/'config'
    node.clearance = CadClearanceModel.from_yaml(config/'field_planning.yaml',config/'competition_footprints.yaml')
    node.heading_stage = HeadingStage(0.,0.,phase='TRANSLATE')
    initial = node.clearance.body_clearance(node.pose[:2],0.)
    assert .03 < initial < .041
    vx,vy,wz = node._stage_safe_command(.12,0.,.1)
    assert 0. < vx <= .05 and vy == wz == 0.
    assert np.linalg.norm(node._stage_safe_command(.01,0.,0.)[:2]) <= .01
    assert node._stage_safe_command(0.,0.,.1) == (0.,0.,0.)
    assert node._stage_safe_command(-.05,0.,0.) == (0.,0.,0.)
    node.velocity[0] = -.1
    assert node._stage_safe_command(.05,0.,0.) == (0.,0.,0.)


def test_low_clearance_connector_must_return_to_the_normal_corridor():
    config = Path(tracker.__file__).resolve().parents[1]/'config'
    model = CadClearanceModel.from_yaml(config/'field_planning.yaml',config/'competition_footprints.yaml')
    start = np.array([-.815,.84,0.])
    state,path,yaw = prepare_stage(model,[start[:2],[-.8,.84],[-.8,1.34]],start,HeadingStage(0.,0.))
    assert state.phase == 'TRANSLATE'
    with pytest.raises(ValueError,match='FIXED_HEADING_PATH_BLOCKED'):
        prepare_stage(model,[start[:2],[-.815,1.2]],start,HeadingStage(0.,0.))


def test_outward_command_cannot_hide_measured_inward_momentum(monkeypatch):
    monkeypatch.setattr(tracker.time, 'monotonic', lambda: 0.)
    node=staged_node(pose=(.70,0.,0.),goal=(2.,0.,0.))
    node.clearance=CadClearanceModel([[[0.,-3.],[0.,3.]]],open_model().footprint)
    node.heading_stage=HeadingStage(0.,0.,phase='TRANSLATE')
    node.velocity[:]=[-.8,0.,0.]
    assert node._stage_safe_command(.05,0.,0.) == (0.,0.,0.)
    assert node.stage_blocked == 'MEASURED_BRAKING_SWEEP_BLOCKED'


@pytest.mark.parametrize('gain,delay,tau', [(1.,.12,.08),(2.8,.2,.12),(2.8,.3,.15)])
@pytest.mark.parametrize('angle', [math.pi/2, -math.pi, .18, 0.])
@pytest.mark.parametrize('bay', [False, True])
def test_delayed_actual_tracker_separates_rotation_and_arrives(monkeypatch, gain, delay, tau, angle, bay):
    node = staged_node(goal=(2.,.5,angle))
    if bay:
        node.active_goal[1] = 0.
        node.clearance = CadClearanceModel([[[-1.,.5],[.6,.5]], [[-1.,-.5],[.6,-.5]]], open_model().footprint)
    actual = np.zeros(3)
    truth = node.pose.copy()
    pipe = deque([np.zeros(3) for _ in range(round(delay/.01))])
    tail = []
    phases = set()
    for step in range(3000):
        now = step*.01
        monkeypatch.setattr(tracker.time, 'monotonic', lambda: now)
        node.pose = truth.copy()
        node.pose_stamp = node.velocity_stamp = now
        node.velocity += (1-math.exp(-.01/.06))*(actual-node.velocity)
        if step % 100 == 0 or node.plan_event.is_set():
            node.plan_event.clear()
            node.last_plan = (np.array([node.pose[:2],node.active_goal[:2]]), angle)
            tracker.TrajectoryTracker._build_trajectory(node, *node.last_plan)
        if step % 5 == 0:
            tracker.TrajectoryTracker._tick(node)
            if node.heading_stage:
                phase = node.heading_stage.phase
                phases.add(phase)
                if phase == 'ROTATE':
                    assert np.linalg.norm(node.command[:2]) == 0.
                elif phase == 'SETTLE':
                    # The captured translation now brakes before rotation.
                    assert np.linalg.norm(node.command[:2]) < .08
                    assert node.command[2] == 0.
                if phase == 'ROTATE':
                    assert np.linalg.norm(actual[:2]) < .03
                if phase == 'TRANSLATE':
                    assert abs(tracker.wrap(truth[2]-angle)) < .02
        pipe.append(node.command.copy())
        actual += (pipe.popleft()*gain-actual)*(.01/tau)
        c,s = math.cos(truth[2]),math.sin(truth[2])
        truth += .01*np.array([actual[0]*c-actual[1]*s,actual[1]*c+actual[0]*s,actual[2]])
        if now > 27.:
            tail.append([np.linalg.norm(truth[:2]-node.active_goal[:2]),abs(tracker.wrap(truth[2]-angle))])
    assert 'TRANSLATE' in phases, (phases, node.stage_blocked, node.statuses[-1])
    assert np.max(np.asarray(tail)[:,0]) < .04, (truth,node.statuses[-1])
    assert np.max(np.asarray(tail)[:,1]) < .02


def test_repair_checks_a_real_smac_corner_cut_without_changing_the_goal():
    config = Path(tracker.__file__).resolve().parents[1]/'config'
    model = CadClearanceModel.from_yaml(config/'field_planning.yaml', config/'competition_footprints.yaml')
    points=np.array([[-1.19,2.54],[-1.185,2.34],[-1.078,2.14],[-.942,1.94],[-.864,1.74],[-.832,1.54],[-.81,1.34]])
    with pytest.raises(ValueError, match='FIXED_HEADING_PATH_BLOCKED'):
        prepare_stage(model,points,np.array([*points[0],0.]),HeadingStage(0.,0.))
    repaired=repair_corridor(model,points,[0.])
    state,path,yaw=prepare_stage(model,repaired,np.array([*points[0],0.]),HeadingStage(0.,0.))
    assert repaired[-1] == pytest.approx(points[-1])
    assert state.phase == 'TRANSLATE'
    assert min(model.body_clearance(p,yaw) for p in path) >= .036


def test_simultaneous_tracker_repairs_fixed_heading_corner_cut():
    config = Path(tracker.__file__).resolve().parents[1]/'config'
    model = CadClearanceModel.from_yaml(config/'field_planning.yaml', config/'competition_footprints.yaml')
    points = np.array([[-1.19,2.54],[-1.185,2.34],[-1.078,2.14],
                       [-.942,1.94],[-.864,1.74],[-.832,1.54],[-.81,1.34]])
    node = make_node((*points[0], 0.), (*points[-1], 0.))
    node.motion_mode = 'simultaneous'
    node.clearance = model
    tracker.TrajectoryTracker._build_trajectory(node, points, 0.)
    assert node.trajectory is not None
    assert node.trajectory.points[-1] == pytest.approx(points[-1])
    assert min(model.body_clearance(p, 0.) for p in node.trajectory.points) >= .04


@pytest.mark.parametrize('goal_id,side', [('4','upper'),('4','lower'),('5','upper'),('5','lower')])
def test_fixed_bucket_final_lanes_have_checked_constant_heading(goal_id, side):
    from omni_autonomy_next.configured_goals import load_configured_poses
    from omni_autonomy_next.route_approaches import load_fixed_goal_approaches
    config = Path(tracker.__file__).resolve().parents[1]/'config'
    model = CadClearanceModel.from_yaml(config/'field_planning.yaml', config/'competition_footprints.yaml')
    goal = load_configured_poses(config/'field_poses.yaml')[goal_id]
    waypoints = load_fixed_goal_approaches(config/'routes.yaml')[goal_id][side]['waypoints']
    points = np.array([[p['x'],p['y']] for p in waypoints]+[[goal['x'],goal['y']]])
    points = tracker.smooth_path(tracker.resample(points,.05),.05,.12)
    state, path, yaw = prepare_stage(model, points, np.array([*points[0],0.]), HeadingStage(0.,0.))
    assert state.phase == 'TRANSLATE'
    assert yaw == 0.
    assert min(model.body_clearance(p,0.) for p in path) > .045
