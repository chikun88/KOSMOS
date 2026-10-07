"""Regressions for delayed replans, passed gates and competing wall control."""
import json
import numpy as np
import pytest
from std_msgs.msg import String
from omni_autonomy_next import trajectory_tracker_node as tracker
from omni_autonomy_next.route_approaches import remaining_lane_gates
from omni_autonomy_next.rl_residual import apply_clearance_residual, CadClearanceModel
from omni_autonomy_next.staged_heading import HeadingStage
from test_staged_heading import staged_node, open_model
from test_smooth_arrival import make_node
from pathlib import Path
import xml.etree.ElementTree as ET
import yaml


def test_gate_passage_is_checked_outside_the_slow_planner_rate_gate():
    root = ET.parse(Path(tracker.__file__).resolve().parents[1]
                    /'behavior_trees/follow_fixed_approach.xml').getroot()
    pipeline = root.find('.//PipelineSequence')
    removal = pipeline.find('RemovePassedBucketGoals')
    assert removal is not None
    assert float(removal.attrib['radius']) == .16
    assert pipeline.find('RateController//RemovePassedBucketGoals') is None
    config = yaml.safe_load((Path(tracker.__file__).resolve().parents[1]
                             /'config/nav2_next.yaml').read_text())
    assert config['planner_server']['ros__parameters']['GridBased']['tolerance'] <= .05


def test_delayed_plan_discards_old_vertices_without_cutting_the_next_bend():
    result = tracker.remaining_path([[0.,0.],[.2,0.],[.4,0.],[.4,1.]], [.3,.01])
    assert result == pytest.approx(np.array([[.3,.01],[.3,0.],[.4,0.],[.4,1.]]))


def test_recorded_bucket_corner_is_recognized_before_robot_has_to_reverse():
    # 20260914T134455.591279Z-6199a85a: the closest localized pose to the
    # (-.80, -1.55) gate. Missing it left 0.20-0.70 m backwards plan legs.
    position = np.array([-.916023113961135, -1.5909266122963461])
    gate = np.array([-.80, -1.55])
    tree = ET.parse(Path(tracker.__file__).resolve().parents[1]
                    /'behavior_trees/follow_fixed_approach.xml')
    radius = float(tree.find('.//RemovePassedBucketGoals').attrib['radius'])
    distance = float(np.linalg.norm(position-gate))
    assert .12 < distance < radius
    # Still retain the next inner gate (-.80, -1.95) and the narrow final lane.
    assert np.linalg.norm(position-[-.80, -1.95]) > radius
    assert radius < .18


def test_same_corridor_replan_preserves_reference_clock(monkeypatch):
    node = make_node(goal=(3.,0.,0.))
    monkeypatch.setattr(tracker.time, 'monotonic', lambda: 0.)
    tracker.TrajectoryTracker._build_trajectory(node,np.array([[0.,0.],[3.,0.]]),0.)
    node.pose[0] = 1.
    node.reference_time = .7
    original = node.trajectory
    tracker.TrajectoryTracker._build_trajectory(node,np.array([[1.,0.],[3.,0.]]),0.)
    assert node.trajectory is original
    assert node.reference_time == .7
    # A route change or profile change must still be rebuilt.
    assert not tracker.same_remaining_corridor([[0.,0.],[1.,1.],[3.,0.]],
                                               [[0.,0.],[3.,0.]],[0.,0.])
    node.speed_limit = .4
    tracker.TrajectoryTracker._build_trajectory(node,np.array([[1.,0.],[3.,0.]]),0.)
    assert node.trajectory is not original


@pytest.mark.parametrize('state', ['SUCCEEDED','CANCELED','FAILED','PREEMPTING'])
def test_both_modes_cancel_immediately_and_reject_late_plans(monkeypatch,state):
    node = staged_node(goal=(2.,0.,0.))
    node.motion_mode = 'simultaneous'
    monkeypatch.setattr(tracker.time,'monotonic',lambda:0.)
    tracker.TrajectoryTracker._build_trajectory(node,np.array([[0.,0.],[2.,0.]]),0.)
    node._stage_goal_status(String(data=json.dumps({'state':state})))
    tracker.TrajectoryTracker._build_trajectory(node,np.array([[0.,0.],[2.,0.]]),0.)
    tracker.TrajectoryTracker._tick(node)
    assert node.trajectory is None
    assert node.command == pytest.approx([0.,0.,0.])


def test_passed_lane_gates_are_skipped_only_for_on_lane_starts():
    gates = [dict(x=-.8,y=y) for y in [2.15,1.75]]
    assert remaining_lane_gates(gates,[-.81,1.6],-1.) == []
    assert remaining_lane_gates(gates,[-.81,2.0],-1.) == gates[1:]
    assert remaining_lane_gates(gates,[-1.2,1.6],-1.) == gates
    assert remaining_lane_gates(gates,[-.8,3.],-1.) == gates


def test_heading_disturbance_keeps_the_correcting_servo_active(monkeypatch):
    node = staged_node(goal=(2.,0.,0.))
    monkeypatch.setattr(tracker.time,'monotonic',lambda:0.)
    tracker.TrajectoryTracker._build_trajectory(node,np.array([[0.,0.],[2.,0.]]),0.)
    node.pose[2] = .03
    tracker.TrajectoryTracker._tick(node)
    assert node.command[2] < 0.
    assert node.command[0] > 0.


def test_braking_constraint_reduces_speed_without_latching_a_full_stop(monkeypatch):
    monkeypatch.setattr(tracker.time, 'monotonic', lambda: 0.)
    node = staged_node(pose=(.70,0.,0.),goal=(2.,0.,0.))
    node.clearance = CadClearanceModel([[[0.,-3.],[0.,3.]]],open_model().footprint)
    node.heading_stage = HeadingStage(0.,0.,phase='TRANSLATE')
    vx,vy,wz = node._stage_safe_command(-.8,0.,0.)
    assert -.8 < vx < 0.
    assert vy == wz == 0.
    stopping_distance = abs(vx)*.2 + vx*vx/(2*.85)
    assert node.pose[0] - stopping_distance - .45 >= .035
    assert node.stage_blocked is None
    assert node._stage_safe_command(.4,0.,0.) == pytest.approx((.4,0.,0.))


def test_clearance_correction_preserves_safe_wall_parallel_motion(monkeypatch):
    monkeypatch.setattr(tracker.time, 'monotonic', lambda: 0.)
    node = staged_node(pose=(.49,0.,0.),goal=(.5,2.,0.))
    node.clearance = CadClearanceModel([[[0.,-3.],[0.,3.]]],open_model().footprint)
    node.heading_stage = HeadingStage(0.,0.,phase='TRANSLATE')
    node.velocity[:] = [0.,.3,0.]
    vx,vy,wz = node._stage_safe_command(0.,.3,0.)
    assert 0. < vx <= .035
    assert vy > .29
    assert wz == 0.


def test_geometry_worker_latency_does_not_fabricate_a_planner_outage(monkeypatch):
    node = make_node(goal=(3.,0.,0.))
    monkeypatch.setattr(tracker.time,'monotonic',lambda:0.)
    tracker.TrajectoryTracker._build_trajectory(node,np.array([[0.,0.],[3.,0.]]),0.)
    monkeypatch.setattr(tracker.time,'monotonic',lambda:3.)
    node.velocity_stamp = node.pose_stamp = node.plan_received_stamp = 3.
    tracker.TrajectoryTracker._tick(node)
    assert node.statuses[-1] == 'TRACKING'
    assert node.command[0] > 0.
    monkeypatch.setattr(tracker.time,'monotonic',lambda:6.)
    node.velocity_stamp = node.pose_stamp = 6.
    tracker.TrajectoryTracker._tick(node)
    assert node.statuses[-1] == 'PLAN_STALE'
    assert node.command == pytest.approx([0.,0.,0.])


@pytest.mark.parametrize('yaw', [0.,.7,-1.5])
def test_wall_parallel_and_outward_requests_are_not_pushed_off_path(yaw):
    for world in ([0.,.6],[.1,.6]):
        body = tracker.to_body(np.array(world),yaw)
        adjusted = apply_clearance_residual(body,yaw=yaw,body_clearance=.04,
            gradient=[1.,0.],clearance_push=1.6,include_baseline=True)
        assert adjusted == pytest.approx(body)


def test_previous_bucket_plan_cannot_replace_new_goal_or_refresh_heartbeat(monkeypatch):
    from nav_msgs.msg import Path as Plan
    from geometry_msgs.msg import PoseStamped
    node = make_node(goal=(3., 0., 0.))
    monkeypatch.setattr(tracker.time, 'monotonic', lambda: 0.)
    tracker.TrajectoryTracker._build_trajectory(node, np.array([[0., 0.], [3., 0.]]), 0.)
    current = node.trajectory
    node.plan_received_stamp = 0.
    plan = Plan()
    plan.header.frame_id = 'map'
    plan.header.stamp.sec = 1
    for x in (0., -3.):
        p = PoseStamped()
        p.header = plan.header
        p.pose.position.x = x
        p.pose.orientation.w = 1.
        plan.poses.append(p)
    monkeypatch.setattr(tracker.time, 'monotonic', lambda: 1.)
    tracker.TrajectoryTracker._on_plan(node, plan)
    assert node.pending_plan is None
    assert node.plan_received_stamp == 0.
    tracker.TrajectoryTracker._build_trajectory(node, np.array([[0., 0.], [-3., 0.]]), 0.)
    assert node.trajectory is current
    # A grid endpoint within snap tolerance is still accepted for the new goal.
    plan.poses[-1].pose.position.x = 2.95
    tracker.TrajectoryTracker._on_plan(node, plan)
    assert node.pending_plan is not None
    assert node.plan_received_stamp == 1.
