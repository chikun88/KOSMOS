"""Exercise the deployed control tick, including degenerate navigation goals."""
import math
import ast
import threading
from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest

from omni_autonomy_next import trajectory_tracker_node as tracker
from omni_autonomy_next.omni_yaw import OmniEnvelope, wrap
from omni_autonomy_next.config import load_robot
from pathlib import Path

_tree = ast.parse(Path(tracker.__file__).read_text(encoding='utf-8'))
_defaults = next(n.value for n in ast.walk(_tree) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'defaults' for t in n.targets))
TRACKER_DEFAULTS = ast.literal_eval(_defaults)


def make_node(pose=(0., 0., 0.), goal=(0., 0., math.pi)):
    params = dict(pose_timeout_sec=.3, plan_timeout_sec=2., feedback_delay_sec=.2,
                  max_predicted_yaw_rad=.35, max_reference_lead_m=.35,
                  no_progress_speed_m_s=.05, no_progress_epsilon_m=.02,
                  no_progress_timeout_sec=4., terminal_approach_m=.10,
                  terminal_timeout_sec=4., terminal_hold_sec=3.,
                  position_gain=2.4, position_damping=.25, yaw_gain=1.6,
                  terminal_speed=.3, goal_snap_distance_m=.3,
                  resample_spacing_m=.05, path_smoothing_m=.12, optimize_yaw=False)
    params.update({key: TRACKER_DEFAULTS[key] for key in params})
    params['velocity_timeout_sec'] = TRACKER_DEFAULTS.get('velocity_timeout_sec', .2)
    params['velocity_filter_sec'] = TRACKER_DEFAULTS['velocity_filter_sec']
    params['predictive_sprint'] = TRACKER_DEFAULTS['predictive_sprint']
    params['sprint_turn_everywhere'] = TRACKER_DEFAULTS['sprint_turn_everywhere']
    root = Path(tracker.__file__).resolve().parents[1]
    envelope = OmniEnvelope(load_robot(str(root / 'config/robot.yaml'))['drivetrain'])
    node = SimpleNamespace(
        trajectory=None, pose=np.array(pose, float), pose_stamp=0.,
        reference_time=0., plan_stamp=0., active_goal=np.array(goal, float),
        lock=threading.Lock(), velocity=np.zeros(3), speed_scale=1., period=.05,
        velocity_stamp=0., velocity_source_stamp=None, odometry_paused=False,
        best_distance=math.inf, best_distance_at=0., terminal_since=None,
        terminal_best_yaw=math.inf, tracked_endpoint=None, command=np.zeros(3),
        finished_at=None, speed_limit=.78, lateral_limit=.702, yaw_limit=1.3,
        acceleration=.85, lateral_acceleration=1.2, yaw_acceleration=2.,
        envelope=envelope, plan_build_ms=0., profile_name='balanced',
        snap_offset=0., snapped=True, pending_plan=None, last_plan=None,
        plan_event=threading.Event(), statuses=[], commands=[],
        get_parameter=lambda name: SimpleNamespace(value=params[name]),
        get_logger=lambda: SimpleNamespace(warning=lambda *a, **kw: None),
        _relay_behavior=lambda *args: False,
        _monitor_safe_yaw_rate=lambda rate, *args: rate,
        _track_flow=lambda *args: None, _flow_gap=lambda: 0., _flow_gain=lambda: 1.)
    node._status = lambda state, **kw: node.statuses.append(state)
    def publish(*command):
        node.command = np.array(command)
        node.commands.append(command)
        tracker.record_prediction_command(node, tracker.time.monotonic(), node.command)
    node._publish = publish
    node._rate_limit = lambda *command: tracker.TrajectoryTracker._rate_limit(node, *command)
    return node


@pytest.mark.parametrize('length', [.02, .04, .1, .3, 1., 3.])
@pytest.mark.parametrize('angle', [math.pi / 2, -math.pi, .2])
def test_planned_angular_acceleration_includes_translation_acceleration(length, angle):
    points = tracker.resample(np.array([[0., 0.], [length, 0.]]), .05)
    s = points[:, 0]
    # The deployed ramp retires rotation before the final approach.
    yaw = angle * np.minimum(s / max(length - .1, length / 2), 1.)
    plan = tracker.Trajectory(points, yaw, np.full(len(points), .78),
        acceleration=.85, lateral_acceleration=1.2, entry_speed=0.,
        angular_speed=1.3, angular_acceleration=2.)
    times = np.linspace(0., plan.duration, 2001)
    rates = np.array([plan.sample(t)[3] for t in times])
    assert np.max(np.abs(rates)) <= 1.3 + 1.e-8
    assert np.max(np.abs(np.diff(rates) / np.diff(times))) <= 2. + 1.e-5


@pytest.mark.parametrize('duplicates', [1, 2, 5])
def test_in_place_plan_is_accepted_without_inventing_translation(monkeypatch, duplicates):
    monkeypatch.setattr(tracker.time, 'monotonic', lambda: 0.)
    node = make_node()
    pose = SimpleNamespace(position=SimpleNamespace(x=0., y=0.),
        orientation=SimpleNamespace(x=0., y=0., z=1., w=0.))
    message = SimpleNamespace(poses=[SimpleNamespace(pose=pose)] * duplicates)
    tracker.TrajectoryTracker._on_plan(node, message)
    assert node.pending_plan is not None
    tracker.TrajectoryTracker._build_trajectory(node, *node.pending_plan)
    assert node.trajectory is not None
    tracker.TrajectoryTracker._tick(node)
    assert node.commands[-1][:2] == pytest.approx((0., 0.))
    assert abs(node.commands[-1][2]) > 0.


def test_terminal_translation_has_no_command_step_at_handoff():
    outputs = []
    for distance in [.100001, .1, .099999]:
        outputs.append(tracker.terminal_translation(
            np.array([.05, .15]), np.array([-distance, 0.]), np.zeros(2),
            distance, .1, 2.4, .3))
    assert outputs[0] == pytest.approx(outputs[-1], abs=1.e-5)
    assert outputs[1] == pytest.approx([.24, 0.])


def test_pause_does_not_expire_terminal_hold(monkeypatch):
    node = make_node(goal=(.04, 0., 1.))
    monkeypatch.setattr(tracker.time, 'monotonic', lambda: 0.)
    tracker.TrajectoryTracker._build_trajectory(node, np.array([[0., 0.], [.04, 0.]]), 1.)
    node.finished_at = 0.
    node.terminal_since = 0.
    node.speed_scale = 0.
    for now in range(1, 8):
        monkeypatch.setattr(tracker.time, 'monotonic', lambda: float(now))
        node.velocity_stamp = node.pose_stamp = node.plan_stamp = float(now)
        tracker.TrajectoryTracker._tick(node)
        assert node.commands[-1] == pytest.approx((0., 0., 0.))
    node.speed_scale = 1.
    tracker.TrajectoryTracker._tick(node)
    assert node.statuses[-1] not in ('TERMINAL_TIMEOUT', 'TERMINAL_HOLD_EXPIRED')
    assert abs(node.commands[-1][2]) > 0.


@pytest.mark.parametrize('distance,angle', [(0., math.pi), (.02, -math.pi),
                                          (.3, math.pi / 2), (2., -math.pi / 2)])
@pytest.mark.parametrize('gain', [1., 2.8])
@pytest.mark.parametrize('direction', np.arange(8) * math.pi / 4)
def test_real_tick_arrives_with_delay_and_replanning(monkeypatch, distance, angle, gain, direction):
    node = make_node(goal=(distance * math.cos(direction), distance * math.sin(direction), angle))
    actual = np.zeros(3)
    delay = deque([np.zeros(3) for _ in range(12)])
    accepted = False
    truth = node.pose.copy()
    for step in range(1800):
        now = step * .01
        monkeypatch.setattr(tracker.time, 'monotonic', lambda: now)
        node.pose = truth.copy()
        node.velocity_stamp = node.pose_stamp = now
        node.velocity += (1. - math.exp(-.01 / TRACKER_DEFAULTS['velocity_filter_sec'])) * (actual - node.velocity)
        if step % 100 == 0:
            tracker.TrajectoryTracker._build_trajectory(node,
                np.array([node.pose[:2], node.active_goal[:2]]), angle)
        if step % 5 == 0:
            tracker.TrajectoryTracker._tick(node)
        delay.append(node.command.copy())
        actual += (delay.popleft() * gain - actual) * (.01 / .08)
        c, s = math.cos(truth[2]), math.sin(truth[2])
        truth += .01 * np.array([actual[0]*c - actual[1]*s,
                                actual[0]*s + actual[1]*c, actual[2]])
        if np.linalg.norm(truth[:2] - node.active_goal[:2]) < .04 and abs(wrap(truth[2] - angle)) < .035:
            accepted = True
            break
    assert accepted, (truth, node.statuses[-1])
    assert not {'TERMINAL_TIMEOUT', 'TERMINAL_HOLD_EXPIRED'} & set(node.statuses)
    commands = np.array(node.commands)
    assert np.max(np.abs(np.diff(commands[:, 2]))) <= .100001


def test_stuck_rotation_still_times_out(monkeypatch):
    node = make_node()
    for step in range(120):
        now = .05 * step
        monkeypatch.setattr(tracker.time, 'monotonic', lambda: now)
        node.velocity_stamp = node.pose_stamp = now
        if step % 20 == 0:
            tracker.TrajectoryTracker._build_trajectory(node, np.zeros((2, 2)), math.pi)
        tracker.TrajectoryTracker._tick(node)
    assert node.statuses[-1] in ('TERMINAL_TIMEOUT', 'TERMINAL_HOLD_EXPIRED')
    assert node.commands[-1] == pytest.approx((0., 0., 0.))


@pytest.mark.parametrize('changed', [False, True])
def test_only_a_new_heading_resets_the_same_position_watchdog(changed):
    node = make_node(goal=(0., 0., 0.))
    node.finished_at = node.terminal_since = 2.
    node.terminal_best_yaw = .01
    heading = math.pi if changed else 0.
    pose = SimpleNamespace(position=SimpleNamespace(x=0., y=0.),
        orientation=SimpleNamespace(x=0., y=0., z=math.sin(heading/2),
                                    w=math.cos(heading/2)))
    tracker.TrajectoryTracker._on_goal(node, SimpleNamespace(pose=pose))
    assert node.finished_at == (None if changed else 2.)
    assert node.terminal_since == (None if changed else 2.)
