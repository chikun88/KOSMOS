"""Configured cruise must reach the transport without an extra speed clamp."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace

from geometry_msgs.msg import Twist
import numpy as np
import pytest
import yaml

from omni_autonomy_next.motor_udp_bridge_node import MotorUdpBridge
from omni_autonomy_next.robomas_uart import mix_velocity

PACKAGE = Path(__file__).resolve().parents[1]
RUNTIME = yaml.safe_load((PACKAGE/'config/runtime.yaml').read_text())['runtime_guard']['ros__parameters']
DRIVE = yaml.safe_load((PACKAGE/'config/robot.yaml').read_text())['robot']['drivetrain']


def launched_bridge_limits(runtime):
    # Execute the actual two launch parameter expressions without starting ROS
    # nodes or opening any device. A literal 1.0 regression must fail this test.
    tree = ast.parse((PACKAGE/'launch/system.launch.py').read_text())
    values = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if isinstance(key, ast.Constant) and key.value in ('max_linear_speed', 'max_angular_speed'):
                    values[key.value] = eval(compile(ast.Expression(value), '<launch parameter>', 'eval'),
                                             {'runtime': runtime})
    return values


@pytest.mark.parametrize('axis,sign', [(0, 1), (0, -1), (1, 1), (1, -1)])
def test_full_sprint_reaches_calibrated_wire_without_extra_clipping(axis, sign):
    limits = launched_bridge_limits(RUNTIME)
    node = SimpleNamespace(latest_twist=Twist(), linear_x_sign=1., linear_y_sign=1.,
        angular_z_sign=1., linear_command_scale=DRIVE['linear_command_scale'],
        angular_command_scale=DRIVE['angular_command_scale'], **limits)
    profile = json.loads(RUNTIME['profiles_json'])['sprint']
    velocity = np.zeros(3)
    velocity[axis] = sign*profile['linear' if axis == 0 else 'lateral']
    node.latest_twist.linear.x = float(velocity[0])
    node.latest_twist.linear.y = float(velocity[1])
    wire = MotorUdpBridge._velocity_from_latest_twist(node)
    assert wire == pytest.approx(velocity*DRIVE['linear_command_scale'])
    wheels, _ = mix_velocity(*wire)
    assert max(abs(v) for v in wheels) <= 10000


def test_launch_bridge_envelope_tracks_custom_runtime_limits():
    runtime = dict(RUNTIME, hard_max_linear_speed=.6, hard_max_lateral_speed=.8,
                   hard_max_angular_speed=.9)
    assert launched_bridge_limits(runtime) == dict(max_linear_speed=.8, max_angular_speed=.9)


def test_higher_low_profile_retains_zero_stop_and_continuous_fine_control():
    from omni_autonomy_next.trajectory_tracker_node import terminal_translation
    # The precision profile's cruise cap is NOT a floor on final positioning.
    for distance in [.01, .001, 0.]:
        command = terminal_translation(np.zeros(2), np.array([-distance, 0.]),
            np.zeros(2), distance, .1, 1.6, .3)
        assert np.linalg.norm(command) <= 1.6*distance + 1e-12
    assert command == pytest.approx(np.zeros(2))


def test_two_mps_axes_survive_planner_guard_and_calibrated_uart():
    from omni_autonomy_next.omni_yaw import OmniEnvelope
    from omni_autonomy_next.motor_kinematics import allocate_omni4_wheel_budget
    envelope = OmniEnvelope(DRIVE)
    for profile in ('precision', 'balanced', 'sprint'):
        limits = json.loads(RUNTIME['profiles_json'])[profile]
        for direction in (np.array([1., 0.]), np.array([-1., 0.]),
                          np.array([0., 1.]), np.array([0., -1.])):
            speed = envelope.speed_limit(direction, 0., limits['linear'], limits['lateral'])
            assert speed == pytest.approx(2.)
    # Oversized mixed commands must be limited BEFORE the wire mixer. Check
    # all quadrants and both yaw signs, including different XY/yaw calibration.
    for angle in np.linspace(-np.pi, np.pi, 73):
        for yaw in (-1.8, 0., 1.8):
            safe = allocate_omni4_wheel_budget(4*np.cos(angle), 4*np.sin(angle), yaw,
                wheel_radius=DRIVE['wheel_radius'], wheel_positions=DRIVE['wheel_positions'],
                wheel_drive_angles=np.radians(DRIVE['wheel_drive_angles_deg']),
                wheel_signs=DRIVE['wheel_signs'], maximum=DRIVE['max_wheel_speed'])
            wire = np.asarray(safe)*[DRIVE['linear_command_scale'],
                DRIVE['linear_command_scale'], DRIVE['angular_command_scale']]
            wheels, saturation = mix_velocity(*wire)
            assert saturation == pytest.approx(1.)
            assert max(abs(v) for v in wheels) <= 10000


@pytest.mark.parametrize('distance', [0., .1, .5, 2., 8.])
@pytest.mark.parametrize('delay', [0., .32, .45])
def test_stopping_speed_includes_response_distance(distance, delay):
    from omni_autonomy_next.trajectory_tracker_node import stopping_speed
    deceleration = .75*.85
    speed = stopping_speed(distance, deceleration, delay)
    assert speed >= 0.
    assert speed*delay + speed**2/(2*deceleration) == pytest.approx(distance)


@pytest.mark.parametrize('yaw', [0., np.pi/2, -np.pi])
def test_two_mps_plan_uses_directional_budget_for_heading_changes(monkeypatch, yaw):
    cap = 2.
    from test_smooth_arrival import make_node
    from omni_autonomy_next import trajectory_tracker_node as module
    monkeypatch.setattr(module.time, 'monotonic', lambda: 0.)
    node = make_node(goal=(8., 0., yaw))
    node.speed_limit = node.lateral_limit = 2.
    module.TrajectoryTracker._build_trajectory(node, np.array([[0., 0.], [8., 0.]]), yaw)
    assert node.trajectory is not None
    assert node.trajectory.linear_limit == pytest.approx(cap)
    assert node.trajectory.lateral_limit == pytest.approx(cap)
    assert np.max(node.trajectory.speed) <= cap + 1e-9
    assert node.trajectory.wheel_limit == pytest.approx(node.envelope.max_wheel)
    if not yaw:
        assert np.max(node.trajectory.speed) == pytest.approx(2.)
    for tangent, heading, rate, speed, budgets in zip(
            module.path_tangents(node.trajectory.points), node.trajectory.yaws,
            node.trajectory.yaw_per_metre, node.trajectory.speed,
            node.trajectory.motion_limits):
        velocity = module.to_body(tangent * speed, heading)
        assert node.envelope.wheel_cost(*velocity, rate * speed) <= budgets[2] + 1e-7


def test_heading_settled_section_recovers_full_cruise(monkeypatch):
    from test_smooth_arrival import make_node
    from omni_autonomy_next import trajectory_tracker_node as module
    monkeypatch.setattr(module.time, 'monotonic', lambda: 0.)
    node = make_node(goal=(12., 0., np.pi/2))
    node.speed_limit = node.lateral_limit = 2.
    original = node.get_parameter
    node.get_parameter = lambda name: (SimpleNamespace(value=True)
        if name == 'optimize_yaw' else original(name))
    node._plan_yaw = lambda points, tangents, arc, *args: np.minimum(arc/2., 1.)*np.pi/2
    module.TrajectoryTracker._build_trajectory(node, np.array([[0., 0.], [12., 0.]]), np.pi/2)
    trajectory = node.trajectory
    turning = np.abs(trajectory.yaw_per_metre) > 1.e-6
    assert np.allclose(trajectory.motion_limits[turning, 0], 1.05)
    assert np.allclose(trajectory.motion_limits[turning, 2], 15.709120382)
    assert np.allclose(trajectory.motion_limits[~turning, 0], 2.)
    assert np.max(trajectory.speed[~turning]) == pytest.approx(2.)


def test_feedback_obeys_observed_turn_budget_before_releasing_cruise(monkeypatch):
    from test_smooth_arrival import make_node
    from omni_autonomy_next import trajectory_tracker_node as module
    monkeypatch.setattr(module.time, 'monotonic', lambda: 0.)
    node = make_node(goal=(8., 0., 0.))
    node.speed_limit = node.lateral_limit = 2.
    node._rate_limit = lambda *command: command
    module.TrajectoryTracker._build_trajectory(node, np.array([[0., 0.], [8., 0.]]), 0.)
    node.reference_time = 3.
    module.TrajectoryTracker._tick(node)
    assert np.linalg.norm(node.command[:2]) > 1.05
    # Only the observed interval is restricted. A leading reference must not
    # choose the full-speed budget farther down the route.
    node.trajectory.motion_limits[:2] = [1.05, 1.05, 15.709120382]
    node.reference_time = 3.
    module.TrajectoryTracker._tick(node)
    assert 0. < np.linalg.norm(node.command[:2]) <= 1.05 + 1e-9
    assert node.envelope.wheel_cost(*node.command) <= 15.709120382 + 1e-9
