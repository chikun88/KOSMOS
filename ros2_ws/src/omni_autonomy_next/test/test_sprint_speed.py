"""Exercise live profile changes through the actual guard, tracker and UART encoder."""
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from std_msgs.msg import String
from omni_autonomy_next.config import ConfigError, load_robot
from omni_autonomy_next.motor_udp_bridge_node import MotorUdpBridge
from omni_autonomy_next.motor_udp_protocol import JetsonPacket, decode_v4_command
from omni_autonomy_next.robomas_uart import mix_velocity, parse_robomas_frame
from omni_autonomy_next.runtime_guard import RuntimeGuard, MotionLimits, GuardHealth
from omni_autonomy_next.omni_yaw import OmniEnvelope
from omni_autonomy_next.trajectory_tracker_node import TrajectoryTracker

PACKAGE = Path(__file__).resolve().parents[1]


def test_profile_switch_controls_guard_tracker_and_encoded_wheels():
    runtime = yaml.safe_load((PACKAGE/'config/runtime.yaml').read_text())['runtime_guard']['ros__parameters']
    # Follow the production tracker's loader, not the raw YAML. The normalized
    # loader previously discarded sprint's wheel budget and this test missed it.
    drive = load_robot(str(PACKAGE/'config/robot.yaml'))['drivetrain']
    profiles = json.loads(runtime['profiles_json'])
    hard = MotionLimits(*(runtime['hard_max_'+key] for key in (
        'linear_speed', 'lateral_speed', 'angular_speed', 'linear_acceleration',
        'angular_acceleration', 'linear_jerk', 'angular_jerk')))
    guard = RuntimeGuard(profiles={k: MotionLimits(**v) for k,v in profiles.items()},
        hard_limits=hard, default_profile='balanced', command_timeout_sec=.25,
        red_zone_speed_scale=1., wheel_radius=drive['wheel_radius'],
        wheel_positions=drive['wheel_positions'], wheel_drive_angles_rad=np.radians(drive['wheel_drive_angles_deg']),
        wheel_signs=drive['wheel_signs'], max_wheel_speed=drive['max_wheel_speed'],
        profile_max_wheel_speeds=drive['profile_max_wheel_speeds'])
    tracker = SimpleNamespace(profiles=profiles, envelope=OmniEnvelope(drive),
        default_max_wheel_speed=drive['max_wheel_speed'],
        profile_max_wheel_speeds=drive['profile_max_wheel_speeds'])
    bridge = SimpleNamespace(payload_format='v4_uart', sequence=0, estop=False,
                             _pending_velocity=(4*.38, 0., 0.))
    tick = 0
    for profile, speed, cap in [('balanced',2.,10000),('sprint',4.,10000),
                                 ('precision',2.,10000),('sprint',4.,10000),('unknown',2.,10000)]:
        selected = guard.resolve_profile(profile)
        TrajectoryTracker._apply_profile(tracker, selected)
        assert tracker.envelope.speed_limit(np.array([1.,0.]),0.,tracker.speed_limit,tracker.lateral_limit) == pytest.approx(speed)
        for _ in range(1000):
            tick += 1
            result = guard.step((4.,0.,0.),now_sec=tick*.01,command_age_sec=0.,
                health=GuardHealth(True,False,True,True,True),profile=profile,user_scale=1.,red_zone=False)
        assert result.velocity[0] == pytest.approx(speed)
        MotorUdpBridge._profile_callback(bridge,String(data=profile))
        frame,*_ = decode_v4_command(MotorUdpBridge._encode_udp_payload(bridge,JetsonPacket()))
        assert [v for _,v in parse_robomas_frame(frame)] == [-cap,-cap,cap,cap]
        bridge._pending_velocity=(0.,0.,0.)
        MotorUdpBridge._encode_udp_payload(bridge,JetsonPacket())
        assert bridge._last_wheel_commands == (0,0,0,0)
        bridge._pending_velocity=(4*.38,0.,0.)


@pytest.mark.parametrize('angle',np.linspace(-math.pi,math.pi,17))
@pytest.mark.parametrize('yaw',[-1.8,0.,1.8])
def test_sprint_mixed_motion_preserves_ratio_and_never_exceeds_10000(angle,yaw):
    vx,vy,wz=4*.38*math.cos(angle),4*.38*math.sin(angle),yaw*.34
    wheels,scale=mix_velocity(vx,vy,wz,wheel_limit=10000)
    assert max(map(abs,wheels)) <= 10000
    raw=np.array([vy-vx,-vy-vx,-vy+vx,vy+vx])*7202-wz*4797.569088
    assert wheels == pytest.approx(raw*scale,abs=.5)


@pytest.mark.parametrize('cap',[0,-1,10001,32767,float('nan'),1.5])
def test_unapproved_explicit_uart_limit_is_rejected(cap):
    with pytest.raises(ValueError):
        mix_velocity(4.,0.,0.,wheel_limit=cap)


@pytest.mark.parametrize('angle', [0., math.pi/4, math.pi/2, math.pi])
def test_loaded_sprint_budget_reaches_trajectory_and_returns_to_balanced(monkeypatch, angle):
    from test_smooth_arrival import make_node
    from omni_autonomy_next import trajectory_tracker_node as module
    monkeypatch.setattr(module.time, 'monotonic', lambda: 0.)
    goal = [60.*math.cos(angle), 60.*math.sin(angle), 0.]
    node = make_node(goal=goal)
    drive = load_robot(str(PACKAGE/'config/robot.yaml'))['drivetrain']
    runtime = yaml.safe_load((PACKAGE/'config/runtime.yaml').read_text())['runtime_guard']['ros__parameters']
    node.profiles = json.loads(runtime['profiles_json'])
    node.default_max_wheel_speed = drive['max_wheel_speed']
    node.profile_max_wheel_speeds = drive['profile_max_wheel_speeds']
    for profile, axis_limit in [('sprint', 4.), ('balanced', 2.), ('sprint', 4.)]:
        TrajectoryTracker._apply_profile(node, profile)
        node.trajectory = None
        TrajectoryTracker._build_trajectory(node, np.array([[0., 0.], goal[:2]]), 0.)
        expected = axis_limit/(abs(math.cos(angle))+abs(math.sin(angle)))
        assert max(node.trajectory.speed) == pytest.approx(expected, rel=1e-8)
        assert node.trajectory.wheel_limit == pytest.approx(node.envelope.max_wheel)
        assert node.trajectory.speed[-1] == 0.


@pytest.mark.parametrize('overrides', [[], {'sprint': 0}, {'sprint': -1},
                                      {'sprint': float('nan')}, {'sprint': float('inf')},
                                      {'': 10}, {1: 10}])
def test_invalid_loaded_profile_wheel_limits_fail_closed(tmp_path, overrides):
    raw = yaml.safe_load((PACKAGE/'config/robot.yaml').read_text())
    raw['robot']['cad_model_file'] = str((PACKAGE/'config'/raw['robot']['cad_model_file']).resolve())
    raw['robot']['drivetrain']['profile_max_wheel_speeds'] = overrides
    path = tmp_path/'robot.yaml'
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ConfigError, match='profile_max_wheel_speeds|wheel speed profile names'):
        load_robot(str(path))


def test_legacy_robot_without_profile_wheel_limits_keeps_default(tmp_path):
    raw = yaml.safe_load((PACKAGE/'config/robot.yaml').read_text())
    raw['robot']['cad_model_file'] = str((PACKAGE/'config'/raw['robot']['cad_model_file']).resolve())
    raw['robot']['drivetrain'].pop('profile_max_wheel_speeds')
    path = tmp_path/'robot.yaml'
    path.write_text(yaml.safe_dump(raw))
    drive = load_robot(str(path))['drivetrain']
    assert drive['profile_max_wheel_speeds'] == {}
    assert drive['max_wheel_speed'] == raw['robot']['drivetrain']['max_wheel_speed']
