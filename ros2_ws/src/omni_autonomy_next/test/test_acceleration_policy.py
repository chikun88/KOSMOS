"""Fast launches must retain the independent braking and correction budgets."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from omni_autonomy_next.config import load_robot
from omni_autonomy_next.omni_yaw import OmniEnvelope
from omni_autonomy_next.trajectory_tracker_node import Trajectory, TrajectoryTracker

CONFIG = Path(__file__).resolve().parents[1]/'config'


def test_profile_switch_restores_acceleration_without_changing_braking():
    drive = load_robot(str(CONFIG/'robot.yaml'))['drivetrain']
    runtime = yaml.safe_load((CONFIG/'runtime.yaml').read_text())['runtime_guard']['ros__parameters']
    node = SimpleNamespace(profiles=json.loads(runtime['profiles_json']),
        envelope=OmniEnvelope(drive), default_max_wheel_speed=drive['max_wheel_speed'],
        profile_max_wheel_speeds=drive['profile_max_wheel_speeds'],
        profile_linear_accelerations=drive['profile_linear_accelerations'],
        default_acceleration=.85, deceleration=.85)
    for name in ('sprint', 'balanced', 'precision', 'sprint'):
        TrajectoryTracker._apply_profile(node, name)
        expected = drive['profile_linear_accelerations'].get(name, .85)
        assert node.acceleration == pytest.approx(expected)
        assert node.acceleration < node.profiles[name]['linear_accel']
        assert node.deceleration == .85


@pytest.mark.parametrize('previous,target,expected', [
    ([0.,0.], [4.,0.], [.15,0.]),
    ([1.,0.], [4.,0.], [1.15,0.]),
    ([1.,0.], [0.,0.], [.9575,0.]),
    ([1.,0.], [-4.,0.], [.9575,0.]),
    ([1.,0.], [1.,2.], [1.,.0425]),
])
def test_fast_acceleration_does_not_speed_up_braking_or_lateral_corrections(previous,target,expected):
    node = SimpleNamespace(command=np.array([*previous,0.]), period=.05,
                           acceleration=3., deceleration=.85, yaw_acceleration=2.)
    result = TrajectoryTracker._rate_limit(node, *target, 0.)
    assert result[:2] == pytest.approx(expected)


def test_fast_plan_shortens_launch_but_keeps_braking_distance():
    x = np.linspace(0.,8.,161)
    kwargs = dict(points=np.c_[x,np.zeros_like(x)], yaws=np.zeros_like(x),
                  speed_limits=np.full_like(x,4.), lateral_acceleration=1.2,
                  entry_speed=0., deceleration=.85)
    slow = Trajectory(acceleration=.85, **kwargs)
    fast = Trajectory(acceleration=3., **kwargs)
    assert fast.duration < slow.duration
    ds = np.diff(fast.arclength)
    acceleration = np.diff(fast.speed**2)/(2*ds)
    assert max(acceleration) <= 3.+1.e-8
    assert min(acceleration) >= -.85-1.e-8
    assert fast.speed[-1] == 0.
    # The last metre must obey the same stop envelope regardless of launch.
    assert fast.speed[-20:] == pytest.approx(slow.speed[-20:])
