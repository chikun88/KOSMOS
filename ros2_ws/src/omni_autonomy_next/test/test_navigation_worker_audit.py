"""Real ROS controller regression for a speed-policy change during planning."""
from types import MethodType, SimpleNamespace
import numpy as np
import pytest

pytest.importorskip('rclpy')
from std_msgs.msg import String
from omni_autonomy_next import trajectory_tracker_node as tracker
from test_smooth_arrival import make_node


def test_profile_switch_during_planning_cannot_commit_an_old_speed_envelope(monkeypatch):
    node = make_node(goal=(3., 0., 0.))
    node.stage_revision = 0
    node.stage_continuation = object()
    node.profiles = {'precision': dict(linear=.2, lateral=.2, angular=.3, linear_accel=.2)}
    node._apply_profile = MethodType(tracker.TrajectoryTracker._apply_profile, node)
    node.get_logger = lambda: SimpleNamespace(info=lambda *a: None, warning=lambda *a: None)
    points = np.array([[0., 0.], [3., 0.]])
    node.last_plan = (points, 0.)
    original = tracker.direction_speed_limits
    def change_profile_while_building(*args, **kwargs):
        result = original(*args, **kwargs)
        tracker.TrajectoryTracker._on_safety_state(node, String(data='{"profile":"precision"}'))
        return result
    monkeypatch.setattr(tracker, 'direction_speed_limits', change_profile_while_building)
    tracker.TrajectoryTracker._build_trajectory(node, points, 0.)
    assert node.profile_name == 'precision' and node.stage_revision == 1
    assert node.trajectory is None and node.stage_continuation is None
    assert node.pending_plan is node.last_plan and node.plan_event.is_set()
