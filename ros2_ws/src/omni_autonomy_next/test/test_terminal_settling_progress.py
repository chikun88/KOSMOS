"""Settling progress must begin after a fixed route's earlier gate passage."""
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from omni_autonomy_next import trajectory_tracker_node as tracker
from test_smooth_arrival import make_node


def tick_at(node, monkeypatch, now):
    monkeypatch.setattr(tracker.time, 'monotonic', lambda: now)
    node.pose_stamp = node.velocity_stamp = node.plan_stamp = now
    tracker.TrajectoryTracker._tick(node)


def returning_node(monkeypatch):
    """A passed its gate at 22 mm, then overshot before reference completion."""
    monkeypatch.setattr(tracker.time, 'monotonic', lambda: 0.)
    node = make_node(pose=(.238, 0., 0.), goal=(0., 0., 0.))
    tracker.TrajectoryTracker._build_trajectory(
        node, np.array([node.pose[:2], node.active_goal[:2]]), 0.)
    node.terminal_best_distance = .022
    node.reference_time = node.trajectory.duration
    return node


def test_gate_overshoot_can_converge_to_precise_arrival_with_replans(monkeypatch):
    node = returning_node(monkeypatch)
    payloads = []
    node.last_status = None
    node.get_logger = lambda: SimpleNamespace(info=lambda *a: None)
    node.status_pub = SimpleNamespace(
        publish=lambda message: payloads.append(json.loads(message.data)))
    node._status = lambda state, **values: tracker.TrajectoryTracker._status(
        node, state, **values)
    tick_at(node, monkeypatch, 0.)
    assert node.finished_at == 0.

    # Replay a measured, monotonic return at the speed seen in the full chain.
    # It takes more than the original three-second hold to reach 15 mm. Fresh
    # replans may replace the reference, but must retain measured progress.
    for step in range(1, 101):
        now = step * .05
        distance = .238 * math.exp(-.6 * now)
        node.pose[0] = distance
        node.velocity[0] = -.6 * distance
        monkeypatch.setattr(tracker.time, 'monotonic', lambda: now)
        if step % 20 == 0:
            tracker.TrajectoryTracker._build_trajectory(
                node, np.array([node.pose[:2], node.active_goal[:2]]), 0.)
        tick_at(node, monkeypatch, now)
        assert payloads[-1]['state'] not in (
            'TERMINAL_TIMEOUT', 'TERMINAL_HOLD_EXPIRED')
        if 3. < now < 4.:
            assert node.commands[-1][0] < 0.
    assert payloads[-1]['arrival']['ready']


def test_first_completion_seed_is_retained_by_same_endpoint_replanning(monkeypatch):
    node = returning_node(monkeypatch)
    tick_at(node, monkeypatch, 0.)
    for step in range(1, 3):
        now = float(step)
        monkeypatch.setattr(tracker.time, 'monotonic', lambda: now)
        node.pose[0] = .238 + .01 * step
        tracker.TrajectoryTracker._build_trajectory(
            node, np.array([node.pose[:2], node.active_goal[:2]]), 0.)
        node.reference_time = node.trajectory.duration
        tick_at(node, monkeypatch, now)
        assert node.finished_at == 0.
        assert node.terminal_best_distance == pytest.approx(.238)


@pytest.mark.parametrize('motion', ['stuck', 'worsening', 'orbit', 'noise'])
def test_no_improvement_still_expires_despite_fresh_replanning(monkeypatch, motion):
    node = returning_node(monkeypatch)
    node.pose[:2] = [.06, 0.]
    tick_at(node, monkeypatch, 0.)
    for step in range(1, 72):
        now = step * .05
        if motion == 'worsening':
            node.pose[:2] = [.06 + .001 * now, 0.]
        elif motion == 'orbit':
            node.pose[:2] = .06 * np.array([math.cos(now), math.sin(now)])
        elif motion == 'noise':
            node.pose[:2] = [.06 + .001 * math.sin(now * 7.), 0.]
        monkeypatch.setattr(tracker.time, 'monotonic', lambda: now)
        if step % 20 == 0:
            tracker.TrajectoryTracker._build_trajectory(
                node, np.array([node.pose[:2], node.active_goal[:2]]), 0.)
        node.reference_time = node.trajectory.duration
        tick_at(node, monkeypatch, now)
    assert node.finished_at == 0.
    assert node.statuses[-1] == 'TERMINAL_HOLD_EXPIRED'
    assert node.commands[-1] == pytest.approx((0., 0., 0.))


@pytest.mark.parametrize('gate', ['cancel', 'stale_wheels', 'stale_pose'])
def test_first_completion_does_not_bypass_immediate_safety_gates(monkeypatch, gate):
    node = returning_node(monkeypatch)
    tick_at(node, monkeypatch, 0.)
    monkeypatch.setattr(tracker.time, 'monotonic', lambda: .5)
    node.pose_stamp = node.velocity_stamp = node.plan_stamp = .5
    if gate == 'cancel':
        node.stage_goal_enabled = False
    elif gate == 'stale_wheels':
        node.velocity_stamp = 0.
    else:
        node.pose_stamp = 0.
    tracker.TrajectoryTracker._tick(node)
    assert node.commands[-1] == pytest.approx((0., 0., 0.))
    assert node.statuses[-1] == {
        'cancel': 'IDLE', 'stale_wheels': 'ODOMETRY_STALE',
        'stale_pose': 'POSE_STALE',
    }[gate]
