"""Regressions for half-turn direction chatter and delayed pose delivery."""
import math
import numpy as np
import pytest
from geometry_msgs.msg import PoseWithCovarianceStamped
from omni_autonomy_next import trajectory_tracker_node as tracker
from test_smooth_arrival import make_node
from test_hardware_tracking import clock_at


@pytest.mark.parametrize("direction", [-1., 1.])
def test_half_turn_noise_does_not_reverse_command(monkeypatch, direction):
    node = make_node(goal=(0., 0., math.pi))
    clock_at(node, monkeypatch, 0.)
    tracker.TrajectoryTracker._build_trajectory(node, np.array([[0., 0.]]), math.pi)
    for step in range(8):
        now = step * .05
        clock_at(node, monkeypatch, now)
        node.pose[2] = direction * (.003 if step % 2 == 0 else -.003)
        node.pose_stamp = node.velocity_stamp = now
        tracker.TrajectoryTracker._tick(node)
    rates = direction * np.array(node.commands)[:, 2]
    assert np.all(rates > 0.), rates
    assert rates[-1] > rates[0] + .2


def pose_message(stamp, x=0.):
    msg = PoseWithCovarianceStamped()
    msg.header.frame_id = 'map'
    msg.header.stamp.sec = int(stamp)
    msg.header.stamp.nanosec = int(round((stamp-int(stamp))*1.e9))
    msg.pose.pose.position.x = x
    msg.pose.pose.orientation.w = 1.
    return msg


@pytest.mark.parametrize('stamp,x', [(.5, 1.), (1.1, 1.), (1., float('nan'))])
def test_invalid_pose_does_not_refresh_feedback(monkeypatch, stamp, x):
    node = make_node()
    clock_at(node, monkeypatch, 1.)
    tracker.TrajectoryTracker._on_pose(node, pose_message(stamp, x))
    assert node.pose_stamp == 0.
    assert np.all(node.pose == 0.)


def test_pose_burst_cannot_move_feedback_backwards(monkeypatch):
    node = make_node()
    clock_at(node, monkeypatch, 1.)
    tracker.TrajectoryTracker._on_pose(node, pose_message(.95, .1))
    tracker.TrajectoryTracker._on_pose(node, pose_message(.90, -.1))
    assert node.pose[0] == pytest.approx(.1)
    assert node.pose_stamp == pytest.approx(.95)


def test_odometry_recovery_without_pose_stops_cleanly(monkeypatch):
    node = make_node(goal=(1., 0., 0.))
    clock_at(node, monkeypatch, 1.)
    tracker.TrajectoryTracker._build_trajectory(node, np.array([[0.,0.],[1.,0.]]), 0.)
    node.odometry_paused = True
    node.velocity_stamp = 1.
    node.pose = None
    tracker.TrajectoryTracker._tick(node)
    assert node.commands[-1] == (0.,0.,0.)
    assert node.statuses[-1] == 'POSE_STALE'


@pytest.mark.parametrize('gain,delay,tau', [(1.,.12,.08), (2.8,.3,.15)])
def test_half_turn_settles_with_drive_delay(monkeypatch, gain, delay, tau):
    from collections import deque
    node = make_node(goal=(0.,0.,math.pi))
    clock_at(node, monkeypatch, 0.)
    tracker.TrajectoryTracker._build_trajectory(node, np.array([[0.,0.]]), math.pi)
    pipeline = deque([0.] * round(delay/.01))
    yaw, rate = 0., 0.
    tail = []
    for step in range(2000):
        now = step*.01
        clock_at(node, monkeypatch, now)
        node.pose[2] = tracker.wrap(yaw + (.003 if step % 2 else -.003))
        node.pose_stamp = node.velocity_stamp = node.plan_stamp = now
        node.velocity[2] += (1.-math.exp(-.01/.06))*(rate-node.velocity[2])
        if step % 5 == 0:
            tracker.TrajectoryTracker._tick(node)
        pipeline.append(node.command[2])
        rate += (gain*pipeline.popleft()-rate)*(.01/tau)
        yaw += .01*rate
        if now > 15.:
            tail.append(abs(tracker.wrap(math.pi-yaw)))
    assert max(tail) < .035
