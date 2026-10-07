"""Hardware-facing regressions: stale wheel feedback and settling after arrival."""
import math
from collections import deque
from types import SimpleNamespace
import numpy as np
import pytest
from nav_msgs.msg import Odometry
from omni_autonomy_next import trajectory_tracker_node as tracker
from test_smooth_arrival import make_node, TRACKER_DEFAULTS


def wheel_message(stamp, velocity=(0., 0., 0.)):
    message = Odometry()
    message.header.frame_id = 'odom'
    message.child_frame_id = 'base_link'
    message.header.stamp.sec = int(stamp)
    message.header.stamp.nanosec = int(round((stamp-int(stamp))*1.e9))
    message.twist.twist.linear.x = float(velocity[0])
    message.twist.twist.linear.y = float(velocity[1])
    message.twist.twist.angular.z = float(velocity[2])
    return message


def clock_at(node, monkeypatch, now):
    monkeypatch.setattr(tracker.time, 'monotonic', lambda: now)
    node.get_clock = lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=int(now*1.e9)))


def test_fresh_pose_cannot_keep_stale_wheel_velocity_in_the_feedback(monkeypatch):
    node = make_node(goal=(1., 0., 1.))
    clock_at(node, monkeypatch, 1.)
    tracker.TrajectoryTracker._build_trajectory(node, np.array([[0.,0.],[1.,0.]]), 1.)
    node.pose_stamp = 1.
    node.velocity[:] = [.6, .1, .8]
    node.velocity_stamp = .7
    tracker.TrajectoryTracker._tick(node)
    assert node.commands[-1] == (0.,0.,0.)
    assert node.statuses[-1] == 'ODOMETRY_STALE'
    # A fresh stopped-wheel sample restarts with zero velocity, not the stale
    # pre-dropout prediction. The reference re-enters at the current pose.
    tracker.TrajectoryTracker._on_odom(node, wheel_message(1.))
    tracker.TrajectoryTracker._tick(node)
    assert node.velocity == pytest.approx(np.zeros(3))
    assert node.statuses[-1] == 'TRACKING'
    assert 0. < node.commands[-1][0] <= .85*.05


@pytest.mark.parametrize('stamp,velocity', [(.6,(.1,0.,0.)), (1.1,(.1,0.,0.)),
                                        (1.,(float('nan'),0.,0.))])
def test_invalid_or_delayed_samples_do_not_refresh_feedback(monkeypatch, stamp, velocity):
    node = make_node()
    clock_at(node, monkeypatch, 1.)
    node.velocity_stamp = .5
    tracker.TrajectoryTracker._on_odom(node, wheel_message(stamp, velocity))
    assert node.velocity_stamp == .5


def test_filter_uses_sample_intervals_in_a_callback_burst(monkeypatch):
    node = make_node()
    node.velocity_stamp = None
    for stamp, arrival, velocity in [(1.,1.,(0.,0.,0.)), (1.02,1.09,(1.,0.,0.)),
                                    (1.04,1.091,(1.,0.,0.))]:
        clock_at(node, monkeypatch, arrival)
        tracker.TrajectoryTracker._on_odom(node, wheel_message(stamp, velocity))
    assert node.velocity[0] == pytest.approx(1.-math.exp(-.04/.06))
    before = node.velocity.copy()
    tracker.TrajectoryTracker._on_odom(node, wheel_message(1.02, (-1.,0.,0.)))
    assert node.velocity == pytest.approx(before)
    assert node.velocity_stamp == pytest.approx(1.04)


def test_replanning_does_not_keep_a_constant_radius_orbit_alive(monkeypatch):
    node = make_node(goal=(0.,0.,0.))
    for step in range(140):
        now = .05*step
        clock_at(node, monkeypatch, now)
        node.pose = np.array([.06*math.cos(now), .06*math.sin(now), 0.])
        node.velocity_stamp = node.pose_stamp = now
        node.velocity = np.array([-.06*math.sin(now), .06*math.cos(now), 0.])
        if step % 20 == 0:
            tracker.TrajectoryTracker._build_trajectory(node, np.array([node.pose[:2],[0.,0.]]), 0.)
        tracker.TrajectoryTracker._tick(node)
    assert node.statuses[-1] in ('TERMINAL_TIMEOUT','TERMINAL_HOLD_EXPIRED')
    assert node.commands[-1] == (0.,0.,0.)


@pytest.mark.parametrize('gain,delay,tau', [(1.,.12,.08), (2.8,.12,.08),
    (2.8,.2,.12), (2.8,.3,.15), (3.2,.2,.12)])
@pytest.mark.parametrize('direction', [0., math.pi/4, math.pi/2])
def test_stays_at_goal_after_arrival_with_uncertain_drive(monkeypatch, gain, delay, tau, direction):
    goal = np.array([2.*math.cos(direction), 2.*math.sin(direction), math.pi/2])
    node = make_node(pose=(0.,0.,.1), goal=goal)
    truth = node.pose.copy()
    actual = np.zeros(3)
    pipeline = deque([np.zeros(3) for _ in range(round(delay/.01))])
    rng = np.random.default_rng(13)
    tail = []
    for step in range(2000):
        now = step*.01
        clock_at(node, monkeypatch, now)
        node.pose = truth.copy()
        node.pose[2] += rng.normal(0.,.0025)
        node.velocity_stamp = node.pose_stamp = now
        if step % 2 == 0:
            sample = actual + rng.normal(0., [.02,.02,.04])
            node.velocity += (1.-math.exp(-.02/TRACKER_DEFAULTS['velocity_filter_sec']))*(sample-node.velocity)
        if step % 100 == 0:
            tracker.TrajectoryTracker._build_trajectory(node, np.array([node.pose[:2],goal[:2]]), goal[2])
        if step % 5 == 0:
            tracker.TrajectoryTracker._tick(node)
        pipeline.append(node.command.copy())
        actual += (pipeline.popleft()*gain-actual)*(.01/tau)
        c,s = math.cos(truth[2]),math.sin(truth[2])
        truth += .01*np.array([actual[0]*c-actual[1]*s,actual[0]*s+actual[1]*c,actual[2]])
        if now >= 15.:
            tail.append((np.linalg.norm(truth[:2]-goal[:2]), abs(tracker.wrap(truth[2]-goal[2]))))
    tail = np.array(tail)
    assert np.max(tail[:,0]) < .04, (gain,delay,direction,tail.max(axis=0),node.statuses[-1])
    assert np.max(tail[:,1]) < .035, (gain,delay,direction,tail.max(axis=0),node.statuses[-1])
