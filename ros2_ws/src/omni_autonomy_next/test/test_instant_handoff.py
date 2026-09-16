"""Turn-response estimation and latency regressions using the deployed servo."""
from collections import deque
import math
import numpy as np
import pytest
from omni_autonomy_next.staged_heading import HeadingStage, TurnResponse, wrap
from test_staged_heading import staged_node


def test_response_is_bounded_and_tracks_rotation_across_pi():
    estimate = TurnResponse()
    for step in range(81):
        now = step*.05
        estimate.update(now, wrap(3.+2.8*.2*max(0.,now-.35)), .2, .2)
    assert estimate.gain == pytest.approx(2.8, abs=.05)
    assert len(estimate.history) <= 10
    for step in range(81, 90):
        estimate.update(step*.05, wrap(3.+2.8*.2*(step*.05-.35)), .2, .2)
    assert 1. <= estimate.gain <= 3.5


def test_feedback_gap_does_not_integrate_a_stale_command():
    estimate = TurnResponse(gain=2.)
    estimate.update(0.,0.,1.,.2)
    estimate.update(.05,.05,1.,.2)
    estimate.update(4.,1.,1.,.2)
    assert estimate.requested == estimate.observed == 0.
    assert estimate.gain == 2.
    assert len(estimate.history) == 1


@pytest.mark.parametrize('command', [0., .001, float('nan')])
def test_uninformative_motion_cannot_raise_command_gain(command):
    estimate = TurnResponse()
    for step in range(30):
        estimate.update(step*.05, step*.001, command, .2)
    assert estimate.gain == 1.


@pytest.mark.parametrize('drive,delay,tau', [(1.,.12,.08), (2.8,.2,.12),
    (2.8,.3,.15), (1.,.3,.15), (3.2,.2,.12), (.7,.12,.08)])
@pytest.mark.parametrize('angle', [math.pi/2, -math.pi, .18, -.18])
def test_turn_finishes_without_slow_tail_or_departure_during_rotation(drive, delay, tau, angle):
    node = staged_node(goal=(2.,0.,angle))
    node.heading_stage = HeadingStage(0.,angle,gate=np.zeros(2),phase='ROTATE',rotation_started=0.)
    pipeline = deque([0.] * round(delay/.01))
    actual = 0.
    tail_start = None
    for step in range(1200):
        now = step*.01
        node.velocity[2] += (1.-math.exp(-.01/.06))*(actual-node.velocity[2])
        if tail_start is None and abs(wrap(angle-node.pose[2])) < .12:
            tail_start = now
        if step % 5 == 0:
            node._stage_tick(now,node.pose,node.velocity,1.)
            assert np.linalg.norm(node.command[:2]) == 0.
            if node.heading_stage.phase == 'TRANSLATE':
                assert abs(wrap(angle-node.pose[2])) < .008
                assert abs(node.velocity[2]) < .025
                break
        pipeline.append(node.command[2])
        actual += (pipeline.popleft()*drive-actual)*(.01/tau)
        node.pose[2] += actual*.01
    assert node.heading_stage.phase == 'TRANSLATE'
    if drive == 1. and delay == .12:
        # Previous finishing servo needed 2.49-2.84 s in these cases.
        assert now-tail_start < 2.2
    if drive == 2.8 and delay == .3:
        # Previous servo needed 5.29-7.66 s with the same delays.
        assert now-tail_start < 4.
