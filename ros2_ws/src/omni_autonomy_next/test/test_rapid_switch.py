"""Early turn capture must reserve space for braking and recheck the stop."""
import math
from types import SimpleNamespace

import numpy as np
import pytest

from omni_autonomy_next import staged_heading_node as staged
from omni_autonomy_next import trajectory_tracker_node as tracker
from omni_autonomy_next.staged_heading import HeadingStage
from test_staged_heading import staged_node


def approaching():
    node = staged_node(pose=(-.09, 0., 0.))
    node.heading_stage = HeadingStage(0., math.pi/2, gate=np.zeros(2), phase='APPROACH')
    return node


def test_open_region_stops_early_and_builds_departure_from_actual_stop(monkeypatch):
    node = approaching()
    assert node._stage_tick(0., node.pose, np.array([.05, 0., 0.]), 1.)
    assert node.heading_stage.phase == 'SETTLE'
    assert node.command == pytest.approx([0., 0., 0.])
    assert node.heading_stage.gate == pytest.approx([-.09, 0.])
    # Model uncertain transport/braking travel larger than nominal 25 mm.
    node.pose[0] = -.045
    node._stage_tick(.3, node.pose, np.zeros(3), 1.)
    assert node.heading_stage.phase == 'SETTLE'
    node._stage_tick(.35, node.pose, np.zeros(3), 1.)
    assert node.heading_stage.phase == 'ROTATE'
    assert node.heading_stage.gate == pytest.approx([-.045, 0.])
    assert node.command[2] > 0.
    monkeypatch.setattr(tracker.time, 'monotonic', lambda: .35)
    tracker.TrajectoryTracker._build_trajectory(node, np.array([[0.,0.],[2.,0.]]), math.pi/2)
    assert node.stage_continuation[0].points[0] == pytest.approx([-.045, 0.])


@pytest.mark.parametrize('failure', ['cad_margin', 'live_map', 'stopping_distance', 'command_speed'])
def test_early_capture_rejects_unreserved_or_unobserved_braking_space(monkeypatch, failure):
    node = approaching()
    measured = np.array([.07, 0., 0.])
    if failure == 'cad_margin':
        monkeypatch.setattr(staged, 'rotation_clearance', lambda *args: .19)
    elif failure == 'live_map':
        node._stage_live_turn_clear = lambda *args: False
    elif failure == 'command_speed':
        node.command[0] = .1
    else:
        original = node.get_parameter
        node.get_parameter = lambda key: SimpleNamespace(value=.5) if key == 'feedback_delay_sec' else original(key)
    assert not node._stage_tick(0., node.pose, measured, 1.)
    assert node.heading_stage.phase == 'APPROACH'
    assert node.heading_stage.gate == pytest.approx([0., 0.])


def test_tight_turn_retains_original_capture_near_nominal_gate(monkeypatch):
    node = approaching()
    node.pose[0] = -.015
    monkeypatch.setattr(staged, 'rotation_clearance', lambda *args: .14)
    node._stage_tick(0., node.pose, np.zeros(3), 1.)
    assert node.heading_stage.phase == 'SETTLE'
    assert node.heading_stage.settle_drift == .035


@pytest.mark.parametrize('phase, displacement', [('SETTLE', .101), ('ROTATE', .036)])
def test_braking_region_never_expands_rotation_drift_guard(phase, displacement):
    node = approaching()
    node._stage_tick(0., node.pose, np.zeros(3), 1.)
    node.heading_stage.phase = phase
    node.heading_stage.rotation_started = 0.
    node.pose[0] += displacement
    node._stage_tick(.3, node.pose, np.zeros(3), 1.)
    assert node.command == pytest.approx([0., 0., 0.])
    assert node.statuses[-1] == 'STAGED_BLOCKED'


def test_actual_stop_must_still_have_fresh_turn_clearance():
    node = approaching()
    node._stage_tick(0., node.pose, np.zeros(3), 1.)
    node._stage_live_turn_clear = lambda *args: False
    node._stage_tick(.3, node.pose, np.zeros(3), 1.)
    assert node.command == pytest.approx([0., 0., 0.])
    assert node.statuses[-1] == 'STAGED_WAITING_CLEARANCE'
