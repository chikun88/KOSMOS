"""Use the actual callbacks without requiring a display or ROS on the host."""
import ast
import json
import math
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import pytest


SOURCE = (Path(__file__).parents[1] / 'ros2_ws/src/omni_autonomy_next'
          / 'omni_autonomy_next/speed_gui_node.py')


def method(class_name, name):
    cls = next(node for node in ast.parse(SOURCE.read_text()).body
               if isinstance(node, ast.ClassDef) and node.name == class_name)
    node = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name)
    scope = {'json': json, 'math': math, 'time': time}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), 'exec'), scope)
    return scope[name]


@pytest.mark.parametrize('callback', ['_state_cb', '_navigation_state_cb', '_remembered_poses_cb'])
@pytest.mark.parametrize('invalid', ['null', '[]', '1', '"bad"'])
def test_malformed_status_does_not_crash_operator_bridge(callback, invalid):
    node = SimpleNamespace(lock=threading.Lock(), state={}, state_time=None,
                           navigation_state={}, remembered_pose_state={})
    method('GuiBridge', callback)(node, SimpleNamespace(data=invalid))
    assert node.state_time is None


@pytest.mark.parametrize('scale', ['bad', None, math.nan, math.inf, -1., 2.])
def test_invalid_scale_cannot_refresh_safety_receipt(scale):
    node = SimpleNamespace(lock=threading.Lock(), state={}, state_time=None)
    method('GuiBridge', '_state_cb')(node, SimpleNamespace(data=json.dumps({'applied_scale': scale})))
    assert node.state_time is None


@pytest.mark.parametrize('status', [
    {'armed': 'false'}, {'allowed': 'false'}, {'emergency_stop': []},
    {'tracking_ok': 1}, {'profile': []}, {'reason': {}},
])
def test_malformed_safety_flags_do_not_represent_armed_or_healthy_state(status):
    node = SimpleNamespace(lock=threading.Lock(), state={'armed': False}, state_time=None)
    method('GuiBridge', '_state_cb')(node, SimpleNamespace(
        data=json.dumps({'applied_scale': 0., **status})))
    assert node.state == {'armed': False}
    assert node.state_time is None


@pytest.mark.parametrize('status', [
    {'field_side': []}, {'state': {}}, {'reason': []},
    {'distance_remaining_m': 10**400}, {'distance_remaining_m': math.nan},
])
def test_invalid_navigation_fields_cannot_poison_the_panel_timer(status):
    node = SimpleNamespace(lock=threading.Lock(), navigation_state={'state': 'IDLE'})
    method('GuiBridge', '_navigation_state_cb')(node, SimpleNamespace(data=json.dumps(status)))
    assert node.navigation_state == {'state': 'IDLE'}


@pytest.mark.parametrize('status', [
    {'poses': {'A': 1}}, {'poses': {'A': {}}},
    {'poses': {'A': {'x': '1', 'y': 0., 'yaw': 0.}}},
    {'poses': {'A': {'x': math.nan, 'y': 0., 'yaw': 0.}}},
    {'poses': {'A': {'x': 10**400, 'y': 0., 'yaw': 0.}}},
    {'poses': {}, 'loading_field_poses': []},
    {'poses': {}, 'loading_field_poses': {'left': 1}},
    {'poses': {}, 'defaulted': [[]]}, {'poses': {}, 'field_side': []},
])
def test_invalid_remembered_fields_cannot_poison_pose_formatting(status):
    previous = {'state': 'READY', 'poses': {}}
    node = SimpleNamespace(lock=threading.Lock(), remembered_pose_state=previous)
    method('GuiBridge', '_remembered_poses_cb')(node, SimpleNamespace(data=json.dumps(status)))
    assert node.remembered_pose_state is previous


def test_valid_status_and_nullable_loading_calibrations_are_retained():
    node = SimpleNamespace(lock=threading.Lock(), state={}, state_time=None,
                           navigation_state={}, remembered_pose_state={})
    safety = {'applied_scale': .5, 'armed': False, 'allowed': False, 'reason': 'DISARMED'}
    method('GuiBridge', '_state_cb')(node, SimpleNamespace(data=json.dumps(safety)))
    assert node.state == safety
    assert node.state_time is not None
    poses = {'state': 'READY', 'field_side': 'left', 'defaulted': ['A'],
             'poses': {'A': {'x': 1., 'y': 0., 'yaw': 0.}},
             'loading_field_poses': {'left': None, 'right': {'x': 1., 'y': 0., 'yaw': 0.}}}
    method('GuiBridge', '_remembered_poses_cb')(node, SimpleNamespace(data=json.dumps(poses)))
    assert node.remembered_pose_state == poses


def test_estop_disarms_and_cancels_even_if_button_already_unchecked():
    events = []
    node = SimpleNamespace(
        arm=SimpleNamespace(setChecked=lambda value: events.append(('checkbox', value))),
        bridge=SimpleNamespace(set_arm=lambda value: events.append(('arm', value)),
                               cancel_goal=lambda: events.append(('cancel', True)),
                               set_estop=lambda value: events.append(('estop', value))))
    method('ControlPanel', '_estop')(node)
    assert events == [('checkbox', False), ('arm', False), ('cancel', True), ('estop', True)]
