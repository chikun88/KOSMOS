"""Exercise the actual full-chain assertions without launching ROS or UDP."""
import ast
import math
from pathlib import Path

import pytest


SOURCE = Path(__file__).parents[1] / 'scripts/check_mu3_navigation_full_chain.py'
FUNCTIONS = {'pose_matches', 'current_goal_succeeded', 'arrival_evidence',
             'expected_right_active_goal', 'mirror_goal_is_current', 'mirror_evidence'}
scope = {'math': math}
body = [node for node in ast.parse(SOURCE.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS]
exec(compile(ast.Module(body=body, type_ignores=[]), str(SOURCE), 'exec'), scope)


def evidence(**changes):
    arguments = dict(
        status={'remembered_pose': 'BAKETU2', 'field_side': 'left',
                'state': 'SUCCEEDED', 'request_id': 'current'},
        name='BAKETU2', previous_request_id='earlier',
        start_pose=(0., 0., 0.), target={'x': 0., 'y': 0., 'yaw': 0.},
        finish_pose=(0., 0., 0.), nonzero_uart=False, plan_points=0,
        wheel_commands=[0, 0, 0, 0])
    arguments.update(changes)
    return scope['arrival_evidence'](**arguments)


def test_exact_current_pose_succeeds_without_artificial_motion_or_plan():
    assert evidence() == {'already_settled': True, 'settled_zero_uart': True}


def test_exact_pose_cannot_hide_unrequested_motion():
    with pytest.raises(AssertionError, match='unexpected exact-pose movement'):
        evidence(nonzero_uart=True)


def test_small_localization_correction_within_tolerance_can_settle():
    assert evidence(start_pose=(.004, 0., .003), nonzero_uart=True)['already_settled']


@pytest.mark.parametrize('changes', [
    {'nonzero_uart': False, 'plan_points': 20},
    {'nonzero_uart': True, 'plan_points': 1},
])
def test_real_displacement_requires_both_motion_and_multi_pose_plan(changes):
    with pytest.raises(AssertionError, match='missing displacement'):
        evidence(start_pose=(1., 0., 0.), **changes)


def test_loading_a_keeps_approach_and_reverse_maneuver_evidence():
    status = {'remembered_pose': 'A', 'field_side': 'left',
              'state': 'SUCCEEDED', 'request_id': 'current'}
    with pytest.raises(AssertionError, match='missing displacement'):
        evidence(status=status, name='A')
    assert not evidence(status=status, name='A', nonzero_uart=True,
                        plan_points=12)['already_settled']


@pytest.mark.parametrize('changes', [
    {'request_id': 'earlier'}, {'request_id': None},
    {'field_side': 'right'}, {'remembered_pose': 'BAKETU3'}, {'state': 'ACTIVE'},
])
def test_retained_or_other_goal_success_is_not_current_success(changes):
    status = {'remembered_pose': 'BAKETU2', 'field_side': 'left',
              'state': 'SUCCEEDED', 'request_id': 'current', **changes}
    with pytest.raises(AssertionError):
        evidence(status=status)


@pytest.mark.parametrize('pose', [(0.021, 0., 0.), (0., 0., .021),
                                  (math.nan, 0., 0.)])
def test_success_status_does_not_replace_measured_settlement(pose):
    with pytest.raises(AssertionError, match='unsettled pose'):
        evidence(finish_pose=pose)


@pytest.mark.parametrize('wheels', [[1, 0, 0, 0], [], [0, 0, 0]])
def test_success_requires_complete_stopped_gateway_wheel_telemetry(wheels):
    with pytest.raises(AssertionError, match='unsettled wheels'):
        evidence(wheel_commands=wheels)


def test_yaw_wrap_does_not_invent_required_rotation():
    assert evidence(start_pose=(0., 0., -math.pi), finish_pose=(0., 0., math.pi),
                    target={'x': 0., 'y': 0., 'yaw': math.pi})['already_settled']


def mirror(**changes):
    arguments = dict(
        status={'remembered_pose': 'BAKETU2', 'field_side': 'right',
                'state': 'SENDING', 'request_id': 'current', 'goal_stamp': [123, 456]},
        name='BAKETU2', previous_request_id='earlier', started=10., status_received_at=11.,
        goal={'pose': [.81, 1.34, -math.pi], 'frame_id': 'map',
              'stamp': [123, 456], 'received_at': 11.},
        left_target={'x': -.81, 'y': 1.34, 'yaw': 0., 'frame_id': 'map'})
    arguments.update(changes)
    return scope['mirror_evidence'](**arguments)


def test_right_goal_matches_full_independent_reflection_and_request_stamp():
    result = mirror()
    assert result['goal_pose'] == [.81, 1.34, -math.pi]
    assert result['goal_stamp'] == [123, 456]
    assert result['request_id'] == 'current'


@pytest.mark.parametrize('pose', [[.82, 1.34, -math.pi], [.81, 1.35, -math.pi],
                                  [.81, 1.34, -math.pi+.01]])
def test_positive_x_does_not_mask_wrong_mirrored_coordinate_or_heading(pose):
    with pytest.raises(AssertionError, match='incorrect mirror'):
        mirror(goal={'pose': pose, 'frame_id': 'map', 'stamp': [123, 456], 'received_at': 11.})


@pytest.mark.parametrize('changes', [
    {'request_id': 'earlier'}, {'request_id': None}, {'field_side': 'left'},
    {'remembered_pose': 'BAKETU3'}, {'goal_stamp': [123, 457]}, {'goal_stamp': [0, 0]},
])
def test_right_coordinate_cannot_pass_other_request_or_side_or_timestamp(changes):
    status = {'remembered_pose': 'BAKETU2', 'field_side': 'right', 'state': 'SENDING',
              'request_id': 'current', 'goal_stamp': [123, 456], **changes}
    with pytest.raises(AssertionError, match='unrelated active goal'):
        mirror(status=status)


@pytest.mark.parametrize('part', ['goal', 'status'])
def test_retained_goal_and_status_must_both_arrive_after_current_command(part):
    changes = ({'goal': {'pose': [.81, 1.34, -math.pi], 'frame_id': 'map',
                         'stamp': [123, 456], 'received_at': 9.}}
               if part == 'goal' else {'status_received_at': 9.})
    with pytest.raises(AssertionError, match='unrelated active goal'):
        mirror(**changes)


def test_right_pose_requires_original_coordinate_frame():
    with pytest.raises(AssertionError, match='mirror frame'):
        mirror(goal={'pose': [.81, 1.34, -math.pi], 'frame_id': 'odom',
                     'stamp': [123, 456], 'received_at': 11.})


def test_loading_a_reflects_then_checks_gate_not_final_dock():
    left = {'x': -1.8, 'y': 4.75, 'yaw': -math.pi/2, 'frame_id': 'map'}
    expected = scope['expected_right_active_goal']('A', left)
    assert expected['x'] == pytest.approx(1.8)
    assert expected['y'] == pytest.approx(4.50)
    assert expected['yaw'] == pytest.approx(-math.pi/2)
    status = {'remembered_pose': 'A', 'field_side': 'right', 'request_id': 'current',
              'goal_stamp': [123, 456]}
    assert mirror(name='A', status=status, left_target=left,
                  goal={'pose': [1.8, 4.50, -math.pi/2], 'frame_id': 'map',
                        'stamp': [123, 456], 'received_at': 11.})['field_side'] == 'right'
    with pytest.raises(AssertionError, match='incorrect mirror'):
        mirror(name='A', status=status, left_target=left,
               goal={'pose': [1.8, 4.75, -math.pi/2], 'frame_id': 'map',
                     'stamp': [123, 456], 'received_at': 11.})


def test_oblique_loading_gate_uses_reflected_heading_before_offset():
    expected = scope['expected_right_active_goal']('A',
        {'x': -2., 'y': 3., 'yaw': math.pi/6, 'frame_id': 'map'})
    assert expected['x'] == pytest.approx(2.-.25*math.sqrt(3)/2)
    assert expected['y'] == pytest.approx(3.125)
    assert expected['yaw'] == pytest.approx(5*math.pi/6)
