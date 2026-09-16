"""Sweep the complete connectors, including the intervening fixed buckets."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from omni_autonomy_next.bucket_transit import FixedBucketTransit
from omni_autonomy_next.configured_goals import load_configured_poses
from omni_autonomy_next.field_side import (
    apply_pose, mirror_goal_approaches, mirror_poses, mirror_waypoint_routes,
)
from omni_autonomy_next.rl_residual import CadClearanceModel
from omni_autonomy_next.route_approaches import (
    load_fixed_departures, load_fixed_goal_approaches,
    match_fixed_goal_approach,
    select_fixed_goal_approach, select_fixed_goal_departure,
    select_fixed_pose_departure,
)
from omni_autonomy_next.staged_heading import dense_path


CONFIG = Path(__file__).resolve().parents[1] / 'config'


@pytest.fixture(scope='module')
def geometry():
    model = CadClearanceModel.from_yaml(
        CONFIG/'field_planning.yaml', CONFIG/'competition_footprints.yaml')
    return (model, FixedBucketTransit.from_yaml(model, CONFIG/'routes.yaml'),
            load_configured_poses(CONFIG/'field_poses.yaml'),
            load_fixed_departures(CONFIG/'routes.yaml'),
            load_fixed_goal_approaches(CONFIG/'routes.yaml'))


def xy(pose):
    return pose['x'], pose['y']


@pytest.mark.parametrize('side', ['left', 'right'])
@pytest.mark.parametrize('origin,goal', [
    (origin, goal) for origin in map(str, range(8)) for goal in ('4', '5')
    if origin != goal
])
def test_all_numbered_starts_reach_bucket_lanes_without_crossing_fixed_objects(
        geometry, side, origin, goal):
    model, transit, poses, departures, approaches = geometry
    if side == 'right':
        poses, departures, approaches = (mirror_poses(poses),
            mirror_waypoint_routes(departures), mirror_goal_approaches(approaches))
    start, end = xy(poses[origin]), xy(poses[goal])
    _, arrival = select_fixed_goal_approach(approaches, goal, start)
    _, departure = select_fixed_pose_departure(departures, poses, start, .2, goal)
    if not departure:
        _, departure = select_fixed_goal_departure(
            approaches, poses, start, end, .2, goal)
    start = xy(departure[-1]) if departure else start
    end = xy(arrival[0]) if arrival else end
    gates = transit.select(start, end, side)
    # The old direct connectors hit bucket 2 from the start bay and bucket 1
    # on the 4 <-> 5 shuttle. Check the entire new polyline, not only its gates.
    points = dense_path([start] + [xy(p) for p in gates] + [end], .01)
    yaw = poses[goal]['yaw']
    margins = model.clearance_over_poses(points, np.full(len(points), yaw))
    assert min(margins) >= .043, (origin, goal, side, gates)
    if (origin, goal) in [('1', '5'), ('4', '5'), ('5', '4')]:
        assert gates
        bypass_x = 2.2 if origin == '1' else 1.55
        assert any(abs(p['x']) == bypass_x for p in gates)
    if (origin, goal) in [('4', '5'), ('5', '4')]:
        # Do not restore the old 4.85 m dogleg via x=+/-2.20. The same
        # swept-footprint margin admits this 3.04 m connector around bucket 1.
        assert np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)) < 3.10


@pytest.mark.parametrize('side', ['left', 'right'])
def test_retry_on_outer_corridor_keeps_only_forward_gates(geometry, side):
    _, transit, *_ = geometry
    sign = 1. if side == 'left' else -1.
    gates = transit.select((sign*-2.2, -.2), (sign*-.8, -1.55), side)
    assert gates
    assert all(p['y'] <= -.2 for p in gates)


@pytest.mark.parametrize('side', ['left', 'right'])
@pytest.mark.parametrize('reverse', [False, True])
def test_retreat_bucket_two_uses_short_upper_crossing(geometry, side, reverse):
    model, transit, *_ = geometry
    sign = 1. if side == 'left' else -1.
    start, end = (sign*-3.88, -.55), (sign*-.8, .5)
    if reverse:
        start, end = end, start
    gates = transit.select(start, end, side)
    vertices = [start] + [xy(p) for p in gates] + [end]
    points = dense_path(vertices, .01)
    # Previously 5.33 m via the opposite side of the central pillar.
    assert np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)) < 4.3
    assert all(p['y'] >= -.55 for p in gates)
    yaw = 0. if side == 'left' else np.pi
    for a, b in zip(vertices, vertices[1:]):
        assert transit._clear(a, b, yaw)
    margins = model.clearance_over_poses(points, np.full(len(points), yaw))
    assert min(margins) >= .043


def test_bridge_inserts_transit_between_departure_and_approach(geometry):
    from omni_autonomy_next.goal_bridge_node import GoalBridgeNode, configured_pose_message
    from builtin_interfaces.msg import Time
    _, transit, poses, departures, approaches = geometry
    node = SimpleNamespace(
        current_position=xy(poses['4']), configured_frame='map',
        bucket_transit=transit,
        get_parameter=lambda key: SimpleNamespace(value=.2),
        get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(to_msg=Time)),
        _side_geometry=lambda side: (poses, departures, approaches),
    )
    request = SimpleNamespace(use_fixed_routes=True, goal_id='5', field_side='left',
        pose=configured_pose_message(poses['5'], Time()))
    GoalBridgeNode._refresh_fixed_routes(node, request)
    points = [(p.pose.position.x, p.pose.position.y) for p in request.route_poses]
    assert points == [(-.8, .75), (-.8, .5), (-1.55, .15),
                      (-1.55, -1.25), (-.8, -1.55), (-.8, -1.95)]
    assert request.route_id == (
        'fixed_bucket_2_to_lower+fixed_bucket_transit+fixed_bucket_3_from_upper')


def test_unconnected_pose_preserves_nav2_fallback_and_does_not_invent_a_leg(geometry):
    _, transit, *_ = geometry
    assert transit.select(None, (-.8, -1.55)) == []
    assert transit.select((float('nan'), 0.), (-.8, -1.55)) == []
    assert transit.select((0., 0.), (-.8, -1.55)) == []


def test_waiting_for_nav2_does_not_repeat_expensive_geometry():
    from omni_autonomy_next.goal_bridge_node import GoalBridgeNode
    refreshed = []
    node = SimpleNamespace(
        pending_request=object(), send_future=None, active_request=None,
        retry_not_before=0., not_ready_reason='starting', last_waiting_log=float('inf'),
        _navigation_ready=lambda *args: False,
        _refresh_fixed_routes=refreshed.append,
        _publish_status=lambda *args, **kwargs: None,
    )
    GoalBridgeNode._try_send_pending(node)
    assert refreshed == []


# Exact saved positions in the operator's 2026-09-14 collision recording.
SAVED_BUCKETS = {
    'BAKETU2': dict(frame_id='map', x=-.8, y=1.2870131135691598,
                   yaw=-.011947584995234269),
    'BAKETU3': dict(frame_id='map', x=-.8009525183256301, y=-2.4165472773112677,
                   yaw=-.026030934367592913),
}


@pytest.mark.parametrize('side', ['left', 'right'])
@pytest.mark.parametrize('name,goal_id', [('BAKETU2', '4'), ('BAKETU3', '5')])
@pytest.mark.parametrize('renamed', [False, True])
def test_saved_goal_callback_uses_bucket_routes_without_changing_saved_target(
        geometry, side, name, goal_id, renamed):
    import math
    from builtin_interfaces.msg import Time
    from std_msgs.msg import String
    from omni_autonomy_next.goal_bridge_node import GoalBridgeNode
    _, transit, poses, departures, approaches = geometry
    if side == 'right':
        poses, departures, approaches = (mirror_poses(poses),
            mirror_waypoint_routes(departures), mirror_goal_approaches(approaches))
    target = apply_pose(SAVED_BUCKETS[name], side)
    selected_name = '射撃位置を保存' if renamed else name
    requests = []
    node = SimpleNamespace(
        field_side=side, configured_frame='map', current_position=xy(poses['1']),
        bucket_transit=transit,
        effective_remembered_poses=lambda: {selected_name: SAVED_BUCKETS[name]},
        _parse_remembered_request=GoalBridgeNode._parse_remembered_request,
        get_parameter=lambda key: SimpleNamespace(value=.2),
        get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(to_msg=Time)),
        _side_geometry=lambda side: (poses, departures, approaches),
    )
    def queue(request):
        GoalBridgeNode._refresh_fixed_routes(node, request)
        requests.append(request)
    node._queue_request = queue
    GoalBridgeNode._remembered_goal_cb(node, String(data=selected_name))
    assert len(requests) == 1
    request = requests[0]
    assert request.goal_id is None
    assert request.remembered_pose_name == selected_name
    assert request.pose.pose.position.x == target['x']
    assert request.pose.pose.position.y == target['y']
    assert request.pose.pose.orientation.z == pytest.approx(math.sin(target['yaw']/2))
    assert request.pose.pose.orientation.w == pytest.approx(math.cos(target['yaw']/2))
    assert 'fixed_bucket_transit' in request.route_id
    assert approaches[goal_id]['upper']['id'] in request.route_id
    assert request.route_poses[-1].pose.position.y == approaches[goal_id]['upper']['waypoints'][-1]['y']


@pytest.mark.parametrize('name,goal_id', [('BAKETU2', '4'), ('BAKETU3', '5')])
def test_saved_lower_approach_does_not_overshoot_calibrated_target(geometry, name, goal_id):
    _, _, poses, _, approaches = geometry
    target = xy(SAVED_BUCKETS[name])
    assert match_fixed_goal_approach(approaches, poses, target) == goal_id
    _, gates = select_fixed_goal_approach(approaches, goal_id,
        (-2.2, -4.), destination_position=target)
    assert gates
    assert all(gate['y'] <= target[1] for gate in gates)


@pytest.mark.parametrize('position', [(-2., 1.3), (-1.8, 4.1), (0., 0.), (float('nan'), 1.)])
def test_unrelated_saved_goals_are_not_assigned_bucket_lanes(geometry, position):
    _, _, poses, _, approaches = geometry
    assert match_fixed_goal_approach(approaches, poses, position) is None


@pytest.mark.parametrize('side', ['left', 'right'])
def test_async_preparation_survives_slow_geometry_and_localization_jitter(geometry, side):
    from concurrent.futures import Future
    from types import MethodType
    from builtin_interfaces.msg import Time
    from omni_autonomy_next.goal_bridge_node import (
        GoalBridgeNode, GoalRequest, configured_pose_message,
    )
    _, transit, poses, departures, approaches = geometry
    if side == 'right':
        poses, departures, approaches = (mirror_poses(poses),
            mirror_waypoint_routes(departures), mirror_goal_approaches(approaches))
    future = Future()
    submissions, sent, statuses = [], [], []
    ready = [True]
    def submit(fn, *args):
        submissions.append((fn, args))
        return future
    client = SimpleNamespace(send_goal_async=lambda goal, **kw: (
        sent.append(goal) or Future()))
    request = GoalRequest(configured_pose_message(poses['5'], Time()), 'bucket3',
                          goal_id='5', use_fixed_routes=True, field_side=side)
    node = SimpleNamespace(
        current_position=xy(poses['3']), configured_frame='map',
        bucket_transit=transit, route_executor=SimpleNamespace(submit=submit),
        get_parameter=lambda key: SimpleNamespace(value=.2),
        get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(to_msg=Time)),
        _side_geometry=lambda side: (poses, departures, approaches),
        _publish_status=lambda state, **kw: statuses.append(state),
        pending_request=request, active_request=None, send_future=None,
        retry_not_before=0., last_waiting_log=float('inf'),
        not_ready_reason='not active: controller_server',
        _navigation_ready=lambda *args: ready[0],
        action_client=client, route_action_client=client,
        route_behavior_tree='fixed.xml',
        active_goal_pub=SimpleNamespace(publish=lambda msg: None),
        _feedback_cb=lambda *args: None, _goal_response_cb=lambda *args: None,
        get_logger=lambda: SimpleNamespace(info=lambda msg: None),
    )
    node._refresh_fixed_routes = MethodType(GoalBridgeNode._refresh_fixed_routes, node)
    # A sweep taking longer than the lifecycle deadline must not occupy the
    # ROS executor or trigger a new sweep at every localized pose update.
    for tick in range(40):
        GoalBridgeNode._try_send_pending(node)
        node.current_position = (node.current_position[0]+.00001,
                                 node.current_position[1])
    assert len(submissions) == 1
    assert not sent
    assert statuses == ['PLANNING_ROUTE']
    fn, args = submissions[0]
    future.set_result(fn(*args))
    ready[0] = False
    GoalBridgeNode._try_send_pending(node)
    assert not sent
    assert statuses[-1] == 'WAITING_FOR_NAV2'
    ready[0] = True
    GoalBridgeNode._try_send_pending(node)
    assert len(submissions) == 1
    assert len(sent) == 1
    assert sent[0].behavior_tree == 'fixed.xml'
    assert len(sent[0].poses) > 2
    assert node.active_request is request
    assert node.pending_request is None


def test_cancel_during_route_preparation_never_sends_old_goal():
    from concurrent.futures import Future
    from omni_autonomy_next.goal_bridge_node import GoalBridgeNode
    future = Future()
    node = SimpleNamespace(
        pending_request=SimpleNamespace(route_preparation=(future,)),
        active_request=None, active_goal_handle=None, send_future=None,
        _publish_status=lambda *a, **kw: None,
    )
    GoalBridgeNode._request_cancel(node, explicit=True)
    assert future.cancelled()
    assert node.pending_request is None
    assert node.cancel_requested


def test_sparse_rejection_preserves_full_centimetre_sweep(geometry):
    model, transit, *_ = geometry
    # Include collision, zero length, narrow valid lanes and mirrored routes.
    for side in ('left', 'right'):
        sign = 1 if side == 'left' else -1
        yaw = 0 if side == 'left' else np.pi
        start = (sign*-3.879, -.551)
        for gate in transit.waypoints[side]:
            end = xy(gate)
            points = dense_path([start, end], .01)
            values = model.clearance_over_poses(points, np.full(len(points), yaw), cap=.15)
            reserve = .005 + model.radius*.02
            margin = np.minimum(
                values[0]-reserve-.015+.2*np.linalg.norm(points-points[0], axis=1),
                values[-1]-reserve-.015+.2*np.linalg.norm(points-points[-1], axis=1))
            expected = bool(np.all(values-reserve >= np.maximum(.025, np.minimum(.10, margin))-1.e-9))
            assert transit._clear(start, end, yaw) == expected
        assert transit._clear(start, start, yaw)


def test_new_goal_preempts_before_any_route_geometry():
    from concurrent.futures import Future
    from omni_autonomy_next.goal_bridge_node import GoalBridgeNode
    old_future = Future()
    replacement = SimpleNamespace(label='new bucket', goal_id='5')
    cancellations = []
    node = SimpleNamespace(
        pending_request=SimpleNamespace(route_preparation=(old_future,)),
        active_request=SimpleNamespace(goal_id='3'), send_future=None,
        _publish_status=lambda *a, **kw: None,
        get_logger=lambda: SimpleNamespace(info=lambda msg: None),
        _request_cancel=lambda **kw: cancellations.append(kw),
    )
    assert GoalBridgeNode._queue_request(node, replacement)
    assert old_future.cancelled()
    assert node.pending_request is replacement
    assert cancellations == [{'explicit': False}]


@pytest.mark.parametrize('state,age,action_ready,expected', [
    (3, .1, True, True), (3, 2., True, False),
    (2, .1, True, False), (3, .1, False, False),
])
def test_prepared_route_still_requires_fresh_active_nav2(state, age, action_ready, expected):
    from omni_autonomy_next.goal_bridge_node import GoalBridgeNode
    node = SimpleNamespace(
        pending_request=SimpleNamespace(route_poses=[object()]),
        _poll_lifecycle_states=lambda now: None,
        route_action_client=SimpleNamespace(server_is_ready=lambda: action_ready),
        lifecycle_clients={'collision_monitor': object()},
        lifecycle_states={'collision_monitor': state},
        lifecycle_state_times={'collision_monitor': 10.-age},
        get_parameter=lambda name: SimpleNamespace(value=1.5),
    )
    assert GoalBridgeNode._navigation_ready(node, 10.) is expected
