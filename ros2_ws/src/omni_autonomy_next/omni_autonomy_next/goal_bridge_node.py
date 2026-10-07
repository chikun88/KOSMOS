"""Execute RViz or preconfigured goals through Nav2 with result monitoring."""

import copy
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import time

from action_msgs.msg import GoalStatus
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from lifecycle_msgs.msg import State
from lifecycle_msgs.srv import GetState
from nav2_msgs.action import NavigateThroughPoses, NavigateToPose
import rclpy
from rclpy.action import ActionClient
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, String

from .configured_goals import load_configured_poses, load_remembered_defaults
from .field_side import (
    LEFT,
    apply_pose,
    mirror_goal_approaches,
    mirror_waypoint_routes,
    mirror_poses,
    normalize_side,
    store_pose,
)
from .reverse_approach import reverse_gate, reverse_target, TIMEOUT as REVERSE_TIMEOUT
from .rl_residual import CadClearanceModel
from .remembered_poses import (
    load_remembered_poses,
    normalize_pose_name,
    save_remembered_poses,
    resolve_remembered_pose,
    remember_field_pose,
)
from .bucket_transit import FixedBucketTransit
from .source_freshness import message_stamp_nanoseconds
from .route_approaches import (
    load_fixed_departures,
    load_fixed_goal_approaches,
    match_fixed_goal_approach,
    select_fixed_goal_approach,
    select_fixed_goal_departure,
    select_fixed_pose_departure,
)


def pose_is_valid(message: PoseStamped) -> bool:
    pose = message.pose
    values = (
        pose.position.x,
        pose.position.y,
        pose.position.z,
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w,
    )
    quaternion_norm = math.hypot(*values[3:])
    return (
        bool(message.header.frame_id)
        and all(math.isfinite(value) for value in values)
        and math.isfinite(quaternion_norm)
        and quaternion_norm > 1.0e-6
    )


def configured_pose_message(configured, stamp):
    message = PoseStamped()
    message.header.frame_id = configured['frame_id']
    if stamp is not None:
        message.header.stamp = stamp
    message.pose.position.x = configured['x']
    message.pose.position.y = configured['y']
    half_yaw = 0.5 * configured['yaw']
    message.pose.orientation.z = math.sin(half_yaw)
    message.pose.orientation.w = math.cos(half_yaw)
    return message


@dataclass
class GoalRequest:
    pose: PoseStamped
    label: str
    goal_id: str | None = None
    attempts: int = 0
    route_id: str | None = None
    route_poses: list[PoseStamped] = field(default_factory=list)
    use_fixed_routes: bool = False
    remembered_pose_name: str | None = None
    reverse_final_pose: PoseStamped | None = None
    reversing: bool = False
    # Which field this request was planned for. Captured per request so a side
    # change that arrives mid-flight cannot re-target the goal already running.
    field_side: str = LEFT
    routes_prepared: bool = False
    route_preparation: tuple | None = None
    goal_stamp: list[int] | None = None
    request_id: str | None = None


class GoalBridgeNode(Node):
    """Validated goal executor with bounded retries and observable status."""

    def __init__(self) -> None:
        super().__init__('goal_bridge')
        share = get_package_share_directory('omni_autonomy_next')
        self.declare_parameter('field_config_file', str(Path(share) / 'config/field_planning.yaml'))
        self.declare_parameter('footprint_config_file', str(Path(share) / 'config/competition_footprints.yaml'))
        self.reverse_clearance = CadClearanceModel.from_yaml(
            self.get_parameter('field_config_file').value,
            self.get_parameter('footprint_config_file').value)
        self.declare_parameter('input_topic', '/goal_request')
        self.declare_parameter('goal_id_topic', '/navigation/goal_id_request')
        self.declare_parameter(
            'remember_pose_topic', '/navigation/remember_pose_request'
        )
        self.declare_parameter(
            'remembered_goal_topic', '/navigation/remembered_goal_request'
        )
        self.declare_parameter(
            'remembered_poses_topic', '/navigation/remembered_poses'
        )
        self.declare_parameter('field_side_topic', '/navigation/field_side')
        self.declare_parameter(
            'remembered_poses_file',
            str(Path.home() / '.ros' / 'omni_autonomy_next' / 'remembered_poses.json'),
        )
        self.declare_parameter('remember_pose_max_age_sec', 1.0)
        self.declare_parameter('verify_tracker_arrival', False)
        self.verify_tracker_arrival = bool(self.get_parameter('verify_tracker_arrival').value)
        self.finalizing_since = None
        self.tracker_arrival = None
        self.tracker_arrival_stamp = -math.inf
        self.declare_parameter('cancel_topic', '/navigation/cancel_request')
        self.declare_parameter('status_topic', '/navigation/goal_status')
        self.declare_parameter('action_name', '/navigate_to_pose')
        self.declare_parameter('route_action_name', '/navigate_through_poses')
        self.declare_parameter(
            'route_behavior_tree',
            share + '/behavior_trees/follow_fixed_approach.xml',
        )
        self.declare_parameter(
            'field_poses_file', share + '/config/field_poses.yaml'
        )
        self.declare_parameter('routes_file', share + '/config/routes.yaml')
        self.declare_parameter('pose_topic', '/localization/pose')
        # If a goal is selected again while already sitting on it, do not drive
        # away to its entry gate and return.
        self.declare_parameter('route_bypass_distance_m', 0.20)
        self.declare_parameter('startup_goal_id', '')
        self.declare_parameter('max_retries', 2)
        self.declare_parameter('retry_delay_sec', 1.0)
        self.declare_parameter('cancel_request_timeout_sec', 1.0)
        self.declare_parameter('active_goal_topic',
                               '/navigation/active_goal')
        self.declare_parameter('lifecycle_poll_period_sec', 0.50)
        self.declare_parameter('lifecycle_state_max_age_sec', 1.50)
        # An unanswered rclpy service future stays pending forever.  A Nav2
        # lifecycle node that is busy transitioning drops the response and logs
        # "client will not receive response"; without a deadline the request
        # slot for that node is never freed, so it is never polled again,
        # _navigation_ready() stays false and every goal the operator selects
        # sits in the queue for the rest of the session.  Reproduced on egg8:
        # velocity_smoother timed out one get_state during bringup and goals
        # 4, 5 and 2 were all logged as queued and never executed.
        self.declare_parameter('lifecycle_request_timeout_sec', 1.0)
        self.declare_parameter('required_lifecycle_nodes', [
            'controller_server', 'planner_server', 'bt_navigator',
            'velocity_smoother', 'collision_monitor',
        ])

        input_topic = str(self.get_parameter('input_topic').value)
        goal_id_topic = str(self.get_parameter('goal_id_topic').value)
        remember_pose_topic = str(
            self.get_parameter('remember_pose_topic').value
        )
        remembered_goal_topic = str(
            self.get_parameter('remembered_goal_topic').value
        )
        cancel_topic = str(self.get_parameter('cancel_topic').value)
        action_name = str(self.get_parameter('action_name').value)
        route_action_name = str(
            self.get_parameter('route_action_name').value
        )
        self.route_behavior_tree = str(
            self.get_parameter('route_behavior_tree').value
        )
        field_poses_file = self.get_parameter('field_poses_file').value
        # Everything loaded here is the LEFT-field original. The right field is
        # the reflection about x=0 and is derived on demand, so a single edited
        # coordinate can never be applied to one field and forgotten on the
        # other.
        self.left_configured_poses = load_configured_poses(field_poses_file)
        self.configured_frame = next(
            iter(self.left_configured_poses.values())
        )['frame_id']
        self.remembered_pose_defaults = load_remembered_defaults(
            field_poses_file, self.left_configured_poses
        )
        self.remembered_poses_file = str(
            self.get_parameter('remembered_poses_file').value
        )
        try:
            self.remembered_poses = load_remembered_poses(
                self.remembered_poses_file
            )
        except (OSError, ValueError, UnicodeError) as error:
            self.get_logger().error(
                f'Could not load remembered poses from '
                f'{self.remembered_poses_file}: {error}'
            )
            self.remembered_poses = {}
        routes_file = self.get_parameter('routes_file').value
        self.bucket_transit = FixedBucketTransit.from_yaml(
            self.reverse_clearance, routes_file)
        self.left_fixed_departures = load_fixed_departures(routes_file)
        self.left_fixed_goal_approaches = load_fixed_goal_approaches(routes_file)
        self.right_configured_poses = mirror_poses(self.left_configured_poses)
        self.right_fixed_departures = mirror_waypoint_routes(
            self.left_fixed_departures
        )
        self.right_fixed_goal_approaches = mirror_goal_approaches(
            self.left_fixed_goal_approaches
        )
        self.field_side = LEFT
        self.configured_poses = self.left_configured_poses
        self.fixed_departures = self.left_fixed_departures
        self.fixed_goal_approaches = self.left_fixed_goal_approaches
        self.action_client = ActionClient(self, NavigateToPose, action_name)
        self.route_action_client = ActionClient(
            self, NavigateThroughPoses, route_action_name
        )
        self.current_position = None
        self.current_pose = None
        self.current_pose_received_at = -math.inf
        self.pending_request = None
        self.route_executor = ThreadPoolExecutor(max_workers=1)
        self.active_request = None
        self.active_goal_handle = None
        self.send_future = None
        self.cancel_future = None
        self.cancel_accepted = False
        self.cancel_requested_at = -math.inf
        self.cancel_retry_not_before = -math.inf
        self.result_future = None
        self.result_monitor_cancel = False
        self.result_retry_not_before = -math.inf
        self.cancel_requested = False
        self.retry_not_before = -math.inf
        self.not_ready_reason = ''
        self.last_waiting_log = -math.inf

        lifecycle_nodes = [
            str(name).strip()
            for name in self.get_parameter('required_lifecycle_nodes').value
            if str(name).strip()
        ]
        self.lifecycle_clients = {
            name: self.create_client(GetState, f'/{name}/get_state')
            for name in lifecycle_nodes
        }
        self.lifecycle_futures = {name: None for name in lifecycle_nodes}
        self.lifecycle_states = {name: None for name in lifecycle_nodes}
        self.lifecycle_state_times = {name: -math.inf for name in lifecycle_nodes}
        self.lifecycle_request_times = {
            name: -math.inf for name in lifecycle_nodes
        }
        self.lifecycle_timeouts = {name: 0 for name in lifecycle_nodes}
        self.last_lifecycle_poll = -math.inf

        status_qos = QoSProfile(depth=1)
        status_qos.reliability = ReliabilityPolicy.RELIABLE
        status_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        # 実行中の目標姿勢そのものを出す。/plan の終端は目標ではない:
        # SmacPlanner2D は tolerance 0.20 m を持つので、目標セルが
        # 占有されているとき（地点4〜7は余裕47〜59 mm）最大 0.20 m 手前で
        # 終わる経路を返す。MPPI は GoalCritic で真の目標を見るが、経路を
        # 追う制御器は経路しか知らないため、そのぶんずれて停まる。
        self.active_goal_pub = self.create_publisher(
            PoseStamped,
            str(self.get_parameter('active_goal_topic').value),
            QoSProfile(
                depth=1, reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL),
        )
        self.status_pub = self.create_publisher(
            String, str(self.get_parameter('status_topic').value), status_qos
        )
        self.create_subscription(String, '/trajectory_tracker/status',
                                 self._tracker_status_cb, status_qos)
        self.remembered_poses_pub = self.create_publisher(
            String,
            str(self.get_parameter('remembered_poses_topic').value),
            status_qos,
        )
        self.create_subscription(PoseStamped, input_topic, self._goal_cb, 5)
        self.create_subscription(String, goal_id_topic, self._goal_id_cb, 5)
        self.create_subscription(
            String, remember_pose_topic, self._remember_pose_cb, 5
        )
        self.create_subscription(
            String, remembered_goal_topic, self._remembered_goal_cb, 5
        )
        # Latched: a panel or radio bridge that starts later must still learn
        # which field is selected before the operator sends the first goal.
        self.create_subscription(
            String, str(self.get_parameter('field_side_topic').value),
            self._field_side_cb, status_qos,
        )
        self.create_subscription(Bool, cancel_topic, self._cancel_cb, 5)
        self.create_subscription(
            PoseWithCovarianceStamped,
            str(self.get_parameter('pose_topic').value), self._pose_cb, 5
        )
        self.create_timer(0.05, self._try_send_pending)
        self.get_logger().info(
            f'Goal executor ready: RViz={input_topic}, preset={goal_id_topic}, '
            f'remember={remember_pose_topic}, recall={remembered_goal_topic}, '
            f'cancel={cancel_topic}, action={action_name}, '
            f'fixed-route-action={route_action_name}'
        )
        self._publish_remembered_poses('READY')

        startup_goal_id = str(self.get_parameter('startup_goal_id').value).strip()
        if startup_goal_id:
            self._queue_configured_goal(startup_goal_id, source='startup')
        else:
            self._publish_status('IDLE')

    def _publish_status(self, state, request=None, **details):
        request = request or self.active_request or self.pending_request
        data = {
            'state': state,
            'goal_id': request.goal_id if request else None,
            'label': request.label if request else None,
            'attempt': request.attempts if request else 0,
            'route_id': request.route_id if request else None,
            'route_waypoints': len(request.route_poses) if request else 0,
            'remembered_pose': (
                request.remembered_pose_name if request else None
            ),
            'field_side': request.field_side if request else self.field_side,
            'goal_stamp': getattr(request, 'goal_stamp', None),
            'request_id': getattr(request, 'request_id', None),
            **details,
        }
        self.status_pub.publish(String(data=json.dumps(data, separators=(',', ':'))))

    def effective_remembered_poses(self):
        """Recorded poses win; unrecorded slots fall back to their seed."""
        return {**self.remembered_pose_defaults, **self.remembered_poses}

    def _publish_remembered_poses(self, state, **details):
        data = {
            'state': state,
            'poses': {name: resolve_remembered_pose(pose, self.field_side)
                      for name, pose in self.effective_remembered_poses().items()},
            'loading_calibrated_fields': sorted(
                self.remembered_poses.get('A', {}).get('field_poses', {})),
            'loading_field_poses': self.remembered_poses.get('A', {}).get('field_poses', {}),
            # Names still sitting on their configured seed. The operator has to
            # be able to see which points have never been measured in place.
            'defaulted': sorted(
                name for name in self.remembered_pose_defaults
                if name not in self.remembered_poses
            ),
            'field_side': self.field_side,
            **details,
        }
        self.remembered_poses_pub.publish(
            String(data=json.dumps(data, ensure_ascii=False, separators=(',', ':')))
        )

    def _side_geometry(self, side):
        """Return (configured poses, departures, approaches) for one field."""
        if normalize_side(side) == LEFT:
            return (self.left_configured_poses, self.left_fixed_departures,
                    self.left_fixed_goal_approaches)
        return (self.right_configured_poses, self.right_fixed_departures,
                self.right_fixed_goal_approaches)

    def _set_field_side(self, side, source, cancel_active=True):
        """Switch fields, abandoning any motion planned for the previous one.

        ``cancel_active`` is false only when the caller queues a replacement
        goal in the same callback: an explicit cancel there would clear the
        pending slot the replacement is about to occupy.
        """
        side = normalize_side(side)
        if side == self.field_side:
            return False
        moving = (
            self.active_request is not None
            or self.pending_request is not None
            or self.send_future is not None
        )
        self.field_side = side
        (self.configured_poses, self.fixed_departures,
         self.fixed_goal_approaches) = self._side_geometry(side)
        self.get_logger().warning(
            f'Field side changed to {side} by {source}'
        )
        canceled = moving and cancel_active
        if canceled:
            # The running goal was planned in the other field. Continuing it
            # would drive the robot across the divider.
            self._request_cancel(explicit=True)
        self._publish_remembered_poses('FIELD_SIDE')
        self._publish_status('FIELD_SIDE_CHANGED', field_side=side,
                             canceled_active=canceled)
        return True

    def _field_side_cb(self, message: String) -> None:
        try:
            self._set_field_side(message.data, 'field_side topic')
        except ValueError as error:
            self.get_logger().error(f'Ignoring field side request: {error}')
            self._publish_remembered_poses(
                'INVALID_FIELD_SIDE', detail=str(error)
            )

    def _queue_request(self, request):
        if self.pending_request is not None:
            preparation = self.pending_request.route_preparation
            if preparation is not None:
                preparation[0].cancel()
        if self.active_request is not None or self.send_future is not None:
            # Operator selection is latest-wins. Rejecting a GUI click as BUSY
            # leaves the panel showing the new number while the robot continues
            # toward the old one. Queue the replacement and cancel the active
            # Nav2 action first so two goals can never execute concurrently.
            self.pending_request = request
            self.get_logger().info(
                f'Replacing active goal with {request.label}'
            )
            self._publish_status(
                'PREEMPTING', request=request,
                active_goal_id=(
                    self.active_request.goal_id if self.active_request else None
                ),
                cancel_goal_stamp=getattr(self.active_request, 'goal_stamp', None),
            )
            self._request_cancel(explicit=False)
            return True
        if self.pending_request is not None:
            self.get_logger().info(
                f'Replacing queued goal with {request.label}'
            )
        self.pending_request = request
        self.cancel_requested = False
        self.retry_not_before = time.monotonic()
        self._publish_status('QUEUED', request=request)
        self._try_send_pending()
        return True

    def _refresh_fixed_routes(self, request) -> bool:
        """Prepare one route snapshot without blocking ROS service responses."""
        if not request.use_fixed_routes or getattr(request, 'routes_prepared', False):
            return True
        preparation = getattr(request, 'route_preparation', None)
        if preparation is not None:
            future, departure_id, departure_points, approach_id, approach_points = preparation
            if not future.done():
                return False
            transit_points = future.result()
            request.route_preparation = None
            GoalBridgeNode._set_fixed_route(self, request, departure_id, departure_points,
                                           approach_id, approach_points, transit_points)
            return True
        destination = (
            float(request.pose.pose.position.x),
            float(request.pose.pose.position.y),
        )
        if request.pose.header.frame_id != self.configured_frame:
            request.route_id = None
            request.route_poses = []
            request.routes_prepared = True
            return True
        bypass_distance = max(
            0.0, float(self.get_parameter('route_bypass_distance_m').value)
        )
        # Use the lanes of the field this request was created for, not the
        # currently selected one: a side switch preempts the request instead of
        # silently re-planning it through the other field's throat.
        side_poses, side_departures, side_approaches = self._side_geometry(
            request.field_side
        )
        # Saved goals and RViz requests do not carry a numbered goal id.
        # Recognize the actual bucket destination for routing, while keeping
        # the saved coordinates, heading, name and public goal id unchanged.
        route_goal_id = request.goal_id or match_fixed_goal_approach(
            side_approaches, side_poses, destination)
        approach_id, approach_points = None, []
        if route_goal_id is not None:
            approach_id, approach_points = select_fixed_goal_approach(
                side_approaches,
                route_goal_id,
                self.current_position,
                destination_position=destination,
            )
            if self.current_position is not None and math.hypot(
                float(self.current_position[0]) - destination[0],
                float(self.current_position[1]) - destination[1],
            ) <= bypass_distance:
                approach_id, approach_points = None, []

        meaningful_rviz_motion = (
            route_goal_id is not None
            or self.current_position is None
            or math.hypot(
                float(self.current_position[0]) - destination[0],
                float(self.current_position[1]) - destination[1],
            ) > bypass_distance
        )
        departure_id, departure_points = None, []
        if meaningful_rviz_motion:
            departure_id, departure_points = select_fixed_pose_departure(
                side_departures,
                side_poses,
                self.current_position,
                bypass_distance,
                destination_goal_id=route_goal_id,
            )
            if departure_id is None:
                departure_id, departure_points = select_fixed_goal_departure(
                    side_approaches,
                    side_poses,
                    self.current_position,
                    destination,
                    bypass_distance,
                    destination_goal_id=route_goal_id,
                )
        transit_points = []
        if approach_points or (departure_id and departure_id.startswith('fixed_bucket_')):
            # Connect from the END of the departure to the START of the
            # approach. Using the original pose here can undo the exit lane.
            start = ((departure_points[-1]['x'], departure_points[-1]['y'])
                     if departure_points else self.current_position)
            end = ((approach_points[0]['x'], approach_points[0]['y'])
                   if approach_points else destination)
            if getattr(self, 'route_executor', None) is not None:
                request.route_preparation = (
                    self.route_executor.submit(
                        self.bucket_transit.select, start, end, request.field_side),
                    departure_id, departure_points, approach_id, approach_points,
                )
                self._publish_status('PLANNING_ROUTE', request=request)
                return False
            transit_points = self.bucket_transit.select(start, end, request.field_side)
        GoalBridgeNode._set_fixed_route(self, request, departure_id, departure_points,
                                       approach_id, approach_points, transit_points)
        return True

    def _set_fixed_route(self, request, departure_id, departure_points,
                         approach_id, approach_points, transit_points):
        route_ids = [route_id for route_id in (
            departure_id, 'fixed_bucket_transit' if transit_points else None,
            approach_id,
        ) if route_id]
        stamp = self.get_clock().now().to_msg()
        request.route_id = '+'.join(route_ids) if route_ids else None
        request.route_poses = [
            configured_pose_message(point, stamp)
            for point in departure_points + transit_points + approach_points
        ]
        request.routes_prepared = True

    def _queue_configured_goal(self, goal_id, source='topic'):
        goal_id = str(goal_id).strip()
        configured = self.configured_poses.get(goal_id)
        if configured is None:
            self.get_logger().error(f'Unknown configured goal id: {goal_id!r}')
            self._publish_status('INVALID_GOAL_ID', goal_id=goal_id)
            return False
        stamp = self.get_clock().now().to_msg()
        pose = configured_pose_message(configured, stamp)
        request = GoalRequest(
            pose=pose,
            label=f'{source}:{goal_id}:{configured["name"]}:{self.field_side}',
            goal_id=goal_id,
            use_fixed_routes=True,
            field_side=self.field_side,
        )
        return self._queue_request(request)

    def _goal_id_cb(self, message: String) -> None:
        self._queue_configured_goal(message.data)

    def _remember_pose_cb(self, message: String) -> None:
        try:
            requested_name, requested_side = self._parse_remembered_request(message.data)
            name = normalize_pose_name(requested_name)
            side = requested_side or self.field_side
            if side != self.field_side:
                self._publish_remembered_poses(
                    'SAVE_FAILED', name=name, detail='フィールドの切替完了後に保存してください。')
                return
        except (ValueError, TypeError) as error:
            self.get_logger().error(f'Invalid remembered pose name: {error}')
            self._publish_remembered_poses(
                'INVALID_NAME', detail=str(error)
            )
            return
        if self.current_pose is None:
            self.get_logger().warning(
                f'Cannot remember {name!r}: no localization pose received'
            )
            self._publish_remembered_poses(
                'NO_CURRENT_POSE', name=name,
                detail='localization pose has not been received',
            )
            return
        pose_age = time.monotonic() - self.current_pose_received_at
        maximum_age = max(
            0.0,
            float(self.get_parameter('remember_pose_max_age_sec').value),
        )
        if pose_age > maximum_age:
            self.get_logger().warning(
                f'Cannot remember {name!r}: localization pose is '
                f'{pose_age:.2f} s old'
            )
            self._publish_remembered_poses(
                'STALE_CURRENT_POSE', name=name, pose_age_sec=pose_age,
            )
            return

        updated = dict(self.remembered_poses)
        # Recording a measured pose is independent of CAD route feasibility.
        # Keep even CAD-overlapping poses exactly as measured on each field.
        try:
            previous = self.effective_remembered_poses().get(name)
            if previous is None:
                previous = store_pose(dict(self.current_pose), side)
            updated[name] = remember_field_pose(previous, self.current_pose, side)
            save_remembered_poses(self.remembered_poses_file, updated)
        except (OSError, ValueError) as error:
            self.get_logger().error(
                f'Could not save remembered pose {name!r}: {error}'
            )
            self._publish_remembered_poses(
                'SAVE_FAILED', name=name, detail=str(error)
            )
            return
        self.remembered_poses = updated
        self.get_logger().info(
            f'Remembered pose {name!r}: '
            f'x={self.current_pose["x"]:.3f}, '
            f'y={self.current_pose["y"]:.3f}, '
            f'yaw={self.current_pose["yaw"]:.3f}'
        )
        self._publish_remembered_poses('SAVED', name=name)

    @staticmethod
    def _parse_remembered_request(payload):
        """Accept a bare pose name or {"name": ..., "side": ...}.

        The radio bridge carries the field in the same message as the name so
        the two can never be applied out of order; the local panel keeps
        sending the plain name and uses whichever field is selected.
        """
        text = str(payload).strip()
        if not text.startswith('{'):
            return text, None
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError('remembered goal request must be an object')
        request_id = data.get('request_id')
        if (request_id is not None
                and (not isinstance(request_id, str) or not request_id.strip()
                     or len(request_id) > 128)):
            raise ValueError('request_id must be non-empty text of at most 128 characters')
        side = data.get('side')
        return data.get('name', ''), (None if side is None else normalize_side(side))

    def _remembered_goal_cb(self, message: String) -> None:
        try:
            requested_name, side = self._parse_remembered_request(message.data)
            name = normalize_pose_name(requested_name)
            request_id = (json.loads(message.data).get('request_id')
                          if str(message.data).strip().startswith('{') else None)
        except (ValueError, TypeError) as error:
            self._publish_status(
                'INVALID_REMEMBERED_POSE', reason=str(error)
            )
            return
        configured = self.effective_remembered_poses().get(name)
        if configured is None:
            self.get_logger().error(f'Unknown remembered pose: {name!r}')
            self._publish_status(
                'INVALID_REMEMBERED_POSE',
                remembered_pose=name,
                reason='NOT_FOUND',
                request_id=request_id,
            )
            return
        if configured['frame_id'] != self.configured_frame:
            self.get_logger().error(
                f'Remembered pose {name!r} uses frame '
                f'{configured["frame_id"]!r}, expected {self.configured_frame!r}'
            )
            self._publish_status(
                'INVALID_REMEMBERED_POSE',
                remembered_pose=name,
                reason='FRAME_MISMATCH',
                request_id=request_id,
            )
            return
        # Validate the entire request before changing the field. An invalid
        # saved name/frame must not switch route geometry while an old action
        # continues running in the other field.
        if side is not None:
            self._set_field_side(
                side, 'remembered goal request', cancel_active=False
            )
        # Reflect the stored left-field pose before the gate is derived, so the
        # straight reverse run also points into the mirrored bay.
        target = resolve_remembered_pose(configured, self.field_side)
        # Saved loading targets are operator-authorized even when CAD overlaps.
        # Keep Nav2's checks on the route to the gate; only the bounded final
        # docking stage bypasses CAD. Live collision gating stays downstream.
        pose = configured_pose_message(
            target, self.get_clock().now().to_msg()
        )
        final_pose = None
        if name == 'A':
            final_pose = pose
            pose = configured_pose_message(
                reverse_gate(target), self.get_clock().now().to_msg())
        self._queue_request(GoalRequest(
            pose=pose,
            label=f'remembered:{name}:{self.field_side}',
            use_fixed_routes=True,
            remembered_pose_name=name,
            reverse_final_pose=final_pose,
            field_side=self.field_side,
            request_id=request_id,
        ))

    def _goal_cb(self, message: PoseStamped) -> None:
        if not pose_is_valid(message) or message.header.frame_id != self.configured_frame:
            self.get_logger().error('Ignoring invalid RViz goal pose')
            self._publish_status('INVALID_RVIZ_GOAL')
            return
        self._queue_request(GoalRequest(
            copy.deepcopy(message),
            'rviz',
            use_fixed_routes=True,
            field_side=self.field_side,
        ))

    def _pose_cb(self, message: PoseWithCovarianceStamped) -> None:
        pose = message.pose.pose
        values = (
            float(pose.position.x),
            float(pose.position.y),
            float(pose.position.z),
            float(pose.orientation.x),
            float(pose.orientation.y),
            float(pose.orientation.z),
            float(pose.orientation.w),
        )
        quaternion_norm = math.hypot(*values[3:])
        if not all(math.isfinite(value) for value in values):
            return
        if not math.isfinite(quaternion_norm) or quaternion_norm <= 1.0e-6:
            return
        x, y = values[:2]
        frame_id = str(message.header.frame_id).strip()
        if frame_id != self.configured_frame:
            return
        source_ns = message_stamp_nanoseconds(message.header.stamp)
        if source_ns is None:
            return
        source_stamp = source_ns * 1.e-9
        age = self.get_clock().now().nanoseconds * 1.e-9 - source_stamp
        maximum_age = max(0., float(self.get_parameter('remember_pose_max_age_sec').value))
        if not math.isfinite(age) or age < -.02 or age > maximum_age:
            return
        now = time.monotonic()
        previous = getattr(self, 'current_pose_source_stamp', None)
        if (previous is not None and source_stamp <= previous
                and now-self.current_pose_received_at <= maximum_age):
            return
        qx, qy, qz, qw = (
            value / quaternion_norm for value in values[3:]
        )
        yaw = math.atan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy * qy + qz * qz),
        )
        self.current_pose = {
            'frame_id': frame_id,
            'x': x,
            'y': y,
            'yaw': yaw,
        }
        self.current_position = (x, y)
        self.current_pose_received_at = now - max(0., age)
        self.current_pose_source_stamp = source_stamp

    def _tracker_status_cb(self, message):
        try:
            self.tracker_arrival = json.loads(message.data).get('arrival')
            self.tracker_arrival_stamp = time.monotonic()
        except (ValueError, AttributeError):
            self.tracker_arrival = None
        # A loading gate handoff need not wait for the 200 ms retry timer.
        # Keep the measured stop/goal checks and Nav2 completion prerequisite.
        if (self.finalizing_since is not None
                and getattr(self.active_request, 'reverse_final_pose', None) is not None
                and self.active_goal_handle is None
                and self.pending_request is None and not self.cancel_requested):
            self._finish_tracker_arrival(time.monotonic())

    def _tracker_arrived(self, now):
        arrival = self.tracker_arrival
        request = self.active_request
        if (not isinstance(arrival, dict) or request is None
                or now-self.tracker_arrival_stamp > .3 or arrival.get('ready') is not True):
            return False
        expected_stamp = getattr(request, 'goal_stamp', None)
        if expected_stamp is not None and arrival.get('goal_stamp') != expected_stamp:
            return False
        target = arrival.get('goal', [])
        pose = request.pose.pose
        q = pose.orientation
        yaw = math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))
        return (isinstance(target, (list, tuple)) and len(target)==3
                and all(type(v) in (int, float) and math.isfinite(v) for v in target)
                and math.hypot(target[0]-pose.position.x,target[1]-pose.position.y)<1.e-6
                and abs((target[2]-yaw+math.pi)%(2*math.pi)-math.pi)<1.e-6)

    def _finish_tracker_arrival(self, now):
        if self.finalizing_since is None:
            return
        if self._tracker_arrived(now):
            if getattr(self.active_request, 'reverse_final_pose', None) is not None:
                self._begin_reverse(now)
                return
            self._publish_status('SUCCEEDED', request=self.active_request)
            self.get_logger().info(f'Goal precisely settled: {self.active_request.label}')
        elif now-self.finalizing_since > (REVERSE_TIMEOUT if getattr(self.active_request, 'reversing', False) else 5.):
            self._publish_status('FAILED', request=self.active_request,
                                 reason='FINAL_APPROACH_TIMEOUT')
        else:
            if getattr(self.active_request, 'reversing', False):
                self._publish_reverse()
            return
        self.finalizing_since = None
        self.active_request = None

    def _publish_reverse(self):
        pose = self.active_request.pose.pose
        q = pose.orientation
        yaw = math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))
        self._publish_status('REVERSE_APPROACH',
                             ignore_cad=True,
                             reverse_goal=[pose.position.x, pose.position.y, yaw])

    def _begin_reverse(self, now):
        request = self.active_request
        request.pose = request.reverse_final_pose
        request.reverse_final_pose = None
        request.reversing = True
        self.tracker_arrival = None
        self.finalizing_since = now
        self._publish_reverse()

    def _cancel_cb(self, message: Bool) -> None:
        if message.data:
            self._request_cancel(explicit=True)

    def _request_cancel(self, *, explicit: bool) -> None:
        canceled_request = self.active_request
        if getattr(self, 'finalizing_since', None) is not None:
            self.finalizing_since = None
            self.active_request = None
        if explicit:
            preparation = getattr(self.pending_request, 'route_preparation', None)
            if preparation is not None:
                preparation[0].cancel()
            self.pending_request = None
            self.cancel_requested = True
        if self.active_goal_handle is not None:
            if self.cancel_future is None and not getattr(self, 'cancel_accepted', False):
                try:
                    self.cancel_requested_at = time.monotonic()
                    self.cancel_future = self.active_goal_handle.cancel_goal_async()
                    self.cancel_future.add_done_callback(self._cancel_response_cb)
                except Exception as error:
                    self.cancel_future = None
                    self.cancel_retry_not_before = time.monotonic() + .25
                    self.get_logger().error(f'Cancel request failed: {error}; retrying')
            state = 'CANCELING' if explicit else 'PREEMPTING'
            self._publish_status(
                state,
                request=(self.active_request if explicit else self.pending_request),
                cancel_goal_stamp=getattr(canceled_request, 'goal_stamp', None),
            )
            return
        if self.send_future is not None:
            # A goal handle cannot be canceled until Nav2 accepts it. The goal
            # response callback observes this flag / pending replacement and
            # cancels immediately without ever starting a second action.
            self.cancel_requested = self.cancel_requested or explicit
            return
        if explicit:
            self.active_request = None
            self._publish_status('CANCELED', request=canceled_request)

    def _cancel_response_cb(self, future) -> None:
        if future is not self.cancel_future:
            return
        self.cancel_future = None
        try:
            response = future.result()
            accepted = bool(response.goals_canceling)
        except Exception as error:  # rclpy action transport failures
            self.get_logger().error(f'Cancel request failed: {error}')
            self.cancel_retry_not_before = time.monotonic() + .25
            return
        if not accepted:
            self.get_logger().warning('Nav2 did not accept the cancel request')
            self.cancel_retry_not_before = time.monotonic() + .25
        self.cancel_accepted = accepted

    def _abandon_lifecycle_request(self, name: str) -> None:
        """Drop a get_state request that will never be answered.

        rclpy keeps the future pending forever, so without this the slot for
        that node is never freed and it is never polled again.
        """
        future = self.lifecycle_futures[name]
        self.lifecycle_futures[name] = None
        self.lifecycle_states[name] = None
        self.lifecycle_state_times[name] = -math.inf
        if future is None:
            return
        client = self.lifecycle_clients.get(name)
        try:
            if client is not None and hasattr(client, 'remove_pending_request'):
                client.remove_pending_request(future)
            else:
                future.cancel()
        except Exception:
            pass

    def _poll_lifecycle_states(self, now: float) -> None:
        request_timeout = max(
            0.20,
            float(self.get_parameter('lifecycle_request_timeout_sec').value),
        )
        for name, future in tuple(self.lifecycle_futures.items()):
            if future is None:
                continue
            if not future.done():
                if now - self.lifecycle_request_times[name] > request_timeout:
                    self.lifecycle_timeouts[name] += 1
                    if self.lifecycle_timeouts[name] in (1, 20):
                        self.get_logger().warning(
                            f'{name}/get_state did not answer within '
                            f'{request_timeout:.2f} s; retrying '
                            f'(timeouts={self.lifecycle_timeouts[name]})'
                        )
                    self._abandon_lifecycle_request(name)
                continue
            try:
                self.lifecycle_states[name] = int(
                    future.result().current_state.id
                )
                self.lifecycle_state_times[name] = now
            except Exception:
                self.lifecycle_states[name] = None
                self.lifecycle_state_times[name] = -math.inf
            self.lifecycle_futures[name] = None

        poll_period = max(
            0.10, float(self.get_parameter('lifecycle_poll_period_sec').value)
        )
        if now - self.last_lifecycle_poll < poll_period:
            return
        self.last_lifecycle_poll = now
        for name, client in self.lifecycle_clients.items():
            if self.lifecycle_futures[name] is not None:
                continue
            if not client.service_is_ready():
                self.lifecycle_states[name] = None
                self.lifecycle_state_times[name] = -math.inf
                continue
            try:
                self.lifecycle_futures[name] = client.call_async(GetState.Request())
                self.lifecycle_request_times[name] = now
            except Exception as error:  # The service can disappear after discovery.
                self.lifecycle_states[name] = None
                self.lifecycle_state_times[name] = -math.inf
                self.get_logger().warning(f'{name}/get_state request failed: {error}')

    def _navigation_ready(self, now: float, request=None) -> bool:
        self.not_ready_reason = ''
        request = request or self.pending_request
        action_client = (
            self.route_action_client
            if request is not None and request.route_poses
            else self.action_client
        )
        if not action_client.server_is_ready():
            self.not_ready_reason = (
                'navigate_through_poses action server'
                if request is not None and request.route_poses
                else 'navigate_to_pose action server'
            )
            return False
        # Nav2 creates its navigation action servers only after configuring
        # the navigator. Polling five lifecycle services before that point
        # competes with the manager's configure/get_state transactions and
        # can make bringup abort under load. Discovery is passive; keep the
        # queued goal here until the navigator can answer, then verify every
        # required node is active before sending the action.
        self._poll_lifecycle_states(now)
        if not self.lifecycle_clients:
            return True
        maximum_age = max(
            0.50,
            float(self.get_parameter('lifecycle_state_max_age_sec').value),
        )
        blocking = [
            name for name in self.lifecycle_clients
            if not (
                self.lifecycle_states[name] == State.PRIMARY_STATE_ACTIVE
                and now - self.lifecycle_state_times[name] <= maximum_age
            )
        ]
        if blocking:
            # Name the node.  A goal that silently stays queued is the single
            # hardest failure to diagnose from the operator panel.
            self.not_ready_reason = 'not active: ' + ', '.join(sorted(blocking))
            return False
        return True

    def _try_send_pending(self) -> None:
        now = time.monotonic()
        if getattr(self, 'finalizing_since', None) is not None:
            self._finish_tracker_arrival(now)
        if (getattr(self, 'active_goal_handle', None) is not None
                and getattr(self, 'result_future', None) is None
                and now >= getattr(self, 'result_retry_not_before', -math.inf)):
            self._watch_goal_result()
        if (getattr(self, 'active_goal_handle', None) is not None
                and (self.cancel_requested or self.pending_request is not None
                     or getattr(self, 'result_monitor_cancel', False))):
            future = self.cancel_future
            timeout = max(.20, float(self.get_parameter('cancel_request_timeout_sec').value))
            if future is not None and now-self.cancel_requested_at > timeout:
                # Canceling the local future clears the action client's pending
                # request. It does not mean the remote goal has stopped.
                self.cancel_future = None
                future.cancel()
                self.cancel_retry_not_before = now
                self.get_logger().warning('Nav2 cancel response timed out; retrying')
            if (self.cancel_future is None and not getattr(self, 'cancel_accepted', False)
                    and now >= getattr(self, 'cancel_retry_not_before', -math.inf)):
                self._request_cancel(explicit=self.cancel_requested)
        # There is no reason to query every Nav2 lifecycle service while the
        # operator has not requested motion. On a cold Jetson startup those
        # extra requests competed with lifecycle transitions and could make
        # controller_server configuration time out.
        if self.pending_request is None:
            return
        if (
            self.send_future is not None
            or self.active_request is not None
            or now < self.retry_not_before
        ):
            return
        if not self._navigation_ready(now, self.pending_request):
            self._publish_status(
                'WAITING_FOR_NAV2', request=self.pending_request,
                detail=self.not_ready_reason,
            )
            if now - self.last_waiting_log >= 5.0:
                self.last_waiting_log = now
                self.get_logger().warning(
                    f'Goal {self.pending_request.label} is queued: waiting for '
                    f'{self.not_ready_reason or "Nav2"}'
                )
            return

        # Prepare once from the current pose. CAD sweeps run off the ROS
        # executor so lifecycle responses and cancel requests remain live.
        # Recheck readiness after preparation without rebuilding on TF jitter.
        try:
            if not self._refresh_fixed_routes(self.pending_request):
                return
        except Exception as error:
            self.get_logger().error(f'Route preparation failed: {error}')
            self._publish_status('FAILED', request=self.pending_request,
                                 reason=f'ROUTE_ERROR:{error}')
            self.pending_request = None
            return
        if not self._navigation_ready(time.monotonic(), self.pending_request):
            return
        self.active_request = self.pending_request
        self.pending_request = None
        self.active_request.attempts += 1
        stamp = self.get_clock().now().to_msg()
        if self.active_request.route_poses:
            goal = NavigateThroughPoses.Goal()
            goal.behavior_tree = self.route_behavior_tree
            goal.poses = [
                copy.deepcopy(pose) for pose in self.active_request.route_poses
            ] + [copy.deepcopy(self.active_request.pose)]
            for route_pose in goal.poses:
                route_pose.header.stamp = stamp
            final_pose = goal.poses[-1]
            action_client = self.route_action_client
        else:
            goal = NavigateToPose.Goal()
            goal.pose = copy.deepcopy(self.active_request.pose)
            goal.pose.header.stamp = stamp
            final_pose = goal.pose
            action_client = self.action_client
        self.active_request.goal_stamp = [stamp.sec, stamp.nanosec]
        self.tracker_arrival = None
        self.active_goal_pub.publish(final_pose)
        try:
            request = self.active_request
            self.send_future = action_client.send_goal_async(
                goal, feedback_callback=lambda feedback: self._feedback_cb(feedback, request=request)
            )
        except Exception as error:  # Discovery and transport can race shutdown.
            self._retry_or_finish(f'SEND_ERROR:{error}')
            return
        self.send_future.add_done_callback(self._goal_response_cb)
        self._publish_status('SENDING')
        self.get_logger().info(
            f'Sending goal {self.active_request.label}, '
            f'attempt={self.active_request.attempts}, '
            f'x={final_pose.pose.position.x:.3f}, '
            f'y={final_pose.pose.position.y:.3f}, '
            f'route={self.active_request.route_id or "dynamic"}'
        )

    def _goal_response_cb(self, future) -> None:
        if future is not self.send_future:
            return
        self.send_future = None
        try:
            goal_handle = future.result()
        except Exception as error:  # rclpy action transport failures
            self._retry_or_finish(f'SEND_ERROR:{error}')
            return
        if not goal_handle.accepted:
            if self.pending_request is not None:
                self.active_request = None
                self._try_send_pending()
                return
            self._retry_or_finish('REJECTED')
            return
        self.active_goal_handle = goal_handle
        self.cancel_accepted = False
        self.result_monitor_cancel = False
        if not self._watch_goal_result():
            return
        if self.cancel_requested or self.pending_request is not None:
            self._request_cancel(explicit=self.cancel_requested)
            return
        self._publish_status('ACTIVE')

    def _watch_goal_result(self):
        """Recover result subscription failures without sending a second goal."""
        goal_handle, request = self.active_goal_handle, self.active_request
        try:
            self.result_future = goal_handle.get_result_async()
            self.result_future.add_done_callback(
                lambda result: self._result_cb(result, request=request, goal_handle=goal_handle))
            return True
        except Exception as error:
            self.result_future = None
            self.result_monitor_cancel = True
            self.result_retry_not_before = time.monotonic() + .25
            self.get_logger().error(f'Goal result subscription failed: {error}; stopping')
            self._request_cancel(explicit=False)
            self._publish_status('CANCELING', reason=f'RESULT_REQUEST_ERROR:{error}')
            return False

    def _feedback_cb(self, feedback_message, *, request=None) -> None:
        if request is not None and request is not self.active_request:
            return
        feedback = feedback_message.feedback
        distance = float(feedback.distance_remaining)
        self._publish_status(
            'ACTIVE', distance_remaining_m=distance if math.isfinite(distance) else None
        )

    def _result_cb(self, future, *, request=None, goal_handle=None) -> None:
        if ((request is not None and request is not self.active_request)
                or (goal_handle is not None and goal_handle is not self.active_goal_handle)):
            return
        try:
            status = future.result().status
        except Exception as error:  # rclpy action transport failures
            self.result_future = None
            self.result_monitor_cancel = True
            self.result_retry_not_before = time.monotonic() + .25
            # A broken result transport does not prove the action stopped.
            # Retain its handle, stop tracking, and recover/cancel this action
            # before any queued replacement can be sent.
            self._request_cancel(explicit=False)
            self._publish_status('CANCELING', reason=f'RESULT_ERROR:{error}')
            return
        self.active_goal_handle = None
        self.result_future = None
        self.result_monitor_cancel = False
        self.cancel_future = None
        self.cancel_accepted = False
        if status == GoalStatus.STATUS_SUCCEEDED:
            request = self.active_request
            if ((getattr(self, 'verify_tracker_arrival', False)
                    or getattr(request, 'reverse_final_pose', None) is not None)
                    and self.pending_request is None and not self.cancel_requested
                    and not self._tracker_arrived(time.monotonic())):
                # Smac's last cell is not the operator's exact goal. Let the
                # existing bounded terminal servo settle the requested pose,
                # then acknowledge success using measured motion as well.
                self.finalizing_since = time.monotonic()
                self._publish_status('FINAL_APPROACH', request=request)
                return
            if (getattr(request, 'reverse_final_pose', None) is not None
                    and self.pending_request is None and not self.cancel_requested):
                self._begin_reverse(time.monotonic())
                return
            self._publish_status('SUCCEEDED', request=request)
            self.get_logger().info(f'Goal succeeded: {request.label}')
            self.active_request = None
        elif status == GoalStatus.STATUS_CANCELED:
            request = self.active_request
            self._publish_status('CANCELED', request=request)
            self.active_request = None
        else:
            if self.pending_request is not None or self.cancel_requested:
                request = self.active_request
                self._publish_status(
                    'CANCELED', request=request,
                    reason=f'NAV2_STATUS_{status}',
                )
                self.active_request = None
            else:
                self._retry_or_finish(f'NAV2_STATUS_{status}')
                return
        self.cancel_requested = False
        if self.pending_request is not None:
            self.retry_not_before = time.monotonic()
            self._try_send_pending()

    def _retry_or_finish(self, reason) -> None:
        request = self.active_request
        if request is None:
            self._publish_status('FAILED', reason=reason)
            return
        if self.pending_request is not None or self.cancel_requested:
            self._publish_status('CANCELED', request=request, reason=reason)
            self.active_request = None
            self.cancel_requested = False
            self.retry_not_before = time.monotonic()
            self._try_send_pending()
            return
        max_retries = max(0, int(self.get_parameter('max_retries').value))
        if request.attempts <= max_retries:
            self.get_logger().warning(
                f'Goal {request.label} failed ({reason}); retrying'
            )
            self.active_request = None
            request.routes_prepared = False
            request.route_preparation = None
            self.pending_request = request
            self.retry_not_before = time.monotonic() + max(
                0.0, float(self.get_parameter('retry_delay_sec').value)
            )
            self._publish_status('RETRY_WAIT', request=request, reason=reason)
            return
        self._publish_status('FAILED', request=request, reason=reason)
        self.get_logger().error(
            f'Goal failed after {request.attempts} attempts: {request.label} ({reason})'
        )
        self.active_request = None


    def destroy_node(self):
        self.route_executor.shutdown(wait=True, cancel_futures=True)
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = GoalBridgeNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
