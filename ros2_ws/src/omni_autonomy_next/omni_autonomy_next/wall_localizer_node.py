import json
import math
import threading
import time
from bisect import bisect_left
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import (
    Point,
    PoseWithCovarianceStamped,
    TransformStamped,
)
from omni_autonomy_interfaces.srv import SetMotionContext
from nav_msgs.msg import Odometry
from rcl_interfaces.msg import SetParametersResult
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy._rclpy_pybind11 import RCLError
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray

from .config import ConfigError, load_field, load_robot
from .competition_footprint import (
    FootprintGeometryError,
    canonical_convex_polygon,
    footprint_digest,
)
from .geometry import (
    WallLookupGrid,
    compose_pose,
    interpolate_pose,
    interpolate_poses_at,
    inverse_pose,
    normalize_angle,
    optimize_pose,
    points_in_polygon,
    relative_pose,
    transform_points_from_poses,
)
from .scan_freshness import scan_generations_changed

# Parameters mirrored into an attribute cache so the 100 Hz+ hot paths avoid
# per-call rcl parameter lookups. Updated live via the parameter callback.
CACHED_PARAMETERS = (
    'max_iterations',
    'max_correspondence_distance',
    'huber_delta',
    'min_correspondences',
    'max_translation_step',
    'max_rotation_step',
    'max_accepted_rmse',
    'max_pose_jump_translation',
    'max_pose_jump_rotation',
    'correspondence_trim_ratio',
    'lidar_correction_gain',
    'max_lidar_correction_translation',
    'max_lidar_correction_rotation',
    'settled_lidar_correction_gain',
    'settled_max_lidar_correction_translation',
    'settled_max_lidar_correction_rotation',
    'settled_wheel_linear_speed',
    'settled_wheel_angular_speed',
    'wheel_odom_freshness_sec',
    'recovery_enabled',
    'recovery_after_rejections',
    'recovery_search_translation',
    'recovery_search_rotation',
    'recovery_max_jump_translation',
    'recovery_max_jump_rotation',
)


def yaw_from_quaternion(quaternion) -> float:
    sin_yaw = 2.0 * (
        quaternion.w * quaternion.z + quaternion.x * quaternion.y
    )
    cos_yaw = 1.0 - 2.0 * (
        quaternion.y * quaternion.y + quaternion.z * quaternion.z
    )
    return math.atan2(sin_yaw, cos_yaw)


def set_yaw(quaternion, yaw: float) -> None:
    quaternion.x = 0.0
    quaternion.y = 0.0
    quaternion.z = math.sin(0.5 * yaw)
    quaternion.w = math.cos(0.5 * yaw)


def message_stamp_nanoseconds(stamp) -> Optional[int]:
    seconds = int(stamp.sec)
    nanoseconds = int(stamp.nanosec)
    if seconds == 0 and nanoseconds == 0:
        return None
    return seconds * 1_000_000_000 + nanoseconds


def advance_corrected_pose_to_wheel_snapshot(
    corrected_pose: np.ndarray,
    *,
    snapshot_wheel_pose: Optional[np.ndarray],
    snapshot_stamp_ns: Optional[int],
    snapshot_generation: Optional[int],
    latest_wheel_pose: Optional[np.ndarray],
    latest_stamp_ns: Optional[int],
    latest_generation: Optional[int],
) -> Optional[np.ndarray]:
    """Advance an ICP pose through wheel motion that arrived during ICP.

    The wheel callback runs independently from the comparatively long ICP
    solve.  ``corrected_pose`` therefore belongs to the wheel snapshot taken
    before optimization.  Re-applying the post-snapshot wheel delta prevents
    the ICP commit from erasing real motion received while it was solving.

    ``None`` means that the snapshot/current pair is inconsistent.  A caller
    must reject that ICP commit rather than guessing across a timestamp or
    state-generation rollback.
    """
    corrected = np.asarray(corrected_pose, dtype=float)
    if corrected.shape != (3,) or not np.all(np.isfinite(corrected)):
        return None
    if snapshot_wheel_pose is None:
        # No wheel sample existed when ICP started.  It is safe to commit only
        # if no wheel sample appeared while ICP was running; otherwise there is
        # no pose from which to reconstruct that concurrent motion.
        if latest_wheel_pose is None and latest_generation in (None, 0):
            return corrected.copy()
        return None
    if (
        snapshot_stamp_ns is None
        or snapshot_generation is None
        or latest_wheel_pose is None
        or latest_stamp_ns is None
        or latest_generation is None
    ):
        return None

    snapshot = np.asarray(snapshot_wheel_pose, dtype=float)
    latest = np.asarray(latest_wheel_pose, dtype=float)
    if (
        snapshot.shape != (3,)
        or latest.shape != (3,)
        or not np.all(np.isfinite(snapshot))
        or not np.all(np.isfinite(latest))
    ):
        return None

    snapshot_stamp = int(snapshot_stamp_ns)
    latest_stamp = int(latest_stamp_ns)
    snapshot_sequence = int(snapshot_generation)
    latest_sequence = int(latest_generation)
    if latest_sequence < snapshot_sequence or latest_stamp < snapshot_stamp:
        return None
    if latest_sequence == snapshot_sequence:
        if latest_stamp != snapshot_stamp or not np.array_equal(latest, snapshot):
            return None
        return corrected.copy()
    if latest_stamp <= snapshot_stamp:
        return None

    post_snapshot_motion = relative_pose(snapshot, latest)
    if not np.all(np.isfinite(post_snapshot_motion)):
        return None
    return compose_pose(corrected, post_snapshot_motion)


def localizer_tf_chain(
    pose,
    wheel_pose,
    odom_source_alive: bool,
    *,
    map_frame: str,
    odom_frame: str,
    base_frame: str,
) -> List[Tuple[str, str, np.ndarray]]:
    """Return the ``(parent, child, [x, y, yaw])`` transforms to broadcast.

    ``map -> odom`` is always ours.  ``odom -> base_link`` belongs to
    measurement_wheel, so it is only added while that source is not alive --
    otherwise ``base_link`` would have two parents.  The substitute reuses the
    last known wheel pose, which keeps ``odom`` continuous if the counters drop
    out mid-run, and it keeps ``base_link`` attached to the tree when they never
    start at all.  Without it a missing counter board removes the robot, the
    LiDAR point clouds, and the odom-frame local costmap from RViz.
    """
    if wheel_pose is not None:
        map_to_odom = compose_pose(pose, inverse_pose(wheel_pose))
        odom_to_base = np.asarray(wheel_pose, dtype=float)
    else:
        map_to_odom = np.asarray(pose, dtype=float)
        odom_to_base = np.zeros(3)
    chain = [(map_frame, odom_frame, map_to_odom)]
    if not odom_source_alive:
        chain.append((odom_frame, base_frame, odom_to_base))
    return chain


class WallLocalizer(Node):
    def __init__(self) -> None:
        super().__init__('wall_localizer')
        self._declare_parameters()

        field_path = self.get_parameter('field_config_file').value
        robot_path = self.get_parameter('robot_config_file').value
        if not field_path or not robot_path:
            raise ConfigError(
                'field_config_file and robot_config_file must be set by the launch file'
            )

        self.field = load_field(field_path)
        self.robot = load_robot(robot_path)
        self.params: Dict[str, object] = {}
        self._refresh_cached_parameters()
        self.add_on_set_parameters_callback(self._cached_parameters_callback)
        lookup_started = time.perf_counter()
        self.wall_lookup = WallLookupGrid(
            self.field['walls'],
            resolution=float(
                self.get_parameter('lookup_grid_resolution').value
            ),
            margin=float(self.params['max_correspondence_distance']) + 0.6,
        )
        self.get_logger().info(
            'Wall lookup grid built in '
            f'{time.perf_counter() - lookup_started:.2f}s '
            f'({self.wall_lookup.cells_x}x{self.wall_lookup.cells_y} cells)'
        )
        self.rejected_streak = 0
        self.map_frame = self.field['frame_id']
        self.base_frame = self.robot['base_frame_id']
        self.pose = np.asarray(
            self.get_parameter('initial_pose').value, dtype=float
        )
        if self.pose.shape != (3,):
            raise ConfigError('initial_pose must be [x, y, yaw]')
        self.pose[2] = normalize_angle(float(self.pose[2]))
        self.covariance = np.diag([0.25, 0.25, math.radians(15.0) ** 2])
        # 高速配信コールバックは別スレッド(別コールバックグループ)で走るため、
        # pose/covariance/latest_wheel_poseの置き換えと読み出しを対にして守る
        # Wheel odometry is handled in its own callback group so a long ICP
        # solve cannot starve the 100 Hz history needed for scan deskew.  An
        # RLock lets the wheel callback protect one atomic state update while
        # reusing the existing small locked sections below.
        self._pose_lock = threading.RLock()
        # Operator /initialpose resets must win over any ICP solve that was
        # already in flight when the reset arrived.
        self.pose_reset_generation = 0

        self.scan_timeout = Duration(
            seconds=float(self.get_parameter('scan_timeout_sec').value)
        )
        self.min_range = float(self.get_parameter('min_range').value)
        self.max_range = float(self.get_parameter('max_range').value)
        self.beam_stride = max(1, int(self.get_parameter('beam_stride').value))
        self.max_points = max(
            10, int(self.get_parameter('max_points_per_lidar').value)
        )
        self.footprint = self._padded_footprint(
            self.robot['footprint'],
            float(self.get_parameter('footprint_padding').value),
        )
        self._motion_context_revision = 0
        self._motion_context_execution_id = 0
        self._motion_context_digest = ''

        self.use_wheel_odometry = bool(
            self.get_parameter('use_wheel_odometry').value
        )
        self.motion_compensate_scans = bool(
            self.get_parameter('motion_compensate_scans').value
        )
        self.wheel_history_duration_ns = int(
            max(
                0.1,
                float(self.get_parameter('wheel_odom_history_sec').value),
            )
            * 1.0e9
        )
        self.wheel_stamp_tolerance_ns = int(
            max(
                0.0,
                float(
                    self.get_parameter('wheel_odom_stamp_tolerance_sec').value
                ),
            )
            * 1.0e9
        )
        self.publish_tf = bool(self.get_parameter('publish_tf').value)
        self.tf_future_tolerance = max(
            0.0,
            float(self.get_parameter('tf_future_tolerance_sec').value),
        )
        self.wheel_odom_tf_timeout_ns = int(
            max(
                0.0,
                float(self.get_parameter('wheel_odom_tf_timeout_sec').value),
            )
            * 1.0e9
        )
        self._odom_source_alive: Optional[bool] = None
        self.tracking_solution_timeout_sec = max(
            0.1,
            float(self.get_parameter('tracking_solution_timeout_sec').value),
        )
        self.odom_frame = 'odom'
        self.latest_wheel_pose: Optional[np.ndarray] = None
        self.latest_wheel_stamp_ns: Optional[int] = None
        self.wheel_generation = 0
        self.latest_wheel_received_ns: Optional[int] = None
        self.previous_wheel_pose: Optional[np.ndarray] = None
        self.previous_wheel_stamp_ns: Optional[int] = None
        self.latest_wheel_linear_speed = 0.0
        self.latest_wheel_angular_speed = 0.0
        self.wheel_history: List[Tuple[int, np.ndarray]] = []
        self.wheel_history_stamps: List[int] = []

        self.latest_scans: Dict[str, Dict[str, object]] = {}
        self.scan_generations = {
            str(lidar['name']): 0 for lidar in self.robot['lidars']
        }
        self.last_processed_scan_generations: Optional[
            Tuple[Tuple[str, int], ...]
        ] = None
        self.min_active_lidars = max(
            1,
            min(
                len(self.robot['lidars']),
                int(self.get_parameter('min_active_lidars').value),
            ),
        )
        self.active_lidar_count = 0
        self.missing_lidars: Tuple[str, ...] = ()
        self.frame_warning_sent = set()
        self.last_result = {
            'state': 'WAITING_FOR_SCANS',
            'accepted': False,
            'correspondences': 0,
            'rmse': math.inf,
        }
        self.last_tracking_accept_time = -math.inf

        pose_topic = str(self.get_parameter('pose_topic').value)
        marker_topic = str(self.get_parameter('marker_topic').value)
        self.pose_publisher = self.create_publisher(
            PoseWithCovarianceStamped, pose_topic, 10
        )
        marker_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.marker_publisher = self.create_publisher(
            MarkerArray, marker_topic, marker_qos
        )
        self.diagnostic_publisher = self.create_publisher(
            DiagnosticArray, '/diagnostics', 10
        )
        # Fast, machine-readable health heartbeat for motion controllers.
        # /localization/pose is intentionally republished at the TF rate, so
        # its receipt time alone cannot reveal rejected/stale ICP updates.
        self.tracking_publisher = self.create_publisher(
            Bool, str(self.get_parameter('tracking_topic').value), 10
        )
        self.create_service(
            SetMotionContext,
            str(self.get_parameter('motion_context_service').value),
            self._set_motion_context,
        )
        self.tf_broadcaster = TransformBroadcaster(self)

        latest_scan_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        # Scan ingestion must not share the default mutually-exclusive group
        # with the CPU-heavy ICP timer. The old arrangement blocked both scan
        # callbacks for up to 0.52 s while optimizing, aged the cached scans
        # past scan_timeout_sec, and then reported WAITING_FOR_SCANS even
        # though both synthetic/physical LiDARs were still publishing.
        self.scan_callback_group = MutuallyExclusiveCallbackGroup()
        for lidar in self.robot['lidars']:
            self.create_subscription(
                LaserScan,
                lidar['topic'],
                lambda message, config=lidar: self._scan_callback(message, config),
                latest_scan_qos,
                callback_group=self.scan_callback_group,
            )

        self.create_subscription(
            PoseWithCovarianceStamped,
            '/initialpose',
            self._initial_pose_callback,
            10,
        )
        self.wheel_callback_group = MutuallyExclusiveCallbackGroup()
        if self.use_wheel_odometry:
            self.create_subscription(
                Odometry,
                str(self.get_parameter('wheel_odom_topic').value),
                self._wheel_odom_callback,
                qos_profile_sensor_data,
                callback_group=self.wheel_callback_group,
            )

        update_rate = max(
            0.2, float(self.get_parameter('update_rate_hz').value)
        )
        self.timer = self.create_timer(1.0 / update_rate, self._timer_callback)
        self.fast_publish_group = MutuallyExclusiveCallbackGroup()
        tf_publish_rate = max(
            update_rate,
            float(self.get_parameter('tf_publish_rate_hz').value),
        )
        self.tf_timer = self.create_timer(
            1.0 / tf_publish_rate,
            self._fast_publish_callback,
            callback_group=self.fast_publish_group,
        )
        self.marker_publish_period_ns = self._period_ns_from_rate(
            float(self.get_parameter('marker_publish_rate_hz').value)
        )
        self.diagnostic_publish_period_ns = self._period_ns_from_rate(
            float(self.get_parameter('diagnostic_publish_rate_hz').value)
        )
        self.last_marker_publish_ns = 0
        self.last_diagnostic_publish_ns = 0
        # The field geometry never changes; rebuilding ~140 wall markers in
        # Python every marker cycle costs more than the ICP solve itself.
        # Republish them slowly just for late-joining RViz instances.
        self.field_marker_period_ns = int(5.0e9)
        self.last_field_marker_publish_ns = 0
        self._field_markers_cache: Optional[List[Marker]] = None
        self.get_logger().info(
            f'Loaded {len(self.field["walls"])} walls and '
            f'{len(self.robot["lidars"])} LiDARs; initial pose='
            f'[{self.pose[0]:.3f}, {self.pose[1]:.3f}, {self.pose[2]:.3f}]'
        )

    def _set_motion_context(self, request, response):
        revision = int(request.revision)
        if revision <= 0 or revision < self._motion_context_revision:
            response.success = False
            response.message = 'stale/invalid motion context revision'
            return response
        if revision == self._motion_context_revision:
            matches = bool(
                int(request.execution_id) == self._motion_context_execution_id
                and str(request.footprint_digest) == self._motion_context_digest
            )
            response.success = matches
            response.applied_revision = self._motion_context_revision
            response.applied_digest = self._motion_context_digest
            response.message = (
                'idempotent motion context ACK' if matches
                else 'revision reused with different context'
            )
            return response
        if request.footprint.header.frame_id != self.base_frame:
            response.success = False
            response.message = f'footprint frame must be {self.base_frame}'
            return response
        try:
            raw = np.asarray(canonical_convex_polygon(
                (point.x, point.y)
                for point in request.footprint.polygon.points
            ))
        except FootprintGeometryError as error:
            response.success = False
            response.message = str(error)
            return response
        digest = footprint_digest(raw)
        if str(request.footprint_digest) != digest:
            response.success = False
            response.message = 'footprint digest does not match polygon'
            return response
        padded = self._padded_footprint(
            raw, float(self.get_parameter('footprint_padding').value)
        )
        # Invalidate every scan collected with the previous self-mask. ICP is
        # deliberately unavailable until one fresh scan from both LiDARs has
        # arrived under the new footprint.
        with self._pose_lock:
            self.footprint = padded
            self.latest_scans.clear()
            self.last_processed_scan_generations = None
            self.rejected_streak = 0
            self.last_result = {
                'state': 'WAITING_FOR_SCANS_AFTER_FOOTPRINT_CHANGE',
                'accepted': False,
                'correspondences': 0,
                'rmse': math.inf,
            }
            self.last_tracking_accept_time = -math.inf
        self.tracking_publisher.publish(Bool(data=False))
        self._motion_context_revision = revision
        self._motion_context_execution_id = int(request.execution_id)
        self._motion_context_digest = digest
        response.success = True
        response.applied_revision = int(request.revision)
        response.applied_digest = digest
        response.message = (
            f'applied {request.footprint_profile} for execution '
            f'{request.execution_id}; old scans discarded'
        )
        return response

    def _declare_parameters(self) -> None:
        defaults = {
            'field_config_file': '',
            'robot_config_file': '',
            'initial_pose': [1.0, 1.0, 0.0],
            'update_rate_hz': 12.0,
            'tf_publish_rate_hz': 60.0,
            'tf_future_tolerance_sec': 0.20,
            'marker_publish_rate_hz': 5.0,
            'diagnostic_publish_rate_hz': 5.0,
            'tracking_solution_timeout_sec': 1.0,
            'scan_timeout_sec': 0.35,
            # How many LiDARs must be publishing fresh scans to localize at all.
            # 1 keeps autonomy available on a single unit (degraded but moving);
            # raise to 2 to demand both, which stops the robot when one dies.
            'min_active_lidars': 1,
            'min_range': 0.15,
            'max_range': 12.0,
            'beam_stride': 3,
            'max_points_per_lidar': 500,
            'footprint_padding': 0.04,
            'max_iterations': 18,
            'max_correspondence_distance': 0.45,
            'huber_delta': 0.08,
            'min_correspondences': 50,
            'max_translation_step': 0.25,
            'max_rotation_step': 0.30,
            'max_accepted_rmse': 0.18,
            'max_pose_jump_translation': 0.25,
            'max_pose_jump_rotation': 0.45,
            'correspondence_trim_ratio': 0.10,
            'lookup_grid_resolution': 0.04,
            'recovery_enabled': True,
            'recovery_after_rejections': 20,
            'recovery_search_translation': 0.20,
            'recovery_search_rotation': 0.17,
            # 回復探索の受入上限。フィールドは対称性が強く、遠い偽の
            # 局所解でもRMSEが小さく出ることがある(実機で1.4 m先へ
            # 誤ロックした事例あり)。探索は現在姿勢の±0.2 mが前提
            # なので、それを大きく超える解は「自信を持って間違う」より
            # REJECTEDのままにして操作者のRViz再指定を待つ。
            'recovery_max_jump_translation': 0.30,
            'recovery_max_jump_rotation': 0.30,
            'lidar_correction_gain': 0.35,
            'max_lidar_correction_translation': 0.04,
            'max_lidar_correction_rotation': 0.08,
            'settled_lidar_correction_gain': 0.50,
            'settled_max_lidar_correction_translation': 0.08,
            'settled_max_lidar_correction_rotation': 0.12,
            'settled_wheel_linear_speed': 0.04,
            'settled_wheel_angular_speed': 0.10,
            'wheel_odom_freshness_sec': 0.30,
            'use_wheel_odometry': False,
            'wheel_odom_topic': '/wheel/odometry',
            'motion_compensate_scans': True,
            'wheel_odom_history_sec': 2.0,
            'wheel_odom_stamp_tolerance_sec': 0.08,
            'wheel_odom_tf_timeout_sec': 1.5,
            'publish_tf': True,
            'pose_topic': '/localization/pose',
            'tracking_topic': '/localization/tracking_ok',
            'marker_topic': '/localization/markers',
            'motion_context_service': '/wall_localizer/set_motion_context',
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

    def _refresh_cached_parameters(self) -> None:
        for name in CACHED_PARAMETERS:
            self.params[name] = self.get_parameter(name).value

    def _cached_parameters_callback(self, parameters) -> SetParametersResult:
        for parameter in parameters:
            if parameter.name in self.params:
                self.params[parameter.name] = parameter.value
        return SetParametersResult(successful=True)

    @staticmethod
    def _period_ns_from_rate(rate_hz: float) -> int:
        if rate_hz <= 0.0:
            return 0
        return int(1.0e9 / max(0.1, rate_hz))

    @staticmethod
    def _publish_due(last_ns: int, now_ns: int, period_ns: int) -> bool:
        return period_ns > 0 and now_ns - last_ns >= period_ns

    @staticmethod
    def _padded_footprint(footprint: np.ndarray, padding: float) -> np.ndarray:
        center = np.mean(footprint, axis=0)
        vectors = footprint - center
        lengths = np.linalg.norm(vectors, axis=1)
        safe_lengths = np.maximum(lengths, 1.0e-9)
        return footprint + vectors / safe_lengths[:, None] * padding

    def _scan_callback(self, message: LaserScan, lidar: dict) -> None:
        if message.header.frame_id and message.header.frame_id != lidar['frame_id']:
            name = lidar['name']
            if name not in self.frame_warning_sent:
                self.get_logger().warning(
                    f'{name}: scan frame "{message.header.frame_id}" differs from '
                    f'configured frame "{lidar["frame_id"]}". Using robot.yaml pose.'
                )
                self.frame_warning_sent.add(name)

        ranges = np.asarray(message.ranges, dtype=float)
        if len(ranges) == 0:
            return
        indices = np.arange(0, len(ranges), self.beam_stride)
        selected_ranges = ranges[indices]
        valid = (
            np.isfinite(selected_ranges)
            & (selected_ranges >= max(self.min_range, float(message.range_min)))
            & (selected_ranges <= min(self.max_range, float(message.range_max)))
        )
        # An all-inf revolution is still a fresh, healthy sensor message. It
        # occurs when one LiDAR's 0.15 m blind zone is very close to a fixture
        # in the image-marked narrow lanes. The old early return left that
        # sensor's cache to expire and disabled otherwise well-constrained ICP
        # even though the opposite 360-degree LiDAR still supplied hundreds of
        # wall points. Keep an empty generation so freshness and geometric
        # observability are handled independently; optimize_pose will still
        # reject the combined set if it lacks min_correspondences.
        indices = indices[valid]
        selected_ranges = selected_ranges[valid]
        angles = float(message.angle_min) + indices * float(message.angle_increment)
        lidar_points = np.column_stack((
            selected_ranges * np.cos(angles),
            selected_ranges * np.sin(angles),
        ))

        lidar_pose = lidar['pose']
        c = math.cos(lidar_pose['yaw'])
        s = math.sin(lidar_pose['yaw'])
        base_points = np.column_stack((
            lidar_pose['x'] + c * lidar_points[:, 0] - s * lidar_points[:, 1],
            lidar_pose['y'] + s * lidar_points[:, 0] + c * lidar_points[:, 1],
        ))
        outside_footprint = ~points_in_polygon(base_points, self.footprint)
        base_points = base_points[outside_footprint]
        point_indices = indices[outside_footprint]

        if len(base_points) > self.max_points:
            sample_indices = np.linspace(
                0, len(base_points) - 1, self.max_points, dtype=int
            )
            base_points = base_points[sample_indices]
            point_indices = point_indices[sample_indices]

        received = self.get_clock().now()
        stamp_ns = message_stamp_nanoseconds(message.header.stamp)
        stamp_from_header = stamp_ns is not None
        if stamp_ns is None:
            stamp_ns = received.nanoseconds
        point_stamp_ns = (
            self._scan_point_stamps(message, point_indices, stamp_ns, len(ranges))
            if stamp_from_header
            else np.full(len(base_points), stamp_ns, dtype=np.int64)
        )

        name = str(lidar['name'])
        self.scan_generations[name] += 1
        self.latest_scans[name] = {
            'name': name,
            'points': base_points,
            'received': received,
            'stamp_ns': stamp_ns,
            'point_stamp_ns': point_stamp_ns,
            'generation': self.scan_generations[name],
        }

    @staticmethod
    def _scan_point_stamps(
        message: LaserScan,
        indices: np.ndarray,
        base_stamp_ns: int,
        range_count: int,
    ) -> np.ndarray:
        if len(indices) == 0:
            return np.zeros(0, dtype=np.int64)
        time_increment = float(message.time_increment)
        if time_increment <= 0.0 and range_count > 1:
            scan_time = float(message.scan_time)
            if scan_time > 0.0:
                time_increment = scan_time / float(range_count - 1)
        offsets = np.rint(indices.astype(float) * time_increment * 1.0e9)
        return int(base_stamp_ns) + offsets.astype(np.int64)

    def _initial_pose_callback(self, message: PoseWithCovarianceStamped) -> None:
        if message.header.frame_id and message.header.frame_id != self.map_frame:
            self.get_logger().warning(
                f'Ignoring /initialpose in frame "{message.header.frame_id}"; '
                f'expected "{self.map_frame}"'
            )
            return
        incoming = np.asarray(message.pose.covariance, dtype=float).reshape(6, 6)
        with self._pose_lock:
            self.pose = np.array([
                message.pose.pose.position.x,
                message.pose.pose.position.y,
                yaw_from_quaternion(message.pose.pose.orientation),
            ])
            self.covariance = incoming[np.ix_([0, 1, 5], [0, 1, 5])]
            self.pose_reset_generation += 1
        # Re-evaluate the most recent scan set against the operator-provided
        # pose immediately instead of waiting for the next LiDAR revolution.
        self.last_processed_scan_generations = None
        self.last_result = {
            'state': 'WAITING_FOR_SCANS_AFTER_INITIAL_POSE',
            'accepted': False,
            'correspondences': 0,
            'rmse': math.inf,
        }
        self.last_tracking_accept_time = -math.inf
        self.get_logger().info(
            f'Pose reset from RViz: [{self.pose[0]:.3f}, '
            f'{self.pose[1]:.3f}, {self.pose[2]:.3f}]'
        )

    def _wheel_odom_callback(self, message: Odometry) -> None:
        received = self.get_clock().now()
        stamp_ns = message_stamp_nanoseconds(message.header.stamp)
        if stamp_ns is None:
            stamp_ns = received.nanoseconds
        wheel_pose = np.array([
            message.pose.pose.position.x,
            message.pose.pose.position.y,
            yaw_from_quaternion(message.pose.pose.orientation),
        ])
        if not np.all(np.isfinite(wheel_pose)):
            self.get_logger().warning(
                'Ignoring non-finite wheel odometry pose',
                throttle_duration_sec=2.0,
            )
            return
        with self._pose_lock:
            if (
                self.latest_wheel_stamp_ns is not None
                and stamp_ns <= self.latest_wheel_stamp_ns
            ):
                return
            if message.header.frame_id:
                self.odom_frame = message.header.frame_id

            if self.previous_wheel_pose is not None:
                motion = relative_pose(self.previous_wheel_pose, wheel_pose)
                self.pose = compose_pose(self.pose, motion)
                self.latest_wheel_pose = wheel_pose
                if self.previous_wheel_stamp_ns is not None:
                    dt = (stamp_ns - self.previous_wheel_stamp_ns) * 1.0e-9
                    if dt > 1.0e-6:
                        self.latest_wheel_linear_speed = (
                            float(np.linalg.norm(motion[:2])) / dt
                        )
                        self.latest_wheel_angular_speed = (
                            abs(float(motion[2])) / dt
                        )
            else:
                self.latest_wheel_pose = wheel_pose
            self.previous_wheel_pose = wheel_pose.copy()
            self.previous_wheel_stamp_ns = stamp_ns
            self.latest_wheel_stamp_ns = stamp_ns
            self.wheel_generation += 1
            self.latest_wheel_received_ns = received.nanoseconds
            self._append_wheel_history(stamp_ns, wheel_pose)

    def _append_wheel_history(self, stamp_ns: int, pose: np.ndarray) -> None:
        entry = (int(stamp_ns), np.asarray(pose, dtype=float).copy())
        if self.wheel_history and stamp_ns < self.wheel_history[-1][0]:
            index = bisect_left(self.wheel_history_stamps, stamp_ns)
            if (
                index < len(self.wheel_history)
                and self.wheel_history[index][0] == stamp_ns
            ):
                self.wheel_history[index] = entry
                self.wheel_history_stamps[index] = int(stamp_ns)
            else:
                self.wheel_history.insert(index, entry)
                self.wheel_history_stamps.insert(index, int(stamp_ns))
        elif self.wheel_history and stamp_ns == self.wheel_history[-1][0]:
            self.wheel_history[-1] = entry
            self.wheel_history_stamps[-1] = int(stamp_ns)
        else:
            self.wheel_history.append(entry)
            self.wheel_history_stamps.append(int(stamp_ns))

        cutoff = stamp_ns - self.wheel_history_duration_ns
        while len(self.wheel_history) > 1 and self.wheel_history[0][0] < cutoff:
            self.wheel_history.pop(0)
            self.wheel_history_stamps.pop(0)

    def _wheel_pose_at(self, stamp_ns: int) -> Optional[np.ndarray]:
        if not self.wheel_history:
            return None

        first_stamp, first_pose = self.wheel_history[0]
        last_stamp, last_pose = self.wheel_history[-1]
        if stamp_ns <= first_stamp:
            if first_stamp - stamp_ns <= self.wheel_stamp_tolerance_ns:
                return first_pose.copy()
            return None
        if stamp_ns >= last_stamp:
            if stamp_ns - last_stamp <= self.wheel_stamp_tolerance_ns:
                return last_pose.copy()
            return None

        index = bisect_left(self.wheel_history_stamps, stamp_ns)
        before_stamp, before_pose = self.wheel_history[index - 1]
        after_stamp, after_pose = self.wheel_history[index]
        span = after_stamp - before_stamp
        if span <= 0:
            return after_pose.copy()
        fraction = (stamp_ns - before_stamp) / span
        return interpolate_pose(before_pose, after_pose, fraction)

    def _motion_compensated_points(
        self,
        scan_sets: Sequence[Dict[str, object]],
    ) -> Optional[Dict[str, object]]:
        reference_stamp_ns = max(
            self._latest_scan_point_stamp(scan) for scan in scan_sets
        )
        # The wheel callback runs concurrently with ICP.  Snapshot one
        # internally-consistent history under the lock, then do all vectorized
        # interpolation outside it so 100 Hz odometry remains responsive.
        with self._pose_lock:
            if (
                self.latest_wheel_pose is None
                or self.latest_wheel_stamp_ns is None
            ):
                return None
            latest_wheel_stamp_ns = int(self.latest_wheel_stamp_ns)
            # Wheel odometry arrives at a finite rate; accept scans that are up
            # to the stamp tolerance newer than the last wheel sample (the pose
            # history interpolation clamps to its newest entry).
            if (
                latest_wheel_stamp_ns + self.wheel_stamp_tolerance_ns
                < reference_stamp_ns
            ):
                return None
            reference_wheel_pose = self._wheel_pose_at(reference_stamp_ns)
            if reference_wheel_pose is None:
                return None
            history_stamps = np.asarray(
                self.wheel_history_stamps, dtype=np.int64
            )
            history_poses = np.asarray(
                [pose for _, pose in self.wheel_history], dtype=float
            )
            latest_wheel_pose = self.latest_wheel_pose.copy()
            wheel_generation = int(self.wheel_generation)
            pose_reset_generation = int(self.pose_reset_generation)
            current_pose = self.pose.copy()
        earliest_allowed = int(history_stamps[0]) - self.wheel_stamp_tolerance_ns
        latest_allowed = int(history_stamps[-1]) + self.wheel_stamp_tolerance_ns

        compensated_points = []
        for scan in scan_sets:
            points = np.asarray(scan['points'], dtype=float)
            point_stamps = np.asarray(
                scan.get(
                    'point_stamp_ns',
                    np.full(len(points), int(scan['stamp_ns']), dtype=np.int64),
                ),
                dtype=np.int64,
            )
            if len(point_stamps) and (
                int(point_stamps.min()) < earliest_allowed
                or int(point_stamps.max()) > latest_allowed
            ):
                return None
            point_wheel_poses = interpolate_poses_at(
                history_stamps, history_poses, point_stamps
            )
            compensated_points.append(
                transform_points_from_poses(
                    points, point_wheel_poses, reference_wheel_pose
                )
            )

        if not compensated_points:
            return None

        reference_to_current = relative_pose(
            reference_wheel_pose,
            latest_wheel_pose,
        )
        initial_pose = compose_pose(
            current_pose, inverse_pose(reference_to_current)
        )
        return {
            'points': np.vstack(compensated_points),
            'initial_pose': initial_pose,
            'reference_to_current': reference_to_current,
            'motion_compensated': True,
            'reference_stamp_ns': reference_stamp_ns,
            'latest_wheel_stamp_ns': latest_wheel_stamp_ns,
            'wheel_snapshot_pose': latest_wheel_pose,
            'wheel_snapshot_stamp_ns': latest_wheel_stamp_ns,
            'wheel_snapshot_generation': wheel_generation,
            'pose_reset_generation': pose_reset_generation,
        }

    @staticmethod
    def _latest_scan_point_stamp(scan: Dict[str, object]) -> int:
        point_stamps = scan.get('point_stamp_ns')
        if point_stamps is None or len(point_stamps) == 0:
            return int(scan['stamp_ns'])
        return int(np.max(point_stamps))

    def _recent_points(self) -> Tuple[Optional[Dict[str, object]], str]:
        now = self.get_clock().now()
        scans = []
        missing = []
        for lidar in self.robot['lidars']:
            name = str(lidar['name'])
            scan = self.latest_scans.get(name)
            if scan is None or now - scan['received'] > self.scan_timeout:
                missing.append(name)
                continue
            scans.append(scan)
        # A single A2M8 sees 360 degrees, so one unit that reaches two
        # non-parallel walls still constrains x, y and yaw.  Refusing to
        # localize at all when one LiDAR is dead only converts a degraded robot
        # into a stopped one, so match on whatever is live and let
        # min_active_lidars be the explicit floor.
        if len(scans) < self.min_active_lidars or not scans:
            self.active_lidar_count = len(scans)
            self.missing_lidars = tuple(missing)
            return None, 'WAITING_FOR_SCANS'
        if missing:
            self.get_logger().warning(
                f'Localizing on {len(scans)}/{len(self.robot["lidars"])} LiDARs; '
                f'no fresh scan from {", ".join(missing)}',
                throttle_duration_sec=5.0,
            )
        self.active_lidar_count = len(scans)
        self.missing_lidars = tuple(missing)

        # Key the generations by LiDAR name.  A bare tuple of counters changes
        # meaning when the set of live LiDARs changes, so (front, 7) could look
        # unchanged next to a previous (rear, 7) and freeze matching at a stale
        # pose while still reporting READY.
        scan_generations = tuple(
            (str(scan['name']), int(scan['generation'])) for scan in scans
        )
        # Keep running the timer so stale inputs still transition to the
        # waiting diagnostic above, but avoid rebuilding motion-compensated
        # point clouds when both LiDAR generations are unchanged.
        if not scan_generations_changed(
            scan_generations,
            self.last_processed_scan_generations,
        ):
            return {'scan_generations': scan_generations}, 'READY'
        if self.use_wheel_odometry and self.motion_compensate_scans:
            compensated = self._motion_compensated_points(scans)
            if compensated is not None:
                compensated['scan_generations'] = scan_generations
                return compensated, 'READY'
            # Wheel odometry missing or stale: degrade to uncompensated
            # matching instead of freezing localization. Uncompensated scans
            # are only wrong while moving fast, whereas no localization at
            # all strands the whole system.
            self.get_logger().warning(
                'Wheel odometry unavailable/stale; localizing without '
                'motion compensation',
                throttle_duration_sec=5.0,
            )

        with self._pose_lock:
            initial_pose = self.pose.copy()
            wheel_snapshot_pose = (
                None
                if self.latest_wheel_pose is None
                else self.latest_wheel_pose.copy()
            )
            wheel_snapshot_stamp_ns = self.latest_wheel_stamp_ns
            wheel_snapshot_generation = int(self.wheel_generation)
            pose_reset_generation = int(self.pose_reset_generation)

        return {
            'points': np.vstack([scan['points'] for scan in scans]),
            'initial_pose': initial_pose,
            'reference_to_current': np.zeros(3),
            'motion_compensated': False,
            'scan_generations': scan_generations,
            'wheel_snapshot_pose': wheel_snapshot_pose,
            'wheel_snapshot_stamp_ns': wheel_snapshot_stamp_ns,
            'wheel_snapshot_generation': wheel_snapshot_generation,
            'pose_reset_generation': pose_reset_generation,
        }, 'READY'

    def _timer_callback(self) -> None:
        started = time.perf_counter()
        scan_data, waiting_state = self._recent_points()
        if scan_data is None:
            self.last_result = {
                'state': waiting_state,
                'accepted': False,
                'correspondences': 0,
                'rmse': math.inf,
            }
        elif scan_generations_changed(
            scan_data['scan_generations'],
            self.last_processed_scan_generations,
        ):
            self.last_processed_scan_generations = tuple(
                scan_data['scan_generations']
            )
            points = scan_data['points']
            before = scan_data['initial_pose']
            result = self._optimize(points, before)
            accepted, settled = self._result_accepted(result, before)
            recovered = False
            if accepted:
                self.rejected_streak = 0
            else:
                self.rejected_streak += 1
                recovery_result = self._attempt_recovery(points, before)
                if recovery_result is not None:
                    result = recovery_result
                    accepted = True
                    recovered = True
                    self.rejected_streak = 0
            if accepted:
                corrected_pose = compose_pose(
                    result.pose,
                    scan_data['reference_to_current'],
                )
                with self._pose_lock:
                    if scan_data.get('pose_reset_generation') != int(
                        self.pose_reset_generation
                    ):
                        corrected_at_commit = None
                    else:
                        corrected_at_commit = (
                            advance_corrected_pose_to_wheel_snapshot(
                                corrected_pose,
                                snapshot_wheel_pose=scan_data.get(
                                    'wheel_snapshot_pose'
                                ),
                                snapshot_stamp_ns=scan_data.get(
                                    'wheel_snapshot_stamp_ns'
                                ),
                                snapshot_generation=scan_data.get(
                                    'wheel_snapshot_generation'
                                ),
                                latest_wheel_pose=self.latest_wheel_pose,
                                latest_stamp_ns=self.latest_wheel_stamp_ns,
                                latest_generation=self.wheel_generation,
                            )
                        )
                    if corrected_at_commit is None:
                        accepted = False
                        recovered = False
                        self.rejected_streak += 1
                        applied_correction = np.zeros(3)
                    else:
                        new_pose, applied_correction = (
                            self._limited_lidar_correction(
                                corrected_at_commit,
                                # Recovery selects a new ICP basin, but its
                                # correction must still enter the control pose
                                # gradually. An instantaneous relock can look
                                # like robot motion and falsely complete a path.
                                bypass_limits=False,
                            )
                        )
                        self.pose = new_pose
                        self.covariance = result.covariance
                if not accepted:
                    self.get_logger().warning(
                        'Rejected ICP commit because the wheel snapshot was '
                        'inconsistent or /initialpose changed during solve',
                        throttle_duration_sec=2.0,
                    )
            else:
                applied_correction = np.zeros(3)
            odom_delta = scan_data['reference_to_current']
            self.last_result = {
                'state': 'TRACKING' if accepted else 'REJECTED',
                'accepted': accepted,
                'recovered': recovered,
                'rejected_streak': int(self.rejected_streak),
                'converged': bool(result.converged),
                'correspondences': int(result.correspondences),
                'rmse': float(result.rmse),
                'iterations': int(result.iterations),
                'final_translation_step': float(result.final_translation_step),
                'final_rotation_step': float(result.final_rotation_step),
                'points': int(len(points)),
                'motion_compensated': bool(scan_data['motion_compensated']),
                'odom_delta_since_scan': [
                    round(float(value), 6) for value in odom_delta
                ],
                'applied_lidar_correction': [
                    round(float(value), 6) for value in applied_correction
                ],
            }
            if accepted:
                self.last_tracking_accept_time = time.monotonic()
        self.last_result['optimization_ms'] = round(
            (time.perf_counter() - started) * 1000.0,
            3,
        )
        self.last_result['wheel_speed'] = {
            'linear': round(float(self.latest_wheel_linear_speed), 4),
            'angular': round(float(self.latest_wheel_angular_speed), 4),
        }

        now = self.get_clock().now()
        stamp = now.to_msg()
        try:
            now_ns = now.nanoseconds
            if self._publish_due(
                self.last_marker_publish_ns,
                now_ns,
                self.marker_publish_period_ns,
            ):
                self._publish_markers(stamp)
                self.last_marker_publish_ns = now_ns
            if self._publish_due(
                self.last_diagnostic_publish_ns,
                now_ns,
                self.diagnostic_publish_period_ns,
            ):
                self._publish_diagnostics(stamp)
                self.last_diagnostic_publish_ns = now_ns
        except RCLError:
            if rclpy.ok():
                raise

    def _optimize(self, points: np.ndarray, initial_pose: np.ndarray):
        return optimize_pose(
            points,
            self.field['walls'],
            initial_pose,
            max_iterations=int(self.params['max_iterations']),
            max_correspondence_distance=float(
                self.params['max_correspondence_distance']
            ),
            huber_delta=float(self.params['huber_delta']),
            min_correspondences=int(self.params['min_correspondences']),
            max_translation_step=float(self.params['max_translation_step']),
            max_rotation_step=float(self.params['max_rotation_step']),
            lookup=self.wall_lookup,
            trim_ratio=float(self.params['correspondence_trim_ratio']),
        )

    def _result_accepted(self, result, before: np.ndarray) -> Tuple[bool, bool]:
        jump = relative_pose(before, result.pose)
        settled = bool(
            result.converged
            or (
                result.final_translation_step <= 0.025
                and result.final_rotation_step <= 0.010
            )
        )
        accepted = bool(
            settled
            and result.correspondences >= int(self.params['min_correspondences'])
            and math.isfinite(result.rmse)
            and result.rmse <= float(self.params['max_accepted_rmse'])
            and np.linalg.norm(jump[:2])
            <= float(self.params['max_pose_jump_translation'])
            and abs(jump[2]) <= float(self.params['max_pose_jump_rotation'])
        )
        return accepted, settled

    def _wheel_stationary(self) -> bool:
        if not self.use_wheel_odometry:
            return True
        with self._pose_lock:
            latest_wheel_pose = self.latest_wheel_pose
            latest_wheel_received_ns = self.latest_wheel_received_ns
            latest_wheel_linear_speed = self.latest_wheel_linear_speed
            latest_wheel_angular_speed = self.latest_wheel_angular_speed
        if latest_wheel_pose is None or latest_wheel_received_ns is None:
            return False
        age = (
            self.get_clock().now().nanoseconds
            - latest_wheel_received_ns
        ) * 1.0e-9
        if (
            not math.isfinite(age)
            or age < 0.0
            or age > float(self.params['wheel_odom_freshness_sec'])
        ):
            return False
        return (
            latest_wheel_linear_speed
            <= float(self.params['settled_wheel_linear_speed'])
            and latest_wheel_angular_speed
            <= float(self.params['settled_wheel_angular_speed'])
        )

    def _attempt_recovery(self, points: np.ndarray, before: np.ndarray):
        """Multi-start search around the current pose after persistent rejection.

        Only runs while the wheels report the robot as stationary, so a large
        re-lock jump cannot happen mid-motion; the path follower's progress
        watchdog already stops the robot when localization stalls.
        """
        if not bool(self.params['recovery_enabled']):
            return None
        if self.rejected_streak < int(self.params['recovery_after_rejections']):
            return None
        if not self._wheel_stationary():
            return None

        radius = float(self.params['recovery_search_translation'])
        rotation = float(self.params['recovery_search_rotation'])
        max_jump_translation = float(
            self.params['recovery_max_jump_translation']
        )
        max_jump_rotation = float(self.params['recovery_max_jump_rotation'])
        offsets = [
            (radius, 0.0, 0.0), (-radius, 0.0, 0.0),
            (0.0, radius, 0.0), (0.0, -radius, 0.0),
            (0.0, 0.0, rotation), (0.0, 0.0, -rotation),
            (radius, radius, 0.0), (-radius, -radius, 0.0),
            (radius, -radius, rotation), (-radius, radius, -rotation),
        ]
        best = None
        for offset in offsets:
            candidate = np.array([
                before[0] + offset[0],
                before[1] + offset[1],
                normalize_angle(float(before[2]) + offset[2]),
            ])
            result = self._optimize(points, candidate)
            settled = bool(
                result.converged
                or (
                    result.final_translation_step <= 0.025
                    and result.final_rotation_step <= 0.010
                )
            )
            # 現在姿勢から遠すぎる解は、RMSEが良くても対称な壁配置への
            # 誤ロックの可能性が高いので受け入れない(上のパラメータ参照)。
            jump = relative_pose(before, result.pose)
            if (
                settled
                and float(np.linalg.norm(jump[:2])) <= max_jump_translation
                and abs(float(jump[2])) <= max_jump_rotation
                and result.correspondences
                >= int(self.params['min_correspondences'])
                and math.isfinite(result.rmse)
                and result.rmse <= float(self.params['max_accepted_rmse'])
                and (best is None or result.rmse < best.rmse)
            ):
                best = result
        if best is not None:
            self.get_logger().warning(
                'Recovered localization after '
                f'{self.rejected_streak} rejections: pose='
                f'[{best.pose[0]:.3f}, {best.pose[1]:.3f}, {best.pose[2]:.3f}], '
                f'rmse={best.rmse:.4f}'
            )
        return best

    def _limited_lidar_correction(
        self,
        corrected_pose: np.ndarray,
        bypass_limits: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        correction = relative_pose(self.pose, np.asarray(corrected_pose, dtype=float))
        if bypass_limits:
            return np.asarray(corrected_pose, dtype=float).copy(), correction
        settled = self._wheel_stationary()
        if settled:
            gain = float(self.params['settled_lidar_correction_gain'])
            max_translation = float(
                self.params['settled_max_lidar_correction_translation']
            )
            max_rotation = float(
                self.params['settled_max_lidar_correction_rotation']
            )
        else:
            gain = float(self.params['lidar_correction_gain'])
            max_translation = float(
                self.params['max_lidar_correction_translation']
            )
            max_rotation = float(self.params['max_lidar_correction_rotation'])
        gain = max(0.0, min(1.0, gain))
        correction = correction * gain

        if max_translation >= 0.0:
            translation = float(np.linalg.norm(correction[:2]))
            if translation > max_translation and translation > 1.0e-12:
                correction[:2] *= max_translation / translation

        if max_rotation >= 0.0:
            correction[2] = float(
                np.clip(correction[2], -max_rotation, max_rotation)
            )

        correction[2] = normalize_angle(float(correction[2]))
        return compose_pose(self.pose, correction), correction

    def _fast_publish_callback(self) -> None:
        now = self.get_clock().now()
        pose_stamp = now.to_msg()
        tf_stamp = (
            now + Duration(seconds=self.tf_future_tolerance)
        ).to_msg()
        try:
            self._publish_pose(pose_stamp)
            self._publish_tf(tf_stamp)
            # Health is a heartbeat, not an ICP completion event. Publishing
            # it only at the 4 Hz solve rate made an otherwise valid 0.5--0.7 s
            # solve look like a dead localizer to RuntimeGuard. Wheel odometry
            # continues propagating the pose while ICP is in flight, so retain
            # the last accepted solution for one bounded solve interval. A
            # rejected update, missing scans, pose reset, or genuinely stuck
            # optimizer still changes this to false immediately / on timeout.
            tracking = Bool()
            tracking.data = bool(
                self.last_result.get('state') == 'TRACKING'
                and time.monotonic() - self.last_tracking_accept_time
                <= self.tracking_solution_timeout_sec
            )
            self.tracking_publisher.publish(tracking)
        except RCLError:
            if rclpy.ok():
                raise

    def _publish_pose(self, stamp) -> None:
        with self._pose_lock:
            pose = self.pose
            pose_covariance = self.covariance
        message = PoseWithCovarianceStamped()
        message.header.stamp = stamp
        message.header.frame_id = self.map_frame
        message.pose.pose.position.x = float(pose[0])
        message.pose.pose.position.y = float(pose[1])
        set_yaw(message.pose.pose.orientation, float(pose[2]))

        covariance = np.zeros((6, 6))
        covariance[2, 2] = 1.0e3
        covariance[3, 3] = 1.0e3
        covariance[4, 4] = 1.0e3
        covariance[np.ix_([0, 1, 5], [0, 1, 5])] = pose_covariance
        message.pose.covariance = covariance.reshape(-1).tolist()
        self.pose_publisher.publish(message)

    def _wheel_odometry_alive(self, received_ns: Optional[int]) -> bool:
        """Report whether measurement_wheel is currently supplying odom->base_link."""
        if not self.use_wheel_odometry or received_ns is None:
            return False
        age_ns = self.get_clock().now().nanoseconds - int(received_ns)
        return age_ns <= self.wheel_odom_tf_timeout_ns

    @staticmethod
    def _transform_message(stamp, parent: str, child: str, pose) -> TransformStamped:
        transform = TransformStamped()
        transform.header.stamp = stamp
        transform.header.frame_id = parent
        transform.child_frame_id = child
        transform.transform.translation.x = float(pose[0])
        transform.transform.translation.y = float(pose[1])
        set_yaw(transform.transform.rotation, float(pose[2]))
        return transform

    def _publish_tf(self, stamp) -> None:
        if not self.publish_tf:
            return
        with self._pose_lock:
            pose = self.pose
            wheel_pose = self.latest_wheel_pose
            wheel_received_ns = self.latest_wheel_received_ns

        # map->odomはLiDAR補正、odom->base_linkはmeasurement_wheelの車輪
        # オドメトリ、というのが正常時の分担。計測輪ボード未接続などで
        # measurement_wheelが起動できないと後者の配信者が誰もいなくなり、
        # base_link以下(ロボット表示・LiDARの点群・Nav2のローカルコスト
        # マップ)がTF木から切り離されてRVizから丸ごと消える。車輪が居ない
        # 間はこのノードが代替配信してTF木を必ず繋いだままにする。
        odom_source_alive = self._wheel_odometry_alive(wheel_received_ns)
        self._log_odom_source_change(odom_source_alive)
        self.tf_broadcaster.sendTransform([
            self._transform_message(stamp, parent, child, transform_pose)
            for parent, child, transform_pose in localizer_tf_chain(
                pose,
                wheel_pose,
                odom_source_alive,
                map_frame=self.map_frame,
                odom_frame=self.odom_frame,
                base_frame=self.base_frame,
            )
        ])

    def _log_odom_source_change(self, odom_source_alive: bool) -> None:
        if odom_source_alive == self._odom_source_alive:
            return
        self._odom_source_alive = odom_source_alive
        # A CPU spike on the Jetson can delay the wheel callback past the
        # timeout even though measurement_wheel is healthy. That substitute is
        # harmless (it repeats the last wheel transform), so throttle the log
        # rather than letting a stall spam the console.
        if odom_source_alive:
            self.get_logger().info(
                'Wheel odometry is publishing '
                f'{self.odom_frame}->{self.base_frame}; '
                'stopped the localizer TF substitute',
                throttle_duration_sec=5.0,
            )
        elif self.use_wheel_odometry:
            self.get_logger().warning(
                'No wheel odometry on '
                f'{self.get_parameter("wheel_odom_topic").value}; publishing '
                f'{self.odom_frame}->{self.base_frame} here so the TF tree '
                'stays connected. Localization is LiDAR-only until the '
                'measurement wheels come back.',
                throttle_duration_sec=5.0,
            )

    def _publish_markers(self, stamp) -> None:
        markers = MarkerArray()
        now_ns = self.get_clock().now().nanoseconds
        if self._publish_due(
            self.last_field_marker_publish_ns,
            now_ns,
            self.field_marker_period_ns,
        ):
            markers.markers.extend(self._field_markers(stamp))
            self.last_field_marker_publish_ns = now_ns
        markers.markers.extend(self._robot_markers(stamp))
        self.marker_publisher.publish(markers)

    def _field_markers(self, stamp) -> List[Marker]:
        if self._field_markers_cache is not None:
            for marker in self._field_markers_cache:
                marker.header.stamp = stamp
            return self._field_markers_cache
        field_markers: List[Marker] = []

        walls = Marker()
        walls.header.frame_id = self.map_frame
        walls.header.stamp = stamp
        walls.ns = 'field'
        walls.id = 0
        walls.type = Marker.LINE_LIST
        walls.action = Marker.ADD
        walls.pose.orientation.w = 1.0
        walls.scale.x = 0.07
        walls.color.r = 0.05
        walls.color.g = 0.95
        walls.color.b = 1.0
        walls.color.a = 1.0
        for x1, y1, x2, y2 in self.field['walls']:
            walls.points.append(Point(x=float(x1), y=float(y1), z=0.04))
            walls.points.append(Point(x=float(x2), y=float(y2), z=0.04))
        field_markers.append(walls)

        wall_surfaces = Marker()
        wall_surfaces.header.frame_id = self.map_frame
        wall_surfaces.header.stamp = stamp
        wall_surfaces.ns = 'field'
        wall_surfaces.id = 6
        wall_surfaces.type = Marker.TRIANGLE_LIST
        wall_surfaces.action = Marker.ADD
        wall_surfaces.pose.orientation.w = 1.0
        wall_surfaces.scale.x = 1.0
        wall_surfaces.scale.y = 1.0
        wall_surfaces.scale.z = 1.0
        wall_surfaces.color.r = 0.0
        wall_surfaces.color.g = 0.85
        wall_surfaces.color.b = 1.0
        wall_surfaces.color.a = 0.80
        half_width = 0.045
        for x1, y1, x2, y2 in self.field['walls']:
            direction = np.array([x2 - x1, y2 - y1], dtype=float)
            length = float(np.linalg.norm(direction))
            if length < 1.0e-9:
                continue
            perpendicular = (
                np.array([-direction[1], direction[0]]) / length * half_width
            )
            start = np.array([x1, y1], dtype=float)
            end = np.array([x2, y2], dtype=float)
            corners = (
                start + perpendicular,
                start - perpendicular,
                end - perpendicular,
                end + perpendicular,
            )
            for index in (0, 1, 2, 0, 2, 3):
                point = corners[index]
                wall_surfaces.points.append(
                    Point(x=float(point[0]), y=float(point[1]), z=0.035)
                )
        field_markers.append(wall_surfaces)

        for wall_index, (x1, y1, x2, y2) in enumerate(self.field['walls']):
            wall_block = Marker()
            wall_block.header.frame_id = self.map_frame
            wall_block.header.stamp = stamp
            wall_block.ns = 'field'
            wall_block.id = 1000 + wall_index
            wall_block.type = Marker.CUBE
            wall_block.action = Marker.ADD
            wall_block.pose.position.x = float((x1 + x2) * 0.5)
            wall_block.pose.position.y = float((y1 + y2) * 0.5)
            wall_block.pose.position.z = 0.10
            set_yaw(
                wall_block.pose.orientation,
                math.atan2(float(y2 - y1), float(x2 - x1)),
            )
            wall_block.scale.x = float(math.hypot(x2 - x1, y2 - y1))
            wall_block.scale.y = 0.10
            wall_block.scale.z = 0.20
            wall_block.color.r = 0.0
            wall_block.color.g = 0.85
            wall_block.color.b = 1.0
            wall_block.color.a = 0.85
            field_markers.append(wall_block)

        self._field_markers_cache = field_markers
        return field_markers

    def _robot_markers(self, stamp) -> List[Marker]:
        markers = MarkerArray()

        footprint = Marker()
        footprint.header.frame_id = self.base_frame
        footprint.header.stamp = stamp
        footprint.ns = 'robot'
        footprint.id = 1
        footprint.type = Marker.LINE_STRIP
        footprint.action = Marker.ADD
        footprint.pose.orientation.w = 1.0
        footprint.scale.x = 0.035
        footprint.color.r = 1.0
        footprint.color.g = 0.75
        footprint.color.b = 0.1
        footprint.color.a = 1.0
        closed = np.vstack((self.robot['footprint'], self.robot['footprint'][0]))
        for x, y in closed:
            footprint.points.append(Point(x=float(x), y=float(y), z=0.05))
        markers.markers.append(footprint)

        body = Marker()
        body.header.frame_id = self.base_frame
        body.header.stamp = stamp
        body.ns = 'robot'
        body.id = 4
        body.type = Marker.TRIANGLE_LIST
        body.action = Marker.ADD
        body.pose.orientation.w = 1.0
        body.scale.x = 1.0
        body.scale.y = 1.0
        body.scale.z = 1.0
        body.color.r = 1.0
        body.color.g = 0.1
        body.color.b = 0.8
        body.color.a = 0.18
        footprint_center = np.mean(self.robot['footprint'], axis=0)
        for index in range(len(self.robot['footprint'])):
            start = self.robot['footprint'][index]
            end = self.robot['footprint'][
                (index + 1) % len(self.robot['footprint'])
            ]
            for x, y in (footprint_center, start, end):
                body.points.append(Point(x=float(x), y=float(y), z=0.035))
        markers.markers.append(body)

        if self.robot['cad_model_file']:
            robot_mesh = Marker()
            robot_mesh.header.frame_id = self.base_frame
            robot_mesh.header.stamp = stamp
            robot_mesh.ns = 'robot'
            robot_mesh.id = 7
            robot_mesh.type = Marker.MESH_RESOURCE
            robot_mesh.action = Marker.ADD
            robot_mesh.mesh_resource = Path(
                self.robot['cad_model_file']
            ).expanduser().resolve().as_uri()
            robot_mesh.mesh_use_embedded_materials = False
            scale = float(self.robot['cad_scale'])
            robot_mesh.scale.x = scale
            robot_mesh.scale.y = scale
            robot_mesh.scale.z = scale
            yaw = float(self.robot['cad_to_base_yaw'])
            set_yaw(robot_mesh.pose.orientation, yaw)
            origin = self.robot['cad_origin_mm'] * scale
            cosine = math.cos(yaw)
            sine = math.sin(yaw)
            robot_mesh.pose.position.x = float(
                -(cosine * origin[0] - sine * origin[1])
            )
            robot_mesh.pose.position.y = float(
                -(sine * origin[0] + cosine * origin[1])
            )
            robot_mesh.pose.position.z = float(-origin[2])
            robot_mesh.color.r = 0.72
            robot_mesh.color.g = 0.76
            robot_mesh.color.b = 0.82
            robot_mesh.color.a = 0.88
            markers.markers.append(robot_mesh)

        position_marker = Marker()
        position_marker.header.frame_id = self.base_frame
        position_marker.header.stamp = stamp
        position_marker.ns = 'robot'
        position_marker.id = 5
        position_marker.type = Marker.SPHERE
        position_marker.action = Marker.ADD
        position_marker.pose.position.z = 0.14
        position_marker.pose.orientation.w = 1.0
        position_marker.scale.x = 0.36
        position_marker.scale.y = 0.36
        position_marker.scale.z = 0.18
        position_marker.color.r = 1.0
        position_marker.color.g = 0.0
        position_marker.color.b = 0.8
        position_marker.color.a = 0.95
        markers.markers.append(position_marker)

        heading = Marker()
        heading.header.frame_id = self.base_frame
        heading.header.stamp = stamp
        heading.ns = 'robot'
        heading.id = 2
        heading.type = Marker.ARROW
        heading.action = Marker.ADD
        heading.pose.orientation.w = 1.0
        heading.scale.x = 0.80
        heading.scale.y = 0.14
        heading.scale.z = 0.14
        heading.color.r = 1.0
        heading.color.g = 0.3
        heading.color.b = 0.1
        heading.color.a = 1.0
        markers.markers.append(heading)

        status = Marker()
        status.header.frame_id = self.map_frame
        status.header.stamp = stamp
        status.ns = 'status'
        status.id = 3
        status.type = Marker.TEXT_VIEW_FACING
        status.action = Marker.ADD
        status.pose.position.x = float(self.pose[0])
        status.pose.position.y = float(self.pose[1])
        status.pose.position.z = 0.85
        status.pose.orientation.w = 1.0
        status.scale.z = 0.25
        status.color.a = 1.0
        status.color.g = 1.0 if self.last_result['accepted'] else 0.35
        status.color.r = 0.15 if self.last_result['accepted'] else 1.0
        rmse = self.last_result.get('rmse', math.inf)
        rmse_text = f'{rmse:.3f} m' if math.isfinite(rmse) else '--'
        status.text = (
            f'{self.last_result["state"]}  '
            f'x={self.pose[0]:.2f}  y={self.pose[1]:.2f}  '
            f'yaw={math.degrees(self.pose[2]):.1f} deg  RMSE={rmse_text}'
        )
        markers.markers.append(status)

        return markers.markers

    def _publish_diagnostics(self, stamp) -> None:
        array = DiagnosticArray()
        array.header.stamp = stamp
        status = DiagnosticStatus()
        status.name = 'dual_rplidar_localization/wall_localizer'
        status.hardware_id = 'dual_rplidar_a2m8'
        state = self.last_result['state']
        degraded = bool(self.missing_lidars)
        if state == 'TRACKING' and not degraded:
            status.level = DiagnosticStatus.OK
            status.message = 'Wall localization is tracking'
        elif state == 'TRACKING':
            # Still usable, so autonomy stays enabled, but the operator must be
            # able to see from /diagnostics that a LiDAR is gone.
            status.level = DiagnosticStatus.WARN
            status.message = (
                'Wall localization is tracking on '
                f'{self.active_lidar_count}/{len(self.robot["lidars"])} LiDARs '
                f'(no scan from {", ".join(self.missing_lidars)})'
            )
        else:
            status.level = DiagnosticStatus.WARN
            status.message = state

        values = dict(self.last_result)
        values['active_lidars'] = self.active_lidar_count
        values['missing_lidars'] = list(self.missing_lidars)
        values['pose'] = [round(float(value), 5) for value in self.pose]
        for key, value in values.items():
            if isinstance(value, np.generic):
                value = value.item()
            if isinstance(value, float) and not math.isfinite(value):
                text = 'inf'
            else:
                text = json.dumps(value)
            status.values.append(KeyValue(key=str(key), value=text))
        array.status.append(status)
        self.diagnostic_publisher.publish(array)


def main(args=None) -> None:
    rclpy.init(args=args)
    try:
        node = WallLocalizer()
    except (ConfigError, ValueError) as error:
        rclpy.logging.get_logger('wall_localizer').fatal(str(error))
        rclpy.shutdown()
        raise
    executor = None
    try:
        # Keep TF, wheel odometry, and scan ingestion responsive while the
        # CPU-heavy ICP callback is running on the Jetson's six CPU cores.
        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(node)
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except RCLError:
        if rclpy.ok():
            raise
    finally:
        if executor is not None:
            try:
                executor.shutdown(timeout_sec=1.0)
            except RCLError:
                pass
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
