"""Self-hit filter for the Nav2 costmap scan inputs.

両LiDARはロボットの対角コーナーに載っており、機体フレーム・配線・ホイールが
ビームに映る。wall_localizerとgoal_orchestratorは自前でフットプリント内の点を
除去しているが、Nav2のobstacle_layerだけは生の/scan_front・/scan_rearを購読
していたため、自機反射がフットプリント内の致死セルとして焼き付き、DWBが
「No valid trajectories out of 474!」で全軌道を棄却→failure_tolerance経由の
アボート→リカバリ、という周期的な停止(移動障害物が無い場所での「迷い」)を
起こしていた。

このノードは各LiDARについて、フットプリント凸包(+padding)から抜けるまでの
ビーム距離をマウント姿勢から角度ごとに事前計算し、それ以下のレンジをNaNに
落とした <topic>_filtered を出力する。NaN化した方向は機体が物理的に視界を
塞いでいるので、マーキングもクリアも情報損失はない。

field_config_file を渡すと、さらに既知の静的壁(field_planning.yaml の線分
から wall_match_distance 以内)のヒットを +inf に置き換えた <topic>_nowalls
も出力する。これはグローバルコストマップの obstacle_layer 専用の入力:
壁は static_layer が既に持っているので、未知の動的障害物(人・相手ロボット)
だけをマーキングさせ、(1) 自己位置の数cm誤差で壁セルがフィールド内側へ
膨らんでスタートゾーンの計画が壊れるのを防ぎ、(2) inf は inf_is_valid: true
でクリアレイに変換されるため、障害物が退いた跡のセルが壁裏のレイに遮られず
確実に消える。自己位置が古い間は壁も含めてそのまま流す(短時間のドロップ
アウトで障害物検出を止めない安全側)。
"""

import math
from typing import Dict, Optional

import numpy as np
import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from omni_autonomy_interfaces.srv import SetMotionContext
from rclpy._rclpy_pybind11 import RCLError
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import LaserScan

from .config import ConfigError, load_field, load_robot
from .competition_footprint import (
    FootprintGeometryError,
    canonical_convex_polygon,
    footprint_digest,
)
from .geometry import WallLookupGrid, transform_points
from .scan_self_reflections import self_reflection_mask
from .scan_freshness import scan_metadata_valid, timestamp_is_fresh


def ray_exit_distances(
    origin: np.ndarray,
    directions: np.ndarray,
    polygon: np.ndarray,
) -> np.ndarray:
    """Distance from `origin` along each direction to the LAST crossing of
    the polygon boundary (0 when the ray never crosses, e.g. the origin is
    outside and pointing away)."""
    starts = polygon
    vectors = np.roll(polygon, -1, axis=0) - polygon
    offsets = starts[None, :, :] - origin[None, None, :]

    # t*d x v = (S-O) x v / (d x v), s from the edge side. (N rays, M edges)
    d_cross_v = (
        directions[:, None, 0] * vectors[None, :, 1]
        - directions[:, None, 1] * vectors[None, :, 0]
    )
    off_cross_v = (
        offsets[:, :, 0] * vectors[None, :, 1]
        - offsets[:, :, 1] * vectors[None, :, 0]
    )
    off_cross_d = (
        offsets[:, :, 0] * directions[:, None, 1]
        - offsets[:, :, 1] * directions[:, None, 0]
    )
    with np.errstate(divide='ignore', invalid='ignore'):
        t = off_cross_v / d_cross_v
        s = off_cross_d / d_cross_v
    valid = (
        np.isfinite(t)
        & (t > 0.0)
        & (s >= -1.0e-9)
        & (s <= 1.0 + 1.0e-9)
    )
    t = np.where(valid, t, -np.inf)
    exits = np.max(t, axis=1)
    return np.where(np.isfinite(exits), np.maximum(exits, 0.0), 0.0)


class ScanFootprintFilter(Node):
    def __init__(self) -> None:
        super().__init__('scan_footprint_filter')
        self.declare_parameter('robot_config_file', '')
        # 凸包の外側にはみ出す配線・ネジ頭などの取り残し防止マージン。
        # localizerのfootprint_padding(0.04)と同思想で、少し広めにする。
        self.declare_parameter('footprint_padding', 0.05)
        self.declare_parameter('output_topic_suffix', '_filtered')
        # 既知壁の分類(follower/orchestratorと同じ考え方): 静的壁の線分から
        # この距離以内のヒットは「既知の壁」。自己位置の受入RMSE上限0.14 m
        # より広め。
        self.declare_parameter('field_config_file', '')
        self.declare_parameter('nowalls_topic_suffix', '_nowalls')
        self.declare_parameter('wall_match_distance', 0.20)
        self.declare_parameter('pose_topic', '/localization/pose')
        self.declare_parameter('pose_timeout_sec', 0.5)
        self.declare_parameter(
            'motion_context_service',
            '/scan_footprint_filter/set_motion_context',
        )

        robot_path = str(self.get_parameter('robot_config_file').value)
        if not robot_path:
            raise ConfigError(
                'robot_config_file must be set by the launch file'
            )
        robot = load_robot(robot_path)
        self.base_frame = str(robot['base_frame_id'])
        self.footprint = np.asarray(robot['footprint'], dtype=float)
        self._motion_context_revision = 0
        self._motion_context_execution_id = 0
        self._motion_context_digest = ''
        self.padding = float(self.get_parameter('footprint_padding').value)
        suffix = str(self.get_parameter('output_topic_suffix').value)
        nowalls_suffix = str(
            self.get_parameter('nowalls_topic_suffix').value
        )

        self.wall_lookup: Optional[WallLookupGrid] = None
        self.pose: Optional[np.ndarray] = None
        self.pose_received = None
        self.pose_stamp_ns = None
        field_path = str(self.get_parameter('field_config_file').value)
        if field_path:
            field = load_field(field_path)
            self.wall_lookup = WallLookupGrid(field['walls'])
            self.map_frame = field['frame_id']
            self.create_subscription(
                PoseWithCovarianceStamped,
                str(self.get_parameter('pose_topic').value),
                self._pose_callback,
                10,
            )

        # LiDARごとの状態: マウント姿勢と、スキャン形状(角度列)ごとに
        # キャッシュした角度別カットオフ距離。
        # A collision pipeline must process the newest completed scan, never a
        # backlog. SensorDataQoS normally keeps five samples; depth one drops
        # obsolete revolutions when the Jetson is momentarily busy.
        latest_scan_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.lidar_states: Dict[str, dict] = {}
        for lidar in robot['lidars']:
            topic = str(lidar['topic'])
            state = {
                'origin': np.array([
                    float(lidar['pose']['x']),
                    float(lidar['pose']['y']),
                ]),
                'yaw': float(lidar['pose']['yaw']),
                'self_reflection_windows': lidar['self_reflection_windows'],
                'frame_id': lidar['frame_id'],
                'cache_key': None,
                'cutoffs': None,
                'directions': None,
                'publisher': self.create_publisher(
                    LaserScan, topic + suffix, qos_profile_sensor_data
                ),
                'nowalls_publisher': (
                    self.create_publisher(
                        LaserScan,
                        topic + nowalls_suffix,
                        qos_profile_sensor_data,
                    )
                    if self.wall_lookup is not None
                    else None
                ),
            }
            self.lidar_states[topic] = state
            self.create_subscription(
                LaserScan,
                topic,
                lambda message, key=topic: self._scan_callback(message, key),
                latest_scan_qos,
            )
        self.create_service(
            SetMotionContext,
            str(self.get_parameter('motion_context_service').value),
            self._set_motion_context,
        )
        self.get_logger().info(
            'Scan footprint filter ready: '
            + ', '.join(
                f'{topic} -> {topic}{suffix}' for topic in self.lidar_states
            )
            + f' (padding {self.padding:.3f} m)'
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
            polygon = canonical_convex_polygon(
                (point.x, point.y)
                for point in request.footprint.polygon.points
            )
        except FootprintGeometryError as error:
            response.success = False
            response.message = str(error)
            return response
        digest = footprint_digest(polygon)
        if str(request.footprint_digest) != digest:
            response.success = False
            response.message = 'footprint digest does not match polygon'
            return response
        self.footprint = np.asarray(polygon, dtype=float)
        self._motion_context_revision = revision
        self._motion_context_execution_id = int(request.execution_id)
        self._motion_context_digest = digest
        for state in self.lidar_states.values():
            state['cache_key'] = None
            state['cutoffs'] = None
            state['directions'] = None
        response.success = True
        response.applied_revision = int(request.revision)
        response.applied_digest = digest
        response.message = (
            f'applied {request.footprint_profile} for execution '
            f'{request.execution_id}; all LiDAR cutoff caches invalidated'
        )
        return response

    def _cutoffs_for(self, message: LaserScan, state: dict) -> np.ndarray:
        count = len(message.ranges)
        key = (
            round(float(message.angle_min), 9),
            round(float(message.angle_increment), 9),
            count,
        )
        if state['cache_key'] == key and state['cutoffs'] is not None:
            return state['cutoffs']
        angles = (
            float(message.angle_min)
            + np.arange(count, dtype=float) * float(message.angle_increment)
            + state['yaw']
        )
        directions = np.column_stack((np.cos(angles), np.sin(angles)))
        cutoffs = ray_exit_distances(
            state['origin'], directions, self.footprint
        )
        # 抜け距離0(=原点がフットプリント外を向く異常な設定)は素通し。
        cutoffs = np.where(cutoffs > 0.0, cutoffs + self.padding, 0.0)
        state['cache_key'] = key
        state['cutoffs'] = cutoffs
        state['directions'] = directions
        return cutoffs

    def _pose_callback(self, message: PoseWithCovarianceStamped) -> None:
        if message.header.frame_id and message.header.frame_id != self.map_frame:
            return
        orientation = message.pose.pose.orientation
        components = [float(getattr(orientation, name)) for name in ('x', 'y', 'z', 'w')]
        if not all(math.isfinite(value) for value in components):
            return
        magnitude = math.hypot(*components)
        if magnitude <= 1.0e-12:
            return
        x, y, z, w = [value / magnitude for value in components]
        yaw = math.atan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y ** 2 + z ** 2),
        )
        pose = np.array([
            message.pose.pose.position.x,
            message.pose.pose.position.y,
            yaw,
        ])
        if not np.all(np.isfinite(pose)):
            return
        received = self.get_clock().now()
        stamp_ns = (
            int(message.header.stamp.sec) * 1_000_000_000
            + int(message.header.stamp.nanosec)
        ) or received.nanoseconds
        if not timestamp_is_fresh(
            received.nanoseconds, stamp_ns,
            int(float(self.get_parameter('pose_timeout_sec').value) * 1.0e9),
            future_tolerance_ns=50_000_000,
        ):
            return
        self.pose = pose
        self.pose_stamp_ns = stamp_ns
        self.pose_received = received

    def _pose_fresh(self) -> bool:
        if self.pose is None or self.pose_received is None:
            return False
        age = (
            self.get_clock().now() - self.pose_received
        ).nanoseconds * 1.0e-9
        timeout = float(self.get_parameter('pose_timeout_sec').value)
        return 0.0 <= age <= timeout and timestamp_is_fresh(
            self.get_clock().now().nanoseconds, self.pose_stamp_ns,
            int(timeout * 1.0e9), future_tolerance_ns=50_000_000,
        )

    def _scan_callback(self, message: LaserScan, topic: str) -> None:
        state = self.lidar_states[topic]
        if not scan_metadata_valid(message):
            return
        if message.header.frame_id != state['frame_id']:
            # Filtering with another sensor's mount can erase a real obstacle.
            # Preserve its original frame and ranges for the safety consumer.
            state['publisher'].publish(message)
            if state['nowalls_publisher'] is not None:
                state['nowalls_publisher'].publish(message)
            return
        cutoffs = self._cutoffs_for(message, state)
        ranges = np.asarray(message.ranges, dtype=np.float32)
        self_hits = ranges <= cutoffs[: len(ranges)]
        # Only this physical sensor's surveyed angle/range window is masked.
        # A mismatched frame must never apply another sensor's exclusion.
        if message.header.frame_id == state['frame_id']:
            self_hits |= self_reflection_mask(
                ranges, message.angle_min, message.angle_increment,
                state['self_reflection_windows'])
        if np.any(self_hits):
            ranges = ranges.copy()
            ranges[self_hits] = math.nan
            message.ranges = ranges.tolist()
        state['publisher'].publish(message)
        if (
            state['nowalls_publisher'] is None
            or state['nowalls_publisher'].get_subscription_count() == 0
        ):
            return
        # 既知壁のヒットを+infへ(グローバル障害物レイヤー用のクリアレイ)。
        # 自己位置が古い間は分類できないので、壁込みのまま流す(安全側)。
        nowalls = ranges
        if self._pose_fresh():
            finite = np.isfinite(ranges) & (ranges > 0.0)
            if np.any(finite):
                directions = state['directions'][: len(ranges)][finite]
                base_points = (
                    state['origin'][None, :]
                    + ranges[finite, None].astype(float) * directions
                )
                map_points = transform_points(base_points, self.pose)
                _, distances, _, _ = self.wall_lookup.query(map_points)
                wall_hits = np.zeros(len(ranges), dtype=bool)
                wall_hits[np.flatnonzero(finite)] = distances <= float(
                    self.get_parameter('wall_match_distance').value
                )
                if np.any(wall_hits):
                    nowalls = ranges.copy()
                    nowalls[wall_hits] = np.inf
        if nowalls is not ranges:
            message.ranges = nowalls.tolist()
        state['nowalls_publisher'].publish(message)


def main(args=None) -> None:
    rclpy.init(args=args)
    try:
        node = ScanFootprintFilter()
    except (ConfigError, ValueError) as error:
        rclpy.logging.get_logger('scan_footprint_filter').fatal(str(error))
        rclpy.try_shutdown()
        raise
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except RCLError:
        if rclpy.ok():
            raise
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
