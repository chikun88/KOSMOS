import hashlib
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import rclpy
from geometry_msgs.msg import Point
from nav_msgs.msg import MapMetaData, OccupancyGrid
from rclpy._rclpy_pybind11 import RCLError
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

from .cad_import import (
    apply_xy_relocations,
    load_layout_corrections,
    read_binary_stl,
    solid_section_mask,
)


class CadVisualizer(Node):
    def __init__(self) -> None:
        super().__init__('cad_visualizer')
        self._declare_parameters()
        model_path = Path(
            str(self.get_parameter('cad_model_file').value)
        ).expanduser()
        if not model_path.is_file():
            raise FileNotFoundError(f'CAD STL does not exist: {model_path}')
        self.frame_id = str(self.get_parameter('frame_id').value)
        self.triangles = read_binary_stl(model_path)
        layout_value = str(self.get_parameter('field_layout_file').value).strip()
        if layout_value:
            layout_path = Path(layout_value).expanduser()
            self.triangles = apply_xy_relocations(
                self.triangles,
                load_layout_corrections(layout_path),
            )
        minimum = self.triangles.min(axis=(0, 1))
        maximum = self.triangles.max(axis=(0, 1))
        self.origin_mm = 0.5 * (minimum[:2] + maximum[:2])
        self.minimum = minimum
        self.maximum = maximum

        transient_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.marker_publisher = self.create_publisher(
            MarkerArray,
            str(self.get_parameter('marker_topic').value),
            transient_qos,
        )
        self.map_publisher = self.create_publisher(
            OccupancyGrid,
            str(self.get_parameter('map_topic').value),
            transient_qos,
        )
        self.obstacle_map_publisher = self.create_publisher(
            OccupancyGrid,
            str(self.get_parameter('obstacle_map_topic').value),
            transient_qos,
        )
        self.model_marker = self._build_model_marker()
        self.start_zone_markers = self._build_start_zone_markers()
        self.field_map = self._build_field_map(
            resolution=float(self.get_parameter('map_resolution').value),
            slices_mm=[float(self.get_parameter('slice_z_mm').value)],
        )
        self.obstacle_map = self._build_field_map(
            resolution=float(
                self.get_parameter('obstacle_map_resolution').value
            ),
            slices_mm=[
                float(value) for value in
                self.get_parameter('obstacle_slices_z_mm').value
            ],
        )
        self._publish()
        digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
        self.get_logger().info(
            f'Loaded exact CAD model: {len(self.triangles)} triangles, '
            f'{maximum[0] - minimum[0]:.0f}x'
            f'{maximum[1] - minimum[1]:.0f}x'
            f'{maximum[2] - minimum[2]:.0f} mm, sha256={digest}'
        )

    def _declare_parameters(self) -> None:
        defaults = {
            'cad_model_file': '',
            'field_layout_file': '',
            'frame_id': 'map',
            'marker_topic': '/field/cad_markers',
            'map_topic': '/field/map',
            'map_resolution': 0.025,
            'slice_z_mm': 180.0,
            # Union of every section height whose geometry the robot body can
            # collide with; must stay consistent with field_planning.yaml.
            'obstacle_map_topic': '/field/obstacle_map',
            'obstacle_map_resolution': 0.02,
            'obstacle_slices_z_mm': [
                60.0, 130.0, 190.0, 250.0, 300.0,
                400.0, 520.0, 700.0, 1000.0, 1500.0,
            ],
            'mesh_alpha': 0.92,
            # Robot start zones as a flat [x1, y1, x2, y2, ...] list in the
            # map frame, drawn as translucent squares. Defaults to the two
            # 1000x1000 mm start squares of the 2026 field CAD (STEP faces
            # #3541/#3818, centers (-1800, 4750) and (+1800, 4750) mm).
            'start_zones': [-1.8, 4.75, 1.8, 4.75],
            'start_zone_size': 1.0,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

    @staticmethod
    def _triangle_color(mean_height: float, normal_z: float, alpha: float):
        if mean_height < 0.04 and normal_z > 0.5:
            values = (0.16, 0.20, 0.24)
        elif normal_z > 0.65:
            values = (0.20, 0.78, 0.86)
        elif mean_height > 1.5:
            values = (0.95, 0.72, 0.18)
        else:
            values = (0.08, 0.52, 0.62)
        return ColorRGBA(r=values[0], g=values[1], b=values[2], a=alpha)

    def _build_model_marker(self) -> Marker:
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.ns = 'exact_field_cad'
        marker.id = 0
        marker.type = Marker.TRIANGLE_LIST
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 1.0
        marker.scale.y = 1.0
        marker.scale.z = 1.0
        alpha = float(self.get_parameter('mesh_alpha').value)
        for triangle in self.triangles:
            centered = triangle.copy()
            centered[:, :2] -= self.origin_mm
            centered /= 1000.0
            normal = np.cross(centered[1] - centered[0], centered[2] - centered[0])
            norm = float(np.linalg.norm(normal))
            normal_z = float(normal[2] / norm) if norm > 1.0e-12 else 0.0
            color = self._triangle_color(
                float(np.mean(centered[:, 2])), normal_z, alpha
            )
            for vertex in centered:
                marker.points.append(Point(
                    x=float(vertex[0]),
                    y=float(vertex[1]),
                    z=float(vertex[2]),
                ))
                marker.colors.append(color)
        return marker

    def _build_field_map(
        self,
        *,
        resolution: float,
        slices_mm: Sequence[float],
    ) -> OccupancyGrid:
        width = int(math.ceil(
            (self.maximum[0] - self.minimum[0]) / 1000.0 / resolution
        ))
        height = int(math.ceil(
            (self.maximum[1] - self.minimum[1]) / 1000.0 / resolution
        ))
        origin_x = (self.minimum[0] - self.origin_mm[0]) / 1000.0
        origin_y = (self.minimum[1] - self.origin_mm[1]) / 1000.0
        occupied = np.zeros((height, width), dtype=bool)

        for slice_z in slices_mm:
            # Exact solid material of the section (even-odd over every loop),
            # so the hollow field enclosure never fills the drivable area.
            occupied |= solid_section_mask(
                self.triangles,
                float(slice_z),
                origin_mm=self.origin_mm,
                origin_x=origin_x,
                origin_y=origin_y,
                resolution=resolution,
                width=width,
                height=height,
            )

        message = OccupancyGrid()
        message.header.frame_id = self.frame_id
        message.info = MapMetaData()
        message.info.resolution = resolution
        message.info.width = width
        message.info.height = height
        message.info.origin.position.x = float(origin_x)
        message.info.origin.position.y = float(origin_y)
        message.info.origin.orientation.w = 1.0
        message.data = (occupied.astype(np.int8) * 100).reshape(-1).tolist()
        return message

    def _build_start_zone_markers(self) -> list:
        flat = [
            float(value)
            for value in self.get_parameter('start_zones').value
        ]
        size = float(self.get_parameter('start_zone_size').value)
        markers = []
        for index in range(0, len(flat) - 1, 2):
            marker = Marker()
            marker.header.frame_id = self.frame_id
            marker.ns = 'start_zones'
            marker.id = index // 2
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose.position.x = flat[index]
            marker.pose.position.y = flat[index + 1]
            marker.pose.position.z = 0.02
            marker.pose.orientation.w = 1.0
            marker.scale.x = size
            marker.scale.y = size
            marker.scale.z = 0.004
            marker.color = ColorRGBA(r=0.10, g=0.85, b=0.95, a=0.55)
            markers.append(marker)
        return markers

    def _publish(self) -> None:
        stamp = self.get_clock().now().to_msg()
        self.model_marker.header.stamp = stamp
        markers = MarkerArray()
        markers.markers.append(self.model_marker)
        for marker in self.start_zone_markers:
            marker.header.stamp = stamp
            markers.markers.append(marker)
        self.marker_publisher.publish(markers)
        self.field_map.header.stamp = stamp
        self.map_publisher.publish(self.field_map)
        self.obstacle_map.header.stamp = stamp
        self.obstacle_map_publisher.publish(self.obstacle_map)


def main(args=None) -> None:
    rclpy.init(args=args)
    try:
        node = CadVisualizer()
    except (FileNotFoundError, ValueError) as error:
        rclpy.logging.get_logger('cad_visualizer').fatal(str(error))
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
