import argparse
from collections import defaultdict
import hashlib
import math
from pathlib import Path
import struct
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import yaml


PointKey = Tuple[float, float]


def load_layout_corrections(path: Path) -> List[dict]:
    """Load drawing-derived XY relocations applied on top of the source STL."""
    config_path = Path(path).expanduser()
    with config_path.open('r', encoding='utf-8') as stream:
        data = yaml.safe_load(stream)
    layout = data.get('field_layout') if isinstance(data, dict) else None
    relocations = layout.get('obstacle_relocations') if isinstance(layout, dict) else None
    if not isinstance(relocations, list):
        raise ValueError(
            f'{config_path} must contain field_layout.obstacle_relocations'
        )
    return relocations


def apply_xy_relocations(
    triangles: np.ndarray,
    relocations: Sequence[dict],
    *,
    origin_mm: np.ndarray = None,
) -> np.ndarray:
    """Relocate isolated CAD objects to surveyed/drawing map coordinates.

    Coordinates in the layout file are metres in the centred ROS ``map``
    frame.  Selection is deliberately based on *all* triangle vertices being
    inside the source cylinder, so a floor or nearby wall cannot accidentally
    be dragged along with an obstacle.
    """
    corrected = np.asarray(triangles, dtype=float).copy()
    if origin_mm is None:
        minimum = corrected.min(axis=(0, 1))
        maximum = corrected.max(axis=(0, 1))
        origin_mm = 0.5 * (minimum[:2] + maximum[:2])
    else:
        origin_mm = np.asarray(origin_mm, dtype=float)

    claimed = np.zeros(len(corrected), dtype=bool)
    for index, relocation in enumerate(relocations):
        if not isinstance(relocation, dict):
            raise ValueError(f'Obstacle relocation {index} must be a mapping')
        name = str(relocation.get('name', f'relocation_{index}'))
        source = np.asarray(relocation.get('source_center', []), dtype=float)
        target = np.asarray(relocation.get('target_center', []), dtype=float)
        if source.shape != (2,) or target.shape != (2,):
            raise ValueError(f'{name}: source_center/target_center need [x, y]')
        radius_mm = 1000.0 * float(relocation.get('selection_radius', 0.0))
        if radius_mm <= 0.0:
            raise ValueError(f'{name}: selection_radius must be positive')
        z_min_mm = float(relocation.get('z_min_mm', -math.inf))
        z_max_mm = float(relocation.get('z_max_mm', math.inf))

        source_mm = origin_mm + 1000.0 * source
        vertex_distance = np.linalg.norm(
            corrected[:, :, :2] - source_mm, axis=2
        )
        selected = (
            (np.max(vertex_distance, axis=1) <= radius_mm + 1.0e-6)
            & (np.min(corrected[:, :, 2], axis=1) >= z_min_mm - 1.0e-6)
            & (np.max(corrected[:, :, 2], axis=1) <= z_max_mm + 1.0e-6)
        )
        if not np.any(selected):
            raise ValueError(f'{name}: no CAD triangles matched the source region')
        if np.any(claimed & selected):
            raise ValueError(f'{name}: source region overlaps another relocation')
        corrected[selected, :, :2] += 1000.0 * (target - source)
        claimed |= selected
    return corrected


def read_binary_stl(path: Path) -> np.ndarray:
    """Read triangle vertices from a binary STL file."""
    file_size = path.stat().st_size
    with path.open('rb') as stream:
        stream.read(80)
        triangle_count_data = stream.read(4)
        if len(triangle_count_data) != 4:
            raise ValueError(f'Invalid binary STL header: {path}')
        triangle_count = struct.unpack('<I', triangle_count_data)[0]
        expected_size = 84 + triangle_count * 50
        if file_size != expected_size:
            raise ValueError(
                f'Only binary STL is supported; expected {expected_size} bytes, '
                f'got {file_size}: {path}'
            )

        triangles = np.empty((triangle_count, 3, 3), dtype=float)
        for index in range(triangle_count):
            record = stream.read(50)
            if len(record) != 50:
                raise ValueError(f'Truncated STL triangle {index}: {path}')
            values = struct.unpack('<12fH', record)
            triangles[index] = np.asarray(values[3:12]).reshape(3, 3)
    return triangles


def section_segments(triangles: np.ndarray, z_mm: float) -> List[np.ndarray]:
    """Intersect STL triangles with a non-coplanar horizontal plane."""
    segments = []
    for triangle in triangles:
        intersections = []
        for first, second in ((0, 1), (1, 2), (2, 0)):
            first_height = triangle[first, 2] - z_mm
            second_height = triangle[second, 2] - z_mm
            if first_height * second_height < 0.0:
                fraction = -first_height / (second_height - first_height)
                intersections.append(
                    triangle[first, :2]
                    + fraction * (triangle[second, :2] - triangle[first, :2])
                )
            elif abs(first_height) < 1.0e-9 and abs(second_height) >= 1.0e-9:
                intersections.append(triangle[first, :2])

        unique = []
        for point in intersections:
            if not any(np.linalg.norm(point - other) < 1.0e-5 for other in unique):
                unique.append(point)
        if len(unique) >= 2 and np.linalg.norm(unique[0] - unique[1]) > 1.0e-4:
            segments.append(np.asarray(unique[:2]))
    return segments


def _point_key(point: np.ndarray, quantization_mm: float) -> PointKey:
    quantized = np.round(point / quantization_mm) * quantization_mm
    return float(quantized[0]), float(quantized[1])


def segments_to_loops(
    segments: Iterable[np.ndarray],
    quantization_mm: float = 0.01,
) -> List[np.ndarray]:
    """Join triangle-section fragments into closed cross-section loops."""
    edges = set()
    adjacency: Dict[PointKey, set] = defaultdict(set)
    for segment in segments:
        first = _point_key(segment[0], quantization_mm)
        second = _point_key(segment[1], quantization_mm)
        if first == second:
            continue
        edge = tuple(sorted((first, second)))
        if edge in edges:
            continue
        edges.add(edge)
        adjacency[first].add(second)
        adjacency[second].add(first)

    loops = []
    unused = set(edges)
    while unused:
        first_edge = next(iter(unused))
        start, current = first_edge
        loop = [start]
        previous = start
        unused.remove(first_edge)

        while current != start:
            loop.append(current)
            candidates = adjacency[current] - {previous}
            if not candidates:
                raise ValueError('STL section contains an open contour')
            following = next(iter(candidates))
            edge = tuple(sorted((current, following)))
            if edge not in unused and following != start:
                raise ValueError('STL section contains a branched contour')
            unused.discard(edge)
            previous, current = current, following
            if len(loop) > len(edges) + 1:
                raise ValueError('Failed to close STL section contour')
        loops.append(np.asarray(loop, dtype=float))
    return loops


def signed_area(loop: np.ndarray) -> float:
    shifted = np.roll(loop, -1, axis=0)
    return 0.5 * float(
        np.sum(loop[:, 0] * shifted[:, 1] - shifted[:, 0] * loop[:, 1])
    )


def point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    x, y = point
    inside = False
    previous = len(polygon) - 1
    for current in range(len(polygon)):
        xi, yi = polygon[current]
        xj, yj = polygon[previous]
        if ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / (yj - yi + 1.0e-15) + xi
        ):
            inside = not inside
        previous = current
    return inside


def _loop_nesting_depth(
    index: int,
    loops: Sequence[np.ndarray],
    areas: Sequence[float],
    representatives: Sequence[np.ndarray],
) -> int:
    """Count how many strictly larger loops enclose this loop."""
    return sum(
        1
        for other_index, other in enumerate(loops)
        if other_index != index
        and areas[other_index] > areas[index]
        and point_in_polygon(representatives[index], other)
    )


def free_space_depth(loops: Sequence[np.ndarray], grid: int = 61) -> int:
    """Nesting depth of the dominant free-space region.

    A point's depth is the number of loops that contain it. The region a robot
    drives in (the perimeter interior of a field, or the open space around a few
    isolated obstacles) covers the largest area, so its depth is the modal depth
    over a grid spanning every loop.
    """
    points = np.vstack(list(loops))
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    xs = np.linspace(minimum[0], maximum[0], grid)
    ys = np.linspace(minimum[1], maximum[1], grid)
    counts: Dict[int, int] = defaultdict(int)
    for x in xs:
        for y in ys:
            probe = np.array([x, y])
            depth = sum(
                1 for loop in loops if point_in_polygon(probe, loop)
            )
            counts[depth] += 1
    return max(counts, key=counts.get)


def select_lidar_visible_loops(
    loops: Sequence[np.ndarray],
) -> List[np.ndarray]:
    """Keep only the cross-section loops a LiDAR in the free space can see.

    Every loop bounds free space on exactly one side. With the free-space region
    at nesting depth ``d``:

    * loops at depth ``d - 1`` are containers the robot sits inside (the field
      perimeter wall's inner face, and the central divider) -> keep,
    * loops at depth ``d`` are obstacles standing in the free space (pillars,
      poles, game elements) -> keep,
    * shallower loops (e.g. the perimeter wall's outer face, which faces outside
      the field) and deeper loops (hollow interiors of a pole) are never visible
      from the free space -> drop.

    This replaces a naive "drop everything enclosed by a larger loop" rule, which
    silently deleted the perimeter inner face, the central wall and every pillar
    once a closed outer boundary was present.
    """
    loops = list(loops)
    if len(loops) <= 1:
        return loops
    areas = [abs(signed_area(loop)) for loop in loops]
    representatives = [np.mean(loop, axis=0) for loop in loops]
    interior_depth = free_space_depth(loops)
    visible_depths = {interior_depth - 1, interior_depth}
    keep = []
    for index, loop in enumerate(loops):
        depth = _loop_nesting_depth(index, loops, areas, representatives)
        if depth in visible_depths:
            keep.append(loop)
    return keep


def rasterize_loops(
    loops: Sequence[np.ndarray],
    *,
    origin_x: float,
    origin_y: float,
    resolution: float,
    width: int,
    height: int,
    occupied: 'np.ndarray' = None,
    combine: str = 'or',
) -> np.ndarray:
    """Rasterize the interiors of closed loops into an occupancy bitmap.

    Loops are in the same metric frame as the grid origin. Each loop is
    rasterized only inside its own bounding-box window, so a centimeter
    resolution field map stays fast even with many section slices.

    combine='or' unions the loop interiors. combine='xor' applies even-odd
    parity, which turns the complete loop set of one STL cross section into
    the exact solid-material mask: a hollow perimeter wall becomes a thin
    ring instead of a filled rectangle, and a pillar inside the field (inside
    the wall's outer and inner boundary loops plus its own loop, i.e. an odd
    count) stays solid.
    """
    if combine not in ('or', 'xor'):
        raise ValueError("combine must be 'or' or 'xor'")
    if occupied is None:
        occupied = np.zeros((height, width), dtype=bool)
    for loop in loops:
        polygon = np.asarray(loop, dtype=float)
        min_cx = max(0, int(math.floor(
            (polygon[:, 0].min() - origin_x) / resolution
        )))
        max_cx = min(width - 1, int(math.ceil(
            (polygon[:, 0].max() - origin_x) / resolution
        )))
        min_cy = max(0, int(math.floor(
            (polygon[:, 1].min() - origin_y) / resolution
        )))
        max_cy = min(height - 1, int(math.ceil(
            (polygon[:, 1].max() - origin_y) / resolution
        )))
        if min_cx > max_cx or min_cy > max_cy:
            continue
        xs = origin_x + (np.arange(min_cx, max_cx + 1) + 0.5) * resolution
        ys = origin_y + (np.arange(min_cy, max_cy + 1) + 0.5) * resolution
        grid_x, grid_y = np.meshgrid(xs, ys)
        window_points = np.column_stack((
            grid_x.reshape(-1), grid_y.reshape(-1)
        ))
        inside = points_in_polygon_vectorized(window_points, polygon)
        window = occupied[min_cy:max_cy + 1, min_cx:max_cx + 1]
        if combine == 'or':
            window |= inside.reshape(grid_x.shape)
        else:
            window ^= inside.reshape(grid_x.shape)
    return occupied


def solid_section_mask(
    triangles: np.ndarray,
    z_mm: float,
    *,
    origin_mm: np.ndarray,
    origin_x: float,
    origin_y: float,
    resolution: float,
    width: int,
    height: int,
    simplify_tolerance_mm: float = 0.05,
) -> np.ndarray:
    """Rasterize the exact solid material of one horizontal STL section.

    Uses every section loop (outer and inner boundaries alike) with even-odd
    parity, so hollow enclosures stay hollow. Loops are simplified to the
    given tolerance first; rasterization cost scales with vertex count.
    """
    raw_segments = section_segments(triangles, float(z_mm))
    if not raw_segments:
        return np.zeros((height, width), dtype=bool)
    loops = [
        simplify_closed_loop(loop, simplify_tolerance_mm)
        for loop in segments_to_loops(raw_segments)
    ]
    return rasterize_loops(
        [(loop - origin_mm) / 1000.0 for loop in loops],
        origin_x=origin_x,
        origin_y=origin_y,
        resolution=resolution,
        width=width,
        height=height,
        combine='xor',
    )


def points_in_polygon_vectorized(
    points: np.ndarray,
    polygon: np.ndarray,
) -> np.ndarray:
    """Vectorized ray-casting test, boundary counts as inside.

    Kept local (duplicating geometry.points_in_polygon) so cad_import stays
    runnable as a standalone script without the package on sys.path.
    """
    if len(points) == 0:
        return np.zeros(0, dtype=bool)
    x = points[:, 0]
    y = points[:, 1]
    inside = np.zeros(len(points), dtype=bool)
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        intersects = ((yi > y) != (yj > y)) & (
            x <= (xj - xi) * (y - yi) / (yj - yi + 1.0e-15) + xi
        )
        inside ^= intersects
        j = i
    return inside


def _point_line_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    vector = end - start
    length = float(np.linalg.norm(vector))
    if length < 1.0e-12:
        return float(np.linalg.norm(point - start))
    return abs(float(np.cross(vector, point - start))) / length


def simplify_closed_loop(loop: np.ndarray, tolerance_mm: float) -> np.ndarray:
    """Remove near-collinear vertices while preserving a closed contour."""
    points = [point.copy() for point in loop]
    changed = True
    while changed and len(points) > 3:
        changed = False
        for index in range(len(points)):
            previous = points[index - 1]
            current = points[index]
            following = points[(index + 1) % len(points)]
            if _point_line_distance(current, previous, following) <= tolerance_mm:
                del points[index]
                changed = True
                break
    return np.asarray(points)


def slice_visible_loops(
    triangles: np.ndarray,
    z_mm: float,
    simplify_tolerance_mm: float,
) -> List[np.ndarray]:
    """Section the STL at one height and keep LiDAR/robot-facing loops."""
    raw_segments = section_segments(triangles, z_mm)
    if not raw_segments:
        return []
    loops = segments_to_loops(raw_segments)
    loops = select_lidar_visible_loops(loops)
    return [
        simplify_closed_loop(loop, simplify_tolerance_mm)
        for loop in loops
    ]


def _loop_walls(
    loops: Sequence[np.ndarray],
    origin_mm: np.ndarray,
    name_prefix: str,
) -> List[dict]:
    loops = sorted(
        loops,
        key=lambda loop: (float(np.mean(loop[:, 1])), float(np.mean(loop[:, 0]))),
    )
    walls = []
    seen_edges = set()
    for surface_index, loop in enumerate(loops):
        centered = (loop - origin_mm) / 1000.0
        for edge_index, start in enumerate(centered):
            end = centered[(edge_index + 1) % len(centered)]
            # Drop edges that collapse to a point so load_field never sees a
            # zero-length wall after rounding.
            if math.hypot(end[0] - start[0], end[1] - start[1]) < 1.0e-4:
                continue
            start_value = (round(float(start[0]), 6), round(float(start[1]), 6))
            end_value = (round(float(end[0]), 6), round(float(end[1]), 6))
            # Identical wall faces show up in several slices of a multi-slice
            # planning map; keep only one copy per undirected edge.
            edge_key = tuple(sorted((start_value, end_value)))
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)
            walls.append({
                'name': f'{name_prefix}_{surface_index:02d}_{edge_index:02d}',
                'start': list(start_value),
                'end': list(end_value),
            })
    return walls


def build_field_data(
    triangles: np.ndarray,
    z_mm: float,
    simplify_tolerance_mm: float,
    source_stl: str = '',
    source_stl_sha256: str = '',
    extra_slices_mm: Sequence[float] = (),
    source_layout: str = '',
    source_layout_sha256: str = '',
) -> dict:
    """Build the field YAML payload.

    With extra_slices_mm the walls are the union of the section at z_mm and
    every extra height, deduplicated. That is wrong for localization (the
    LiDAR only sees its own scan plane) but exactly right for path planning,
    where anything the robot body can hit at any height must be an obstacle.
    """
    minimum = triangles.min(axis=(0, 1))
    maximum = triangles.max(axis=(0, 1))
    origin_mm = 0.5 * (minimum[:2] + maximum[:2])

    walls: List[dict] = []
    seen = set()
    for slice_index, slice_z in enumerate([float(z_mm)] + [
        float(value) for value in extra_slices_mm
    ]):
        loops = slice_visible_loops(triangles, slice_z, simplify_tolerance_mm)
        prefix = (
            'cad_surface'
            if slice_index == 0
            else f'cad_z{int(round(slice_z)):04d}'
        )
        for wall in _loop_walls(loops, origin_mm, prefix):
            edge_key = tuple(sorted((tuple(wall['start']), tuple(wall['end']))))
            if edge_key in seen:
                continue
            seen.add(edge_key)
            walls.append(wall)

    field = {
        'frame_id': 'map',
        'source': 'CAD horizontal section',
    }
    if source_stl:
        field['source_stl'] = source_stl
    if source_stl_sha256:
        field['source_stl_sha256'] = source_stl_sha256
    if source_layout:
        field['source_layout'] = source_layout
    if source_layout_sha256:
        field['source_layout_sha256'] = source_layout_sha256
    field['source_slice_z_mm'] = float(z_mm)
    if extra_slices_mm:
        field['extra_slices_z_mm'] = [float(value) for value in extra_slices_mm]
    field['cad_origin_mm'] = [
        round(float(origin_mm[0]), 6),
        round(float(origin_mm[1]), 6),
    ]
    field['cad_bounds_mm'] = {
        'min': [float(value) for value in minimum],
        'max': [float(value) for value in maximum],
    }
    field['walls'] = walls
    return {'field': field}


def convert_stl(
    input_path: Path,
    output_path: Path,
    z_mm: float,
    simplify_tolerance_mm: float,
    extra_slices_mm: Sequence[float] = (),
    layout_config: Path = None,
) -> dict:
    triangles = read_binary_stl(input_path)
    digest = hashlib.sha256(input_path.read_bytes()).hexdigest()
    layout_path = None
    layout_digest = ''
    if layout_config is not None:
        layout_path = Path(layout_config).expanduser().resolve()
        relocations = load_layout_corrections(layout_path)
        triangles = apply_xy_relocations(triangles, relocations)
        layout_digest = hashlib.sha256(layout_path.read_bytes()).hexdigest()
    field_data = build_field_data(
        triangles,
        z_mm,
        simplify_tolerance_mm,
        source_stl=str(input_path),
        source_stl_sha256=digest,
        extra_slices_mm=extra_slices_mm,
        source_layout=str(layout_path) if layout_path is not None else '',
        source_layout_sha256=layout_digest,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as stream:
        stream.write(
            '# Generated from the field CAD STL. Coordinates are meters from CAD center.\n'
        )
        yaml.safe_dump(field_data, stream, sort_keys=False, allow_unicode=True)
    return field_data


def main(args=None) -> None:
    parser = argparse.ArgumentParser(
        description='Convert a binary STL horizontal section to field wall YAML'
    )
    parser.add_argument('input_stl', type=Path)
    parser.add_argument('output_yaml', type=Path)
    parser.add_argument(
        '--slice-z-mm',
        type=float,
        default=130.0,
        help='Global CAD height of the horizontal LiDAR scan plane. Must match '
        'the LiDAR pose z in robot.yaml. The field perimeter wall tops out at '
        '164 mm, so keep this below ~150 mm to see the field edge.',
    )
    parser.add_argument(
        '--simplify-mm',
        type=float,
        default=0.05,
        help='Maximum near-collinear simplification error',
    )
    parser.add_argument(
        '--extra-slice-z-mm',
        type=float,
        nargs='+',
        default=[],
        help='Additional section heights whose walls are unioned into the '
        'output. Use for a path-planning map that must contain everything '
        'the robot body can hit (centre wall top, poles), not only what the '
        'LiDAR scan plane sees. Do not use for the localization map.',
    )
    parser.add_argument(
        '--layout-config',
        type=Path,
        default=None,
        help='Optional drawing/survey correction YAML applied to isolated '
        'objects before extracting section walls.',
    )
    parsed = parser.parse_args(args)
    data = convert_stl(
        parsed.input_stl,
        parsed.output_yaml,
        parsed.slice_z_mm,
        parsed.simplify_mm,
        extra_slices_mm=parsed.extra_slice_z_mm,
        layout_config=parsed.layout_config,
    )
    print(
        f'Wrote {len(data["field"]["walls"])} wall segments to '
        f'{parsed.output_yaml}'
    )


if __name__ == '__main__':
    main()
