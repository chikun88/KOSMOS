"""Geometry contracts shared by competition footprint participants."""

from __future__ import annotations

import hashlib
import json
from typing import Iterable

import numpy as np


class FootprintGeometryError(ValueError):
    pass


def _cross(a, b, c) -> float:
    return float(np.cross(np.asarray(b) - a, np.asarray(c) - b))


def _point_on_segment(point, start, end, tolerance=1.0e-9) -> bool:
    point = np.asarray(point, dtype=float)
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    vector = end - start
    offset = point - start
    if abs(float(np.cross(vector, offset))) > tolerance:
        return False
    projection = float(np.dot(offset, vector))
    return -tolerance <= projection <= float(np.dot(vector, vector)) + tolerance


def point_in_or_on_convex_polygon(point, polygon) -> bool:
    points = np.asarray(polygon, dtype=float)
    signs = []
    for index, start in enumerate(points):
        end = points[(index + 1) % len(points)]
        if _point_on_segment(point, start, end):
            return True
        signs.append(float(np.cross(end - start, np.asarray(point) - start)))
    return all(value >= -1.0e-9 for value in signs) or all(
        value <= 1.0e-9 for value in signs
    )


def canonical_convex_polygon(points: Iterable[Iterable[float]]) -> tuple:
    """Validate a strict convex base_link polygon and return canonical CCW."""
    polygon = np.asarray(tuple(tuple(value) for value in points), dtype=float)
    if polygon.ndim != 2 or polygon.shape[1:] != (2,) or len(polygon) < 3:
        raise FootprintGeometryError('footprint needs at least three [x, y] points')
    if not np.all(np.isfinite(polygon)):
        raise FootprintGeometryError('footprint contains non-finite coordinates')
    if float(np.max(np.linalg.norm(polygon, axis=1))) > 3.0:
        raise FootprintGeometryError('footprint exceeds the 3 m sanity bound')
    rounded = {tuple(np.round(point, 12)) for point in polygon}
    if len(rounded) != len(polygon):
        raise FootprintGeometryError('footprint contains duplicate vertices')
    signed_twice_area = float(sum(
        polygon[index, 0] * polygon[(index + 1) % len(polygon), 1]
        - polygon[(index + 1) % len(polygon), 0] * polygon[index, 1]
        for index in range(len(polygon))
    ))
    if abs(signed_twice_area) <= 1.0e-6:
        raise FootprintGeometryError('footprint area is zero')
    if signed_twice_area < 0.0:
        polygon = polygon[::-1].copy()
    turns = [
        _cross(
            polygon[index - 1], polygon[index],
            polygon[(index + 1) % len(polygon)],
        )
        for index in range(len(polygon))
    ]
    if any(value <= 1.0e-9 for value in turns):
        raise FootprintGeometryError('footprint must be strictly convex and CCW')
    if not point_in_or_on_convex_polygon((0.0, 0.0), polygon):
        raise FootprintGeometryError('footprint must contain base_link origin')
    # Rotate to a deterministic first vertex for stable hashes.
    first = min(
        range(len(polygon)),
        key=lambda index: (polygon[index, 0], polygon[index, 1]),
    )
    polygon = np.roll(polygon, -first, axis=0)
    return tuple((float(x), float(y)) for x, y in polygon)


def polygon_contains(container, contained) -> bool:
    return all(
        point_in_or_on_convex_polygon(point, container)
        for point in contained
    )


def footprint_digest(polygon) -> str:
    canonical = canonical_convex_polygon(polygon)
    payload = json.dumps(
        [[round(x, 9), round(y, 9)] for x, y in canonical],
        separators=(',', ':'),
    ).encode('ascii')
    return hashlib.sha256(payload).hexdigest()


def polygons_match(first, second, tolerance: float = 1.0e-5) -> bool:
    try:
        one = np.asarray(canonical_convex_polygon(first), dtype=float)
        two = np.asarray(canonical_convex_polygon(second), dtype=float)
    except (FootprintGeometryError, TypeError, ValueError):
        return False
    return one.shape == two.shape and bool(np.allclose(one, two, atol=tolerance))


def polygons_congruent(first, second, tolerance: float = 1.0e-5) -> bool:
    """Compare an ordered polygon up to rigid translation/yaw and reversal.

    Nav2 publishes its footprint in the costmap global frame, whereas the
    configured profile is expressed in ``base_link``. Pairwise distances are
    frame invariant and still detect the wrong profile or vertex ordering.
    """
    try:
        one = np.asarray(first, dtype=float)
        two = np.asarray(second, dtype=float)
    except (TypeError, ValueError):
        return False
    if (
        one.ndim != 2
        or two.ndim != 2
        or one.shape != two.shape
        or one.shape[0] < 3
        or one.shape[1] != 2
        or not np.all(np.isfinite(one))
        or not np.all(np.isfinite(two))
    ):
        return False
    distances_one = np.linalg.norm(one[:, None, :] - one[None, :, :], axis=2)
    for candidate in (two, two[::-1]):
        for shift in range(len(two)):
            rolled = np.roll(candidate, shift, axis=0)
            distances_two = np.linalg.norm(
                rolled[:, None, :] - rolled[None, :, :], axis=2
            )
            if np.allclose(
                distances_one, distances_two, atol=tolerance, rtol=0.0
            ):
                return True
    return False
