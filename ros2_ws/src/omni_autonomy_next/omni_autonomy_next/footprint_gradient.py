"""Translation direction away from the closest features of a rotated outline."""
import numpy as np


def footprint_outward_direction(polygon, starts, ends, clearance):
    """Return a map-frame ascent direction of positive footprint clearance.

    The caller supplies exact polygon/segment clearance, including intersection
    and containment rejection. Closest features can be a polygon vertex against
    a wall, or a wall endpoint against a polygon edge. Both normals point from
    the wall to the outline. At ties an arbitrary nearest feature is unsafe:
    retain a direction only if it does not approach any equally close feature.
    Contact and incompatible tied normals have no invented outward direction.
    """
    polygon = np.asarray(polygon, dtype=float)
    starts = np.asarray(starts, dtype=float)
    ends = np.asarray(ends, dtype=float)
    clearance = float(clearance)
    if (polygon.ndim != 2 or polygon.shape[1:] != (2,) or len(polygon) < 3
            or starts.ndim != 2 or starts.shape[1:] != (2,)
            or ends.shape != starts.shape or not len(starts)
            or not np.isfinite(polygon).all() or not np.isfinite(starts).all()
            or not np.isfinite(ends).all() or not np.isfinite(clearance)
            or clearance < 0.):
        raise ValueError('footprint gradient requires finite polygon/wall clearance')
    zero = np.zeros(2)
    if clearance <= 1.e-12:
        return zero
    edges = np.roll(polygon, -1, axis=0) - polygon
    edge_length2 = np.einsum('ij,ij->i', edges, edges)
    if np.any(edge_length2 <= 1.e-18):
        raise ValueError('footprint gradient requires distinct adjacent vertices')

    walls = ends - starts
    wall_length2 = np.maximum(np.einsum('ij,ij->i', walls, walls), 1.e-18)
    relative = polygon[:, None, :] - starts[None, :, :]
    t = np.clip(np.einsum('ijk,jk->ij', relative, walls) / wall_length2, 0., 1.)
    vertex_offsets = relative - t[..., None] * walls[None, :, :]

    endpoints = np.concatenate((starts, ends))
    relative = endpoints[None, :, :] - polygon[:, None, :]
    t = np.clip(np.einsum('ijk,ik->ij', relative, edges)
                / edge_length2[:, None], 0., 1.)
    endpoint_offsets = t[..., None] * edges[:, None, :] - relative
    offsets = np.concatenate((vertex_offsets.reshape(-1, 2),
                              endpoint_offsets.reshape(-1, 2)))
    distances = np.linalg.norm(offsets, axis=1)
    minimum = float(distances.min())
    # A saturated or inconsistent scalar must not select unrelated features.
    tolerance = 1.e-9 * max(1., clearance)
    if minimum <= 1.e-12 or abs(minimum - clearance) > tolerance:
        return zero
    nearest = offsets[distances <= minimum + tolerance]
    normals = nearest / np.linalg.norm(nearest, axis=1)[:, None]
    # Identical CAD facets/vertices must not bias an equal-feature decision.
    _, unique = np.unique(np.round(normals, 10), axis=0, return_index=True)
    normals = normals[unique]
    direction = normals.mean(axis=0)
    norm = float(np.linalg.norm(direction))
    if norm <= 1.e-10:
        return zero
    direction /= norm
    if np.any(normals @ direction < -1.e-10):
        return zero
    return direction
