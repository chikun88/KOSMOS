"""ROS-independent geometry for stop, rotate, then holonomic translation.

The selected route is checked at both headings, including the swept polygon
between samples. A rejected route never falls back to simultaneous rotation.
Margins are physical clearance, not costmap inflation costs.
"""
from collections import deque
from dataclasses import dataclass, field
import math
import numpy as np


def wrap(value):
    return (float(value) + math.pi) % (2 * math.pi) - math.pi


@dataclass
class TurnResponse:
    """Estimate turn response from accumulated motion after transport lag.

    This only reduces the requested rate. Integrating angles avoids dividing
    noisy instantaneous wheel speeds; the delay queue is bounded in time.
    A feedback interruption restarts the observation, retaining the reduction.
    """
    gain: float = 1.0
    stamp: object = None
    yaw: float = 0.0
    observed: float = 0.0
    requested: float = 0.0
    history: object = field(default_factory=deque)

    def update(self, now, yaw, command, delay):
        if not all(math.isfinite(v) for v in (now, yaw, command, delay)):
            self.stamp = None
            return self.gain
        if self.stamp is None or not 0. < now-self.stamp <= .15:
            self.observed = self.requested = 0.
            self.history.clear()
        else:
            self.observed += wrap(yaw-self.yaw)
            self.requested += command*(now-self.stamp)
        self.stamp, self.yaw = now, yaw
        self.history.append((now, self.requested))
        cutoff = now-max(.0, delay)-.15
        while len(self.history) > 1 and self.history[1][0] <= cutoff:
            self.history.popleft()
        stamp, requested = self.history[0]
        if stamp <= cutoff and abs(requested) > .015 and requested*self.observed > 0.:
            self.gain = min(3.5, max(1., self.observed/requested))
        return self.gain


@dataclass
class HeadingStage:
    heading: float
    target: float
    gate: object = None
    phase: str = 'SELECT'
    settled_since: object = None
    rotation_started: object = None
    clearance: float = 0.0
    rotation_finishing: bool = False
    braking_started: object = None
    settle_drift: float = .035
    turn_response: TurnResponse = field(default_factory=TurnResponse)


def dense_path(points, spacing=.02):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not len(points):
        raise ValueError('invalid path')
    if not np.isfinite(points).all():
        raise ValueError('nonfinite path')
    result = [points[0]]
    for a, b in zip(points[:-1], points[1:]):
        if np.linalg.norm(b-a) < 1.e-9:
            continue
        n = max(1, math.ceil(float(np.linalg.norm(b-a)) / spacing - 1.e-9))
        result.extend(a + (b-a)*i/n for i in range(1, n+1))
    return np.asarray(result)


def remaining_path(points, position):
    """Keep the continuous projection and suffix, including all future bends."""
    points = np.asarray(points, dtype=float)
    position = np.asarray(position, dtype=float)[:2]
    if len(points) < 2:
        return np.vstack((position, points))
    delta = np.diff(points, axis=0)
    lengths2 = np.einsum('ij,ij->i', delta, delta)
    fractions = np.clip(np.einsum('ij,ij->i', position-points[:-1], delta)
                        / np.maximum(lengths2, 1.e-12), 0., 1.)
    projected = points[:-1] + fractions[:, None]*delta
    index = int(np.argmin(np.linalg.norm(projected-position, axis=1)))
    return np.vstack((position, projected[index], points[index+1:]))


def path_clearances(model, points, yaw, margin=.025, yaw_reserve=.02):
    # Distance is 1-Lipschitz in translation. Subtract half the maximum
    # segment length so a wall between samples cannot slip through. Reserve
    # radius*angle for tracking error at ANY intermediate heading.
    points = dense_path(points, .01)
    reserve = model.radius * yaw_reserve + .005
    # Only the threshold matters here. Capping nearby limits the expensive
    # exact polygon operation to walls which can actually reject this route.
    values = model.clearance_over_poses(
        points, np.full(len(points), yaw), cap=margin+reserve+.02) - reserve
    return points, values


def rotation_clearance(model, point, start, target, cap=.50):
    delta = wrap(target-start)
    n = max(1, math.ceil(abs(delta)/.025))
    angles = start + np.linspace(0., delta, n+1)
    values = model.clearance_over_rotation(point, angles, cap=cap)
    # Upper bound on corner travel between yaw samples, plus gate settling
    # and localization allowance. Covers the entire sweep, not just endpoints.
    return float(np.min(values)) - model.radius*abs(delta)/(2*n) - .025


def repair_corridor(model, points, headings, margin=.025):
    """Repair small Smac/smoothing corner cuts within 0.12 m of its corridor.

    Cell-center paths can miss a requested gate by several centimetres even
    when the gate itself fits. Search the nearest clearance-valid offset; the
    complete swept route is still checked by prepare_stage afterwards.
    This never changes the requested start or goal pose.
    """
    points = dense_path(points, .025)
    repaired = points.copy()
    threshold = margin + model.radius*.02 + .005 + .002
    def clearance(point):
        return max(model.body_clearance(point, yaw, cap=threshold+.02) for yaw in headings)
    directions = np.array([[math.cos(a), math.sin(a)] for a in np.arange(16)*math.pi/8])
    initial = np.max([model.clearance_over_poses(
        points, np.full(len(points), yaw), cap=threshold+.02)
        for yaw in headings], axis=0)
    affected = np.zeros(len(points), dtype=bool)
    for index in range(1, len(points)-1):
        if initial[index] >= threshold:
            continue
        affected[max(1,index-5):min(len(points)-1,index+6)] = True
        # Most corner cuts have a nearby outward normal. Solve those locally
        # before the 16-direction radial fallback (up to 384 polygon checks
        # per point). Every proposal still passes the same exact footprint
        # test and 0.12 m corridor bound; the final swept check is unchanged.
        candidate = points[index].copy()
        for _ in range(6):
            value = clearance(candidate)
            if value >= threshold:
                break
            gradient = np.array([
                clearance(candidate+offset)-clearance(candidate-offset)
                for offset in (np.array([.002,0.]),np.array([0.,.002]))])
            norm = float(np.linalg.norm(gradient))
            if norm < 1.e-9:
                # Collision clearance is flat at zero. A center-wall normal
                # supplies a seed, whose actual body clearance is checked.
                _,gradient = model.clearance_and_gradient(candidate)
                norm = float(np.linalg.norm(gradient))
            if norm < 1.e-9:
                break
            candidate += min(.03,threshold-value+.002)*gradient/norm
            if np.linalg.norm(candidate-points[index]) > .12:
                break
        if (np.linalg.norm(candidate-points[index]) <= .12
                and clearance(candidate) >= threshold):
            repaired[index] = candidate
            continue
        for radius in np.arange(.005, .121, .005):
            options = points[index] + radius*directions
            values = np.array([clearance(p) for p in options])
            best = int(np.argmax(values))
            if values[best] >= threshold:
                repaired[index] = options[best]
                break
    if not np.any(affected):
        return repaired
    for _ in range(8):
        proposed = .25*repaired[:-2]+.5*repaired[1:-1]+.25*repaired[2:]
        for index, point in enumerate(proposed,1):
            if (affected[index] and np.linalg.norm(point-points[index]) <= .12
                    and clearance(point) >= threshold):
                repaired[index] = point
    return repaired


def prepare_stage(model, points, pose, state, travel_margin=.025,
                  turn_margin=.12, stopping_distance=0.):
    """Return (new state, translation path, fixed heading), or raise.

    A gate remains fixed across replans. Selection minimizes a simple cost on
    the supplied route: prefer clearance up to 0.35 m, then an earlier turn.
    This is a bounded search on the Nav2 route, not a global SE(2) optimum.
    """
    if model is None or model.footprint is None:
        raise ValueError('FOOTPRINT_UNAVAILABLE')
    points = dense_path(points)
    # Discard the already traversed prefix of a replan without skipping the
    # connector from the measured pose to the remaining checked route.
    points = dense_path(remaining_path(points, pose[:2]))
    if state.phase in ('ROTATE', 'SETTLE'):
        return state, np.array([state.gate]), state.heading
    heading = state.target if state.phase == 'TRANSLATE' else state.heading
    points, old = path_clearances(model, points, heading)
    required = np.full(len(points), travel_margin)
    # An actual tracking error may leave the base inside the planning reserve
    # while its polygon still has >=30 mm clearance. Permit only a short
    # connector back into the normal corridor. The command gate executes this
    # at <=0.05 m/s with zero yaw and requires increasing clearance.
    if (old[0] < travel_margin and model.body_clearance(pose[:2], pose[2]) >= .03
            and abs(wrap(heading-pose[2])) < .01):
        arc = np.r_[0.,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
        required = np.interp(arc,[0.,.10],[old[0]-.0001,travel_margin])
        required[-1] = travel_margin
    if state.phase == 'TRANSLATE':
        if np.any(old < required):
            raise ValueError('FIXED_HEADING_PATH_BLOCKED')
        return state, points, heading
    if abs(wrap(state.target-state.heading)) < .01:
        if np.any(old < required):
            raise ValueError('FIXED_HEADING_PATH_BLOCKED')
        state.phase = 'TRANSLATE'
        return state, points, state.target
    _, new = path_clearances(model, points, state.target)
    prefix = np.minimum.accumulate(old-required)
    suffix = np.minimum.accumulate(new[::-1]-travel_margin)[::-1]
    if state.gate is not None:
        index = int(np.argmin(np.linalg.norm(points-state.gate, axis=1)))
        # Global replanning can move its route sideways while we approach an
        # already selected open turning area. Keep that area, with explicit
        # checked connectors on both sides, instead of latching a stop because
        # the new cell-center path misses it by 10 cm.
        prefix_points = np.vstack((points[:index], state.gate))
        suffix_points = np.vstack((state.gate, points[index+1:]))
        if (np.min(path_clearances(model, prefix_points, heading)[1]) < travel_margin
                or np.min(path_clearances(model, suffix_points, state.target)[1]) < travel_margin
                or rotation_clearance(model, state.gate, heading, state.target) < turn_margin):
            raise ValueError('ROTATION_GATE_BLOCKED')
        return state, prefix_points, heading
    candidates = np.flatnonzero((prefix >= 0.) & (suffix >= 0.))
    # At most one candidate per 0.10 m, plus the last feasible candidate.
    candidates = np.unique(np.r_[candidates[::5], candidates[-1:]])
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    feasible = []
    best_score = math.inf
    for index in candidates:
        if arc[index]*.035-.35 >= best_score:
            continue
        if arc[index] < stopping_distance:
            continue
        clearance = rotation_clearance(model, points[index], heading, state.target)
        if clearance >= turn_margin:
            score = arc[index] * .035 - min(clearance, .35)
            feasible.append((score, int(index), clearance))
            best_score = min(best_score, score)
    if not feasible:
        raise ValueError('NO_SAFE_ROTATION_GATE')
    _, index, state.clearance = min(feasible)
    state.gate = points[index].copy()
    state.phase = 'APPROACH'
    return state, points[:index+1], heading


def free_rotation_disk(data, resolution, origin, center, radius):
    """A full disk must fit in observed free space of a local OccupancyGrid.

    Occupied/unknown cells are closed squares; inflation-only costs <100 are
    not obstacles. Out-of-map and malformed data fail closed.
    """
    grid = np.asarray(data)
    if grid.ndim != 2 or not grid.size or resolution <= 0:
        return False
    xy = (np.asarray(center)-np.asarray(origin))/resolution
    r = radius/resolution
    h, w = grid.shape
    if xy[0]-r < 0 or xy[1]-r < 0 or xy[0]+r >= w or xy[1]+r >= h:
        return False
    x0, x1 = int(math.floor(xy[0]-r)), int(math.floor(xy[0]+r))+1
    y0, y1 = int(math.floor(xy[1]-r)), int(math.floor(xy[1]+r))+1
    iy, ix = np.indices((y1-y0, x1-x0))
    dx = np.maximum(np.abs(ix+x0+.5-xy[0])-.5, 0.)
    dy = np.maximum(np.abs(iy+y0+.5-xy[1])-.5, 0.)
    overlaps = dx*dx+dy*dy <= r*r
    cells = grid[y0:y1, x0:x1]
    return not np.any(overlaps & ((cells < 0) | (cells >= 100)))
