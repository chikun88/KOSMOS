"""Bounded straight reverse docking, in the saved goal's body direction."""
import math
import numpy as np
from .staged_heading import wrap
from .field_side import apply_pose

DISTANCE = .25
# Cruise faster while retaining the delay-compensated terminal slowdown.
SPEED = .10
TIMEOUT = 20.
# End docking without a prolonged millimetre-scale correction tail.
# The tracker uses these same bounds before confirming the measured stop.
POSITION_TOLERANCE = .010
YAW_TOLERANCE = .015


def reverse_target(configured, side, clearance):
    """Validate the exact saved docking pose without moving its endpoint."""
    target = apply_pose(configured, side)

    def corridor_clear(pose, floor=.026):
        gate = reverse_gate(pose)
        points = np.linspace([gate['x'], gate['y']], [pose['x'], pose['y']], 61)
        return np.min(clearance.clearance_over_poses(
            points, np.full(len(points), pose['yaw']), cap=.1)) >= floor

    if corridor_clear(target):
        return target, 0.
    raise ValueError('REVERSE_CLEARANCE_BLOCKED')


def reverse_gate(configured):
    gate = dict(configured)
    gate['x'] += DISTANCE * math.cos(gate['yaw'])
    gate['y'] += DISTANCE * math.sin(gate['yaw'])
    return gate


def reverse_command(pose, goal, measured, scale, delay):
    """Bounded docking with lateral/overshoot correction near the goal."""
    c, s = math.cos(goal[2]), math.sin(goal[2])
    delta = np.asarray(pose[:2]) - goal[:2]
    along = c*delta[0] + s*delta[1]
    cross = -s*delta[0] + c*delta[1]
    error = wrap(goal[2]-pose[2])
    if (along < -.015 or along > .28 or abs(cross) > .02
            or abs(error) > .04):
        return (0., 0., 0.), 'REVERSE_DEVIATION'
    angle = pose[2]-goal[2]
    ca, sa = math.cos(angle), math.sin(angle)
    forward = ca*measured[0]-sa*measured[1]
    lateral = sa*measured[0]+ca*measured[1]
    along_speed = float(np.clip(-.8*(along+forward*delay), -SPEED*scale, .01*scale))
    cross_speed = float(np.clip(-(cross+lateral*delay), -.01*scale, .01*scale))
    vx = ca*along_speed+sa*cross_speed
    vy = -sa*along_speed+ca*cross_speed
    wz = float(np.clip(1.2*wrap(error-measured[2]*delay), -.08*scale, .08*scale))
    if math.hypot(along, cross) <= POSITION_TOLERANCE and abs(error) <= YAW_TOLERANCE:
        vx = vy = wz = 0.
    return tuple(0. if abs(v) < 1.e-12 else v for v in (vx, vy, wz)), 'REVERSING'
