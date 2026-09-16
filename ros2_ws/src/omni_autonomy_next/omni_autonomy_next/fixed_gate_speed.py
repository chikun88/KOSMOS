"""Preview braking only for tight turns at fixed lane mouths."""
from pathlib import Path

import numpy as np
import yaml

from .route_approaches import _load_waypoints


class FixedGateSpeed:
    def __init__(self, gates, speed=.20, radius=.35, reaction_sec=.30,
                 corner_only=True, exit_hold_m=.25, entry_hold_m=.25, transit_speed=.40):
        self.gates = np.asarray(gates, dtype=float).reshape(-1, 2)
        self.speed, self.radius, self.reaction_sec = speed, radius, reaction_sec
        self.corner_only, self.exit_hold_m = corner_only, exit_hold_m
        self.entry_hold_m, self.transit_speed = entry_hold_m, transit_speed
        self.transit_gates = np.empty((0, 2))
        self.gate_speed_overrides = {}
        if (not np.isfinite(self.gates).all()
                or not np.isfinite([speed, radius, reaction_sec, exit_hold_m, entry_hold_m, transit_speed]).all()
                or speed <= 0. or radius <= 0. or reaction_sec < 0. or exit_hold_m < 0.
                or entry_hold_m < 0. or transit_speed <= 0.):
            raise ValueError('invalid fixed gate speed configuration')

    @classmethod
    def from_yaml(cls, path):
        data = yaml.safe_load(Path(path).read_text(encoding='utf-8')) or {}
        frame = data.get('frame_id', 'map')
        if frame != 'map':
            raise ValueError('fixed gate speed geometry must use map frame')
        entries = list((data.get('fixed_departures') or {}).values())
        for approach in (data.get('fixed_goal_approaches') or {}).values():
            entries.extend(approach[side] for side in ('upper', 'lower'))
        gates = set()
        for entry in entries:
            for point in _load_waypoints(entry.get('waypoints'), frame, 'speed gate'):
                gates.add((point['x'], point['y']))
                gates.add((-point['x'], point['y']))
        tuning = data.get('fixed_gate_tracking') or {}
        result = cls(sorted(gates), speed=float(tuning.get('speed_m_s', .20)),
                   radius=float(tuning.get('slow_radius_m', .35)),
                   reaction_sec=float(tuning.get('reaction_sec', .30)),
                   exit_hold_m=float(tuning.get('exit_hold_m', .25)),
                   entry_hold_m=float(tuning.get('entry_hold_m', .25)),
                   transit_speed=float(tuning.get('transit_turn_speed_m_s', .40)))
        result.transit_gates = np.array([
            [sign*p['x'], p['y']] for p in data.get('fixed_bucket_transit', {}).get('waypoints', [])
            if p.get('outer_y_passage') for sign in (-1., 1.)], dtype=float).reshape(-1, 2)
        if not np.isfinite(result.transit_gates).all():
            raise ValueError('invalid transit turn geometry')
        gate_caps = {}
        for entry in entries:
            cap = float(entry.get('turn_speed_m_s', result.speed))
            if not np.isfinite(cap) or cap <= 0.:
                raise ValueError('invalid fixed gate turn_speed_m_s')
            for point in _load_waypoints(entry.get('waypoints'), frame, 'speed gate'):
                for sign in (-1., 1.):
                    key = (sign*point['x'], point['y'])
                    gate_caps[key] = min(cap, gate_caps.get(key, np.inf))
        result.gate_speed_overrides = {key: cap for key, cap in gate_caps.items()
                                       if cap != result.speed}
        return result

    def limits(self, points, acceleration, *, curvature=None, arclength=None):
        return self.plan_limits(points, acceleration, curvature=curvature,
                                arclength=arclength)[0]

    def plan_limits(self, points, acceleration, *, curvature=None, arclength=None,
                    retained_turns=()):
        """Bound speed before the gate, allowing for response delay.

        At distance d outside the slow region, solve
        (v-v_gate)*reaction + (v**2-v_gate**2)/(2*a) <= d.
        The trajectory's backward acceleration sweep also consumes this cap;
        this is not a sudden output clamp at the waypoint. Euclidean distance
        is conservative for curved approaches and does not prune any gates.
        """
        points = np.asarray(points, dtype=float)
        if not len(self.gates):
            return np.full(len(points), np.inf), ()
        distance = np.min(np.linalg.norm(
            points[:, None, :] - self.gates[None, :, :], axis=2), axis=1)
        a = max(float(acceleration), 1.e-9)
        delay = a*self.reaction_sec
        if self.corner_only:
            if curvature is None or arclength is None:
                raise ValueError('corner speed planning requires curvature and arclength')
            arc = np.asarray(arclength)
            turns = {(x, y): cap for x, y, cap in retained_turns}
            distances = np.linalg.norm(points[:, None, :] - self.gates[None, :, :], axis=2)
            nearest = np.argmin(distances, axis=1)
            # Ignore localization correction kinks at a new plan's start.
            turning = (np.asarray(curvature) >= 2.) & (arc >= .10)
            turning &= distances[np.arange(len(points)), nearest] <= self.radius
            for index in np.unique(nearest[turning]):
                key = tuple(self.gates[index])
                cap = self.gate_speed_overrides.get(key, self.speed)
                turns[key] = min(cap, turns.get(key, np.inf))
            limits = np.full(len(points), np.inf)
            # A replan can begin inside a turn, where its first cells no longer
            # contain the old bend. Retain that turn until the physical path
            # has cleared it; otherwise every replan can release braking early.
            delta = np.diff(points, axis=0)
            length2 = np.sum(delta*delta, axis=1)
            for (x, y), speed in turns.items():
                if not len(delta):
                    continue
                fraction = np.clip(np.sum((np.array([x, y])-points[:-1])*delta, axis=1)
                                   /np.maximum(length2, 1.e-12), 0., 1.)
                projection = points[:-1]+fraction[:, None]*delta
                distances = np.linalg.norm(projection-[x, y], axis=1)
                index = int(np.argmin(distances))
                at = arc[index]+fraction[index]*np.sqrt(length2[index])
                if distances[index] > self.radius:
                    continue
                if at < 1.e-6 and distances[index] > self.exit_hold_m+.05:
                    continue
                remaining = at-arc
                # Reach turn speed before the corner, then keep it through
                # the short exit while delayed lateral momentum drains.
                preview = np.sqrt((speed+delay)**2+2*a*np.maximum(0., remaining-self.entry_hold_m))-delay
                limits = np.minimum(limits, np.where(remaining >= -self.exit_hold_m, preview, np.inf))
            if len(self.transit_gates):
                transit_distance = np.min(np.linalg.norm(
                    points[:, None, :] - self.transit_gates[None, :, :], axis=2), axis=1)
                bends = (transit_distance <= .50) & (np.asarray(curvature) >= .75) & (arc >= .10)
                if np.any(bends):
                    remaining = arc[bends][None, :] - arc[:, None]
                    preview = np.sqrt((self.transit_speed+delay)**2+2*a*np.maximum(0., remaining))-delay
                    limits = np.minimum(limits, np.min(np.where(
                        remaining >= -self.exit_hold_m, preview, np.inf), axis=1))
            return limits, tuple((x, y, cap) for (x, y), cap in turns.items())
        remaining = np.maximum(0., distance-self.radius)
        return np.sqrt((self.speed+delay)**2 + 2*a*remaining)-delay, ()
