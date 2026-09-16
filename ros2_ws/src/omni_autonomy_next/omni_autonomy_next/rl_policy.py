"""Pure runtime representation of a promoted reinforcement-learning policy."""

from bisect import bisect_right
from dataclasses import dataclass
import math
from pathlib import Path

import yaml


# Must stay in the same order as simulation/reinforcement_learning.DEFAULT_BINS.
# ``clearance_margin`` is the footprint clearance, not a base_link margin.
STATE_FIELDS = (
    'remaining_distance',
    'clearance_margin',
    'turn_error',
    'speed_fraction',
    'goal_clearance_margin',
    'yaw_error',
)


@dataclass(frozen=True)
class RLObservation:
    remaining_distance: float
    clearance_margin: float
    turn_error: float
    speed_fraction: float
    goal_clearance_margin: float
    yaw_error: float


@dataclass(frozen=True)
class RLAction:
    speed_scale: float
    clearance_push: float


@dataclass(frozen=True)
class PolicyDecision:
    state: tuple[int, ...]
    action_index: int
    action: RLAction
    learned_override: bool


class CompactRLPolicy:
    """Validated greedy table exported from the offline Q learner.

    Only state/action pairs with enough training visits are exported. Every
    other state falls back to the unmodified deterministic controller.
    """

    def __init__(self, data):
        if int(data.get('format_version', 0)) != 2:
            raise ValueError('unsupported deployed RL policy format')
        if data.get('algorithm') != 'tabular_q_learning_greedy':
            raise ValueError('unsupported deployed RL policy algorithm')
        certification = data.get('certification', {})
        # A residual may be slower than the deterministic controller: on the
        # bucket-lane transits the deterministic controller does not arrive at
        # all.  It may never regress a route that already worked, and it may
        # never have collided in evaluation.
        if certification.get('paired_regression_passed') is not True:
            raise ValueError(
                'RL policy carries no paired non-regression certification'
            )
        if certification.get('evaluation_collisions') != 0:
            raise ValueError('RL policy collided during evaluation')

        self.bins = {
            name: tuple(float(value) for value in data['bins'][name])
            for name in STATE_FIELDS
        }
        for name, values in self.bins.items():
            if values != tuple(sorted(set(values))):
                raise ValueError(f'{name} bins must be finite and increasing')
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f'{name} bins must be finite and increasing')

        self.actions = tuple(
            RLAction(
                speed_scale=float(item['speed_scale']),
                clearance_push=float(item['clearance_push']),
            )
            for item in data['actions']
        )
        if not self.actions:
            raise ValueError('deployed RL policy has no actions')
        for action in self.actions:
            if not 0.0 < action.speed_scale <= 1.0:
                raise ValueError('RL speed_scale must be within (0, 1]')
            if not 1.0 <= action.clearance_push <= 2.0:
                raise ValueError('RL clearance_push must be within [1, 2]')

        self.default_action = int(data['default_action'])
        if not 0 <= self.default_action < len(self.actions):
            raise ValueError('default_action is out of range')
        fallback = self.actions[self.default_action]
        if fallback != RLAction(1.0, 1.0):
            raise ValueError('default RL action must leave the controller unchanged')

        self.overrides = {}
        state_shape = tuple(len(self.bins[name]) + 1 for name in STATE_FIELDS)
        for encoded, raw_action in data.get('overrides', {}).items():
            state = tuple(int(value) for value in str(encoded).split(','))
            if len(state) != len(STATE_FIELDS) or any(
                value < 0 or value >= maximum
                for value, maximum in zip(state, state_shape)
            ):
                raise ValueError(f'invalid RL state override: {encoded}')
            action_index = int(raw_action)
            if not 0 <= action_index < len(self.actions):
                raise ValueError(f'invalid RL action override: {raw_action}')
            self.overrides[state] = action_index

        self.decision_period_sec = float(data.get('decision_period_sec', 0.25))
        if not 0.05 <= self.decision_period_sec <= 1.0:
            raise ValueError('decision_period_sec must be within [0.05, 1.0]')
        self.convergence_distance = float(
            data.get('convergence_distance_m', 0.35)
        )
        if not 0.0 <= self.convergence_distance <= 1.0:
            raise ValueError('convergence_distance_m must be within [0, 1]')
        # The residual can only slow the controller or push it away from a wall,
        # so it is inert when no wall is near.  Enforcing that here as well as
        # at export time keeps an open-field command bit-identical.
        self.active_clearance_margin = float(
            data.get('active_clearance_margin_m', 0.10)
        )
        if not 0.0 < self.active_clearance_margin <= 1.0:
            raise ValueError('active_clearance_margin_m must be within (0, 1]')
        self._decision = None
        self._next_decision_time = -math.inf

    @classmethod
    def from_yaml(cls, path):
        with Path(path).open(encoding='utf-8') as stream:
            return cls(yaml.safe_load(stream))

    def reset(self):
        self._decision = None
        self._next_decision_time = -math.inf

    def state(self, observation):
        if not isinstance(observation, RLObservation):
            raise TypeError('observation must be RLObservation')
        values = tuple(float(getattr(observation, name)) for name in STATE_FIELDS)
        if not all(math.isfinite(value) for value in values):
            raise ValueError('RL observation values must be finite')
        return tuple(
            bisect_right(self.bins[name], value)
            for name, value in zip(STATE_FIELDS, values)
        )

    def decide(self, observation, now_sec):
        now_sec = float(now_sec)
        if self._decision is not None and now_sec < self._next_decision_time:
            return self._decision
        state = self.state(observation)
        shielded = (
            observation.remaining_distance <= self.convergence_distance
            or observation.clearance_margin >= self.active_clearance_margin
        )
        action_index = (
            self.default_action if shielded
            else self.overrides.get(state, self.default_action)
        )
        self._decision = PolicyDecision(
            state=state,
            action_index=action_index,
            action=self.actions[action_index],
            learned_override=not shielded and state in self.overrides,
        )
        self._next_decision_time = now_sec + self.decision_period_sec
        return self._decision
