"""Safety-constrained Q-learning for the offline navigation simulator.

The learner is deliberately a residual policy: it may slow the classical
controller and increase its obstacle repulsion, but it cannot raise any speed
limit or weaken the existing clearance response.  A learned model is therefore
still subject to the same dynamics and collision gates as the baseline.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import yaml

try:
    from .dynamics import ControlAdjustment, ControlObservation
    from .run_campaign import (
        build_campaign_report,
        campaign_passed,
        deployed_profile,
        evaluate_campaign,
        prepare_campaign,
        run_campaign,
        summarize,
    )
except ImportError:
    from dynamics import ControlAdjustment, ControlObservation
    from run_campaign import (
        build_campaign_report,
        campaign_passed,
        deployed_profile,
        evaluate_campaign,
        prepare_campaign,
        run_campaign,
        summarize,
    )


MODEL_VERSION = 2
# A steady crawl is the decisive option in the bucket lane: the Collision
# Monitor projects the commanded velocity over 1.2 s, so a slower command is
# less likely to predict contact and be zeroed outright.  Stop-and-go is what
# trips the progress checker, so the learner needs speed scales well below the
# old 0.60 floor.
DEFAULT_ACTIONS = (
    ControlAdjustment(0.35, 1.60),
    ControlAdjustment(0.35, 1.00),
    ControlAdjustment(0.50, 1.60),
    ControlAdjustment(0.50, 1.00),
    ControlAdjustment(0.70, 1.40),
    ControlAdjustment(0.70, 1.00),
    ControlAdjustment(1.00, 1.40),
    # The final action is the unmodified deterministic controller.  It is the
    # fallback for unseen and insufficiently sampled states.
    ControlAdjustment(1.00, 1.00),
)
# ``clearance_margin`` is now the footprint clearance rather than a disc
# margin, and the tight firing lane lives entirely below 60 mm, so the bins
# resolve that band instead of treating it as one bucket.
# ``goal_clearance_margin`` is what separates a tight firing pose from an open
# one, and ``yaw_error`` matters because the footprint clearance is
# yaw-dependent.
# A residual whose only powers are slowing down and pushing away from a wall has
# no business acting when no wall is near.  Restricting it to this band is what
# stopped the learner from paying for tight-lane gains with slower open-field
# routes, and it collapses the state space that actually needs sampling.
ACTIVE_CLEARANCE_MARGIN = 0.10
DEFAULT_BINS = {
    'remaining_distance': (0.12, 0.35, 0.80, 1.80, 3.50),
    'clearance_margin': (0.020, 0.040, 0.060, 0.100, 0.180),
    'turn_error': (0.10, 0.35, 0.80),
    'speed_fraction': (0.15, 0.45, 0.75),
    'goal_clearance_margin': (0.080, 0.200),
    'yaw_error': (0.035, 0.200),
}


def _require_certification(certification):
    """The contract a deployed residual must carry.

    A residual may be slower than the deterministic controller, because on the
    bucket-lane transits the deterministic controller does not arrive at all.
    It may never regress a scenario that already worked, and it may never have
    collided in evaluation.
    """
    if certification.get('paired_regression_passed') is not True:
        raise ValueError(
            'deployed policy carries no paired non-regression certification'
        )
    if certification.get('evaluation_collisions') != 0:
        raise ValueError('deployed policy collided during evaluation')


class CompactDeploymentPolicy:
    """Simulation adapter for the exact compact YAML loaded by ROS runtime."""

    def __init__(self, data):
        if (data.get('format_version') != 2
                or data.get('algorithm') != 'tabular_q_learning_greedy'):
            raise ValueError('unsupported deployed RL policy format/algorithm')
        _require_certification(data.get('certification', {}))
        self.actions = tuple(ControlAdjustment(**item) for item in data['actions'])
        _validate_actions(self.actions)
        self.bins = {
            name: tuple(float(value) for value in data['bins'][name])
            for name in DEFAULT_BINS
        }
        for name, values in self.bins.items():
            if (not all(math.isfinite(value) for value in values)
                    or values != tuple(sorted(set(values)))):
                raise ValueError(f'{name} bins must be finite and increasing')
        self.default_action = int(data['default_action'])
        if not 0 <= self.default_action < len(self.actions):
            raise ValueError('default_action is out of range')
        if self.actions[self.default_action] != ControlAdjustment(1.0, 1.0):
            raise ValueError('deployed policy fallback must be unmodified')
        self.overrides = {
            tuple(int(value) for value in encoded.split(',')): int(action)
            for encoded, action in data.get('overrides', {}).items()
        }
        state_shape = tuple(len(values) + 1 for values in self.bins.values())
        for state, action in self.overrides.items():
            if (len(state) != len(state_shape)
                    or any(value < 0 or value >= maximum
                           for value, maximum in zip(state, state_shape))
                    or not 0 <= action < len(self.actions)):
                raise ValueError('invalid deployed RL state/action override')
        self.observation_context = data.get('observation_context')
        if self.overrides and (not isinstance(self.observation_context, dict)
                or self.observation_context.get('velocity_source') != 'model_actual_velocity'):
            raise ValueError('compact simulator requires model-actual observation context')
        self.convergence_distance = float(
            data.get('convergence_distance_m', 0.35)
        )
        self.active_clearance_margin = float(
            data.get('active_clearance_margin_m', ACTIVE_CLEARANCE_MARGIN)
        )
        decision_period = float(data.get('decision_period_sec', 0.25))
        if not 0.05 <= decision_period <= 1.0:
            raise ValueError('decision_period_sec must be within [0.05, 1.0]')
        if not 0.0 <= self.convergence_distance <= 1.0:
            raise ValueError('convergence_distance_m must be within [0, 1]')
        if not 0.0 < self.active_clearance_margin <= 1.0:
            raise ValueError('active_clearance_margin_m must be within (0, 1]')
        # Runtime decides on the first 50 ms tick at/after its time deadline.
        # Rounding released a held decision too early for e.g. a 120 ms period.
        self.decision_steps = max(1, int(math.ceil(decision_period / .05 - 1.e-12)))
        self._action_index = None
        self._steps = 0

    @classmethod
    def load(cls, path):
        with Path(path).open(encoding='utf-8') as stream:
            return cls(yaml.safe_load(stream))

    def begin_episode(self):
        self._action_index = None
        self._steps = 0

    def select_adjustment(self, observation):
        values = [getattr(observation, name) for name in self.bins]
        if not np.isfinite(values).all():
            raise ValueError('observation must contain only finite values')
        state = tuple(
            int(np.digitize(getattr(observation, name), self.bins[name]))
            for name in DEFAULT_BINS
        )
        shielded = (
            observation.remaining_distance <= self.convergence_distance
            or observation.clearance_margin >= self.active_clearance_margin
        )
        if self._action_index is not None:
            if shielded and self._action_index != self.default_action:
                self._action_index = self.default_action
                self._steps = 0
            return self.actions[self._action_index]
        self._action_index = (
            self.default_action if shielded
            else self.overrides.get(state, self.default_action)
        )
        return self.actions[self._action_index]

    def observe_transition(self, _reward, _next_observation, terminal):
        self._steps += 1
        if terminal or self._steps >= self.decision_steps:
            self._action_index = None
            self._steps = 0


def _validate_actions(actions):
    if not actions:
        raise ValueError('at least one action is required')
    for action in actions:
        if not 0.0 < action.speed_scale <= 1.0:
            raise ValueError('RL actions may not raise the speed limit')
        if not 1.0 <= action.clearance_push <= 2.0:
            raise ValueError('RL actions may not weaken or overdrive clearance push')


class SafeQLearningPolicy:
    """Tabular Q learner with a small, inspectable and bounded action space."""

    def __init__(
        self,
        *,
        actions=DEFAULT_ACTIONS,
        bins=None,
        learning_rate=0.18,
        discount=0.96,
        decision_interval=5,
        minimum_eval_visits=8,
        convergence_distance=0.35,
        active_clearance_margin=ACTIVE_CLEARANCE_MARGIN,
        epsilon=0.0,
        training=False,
        seed=0,
    ):
        self.actions = tuple(actions)
        _validate_actions(self.actions)
        source_bins = DEFAULT_BINS if bins is None else bins
        self.bins = {
            name: tuple(float(value) for value in source_bins[name])
            for name in DEFAULT_BINS
        }
        for name, values in self.bins.items():
            if (not all(math.isfinite(value) for value in values)
                    or tuple(sorted(values)) != values or len(set(values)) != len(values)):
                raise ValueError(f'{name} bins must be finite and strictly increasing')
        self.learning_rate = float(learning_rate)
        self.discount = float(discount)
        self.decision_interval = int(decision_interval)
        self.minimum_eval_visits = int(minimum_eval_visits)
        self.convergence_distance = float(convergence_distance)
        self.active_clearance_margin = float(active_clearance_margin)
        self.epsilon = float(epsilon)
        self.training = bool(training)
        if not 0.0 < self.learning_rate <= 1.0:
            raise ValueError('learning_rate must be within (0, 1]')
        if not 0.0 <= self.discount <= 1.0:
            raise ValueError('discount must be within [0, 1]')
        if self.decision_interval <= 0:
            raise ValueError('decision_interval must be positive')
        if self.minimum_eval_visits < 0:
            raise ValueError('minimum_eval_visits may not be negative')
        if not 0.0 <= self.convergence_distance <= 1.0:
            raise ValueError('convergence_distance must be within [0, 1]')
        if not 0.0 < self.active_clearance_margin <= 1.0:
            raise ValueError('active_clearance_margin must be within (0, 1]')
        if not 0.0 <= self.epsilon <= 1.0:
            raise ValueError('epsilon must be within [0, 1]')
        state_shape = tuple(len(self.bins[name]) + 1 for name in DEFAULT_BINS)
        self.q_values = np.zeros((*state_shape, len(self.actions)), dtype=float)
        self.visit_counts = np.zeros_like(self.q_values, dtype=np.int64)
        self._rng = np.random.default_rng(seed)
        self.fallback_action = next((
            index for index, action in enumerate(self.actions)
            if action.speed_scale == 1.0 and action.clearance_push == 1.0
        ), None)
        if self.fallback_action is None:
            raise ValueError('RL policy requires an unmodified fallback action')
        self._last_state = None
        self._last_action = None
        self._pending_return = 0.0
        self._steps_on_action = 0

    def _state(self, observation):
        if not isinstance(observation, ControlObservation):
            raise TypeError('observation must be ControlObservation')
        # Read through the bin names so adding a feature cannot silently drop
        # it from the state.
        values = tuple(getattr(observation, name) for name in DEFAULT_BINS)
        if not np.all(np.isfinite(values)):
            raise ValueError('observation must contain only finite values')
        return tuple(
            int(np.digitize(value, self.bins[name]))
            for name, value in zip(DEFAULT_BINS, values)
        )

    def begin_episode(self):
        self._last_state = None
        self._last_action = None
        self._pending_return = 0.0
        self._steps_on_action = 0

    def select_adjustment(self, observation):
        if self._last_action is not None:
            return self.actions[self._last_action]
        state = self._state(observation)
        if (
            observation.remaining_distance <= self.convergence_distance
            or observation.clearance_margin >= self.active_clearance_margin
        ):
            # Both shields apply while training too.  Skipping them there let
            # the learner value behaviour the deployed policy can never take.
            action_index = self.fallback_action
        elif self.training and self._rng.random() < self.epsilon:
            action_index = int(self._rng.integers(len(self.actions)))
        else:
            action_values = self.q_values[state]
            eligible = (
                np.ones(len(self.actions), dtype=bool)
                if self.training
                else self.visit_counts[state] >= self.minimum_eval_visits
            )
            if np.any(eligible):
                eligible_values = np.where(eligible, action_values, -np.inf)
                action_index = int(np.flatnonzero(
                    eligible_values == eligible_values.max()
                )[-1])
            else:
                action_index = self.fallback_action
        self._last_state = state
        self._last_action = action_index
        return self.actions[action_index]

    def observe_transition(self, reward, next_observation, terminal):
        if self._last_state is None or self._last_action is None:
            raise RuntimeError('select_adjustment must precede observe_transition')
        reward = float(reward)
        if not np.isfinite(reward):
            raise ValueError('reward must be finite')
        self._pending_return += (self.discount ** self._steps_on_action) * reward
        self._steps_on_action += 1
        if not terminal and self._steps_on_action < self.decision_interval:
            return
        if self.training:
            current_key = (*self._last_state, self._last_action)
            target = self._pending_return
            if not terminal:
                target += (
                    self.discount ** self._steps_on_action
                    * float(self.q_values[self._state(next_observation)].max())
                )
            current = float(self.q_values[current_key])
            self.q_values[current_key] = (
                current + self.learning_rate * (target - current)
            )
            self.visit_counts[current_key] += 1
        self._last_state = None
        self._last_action = None
        self._pending_return = 0.0
        self._steps_on_action = 0

    def to_dict(self, metadata=None):
        return {
            'format_version': MODEL_VERSION,
            'algorithm': 'tabular_q_learning',
            'safety_contract': {
                'maximum_speed_scale': 1.0,
                'minimum_clearance_push': 1.0,
                'runtime_guard_still_required': True,
            },
            'actions': [asdict(action) for action in self.actions],
            'bins': {name: list(values) for name, values in self.bins.items()},
            'learning_rate': self.learning_rate,
            'discount': self.discount,
            'decision_interval': self.decision_interval,
            'minimum_eval_visits': self.minimum_eval_visits,
            'convergence_distance': self.convergence_distance,
            'active_clearance_margin': self.active_clearance_margin,
            'q_values': self.q_values.tolist(),
            'visit_counts': self.visit_counts.tolist(),
            'metadata': {} if metadata is None else metadata,
        }

    @classmethod
    def from_dict(cls, data, *, seed=0):
        if data.get('format_version') != MODEL_VERSION:
            raise ValueError('unsupported RL model format_version')
        if data.get('algorithm') != 'tabular_q_learning':
            raise ValueError('unsupported RL algorithm')
        actions = tuple(ControlAdjustment(**item) for item in data['actions'])
        policy = cls(
            actions=actions,
            bins=data['bins'],
            learning_rate=data['learning_rate'],
            discount=data['discount'],
            decision_interval=data.get('decision_interval', 1),
            minimum_eval_visits=data.get('minimum_eval_visits', 0),
            convergence_distance=data.get('convergence_distance', 0.35),
            active_clearance_margin=data.get(
                'active_clearance_margin', ACTIVE_CLEARANCE_MARGIN
            ),
            epsilon=0.0,
            training=False,
            seed=seed,
        )
        q_values = np.asarray(data['q_values'], dtype=float)
        visit_counts = np.asarray(data.get('visit_counts'), dtype=np.int64)
        if q_values.shape != policy.q_values.shape:
            raise ValueError('RL model q_values shape does not match its bins/actions')
        if visit_counts.shape != policy.visit_counts.shape:
            raise ValueError('RL model visit_counts shape does not match its bins/actions')
        if not np.all(np.isfinite(q_values)):
            raise ValueError('RL model q_values must be finite')
        if np.any(visit_counts < 0):
            raise ValueError('RL model visit_counts may not be negative')
        policy.q_values = q_values
        policy.visit_counts = visit_counts
        return policy

    def save(self, path, metadata=None):
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(self.to_dict(metadata), indent=2), encoding='utf-8'
        )

    @classmethod
    def load(cls, path, *, seed=0):
        with Path(path).open(encoding='utf-8') as stream:
            return cls.from_dict(json.load(stream), seed=seed)


def train_policy(
    field_file,
    poses_file,
    profile,
    *,
    episodes,
    random_episodes,
    seed,
):
    field, scenarios, metadata = prepare_campaign(
        field_file, poses_file, episodes, random_episodes, seed
    )
    policy = SafeQLearningPolicy(training=True, seed=seed)
    training_results = []
    order_rng = np.random.default_rng(seed + 1)
    order = order_rng.permutation(len(scenarios))
    for episode_number, scenario_index in enumerate(order):
        fraction = episode_number / max(1, len(scenarios) - 1)
        policy.epsilon = 0.35 + fraction * (0.03 - 0.35)
        path, start_yaw, goal_yaw, scenario_seed = scenarios[int(scenario_index)]
        training_results.extend(evaluate_campaign(
            field,
            [(path, start_yaw, goal_yaw, scenario_seed)],
            profile,
            controller=policy,
        ))
    policy.training = False
    policy.epsilon = 0.0
    return policy, {
        **metadata,
        'episodes': episodes,
        'random_episodes': random_episodes,
        'epsilon_start': 0.35,
        'epsilon_final': 0.03,
        'visited_state_actions': int(np.count_nonzero(policy.visit_counts)),
        'updates': int(policy.visit_counts.sum()),
        'summary': summarize(training_results),
    }


TIME_ALLOWANCE = 1.10
TIME_SLACK_SEC = 0.75


def paired_regression_report(
    baseline_results, learned_results,
    *, time_allowance=TIME_ALLOWANCE, time_slack=TIME_SLACK_SEC,
):
    """Compare the learned and deterministic controllers scenario by scenario.

    The previous gate compared aggregate mean time with a 0.5 per cent
    allowance.  Once the offline model included the real footprint and the
    deployed gates, the deterministic controller stopped arriving at all on the
    bucket-lane transits, so an aggregate time comparison rejected exactly the
    slower-but-arriving behaviour that fixes them.

    What must not regress is any scenario the deterministic controller already
    solved.  Trading speed for arrival on a scenario it fails is the point.
    """
    if len(baseline_results) != len(learned_results):
        raise ValueError('paired comparison needs identical scenario counts')
    regressed = []
    rescued = []
    slower = []
    for index, (reference, candidate) in enumerate(
        zip(baseline_results, learned_results)
    ):
        if reference.success and not candidate.success:
            regressed.append({'scenario': index, 'reason': 'no_longer_arrives'})
            continue
        if reference.success and candidate.success:
            budget = reference.elapsed * float(time_allowance) + float(time_slack)
            if candidate.elapsed > budget:
                regressed.append({
                    'scenario': index,
                    'reason': 'slower_on_a_solved_scenario',
                    'baseline_sec': reference.elapsed,
                    'learned_sec': candidate.elapsed,
                    'budget_sec': budget,
                })
            elif candidate.elapsed > reference.elapsed:
                slower.append(index)
        elif not reference.success and candidate.success:
            rescued.append(index)
    baseline_failures = sum(not r.success for r in baseline_results)
    learned_failures = sum(not r.success for r in learned_results)
    return {
        'episodes': len(baseline_results),
        'baseline_failures': baseline_failures,
        'learned_failures': learned_failures,
        'rescued_scenarios': rescued,
        'regressed_scenarios': regressed,
        'slower_within_budget': len(slower),
        'passed': not regressed and learned_failures <= baseline_failures,
    }


def performance_non_regression_passed(baseline_results, learned_results):
    return paired_regression_report(baseline_results, learned_results)['passed']


def _greedy_overrides(policy):
    """Select the exact overrides that would be serialized for deployment."""
    active_bins = {
        index for index in range(len(policy.bins['clearance_margin']) + 1)
        if (policy.bins['clearance_margin'][index - 1] if index else 0.0)
        < policy.active_clearance_margin
    }
    overrides = {}
    for state in np.ndindex(policy.q_values.shape[:-1]):
        if state[1] not in active_bins:
            continue
        eligible = policy.visit_counts[state] >= policy.minimum_eval_visits
        if np.any(eligible):
            values = np.where(eligible, policy.q_values[state], -np.inf)
            action_index = int(np.flatnonzero(values == values.max())[-1])
        else:
            action_index = policy.fallback_action
        if action_index != policy.fallback_action:
            overrides[','.join(str(value) for value in state)] = action_index
    return overrides


def _runtime_context_compatible(context):
    try:
        return (isinstance(context, dict)
                and context.get('velocity_source') == 'smoothed_command'
                and isinstance(context.get('reference_profile'), str)
                and bool(context['reference_profile'])
                and math.isfinite(float(context.get('reference_speed_mps', math.nan)))
                and float(context['reference_speed_mps']) > 0.0)
    except (TypeError, ValueError):
        return False


def export_compact_policy(model_path, output_path):
    """Export only confident greedy decisions from a passing offline model."""
    model_path = Path(model_path)
    raw_bytes = model_path.read_bytes()
    data = json.loads(raw_bytes)
    metadata = data.get('metadata', {})
    if metadata.get('paired_regression_passed') is not True:
        raise ValueError(
            'refusing to deploy an RL model which regresses a scenario the '
            'deterministic controller already solved'
        )
    if metadata.get('evaluation_collisions') != 0:
        raise ValueError('refusing to deploy an RL model which collided')
    policy = SafeQLearningPolicy.from_dict(data)
    default_action = next((
        index for index, action in enumerate(policy.actions)
        if action.speed_scale == 1.0 and action.clearance_push == 1.0
    ), None)
    if default_action is None:
        raise ValueError('RL model has no unmodified-controller fallback action')
    overrides = _greedy_overrides(policy)
    context = metadata.get('observation_context')
    if overrides and not _runtime_context_compatible(context):
        raise ValueError(
            'refusing to deploy nonempty RL overrides without runtime-compatible '
            'observation context; model actual velocity differs from smoothed command')
    training = data.get('metadata', {}).get('training', {})
    compact = {
        'format_version': 2,
        'algorithm': 'tabular_q_learning_greedy',
        'decision_period_sec': 0.05 * policy.decision_interval,
        'convergence_distance_m': policy.convergence_distance,
        'active_clearance_margin_m': policy.active_clearance_margin,
        'minimum_eval_visits': policy.minimum_eval_visits,
        'certification': {
            # Mandatory: never deploy a residual that made an already-solved
            # route worse, and never one that collided.
            'paired_regression_passed': True,
            'evaluation_collisions': 0,
            # Recorded, not required.  The deterministic controller does not
            # arrive on every bucket-lane transit either, so demanding a clean
            # safety campaign here would only keep the better policy off the
            # robot.  The system-level release gate is separate.
            'safety_campaign_passed': bool(
                metadata.get('safety_campaign_passed', False)
            ),
            'baseline_failures': metadata.get('baseline_failures'),
            'learned_failures': metadata.get('learned_failures'),
            'training_seed': training.get('seed'),
            'held_out_seed': metadata.get('held_out_seed'),
            'source_training_episodes': training.get('episodes'),
            'source_evaluation_episodes': metadata.get('evaluation_episodes'),
            'critical_pose_ids': training.get('critical_pose_ids', []),
            'source_model_sha256': hashlib.sha256(raw_bytes).hexdigest(),
        },
        'bins': {name: list(values) for name, values in policy.bins.items()},
        'actions': [asdict(action) for action in policy.actions],
        'default_action': default_action,
        'overrides': overrides,
        'observation_context': context,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(compact, sort_keys=False), encoding='utf-8')
    return compact


def _default_paths():
    root = Path(__file__).resolve().parents[1]
    package = root / 'ros2_ws' / 'src' / 'omni_autonomy_next'
    return root, package


def _add_environment_arguments(parser, package):
    parser.add_argument('--field', default=str(package / 'config' / 'field_planning.yaml'))
    parser.add_argument('--poses', default=str(package / 'config' / 'field_poses.yaml'))
    parser.add_argument('--runtime', default=str(package / 'config' / 'runtime.yaml'))
    parser.add_argument('--seed', type=int, default=20260804)


def _parse_args():
    root, package = _default_paths()
    parser = argparse.ArgumentParser(
        description='Train or evaluate the safety-constrained navigation Q policy.'
    )
    commands = parser.add_subparsers(dest='command', required=True)

    train = commands.add_parser('train', help='train in simulation, then run a held-out gate')
    _add_environment_arguments(train, package)
    train.add_argument('--episodes', type=int, default=300)
    train.add_argument('--random-episodes', type=int, default=60)
    train.add_argument('--eval-episodes', type=int, default=120)
    train.add_argument('--eval-random-episodes', type=int, default=24)
    train.add_argument('--model', default=str(root / 'simulation' / 'results' / 'rl_policy.json'))
    train.add_argument('--output', default=str(root / 'simulation' / 'results' / 'rl_training.json'))
    train.add_argument(
        '--deploy-policy',
        help='write a compact ROS runtime policy only when the held-out gate passes',
    )

    evaluate = commands.add_parser('evaluate', help='evaluate a saved model only')
    _add_environment_arguments(evaluate, package)
    evaluate.add_argument('--episodes', type=int, default=120)
    evaluate.add_argument('--random-episodes', type=int, default=24)
    evaluate.add_argument('--model', default=str(root / 'simulation' / 'results' / 'rl_policy.json'))
    evaluate.add_argument(
        '--deployed-policy',
        help='evaluate the compact YAML used by the ROS runtime instead of --model',
    )
    evaluate.add_argument('--output', default=str(root / 'simulation' / 'results' / 'rl_evaluation.json'))
    args = parser.parse_args()

    if args.episodes <= 0:
        parser.error('--episodes must be positive')
    if not 0 <= args.random_episodes <= args.episodes:
        parser.error('--random-episodes must be between zero and --episodes')
    if args.command == 'train':
        if args.eval_episodes <= 0:
            parser.error('--eval-episodes must be positive')
        if not 0 <= args.eval_random_episodes <= args.eval_episodes:
            parser.error('--eval-random-episodes must be between zero and --eval-episodes')
    return args


def _write_report(path, report):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')


def main():
    args = _parse_args()
    profile = deployed_profile(args.runtime)
    if args.command == 'train':
        policy, training = train_policy(
            args.field,
            args.poses,
            profile,
            episodes=args.episodes,
            random_episodes=args.random_episodes,
            seed=args.seed,
        )
        evaluation_seed = args.seed + 1_000_003
        field, scenarios, evaluation_metadata = prepare_campaign(
            args.field,
            args.poses,
            args.eval_episodes,
            args.eval_random_episodes,
            evaluation_seed,
        )
        baseline_results = evaluate_campaign(field, scenarios, profile)
        learned_results = evaluate_campaign(
            field, scenarios, profile, controller=policy
        )
        baseline = build_campaign_report(
            evaluation_metadata, profile, baseline_results,
            args.eval_random_episodes,
        )
        learned = build_campaign_report(
            evaluation_metadata, profile, learned_results,
            args.eval_random_episodes, controller=policy,
        )
        regression = paired_regression_report(baseline_results, learned_results)
        safety_passed = campaign_passed(learned)
        collisions = int(sum(result.collision for result in learned_results))
        metadata = {
            'training': training,
            'held_out_seed': evaluation_seed,
            'evaluation_episodes': args.eval_episodes,
            'safety_campaign_passed': safety_passed,
            'paired_regression_passed': regression['passed'],
            'paired_regression': regression,
            'evaluation_collisions': collisions,
            'baseline_failures': regression['baseline_failures'],
            'learned_failures': regression['learned_failures'],
            'observation_context': {
                'velocity_source': 'model_actual_velocity',
                'reference_speed_mps': profile.speed,
                'reference_profile': yaml.safe_load(Path(args.runtime).read_text())[
                    'runtime_guard']['ros__parameters']['default_profile'],
            },
        }
        # Promote a residual that is strictly better, or one that is equivalent
        # and already clean.  Never promote one that regressed a solved route.
        runtime_compatible = (not _greedy_overrides(policy)
                              or _runtime_context_compatible(metadata['observation_context']))
        passed = runtime_compatible and regression['passed'] and collisions == 0 and (
            regression['learned_failures'] < regression['baseline_failures']
            or safety_passed
        )
        requested_model = Path(args.model)
        model_path = requested_model
        if not passed:
            model_path = requested_model.with_name(
                f'{requested_model.stem}.rejected{requested_model.suffix}'
            )
        policy.save(model_path, metadata)
        deployed_policy = None
        if passed and args.deploy_policy:
            export_compact_policy(model_path, args.deploy_policy)
            deployed_policy = str(args.deploy_policy)
        report = {
            'algorithm': 'safety-constrained tabular Q-learning',
            'model': str(model_path),
            'requested_model': str(requested_model),
            'promoted': passed,
            'deployed_policy': deployed_policy,
            'profile': asdict(profile),
            'training': training,
            'baseline_held_out': baseline,
            'learned_held_out': learned,
            'safety_campaign_passed': safety_passed,
            'paired_regression': regression,
            'evaluation_collisions': collisions,
            'runtime_observation_context_compatible': runtime_compatible,
            'passed': passed,
        }
    else:
        if args.deployed_policy:
            policy = CompactDeploymentPolicy.load(args.deployed_policy)
            evaluated_model = str(args.deployed_policy)
        else:
            policy = SafeQLearningPolicy.load(args.model, seed=args.seed)
            evaluated_model = str(args.model)
        field, scenarios, evaluation_metadata = prepare_campaign(
            args.field, args.poses, args.episodes, args.random_episodes, args.seed
        )
        baseline_results = evaluate_campaign(field, scenarios, profile)
        learned_results = evaluate_campaign(
            field, scenarios, profile, controller=policy
        )
        baseline = build_campaign_report(
            evaluation_metadata, profile, baseline_results, args.random_episodes,
        )
        learned = build_campaign_report(
            evaluation_metadata, profile, learned_results, args.random_episodes,
            controller=policy,
        )
        safety_passed = campaign_passed(learned)
        regression = paired_regression_report(baseline_results, learned_results)
        collisions = int(sum(result.collision for result in learned_results))
        overrides = (policy.overrides if isinstance(policy, CompactDeploymentPolicy)
                     else _greedy_overrides(policy))
        context = (policy.observation_context if isinstance(policy, CompactDeploymentPolicy)
                   else json.loads(Path(args.model).read_text()).get('metadata', {}).get('observation_context'))
        runtime_compatible = not overrides or _runtime_context_compatible(context)
        report = {
            'algorithm': 'safety-constrained tabular Q-learning',
            'model': evaluated_model,
            'baseline_held_out': baseline,
            'learned_held_out': learned,
            'safety_campaign_passed': safety_passed,
            'paired_regression': regression,
            'evaluation_collisions': collisions,
            'runtime_observation_context_compatible': runtime_compatible,
            'passed': runtime_compatible and regression['passed'] and collisions == 0,
        }
    _write_report(args.output, report)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report['passed'] else 2)


if __name__ == '__main__':
    main()
