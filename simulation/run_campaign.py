from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import hashlib
import math
from pathlib import Path
import statistics

import numpy as np
import yaml

try:
    from .dynamics import SimProfile, simulate_episode
    from .field_model import GridField
except ImportError:
    from dynamics import SimProfile, simulate_episode
    from field_model import GridField


def deployed_profile(runtime_file, base=None, nav2_file=None):
    """Read deployed limits for the MPPI-style offline stand-in.

    The gate is only meaningful if it measures the speeds the robot uses.  A
    hardcoded profile here silently stopped describing the robot the moment
    ``default_speed_scale`` was lowered, so the limits are read from
    ``config/runtime.yaml`` instead.  The arrival tolerances, progress
    allowance and Collision Monitor behaviour are read from
    ``config/nav2_next.yaml`` for the same reason: the offline model previously
    used a disc robot and its own success test, and so reported full success on
    exactly the routes that fail on the field.  The tracking gains stay with
    :class:`SimProfile` because they belong to this offline model, not to the
    deployed configuration.
    """
    with Path(runtime_file).open(encoding='utf-8') as stream:
        runtime = yaml.safe_load(stream)['runtime_guard']['ros__parameters']
    profiles = json.loads(runtime['profiles_json'])
    name = runtime['default_profile']
    if name not in profiles:
        raise ValueError(f'runtime default_profile {name!r} is not defined')
    limits = profiles[name]
    scale = float(runtime['default_speed_scale'])
    if not 0.0 < scale <= 1.0:
        raise ValueError('default_speed_scale must be within (0, 1]')
    speed_limits = {
        'speed': float(limits['linear']) * scale,
        'lateral_speed': float(limits['lateral']) * scale,
        'angular_speed': float(limits['angular']) * scale,
    }
    if any(not math.isfinite(value) or value <= 0.0
           for value in speed_limits.values()):
        raise ValueError('deployed speed limits must be finite and positive')
    profile = replace(
        base if base is not None else SimProfile(),
        **speed_limits,
    )
    if nav2_file is None:
        nav2_file = Path(runtime_file).with_name('nav2_next.yaml')
    if not Path(nav2_file).exists():
        raise FileNotFoundError(f'nav2 configuration {nav2_file} is required')
    # Acceleration comes from velocity_smoother, not from the guard profile.
    # The guard's acceleration and jerk are deliberately held above the
    # shaping stage so it stops re-limiting an already-limited command and
    # stops acting as a lag element inside the control loop; velocity_smoother
    # is therefore the stage that decides how fast the base actually changes
    # speed.  Reading the guard's headroom here would certify a robot that
    # accelerates half again as hard as the deployed chain ever commands.
    return replace(
        profile,
        **_deployed_acceleration(nav2_file, scale),
        **_deployed_gates(nav2_file),
    )


def _deployed_acceleration(nav2_file, scale):
    with Path(nav2_file).open(encoding='utf-8') as stream:
        smoother = yaml.safe_load(stream)['velocity_smoother']['ros__parameters']
    max_accel = smoother['max_accel']
    result = {
        'acceleration': float(max_accel[0]) * scale,
        'angular_acceleration': float(max_accel[2]) * scale,
    }
    if any(not math.isfinite(value) or value <= 0.0 for value in result.values()):
        raise ValueError('deployed acceleration limits must be finite and positive')
    return result


def _deployed_gates(nav2_file):
    with Path(nav2_file).open(encoding='utf-8') as stream:
        nav2 = yaml.safe_load(stream)
    controller = nav2['controller_server']['ros__parameters']
    goal_checker = controller['goal_checker']
    progress_checker = controller['progress_checker']
    monitor = nav2['collision_monitor']['ros__parameters']
    approach = monitor['FootprintApproach']
    gates = {
        'xy_goal_tolerance': float(goal_checker['xy_goal_tolerance']),
        'yaw_goal_tolerance': float(goal_checker['yaw_goal_tolerance']),
        'progress_radius': float(progress_checker['required_movement_radius']),
        'progress_angle': float(progress_checker['required_movement_angle']),
        'progress_time_allowance': float(
            progress_checker['movement_time_allowance']
        ),
        'approach_horizon_sec': float(approach['time_before_collision']),
        'approach_step_sec': float(approach['simulation_time_step']),
    }
    slow_zone = monitor.get('SlowZone')
    if slow_zone is not None and slow_zone.get('action_type') == 'slowdown':
        gates['slowdown_ratio'] = float(slow_zone['slowdown_ratio'])
    for name, value in gates.items():
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f'deployed gate {name} must be finite and positive')
    return gates


def _load_poses(path):
    with Path(path).open(encoding='utf-8') as stream:
        data = yaml.safe_load(stream)
    poses = {
        str(key): np.asarray([value['x'], value['y'], value['yaw']], dtype=float)
        for key, value in data['poses'].items() if value.get('configured', False)
    }
    if not poses or any(not np.isfinite(pose).all() for pose in poses.values()):
        raise ValueError('campaign requires finite configured poses')
    return poses


def summarize(results):
    successful = [result for result in results if result.success]
    return {
        'episodes': len(results),
        'successes': len(successful),
        'success_rate': len(successful) / max(1, len(results)),
        'collisions': sum(result.collision for result in results),
        'timeouts': sum(result.timeout for result in results),
        'progress_aborts': sum(result.aborted for result in results),
        'p95_position_error_m': (
            float(np.percentile([r.final_position_error for r in successful], 95))
            if successful else None
        ),
        'p95_yaw_error_deg': (
            float(np.degrees(np.percentile([r.final_yaw_error for r in successful], 95)))
            if successful else None
        ),
        'mean_time_sec': (
            statistics.fmean(r.elapsed for r in successful) if successful else None
        ),
        'minimum_clearance_m': (
            min(r.minimum_clearance for r in successful) if successful else None
        ),
        'mean_braked_fraction': (
            statistics.fmean(r.braked_fraction for r in results) if results else None
        ),
        'mean_command_reversals': (
            statistics.fmean(r.command_reversals for r in results) if results else None
        ),
    }


def prepare_campaign(
    field_file, poses_file, episodes, random_episodes, seed, footprint_file=None
):
    if isinstance(episodes, bool) or not isinstance(episodes, (int, np.integer)) or episodes <= 0:
        raise ValueError('episodes must be a positive integer')
    if (isinstance(random_episodes, bool)
            or not isinstance(random_episodes, (int, np.integer))
            or not 0 <= random_episodes <= episodes):
        raise ValueError('random_episodes must be between zero and episodes')
    rng = np.random.default_rng(seed)
    field = GridField.from_yaml(field_file, footprint_file)
    poses = _load_poses(poses_file)
    side_sign = int(np.sign(np.median([pose[0] for pose in poses.values()])))
    critical = [key for key in ['0', '2', '3', '4', '5', '6', '7'] if key in poses]
    starts = ['1'] if '1' in poses else [next(iter(poses))]
    named_pairs = []
    for start in starts + critical:
        for goal in critical:
            if start != goal:
                named_pairs.append((start, goal))
    path_cache = {}
    scenarios = []
    named_count = max(0, episodes - random_episodes)
    if named_count and not named_pairs:
        raise ValueError('campaign requires distinct configured named route poses')
    for i in range(named_count):
        start_key, goal_key = named_pairs[i % len(named_pairs)]
        cache_key = (start_key, goal_key)
        if cache_key not in path_cache:
            # The route is planned for the footprint held at the goal yaw,
            # which is the orientation the robot holds for all but the first
            # rotation and, critically, at the tight firing pose itself.
            path_cache[cache_key] = field.plan(
                poses[start_key][:2], poses[goal_key][:2],
                float(poses[goal_key][2]),
            )
        scenarios.append((
            path_cache[cache_key], float(poses[start_key][2]),
            float(poses[goal_key][2]), int(rng.integers(0, 2**31 - 1)),
        ))
    for _ in range(random_episodes):
        path = None
        for _attempt in range(100):
            goal_yaw = float(rng.uniform(-np.pi, np.pi))
            start_yaw = float(rng.uniform(-np.pi, np.pi))
            start = field.random_free_point(rng, goal_yaw, side_sign=side_sign)
            goal = field.random_free_point(rng, goal_yaw, side_sign=side_sign)
            # A start free at the travel yaw may collide at the independently
            # sampled initial yaw. Such a scenario is invalid before motion.
            if (field.body_clearance(start, start_yaw) < field.planning_margin
                    or field.body_clearance(goal, goal_yaw) < field.planning_margin):
                continue
            path = field.plan(start, goal, goal_yaw)
            if path is not None:
                break
        if path is None:
            raise RuntimeError('failed to sample a connected same-side field route')
        scenarios.append((
            path, start_yaw, goal_yaw,
            int(rng.integers(0, 2**31 - 1)),
        ))
    metadata = {
        'seed': seed, 'critical_pose_ids': critical,
        'named_path_count': len(path_cache),
        'planning_margin_m': field.planning_margin,
        'footprint_circumscribed_radius_m': field.body.radius,
        'footprint_inscribed_radius_m': field.body.inscribed_radius,
        'source_sha256': {
            str(Path(path).resolve()): hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for path in (field_file, poses_file,
                         footprint_file or Path(field_file).with_name('competition_footprints.yaml'))
        },
        # Which goal-number pair each named scenario is, so a failing gate can
        # name the route instead of only counting it.
        'named_routes': [
            f'{named_pairs[i % len(named_pairs)][0]}->'
            f'{named_pairs[i % len(named_pairs)][1]}'
            for i in range(named_count)
        ],
    }
    return field, scenarios, metadata


def evaluate_campaign(field, scenarios, profile, controller=None):
    return [
        simulate_episode(
            field, path, start_yaw, goal_yaw, profile,
            np.random.default_rng(scenario_seed),
            controller=controller,
        )
        for path, start_yaw, goal_yaw, scenario_seed in scenarios
    ]


def failures_by_route(named_routes, results):
    """Attribute every non-arrival to the goal-number pair that produced it."""
    tally = {}
    for route, result in zip(named_routes, results):
        attempts, failures, reasons = tally.get(route, (0, 0, []))
        attempts += 1
        if not result.success:
            failures += 1
            reasons.append(
                'collision' if result.collision
                else 'progress_abort' if result.aborted
                else 'timeout' if result.timeout
                else 'unknown'
            )
        tally[route] = (attempts, failures, reasons)
    return {
        route: {
            'attempts': attempts,
            'failures': failures,
            'reasons': sorted(set(reasons)),
        }
        for route, (attempts, failures, reasons) in sorted(tally.items())
        if failures
    }


def build_campaign_report(metadata, profile, results, random_episodes, controller=None):
    named_count = max(0, len(results) - random_episodes)
    named_routes = metadata.get('named_routes', [])
    return {
        **{key: value for key, value in metadata.items() if key != 'named_routes'},
        'profile': asdict(profile),
        'model': {
            'version': 3,
            'kind': 'MPPI-style lightweight offline stand-in',
            'physical_acceptance_evaluated': False,
            'limitations': [
                'Not a replay of the deployed trajectory tracker or live Nav2 stack.',
                'Acceleration uses the MPPI velocity smoother, not tracker profile acceleration.',
                'CAD contact is model overlap, not an observed physical collision.',
                'No sensor failures, moving obstacles, load identification or hardware I/O.',
                'Projected intervals are conservatively certified; dynamics contact is sampled per step.',
            ],
        },
        'controller': (
            'baseline' if controller is None else controller.__class__.__name__
        ),
        'summary': summarize(results),
        'critical_routes_summary': summarize(results[:named_count]),
        'random_same_side_summary': summarize(results[named_count:]),
        'failing_routes': failures_by_route(
            named_routes, results[:named_count]
        ),
    }


def run_campaign(
    field_file,
    poses_file,
    episodes,
    random_episodes,
    seed,
    profile,
    controller=None,
):
    field, scenarios, metadata = prepare_campaign(
        field_file, poses_file, episodes, random_episodes, seed
    )
    results = evaluate_campaign(field, scenarios, profile, controller=controller)
    return build_campaign_report(
        metadata, profile, results, random_episodes, controller=controller
    )


def campaign_passed(report):
    """Require nonempty, finite offline evidence with every scenario solved.

    Passing this surrogate gate does not establish physical acceptance.
    """
    try:
        total = report['summary']['episodes']
        named = report['critical_routes_summary']['episodes']
        random = report['random_same_side_summary']['episodes']
        if any(isinstance(count, bool) or not isinstance(count, int) or count < 0
               for count in (total, named, random)):
            return False
        if total <= 0 or named + random != total:
            return False
        profile = report.get('profile', {})
        position_tolerance = float(profile.get('xy_goal_tolerance', 0.04))
        yaw_tolerance_deg = math.degrees(float(profile.get('yaw_goal_tolerance', math.radians(2.0))))
        if (not math.isfinite(position_tolerance) or position_tolerance <= 0.0
                or not math.isfinite(yaw_tolerance_deg) or yaw_tolerance_deg <= 0.0):
            return False
    except (KeyError, TypeError, ValueError):
        return False
    for key in ('summary', 'critical_routes_summary', 'random_same_side_summary'):
        summary = report[key]
        if summary['episodes'] == 0:
            continue
        try:
            position_error = float(summary['p95_position_error_m'])
            yaw_error = float(summary['p95_yaw_error_deg'])
            clearance = float(summary['minimum_clearance_m'])
            if (
                summary['successes'] != summary['episodes']
                or summary['collisions'] != 0
                or summary['timeouts'] != 0
                or summary['progress_aborts'] != 0
                or not all(math.isfinite(value) for value in (position_error, yaw_error, clearance))
                or not 0.0 <= position_error <= position_tolerance
                or not 0.0 <= yaw_error <= yaw_tolerance_deg
                or clearance <= 0.0
            ):
                return False
        except (KeyError, TypeError, ValueError):
            return False
    return True


def main():
    root = Path(__file__).resolve().parents[1]
    package = root / 'ros2_ws' / 'src' / 'omni_autonomy_next'
    parser = argparse.ArgumentParser()
    parser.add_argument('--field', default=str(package / 'config' / 'field_planning.yaml'))
    parser.add_argument('--poses', default=str(package / 'config' / 'field_poses.yaml'))
    parser.add_argument('--runtime', default=str(package / 'config' / 'runtime.yaml'))
    parser.add_argument('--episodes', type=int, default=1200)
    parser.add_argument('--random-episodes', type=int, default=120)
    parser.add_argument('--seed', type=int, default=20260801)
    parser.add_argument('--output', default=str(root / 'simulation' / 'results' / 'campaign.json'))
    args = parser.parse_args()
    report = run_campaign(
        args.field, args.poses, args.episodes, args.random_episodes, args.seed,
        deployed_profile(args.runtime),
    )
    report['profile_source'] = args.runtime
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if campaign_passed(report) else 2)


if __name__ == '__main__':
    main()
