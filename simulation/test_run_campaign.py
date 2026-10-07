from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from simulation.dynamics import SimProfile
from simulation.run_campaign import campaign_passed, deployed_profile
from simulation.run_campaign import model_source_fingerprints


RUNTIME = (
    Path(__file__).resolve().parents[1]
    / 'ros2_ws' / 'src' / 'omni_autonomy_next' / 'config' / 'runtime.yaml'
)


def test_campaign_source_evidence_includes_the_runtime_shared_gradient():
    import hashlib
    sources = model_source_fingerprints()
    helper = 'ros2_ws/src/omni_autonomy_next/omni_autonomy_next/footprint_gradient.py'
    root = RUNTIME.parents[4]
    assert sources[helper] == hashlib.sha256((root / helper).read_bytes()).hexdigest()
    assert 'simulation/footprint_gradient.py' in sources
    assert 'simulation/dynamics.py' in sources


def passing_report():
    summary = {
        'episodes': 10,
        'successes': 10,
        'collisions': 0,
        'timeouts': 0,
        'progress_aborts': 0,
        'p95_position_error_m': 0.04,
        'p95_yaw_error_deg': 2.0,
        'minimum_clearance_m': 0.01,
    }
    return {
        'summary': {**deepcopy(summary), 'episodes': 20, 'successes': 20},
        'critical_routes_summary': deepcopy(summary),
        'random_same_side_summary': deepcopy(summary),
    }


def test_campaign_gate_accepts_only_complete_success():
    assert campaign_passed(passing_report())


def test_campaign_gate_rejects_a_single_timeout():
    report = passing_report()
    report['random_same_side_summary']['successes'] = 9
    report['random_same_side_summary']['timeouts'] = 1
    assert not campaign_passed(report)


def test_campaign_gate_checks_each_route_class():
    report = passing_report()
    report['critical_routes_summary']['p95_position_error_m'] = 0.040001
    assert not campaign_passed(report)


def test_campaign_gate_rejects_a_progress_abort():
    """A goal the controller gives up on is the field failure being modelled."""
    report = passing_report()
    report['critical_routes_summary']['successes'] = 9
    report['critical_routes_summary']['progress_aborts'] = 1
    assert not campaign_passed(report)


def test_campaign_gate_allows_an_explicitly_empty_route_class():
    report = passing_report()
    report['summary']['episodes'] = 10
    report['summary']['successes'] = 10
    report['random_same_side_summary'] = {
        'episodes': 0, 'successes': 0, 'collisions': 0, 'timeouts': 0,
        'progress_aborts': 0,
        'p95_position_error_m': None, 'p95_yaw_error_deg': None,
        'minimum_clearance_m': None,
    }
    assert campaign_passed(report)


def test_gate_measures_the_deployed_profile_not_model_defaults():
    """The gate must move at the speeds the deployed guard will allow.

    Configured speeds can change independently of this model's illustrative
    defaults. Every campaign must read the actual runtime limits.
    """
    profile = deployed_profile(RUNTIME)
    tuned = SimProfile()
    import json
    runtime = yaml.safe_load(RUNTIME.read_text(encoding='utf-8'))[
        'runtime_guard']['ros__parameters']
    deployed = json.loads(runtime['profiles_json'])[runtime['default_profile']]
    scale = runtime['default_speed_scale']
    assert profile.speed == pytest.approx(deployed['linear'] * scale)
    assert profile.lateral_speed == pytest.approx(deployed['lateral'] * scale)
    assert profile.angular_speed == pytest.approx(deployed['angular'] * scale)
    # Acceleration is the shaping stage's, not the guard's headroom. The guard
    # is deliberately allowed to accelerate harder than velocity_smoother ever
    # asks, so that it stops re-limiting an already-limited command; taking its
    # value here would certify a robot that accelerates half again as hard as
    # the deployed chain commands.
    smoother = yaml.safe_load(
        (RUNTIME.parent / 'nav2_next.yaml').read_text(encoding='utf-8')
    )['velocity_smoother']['ros__parameters']
    scale = yaml.safe_load(RUNTIME.read_text(encoding='utf-8'))[
        'runtime_guard']['ros__parameters']['default_speed_scale']
    assert profile.acceleration == pytest.approx(
        float(smoother['max_accel'][0]) * scale
    )
    assert profile.angular_acceleration == pytest.approx(
        float(smoother['max_accel'][2]) * scale
    )
    # The offline tracking gains belong to this model, not to the robot config.
    assert profile.lookahead == tuned.lookahead
    assert profile.position_gain == tuned.position_gain
    assert profile.yaw_gain == tuned.yaw_gain


def test_gate_reads_arrival_and_collision_gates_from_the_deployed_nav2_config():
    """The gates must be the deployed ones, not this model's own opinion.

    An offline success test that is looser than nav2's goal checker, or that
    ignores the progress checker and Collision Monitor, reports success on
    exactly the routes that fail on the field.
    """
    nav2 = yaml.safe_load(
        (RUNTIME.parent / 'nav2_next.yaml').read_text(encoding='utf-8')
    )
    controller = nav2['controller_server']['ros__parameters']
    monitor = nav2['collision_monitor']['ros__parameters']
    profile = deployed_profile(RUNTIME)
    assert profile.xy_goal_tolerance == pytest.approx(
        controller['goal_checker']['xy_goal_tolerance']
    )
    assert profile.yaw_goal_tolerance == pytest.approx(
        controller['goal_checker']['yaw_goal_tolerance']
    )
    assert profile.progress_radius == pytest.approx(
        controller['progress_checker']['required_movement_radius']
    )
    assert profile.progress_time_allowance == pytest.approx(
        controller['progress_checker']['movement_time_allowance']
    )
    assert profile.approach_horizon_sec == pytest.approx(
        monitor['FootprintApproach']['time_before_collision']
    )
    assert profile.slowdown_ratio == pytest.approx(
        monitor['SlowZone']['slowdown_ratio']
    )


def test_deployed_profile_requires_the_nav2_configuration(tmp_path):
    runtime = tmp_path / 'runtime.yaml'
    runtime.write_text(RUNTIME.read_text(encoding='utf-8'), encoding='utf-8')
    with pytest.raises(FileNotFoundError):
        deployed_profile(runtime)


def test_deployed_profile_rejects_an_unusable_speed_scale(tmp_path):
    broken = tmp_path / 'runtime.yaml'
    broken.write_text(
        RUNTIME.read_text(encoding='utf-8').replace(
            'default_speed_scale: 1.0', 'default_speed_scale: 0.0'
        ),
        encoding='utf-8',
    )
    with pytest.raises(ValueError):
        deployed_profile(broken)
