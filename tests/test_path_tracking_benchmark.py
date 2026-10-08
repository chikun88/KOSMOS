"""Independent replay measurements must not hide tracking or settling failures."""
import importlib.util
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/benchmark_path_tracking.py'
SPEC = importlib.util.spec_from_file_location('path_tracking_benchmark', SCRIPT)
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


def test_cross_track_uses_segments_instead_of_vertex_or_goal_distance():
    points = np.array([[0., 0.], [2., 0.], [2., 2.]])
    distance, progress = benchmark.polyline_projection(points, [[1., .03], [2.04, 1.5], [3., 3.]])
    assert distance == pytest.approx([.03, .04, 2.**.5])
    assert progress == pytest.approx([1., 3.5, 4.])


def test_duplicate_segments_and_single_pose_are_measurable():
    for path in ([[1., 2.]], [[1., 2.], [1., 2.]]):
        distance, progress = benchmark.polyline_projection(path, [[1., 2.5]])
        assert distance == pytest.approx([.5])
        assert progress == pytest.approx([0.])


@pytest.mark.parametrize('points,positions', [([], [[0., 0.]]),
    ([[0., 0.], [float('nan'), 1.]], [[0., 0.]]), ([[0., 0.]], [[float('inf'), 0.]])])
def test_nonfinite_or_empty_geometry_never_yields_success(points, positions):
    with pytest.raises(ValueError):
        benchmark.polyline_projection(points, positions)


def test_last_complete_settling_window_counts_but_a_transient_crossing_does_not():
    times = np.arange(101)*.01
    positions = np.zeros((101, 3))
    velocities = np.zeros_like(positions)
    positions[:50, 0] = .1
    assert benchmark.sustained_arrival(times, positions, velocities, np.zeros(3)) == pytest.approx(.5)
    positions[-1, 0] = .016
    assert benchmark.sustained_arrival(times, positions, velocities, np.zeros(3)) is None


def test_pose_error_does_not_hide_motion_or_yaw_and_wrap_is_respected():
    times = np.arange(101)*.01
    positions = np.zeros((101, 3))
    velocities = np.zeros_like(positions)
    velocities[:, 0] = .026
    assert benchmark.sustained_arrival(times, positions, velocities, np.zeros(3)) is None
    velocities[:] = 0.
    positions[:, 2] = 2.*np.pi
    assert benchmark.sustained_arrival(times, positions, velocities, np.zeros(3)) == 0.
    positions[:, 2] += .016
    assert benchmark.sustained_arrival(times, positions, velocities, np.zeros(3)) is None


def test_loader_never_adds_ros_stubs_or_initializes_a_node():
    code = '''
import sys
sys.path.insert(0, sys.argv[1])
import benchmark_path_tracking as benchmark
before = {k: v for k, v in sys.modules.items() if k.startswith(('rclpy', 'geometry_msgs', 'nav_msgs'))}
module, clock, defaults, bridge = benchmark.load_controller(benchmark.ROOT)
after = {k: v for k, v in sys.modules.items() if k.startswith(('rclpy', 'geometry_msgs', 'nav_msgs'))}
assert before == after
assert module.Trajectory is not None
assert defaults['control_rate_hz'] >= 20
'''
    subprocess.run([sys.executable, '-c', code, str(SCRIPT.parent)], check=True)


def test_plant_truth_matches_independent_first_order_straight_solution():
    pose, velocity = np.zeros(3), np.zeros(3)
    target, tau = np.array([3.5, 0., 0.]), .12
    for _ in range(100):
        pose, velocity = benchmark.integrate_plant(pose, velocity, target, .01, tau)
    assert velocity[0] == pytest.approx(3.5*(1.-np.exp(-1./tau)), abs=1.e-12)
    assert pose[0] == pytest.approx(3.5*(1.-tau*(1.-np.exp(-1./tau))), abs=1.e-8)
    assert pose[1:] == pytest.approx([0., 0.])


def test_plant_truth_rotating_body_twist_matches_exact_circle():
    pose = np.zeros(3)
    velocity = np.array([3.5, 0., 1.3])
    for _ in range(100):
        pose, velocity = benchmark.integrate_plant(pose, velocity, velocity, .01, .12)
    expected = [3.5*np.sin(1.3)/1.3, 3.5*(1.-np.cos(1.3))/1.3, 1.3]
    assert pose == pytest.approx(expected, abs=1.e-10)


def test_delayed_production_controller_stays_on_curved_routes(tmp_path):
    """Old code arrives eventually but misses these fixed routes by about 1 m."""
    import json
    output = tmp_path / 'curved-controller.json'
    subprocess.run([sys.executable, str(SCRIPT), '--output', str(output),
                    '--geometry', 'none', '--paths', 'quarter_circle_r3', 's_curve_8m',
                    '--delays', '.2', '--duration', '16', '--require-arrival'],
                   check=True, capture_output=True, text=True)
    report = json.loads(output.read_text())
    assert report['all_arrived']
    assert len(report['cases']) == 2
    for case in report['cases']:
        result = case['result']
        assert result['cross_track_peak_m'] < .10, (case['path'], result)
        assert result['tail_1s_goal_error_max_m'] < .015
        assert result['uart_peak_units'] <= 10000
        assert not result['controller_failures_before_arrival']
