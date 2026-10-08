#!/usr/bin/env python3
"""Compare CAD geometry work with a fixed unmodified git revision.

Checks 10,000 pose/cap combinations for exact distance equivalence before
measuring work on stored failure contexts and a synthetic 3.5 m/s envelope.
This measures computation, not physical speed or route acceptance.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ros2_ws/src/omni_autonomy_next'))
from omni_autonomy_next.execution_clearance import braking_pose_path
from omni_autonomy_next.rl_residual import CadClearanceModel

SOURCE_PATH = 'ros2_ws/src/omni_autonomy_next/omni_autonomy_next/rl_residual.py'


def recorded_contexts(document):
    cases = []

    def walk(value):
        if isinstance(value, dict):
            context = value.get('execution_certificate_context')
            if (context and context.get('pose') and context.get('raw_twist')
                    and context.get('initial_delay_sec') is not None):
                cases.append((context['pose'], context['raw_twist'],
                              context['initial_delay_sec']))
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(document)
    unique = {tuple(np.round(np.r_[pose, command, delay], 4)):
              (pose, command, delay) for pose, command, delay in cases}
    return list(unique.values())[::max(1, len(unique)//30)][:30]


def compare_distances(models):
    rng = np.random.default_rng(260108)
    points = rng.uniform([-5., -2.], [1., 5.], (2500, 2))
    yaws = rng.uniform(-np.pi, np.pi, len(points))
    for cap in (.01, .06, .35, 2.):
        before = models[0].clearance_over_poses(points, yaws, cap)
        after = models[1].clearance_over_poses(points, yaws, cap)
        if not np.array_equal(before, after):
            raise AssertionError(f'batch clearance changed with cap {cap}')
        for point, yaw in zip(points[::10], yaws[::10]):
            if (models[0].body_clearance(point, yaw, cap)
                    != models[1].body_clearance(point, yaw, cap)):
                raise AssertionError(f'scalar clearance changed with cap {cap}')


def measure(models, label, cases):
    result = dict(label=label, inputs=cases, samples=sum(
        len(braking_pose_path(pose, command, delay, .85, 1.2,
                             models[0].radius)[1])
        for pose, command, delay in cases))
    outputs = []
    for name, model in zip(('before', 'after'), models):
        timings = []
        for _ in range(8):
            begin, cpu = time.perf_counter(), time.process_time()
            values = []
            for pose, command, delay in cases:
                points, yaws, _, _ = braking_pose_path(
                    pose, command, delay, .85, 1.2, model.radius)
                values.extend(model.clearance_over_poses(points, yaws, .15))
            timings.append(dict(wall_ms=(time.perf_counter()-begin)*1000,
                                cpu_ms=(time.process_time()-cpu)*1000))
        result[name] = dict(
            runs=timings,
            median_wall_ms=statistics.median(row['wall_ms'] for row in timings),
            median_cpu_ms=statistics.median(row['cpu_ms'] for row in timings))
        outputs.append(values)
    if not np.array_equal(*outputs):
        raise AssertionError(f'clearance changed in benchmark {label}')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-ref', required=True,
                        help='Unmodified git revision to compare')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    baseline_ref = subprocess.check_output(
        ['git', 'rev-parse', '--verify', args.baseline_ref], cwd=ROOT,
        text=True).strip()
    baseline = subprocess.check_output(
        ['git', 'show', baseline_ref+':'+SOURCE_PATH], cwd=ROOT)
    with tempfile.TemporaryDirectory(prefix='kosmos-geometry-') as directory:
        original = Path(directory) / 'rl_residual.py'
        original.write_bytes(baseline)
        spec = importlib.util.spec_from_file_location(
            'omni_autonomy_next._original', original)
        old = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old)
    config = ROOT / 'ros2_ws/src/omni_autonomy_next/config'
    models = [cls.from_yaml(config/'field_planning.yaml',
                           config/'competition_footprints.yaml')
              for cls in (old.CadClearanceModel, CadClearanceModel)]
    contexts = recorded_contexts(json.loads((ROOT / 'docs' /
        'SYSTEM_AUDIT_CRITICAL_PROOF_CONTEXT_20261007.json').read_text()))
    if not contexts:
        raise ValueError('recorded certificate contexts are missing')
    compare_distances(models)
    report = dict(
        classification='Offline geometry benchmark; not physical or ROS timing acceptance',
        random_equality_poses=10000,
        captured_context_count=len(contexts),
        cases=[measure(models, 'recorded_contexts', contexts),
               measure(models, '3.5_mps_braking_samples',
                       [([-3., 0., .2], [3.5, 0., .3], .32)])],
        baseline_git_commit=baseline_ref,
        baseline_source_sha256=hashlib.sha256(baseline).hexdigest(),
        optimized_source_sha256=hashlib.sha256((ROOT/SOURCE_PATH).read_bytes()).hexdigest(),
        benchmark_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        qualification=(
            'Eight serial measurements per variant on this host. The synthetic '
            '3.5 m/s work is not a collision-free field route. This does not prove '
            'control-loop WCET, route acceptance, or attainable physical speed.'))
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
