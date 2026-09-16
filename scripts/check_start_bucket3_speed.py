#!/usr/bin/env python3
"""Offline start-to-bucket-3 candidate comparison; no ROS publishers or motor I/O."""
import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import check_bucket_gate_replay as replay_model
from omni_autonomy_next.staged_heading import dense_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', type=Path, default=replay_model.ROOT /
                        'docs/start_bucket3_speed_candidates_20260916.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.fixture.resolve() == args.output.resolve():
        parser.error('output must differ from the recorded fixture')
    fixture = json.loads(args.fixture.read_text())
    base = fixture['baseline_case']
    original_tuning = replay_model.calibrated_tracking_parameters
    source_paths = [replay_model.ROOT / p for p in fixture['sources']]
    source_paths += [Path(__file__).resolve(), args.fixture.resolve()]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    results = []
    for recorded in fixture['results']:
        gain = recorded.get('gain', 1.6)
        gates = [list(p) for p in base['gates']]
        if recorded['candidate'] == 'middle_gate_y_0':
            gates[2][1] = 0.
        elif 'y' in recorded:
            gates[1][1] = recorded['y']
        vertices = [base['points'][0]] + gates + [base['goal'][:2]]
        case = dict(base, points=dense_path(vertices, .025).tolist(), gates=gates)

        def tuning(*a, **kw):
            return dict(original_tuning(*a, **kw), position_gain=gain)

        with patch.object(replay_model, 'calibrated_tracking_parameters', tuning):
            result = replay_model.replay(case, True, recorded['delay'],
                                         recorded['tau'], mode='fast')
        row = dict(candidate=recorded['candidate'], delay=recorded['delay'],
                   tau=recorded['tau'], **result)
        results.append(row)
        print(json.dumps(row), flush=True)
    for p in source_paths:
        if hashlib.sha256(p.read_bytes()).hexdigest() != hashes[str(p)]:
            raise RuntimeError(f'source changed during replay: {p}')
    args.output.write_text(json.dumps(dict(
        applied=False, physical_improvement_verified=False,
        limitations=fixture['limitations'], sources=hashes, results=results),
        indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
