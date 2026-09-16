from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np

try:
    from .dynamics import SimProfile
    from .run_campaign import evaluate_campaign, prepare_campaign, summarize
except ImportError:
    from dynamics import SimProfile
    from run_campaign import evaluate_campaign, prepare_campaign, summarize


def main():
    root = Path(__file__).resolve().parents[1]
    package = root / 'ros2_ws' / 'src' / 'omni_autonomy_next'
    field = package / 'config' / 'field_planning.yaml'
    poses = package / 'config' / 'field_poses.yaml'
    field_model, scenarios, _ = prepare_campaign(
        field, poses, episodes=120, random_episodes=12, seed=20260731
    )
    candidates = []
    base = SimProfile()
    # Deterministic coarse-to-fine search.  The winner is re-tested by the full
    # campaign; tuning data never bypasses hard runtime limits.
    for speed in [0.62, 0.68, 0.72, 0.78]:
        for acceleration in [0.85, 1.05, 1.25]:
            for gain in [1.5, 1.8, 2.1]:
                profile = replace(
                    base, speed=speed, lateral_speed=0.90 * speed,
                    acceleration=acceleration, position_gain=gain,
                )
                summary = summarize(evaluate_campaign(field_model, scenarios, profile))
                error = summary['p95_position_error_m'] or 9.0
                time_value = summary['mean_time_sec'] or 999.0
                score = (
                    1000.0 * summary['success_rate']
                    - 200.0 * summary['collisions']
                    - 100.0 * error
                    - time_value
                )
                candidates.append((score, profile, summary))
    candidates.sort(key=lambda item: item[0], reverse=True)
    score, best, summary = candidates[0]
    output = {
        'method': 'deterministic grid search + held-out full campaign',
        'candidate_count': len(candidates), 'score': score,
        'learned_profile': asdict(best), 'training_summary': summary,
        'top_five': [
            {'score': s, 'profile': asdict(p), 'summary': m}
            for s, p, m in candidates[:5]
        ],
    }
    path = root / 'simulation' / 'results' / 'autotune.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2), encoding='utf-8')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
