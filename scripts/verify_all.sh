#!/usr/bin/env bash
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
bash "${ROOT}/scripts/verify_software.sh"
set -u
cd "${ROOT}"

# Keep software success visible even when either model acceptance gate fails.
echo "Software regression gates passed."
CAMPAIGN_EPISODES="${CAMPAIGN_EPISODES:-1200}"
CAMPAIGN_RANDOM_EPISODES="${CAMPAIGN_RANDOM_EPISODES:-120}"
acceptance_failed=0

# Gate 1: the learned residual may not regress any route the deterministic
# controller already solved, and may not collide.  This is the RL gate.
if python3 -m simulation.reinforcement_learning evaluate \
  --episodes 98 --random-episodes 0 --seed 20260808 \
  --deployed-policy "${ROOT}/ros2_ws/src/omni_autonomy_next/config/rl_policy.yaml" \
  --output "${ROOT}/simulation/results/rl_configured_goals.json"; then
  echo "RL model acceptance gate passed."
else
  echo "RL MODEL ACCEPTANCE GATE FAILED." >&2
  echo "Read simulation/results/rl_configured_goals.json." >&2
  acceptance_failed=1
fi

# Gate 2: field acceptance.  This is a separate, stricter question — whether the
# robot arrives on every route. The current model has known failures;
# docs/SYSTEM_AUDIT_20261007.md describes its scope and remaining gates.
# 'failing_routes' in the report names the goal-number pairs that did not arrive.
if python3 -m simulation.run_campaign \
    --episodes "${CAMPAIGN_EPISODES}" \
    --random-episodes "${CAMPAIGN_RANDOM_EPISODES}" \
    --seed 20260801; then
  echo "Field acceptance campaign passed. Physical acceptance is still required."
else
  echo "FIELD ACCEPTANCE CAMPAIGN FAILED." >&2
  echo "Do not treat software regression success as field acceptance." >&2
  echo "Read 'failing_routes' in simulation/results/campaign.json." >&2
  acceptance_failed=1
fi
exit "${acceptance_failed}"
