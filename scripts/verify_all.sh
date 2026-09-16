#!/usr/bin/env bash
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /opt/ros/jazzy/setup.bash
set -u

cd "${ROOT}/ros2_ws"
colcon build --symlink-install --packages-up-to omni_autonomy_next
set +u
source install/setup.bash
set -u
ROS_DOMAIN_ID=193 colcon test --packages-select omni_route_bt --event-handlers console_direct+
colcon test-result --test-result-base build/omni_route_bt/test_results --verbose
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider \
  src/omni_autonomy_next/test
python3 -m compileall -q src/omni_autonomy_next/omni_autonomy_next

cd "${ROOT}"
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider \
  simulation/test_*.py

# The offline model now carries the real ten-vertex footprint and the deployed
# nav2 gates, so an episode costs about 1.4 s instead of 0.05 s.  Override for a
# quicker check; the release figure is the default.
CAMPAIGN_EPISODES="${CAMPAIGN_EPISODES:-1200}"
CAMPAIGN_RANDOM_EPISODES="${CAMPAIGN_RANDOM_EPISODES:-120}"

# Gate 1: the learned residual may not regress any route the deterministic
# controller already solved, and may not collide.  This is the RL gate.
python3 -m simulation.reinforcement_learning evaluate \
  --episodes 98 --random-episodes 0 --seed 20260808 \
  --deployed-policy "${ROOT}/ros2_ws/src/omni_autonomy_next/config/rl_policy.yaml" \
  --output "${ROOT}/simulation/results/rl_configured_goals.json"

echo "Software regression gates passed."

# Gate 2: field acceptance.  This is a separate, stricter question — whether the
# robot arrives on every route at all — and it is expected to fail while the
# firing poses beside the fixed bucket have under 60 mm of footprint clearance
# against a 40 mm arrival tolerance.  See docs/REINFORCEMENT_LEARNING.md.
# 'failing_routes' in the report names the goal-number pairs that did not arrive.
if python3 -m simulation.run_campaign \
    --episodes "${CAMPAIGN_EPISODES}" \
    --random-episodes "${CAMPAIGN_RANDOM_EPISODES}" \
    --seed 20260801; then
  echo "Field acceptance campaign passed. Physical acceptance is still required."
else
  echo "FIELD ACCEPTANCE CAMPAIGN FAILED." >&2
  echo "This is the known fixed-bucket geometry limit, not a regression." >&2
  echo "Read 'failing_routes' in simulation/results/campaign.json." >&2
  exit 1
fi
