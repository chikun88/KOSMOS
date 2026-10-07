#!/usr/bin/env bash
# Reproducible software gates; field acceptance remains a separate failing gate.
set -eo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /opt/ros/jazzy/setup.bash
set -u
cd "${ROOT}"
python3 run.py ros-build
set +u
source "${ROOT}/ros2_ws/install/setup.bash"
set -u
cd "${ROOT}/ros2_ws"
ROS_DOMAIN_ID=193 colcon test --packages-select omni_route_bt --event-handlers console_direct+
colcon test-result --test-result-base build/omni_route_bt/test_results --verbose
cd "${ROOT}"
QT_QPA_PLATFORM=offscreen ROS_DOMAIN_ID=193 PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider \
  tests ros2_ws/src/omni_autonomy_next/test simulation
python3 -m compileall -q run.py start_robot.py ros2_ws/src/omni_autonomy_next simulation scripts tests
python3 run.py gateway-test
