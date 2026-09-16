#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /opt/ros/jazzy/setup.bash
source "${ROOT}/ros2_ws/install/setup.bash"
exec ros2 launch omni_autonomy_next system.launch.py \
  demo:=true lidars:=false wheels:=false motors:=false gui:=true rviz:=true

