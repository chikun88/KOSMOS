#!/usr/bin/env bash
# egg8で実行: 実運用の motor_udp_bridge ノードで v4 パススルーを通す。
#
# 本番のポート(8888)・トピックには触らず、テスト用のポートとトピックで
# ノードを起動する。bacon6 側では scripts/passthrough_wire_check.py を
# 同時に走らせて、UARTへ出たバイト列を確認すること。
# ROSのsetup.bashは未定義変数を踏むので set -u は使わない
set -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_IP="${REMOTE_IP:-192.168.60.2}"
REMOTE_PORT="${REMOTE_PORT:-8899}"
LOCAL_PORT="${LOCAL_PORT:-8890}"
DRIVE_SECONDS="${DRIVE_SECONDS:-5}"

source /opt/ros/jazzy/setup.bash
source "${ROOT}/ros2_ws/install/setup.bash"

cleanup() {
    for pid in "${PIDS[@]:-}"; do
        kill "${pid}" 2>/dev/null || true
    done
    wait 2>/dev/null || true
}
trap cleanup EXIT

PIDS=()

ros2 run omni_autonomy_next motor_udp_bridge --ros-args \
    -p payload_format:=v4_uart \
    -p cmd_vel_topic:=/cmd_vel_pttest \
    -p enable_topic:=/motion/enable_pttest \
    -p estop_topic:=/motion/estop_pttest \
    -p status_topic:=/motor/status_pttest \
    -p telemetry_topic:=/motor/telemetry_pttest \
    -p local_ip:=192.168.60.1 -p local_port:="${LOCAL_PORT}" \
    -p remote_ip:="${REMOTE_IP}" -p remote_port:="${REMOTE_PORT}" \
    -p send_rate_hz:=100.0 \
    -p max_linear_speed:=0.55 -p max_angular_speed:=1.2 \
    > /tmp/pt_bridge.log 2>&1 &
PIDS+=($!)

ros2 topic pub -r 10 /motion/enable_pttest std_msgs/msg/Bool '{data: true}' \
    > /dev/null 2>&1 &
PIDS+=($!)

# まずゼロ指令で係合させる（Piのdisarm->rearmハンドシェイク）
ros2 topic pub -r 50 /cmd_vel_pttest geometry_msgs/msg/Twist \
    '{linear: {x: 0.0, y: 0.0}, angular: {z: 0.0}}' > /dev/null 2>&1 &
ZERO_PID=$!
PIDS+=("${ZERO_PID}")
sleep 4
kill "${ZERO_PID}" 2>/dev/null || true

ros2 topic pub -r 50 /cmd_vel_pttest geometry_msgs/msg/Twist \
    '{linear: {x: 0.30, y: 0.0}, angular: {z: 0.0}}' > /dev/null 2>&1 &
PIDS+=($!)

sleep "${DRIVE_SECONDS}"

echo "=== bridge が最後に発行したテレメトリ ==="
timeout 5 ros2 topic echo --once /motor/telemetry_pttest std_msgs/msg/String \
    || echo '(テレメトリが取れませんでした)'
echo
echo "=== bridge の状態 ==="
timeout 5 ros2 topic echo --once /motor/status_pttest std_msgs/msg/String \
    || echo '(status が取れませんでした)'
echo
echo "=== bridge ログ末尾 ==="
tail -12 /tmp/pt_bridge.log
