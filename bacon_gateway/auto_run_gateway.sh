#!/bin/bash
# omni_autonomy_next gateway launcher for bacon6.
#
# Mirrors the guard in MU3_robomas/auto_run_rx.sh: refuse to start a second
# receiver. The binary also uses exclusive UDP/TTY ownership. These checks
# provide a useful error when an older receiver already owns either endpoint.
set -euo pipefail

GATEWAY_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${GATEWAY_DIR}/build"
BINARY="${BUILD_DIR}/omni_gateway_next"
JETSON_PORT="${MU3_JETSON_PORT:-8888}"
MOTOR_DEVICE="${MU3_MOTOR_DEVICE:-/dev/serial0}"

cd "$GATEWAY_DIR"

RECEIVER_PATTERN='(^|[[:space:]])(\./_build/bacon6|/home/bacon6/MU3_robomas/_build/bacon6|'"${BINARY//\//\\/}"')([[:space:]]|$)'

existing_receivers="$(pgrep -af "$RECEIVER_PATTERN" | grep -v "^$$ " || true)"
port_users="$(ss -H -lunp "sport = :${JETSON_PORT}" 2>/dev/null || true)"

if [[ -n "$existing_receivers" || -n "$port_users" ]]; then
    echo "[ERROR] A receiver is already running or UDP port ${JETSON_PORT} is busy." >&2
    [[ -n "$existing_receivers" ]] && { echo "[ERROR] Existing receiver process(es):" >&2; echo "$existing_receivers" >&2; }
    [[ -n "$port_users" ]] && { echo "[ERROR] UDP port ${JETSON_PORT} owner:" >&2; echo "$port_users" >&2; }
    for unit in mu3-robomas-receiver.service rx_run.service; do
        if systemctl is-active --quiet "$unit" 2>/dev/null; then
            echo "[INFO] ${unit} is active and must be stopped before this gateway runs." >&2
        fi
    done
    exit 1
fi

if [[ -e "$MOTOR_DEVICE" ]] && command -v fuser >/dev/null 2>&1; then
    if fuser "$MOTOR_DEVICE" >/dev/null 2>&1; then
        echo "[ERROR] ${MOTOR_DEVICE} is already held by another process:" >&2
        fuser -v "$MOTOR_DEVICE" >&2 || true
        exit 1
    fi
fi

if [[ ! -x "$BINARY" ]]; then
    echo "=== Building omni_gateway_next ===" >&2
    cmake -S "$GATEWAY_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=RelWithDebInfo
    cmake --build "$BUILD_DIR" -j2
fi

echo "=== Starting omni_gateway_next on UDP ${JETSON_PORT} ===" >&2
exec "$BINARY" "$@"
