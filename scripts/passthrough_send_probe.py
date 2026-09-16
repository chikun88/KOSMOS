#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""egg8で実行: v4(UARTパススルー)コマンドを既知の値で送り込む。

bacon6 側の scripts/passthrough_wire_check.py と対で使う検証用ツール。
ROS抜きで、実運用と同じ robomas_uart / motor_udp_protocol を使って
バイト列を組み、bacon6 のテスト用ゲートウェイへ送る。

送ったフレームと受け取ったテレメトリをJSONで出すので、bacon6側が
UARTへ出したバイト列と1バイト単位で突き合わせられる。

使い方（bacon6でwire_checkを起動してから）:
    python3 scripts/passthrough_send_probe.py --host 192.168.60.2 --port 8899
"""
import argparse
import json
import os
import socket
import sys
import time

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'ros2_ws', 'src', 'omni_autonomy_next'))

from omni_autonomy_next.motor_udp_protocol import (  # noqa: E402
    TELEMETRY_EXT_SIZE,
    TELEMETRY_SIZE,
    decode_telemetry,
    encode_v4_command,
)
from omni_autonomy_next.robomas_uart import drive_frame  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='192.168.60.2')
    parser.add_argument('--port', type=int, default=8899)
    parser.add_argument('--vx', type=float, default=0.30)
    parser.add_argument('--vy', type=float, default=0.0)
    parser.add_argument('--wz', type=float, default=0.0)
    parser.add_argument('--seconds', type=float, default=4.0)
    parser.add_argument('--rate', type=float, default=100.0)
    parser.add_argument(
        '--abort', action='store_true',
        help='走行中に無言で送信を止める（リンク断時の減速停止の確認用）')
    parser.add_argument(
        '--corrupt', choices=('none', 'crc', 'payload', 'length'),
        default='none',
        help='走行フェーズで意図的に壊したv4を送る（中継しないことの確認用）')
    args = parser.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.connect((args.host, args.port))
    sock.setblocking(False)

    period = 1.0 / max(1.0, args.rate)
    sequence = 1
    telemetry_seen = []
    sent_drive_frame = None

    def send(vx, vy, wz, auto_request, corrupt='none'):
        nonlocal sequence, sent_drive_frame
        frame, wheels, scale = drive_frame(vx, vy, wz)
        if auto_request and (vx or vy or wz):
            sent_drive_frame = {
                'hex': frame.hex(), 'wheels': list(wheels), 'scale': scale,
            }
        data = encode_v4_command(
            uart_frame=frame,
            seq=sequence,
            t_tx_us=(time.clock_gettime_ns(time.CLOCK_REALTIME) // 1000)
            & 0xFFFFFFFF,
            vx_mps=vx, vy_mps=vy, wz_radps=wz,
            buttons0=0x02 if auto_request else 0x00,
            auto_request=auto_request,
        )
        sequence = (sequence + 1) & 0xFFFF
        if corrupt != 'none':
            data = bytearray(data)
            if corrupt == 'crc':
                data[-1] ^= 0xFF
            elif corrupt == 'payload':
                # CRCはそのままに中身だけ壊す。CRC検証で弾かれるべき。
                data[20] ^= 0xFF
            elif corrupt == 'length':
                data[6] = 0xFF  # uart_lenと実長の不一致
            data = bytes(data)
        sock.send(data)

    def drain():
        while True:
            try:
                data = sock.recv(128)
            except BlockingIOError:
                return
            except OSError:
                return
            if len(data) in (TELEMETRY_SIZE, TELEMETRY_EXT_SIZE):
                try:
                    telemetry_seen.append(decode_telemetry(data))
                except ValueError:
                    pass

    # 1) disarm（自動要求なしの停止フレーム）でハンドシェイクの前半を成立させる
    for _ in range(30):
        send(0.0, 0.0, 0.0, False)
        drain()
        time.sleep(period)

    # 2) 自動要求の立ち上がりエッジ。ゼロ速度のまま係合させる
    for _ in range(30):
        send(0.0, 0.0, 0.0, True)
        drain()
        time.sleep(period)

    # 3) 実際の指令
    deadline = time.monotonic() + args.seconds
    while time.monotonic() < deadline:
        send(args.vx, args.vy, args.wz, True, corrupt=args.corrupt)
        drain()
        time.sleep(period)
    # 走行中のテレメトリを、離脱で0に戻る前に取っておく
    drive_telemetry = telemetry_seen[-1] if telemetry_seen else None

    # 4) 停止して離脱。--abort のときは何も送らずに黙って消える
    if not args.abort:
        for _ in range(20):
            send(0.0, 0.0, 0.0, False)
            drain()
            time.sleep(period)

    last = drive_telemetry
    json.dump({
        'sent_drive_frame': sent_drive_frame,
        'telemetry_count': len(telemetry_seen),
        'telemetry_during_drive': None if last is None else {
            'protocol': last.protocol_name,
            'auto_engaged': last.auto_engaged,
            'uart_open': last.uart_open,
            'link_alive': last.link_alive,
            'estop_active': last.estop_active,
            'controlled_stop': last.controlled_stop,
            'rearm_required': last.rearm_required,
            'wheel_commands': (
                None if last.wheel_commands is None
                else list(last.wheel_commands)
            ),
            'applied_velocity': [
                last.applied_vx_mmps, last.applied_vy_mmps,
                last.applied_w_mradps,
            ],
            'crc_error_count': last.crc_error_count,
            'stale_drop_count': last.stale_drop_count,
            'cmd_count': last.cmd_count,
        },
    }, sys.stdout, indent=1)
    print()
    return 0


if __name__ == '__main__':
    sys.exit(main())
