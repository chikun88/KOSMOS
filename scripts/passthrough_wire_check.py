#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""bacon6で実行: パススルーでUARTへ実際に出たバイト列を取り込む。

ロボマス開発ボードを回さずに「egg8が送ったバイト列が、書き換えられずに
UARTへ出ているか」を確かめるための検証用ツール。

  1. ptyを1組作り、そのslave側をゲートウェイのモータ出力先に指定する
     （MU3_MOTOR_DEVICE。本番の /dev/serial0 には一切触れない）
  2. テスト用ポート(既定8899)でゲートウェイを起動する
     （本番サービスは8888と/dev/serial0のままなので併走できる）
  3. master側に出てきたバイト列をCOBSの区切り0x00でフレームに割り、
     [コマンドID, 値] へ復号して表示する

使い方:
    python3 scripts/passthrough_wire_check.py --seconds 6
    # 別ホスト(egg8)から同時に v4 コマンドを送る
"""
import argparse
import json
import os
import pty
import select
import signal
import subprocess
import sys
import time


GATEWAY = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'bacon_gateway', 'build', 'omni_gateway_next')


def decode_cobs(encoded):
    out = bytearray()
    index = 0
    while index < len(encoded):
        code = encoded[index]
        index += 1
        if code == 0:
            return None
        for _ in range(1, code):
            if index >= len(encoded):
                return None
            out.append(encoded[index])
            index += 1
        if code != 0xFF and index < len(encoded):
            out.append(0)
    return bytes(out)


def decode_frame(encoded):
    payload = decode_cobs(encoded)
    if payload is None or not payload or len(payload) % 3 != 0:
        return None
    commands = []
    for offset in range(0, len(payload), 3):
        raw = (payload[offset + 1] << 8) | payload[offset + 2]
        value = raw - 0x10000 if raw & 0x8000 else raw
        commands.append([payload[offset], value])
    return commands


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seconds', type=float, default=6.0)
    parser.add_argument('--port', type=int, default=8899)
    parser.add_argument('--json', action='store_true',
                        help='解析結果をJSONで出す（自動チェック用）')
    parser.add_argument('--log', default='/tmp/passthrough_gateway.log')
    args = parser.parse_args()

    master, slave = pty.openpty()
    slave_name = os.ttyname(slave)

    environment = dict(os.environ)
    environment['MU3_JETSON_PORT'] = str(args.port)
    environment['MU3_MOTOR_DEVICE'] = slave_name

    log = open(args.log, 'wb')
    gateway = subprocess.Popen([GATEWAY], env=environment,
                               stdout=log, stderr=subprocess.STDOUT)
    # slaveは開いたままにしておく。全員が閉じるとmaster側の読み出しが
    # EIOになるので、ゲートウェイが開く前に閉じてはいけない。

    captured = bytearray()
    deadline = time.monotonic() + args.seconds
    try:
        while time.monotonic() < deadline:
            if gateway.poll() is not None:
                break
            ready, _, _ = select.select([master], [], [], 0.2)
            if ready:
                try:
                    captured.extend(os.read(master, 4096))
                except OSError:
                    break
    finally:
        if gateway.poll() is None:
            gateway.send_signal(signal.SIGTERM)
            try:
                gateway.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                gateway.kill()
        # 終了時の停止フレームまで読み切る
        while True:
            ready, _, _ = select.select([master], [], [], 0.3)
            if not ready:
                break
            try:
                chunk = os.read(master, 4096)
            except OSError:
                break
            if not chunk:
                break
            captured.extend(chunk)
        os.close(master)
        os.close(slave)
        log.close()

    frames = [bytes(part) for part in captured.split(b'\x00') if part]
    decoded = [decode_frame(frame) for frame in frames]

    result = {
        'gateway_exit': gateway.returncode,
        'gateway_log': args.log,
        'total_bytes': len(captured),
        'frame_count': len(frames),
        'undecodable': sum(1 for entry in decoded if entry is None),
        'frames': [
            {'hex': frame.hex(), 'commands': entry}
            for frame, entry in zip(frames, decoded)
        ],
    }

    if args.json:
        json.dump(result, sys.stdout)
        return 0

    print('captured %d bytes / %d frames (undecodable=%d)'
          % (result['total_bytes'], result['frame_count'],
             result['undecodable']))
    print('--- UARTへ出た相異なるフレーム（出現回数の多い順） ---')
    counts = {}
    for entry in result['frames']:
        key = entry['hex']
        record = counts.setdefault(key, {'count': 0, 'commands': entry['commands']})
        record['count'] += 1
    ordered = sorted(counts.items(), key=lambda item: -item[1]['count'])
    for hex_frame, record in ordered[:40]:
        print('  x%-6d %-30s %s'
              % (record['count'], hex_frame, record['commands']))
    if len(ordered) > 40:
        print('  ... (%d 種類)' % len(ordered))
    return 0


if __name__ == '__main__':
    sys.exit(main())
