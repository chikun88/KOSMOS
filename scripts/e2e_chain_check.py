#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""egg8 -> Ethernet -> bacon6 -> UART送出値 までを実測で突き合わせる。

/cmd_vel_safe に既知のツイストを流し、Piがテレメトリで返す
「実際にUARTへ送った各輪値」と「適用した機体速度」を読み取り、

  1. Pi がコマンドを復号し自動モードに入ったか（v4ならパススルー経路）
  2. 各輪値がミキサ式と一致するか
     （v4では egg8 の robomas_uart.mix_velocity が組んだ値がそのまま
       開発ボードへ届く。Pi は中身を書き換えないので、ここが一致する
       ことは「Jetsonの計算がそのまま実機に出た」ことの確認になる）
  3. その各輪値を逆変換すると元のツイストに戻るか（指令が失われていないか）
  4. その各輪値が実際のオムニ幾何（車輪位置・ローラ角）と整合するか
  5. 各輪が 8000 の上限を超えていないか

を検証する。4 は「オムニとして正しい速度になっているか」の確認で、
ミキサ式そのものではなく CAD 由来の幾何から独立に計算する。
"""
import json
import math
import sys
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy,
)
from std_msgs.msg import String


import os
import re

# --- bacon_gateway/include/value.hpp から読む ---
# 符号をここに書き写すと、value.hpp を直したときに検証側だけ古いまま通って
# しまう。期待値の出どころは常にゲートウェイのソース1箇所にする。
VALUE_HPP = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'bacon_gateway', 'include', 'value.hpp')


def _constant(name, default):
    try:
        text = open(VALUE_HPP, encoding='utf-8').read()
    except OSError:
        return default
    match = re.search(
        r'constexpr\s+(?:double|int)\s+%s\s*=\s*(-?[0-9.]+)' % name, text)
    return float(match.group(1)) if match else default


UNITS_PER_MPS = _constant('auto_units_per_mps', 7202.0)
LEVER_ARM_M = _constant('auto_rotation_lever_arm_m', 0.666144)
UNITS_PER_RADPS = UNITS_PER_MPS * LEVER_ARM_M
LATERAL_SIGN = _constant('auto_lateral_sign', -1.0)
FORWARD_SIGN = _constant('auto_forward_sign', 1.0)
TURN_SIGN = _constant('auto_turn_sign', -1.0)
WHEEL_LIMIT = int(_constant('omni_max', 8000.0))

# --- config/robot.yaml の drivetrain と一致していること ---
WHEEL_RADIUS = 0.05
HALF = 0.333072
MAX_WHEEL_RADPS = 15.709120382

# ミキサの各輪と物理コーナーの対応。実測済みの契約
#   並進入力が正 -> 前進 / 横入力が正 -> 左 / 旋回入力が正 -> 時計回り
# と、4輪すべてで矛盾なく成立する唯一の族の代表。ローラ角はすべて各コーナーの
# 接線方向を向く。
#
# 2026-08-07 に横方向を実機で測り直して auto_lateral_sign を +1.0 にしたので、
# ここも m1<->m2 / m3<->m4 が入れ替わる。配線がそうなっているという意味で、
# どのコーナーがID1かはどこにも記録されていない量である。
CORNERS = np.array([
    [-HALF, -HALF],   # m1 後右
    [+HALF, -HALF],   # m2 前右
    [+HALF, +HALF],   # m3 前左
    [-HALF, +HALF],   # m4 後左
])
ROLLER_DEG = np.array([135.0, 225.0, -45.0, 45.0])


def mixer(vx, vy, wz):
    """move.cpp の OMNI::mix_velocity と同じ計算。"""
    ux = vy * UNITS_PER_MPS * LATERAL_SIGN
    uy = vx * UNITS_PER_MPS * FORWARD_SIGN
    ut = wz * UNITS_PER_RADPS * TURN_SIGN
    m = [ux - uy + ut, -ux - uy + ut, -ux + uy + ut, ux + uy + ut]
    peak = max(abs(v) for v in m)
    k = WHEEL_LIMIT / peak if peak > WHEEL_LIMIT else 1.0
    return [int(round(v * k)) for v in m], k


def unmix(wheels):
    """各輪値からツイストを復元する。ミキサの逆行列。"""
    m0, m1, m2, m3 = (float(v) for v in wheels)
    ux = (m0 - m1 - m2 + m3) / 4.0
    uy = (-m0 - m1 + m2 + m3) / 4.0
    ut = (m0 + m1 + m2 + m3) / 4.0
    return (
        uy / (UNITS_PER_MPS * FORWARD_SIGN),
        ux / (UNITS_PER_MPS * LATERAL_SIGN),
        ut / (UNITS_PER_RADPS * TURN_SIGN),
    )


def geometric_wheel_radps(vx, vy, wz):
    """CAD幾何から各輪の実回転速度[rad/s]を独立に計算する。"""
    angles = np.radians(ROLLER_DEG)
    directions = np.column_stack((np.cos(angles), np.sin(angles)))
    lever = (-CORNERS[:, 1] * directions[:, 0]
             + CORNERS[:, 0] * directions[:, 1])
    surface = directions[:, 0] * vx + directions[:, 1] * vy + lever * wz
    return surface / WHEEL_RADIUS


CASES = [
    ('前進 0.50 m/s', 0.50, 0.0, 0.0),
    ('後退 0.50 m/s', -0.50, 0.0, 0.0),
    ('左 0.50 m/s', 0.0, 0.50, 0.0),
    ('右 0.50 m/s', 0.0, -0.50, 0.0),
    ('左旋回 1.00 rad/s', 0.0, 0.0, 1.00),
    ('右旋回 1.00 rad/s', 0.0, 0.0, -1.00),
    ('斜め前左 0.35/0.35', 0.35, 0.35, 0.0),
    ('前進+旋回 0.60/0.60', 0.60, 0.0, 0.60),
    ('飽和 0.78/0.00/1.30', 0.78, 0.0, 1.30),
]


class Harness(Node):
    def __init__(self):
        super().__init__('e2e_chain_check')
        control_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST, depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        transient = QoSProfile(
            history=HistoryPolicy.KEEP_LAST, depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel_safe', control_qos)
        self.telemetry = None
        self.create_subscription(
            String, '/motor/telemetry', self._on_telemetry, transient)
        self.status = None
        self.create_subscription(
            String, '/motor/network_status', self._on_status, transient)

    def _on_telemetry(self, message):
        try:
            self.telemetry = json.loads(message.data)
        except ValueError:
            pass

    def _on_status(self, message):
        try:
            self.status = json.loads(message.data)
        except ValueError:
            pass

    def drive(self, vx, vy, wz, seconds):
        """指令を出し続けたまま、その最中のテレメトリを採取して返す。

        指令を止めてから読んではいけない。command_timeout_sec は 0.15 s で、
        ブリッジは指令の途絶を「制御リンク喪失」として rearm をラッチし、
        直ちに 0 を送る（それが正しい安全動作である）。停止後に読むと、
        経路が健全でも全輪 0 が観測される。
        """
        message = Twist()
        message.linear.x, message.linear.y, message.angular.z = vx, vy, wz
        start = time.monotonic()
        deadline = start + seconds
        settled_after = start + seconds * 0.5
        captured = None
        while time.monotonic() < deadline:
            self.cmd_pub.publish(message)
            rclpy.spin_once(self, timeout_sec=0.02)
            if time.monotonic() >= settled_after and self.telemetry is not None:
                captured = self.telemetry
        return captured if captured is not None else self.telemetry


def main():
    rclpy.init()
    node = Harness()
    print('Pi のテレメトリを待っています...')
    deadline = time.monotonic() + 20.0
    while node.telemetry is None and time.monotonic() < deadline:
        node.drive(0.0, 0.0, 0.0, 0.2)
    if node.telemetry is None:
        print('ERROR: /motor/telemetry が来ません。'
              'bridge か Pi 側テストゲートウェイを確認してください。')
        print('  bridge status =', node.status)
        return 1

    # 停止指令で ARM ハンドシェイクを成立させる
    node.drive(0.0, 0.0, 0.0, 3.0)
    telemetry = node.telemetry
    pi = telemetry['pi']
    print()
    print('=== リンクとハンドシェイク ===')
    print('  proto           : %s (v4=UARTパススルー)'
          % pi.get('protocol', 'v3' if pi.get('cmd_was_v3') else '?'))
    print('  auto_engaged    : %s' % pi['auto_engaged'])
    print('  link_alive      : %s' % pi['link_alive'])
    print('  rx_rate         : %s Hz' % pi['rx_rate_hz'])
    print('  crc_error_count : %s' % pi['crc_error_count'])
    print('  stale_drop_count: %s' % pi['stale_drop_count'])
    print('  RTT avg         : %s ms' % telemetry['rtt_ms']['avg'])
    print('  loss            : %s %%' % telemetry['loss_pct'])
    if not pi['auto_engaged']:
        print()
        print('ERROR: Pi が自動モードに入っていません。以降の値は0になります。')
        print('  bridge state =', (node.status or {}).get('state'))
        return 1

    print()
    print('=== 指令 -> egg8ミキサ -> UARTフレーム -> Ethernet -> UART送出値 ===')
    print('%-22s %-22s %-6s %s'
          % ('指令 (vx,vy,wz)', 'UART各輪値 m1..m4', 'k', '判定'))
    failures = []
    for label, vx, vy, wz in CASES:
        telemetry = node.drive(vx, vy, wz, 2.0)
        pi = telemetry['pi']
        wheels = pi['wheel_commands']
        applied = pi['applied_velocity']
        expected, k = mixer(vx, vy, wz)
        problems = []

        # 1) ミキサ式との一致（量子化1LSB許容）
        if wheels is None or any(
            abs(int(a) - int(b)) > 1 for a, b in zip(wheels, expected)
        ):
            problems.append('ミキサ不一致 expected=%s' % expected)

        # 2) 上限
        if wheels is not None and max(abs(int(v)) for v in wheels) > WHEEL_LIMIT:
            problems.append('上限超過')

        # 3) 各輪値の逆変換で元のツイストへ戻るか
        if wheels is not None:
            rx, ry, rw = unmix(wheels)
            if not (
                math.isclose(rx, vx * k, abs_tol=2.0e-3)
                and math.isclose(ry, vy * k, abs_tol=2.0e-3)
                and math.isclose(rw, wz * k, abs_tol=4.0e-3)
            ):
                problems.append(
                    '逆変換不一致 got=(%.3f,%.3f,%.3f) want=(%.3f,%.3f,%.3f)'
                    % (rx, ry, rw, vx * k, vy * k, wz * k))

        # 4) CAD幾何と独立に整合するか（各輪の実回転速度）
        if wheels is not None:
            geometric = geometric_wheel_radps(vx * k, vy * k, wz * k)
            # ミキサ単位 -> rad/s : m / (UNITS_PER_MPS * sqrt(2)) / r
            from_wire = np.asarray(wheels, dtype=float) / (
                UNITS_PER_MPS * math.sqrt(2.0) * WHEEL_RADIUS)
            if np.max(np.abs(from_wire - geometric)) > 0.02:
                problems.append(
                    '幾何不一致 wire=%s geom=%s'
                    % (np.round(from_wire, 3).tolist(),
                       np.round(geometric, 3).tolist()))
            if np.max(np.abs(geometric)) > MAX_WHEEL_RADPS + 1.0e-3:
                problems.append('車輪速度上限超過 %.3f rad/s'
                                % float(np.max(np.abs(geometric))))

        # 5) Piが報告する適用速度
        if applied is not None:
            got = (applied['vx_mmps'] / 1000.0, applied['vy_mmps'] / 1000.0,
                   applied['w_mradps'] / 1000.0)
            want = (vx * k, vy * k, wz * k)
            if any(abs(a - b) > 2.0e-3 for a, b in zip(got, want)):
                problems.append('適用速度不一致 got=%s want=%s'
                                % (tuple(round(v, 3) for v in got),
                                   tuple(round(v, 3) for v in want)))

        verdict = 'OK' if not problems else 'NG: ' + ' / '.join(problems)
        if problems:
            failures.append((label, problems))
        print('%-22s %-22s %-6.3f %s'
              % ('%s' % label, str(wheels), k, verdict))

    node.drive(0.0, 0.0, 0.0, 1.0)
    print()
    if failures:
        print('=== %d 件不一致 ===' % len(failures))
        for label, problems in failures:
            print('  %s: %s' % (label, '; '.join(problems)))
    else:
        print('=== 全ケース一致 ===')
        print('  指令が Ethernet を渡り、Pi が v3 として復号し、ミキサが')
        print('  CAD幾何どおりの各輪値を作り、その値が UART へ送られている')
        print('  ところまで、数値として一致を確認しました。')
    telemetry = node.telemetry
    print()
    print('  最終 crc_err=%s stale_drop=%s loss=%s%% RTT=%.2f ms'
          % (telemetry['pi']['crc_error_count'],
             telemetry['pi']['stale_drop_count'],
             telemetry['loss_pct'],
             telemetry['rtt_ms']['avg'] or 0.0))
    node.destroy_node()
    rclpy.shutdown()
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
