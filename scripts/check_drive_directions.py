#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ACCEPTANCE step 3 を自動化する: +x / +y / +yaw の物理方向を実測で確定する。

これは残っている唯一の「ソフトでは決まらない」項目である。ミキサの各輪値
m1..m4 がどの物理コーナーへ配線されているかはどこにも記録されておらず、
`auto_lateral_sign` の -1.0 と +1.0 はどちらも幾何的に成立してしまう。
判定できるのは実機だけなので、基準には検証済みの計測輪オドメトリを使う。

Nav2 も LiDAR も使わない。measurement_wheel と motor_udp_bridge だけを
起動し、1軸ずつ低速で短時間だけ動かして /odom の増分を読む。

  +vx を出して /odom dx が正  -> 前後の向きは正しい
  +vy を出して /odom dy が正  -> 横の向きは正しい
  +wz を出して /odom dyaw が正 -> 旋回の向きは正しい

負なら該当する符号を反転する。

  前後が逆 -> auto_forward_sign
  横が逆   -> auto_lateral_sign
  旋回が逆 -> auto_turn_sign

同じ走行で*大きさ*も出る。既知の速度を既知の時間だけ出しているので、期待
変位は speed x seconds である（駆動系の無駄時間と一次遅れは波形を遅らせる
だけで面積を保つので、停止後の整定まで含めて積めば消える）。実測変位との
比が駆動系ゲイン、すなわち「指令 1 m/s に対して実際に出る速度」であり、
これが auto_units_per_mps の検算になる。あの値は「フルスティック =
0.55 m/s」という仮定から出したもので実測されていない。ゲイン誤差は速度
だけでなく、フィードフォワード・加速度・車輪バジェット・包絡線のすべてを
同じ倍率だけ嘘にし、姿勢のP項の実効ゲインを上げるので、約2.2倍を超えると
姿勢が自励振動する（実機の「フラフラ」）。

同じ定数が3か所にあるので必ず同時に直すこと。実際に走行へ効くのは1番目で
ある（payload_format は v4_uart、つまりUARTフレームを組むのは egg8 側）。

  ros2_ws/src/omni_autonomy_next/omni_autonomy_next/robomas_uart.py  AUTO_*_SIGN
  bacon_gateway/include/value.hpp                                    auto_*_sign
  bacon_gateway/tests/test_omni_velocity.cpp と test/test_robomas_uart.py の期待値

【安全】モーターが回る。実行前に必ず:
  * docs/ACCEPTANCE.md の車輪浮上試験を先に通すこと
  * 周囲 1.5 m を空けること。指令どおりなら1軸あたり 14 cm / 24 度だが、
    駆動系ゲインは未実測（まさにこの試験で測る量）で、実機で記録された
    約2.9倍が本当なら 42 cm / 72 度動く。1.5 m はそれを見込んだ値である
  * 非常停止をすぐ押せる位置にいること
"""
import argparse
import math
import os
import signal
import subprocess
import sys
import time

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy,
)
from std_msgs.msg import Bool, String

import json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROBOT_CONFIG = os.path.join(
    ROOT, 'ros2_ws', 'src', 'omni_autonomy_next', 'config', 'robot.yaml')

LINEAR_SPEED = 0.12       # m/s
ANGULAR_SPEED = 0.35      # rad/s
DRIVE_SECONDS = 1.2
SETTLE_SECONDS = 1.5

STEPS = (
    ('前後 (+vx: 前へ)', 'x', (LINEAR_SPEED, 0.0, 0.0), 'auto_forward_sign'),
    ('横   (+vy: 左へ)', 'y', (0.0, LINEAR_SPEED, 0.0), 'auto_lateral_sign'),
    ('旋回 (+wz: 左回り)', 'yaw', (0.0, 0.0, ANGULAR_SPEED), 'auto_turn_sign'),
)
AXIS_INDEX = {'x': 0, 'y': 1, 'yaw': 2}
AXIS_UNIT = {'x': 'm', 'y': 'm', 'yaw': 'rad'}


def units_per_mps():
    """実際に送っている換算係数。v4_uart 経路では egg8 側の値が効く。"""
    try:
        from omni_autonomy_next.robomas_uart import AUTO_UNITS_PER_MPS
        return float(AUTO_UNITS_PER_MPS)
    except ImportError:
        return 7202.0


def report_drive_gain(results):
    """3軸のゲインから、書き込むべき auto_units_per_mps を出す。"""
    translation = [
        gain for _, axis, _, _, gain in results
        if axis in ('x', 'y') and gain
    ]
    if not translation:
        return
    current = units_per_mps()
    measured = sum(translation) / len(translation)
    print()
    print('  駆動系ゲイン: 並進 %s（平均 %.2f 倍）'
          % ('/'.join('%.2f' % g for g in translation), measured))
    print('  現在の設定  : auto_units_per_mps = %.0f' % current)
    if abs(measured - 1.0) <= 0.10:
        print('  -> 較正は合っています（ずれ 10% 以内）。')
        return
    corrected = current / measured
    print('  -> 較正が違います。%.0f へ直してください（= %.0f / %.2f）:'
          % (corrected, current, measured))
    print('       ros2_ws/.../omni_autonomy_next/robomas_uart.py'
          '  AUTO_UNITS_PER_MPS')
    print('       bacon_gateway/include/value.hpp'
          '                auto_units_per_mps')
    print('       config/robot.yaml  max_wheel_speed = 8000/単位 * '
          'cos(45deg)/r = %.3f rad/s'
          % (8000.0 / corrected * math.cos(math.radians(45.0)) / 0.05))
    print('     直したら pytest test_robomas_uart.py と bacon6 の'
          ' ctest (omni_velocity) を通すこと。')
    if measured > 2.2:
        print('  ※ 2.2 倍を超えているので、これだけで姿勢の自励振動'
              '（走行中のフラフラ）が出ます。')


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


class DriveDirectionCheck(Node):
    def __init__(self):
        super().__init__('drive_direction_check')
        control_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST, depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE)
        transient = QoSProfile(
            history=HistoryPolicy.KEEP_LAST, depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel_safe', control_qos)
        self.arm_pub = self.create_publisher(Bool, '/system/armed', transient)
        self.pose = None
        self.telemetry = None
        self.create_subscription(Odometry, '/odom', self._on_odom, 20)
        self.create_subscription(
            String, '/motor/telemetry', self._on_telemetry, transient)

    def _on_odom(self, message):
        self.pose = (message.pose.pose.position.x,
                     message.pose.pose.position.y,
                     yaw_of(message.pose.pose.orientation))

    def _on_telemetry(self, message):
        try:
            self.telemetry = json.loads(message.data)
        except ValueError:
            pass

    def publish(self, vx, vy, wz):
        message = Twist()
        message.linear.x, message.linear.y, message.angular.z = vx, vy, wz
        self.cmd_pub.publish(message)

    def hold(self, vx, vy, wz, seconds):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self.publish(vx, vy, wz)
            rclpy.spin_once(self, timeout_sec=0.02)

    def arm(self, value):
        message = Bool()
        message.data = bool(value)
        self.arm_pub.publish(message)


def spawn(command, log_path):
    handle = open(log_path, 'wb')
    return subprocess.Popen(
        command, stdout=handle, stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL, preexec_fn=os.setsid), handle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--accept-motor-risk', action='store_true')
    args = parser.parse_args()
    if not args.accept_motor_risk:
        print(__doc__)
        print('モーターが回ります。理解した上で --accept-motor-risk を付けて'
              '再実行してください。')
        return 2

    print('measurement_wheel と motor_udp_bridge を起動します（Nav2/LiDARなし）')
    wheel, wheel_log = spawn(
        ['ros2', 'run', 'omni_autonomy_next', 'measurement_wheel',
         '--ros-args', '-p', f'robot_config_file:={ROBOT_CONFIG}'],
        '/tmp/dircheck_wheel.log')
    bridge, bridge_log = spawn(
        ['ros2', 'run', 'omni_autonomy_next', 'motor_udp_bridge', '--ros-args',
         '-p', 'cmd_vel_topic:=/cmd_vel_safe',
         '-p', 'enable_topic:=/system/armed',
         '-p', 'payload_format:=v4_uart',
         '-p', 'local_ip:=192.168.60.1', '-p', 'local_port:=8888',
         '-p', 'remote_ip:=192.168.60.2', '-p', 'remote_port:=8888',
         '-p', 'send_rate_hz:=100.0',
         '-p', 'max_linear_speed:=0.20', '-p', 'max_angular_speed:=0.50'],
        '/tmp/dircheck_bridge.log')

    rclpy.init()
    node = DriveDirectionCheck()
    results = []
    try:
        print('/odom と Pi テレメトリを待っています...')
        deadline = time.monotonic() + 25.0
        while time.monotonic() < deadline and (
            node.pose is None or node.telemetry is None
        ):
            node.arm(True)
            node.hold(0.0, 0.0, 0.0, 0.3)
        if node.pose is None:
            print('ERROR: /odom が来ません。計測輪(CNT-3204IN-USB)を確認してください。')
            print('  log: /tmp/dircheck_wheel.log')
            return 1
        if node.telemetry is None:
            print('ERROR: Pi テレメトリが来ません。'
                  'bacon6 の omni-gateway-next と有線LANを確認してください。')
            print('  log: /tmp/dircheck_bridge.log')
            return 1

        print('ARM して停止指令でハンドシェイクを成立させます...')
        for _ in range(20):
            node.arm(True)
            node.hold(0.0, 0.0, 0.0, 0.2)
            pi = (node.telemetry or {}).get('pi', {})
            if pi.get('auto_engaged'):
                break
        pi = (node.telemetry or {}).get('pi', {})
        if not pi.get('auto_engaged'):
            print('ERROR: Pi が自動モードに入りません。'
                  'uart_open=%s estop=%s rearm=%s'
                  % (pi.get('uart_open'), pi.get('estop_active'),
                     pi.get('rearm_required')))
            return 1
        print('  Pi auto_engaged=True, proto=%s' % pi.get('protocol'))

        for label, axis, command, constant in STEPS:
            print()
            print('=== %s を %.2f 秒 出します' % (label, DRIVE_SECONDS))
            input('    周囲の安全を確認して Enter（中止は Ctrl+C）: ')
            node.hold(0.0, 0.0, 0.0, 0.4)
            before = node.pose
            node.hold(*command, DRIVE_SECONDS)
            node.hold(0.0, 0.0, 0.0, SETTLE_SECONDS)
            after = node.pose
            delta = (after[0] - before[0], after[1] - before[1],
                     wrap(after[2] - before[2]))
            index = AXIS_INDEX[axis]
            value = delta[index]
            others = max(
                abs(delta[i]) for i in range(3) if i != index)
            print('    計測輪: dx=%+.3f dy=%+.3f dyaw=%+.3f' % delta)
            wheels = (node.telemetry or {}).get('pi', {}).get('wheel_commands')
            print('    直近の各輪指令: %s' % wheels)
            if abs(value) < 0.02:
                verdict = '動いていない（判定不成立）'
            elif value > 0:
                verdict = 'OK（%s は現状のままで正しい）' % constant
            else:
                verdict = '逆。%s を反転してください' % constant
            print('    %s軸 %+.3f %s -> %s'
                  % (axis, value, AXIS_UNIT[axis], verdict))
            expected = abs(command[index]) * DRIVE_SECONDS
            gain = abs(value) / expected if expected > 0.0 else None
            if gain is not None:
                print('    大きさ: 期待 %.3f %s / 実測 %.3f %s '
                      '-> 駆動系ゲイン %.2f 倍'
                      % (expected, AXIS_UNIT[axis], abs(value),
                         AXIS_UNIT[axis], gain))
            if abs(value) >= 0.02 and others > 0.6 * abs(value):
                print('    注意: 他軸にも大きく出ています。'
                      'm1..m4 の配線順（どのコーナーがID1か）を疑ってください。')
            results.append((label, axis, value, verdict, gain))

    except KeyboardInterrupt:
        print()
        print('中断しました。停止指令を送ります。')
    finally:
        try:
            for _ in range(30):
                node.publish(0.0, 0.0, 0.0)
                rclpy.spin_once(node, timeout_sec=0.02)
            node.arm(False)
            for _ in range(20):
                node.publish(0.0, 0.0, 0.0)
                rclpy.spin_once(node, timeout_sec=0.02)
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()
        for process, handle in ((bridge, bridge_log), (wheel, wheel_log)):
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGINT)
                process.wait(timeout=5)
            except Exception:
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                except Exception:
                    pass
            handle.close()

    print()
    if results:
        print('まとめ:')
        for label, axis, value, verdict, gain in results:
            print('  %-18s %+.3f %-4s %s'
                  % (label, value, AXIS_UNIT[axis], verdict))
        report_drive_gain(results)
        flips = [v for v in results if '反転' in v[3]]
        print()
        if not flips:
            print('  3軸とも正しい向きです。value.hpp の符号は現状のままで')
            print('  確定してよく、ACCEPTANCE step 3 は合格です。')
        else:
            print('  bacon_gateway/include/value.hpp を編集し、bacon6 で')
            print('    cmake --build build -j2 && ctest --test-dir build')
            print('  を通してから omni-gateway-next を再起動してください。')
            print('  tests/test_omni_velocity.cpp の期待値も同時に直すこと。')
    return 0


if __name__ == '__main__':
    sys.exit(main())
