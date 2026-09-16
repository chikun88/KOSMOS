#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""1回の自律走行を記録し、ふらふらが指令側か応答側かを切り分ける。

  python3 scripts/record_run.py            # 走行中に実行し Ctrl+C で集計
"""
import datetime
import io
import math
import os
import sys
import signal
import time
from collections import deque

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from std_msgs.msg import String
from omni_autonomy_next.run_recorder_node import Recorder



class _Tee(object):
    """標準出力をログファイルにも複製する。毎回flushするので対話入力も崩れない。"""

    def __init__(self, stream, path):
        self._stream = stream
        self._file = io.open(path, 'w', encoding='utf-8')

    def write(self, data):
        self._stream.write(data)
        self._stream.flush()
        self._file.write(data)
        self._file.flush()

    def flush(self):
        self._stream.flush()
        self._file.flush()

    def isatty(self):
        return self._stream.isatty()


def start_log(name):
    """logs/<name>-<timestamp>.log へ出力を複製し、そのパスを返す。"""
    directory = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
    os.makedirs(directory, exist_ok=True)
    stamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    path = os.path.join(directory, '%s-%s.log' % (name, stamp))
    sys.stdout = _Tee(sys.stdout, path)
    print('log: %s' % path)
    return path


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def sign_changes(values, deadband):
    changes = 0
    previous = 0
    for v in values:
        s = 0 if abs(v) < deadband else (1 if v > 0 else -1)
        if s != 0 and previous != 0 and s != previous:
            changes += 1
        if s != 0:
            previous = s
    return changes



def report(node):
    print()
    print('全記録件数: %s（下記集計は各信号の直近100000件まで）' % node.counts)
    print('=== samples: cmd=%d odom=%d pose=%d ===' % (
        len(node.cmd), len(node.odom), len(node.pose)))
    if not node.cmd:
        print('指令が記録されていません。/cmd_vel_safe が出ているか確認してください。')
        return
    moving = [c for c in node.cmd if abs(c[0]) > 0.02 or abs(c[1]) > 0.02]
    vy_used = [c for c in moving if abs(c[1]) > 0.02]
    print('指令 /cmd_vel_safe:')
    print('  |vx| peak=%.3f   |vy| peak=%.3f   |wz| peak=%.3f' % (
        max(abs(c[0]) for c in node.cmd),
        max(abs(c[1]) for c in node.cmd),
        max(abs(c[2]) for c in node.cmd)))
    if moving:
        print('  横速度が乗っていた割合 = %.0f%% (%d/%d moving samples)' % (
            100.0 * len(vy_used) / len(moving), len(vy_used), len(moving)))
    print('  vy 符号反転回数 = %d   wz 符号反転回数 = %d' % (
        sign_changes([c[1] for c in node.cmd], 0.03),
        sign_changes([c[2] for c in node.cmd], 0.05)))
    if node.odom:
        print('応答 /wheel/odometry twist:')
        print('  vy 符号反転回数 = %d   wz 符号反転回数 = %d' % (
            sign_changes([o[1] for o in node.odom], 0.03),
            sign_changes([o[2] for o in node.odom], 0.05)))
    print()
    print('読み方:')
    print('  横速度の割合がほぼ0%  -> vy が出ていない。ホロノミック探索が効いて')
    print('                          いないか下流で潰れている。MPPI motion_model と')
    print('                          velocity_smoother max_velocity[1] を確認。')
    print('  指令の vy 反転が多い  -> 指令側で振動している。MPPI のチューニング')
    print('                          (PathAlignCritic 24.0 / PathFollowCritic 18.0)。')
    print('  指令は滑らかで応答が  -> 実行側。ゲートウェイの符号かオドメトリ較正。')
    print('  振動している            check_odometry_signs.py を先に回す。')


def main():
    log_path = start_log('record-run')
    rclpy.init()
    node = Recorder()
    print('詳細走行データ: %s' % node.directory, flush=True)
    print('記録中です。走行させて、終わったら Ctrl+C を押してください。', flush=True)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        node.close()
        report(node)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
