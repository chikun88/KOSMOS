#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""計測輪オドメトリを LiDAR 自己位置と突き合わせて確認する（モーター出力なし）。

ロボットを手で押す/回すだけなので、モーターを一切動かさずに
オドメトリの符号とスケールを確定できる。

  /odom               : 計測輪オドメトリ
  /localization/pose  : デュアルLiDARの壁自己位置（基準）
  /wheel/counts       : 生カウント（配線/接地の切り分け用）

計測区間は「動き出し」と「停止」を自動検出して決める。Enter は要らない。
以前の版は Enter 2 回で区間を挟む方式で、押す前に区間が閉じると
生カウント差分が 0 になり、計測輪が正常でも「計測輪が無反応」と誤って
断定していた（2026-08-05 のログがその例で、同じログの /wheel/status には
0.487 m 前進・0.482 m 左・+90.15 度が正しく出ていた）。この誤診断が
オドメトリ較正を疑う調査を一巡させたので、区間の取り方そのものを変えた。

使い方:
  1) python3 run.py real          # LiDAR/計測輪ON・モーターOFF
  2) python3 run.py check-odometry
"""
import datetime
import io
import math
import os
import sys
import threading
import time

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String


# 動き出し判定。2048 CPR・半径25.4 mm なので 200 カウント ≒ 15.6 mm。
COUNT_MOTION_THRESHOLD = 200.0
WHEEL_MOTION_THRESHOLD = 0.02      # m / rad
REFERENCE_MOTION_THRESHOLD = 0.02  # m / rad
STILL_SECONDS = 1.5
MOTION_START_TIMEOUT = 90.0
MOTION_END_TIMEOUT = 120.0
# 動き出し検出より少し前へ戻した値を基準にする（検出遅れ分の取りこぼし防止）
BASELINE_LOOKBACK_SEC = 0.5


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


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


class Probe(Node):
    """購読は別スレッドの executor が回す。main は履歴を見て区間を決める。"""

    HISTORY_SECONDS = 8.0

    def __init__(self):
        super().__init__('odometry_sign_probe')
        self._lock = threading.Lock()
        self.history = []          # (t, wheel, ref, counts)
        self.wheel = None
        self.ref = None
        self.counts = None
        self.status = None
        self.create_subscription(Odometry, '/odom', self._on_wheel, 20)
        self.create_subscription(
            PoseWithCovarianceStamped, '/localization/pose', self._on_ref, 20)
        self.create_subscription(
            Float64MultiArray, '/wheel/counts', self._on_counts, 20)
        self.create_subscription(String, '/wheel/status', self._on_status, 20)
        self.create_timer(0.02, self._sample)

    def _on_wheel(self, m):
        with self._lock:
            self.wheel = (m.pose.pose.position.x, m.pose.pose.position.y,
                          yaw_of(m.pose.pose.orientation))

    def _on_ref(self, m):
        with self._lock:
            self.ref = (m.pose.pose.position.x, m.pose.pose.position.y,
                        yaw_of(m.pose.pose.orientation))

    def _on_counts(self, m):
        with self._lock:
            self.counts = list(m.data)

    def _on_status(self, m):
        with self._lock:
            self.status = m.data

    def _sample(self):
        now = time.monotonic()
        with self._lock:
            self.history.append((now, self.wheel, self.ref, self.counts))
            cutoff = now - self.HISTORY_SECONDS
            while len(self.history) > 2 and self.history[0][0] < cutoff:
                self.history.pop(0)

    def latest(self):
        with self._lock:
            return (self.wheel, self.ref, self.counts, self.status)

    def sample_at(self, target_time):
        """target_time 以前で最も新しいサンプル。無ければ最古のもの。"""
        with self._lock:
            chosen = None
            for sample in self.history:
                if sample[0] <= target_time:
                    chosen = sample
                else:
                    break
            return chosen or (self.history[0] if self.history else None)

    def have_wheel(self):
        with self._lock:
            return self.wheel is not None

    def have_ref(self):
        with self._lock:
            return self.ref is not None


def pose_delta(first, last):
    if first is None or last is None:
        return None
    return (last[0] - first[0], last[1] - first[1], wrap(last[2] - first[2]))


def counts_delta(first, last):
    if first is None or last is None or len(first) != len(last):
        return None
    return [b - a for a, b in zip(first, last)]


def moved(baseline, current):
    """baseline から見て動いたか。生カウント優先、無ければ姿勢で判定。"""
    counts = counts_delta(baseline[3], current[3])
    if counts is not None and max(abs(v) for v in counts) > COUNT_MOTION_THRESHOLD:
        return True
    wheel = pose_delta(baseline[1], current[1])
    if wheel is not None and max(abs(v) for v in wheel) > WHEEL_MOTION_THRESHOLD:
        return True
    reference = pose_delta(baseline[2], current[2])
    if (reference is not None
            and max(abs(v) for v in reference) > REFERENCE_MOTION_THRESHOLD):
        return True
    return False


def wait_for_motion(node, timeout):
    """動き出しを検出し、その直前のサンプルを返す。検出できなければ None。"""
    deadline = time.monotonic() + timeout
    quiet = node.sample_at(time.monotonic())
    while time.monotonic() < deadline:
        time.sleep(0.05)
        now = time.monotonic()
        current = node.sample_at(now)
        if current is None or quiet is None:
            quiet = current
            continue
        if moved(quiet, current):
            return node.sample_at(now - BASELINE_LOOKBACK_SEC) or quiet
        # 静止が続く間は基準を進める（ドリフトで誤検出しないため）
        if now - quiet[0] > 2.0:
            quiet = node.sample_at(now - 1.0) or quiet
    return None


def wait_for_stop(node, timeout, still_seconds=STILL_SECONDS):
    """静止が still_seconds 続くまで待ち、最後のサンプルを返す。"""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        time.sleep(0.1)
        now = time.monotonic()
        recent = node.sample_at(now - still_seconds)
        current = node.sample_at(now)
        if recent is None or current is None:
            continue
        if not moved(recent, current):
            return current
    return node.sample_at(time.monotonic())


STEPS = (
    ('ロボットを前へ約0.5 m 手で押してください', 'x'),
    ('ロボットを左へ約0.5 m 手で押してください', 'y'),
    ('ロボットを反時計回り(左回り)に約90度 手で回してください', 'yaw'),
)
AXIS_INDEX = {'x': 0, 'y': 1, 'yaw': 2}
AXIS_UNIT = {'x': 'm', 'y': 'm', 'yaw': 'rad'}


def wait_for(node, predicate, seconds):
    deadline = time.monotonic() + seconds
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.05)
    return predicate()


def stop(executor, spinner, node):
    """spin スレッドを確実に止めてから破棄する。join を省くとabortする。"""
    executor.shutdown()
    spinner.join(timeout=3.0)
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


def run_step(node, instruction, axis, has_reference):
    print()
    print('=== %s' % instruction)
    print('    そのまま動かしてください（Enter不要・動き出しを自動検出します）')
    baseline = wait_for_motion(node, MOTION_START_TIMEOUT)
    if baseline is None:
        print('    -> %.0f 秒待っても動きを検出できませんでした。'
              % MOTION_START_TIMEOUT)
        _, _, _, status = node.latest()
        if status:
            print('    計測輪ステータス: %s' % status)
        return (axis, None, '動きを検出できず')
    print('    動き出しを検出。停止を待っています...')
    final = wait_for_stop(node, MOTION_END_TIMEOUT)

    wheel = pose_delta(baseline[1], final[1])
    reference = pose_delta(baseline[2], final[2])
    counts = counts_delta(baseline[3], final[3])

    if wheel is not None:
        line = '    計測輪: dx=%+.3f dy=%+.3f dyaw=%+.3f' % wheel
        if reference is not None:
            line += '  |  LiDAR: dx=%+.3f dy=%+.3f dyaw=%+.3f' % reference
        print(line)
    if counts is not None:
        print('    生カウント差分: %s' % [int(round(v)) for v in counts])
    _, _, _, status = node.latest()
    if status:
        print('    計測輪ステータス: %s' % status)

    index = AXIS_INDEX[axis]
    counts_moved = counts is not None and max(abs(v) for v in counts) > 0.5
    wheel_moved = wheel is not None and max(abs(v) for v in wheel) > 1.0e-3

    if not counts_moved and not wheel_moved:
        print('    -> 生カウントもオドメトリも動いていません。')
        print('       配線/接地/電源側です（符号やスケールの問題ではありません）。')
        return (axis, None, '計測輪が無反応')
    if counts_moved and not wheel_moved:
        print('    -> カウントは動いていますがオドメトリが0です。'
              'calibration_matrix / 幾何モデル側の問題です。')
        return (axis, None, 'オドメトリ変換が0')
    if not has_reference or reference is None:
        print('    -> LiDAR基準が無いので比を出せません。'
              '計測輪の測定値だけ記録します: %s軸 %+.3f %s'
              % (axis, wheel[index], AXIS_UNIT[axis]))
        return (axis, None, '基準なし(実測 %+.3f %s)'
                % (wheel[index], AXIS_UNIT[axis]))
    if abs(reference[index]) < 5.0e-3:
        print('    -> LiDAR基準がほとんど動いていないため判定不成立です。'
              'もっと大きく動かしてください。')
        return (axis, None, '判定不成立(基準未移動)')

    ratio = wheel[index] / reference[index]
    verdict = 'OK' if 0.85 <= ratio <= 1.15 else (
        '符号が逆' if ratio < 0 else 'スケールずれ')
    # 意図した軸以外に大きく出ていれば軸間の取り違え（配線順）を疑う
    others = [
        (name, wheel[AXIS_INDEX[name]])
        for name in ('x', 'y', 'yaw') if name != axis
    ]
    crosstalk = [
        '%s=%+.3f' % (name, value) for name, value in others
        if abs(value) > 0.4 * abs(wheel[index])
    ]
    print('    %s軸 比 = %+.3f  -> %s' % (axis, ratio, verdict))
    if crosstalk:
        print('    注意: 他軸にも大きく出ています (%s)。'
              'チャンネル順/取付角の取り違えを疑ってください。'
              % ', '.join(crosstalk))
    return (axis, ratio, verdict)


def main():
    start_log('odometry-signs')
    rclpy.init()
    node = Probe()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spinner = threading.Thread(target=executor.spin, daemon=True)
    spinner.start()

    print('/odom と /localization/pose を待っています...')
    if not wait_for(node, node.have_wheel, 15.0):
        print('ERROR: /odom が来ていません。measurement_wheel ノードを確認してください。')
        stop(executor, spinner, node)
        return 1
    has_reference = wait_for(node, node.have_ref, 5.0)
    if not has_reference:
        print('WARN: /localization/pose が来ていません。'
              '比較なしで測定値だけ記録します。')

    results = []
    try:
        for instruction, axis in STEPS:
            results.append(run_step(node, instruction, axis, has_reference))
    except KeyboardInterrupt:
        print()
        print('中断しました。')

    print()
    if results:
        print('まとめ:')
        for axis, ratio, verdict in results:
            if ratio is None:
                print('  %-3s %s' % (axis, verdict))
            else:
                print('  %-3s 比=%+.3f  %s' % (axis, ratio, verdict))
        print()
        verdicts = [r[2] for r in results]
        if any(v == '計測輪が無反応' for v in verdicts):
            print('  1パルスもカウントしていない軸があります。符号やスケールの')
            print('  問題ではないので count_signs や calibration_matrix を触っても')
            print('  直りません。確認する順に:')
            print('   1) 計測輪が床に接地して転がるか（浮いていないか）')
            print('   2) AMT102-V のケーブルとCNT-3204IN-USBのコネクタ・電源')
            print('   3) ros2 topic echo /wheel/status で READ_ERROR が出ていないか')
        elif all(v == 'OK' for v in verdicts):
            print('  オドメトリは3軸とも基準と一致。ふらふらの原因はオドメトリでは')
            print('  ありません。指令チェーン側を見てください:')
            print('   python3 scripts/diagnose_chain.py --seconds 45')
            print('   （走行中に実行。段ごとの遅れと振動の発生段が出ます）')
        elif any(v == '符号が逆' for v in verdicts):
            print('  符号が逆の軸は config/robot.yaml の count_signs を -1 に')
            print('  してください。ただし calibration_matrix がある場合は')
            print('  count_signs は使われないので、行列側を取り直します。')
        elif any(v == 'スケールずれ' for v in verdicts):
            print('  robot.yaml のコメントにある wheel_calibration サービスで')
            print('  calibration_matrix を同定するのが確実です。')
        else:
            print('  判定不成立の軸があります。動かす量を大きくして再実行して')
            print('  ください。')
    else:
        print('測定できた軸がありません。')

    stop(executor, spinner, node)
    return 0


if __name__ == '__main__':
    sys.exit(main())
