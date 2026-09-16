#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""指令チェーン全段を同時記録し、蛇行(weave)の発生段を特定する。

record_run.py は最終指令と応答しか見ないため「指令側か応答側か」までしか
切り分けられない。こちらは MPPI 出力から最終指令までを全段記録し、

  * 段ごとの配信レート/ジッタ
  * 段ごとの vy・wz 符号反転レート（振動がどの段で生まれ、どこで増えるか）
  * 段間の位相遅れ（相互相関のピーク位置）
  * 経路(/plan)への横方向誤差の RMS・最大・卓越周波数
  * 実現加速度と設定上限の比較

を出す。MPPI 出力で既に反転が多ければ制御器のチューニング、下流で増えて
いればその段のレート制限・遅延が発振源である。

使い方:
  python3 scripts/diagnose_chain.py --seconds 45
"""
import argparse
import datetime
import io
import json
import math
import os
import re
import sys
import threading
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
from nav_msgs.msg import Odometry, Path
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String


# v4_uart 経路では、m/s から各輪値への換算は egg8 側の robomas_uart.py が
# 持っている（bacon6 はバイト列を書き換えずに流すだけ）。ここへ数値を書き
# 写すと、較正を直したときに診断側だけ古いまま通ってしまうので、実際に
# 送っているモジュールから読む。
def _units_per_mps():
    try:
        from omni_autonomy_next.robomas_uart import AUTO_UNITS_PER_MPS
        return float(AUTO_UNITS_PER_MPS)
    except ImportError:
        pass
    source = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'ros2_ws', 'src', 'omni_autonomy_next', 'omni_autonomy_next',
        'robomas_uart.py')
    try:
        text = open(source, encoding='utf-8').read()
    except OSError:
        return 7202.0
    match = re.search(r'AUTO_UNITS_PER_MPS\s*=\s*([0-9.]+)', text)
    return float(match.group(1)) if match else 7202.0


UNITS_PER_MPS = _units_per_mps()

STAGES = [
    ('/cmd_vel_nav', 'MPPI'),
    ('/cmd_vel_nav_smoothed', 'smoother'),
    ('/cmd_vel_rl', 'rl_policy'),
    ('/cmd_vel_collision_safe', 'collision'),
    ('/cmd_vel_safe', 'guard'),
]


class _Tee(object):
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


class Collector(Node):
    def __init__(self):
        super().__init__('chain_diagnostic')
        self._lock = threading.Lock()
        self.stages = {topic: [] for topic, _ in STAGES}
        self.odom = []
        self.pose = []
        self.plan = None
        self.plans = []
        self.safety = []
        self.telemetry = []
        self.tracker = []
        for topic, _ in STAGES:
            self.create_subscription(
                Twist, topic, self._make_twist_cb(topic), 50)
        self.create_subscription(Odometry, '/odom', self._on_odom, 50)
        self.create_subscription(
            PoseWithCovarianceStamped, '/localization/pose', self._on_pose, 50)
        self.create_subscription(Path, '/plan', self._on_plan, 5)
        self.create_subscription(
            String, '/system/safety_state', self._on_safety, 5)
        self.create_subscription(
            String, '/motor/telemetry', self._on_telemetry, 5)
        self.create_subscription(
            String, '/trajectory_tracker/status', self._on_tracker, 50)

    def _make_twist_cb(self, topic):
        def _cb(message):
            sample = (time.monotonic(), message.linear.x,
                      message.linear.y, message.angular.z)
            with self._lock:
                self.stages[topic].append(sample)
        return _cb

    def _on_odom(self, message):
        sample = (time.monotonic(),
                  message.twist.twist.linear.x,
                  message.twist.twist.linear.y,
                  message.twist.twist.angular.z)
        with self._lock:
            self.odom.append(sample)

    def _on_pose(self, message):
        sample = (time.monotonic(),
                  message.pose.pose.position.x,
                  message.pose.pose.position.y,
                  yaw_of(message.pose.pose.orientation))
        with self._lock:
            self.pose.append(sample)

    def _on_plan(self, message):
        points = [(p.pose.position.x, p.pose.position.y) for p in message.poses]
        with self._lock:
            if len(points) >= 2:
                self.plan = np.asarray(points, dtype=float)
                # Cross-track error is only meaningful against the plan that
                # was active when the pose was measured. Keeping one plan and
                # comparing the whole session to it reports the distance to a
                # different leg entirely.
                self.plans.append((time.monotonic(), self.plan))

    def _on_safety(self, message):
        with self._lock:
            self.safety.append((time.monotonic(), message.data))

    def _on_telemetry(self, message):
        with self._lock:
            self.telemetry.append((time.monotonic(), message.data))

    def _on_tracker(self, message):
        with self._lock:
            self.tracker.append((time.monotonic(), message.data))

    def snapshot(self):
        with self._lock:
            return (
                {k: list(v) for k, v in self.stages.items()},
                list(self.odom), list(self.pose), list(self.plans),
                list(self.safety), list(self.telemetry), list(self.tracker),
            )


def rate_stats(times):
    if len(times) < 3:
        return None
    gaps = np.diff(np.asarray(times, dtype=float))
    gaps = gaps[gaps > 0.0]
    if gaps.size == 0:
        return None
    return {
        'hz': 1.0 / float(np.mean(gaps)),
        'mean': 1000.0 * float(np.mean(gaps)),
        'p95': 1000.0 * float(np.percentile(gaps, 95)),
        'max': 1000.0 * float(np.max(gaps)),
        'n': int(len(times)),
    }


def reversals_per_sec(times, values, deadband):
    if len(values) < 3:
        return None
    duration = float(times[-1]) - float(times[0])
    if duration <= 0.0:
        return None
    changes = 0
    previous = 0
    for value in values:
        sign = 0 if abs(value) < deadband else (1 if value > 0 else -1)
        if sign != 0 and previous != 0 and sign != previous:
            changes += 1
        if sign != 0:
            previous = sign
    return changes / duration


def resample(times, values, grid):
    if len(times) < 2:
        return None
    return np.interp(grid, np.asarray(times, dtype=float),
                     np.asarray(values, dtype=float))


def lag_seconds(reference, signal, dt, max_lag_sec=0.8):
    if reference is None or signal is None:
        return None
    a = reference - np.mean(reference)
    b = signal - np.mean(signal)
    if np.std(a) < 1.0e-6 or np.std(b) < 1.0e-6:
        return None
    max_lag = int(max_lag_sec / dt)
    best_lag, best_score = 0, -np.inf
    for lag in range(0, max_lag + 1):
        score = float(np.dot(a, b)) if lag == 0 else float(
            np.dot(a[:-lag], b[lag:]))
        if score > best_score:
            best_score, best_lag = score, lag
    return best_lag * dt


def dominant_frequency(values, dt):
    if values is None or len(values) < 64:
        return None
    centred = values - np.mean(values)
    if np.std(centred) < 1.0e-6:
        return None
    spectrum = np.abs(np.fft.rfft(centred * np.hanning(len(centred))))
    freqs = np.fft.rfftfreq(len(centred), dt)
    valid = freqs > 0.05
    if not np.any(valid):
        return None
    return float(freqs[valid][int(np.argmax(spectrum[valid]))])


def _signed_offset(plan, point):
    start = plan[:-1]
    delta = plan[1:] - start
    length_sq = np.sum(delta * delta, axis=1)
    length_sq[length_sq < 1.0e-12] = 1.0e-12
    t = np.clip(np.sum((point - start) * delta, axis=1) / length_sq, 0.0, 1.0)
    offsets = point - (start + t[:, None] * delta)
    distances = np.hypot(offsets[:, 0], offsets[:, 1])
    index = int(np.argmin(distances))
    cross = (delta[index, 0] * offsets[index, 1]
             - delta[index, 1] * offsets[index, 0])
    return math.copysign(distances[index], cross)


def cross_track(plans, pose_samples, settle_sec=0.0):
    """Signed distance to the plan that was active when each pose was taken.

    Comparing a whole session to one stored plan measures the distance to a
    different leg of the route, not the tracking error.

    ``settle_sec`` drops samples taken just after a *new goal's* first plan.
    The robot is legitimately off a path it has only just been given, so those
    transients otherwise dominate the maximum and hide the steady tracking
    quality that a weave actually shows up in.
    """
    if not plans or not pose_samples:
        return None, None
    # A replan for the same goal keeps roughly the same geometry; only treat a
    # plan whose start jumps far from the previous one as a new leg.
    leg_starts = [plans[0][0]]
    for (_, previous), (stamp, plan) in zip(plans, plans[1:]):
        if float(np.hypot(*(plan[0] - previous[0]))) > 0.5:
            leg_starts.append(stamp)
    times, errors = [], []
    index = 0
    for stamp, x, y, _yaw in pose_samples:
        while index + 1 < len(plans) and plans[index + 1][0] <= stamp:
            index += 1
        plan_time, plan = plans[index]
        if stamp < plan_time or len(plan) < 2:
            continue
        if settle_sec > 0.0 and any(
            0.0 <= stamp - start < settle_sec for start in leg_starts
        ):
            continue
        times.append(stamp)
        errors.append(_signed_offset(plan, np.asarray((x, y), dtype=float)))
    if len(errors) < 10:
        return None, None
    return (np.asarray(times, dtype=float),
            np.asarray(errors, dtype=float))


def frame_error(commands, pose_samples, lag=0.20, window=0.12, min_speed=0.15):
    """指令の座標系が実際の動きから何度ずれているかを測る。

    機体系の指令を、その時点の姿勢で世界系へ回したものと、自己位置の差分から
    求めた実際の世界系速度の角度差。0 度なら座標系は合っている。±90 度なら
    指令が真横に出ており、位置のP制御は目標へ寄らず円を描く。180 度なら
    駆動方向の符号が逆。中央値が一定値に偏っていれば構成の誤り（車輪配置・
    符号・yaw の基準）で、ばらつくだけなら追従の質の問題である。

    ``(中央値deg, 四分位範囲deg, サンプル数)`` を返す。
    """
    if len(commands) < 32 or len(pose_samples) < 32:
        return None
    times = np.asarray([s[0] for s in pose_samples])
    x = np.asarray([s[1] for s in pose_samples])
    y = np.asarray([s[2] for s in pose_samples])
    yaw = np.asarray([s[3] for s in pose_samples])
    command_t = np.asarray([s[0] for s in commands])
    command_x = np.asarray([s[1] for s in commands])
    command_y = np.asarray([s[2] for s in commands])

    offsets = []
    for index, start in enumerate(times):
        end = start + window
        if end > times[-1]:
            break
        moved = np.array([
            float(np.interp(end, times, x)) - float(x[index]),
            float(np.interp(end, times, y)) - float(y[index]),
        ])
        realised = moved / window
        if float(np.hypot(*realised)) < min_speed:
            continue
        # 指令は駆動系の遅れのぶん過去のものが効いている。
        stamp = start + 0.5 * window - lag
        if stamp < command_t[0]:
            continue
        body = np.array([
            float(np.interp(stamp, command_t, command_x)),
            float(np.interp(stamp, command_t, command_y)),
        ])
        if float(np.hypot(*body)) < min_speed:
            continue
        heading = float(yaw[index])
        cosine, sine = math.cos(heading), math.sin(heading)
        wanted = np.array([
            body[0] * cosine - body[1] * sine,
            body[0] * sine + body[1] * cosine,
        ])
        offsets.append(math.degrees(math.atan2(
            float(np.cross(wanted, realised)), float(np.dot(wanted, realised)))))
    if len(offsets) < 16:
        return None
    values = np.asarray(offsets)
    return (
        float(np.median(values)),
        float(np.percentile(values, 75) - np.percentile(values, 25)),
        len(values),
    )


def delivered_gain(guard, odom, dt):
    """指令 1 m/s に対して実際に出た速度の比（駆動系ゲイン）を出す。

    これが 1.0 でなければ bacon_gateway/include/value.hpp の
    auto_units_per_mps が違う。あの値は「フルスティック = 0.55 m/s」という
    仮定から出したもので実測されていない。効き方は速度だけではない:

      * フィードフォワードが全区間その倍率だけ外れる
      * 加速度・車輪バジェット・包絡線もすべて同じ倍率だけ嘘になる
      * 姿勢のP項の実効ゲインが同じ倍率だけ上がるので、約 2.2 倍を超えると
        姿勢が自励振動する（trajectory_tracker の yaw_gain の項を参照）

    計測輪オドメトリは LiDAR 自己位置に対して同定済みの行列で解いてあり
    （robot.yaml の calibration_matrix、幾何モデルと 6% 以内で一致）、
    ここでの基準に使える。指令と応答は無駄時間ぶんずれているので、相互
    相関で求めた遅れだけ戻してから原点を通る最小二乗の傾きを取る。
    """
    if len(guard) < 64 or len(odom) < 64:
        return
    start = max(guard[0][0], odom[0][0])
    end = min(guard[-1][0], odom[-1][0])
    if end - start < 4.0:
        return
    grid = np.arange(start, end, dt)
    print()
    print('--- 駆動系ゲイン（指令に対して実際に出た速度）---')
    print('%-8s %8s %10s %12s'
          % ('axis', 'gain', 'n(moving)', 'implied 単位/(m/s)'))
    gains = {}
    for name, column, floor in (('vx', 1, 0.08), ('vy', 2, 0.08),
                                ('wz', 3, 0.15)):
        command = resample([s[0] for s in guard],
                           [s[column] for s in guard], grid)
        response = resample([s[0] for s in odom],
                            [s[column] for s in odom], grid)
        lag = lag_seconds(command, response, dt)
        shift = 0 if lag is None else int(round(lag / dt))
        if shift > 0:
            command, response = command[:-shift], response[shift:]
        moving = np.abs(command) > floor
        if int(np.sum(moving)) < 32:
            print('%-8s %8s %10d' % (name, '-', int(np.sum(moving))))
            continue
        c, r = command[moving], response[moving]
        gain = float(np.dot(c, r) / np.dot(c, c))
        gains[name] = gain
        implied = UNITS_PER_MPS / gain if gain > 1.0e-6 else float('inf')
        print('%-8s %8.2f %10d %12.0f' % (name, gain, len(c), implied))
    translation = [gains[k] for k in ('vx', 'vy') if k in gains]
    if not translation:
        print('  （動いている区間が足りない。目標へ走らせながら記録すること）')
        return
    worst = max(translation, key=lambda value: abs(math.log(max(value, 1e-6))))
    print('  設定: auto_units_per_mps = %.0f （フルスティック = 0.55 m/s の'
          '仮定から。実測ではない）' % UNITS_PER_MPS)
    if abs(worst - 1.0) <= 0.25:
        print('  -> 較正は合っている（ずれ 25% 以内）。')
    else:
        print('  -> ずれている。auto_units_per_mps を %.0f へ（= %.0f / %.2f）、'
              % (UNITS_PER_MPS / worst, UNITS_PER_MPS, worst))
        print('     robomas_uart.py の AUTO_UNITS_PER_MPS と robot.yaml の')
        print('     max_wheel_speed = 8000/単位 * cos45/r も同時に直すこと。')
        if worst > 2.2:
            print('     ※ 2.2 倍を超えているので、これだけで姿勢の自励振動'
                  '（フラフラ）が出る。')


def report(collector, dt=0.01):
    stages, odom, pose, plans, safety, telemetry, tracker = collector.snapshot()
    print()
    print('=' * 70)
    print('COMMAND CHAIN DIAGNOSTIC / 指令チェーン診断')
    print('=' * 70)

    print()
    print('--- 段ごとの配信レート ---')
    print('%-26s %7s %9s %9s %9s %6s'
          % ('topic', 'Hz', 'mean_ms', 'p95_ms', 'max_ms', 'n'))
    rows = [(t, stages[t]) for t, _ in STAGES]
    rows.append(('/odom', odom))
    rows.append(('/localization/pose', pose))
    for name, samples in rows:
        stats = rate_stats([s[0] for s in samples])
        if stats is None:
            print('%-26s   (no data)' % name)
            continue
        print('%-26s %7.1f %9.1f %9.1f %9.1f %6d'
              % (name, stats['hz'], stats['mean'], stats['p95'],
                 stats['max'], stats['n']))

    print()
    print('--- 振動指標（符号反転 回/秒。大きいほど振動）---')
    print('%-26s %9s %9s %9s %9s'
          % ('topic', 'vy_rev/s', 'wz_rev/s', 'max|vy|', 'max|vx|'))
    for name, samples in rows[:5] + [('/odom (response)', odom)]:
        if len(samples) < 3:
            continue
        times = [s[0] for s in samples]
        vy_rate = reversals_per_sec(times, [s[2] for s in samples], 0.03)
        wz_rate = reversals_per_sec(times, [s[3] for s in samples], 0.05)
        print('%-26s %9s %9s %9.3f %9.3f'
              % (name,
                 '-' if vy_rate is None else '%.2f' % vy_rate,
                 '-' if wz_rate is None else '%.2f' % wz_rate,
                 max(abs(s[2]) for s in samples),
                 max(abs(s[1]) for s in samples)))

    reference = stages['/cmd_vel_nav']
    populated = [stages[t] for t, _ in STAGES if len(stages[t]) > 2]
    if len(reference) >= 64 and len(populated) >= 2:
        start = max(s[0][0] for s in populated)
        end = min(s[-1][0] for s in populated)
        if end - start > 3.0:
            grid = np.arange(start, end, dt)
            base_vy = resample([s[0] for s in reference],
                               [s[2] for s in reference], grid)
            base_vx = resample([s[0] for s in reference],
                               [s[1] for s in reference], grid)
            print()
            print('--- MPPI出力からの遅れ（相互相関ピーク）---')
            for topic, _ in STAGES[1:]:
                samples = stages[topic]
                if len(samples) < 32:
                    continue
                vy = resample([s[0] for s in samples],
                              [s[2] for s in samples], grid)
                vx = resample([s[0] for s in samples],
                              [s[1] for s in samples], grid)
                lag_vy = lag_seconds(base_vy, vy, dt)
                lag_vx = lag_seconds(base_vx, vx, dt)
                print('  %-24s vy %6s ms   vx %6s ms'
                      % (topic,
                         '-' if lag_vy is None else '%.0f' % (1000 * lag_vy),
                         '-' if lag_vx is None else '%.0f' % (1000 * lag_vx)))
            if len(odom) >= 32:
                ovy = resample([s[0] for s in odom], [s[2] for s in odom], grid)
                lag_o = lag_seconds(base_vy, ovy, dt)
                print('  %-24s vy %6s ms   <- 指令から実測までの全遅れ'
                      % ('/odom',
                         '-' if lag_o is None else '%.0f' % (1000 * lag_o)))
            guard = stages['/cmd_vel_safe']
            if len(guard) >= 64:
                gvy = resample([s[0] for s in guard],
                               [s[2] for s in guard], grid)
                freq = dominant_frequency(gvy, dt)
                if freq:
                    print()
                    print('  最終指令 vy 卓越振動 = %.2f Hz (周期 %.0f ms)'
                          % (freq, 1000.0 / freq))

    guard = stages['/cmd_vel_safe']
    if len(guard) >= 20:
        times = np.asarray([s[0] for s in guard])
        vx = np.asarray([s[1] for s in guard])
        vy = np.asarray([s[2] for s in guard])
        gaps = np.diff(times)
        valid = gaps > 1.0e-4
        if np.any(valid):
            ax = np.diff(vx)[valid] / gaps[valid]
            ay = np.diff(vy)[valid] / gaps[valid]
            magnitude = np.hypot(ax, ay)
            speed = np.hypot(vx, vy)
            print()
            print('--- 最終指令 /cmd_vel_safe の実現値 ---')
            print('  速度 |v| p95 = %.3f m/s   max = %.3f m/s'
                  % (float(np.percentile(speed, 95)), float(np.max(speed))))
            print('  加速 |a| p95 = %.3f m/s^2 max = %.3f m/s^2'
                  % (float(np.percentile(magnitude, 95)),
                     float(np.max(magnitude))))
            print('  設定: balanced linear=0.78 m/s, linear_accel=0.85 m/s^2')
            print('  （|a| p95 が 0.85 を大きく下回る = 加速が出せていない）')

    delivered_gain(stages['/cmd_vel_safe'], odom, dt)

    times, errors = cross_track(plans, pose)
    if errors is not None:
        print()
        print('--- 経路(/plan)への横方向誤差（蛇行の直接指標）---')
        print('  全区間  RMS = %.3f m  p95 = %.3f m  max = %.3f m  '
              'mean = %+.3f m  n=%d'
              % (float(np.sqrt(np.mean(errors ** 2))),
                 float(np.percentile(np.abs(errors), 95)),
                 float(np.max(np.abs(errors))),
                 float(np.mean(errors)), len(errors)))
        settled_times, settled = cross_track(plans, pose, settle_sec=2.0)
        if settled is not None:
            print('  定常のみ RMS = %.3f m  p95 = %.3f m  max = %.3f m  n=%d'
                  '   (新経路直後2.0秒を除外)'
                  % (float(np.sqrt(np.mean(settled ** 2))),
                     float(np.percentile(np.abs(settled), 95)),
                     float(np.max(np.abs(settled))), len(settled)))
        print('  符号反転 = %.2f 回/秒'
              % (reversals_per_sec(times, errors, 0.02) or 0.0))
        if len(times) > 64 and times[-1] > times[0]:
            grid = np.arange(times[0], times[-1], dt)
            freq = dominant_frequency(np.interp(grid, times, errors), dt)
            if freq:
                print('  卓越周波数 = %.2f Hz (周期 %.0f ms)'
                      % (freq, 1000.0 / freq))
        print('  （mean が 0 から離れている = 片側への定常オフセット。'
              '反転が多く RMS が大きい = 蛇行）')

    offset = frame_error(stages['/cmd_vel_safe'], pose)
    if offset is not None:
        median, spread, count = offset
        print()
        print('--- 指令の座標系と実際の動きのずれ ---')
        print('  中央値 = %+.1f deg   四分位範囲 = %.1f deg   n=%d'
              % (median, spread, count))
        if abs(median) > 25.0:
            print('  -> 一定して %+.0f 度ずれている。位置のP制御は目標へ寄らず'
                  '弧を描く。' % median)
            print('     車輪配置(motor_kinematics)・回転方向の符号・'
                  'LiDAR自己位置の yaw 基準のどれかが違う。')
            print('     python3 run.py check-drive-directions で切り分ける。')
        else:
            print('  -> 座標系は合っている（ずれ < 25 deg）。'
                  '残る誤差は追従の質の問題。')

    if tracker:
        states = {}
        transitions = []
        last_state = None
        first = tracker[0][0]
        for stamp, payload in tracker:
            try:
                parsed = json.loads(payload)
            except (ValueError, TypeError):
                continue
            state = parsed.get('state', '?')
            states[state] = states.get(state, 0) + 1
            if state != last_state:
                transitions.append((stamp - first, state, parsed))
                last_state = state
        print()
        print('--- trajectory_tracker 状態内訳 ---')
        for state, count in sorted(states.items(), key=lambda kv: -kv[1]):
            print('  %-26s %d' % (state, count))
        if transitions:
            print('  状態遷移（先頭20件）:')
            for elapsed, state, parsed in transitions[:20]:
                extra = {
                    key: value for key, value in parsed.items()
                    if key not in ('state', 'position_error', 'plan_build_ms')
                }
                print('    %6.2f s  %-22s %s'
                      % (elapsed, state, json.dumps(
                          extra, separators=(',', ':'), ensure_ascii=False)[:96]))

    if safety:
        reasons = {}
        last = None
        for _, payload in safety:
            try:
                parsed = json.loads(payload)
            except (ValueError, TypeError):
                continue
            last = parsed
            key = parsed.get('reason', '?')
            reasons[key] = reasons.get(key, 0) + 1
        print()
        print('--- RuntimeGuard 状態内訳 ---')
        for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1]):
            print('  %-26s %d' % (reason, count))
        if last:
            print('  最終: profile=%s applied_scale=%s speed_limit_pct=%s'
                  % (last.get('profile'), last.get('applied_scale'),
                     last.get('planner_speed_limit_pct')))

    if telemetry:
        try:
            last = json.loads(telemetry[-1][1])
            pi = last.get('pi', {})
            print()
            print('--- モータリンク（Pi）---')
            print('  RTT avg=%s ms  loss=%s %%  rx_rate=%s Hz'
                  % (last.get('rtt_ms', {}).get('avg'), last.get('loss_pct'),
                     pi.get('rx_rate_hz')))
            print('  crc_err=%s  stale_drop=%s  hold_us=%s'
                  % (pi.get('crc_error_count'), pi.get('stale_drop_count'),
                     pi.get('hold_us')))
            print('  proto=%s  applied_velocity=%s  wheels=%s'
                  % (pi.get('protocol'), pi.get('applied_velocity'),
                     pi.get('wheel_commands')))
        except (ValueError, TypeError):
            pass
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seconds', type=float, default=45.0)
    args = parser.parse_args()

    start_log('diagnose-chain')
    rclpy.init()
    node = Collector()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spinner = threading.Thread(target=executor.spin, daemon=True)
    spinner.start()
    print('recording %.0f s ...' % args.seconds, flush=True)
    deadline = time.monotonic() + args.seconds
    try:
        while time.monotonic() < deadline:
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    report(node)
    executor.shutdown()
    spinner.join(timeout=3.0)
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
