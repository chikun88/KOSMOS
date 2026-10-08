#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""配備値からソフトウェアの速度制約を計算する。実機の限界性能ではない。

「もっと速く・もっと正確に」を議論する前に、どこが効いていて、どこが
効かないかを設定から確認するためのもの。設定を読むだけで、実機も
ROSも要らない。

  python3 scripts/performance_envelope.py
"""
import math
import os
import sys

import numpy as np
import yaml


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = os.path.join(ROOT, 'ros2_ws', 'src', 'omni_autonomy_next', 'config')
sys.path.insert(0, os.path.join(ROOT, 'ros2_ws', 'src', 'omni_autonomy_next'))
# The deployed v4 UART encoder has a different cap from the manual gateway's
# omni_max. Import its pure-Python calibration without requiring ROS.
from omni_autonomy_next.robomas_uart import (  # noqa: E402
    AUTO_UNITS_PER_MPS, AUTO_WHEEL_LIMIT, mix_velocity,
)


def load(name):
    with open(os.path.join(CONFIG, name), encoding='utf-8') as stream:
        return yaml.safe_load(stream)


def wheel_cost(vx, vy, wz, positions, angles, radius):
    """この機体速度が要求する各輪速度の最大値[rad/s]。"""
    directions = np.column_stack((np.cos(angles), np.sin(angles)))
    lever = (-positions[:, 1] * directions[:, 0]
             + positions[:, 0] * directions[:, 1])
    surface = directions[:, 0] * vx + directions[:, 1] * vy + lever * wz
    return float(np.max(np.abs(surface / radius)))


def max_scale(vx, vy, wz, positions, angles, radius, maximum):
    cost = wheel_cost(vx, vy, wz, positions, angles, radius)
    return maximum / cost if cost > 1.0e-12 else float('inf')


def trapezoid_time(distance, speed, accel, decel=None):
    """静止から静止まで。加速と制動を分けた理想所要時間[s]。"""
    decel = accel if decel is None else decel
    if (not all(math.isfinite(v) for v in (distance, speed, accel, decel))
            or distance < 0.0 or min(speed, accel, decel) <= 0.0):
        raise ValueError('distance must be nonnegative and speed/ramps positive and finite')
    reciprocal_ramps = 1.0 / accel + 1.0 / decel
    ramp_distance = 0.5 * speed * speed * reciprocal_ramps
    if distance <= ramp_distance:
        peak = math.sqrt(2.0 * distance / reciprocal_ramps)
        return peak * reciprocal_ramps
    return speed * reciprocal_ramps + (distance - ramp_distance) / speed


def profile_envelopes(drive, runtime, smoother):
    """Report each profile through the deployed wheel/guard/encoder limits.

    Acceleration describes the default trajectory tracker; MPPI-only mode
    uses the smoother's separate acceleration. No physical motor model is
    inferred from a software command limit.
    """
    positions = np.asarray(drive['wheel_positions'], dtype=float)
    angles = np.radians(drive['wheel_drive_angles_deg'])
    radius = float(drive['wheel_radius'])
    linear_scale = float(drive.get('linear_command_scale', 1.0))
    uart_axis_limit = AUTO_WHEEL_LIMIT / (linear_scale * AUTO_UNITS_PER_MPS)
    profiles = yaml.safe_load(runtime['profiles_json'])
    result = {}
    for name, limits in profiles.items():
        maximum = float(drive.get('profile_max_wheel_speeds', {}).get(
            name, drive['max_wheel_speed']))
        wheel_axis_limit = max_scale(1.0, 0.0, 0.0, positions, angles, radius, maximum)
        axis = min(float(limits['linear']), wheel_axis_limit,
                   float(runtime['hard_max_linear_speed']), uart_axis_limit)
        acceleration = min(float(drive.get('profile_linear_accelerations', {}).get(
            name, smoother['max_accel'][0])), float(limits['linear_accel']),
            float(runtime['hard_max_linear_acceleration']))
        wheels, saturation = mix_velocity(axis * linear_scale, 0.0, 0.0)
        result[name] = dict(
            wheel_limit=maximum, wheel_axis_limit=wheel_axis_limit,
            axis_speed=axis, acceleration=acceleration,
            deceleration=abs(float(smoother['max_decel'][0])),
            uart_peak=max(map(abs, wheels)), uart_scale=saturation,
        )
    return result


def main():
    robot = load('robot.yaml')['robot']
    drive = robot['drivetrain']
    runtime = load('runtime.yaml')['runtime_guard']['ros__parameters']
    nav2 = load('nav2_next.yaml')
    controller = nav2['controller_server']['ros__parameters']
    smoother = nav2['velocity_smoother']['ros__parameters']

    positions = np.asarray(drive['wheel_positions'], dtype=float)
    angles = np.radians(drive['wheel_drive_angles_deg'])
    radius = float(drive['wheel_radius'])
    profiles = yaml.safe_load(runtime['profiles_json'])
    envelopes = profile_envelopes(drive, runtime, smoother)
    selected = runtime['default_profile']
    profile = profiles[selected]
    envelope = envelopes[selected]
    maximum = envelope['wheel_limit']

    print('=' * 72)
    print('性能エンベロープ（配備値から計算。実機・ROS不要）')
    print('=' * 72)

    print()
    print('--- 1. 既存の車輪モデルが許可する速度 ---')
    print('  選択プロファイル  : %s（下表は車輪予算のみ、速度指令上限は別）' % selected)
    print('  各輪のモデル上限  : %.3f rad/s (= %.3f m/s 接地速度)'
          % (maximum, maximum * radius))
    print('  設定された公称予算。補正後のUART指令は末尾で別途計算する。')
    forward = max_scale(1.0, 0.0, 0.0, positions, angles, radius, maximum)
    lateral = max_scale(0.0, 1.0, 0.0, positions, angles, radius, maximum)
    diagonal = max_scale(
        1 / math.sqrt(2), 1 / math.sqrt(2), 0.0,
        positions, angles, radius, maximum)
    spin = max_scale(0.0, 0.0, 1.0, positions, angles, radius, maximum)
    print('  前進のみ          : %.3f m/s' % forward)
    print('  横のみ            : %.3f m/s' % lateral)
    print('  斜め45度          : %.3f m/s  <- 45度オムニの最悪方向'
          % diagonal)
    print('  その場旋回のみ    : %.3f rad/s (%.0f deg/s)'
          % (spin, math.degrees(spin)))

    print()
    print('--- 2. 並進と旋回の取り合い（同時に出せる組み合わせ）---')
    print('  %-14s %-14s' % ('旋回 [rad/s]', '同時に出せる前進 [m/s]'))
    for yaw in (0.0, 0.2, 0.4, 0.6, 0.9, 1.30):
        yaw_cost = wheel_cost(0.0, 0.0, yaw, positions, angles, radius)
        remaining = maximum - yaw_cost
        if remaining <= 0.0:
            print('  %-14.2f %s' % (yaw, '旋回だけで上限を使い切る'))
            continue
        per_mps = wheel_cost(1.0, 0.0, 0.0, positions, angles, radius)
        print('  %-14.2f %.3f' % (yaw, remaining / per_mps))
    print('  （旋回1 rad/sあたり %.3f rad/s、前進1 m/sあたり %.3f rad/s を消費）'
          % (wheel_cost(0.0, 0.0, 1.0, positions, angles, radius),
             wheel_cost(1.0, 0.0, 0.0, positions, angles, radius)))

    print()
    print('--- 3. 現在の設定値と、その上限に対する余裕 ---')
    print('  profile "%s": 前進 %.3f / 横 %.3f / 旋回 %.2f'
          % (selected, profile['linear'],
             profile['lateral'], profile['angular']))
    print('  車輪モデル上限に対して: 前進 %.0f%% / 横 %.0f%% / 旋回 %.0f%%'
          % (100 * profile['linear'] / forward,
             100 * profile['lateral'] / lateral,
             100 * profile['angular'] / spin))
    print('  追従器の並進加速度 / 制動減速度 [m/s^2]（速度倍率1.0）:')
    for name, limits in envelopes.items():
        print('    %-10s %.2f / %.2f、直進上限 %.3f m/s、車輪予算 %.3f rad/s'
              % (name, limits['acceleration'], limits['deceleration'],
                 limits['axis_speed'], limits['wheel_limit']))
    print('  MPPI専用の加速度 (velocity_smoother): 並進 %.2f m/s^2 / 旋回 %.2f rad/s^2'
          % (smoother['max_accel'][0], smoother['max_accel'][2]))
    speed = envelope['axis_speed']
    braking_distance = speed * speed / (2.0 * envelope['deceleration'])
    ramp_distance = braking_distance + speed * speed / (2.0 * envelope['acceleration'])
    print('  %s %.3f m/sからの理想制動距離: %.3f m（遅延・ジャークを除く）'
          % (selected, speed, braking_distance))
    print('  静止から最高速を経て停止する最短直線距離: %.3f m' % ramp_distance)

    print()
    print('--- 4. 代表経路の所要時間（台形プロファイル・理想追従の下限）---')
    poses = load('field_poses.yaml')['poses']

    def distance(a, b):
        return math.hypot(poses[a]['x'] - poses[b]['x'],
                          poses[a]['y'] - poses[b]['y'])

    legs = [(1, 4), (4, 5), (1, 2), (3, 6)]
    print('  %-8s %-12s %s'
          % ('区間', '直線距離[m]', ' '.join('%-12s' % name for name in envelopes)))
    for a, b in legs:
        d = distance(a, b)
        times = [trapezoid_time(d, limits['axis_speed'], limits['acceleration'],
                               limits['deceleration']) for limits in envelopes.values()]
        print('  %-8s %-12.2f %s'
              % ('%d->%d' % (a, b), d, ' '.join('%-12.2f' % t for t in times)))
    print('  ※ 秒単位。車体前進方向に直線走行・停止する理想下限。')
    print('     障害物回避・斜行・旋回・通信遅れ・ジャーク・終端整定を含まない。')

    print()
    print('--- 5. 到達精度を決めているもの ---')
    goal = controller['goal_checker']
    print('  goal_checker の受入   : 位置 %.0f mm / 姿勢 %.1f deg'
          % (1000 * goal['xy_goal_tolerance'],
             math.degrees(goal['yaw_goal_tolerance'])))
    print('    -> これは「合格とみなす閾値」であって到達精度ではない。')
    print('       閾値を小さくしても、下の要因が改善しなければ収束しない。')
    print('  計測輪の校正行列だけから、実際の位置推定誤差は確定できない。')
    localization = load('localization.yaml')['wall_localizer']['ros__parameters']
    print('  LiDAR自己位置の更新   : %.1f Hz、1回あたり最大 %.0f mm の補正'
          % (localization['update_rate_hz'],
             1000 * localization['max_lidar_correction_translation']))
    print('    -> 補正できる速度の上限は %.2f m/s。これを超える推定誤差の'
          % (localization['update_rate_hz']
             * localization['max_lidar_correction_translation']))
    print('       増加率には原理的に追いつけない。')
    lead = robot.get('calibrated_tracking', {}).get('feedback_delay_sec', .2)
    print('  補正済み追従の先読み : %.2f s（調整値。実測遅延ではない）' % lead)
    print('  制御周期              : %.0f Hz (MPPI)'
          % controller['controller_frequency'])
    print('    -> 1周期で %.0f mm 進む'
          % (1000 * profile['linear'] / controller['controller_frequency']))

    print()
    print('--- 6. 送信補正後の指令と、未測定の上限 ---')
    linear_scale = float(drive.get('linear_command_scale', 1.))
    angular_scale = float(drive.get('angular_command_scale', 1.))
    print('  送信補正: 並進 %.3f / 旋回 %.3f' % (linear_scale, angular_scale))
    for name, limits in envelopes.items():
        command = limits['uart_peak']
        print('  %s 前進: %.3f m/s -> 車輪指令 %d / %d (%.1f%%)、飽和倍率 %.3f'
              % (name, limits['axis_speed'], command, AUTO_WHEEL_LIMIT,
                 100*command/AUTO_WHEEL_LIMIT, limits['uart_scale']))
    print('  車輪モデル上限の使用率と、補正後UART指令の使用率は異なる。')
    print('  指令単位はRPMではない。実モーターの限界には回転数・電流・温度の実測が必要。')
    print()
    return 0


if __name__ == '__main__':
    sys.exit(main())
