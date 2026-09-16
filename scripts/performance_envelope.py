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
import re
import sys

import numpy as np
import yaml


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = os.path.join(ROOT, 'ros2_ws', 'src', 'omni_autonomy_next', 'config')
VALUE_HPP = os.path.join(ROOT, 'bacon_gateway', 'include', 'value.hpp')


def load(name):
    with open(os.path.join(CONFIG, name), encoding='utf-8') as stream:
        return yaml.safe_load(stream)


def hpp_constant(name, default):
    try:
        text = open(VALUE_HPP, encoding='utf-8').read()
    except OSError:
        return default
    match = re.search(
        r'constexpr\s+(?:double|int)\s+%s\s*=\s*(-?[0-9.]+)' % name, text)
    return float(match.group(1)) if match else default


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


def trapezoid_time(distance, speed, accel):
    """台形/三角速度プロファイルでの所要時間[s]。"""
    ramp_distance = speed * speed / accel
    if distance <= ramp_distance:
        return 2.0 * math.sqrt(distance / accel)
    return speed / accel + (distance - ramp_distance) / speed + speed / accel


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
    maximum = float(drive['max_wheel_speed'])

    units_per_mps = hpp_constant('auto_units_per_mps', 7202.0)
    wheel_limit = hpp_constant('omni_max', 8000.0)

    print('=' * 72)
    print('性能エンベロープ（配備値から計算。実機・ROS不要）')
    print('=' * 72)

    print()
    print('--- 1. 既存の車輪モデルが許可する速度 ---')
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
    profile = yaml.safe_load(runtime['profiles_json'])[runtime['default_profile']]
    print('  profile "%s": 前進 %.3f / 横 %.3f / 旋回 %.2f'
          % (runtime['default_profile'], profile['linear'],
             profile['lateral'], profile['angular']))
    print('  車輪モデル上限に対して: 前進 %.0f%% / 横 %.0f%% / 旋回 %.0f%%'
          % (100 * profile['linear'] / forward,
             100 * profile['lateral'] / lateral,
             100 * profile['angular'] / spin))
    print('  加速度 (velocity_smoother): 並進 %.2f m/s^2 / 旋回 %.2f rad/s^2'
          % (smoother['max_accel'][0], smoother['max_accel'][2]))

    print()
    print('--- 4. 代表経路の所要時間（台形プロファイル・理想追従の下限）---')
    poses = load('field_poses.yaml')['poses']

    def distance(a, b):
        return math.hypot(poses[a]['x'] - poses[b]['x'],
                          poses[a]['y'] - poses[b]['y'])

    legs = [(1, 4), (4, 5), (1, 2), (3, 6)]
    accel = float(smoother['max_accel'][0])
    print('  %-8s %-8s %-12s %-12s'
          % ('区間', '直線距離[m]', '現在%.2f' % profile['linear'], 'モデル上限%.2f' % forward))
    for a, b in legs:
        d = distance(a, b)
        print('  %-8s %-8.2f %-12.2f %-12.2f'
              % ('%d->%d' % (a, b), d,
                 trapezoid_time(d, profile['linear'], accel),
                 trapezoid_time(d, forward, accel)))
    print('  ※ 障害物回避・旋回・通信遅れ・終端整定を含まない理想下限。')

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
    profiles = yaml.safe_load(runtime['profiles_json'])
    for name, limits in profiles.items():
        axis = min(limits['linear'], forward, runtime['hard_max_linear_speed'])
        command = axis * linear_scale * units_per_mps
        print('  %s 前進: %.3f m/s -> 車輪指令 約%.0f / %.0f (%.1f%%)'
              % (name, axis, command, wheel_limit, 100*command/wheel_limit))
    print('  車輪モデル上限の使用率と、補正後UART指令の使用率は異なる。')
    print('  指令単位はRPMではない。実モーターの限界には回転数・電流・温度の実測が必要。')
    print()
    return 0


if __name__ == '__main__':
    sys.exit(main())
