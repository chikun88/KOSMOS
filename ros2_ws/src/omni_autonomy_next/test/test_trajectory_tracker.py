"""速度プロファイル生成と、追従器モードの結線を固定する。

追従器は MPPI と同じ20 Hzで時間パラメータ化したフィードフォワードを
置き換えるが、停止権限は一つも移していない。出力は velocity_smoother と
同じトピックなので rl_policy -> collision_monitor -> runtime_guard の
安全鎖はそのまま通る。ここではその結線と、軌道が物理制約を守ることを
確認する。
"""
import json
import math
import re
from pathlib import Path

import numpy as np
import pytest
import yaml

from omni_autonomy_next.config import load_collision_monitor_horizon
from omni_autonomy_next.configured_goals import load_configured_poses
from omni_autonomy_next.omni_yaw import (
    direction_speed_limits, plan_yaw_profile, smooth_yaw_profile,
    yaw_candidates, yaw_derivative,
)
from omni_autonomy_next.rl_residual import CadClearanceModel, limit_yaw_rate
from omni_autonomy_next.trajectory_tracker_node import (
    Trajectory, direction_gap, entry_time, menger_curvature, path_tangents,
    resample, same_endpoint, smooth_path, terminal_latch, to_body,
    track_progress, wrap,
)

CONFIG = Path(__file__).resolve().parents[1] / 'config'
LAUNCH = Path(__file__).resolve().parents[1] / 'launch'
# ros2_ws/src/omni_autonomy_next/test -> リポジトリ直下
ROOT = Path(__file__).resolve().parents[4]


def straight(length=4.0, spacing=0.05):
    count = int(length / spacing) + 1
    return np.column_stack((np.linspace(0.0, length, count), np.zeros(count)))


def build(points, **overrides):
    """Trajectory は点ごとの速度上限と姿勢列を受け取る。

    速度上限が配列なのは、45 度オムニでは出せる速度が進行方向と姿勢の
    関数だからで、区間ごとに違う。ここでは一定値と線形の姿勢配分で、
    従来と同じ条件を作る。
    """
    speed_limit = overrides.pop('speed_limit', 0.78)
    yaw_start = overrides.pop('yaw_start', 0.0)
    yaw_goal = overrides.pop('yaw_goal', 0.0)
    settings = dict(acceleration=0.85, lateral_acceleration=1.2, entry_speed=0.0)
    settings.update(overrides)
    segments = np.linalg.norm(np.diff(points, axis=0), axis=1)
    arclength = np.concatenate(([0.0], np.cumsum(segments)))
    fraction = arclength / max(float(arclength[-1]), 1.0e-9)
    yaws = yaw_start + wrap(yaw_goal - yaw_start) * fraction
    limits = np.full(len(points), float(speed_limit))
    return Trajectory(points, yaws, limits, **settings)


def test_profile_respects_speed_and_acceleration_limits():
    trajectory = build(straight())
    assert np.max(trajectory.speed) <= 0.78 + 1.0e-9
    # 終端は必ず0。ここが0でないと goal_checker の手前で行き過ぎる。
    assert trajectory.speed[-1] == pytest.approx(0.0)
    segments = np.linalg.norm(np.diff(trajectory.points, axis=0), axis=1)
    implied = np.abs(np.diff(trajectory.speed ** 2)) / (2.0 * segments)
    assert np.max(implied) <= 0.85 + 1.0e-6
    # 4 m の直線なら台形になり、上限速度に達している。
    assert np.max(trajectory.speed) == pytest.approx(0.78, abs=1.0e-3)


def test_profile_starts_from_the_current_speed():
    """走行中の再計画で速度を落とさないこと。

    毎回 0 から作り直すと、Nav2 が経路を出すたびに減速して結局遅くなる。
    """
    moving = build(straight(), entry_speed=0.5)
    assert moving.speed[0] == pytest.approx(0.5)
    stopped = build(straight(), entry_speed=0.0)
    assert stopped.speed[0] == pytest.approx(0.0)
    assert moving.duration < stopped.duration


def test_curvature_limits_speed_through_a_corner():
    angle = np.linspace(0.0, math.pi / 2.0, 120)
    radius = 0.5
    arc = np.column_stack((radius * np.sin(angle), radius * (1 - np.cos(angle))))
    trajectory = build(resample(arc, 0.02))
    # v = sqrt(a_lat / kappa) = sqrt(1.2 * 0.5) = 0.775 -> ほぼ上限だが、
    # 半径を締めれば確実に下回る。
    tight = build(resample(np.column_stack((
        0.15 * np.sin(angle), 0.15 * (1 - np.cos(angle)))), 0.01))
    assert np.max(tight.speed) < 0.78
    assert np.max(tight.speed) == pytest.approx(math.sqrt(1.2 * 0.15), abs=0.05)
    assert trajectory.duration > 0.0


def test_sample_is_monotonic_and_ends_at_the_goal():
    trajectory = build(straight(3.0))
    previous = -1.0
    for fraction in np.linspace(0.0, 1.0, 40):
        _, _, _, _, s = trajectory.sample(fraction * trajectory.duration)
        assert s >= previous - 1.0e-9
        previous = s
    position, velocity, _, _, s = trajectory.sample(trajectory.duration)
    assert position[0] == pytest.approx(3.0, abs=1.0e-3)
    assert np.linalg.norm(velocity) == pytest.approx(0.0, abs=1.0e-6)
    # 終端を過ぎた時刻を引いても飛ばない。
    assert trajectory.sample(trajectory.duration * 5.0)[4] == pytest.approx(s)


def test_yaw_is_retired_along_the_path_not_at_the_end():
    trajectory = build(straight(), yaw_start=0.0, yaw_goal=1.2)
    _, _, half_yaw, _, _ = trajectory.sample(0.5 * trajectory.duration)
    _, _, end_yaw, _, _ = trajectory.sample(trajectory.duration)
    assert 0.2 < half_yaw < 1.0
    assert end_yaw == pytest.approx(1.2, abs=1.0e-3)


def test_projection_finds_the_travelled_arc_length():
    trajectory = build(straight())
    assert trajectory.project(np.array([2.0, 0.3])) == pytest.approx(2.0, abs=0.06)


def test_replanning_keeps_the_reference_on_the_robot():
    """1 Hz の経路差し替えで基準が後ろへ飛び戻らないこと。

    BT の replanning は 1 Hz で、経路を計画して姿勢を最適化する間に機体は
    進む。差し替えのたびに基準時刻を 0 に戻すと、基準はその古い位置から
    やり直すので位置誤差が後ろ向きに残り、P項が制動をかける。基準は実時間で
    進むだけなので追いつけず、次の経路でまた飛び戻る。これが毎秒の脈動と、
    曲がっている区間ではその横成分（左右の振れ）になっていた。
    """
    trajectory = build(straight(4.0), entry_speed=0.5)
    # 機体は経路上 1.8 m 地点にいる。差し替え直後の基準はそこに乗る。
    pose = np.array([1.8, 0.0, 0.0])
    started = entry_time(trajectory, pose)
    assert started > 0.0
    reference, _, _, _, s = trajectory.sample(started)
    assert s == pytest.approx(1.8, abs=0.06)
    assert np.linalg.norm(reference - pose[:2]) < 0.06
    # 経路が計画された時点の位置（先頭）に置き直すと、この距離ぶんまるごと
    # 後ろ向きの誤差になる。
    assert np.linalg.norm(trajectory.points[0] - pose[:2]) > 1.7
    # 自己位置がまだ無いときは従来どおり先頭から。
    assert entry_time(trajectory, None) == 0.0


def test_command_is_expressed_in_the_yaw_the_robot_will_have():
    """指令の回転には、いまの姿勢ではなく lag 後の姿勢を使うこと。"""
    lag, yaw_rate = 0.20, 0.8
    pose_yaw = 0.0
    predicted_yaw = pose_yaw + yaw_rate * lag
    wanted = np.array([0.55, 0.0])  # 世界系でまっすぐ前へ出したい

    command = to_body(wanted, predicted_yaw)
    # 指令が効くころの姿勢で機体系→世界系へ戻すと、狙った方向に一致する。
    cosine, sine = math.cos(predicted_yaw), math.sin(predicted_yaw)
    realised = np.array([
        command[0] * cosine - command[1] * sine,
        command[0] * sine + command[1] * cosine,
    ])
    assert realised == pytest.approx(wanted, abs=1.0e-9)

    # 現在姿勢で回すと、同じ瞬間に yaw_rate * lag ぶん横へ抜ける。
    stale = to_body(wanted, pose_yaw)
    realised_stale = np.array([
        stale[0] * cosine - stale[1] * sine,
        stale[0] * sine + stale[1] * cosine,
    ])
    lateral = abs(float(np.cross(wanted / np.linalg.norm(wanted), realised_stale)))
    assert lateral == pytest.approx(0.55 * math.sin(yaw_rate * lag), abs=1.0e-9)
    assert lateral > 0.08


def test_a_command_frame_error_makes_a_circle_the_watchdog_catches():
    """指令の座標系がずれると目標へ収束せず円を描く。それを検出すること。

    位置のP制御は本来 xdot = -K (x - goal) で目標へまっすぐ寄る。指令を
    機体系へ落とす回転が実際の向きから theta ずれていると、実際に出るのは
    -K R(theta) (x - goal) になり、theta が 90 度に近いほど誤差は縮まず
    横へ回る。theta = 90 度でちょうど円になり、永遠に回り続ける。
    """
    goal = np.zeros(2)
    position = np.array([0.6, 0.0])
    theta = math.pi / 2.0
    rotation = np.array([
        [math.cos(theta), -math.sin(theta)],
        [math.sin(theta), math.cos(theta)],
    ])
    best, best_at, stalled = math.inf, 0.0, 0.0
    for step in range(400):
        now = 0.01 * step
        velocity = -2.4 * rotation.dot(position - goal)
        position = position + velocity * 0.01
        best, best_at, stalled = track_progress(
            float(np.linalg.norm(position - goal)), best, best_at, now, 0.02)
    # 4 秒回っても目標には一度も近づいていない（＝円）。
    assert float(np.linalg.norm(position - goal)) >= 0.6
    assert best >= 0.6
    assert stalled == pytest.approx(3.99, abs=0.02)

    # まっすぐ寄っている間は一度も停滞しない。
    position = np.array([0.6, 0.0])
    best, best_at, stalled = math.inf, 0.0, 0.0
    for step in range(400):
        now = 0.01 * step
        position = position - 2.4 * (position - goal) * 0.01
        best, best_at, stalled = track_progress(
            float(np.linalg.norm(position - goal)), best, best_at, now, 0.02)
        assert stalled < 4.0


def test_the_watchdog_survives_the_one_hertz_replanning():
    """1 Hz の経路差し替えで進捗の記録を捨てないこと。

    捨てると停滞時間は最大 1 秒しか積まれず、no_progress_timeout_sec には
    決して届かない。円を描いている間も BT は経路を出し続けているので、
    捕まえたい状況そのものが検出できなくなる。
    """
    goal = np.zeros(2)
    position = np.array([0.6, 0.0])
    theta = math.pi / 2.0
    rotation = np.array([
        [math.cos(theta), -math.sin(theta)],
        [math.sin(theta), math.cos(theta)],
    ])
    best, best_at, stalled = math.inf, 0.0, 0.0
    endpoint = None
    for step in range(600):
        now = 0.01 * step
        if step % 100 == 0:  # BT の replanning。終点は同じ目標のまま。
            if not same_endpoint(endpoint, goal):
                endpoint, best, best_at = goal.copy(), math.inf, now
        velocity = -2.4 * rotation.dot(position - goal)
        position = position + velocity * 0.01
        best, best_at, stalled = track_progress(
            float(np.linalg.norm(position - goal)), best, best_at, now, 0.02)
    assert stalled > 4.0

    # 次の目標が来たら記録は捨てる（そこからの停滞をあらためて測る）。
    assert same_endpoint(goal, np.array([0.01, 0.0]))
    assert not same_endpoint(goal, np.array([0.4, 0.0]))
    assert not same_endpoint(None, goal)


def test_the_terminal_zone_orbit_is_caught_and_survives_replanning():
    """終端ゾーンの中を回り続ける機体を、時間切れで止めること。

    停滞判定はゾーンの中を除外しているので、そこで破綻するとどこにも
    引っかからない。しかも 1 Hz の経路差し替えが掛け金を戻していたので、
    「目標のまわりを小さく回り続けて永久に止まらない」だけが素通りしていた。

    ゾーン 0.16 m のまわりを半径 0.05 m で回る機体を置くと、残り弧長は
    境界をまたいで往復する。境界で掛け金を外す実装だと周回ごとに戻って
    しまうので、時間切れは成立しない。
    """
    zone, timeout = 0.16, 4.0
    since, fired_at = None, None
    endpoint = None
    goal = np.zeros(2)
    for step in range(1200):
        now = 0.01 * step
        if step % 100 == 0 and not same_endpoint(endpoint, goal):
            endpoint, since = goal.copy(), None   # 目標が変わったときだけ外す
        # 境界をまたいで往復する残り弧長（周期 2 秒の周回）。
        remaining = zone + 0.05 * math.sin(2.0 * math.pi * now / 2.0)
        since = terminal_latch(since, remaining, zone, now)
        if since is not None and now - since > timeout and fired_at is None:
            fired_at = now
    assert fired_at is not None and fired_at < 6.0

    # 正常な整定（ゾーンへ入って 1 秒で公差の中）は切らない。
    since = None
    for step in range(200):
        now = 0.01 * step
        since = terminal_latch(since, max(0.0, 0.16 - 0.16 * now), zone, now)
        assert since is None or now - since <= timeout

    # ゾーンを明確に出たら外れる（次の目標へ向かう走行を切らない）。
    assert terminal_latch(3.0, 0.9, zone, 9.0) is None
    # 境界のすぐ外では保つ。
    assert terminal_latch(3.0, 0.2, zone, 9.0) == 3.0


def test_the_command_frame_error_shows_up_as_an_angle_in_the_status():
    """指令と実測の進行方向のずれを、そのまま角度で出せること。

    円を描く原因は指令の座標系のずれだが、走行ログからは「寄らない」しか
    見えなかった。角度で出しておけば疑う符号がその場で決まる。
    """
    assert direction_gap([0.5, 0.0], [0.5, 0.0]) == pytest.approx(0.0)
    # ミキサの x/y 入れ替え -> 90 度
    assert direction_gap([0.5, 0.0], [0.0, 0.5]) == pytest.approx(90.0)
    # 符号反転 -> 180 度
    assert abs(direction_gap([0.5, 0.0], [-0.5, 0.0])) == pytest.approx(180.0)
    # 向きが定まらないほど遅いときは判定しない。
    assert direction_gap([0.5, 0.0], [0.001, 0.0]) is None
    assert direction_gap([0.0, 0.0], [0.5, 0.0]) is None


def test_going_around_a_wall_is_not_mistaken_for_a_stall():
    """壁を回り込む経路では、終点までの直線距離は増える区間がある。

    そこで直線距離を停滞の指標にすると、正常な走行を停滞と誤判定して止めて
    しまう。指標は残りの弧長でなければならない。
    """
    # 目標 (0.2, 1.4) との間に壁があり、右へ 1.5 m 出て回り込んで戻る経路。
    # 最初の脚では目標から遠ざかっていく。
    detour = np.array(
        [[0.05 * i, 0.0] for i in range(0, 31)]
        + [[1.5, 0.05 * i] for i in range(1, 29)]
        + [[1.5 - 0.05 * i, 1.4] for i in range(1, 27)]
    )
    trajectory = Trajectory(
        detour, np.zeros(len(detour)), np.full(len(detour), 0.5),
        acceleration=0.85, lateral_acceleration=1.0, entry_speed=0.0)
    goal = trajectory.points[-1]

    euclid_best, euclid_at, euclid_stalled = math.inf, 0.0, 0.0
    arc_best, arc_at, arc_stalled = math.inf, 0.0, 0.0
    worst_euclid = 0.0
    for step in range(int(trajectory.duration / 0.01)):
        now = 0.01 * step
        position = trajectory.sample(min(now, trajectory.duration))[0]
        euclid_best, euclid_at, euclid_stalled = track_progress(
            float(np.linalg.norm(goal - position)),
            euclid_best, euclid_at, now, 0.02)
        worst_euclid = max(worst_euclid, euclid_stalled)
        arc_best, arc_at, arc_stalled = track_progress(
            float(trajectory.length - trajectory.project(position)),
            arc_best, arc_at, now, 0.02)
        assert arc_stalled < 1.0
    # 直線距離では実際に長時間「寄っていない」ように見える。
    assert worst_euclid > 1.0


def test_resample_and_curvature_handle_degenerate_input():
    assert len(resample(np.zeros((1, 2)), 0.05)) == 1
    assert np.all(menger_curvature(np.zeros((2, 2))) == 0.0)
    assert wrap(3.5 * math.pi) == pytest.approx(-0.5 * math.pi)


def test_tracker_uses_the_shaping_stage_acceleration_not_the_guard_headroom():
    """軌道は velocity_smoother の加速度で作らなければならない。

    RuntimeGuard の profile 値はゲート用に上へ振ってあるので、そちらで
    時間割りを作ると実際には出せない軌道を追いかけることになる。
    """
    source = (
        Path(__file__).resolve().parents[1]
        / 'omni_autonomy_next' / 'trajectory_tracker_node.py'
    ).read_text(encoding='utf-8')
    assert "smoother['max_accel'][0]" in source
    assert "smoother['max_accel'][2]" in source
    nav2 = yaml.safe_load((CONFIG / 'nav2_next.yaml').read_text(encoding='utf-8'))
    runtime = yaml.safe_load(
        (CONFIG / 'runtime.yaml').read_text(encoding='utf-8')
    )['runtime_guard']['ros__parameters']
    profile = json.loads(runtime['profiles_json'])[runtime['default_profile']]
    smoother = nav2['velocity_smoother']['ros__parameters']
    assert smoother['max_accel'][0] < profile['linear_accel']


def test_tracker_mode_keeps_every_safety_gate_and_has_one_publisher():
    system = (LAUNCH / 'system.launch.py').read_text(encoding='utf-8')
    navigation = (LAUNCH / 'navigation.launch.py').read_text(encoding='utf-8')
    # 既定は追従器（2026-08-07 16:59 の構成）。MPPI は tracker:=false を
    # 明示したときだけ走る。優劣がデモでしか測れていないことは
    # system.launch.py の引数コメントに残してある。
    assert "DeclareLaunchArgument('tracker', default_value='true')" in system
    assert "executable='trajectory_tracker'" in system
    # 追従器の出力は整形段と同じトピック。以降の鎖は変わらない。
    assert "'output_topic': '/cmd_vel_nav_smoothed'" in system
    monitor = yaml.safe_load(
        (CONFIG / 'nav2_next.yaml').read_text(encoding='utf-8')
    )['collision_monitor']['ros__parameters']
    assert monitor['cmd_vel_in_topic'] == '/cmd_vel_rl'
    assert monitor['cmd_vel_out_topic'] == '/cmd_vel_collision_safe'
    # 追従器モードでは整形段の出力を行き止まりにして、同じトピックに
    # 二つの publisher が乗らないようにする。
    assert "'cmd_vel_smoothed_idle'" in navigation
    assert "'cmd_vel_mppi_idle'" in navigation
    assert navigation.count('IfCondition(tracker)') >= 2
    assert navigation.count('UnlessCondition(tracker)') >= 2


def test_tracker_rate_does_not_flood_the_collision_monitor():
    """固定バケツ脇の swept-footprint 判定へ 100 Hz を流し込まないこと。

    100 Hz では collision_monitor の出力が最大 2.2 秒途切れ、0.25 秒の
    RuntimeGuard watchdog が停止させて地点4/5への進捗が失われた。
    30 Hzでも固定バケツ付近の連続判定中に0.47〜0.68秒の途切れが残った。
    Nav2 controllerと同じ20 Hzなら追従帯域を保ったまま判定量を1/3減らす。
    """
    tracker = (Path(__file__).resolve().parents[1] / 'omni_autonomy_next'
               / 'trajectory_tracker_node.py').read_text(encoding='utf-8')
    assert "'control_rate_hz': 20.0," in tracker


def test_the_launcher_can_actually_ask_for_the_tracker():
    """run.py が tracker を渡さなければ、追従器は存在しないのと同じ。

    既定を off に戻すかどうかとは別の話で、ランチャが引数を持たない限り
    比較測定すら運用手順からは踏めない。追従器が走った 16 回はすべて手で
    ros2 launch を叩いたものだった。
    """
    launcher = (ROOT / 'run.py').read_text(encoding='utf-8')
    assert 'tracker:=' in launcher
    assert '--no-tracker' in launcher


def test_the_launcher_does_not_override_the_measured_navigation_delay():
    """run.py が渡すフォールバック期限は launch 側と一致させる。

    通常は局在器の準備完了で前倒し起動する。26.0 はその出力を取得できない
    ときの安全側期限であり、run.py が別の値を毎回渡すと launch 側で決めた
    フォールバックが運用に届かない。
    """
    system = (LAUNCH / 'system.launch.py').read_text(encoding='utf-8')
    declared = re.search(
        r"DeclareLaunchArgument\('navigation_delay', default_value='([^']+)'\)",
        system,
    )
    assert declared is not None
    launcher = (ROOT / 'run.py').read_text(encoding='utf-8')
    assert f'default={declared.group(1)},' in launcher


def test_configured_profiles_plan_inside_the_wheel_budget():
    """An axis cap may exceed the inscribed ellipse; every planned direction
    still intersects the unchanged wheel budget, including concurrent yaw.
    """
    from omni_autonomy_next.omni_yaw import OmniEnvelope
    from omni_autonomy_next.motor_kinematics import allocate_omni4_wheel_budget
    drive = yaml.safe_load((CONFIG / 'robot.yaml').read_text())['robot']['drivetrain']
    envelope = OmniEnvelope(drive)
    runtime = yaml.safe_load((CONFIG / 'runtime.yaml').read_text())['runtime_guard']['ros__parameters']
    for name, profile in json.loads(runtime['profiles_json']).items():
        assert profile['linear'] <= runtime['hard_max_linear_speed']
        assert profile['lateral'] <= runtime['hard_max_lateral_speed']
        for theta in np.linspace(-math.pi, math.pi, 73):
            direction = np.array([math.cos(theta), math.sin(theta)])
            for yaw_per_metre in [-2., 0., 2.]:
                speed = envelope.speed_limit(direction, yaw_per_metre,
                    profile['linear'], profile['lateral'])
                twist = np.array([*(direction * speed), yaw_per_metre * speed])
                assert envelope.wheel_cost(*twist) <= envelope.max_wheel + 1.e-8, name
                assert math.hypot(twist[0]/profile['linear'], twist[1]/profile['lateral']) <= 1. + 1.e-8
                # Feedback can exceed the plan. The independent output
                # allocator must retain curvature while enforcing the limit.
                request = twist * 2.
                safe = np.asarray(allocate_omni4_wheel_budget(*request,
                    wheel_radius=drive['wheel_radius'],
                    wheel_positions=drive['wheel_positions'],
                    wheel_drive_angles=np.radians(drive['wheel_drive_angles_deg']),
                    wheel_signs=drive['wheel_signs'], maximum=drive['max_wheel_speed']))
                assert envelope.wheel_cost(*safe) <= envelope.max_wheel + 1.e-8
                assert np.linalg.norm(np.cross(request, safe)) < 1.e-8


def test_omni_envelope_reports_the_direction_dependent_limit():
    from omni_autonomy_next.omni_yaw import OmniEnvelope
    robot = yaml.safe_load(
        (CONFIG / 'robot.yaml').read_text(encoding='utf-8'))['robot']
    envelope = OmniEnvelope(robot['drivetrain'])
    wide = 9.9  # プロファイルを外して駆動系だけを見る
    axis = envelope.speed_limit(np.array([1.0, 0.0]), 0.0, wide, wide)
    diagonal = envelope.speed_limit(
        np.array([1.0, 1.0]) / math.sqrt(2.0), 0.0, wide, wide)
    assert axis == pytest.approx(2.0, abs=1.0e-3)
    assert diagonal == pytest.approx(2.0 / math.sqrt(2.0), abs=1.0e-3)
    # 45 度方向は体軸の 1/sqrt(2)。ここを一定値だと思うと、斜め区間で
    # 出せない速度を計画してしまう。
    assert diagonal == pytest.approx(axis / math.sqrt(2.0), rel=1.0e-3)
    # 旋回はバジェットを食う。同じ方向でも旋回を足せば上限は下がる。
    with_yaw = envelope.speed_limit(np.array([1.0, 0.0]), 1.0, wide, wide)
    assert with_yaw < axis


def test_yaw_optimisation_is_off_by_default_in_the_launch():
    """既定は無効。現行プロファイルでは速くも滑らかにもならない。

    体軸へ姿勢を合わせる利得は駆動系の菱形では 1.41 倍あるが、運用
    プロファイル 0.78 の円の中では最大 11% しかなく、その 11% は姿勢を
    回すのに使う車輪バジェットと釣り合ってしまう。菱形に内接する円の
    半径は 0.786 なので、balanced の 0.78 はすでに「どの方向へも速い」
    プロファイルの天井である。

    2026-09-01 に同じ機体モデルで 4 区間を測り直した結果（到着時刻、
    既定の線形配分 -> 姿勢最適化）: 6.42->6.45 / 4.99->5.09 /
    6.61->6.61 / 5.56->5.58 s。要求角加速度は 2.5-3.9 -> 17.8-27.5
    rad/s^2 で、滑らかさでも負ける。原因は離散最適化を 1 Hz の再計画の
    中で回していること自体で、再計画のたびに別の枝へ飛ぶと基準角速度が
    段で変わる。

    それでも実現可能性の修正（候補間隔・時間最小化・平滑化・角加速度
    上限）は入れてある。以前この旗を立てると 105 rad/s^2 を要求する計画が
    出ており、罠だった。プロファイルを菱形へ寄せられたときに初めて意味を
    持つ。
    """
    system = (LAUNCH / 'system.launch.py').read_text(encoding='utf-8')
    assert "DeclareLaunchArgument('optimize_yaw', default_value='false')" in system
    # ノード既定も同じであること。食い違っていると、このノードを直接
    # 起動して測ったときに launch とは別のものを測る。
    node = (Path(__file__).resolve().parents[1] / 'omni_autonomy_next'
            / 'trajectory_tracker_node.py').read_text(encoding='utf-8')
    assert "'optimize_yaw': False," in node


def grid_leg(start, goal, cell=0.05):
    """SmacPlanner2D が返すのと同じ、セル中心に量子化された折れ線。"""
    start, goal = np.asarray(start, float), np.asarray(goal, float)
    count = max(2, int(np.linalg.norm(goal - start) / cell) + 1)
    raw = start + np.linspace(0.0, 1.0, count)[:, None] * (goal - start)
    return np.round(raw / cell) * cell


def heading_swing(points):
    tangents = path_tangents(points)
    headings = np.degrees(np.arctan2(tangents[:, 1], tangents[:, 0]))
    return float(headings.max() - headings.min())


def test_the_planner_staircase_would_command_a_weave_if_not_smoothed():
    """平滑化前の接線は 8 近傍グリッドの 45 度をそのまま振る。

    これが蛇行の出どころだった。planner のセルは 0.05 m で、
    resample_spacing_m も 0.05 m なので階段はそのまま残り、中心差分が
    それを拾う。0.78 m/s では横方向に 0.5 m/s 級の指令になる。
    軸に乗った 4->5 区間だけは階段が出ないので、その往復しか走って
    いなかった合成デモでは一度も現れなかった。
    """
    diagonal = resample(grid_leg((-4.19, 4.69), (-0.81, 1.34)), 0.05)
    assert heading_swing(diagonal) > 40.0

    axis_aligned = resample(grid_leg((-0.81, 1.34), (-0.81, -2.31)), 0.05)
    assert heading_swing(axis_aligned) == pytest.approx(0.0, abs=1.0e-6)


def test_smoothing_removes_the_staircase_on_every_diagonal_leg():
    for start, goal in [((-4.19, 4.69), (-0.81, 1.34)),
                        ((-3.88, -0.55), (-0.81, 0.23)),
                        ((-1.80, 4.75), (-0.81, 1.34))]:
        points = resample(grid_leg(start, goal), 0.05)
        smoothed = smooth_path(points, 0.05, 0.12)
        assert heading_swing(points) > 40.0
        assert heading_swing(smoothed) < 12.0


def test_smoothing_never_moves_the_endpoints():
    """始点は機体位置、終点は goal_bridge の実目標へスナップ済み。

    どちらかが動くと、地点4〜7の 57〜59 mm しかない余裕の中で
    狙う場所がずれる。
    """
    points = resample(grid_leg((-4.19, 4.69), (-0.81, 1.34)), 0.05)
    smoothed = smooth_path(points, 0.05, 0.12)
    assert smoothed[0] == pytest.approx(points[0])
    assert smoothed[-1] == pytest.approx(points[-1])


def test_smoothing_cannot_cut_a_real_corner_by_more_than_half_a_cell():
    """消してよいのは量子化ぶんだけ。実在する角は削らない。

    制限が無いと、壁を回り込む 90 度の角で内側を 49 mm 通る。地点4〜7の
    余裕は 57〜59 mm しかないので、それは接触しうる量である。
    """
    corner = np.vstack((
        np.column_stack((np.full(40, -2.0), np.linspace(2.0, 0.05, 40))),
        np.column_stack((np.linspace(-1.95, 0.0, 40), np.zeros(40))),
    ))
    points = resample(np.round(corner / 0.05) * 0.05, 0.05)
    smoothed = smooth_path(points, 0.05, 0.12)
    assert np.max(np.linalg.norm(smoothed - points, axis=1)) <= 0.025 + 1.0e-9


def test_smoothing_is_disabled_by_a_zero_length_and_keeps_short_paths():
    points = resample(grid_leg((-4.19, 4.69), (-0.81, 1.34)), 0.05)
    assert smooth_path(points, 0.05, 0.0) is points
    stub = np.array([[0.0, 0.0], [0.05, 0.0]])
    assert smooth_path(stub, 0.05, 0.12) is stub


def test_the_reference_is_read_at_the_same_instant_the_state_is_predicted():
    """状態を lag 先へ送るなら、基準も lag 先から取る。

    片方だけ進めると、進行方向へ v*lag だけ常に行き過ぎて見えるので、
    P項が走行中ずっと制動をかける。0.7 m/s で 32 mm の定常追従遅れに
    なっていた。
    """
    source = (Path(__file__).resolve().parents[1] / 'omni_autonomy_next'
              / 'trajectory_tracker_node.py').read_text(encoding='utf-8')
    # 基準時刻はゲートが通す倍率 scale で進むので、実時間 lag 秒の先読みは
    # 基準時刻では lag * scale である。
    assert (
        'sample_time = min(reference_time + lag * scale, trajectory.duration)'
        in source)
    assert 'reference = trajectory.sample(sample_time)' in source
    assert 'trajectory.sample(reference_time)' not in source


def test_the_measured_velocity_is_filtered_before_it_reaches_the_command():
    """生の1サンプル差分はゲイン約0.5で指令に入る。

    position_gain*feedback_delay=0.48、position_damping=0.25、
    yaw_gain*feedback_delay=0.32。measurement_wheel は twist を
    body_delta/dt でそのまま出しているので、ここで均さないと
    エンコーダの微分雑音がほぼ等倍で指令に乗る。
    """
    source = (Path(__file__).resolve().parents[1] / 'omni_autonomy_next'
              / 'trajectory_tracker_node.py').read_text(encoding='utf-8')
    assert "'velocity_filter_sec': 0.06," in source
    assert 'math.exp(-dt / tau)) * (sample - self.velocity)' in source


def tracker_default(name):
    """ノードの既定値をソースから読む。テストへ値を写し取らないため。"""
    source = (Path(__file__).resolve().parents[1] / 'omni_autonomy_next'
              / 'trajectory_tracker_node.py').read_text(encoding='utf-8')
    match = re.search(r"'%s': ([0-9.]+)," % name, source)
    assert match, f'{name} の既定値が見つからない'
    return float(match.group(1))


def replay_yaw_loop(yaw_gain, plant_gain, seconds=8.0):
    """姿勢軸だけの閉ループ再生。返すのは最後の2秒の姿勢の peak-to-peak [deg]。

    実測した駆動系モデル（無駄時間 120 ms + 一次遅れ 80 ms）に、配備した
    20 Hz の制御周期、遅れ補償の先読み 0.20 s、角速度上限と角加速度制限を
    入れてある。``plant_gain`` は「指令 1 m/s(rad/s) に対して実際に出る量」
    で、ミキサの auto_units_per_mps が決める。基準角速度は 0、初期姿勢誤差
    6 度で、収束すれば peak-to-peak は 0 になる。
    """
    step, control, dead, tau = 0.01, 1.0 / 20.0, 0.12, 0.08
    yaw_limit = 1.30            # balanced プロファイルの角速度上限
    accel = 2.00                # velocity_smoother の max_accel[2]
    lead = 0.20                 # feedback_delay_sec
    yaw, actual, command = math.radians(6.0), 0.0, 0.0
    pipeline = [0.0] * int(round(dead / step))
    tail = []
    for index in range(int(seconds / step)):
        if index % int(round(control / step)) == 0:
            target = -yaw_gain * (yaw + actual * lead)
            target = max(-yaw_limit, min(yaw_limit, target))
            command += max(-accel * control,
                           min(accel * control, target - command))
        pipeline.append(command)
        actual += (pipeline.pop(0) * plant_gain - actual) * (step / tau)
        yaw += actual * step
        if index * step > seconds - 2.0:
            tail.append(math.degrees(yaw))
    return max(tail) - min(tail)


def test_the_yaw_loop_keeps_margin_against_a_drivebase_gain_error():
    """姿勢のP項は、駆動系ゲイン誤差に対する余裕で選ぶ。

    駆動系ゲイン（指令に対して実際に出る速度の比）は
    bacon_gateway/include/value.hpp の auto_units_per_mps が決めるが、あの
    値は「フルスティック = 0.55 m/s」という仮定から出したもので実測されて
    いない。ゲイン誤差はそのまま実効ループゲインに掛かり、このループは
    遅れで律速されている（無駄時間 120 ms + 一次遅れ 80 ms + 20 Hz 保持）
    ので、ある倍率を超えると自励振動する。外から見ると「走行中に機体の
    向きが左右に振れる」であり、実機で報告された症状そのものである。

    以前の 2.6 は 2.2 倍あたりで発振し、実機で記録された armed 走行
    （logs/diagnose-chain-20260806-204616.log）の応答/指令比は約 2.9 倍
    だった。1.6 は 3.2 倍でも収束する。姿勢はフィードフォワードの角速度が
    運んでいるのでこの項は残差しか直さず、ゲインを下げても追従は落ちない
    （同モデル4区間で到着時刻の差 0.01 s 以内、終端誤差は同じ）。

    scripts/yaw_oscillation_probe.py --margin が全区間版の表を出す。
    """
    yaw_gain = tracker_default('yaw_gain')
    # 較正が正しければどちらでも収束する。
    assert replay_yaw_loop(yaw_gain, 1.0) < 0.05
    assert replay_yaw_loop(2.6, 1.0) < 0.05
    # 実機で記録された倍率では、配備値だけが収束する。
    assert replay_yaw_loop(2.6, 2.9) > 1.0
    assert replay_yaw_loop(yaw_gain, 2.9) < 0.05
    # 3.2 倍まで余裕がある。
    assert replay_yaw_loop(yaw_gain, 3.2) < 0.05


def test_the_tracker_reports_the_delivered_drivebase_gain():
    """較正誤差が走行ログに数字で出ること。

    2.9 倍のゲイン誤差が「速い」以外の症状（姿勢の自励振動、加速度制限の
    飽和、フィードフォワードの全区間のずれ）で出るので、比そのものを状態へ
    出しておかないと原因が分からない。_track_flow は指令と実測の進行方向を
    世界系で均してあるので、そのノルム比が駆動系ゲインである。
    """
    source = (Path(__file__).resolve().parents[1] / 'omni_autonomy_next'
              / 'trajectory_tracker_node.py').read_text(encoding='utf-8')
    assert 'def _flow_gain(self)' in source
    assert source.count('flow_gain=self._flow_gain()') >= 2
    assert 'np.linalg.norm(self.measured_flow)) / commanded' in source


# ---------------------------------------------------------------------------
# 回りながら走る: 姿勢計画の実現可能性と滑らかさ
# ---------------------------------------------------------------------------

def envelope():
    from omni_autonomy_next.omni_yaw import OmniEnvelope
    robot = yaml.safe_load(
        (CONFIG / 'robot.yaml').read_text(encoding='utf-8'))['robot']
    return OmniEnvelope(robot['drivetrain'])


def straight_leg(start=(-1.80, 4.75), goal=(-4.19, 4.69), knots=13):
    """地点1->2 の 2.39 m。直線だが姿勢は -90 度から 0 度へ入れ替える。"""
    start, goal = np.asarray(start, float), np.asarray(goal, float)
    points = start + np.linspace(0.0, 1.0, knots)[:, None] * (goal - start)
    direction = (goal - start) / np.linalg.norm(goal - start)
    arclength = np.concatenate(
        ([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))))
    return points, np.tile(direction, (knots, 1)), arclength


def step_budget(arclength, yaw_limit=1.30, speed=0.78):
    return np.maximum(yaw_limit * np.diff(arclength) / speed,
                      math.radians(5.0))


def test_the_yaw_plan_stays_inside_what_one_segment_can_turn():
    """計画した姿勢変化は、その区間で実際に回れる角度の内側であること。

    ここが破れていたのが `optimize_yaw` を使えなかった本当の理由である。
    候補格子は 45 度固定、1区間で回れるのは 19 度なので、格子上の隣どうしは
    どれも到達不能だった。全滅時のフォールバックが制約を無視して最短の跳びを
    選ぶので、-90 度から 0 度への旋回は最初の 0.199 m に 88.6 度詰め込まれ、
    7.76 rad/m すなわち 105 rad/s^2 を要求する計画になっていた。使えるのは
    2.00 rad/s^2 である。出口の制限がそれを削るため機体は基準姿勢に追従でき
    ず、2.39 m の区間が 4.99 s から 5.43 s へ延びていた。
    """
    points, tangents, arclength = straight_leg()
    budget = step_budget(arclength)
    # Reproduce the historical low-budget optimizer regression independently
    # of today's configured high-speed straight-line wheel budget.
    original_envelope = envelope()
    original_envelope.max_wheel = 15.709120382
    yaws = plan_yaw_profile(
        points, tangents, envelope=original_envelope, ellipse_x=0.78,
        ellipse_y=0.702, current_yaw=-0.5 * math.pi, goal_yaw=0.0,
        max_yaw_step=budget)
    assert yaws[0] == pytest.approx(-0.5 * math.pi)
    assert wrap(yaws[-1]) == pytest.approx(0.0, abs=1.0e-9)
    assert np.all(np.abs(np.diff(yaws)) <= budget + 1.0e-9)
    # 旋回は単調で、90度を区間全体へ配っている（1区間に詰め込まない）。
    assert np.all(np.diff(yaws) >= -1.0e-9)
    assert np.max(np.abs(yaw_derivative(yaws, arclength))) * 0.78 <= 1.30


def test_the_yaw_candidates_are_no_coarser_than_one_segment_can_turn():
    """候補の間隔が「1区間で回れる角度」より粗いと緩やかな旋回が作れない。"""
    limit = math.radians(19.0)
    fine = np.sort(yaw_candidates(
        math.pi, 0.0, -0.5 * math.pi, ramp_yaw=-0.4, step_limit=limit))
    gaps = np.diff(np.concatenate((fine, [fine[0] + 2.0 * math.pi])))
    # 3 度以内の重複は落とすので、その分だけ許容する。
    assert gaps.max() <= limit + math.radians(3.1)
    coarse = np.sort(yaw_candidates(math.pi, 0.0, -0.5 * math.pi))
    coarse_gaps = np.diff(
        np.concatenate((coarse, [coarse[0] + 2.0 * math.pi])))
    assert coarse_gaps.max() > limit


def test_the_default_yaw_ramp_is_inside_the_search_space():
    """既定の配分が候補にあれば、最適化は既定より遅い解を返せない。"""
    ramp = math.radians(-37.0)
    options = yaw_candidates(
        math.pi, 0.0, -0.5 * math.pi, ramp_yaw=ramp,
        step_limit=math.radians(19.0))
    assert min(abs(wrap(value - ramp)) for value in options) <= math.radians(3.0)


def test_the_optimised_yaw_is_not_slower_than_the_linear_ramp():
    """姿勢最適化を入れて所要時間が悪化しないこと。

    以前の目的関数は -(速度 + 余裕) + 0.35 * |姿勢変化| で、m/s と rad を
    足していた。速度は「旋回ゼロ」で評価するので、姿勢を回して得る利得は
    数えるのに、そのために食う車輪バジェットは数えていなかった。いまは
    区間の通過時間[s]そのものを最小化する。
    """
    points, tangents, arclength = straight_leg(knots=49)
    cutoff = float(arclength[-1]) - 0.10
    knots = np.linspace(0.0, cutoff, 13)
    indices = np.searchsorted(arclength, knots).clip(0, len(points) - 1)
    coarse = plan_yaw_profile(
        points[indices], tangents[indices], envelope=envelope(),
        ellipse_x=0.78, ellipse_y=0.702, current_yaw=-0.5 * math.pi,
        goal_yaw=0.0, max_yaw_step=step_budget(knots))
    optimised = smooth_yaw_profile(
        np.interp(np.minimum(arclength, cutoff), arclength[indices], coarse),
        arclength, 0.25)
    ramp = -0.5 * math.pi + 0.5 * math.pi * np.clip(
        arclength / cutoff, 0.0, 1.0)

    def duration(yaws):
        limits, _ = direction_speed_limits(
            tangents, yaws, arclength, envelope=envelope(),
            ellipse_x=0.78, ellipse_y=0.702)
        limits = np.where((arclength[-1] - arclength) <= 0.10,
                          np.minimum(limits, 0.30), limits)
        return Trajectory(
            points, yaws, limits, acceleration=0.85,
            lateral_acceleration=1.2, entry_speed=0.0, angular_speed=1.30,
            angular_acceleration=2.0).duration

    assert duration(optimised) <= 1.02 * duration(ramp)


def test_smoothing_rounds_the_yaw_knots_without_moving_the_ends():
    """折れ線の姿勢列は角速度が階段になる。丸めても端点は動かさない。"""
    arclength = np.linspace(0.0, 3.0, 61)
    knots = np.linspace(0.0, 3.0, 11)
    staircase = np.interp(
        arclength, knots, np.where(knots < 1.5, 0.0, 0.8 * (knots - 1.5)))
    smoothed = smooth_yaw_profile(staircase, arclength, 0.25)

    def bend(values):
        return float(np.max(np.abs(yaw_derivative(
            yaw_derivative(values, arclength), arclength))))

    # 始端は機体の実姿勢、終端は指定姿勢。どちらもディリクレ境界で厳密。
    assert smoothed[0] == pytest.approx(staircase[0])
    assert smoothed[-1] == pytest.approx(staircase[-1])
    assert bend(smoothed) < 0.25 * bend(staircase)
    # 長さ 0 は無効化。
    assert smooth_yaw_profile(
        staircase, arclength, 0.0) == pytest.approx(staircase)


def test_the_time_parameterisation_bounds_the_yaw_rate():
    """角速度は yaw'(s) * v なので、速度側の上限として掛けられること。

    掛けないと、姿勢列に折れ目がある場所で実現不能な角速度をフィード
    フォワードし、出口の加速度制限がそれを削る。削られた分は基準姿勢に
    対する遅れになる。
    """
    points = straight(4.0)
    arclength = np.concatenate(
        ([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))))
    yaws = np.where(arclength < 2.0, 0.0, 1.2 * (arclength - 2.0))
    limits = np.full(len(points), 0.78)
    settings = dict(acceleration=0.85, lateral_acceleration=1.2,
                    entry_speed=0.0)

    def peak_rate(trajectory):
        return max(
            abs(trajectory.sample(t)[3])
            for t in np.linspace(0.0, trajectory.duration, 400))

    unbounded = Trajectory(points, yaws, limits, **settings)
    bounded = Trajectory(points, yaws, limits, angular_speed=0.5,
                         angular_acceleration=2.0, **settings)
    assert peak_rate(unbounded) > 0.5
    assert peak_rate(bounded) <= 0.5 + 1.0e-6
    # 旋回のために遅くなるので、所要時間は伸びる。それが正しい代償である。
    assert bounded.duration > unbounded.duration


def test_the_yaw_is_retired_before_the_terminal_approach_begins():
    """終端の寄せは純粋な並進の整定にする。

    以前は目標位置と同時に姿勢を入れ終えていたので、最後の低速区間でも
    基準姿勢が動き続け、到着判定の瞬間の姿勢誤差はその区間の速度に比例
    していた。終端を 0.16 -> 0.35 m/s へ上げると区間 1->2 の姿勢誤差が
    1.30 -> 1.74 deg（公差 2.00 deg）へ悪化する、という形で出ていた。
    分けたあとは同じ条件で 0.30 -> 0.37 deg である。
    """
    source = (Path(__file__).resolve().parents[1] / 'omni_autonomy_next'
              / 'trajectory_tracker_node.py').read_text(encoding='utf-8')
    assert 'yaw_cutoff = max(' in source
    assert 'points, tangents, arclength, float(pose[2]), goal_yaw,\n' \
           '                yaw_cutoff)' in source

    points = straight(3.0)
    arclength = np.concatenate(
        ([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))))
    cutoff = float(arclength[-1]) - 0.10
    yaws = -0.5 * math.pi + 0.5 * math.pi * np.clip(
        arclength / cutoff, 0.0, 1.0)
    limits = np.where((arclength[-1] - arclength) <= 0.10,
                      np.minimum(np.full(len(points), 0.78), 0.30), 0.78)
    trajectory = Trajectory(
        points, yaws, limits, acceleration=0.85, lateral_acceleration=1.2,
        entry_speed=0.0, angular_speed=1.30, angular_acceleration=2.0)
    _, _, yaw, rate, _ = trajectory.sample(trajectory.duration)
    assert yaw == pytest.approx(0.0, abs=1.0e-6)
    assert rate == pytest.approx(0.0, abs=1.0e-6)
    # 打ち切り点より先はどこを取っても指定姿勢のまま。
    inside = trajectory.arclength >= cutoff + 0.01
    assert np.allclose(trajectory.yaws[inside], 0.0, atol=1.0e-6)


def test_the_terminal_approach_is_the_measured_setting():
    """終端ゾーンは要る。ゼロにすると整定が 0.3 mm から 16.7 mm へ崩れる。

    同じ機体モデル（無駄時間 120 ms・一次遅れ 80 ms）で 4 区間を掃引した
    結果が scripts/yaw_oscillation_probe.py --terminal である。
    """
    source = (Path(__file__).resolve().parents[1] / 'omni_autonomy_next'
              / 'trajectory_tracker_node.py').read_text(encoding='utf-8')
    assert "'terminal_approach_m': 0.10," in source
    assert "'terminal_speed': 0.30," in source


def test_the_tracker_plans_inside_the_envelope_the_guard_executes():
    """追従器は RuntimeGuard が実際に通す包絡線で時間割りを作ること。

    追従器モードでは velocity_smoother を迂回しているので、指令を作るのは
    追従器だけである。それが倍率を知らないと、

      * 操作卓で sprint を選んでも軌道は既定プロファイルのまま作られる。
      * ACCEPTANCE step 3 の 10% では、ゲートが 0.078 m/s しか通さないのに
        0.78 m/s の時間割りを追う。
      * 学習残差 (0.35〜1.0) と red zone (全地点 0.95 m 以内で 0.70) は
        つねに効いているので、目標の手前ではいつも 30% 速い時間割りだった。

    倍率は経路の形を変えない一様なスケールなので、軌道を作り直さずに
    基準時刻の進み方で掛ける（経路速度スケーリング）。
    """
    package = Path(__file__).resolve().parents[1] / 'omni_autonomy_next'
    tracker = (package / 'trajectory_tracker_node.py').read_text(
        encoding='utf-8')
    guard = (package / 'runtime_guard_node.py').read_text(encoding='utf-8')

    assert "'safety_state_topic': '/system/safety_state'," in tracker
    assert "'/system/safety_state'" in guard
    # 両側が同じ鍵を使っていること。
    assert "state.get('applied_scale')" in tracker
    assert "'applied_scale': result.applied_scale," in guard
    assert "state.get('profile')" in tracker
    assert "'profile': result.profile," in guard
    # 倍率は時間割りだけを変え、プロファイル変更は作り直す。
    assert 'reference_time + self.control_dt * scale' in tracker
    assert 'self.pending_plan = self.last_plan' in tracker


def test_the_yaw_derivative_survives_a_duplicated_path_point():
    """重複点があっても指令に NaN を出さないこと。

    姿勢の弧長微分は角速度そのものになるので、ここで inf/NaN が出ると
    そのままフィードフォワードへ入る。
    """
    arclength = np.array([0.0, 0.1, 0.1, 0.2])
    values = np.array([0.0, 0.1, 0.1, 0.3])
    derivative = yaw_derivative(values, arclength)
    assert np.all(np.isfinite(derivative))
    # 退化していない入力では中心差分のまま。
    clean = np.linspace(0.0, 1.0, 5)
    assert yaw_derivative(2.0 * clean, clean) == pytest.approx(
        np.full(5, 2.0))


def test_the_knot_helper_never_puts_a_knot_past_the_cutoff():
    from omni_autonomy_next.trajectory_tracker_node import yaw_plan_knots
    arclength = np.linspace(0.0, 3.551, 72)
    indices, last = yaw_plan_knots(arclength, 0.30, 3.551 - 0.10)
    assert last <= 3.551 - 0.10 + 1.0e-9
    assert indices[0] == 0
    assert np.all(np.diff(indices) > 0)
    # 打ち切りが経路より長くても、終点で止まる。
    _, whole = yaw_plan_knots(arclength, 0.30, 99.0)
    assert whole == pytest.approx(3.551)
    # 節点が2つに満たない短い経路でも成立する。
    short, edge = yaw_plan_knots(np.array([0.0, 0.04]), 0.30, 0.0)
    assert len(short) == 2 and edge == pytest.approx(0.04)


def test_the_tracker_trims_a_yaw_rate_the_monitor_would_veto():
    """壁ぎわで通らない旋回を出さない。並進が絞られるのはこれのせい。

    collision_monitor の approach は、指令角速度を 1.2 秒ぶん一定と見なして
    フットプリントを掃き、接触するなら twist 全体（並進も）を
    contact_time / 1.2 倍にする。balanced の 1.30 rad/s は 1.2 秒で 89 度
    回る指令であり、地点1・4・5・7 の回転余裕は 10〜24 度しかないので、
    壁ぎわの旋回はどれも「接触する」と判定される。実行される角速度は
    要求に関係なく余裕/1.2 に張り付くから、姿勢誤差は開き続け、P 項が
    角速度を上げ、比はさらに小さくなり、並進が道連れになる。
    """
    horizon = load_collision_monitor_horizon(CONFIG / 'nav2_next.yaml')
    field = CadClearanceModel.from_yaml(
        CONFIG / 'field_planning.yaml', CONFIG / 'competition_footprints.yaml'
    )
    runtime = yaml.safe_load(
        (CONFIG / 'runtime.yaml').read_text(encoding='utf-8')
    )['runtime_guard']['ros__parameters']
    yaw_limit = json.loads(
        runtime['profiles_json'])[runtime['default_profile']]['angular']
    poses = load_configured_poses(CONFIG / 'field_poses.yaml')
    for key in ('1', '4', '5', '7'):
        configured = poses[key]
        position = np.asarray([configured['x'], configured['y']])
        limited = limit_yaw_rate(
            yaw_limit, position=position, yaw=configured['yaw'],
            clearance_model=field, horizon=horizon)
        # 掃き角がプロファイル上限では回転余裕を大きく超えている。
        assert yaw_limit * horizon > math.radians(30.0)
        assert 0.0 < limited < yaw_limit
        assert limited * horizon <= math.radians(30.0)


def test_the_tracker_trims_the_yaw_rate_before_it_pays_wheel_budget():
    """順序が要点。通らない角速度のために並進のバジェットを譲らないこと。

    車輪バジェットの一様縮小が先に走ると、balanced の並進 0.78 と旋回
    1.30 rad/s は 23.28 / 15.709 で並進が 0.675 倍になる。その角速度は
    このあと下流で落とされるので、譲った並進はまるまる損である。
    """
    source = (
        Path(__file__).resolve().parents[1]
        / 'omni_autonomy_next' / 'trajectory_tracker_node.py'
    ).read_text(encoding='utf-8')
    trimmed = source.index('_monitor_safe_yaw_rate(\n')
    budget = source.index('budget_scale = wheel_limit / cost')
    assert trimmed < budget
    # 余裕モデルは optimize_yaw とは無関係に読むこと。既定は false なので、
    # 条件付きで読んでいると配備構成では制限そのものが効かない。
    assert "if field_file and bool(self.get_parameter('optimize_yaw')" \
        not in source
    assert "'limit_yaw_rate_to_clearance': True," in source


def test_both_command_producers_apply_the_same_yaw_rate_rule():
    """MPPI モードと復帰動作も同じ壁に当たる。規則は一つで共有する。"""
    package = Path(__file__).resolve().parents[1] / 'omni_autonomy_next'
    tracker = (package / 'trajectory_tracker_node.py').read_text(encoding='utf-8')
    policy = (package / 'rl_policy_node.py').read_text(encoding='utf-8')
    for source in (tracker, policy):
        assert 'limit_yaw_rate' in source
        assert 'load_collision_monitor_horizon' in source
    # rl_policy は monitor の直前段なので、追従器が止まっている間も効く。
    assert "output.angular.z = float(yaw_rate)" in policy
