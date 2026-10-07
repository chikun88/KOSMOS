"""時間パラメータ化した軌道のフィードフォワード追従器（オムニ姿勢最適化つき）。

MPPI は 20 Hz のフィードバックのみで、指令から実測まで実測 100 ms 遅れる。
0.78 m/s ではその間に 78 mm 進む。速度を上げるほどその遅れが追従誤差に
なる。ここでは経路を時間の関数として先に決め、速度・角速度を
フィードフォワードで与えて、フィードバックは残差だけを直す。

さらにホロノミックであることを二重に使う。

  * 45度オムニは進行方向が体軸に一致すると 1.111 m/s、ローラ上に乗ると
    0.785 m/s しか出ない。姿勢は進行方向と独立に選べるので、通過中の
    姿勢を体軸へ合わせるだけで最大 1.41 倍速くなる。
  * フットプリント余裕も姿勢の関数である。地点4/5の射撃レーンは指定姿勢で
    57/59 mm しかないが、通過中は姿勢が自由なので広い姿勢で通り、終端で
    だけ指定姿勢へ入れればよい。

この二つは競合するので、弧長方向の動的計画法で姿勢列を決める（omni_yaw）。
速度上限は姿勢が決まったあとに方向ごとに計算するので、体軸に乗っている
区間では自動的に速く、斜めに乗る区間では自動的に遅くなる。

出力は /cmd_vel_nav_smoothed。rl_policy -> collision_monitor ->
runtime_guard の安全鎖はそのまま通るので、停止権限はどれも変わらない。

なぜ既定は姿勢最適化 false か（2026-09-01 の測定）
--------------------------------------------------
上の 1.41 倍は「駆動系の菱形」での値である。運用プロファイル balanced の
並進包絡線は 0.78/0.702 のほぼ円なので、その中で姿勢を体軸へ合わせて得る
利得は最大 11% しかない。菱形へ内接する円の半径は 1.111/sqrt(2) = 0.786 で
あり、balanced の 0.78 はすでにその天井である。つまり「どの方向へも速い」
プロファイルはこれ以上作れず、体軸の 1.111 m/s を使うには前後に速く横に
遅い非等方プロファイル（sprint 0.95/0.57）と姿勢合わせを組み合わせるしか
ない。それも測ると釣り合わない。旋回と並進は同じ 15.709 rad/s の車輪
バジェットを共有するので、0.95 m/s で走ると旋回に残るのは 0.24 rad/s しか
なく、90 度合わせるだけで 3 秒以上かかる。この競技フィールドの区間は
2.4〜3.7 m なので、巡航で取り戻せない。

同じ機体モデルで 4 区間を比べた結果（到着時刻）:

  区間        既定の線形配分   姿勢最適化   sprint + 姿勢最適化
  1->4 (90度)    6.42 s        6.45 s        6.26 s
  1->2 (90度)    4.99 s        5.09 s        5.50 s
  4->5 (0度)     6.61 s        6.61 s        7.64 s
  3->6 (11度)    5.56 s        5.58 s        5.42 s

さらに滑らかさでも線形配分が勝つ。要求角加速度は線形配分が 2.5〜3.9
rad/s^2（使えるのは 2.00）で出口の制限に当たるのは 0.2〜0.9% の周期だが、
姿勢最適化では 17.8〜27.5 rad/s^2、3.7〜7.0% である。原因は離散最適化を
1 Hz の再計画の中で回していることそのものである。再計画のたびに別の枝へ
飛ぶことがあり、そのとき基準角速度が段で変わる（実測: 区間 1->4 の要求
角加速度のピークは、ちょうど再計画の時刻 t=4.00 s から 0.12 s 続く）。

したがって速度はここからは出ない。出るのは終端の寄せ方（terminal_approach_m
と terminal_speed）で、そちらは 5〜12% 効く。姿勢最適化は
「プロファイルを菱形へ寄せられるようになったとき」に初めて意味を持つので、
壊れたまま残さず、実現可能で滑らかな状態にして既定 false のまま置いてある。
"""
import json
import copy
import math
import threading
import time
from typing import Optional

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist
from nav_msgs.msg import Odometry, Path
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
import yaml

from .config import load_collision_monitor_horizon, load_robot
from .fixed_gate_speed import FixedGateSpeed
from .motion_prediction import DelayedMotionPredictor
from .reverse_approach import POSITION_TOLERANCE, YAW_TOLERANCE
from .omni_yaw import (
    OmniEnvelope,
    direction_speed_limits,
    plan_yaw_profile,
    smooth_yaw_profile,
    wrap,
    yaw_derivative,
)
from .rl_residual import CadClearanceModel, limit_yaw_rate
from .staged_heading import (
    HeadingStage, path_clearances, prepare_stage, repair_corridor, remaining_path,
)
from .staged_heading_node import StagedHeadingMixin
from .source_freshness import message_stamp_nanoseconds


def yaw_of(quaternion) -> float:
    values = (quaternion.x, quaternion.y, quaternion.z, quaternion.w)
    norm = math.hypot(*values)
    if not all(math.isfinite(v) for v in values) or not math.isfinite(norm) or norm <= 1.e-6:
        return math.nan
    x, y, z, w = (v / norm for v in values)
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def terminal_zone(node, stage=None):
    stage = stage if stage is not None else getattr(node, 'heading_stage', None)
    if stage is not None and stage.phase == 'APPROACH':
        # Avoid a 40 cm position-servo crawl before an open turning gate.
        # The trajectory still brakes to rest; capture checks stop distance.
        return .10
    return float(node.get_parameter('terminal_approach_m').value)


def terminal_cap(node):
    return float(node.get_parameter('terminal_speed').value)


def stopping_speed(distance, deceleration, reaction_sec):
    """Solve v*reaction + v**2/(2*a) <= remaining path distance."""
    a = max(float(deceleration), 1.e-9)
    delay = max(float(reaction_sec), 0.)
    return max(0., math.sqrt((a*delay)**2 + 2*a*max(0., distance)) - a*delay)


def record_prediction_command(node, now, command):
    predictor = getattr(node, 'motion_predictor', None)
    if predictor is not None:
        predictor.record(now, command)


def sprint_turn_clearance(points, yaws, model, reserve_m=.35):
    """Release the legacy turn cap only in a checked, open corridor.

    Subtract a translation + rotational sweep bound between samples, then
    reserve 35 cm for tracking deviation along the ENTIRE remaining route.
    Open-corridor delayed-drive replay reached 23 cm lateral deviation; this
    eligibility reserve must exceed that rather than using the old 15 cm.
    Pointwise release caused late speed transitions and clearance regressions
    in delayed-drive replay. This is a static CAD eligibility
    test; live collision monitoring and stopping limits remain mandatory.
    Missing geometry never grants the faster budget.
    """
    allowed = np.zeros(len(points), dtype=bool)
    if model is None or getattr(model, 'footprint', None) is None:
        return allowed
    if not np.isfinite(points).all() or not np.isfinite(yaws).all():
        return allowed
    sweep = np.linalg.norm(np.diff(points, axis=0), axis=1)
    sweep += model.radius*np.abs(np.diff(yaws))
    reserve = np.zeros(len(points))
    reserve[:-1] = np.maximum(reserve[:-1], sweep)
    reserve[1:] = np.maximum(reserve[1:], sweep)
    if not math.isfinite(reserve_m) or reserve_m < .025:
        return allowed
    margins = model.clearance_over_poses(points, yaws, cap=reserve_m+float(np.max(reserve, initial=0.))+.001)
    allowed[:] = np.all(np.isfinite(margins) & (margins-reserve >= reserve_m))
    return allowed


def same_remaining_corridor(old_points, new_points, position, tolerance=.015):
    """Bidirectional, ordered comparison of the remaining geometric path.

    Small planner cell jitter should not restart the acceleration/yaw profile.
    Both directions are checked so a new shortcut cannot erase an old bend.
    """
    old = resample(remaining_path(old_points, position), .05)
    new = resample(remaining_path(new_points, position), .05)
    if np.linalg.norm(old[-1]-new[-1]) > tolerance:
        return False
    for source, target in ((old,new),(new,old)):
        if len(target) < 2:
            if np.max(np.linalg.norm(source-target[0],axis=1)) > tolerance:
                return False
            continue
        delta = np.diff(target,axis=0)
        length = np.linalg.norm(delta,axis=1)
        offset = source[:,None,:]-target[None,:-1,:]
        fraction = np.clip(np.einsum('ijk,jk->ij',offset,delta)
                           /np.maximum(length**2,1.e-12),0.,1.)
        distances = np.linalg.norm(offset-fraction[:,:,None]*delta[None,:,:],axis=2)
        nearest = np.argmin(distances,axis=1)
        if np.max(distances[np.arange(len(source)),nearest]) > tolerance:
            return False
        arc = np.r_[0.,np.cumsum(length)]
        progress = arc[nearest]+fraction[np.arange(len(source)),nearest]*length[nearest]
        if np.any(np.diff(progress) < -tolerance):
            return False
    return True


def resample(points: np.ndarray, spacing: float) -> np.ndarray:
    """経路を等間隔に張り直す。Smac の出力は間隔が一定でない。"""
    if len(points) < 2:
        return points
    segments = np.linalg.norm(np.diff(points, axis=0), axis=1)
    arclength = np.concatenate(([0.0], np.cumsum(segments)))
    total = float(arclength[-1])
    if total < 1.0e-6:
        return points[:1]
    # A rest-to-rest move needs an interior sample at which to accelerate.
    # With only two zero-speed endpoints a 4 cm replan took 400 seconds.
    count = max(3, int(round(total / spacing)) + 1)
    target = np.linspace(0.0, total, count)
    return np.column_stack((
        np.interp(target, arclength, points[:, 0]),
        np.interp(target, arclength, points[:, 1]),
    ))


def smooth_path(points: np.ndarray, spacing: float, smoothing: float) -> np.ndarray:
    """経路のグリッド階段を落とす。端点は動かさない。

    SmacPlanner2D はセル中心を返すので、グリッド軸に乗っていない区間は
    階段状の折れ線になる。それをセル幅と同じ 0.05 m で張り直して中心差分を
    取ると、接線は 8 近傍グリッドの 45 度をそのまま振る。実測で 3 m の斜め
    区間あたり方向反転 31〜44 回、0.7 m/s では横方向に ±0.53 m/s を指令
    していた。機体はそれを追えないので蛇行する。

    ここで消しているずれはセル半分（0.025 m）が上限で、planner 自身の
    tolerance 0.20 m の内側である。実際の経路の曲率半径は平滑長より
    はるかに大きいので残る。

    端点を固定した [1,2,1]/4 の反復（ディリクレ境界）なので、始点＝機体
    位置と終点＝スナップ済み目標はどちらも厳密に保たれる。

    さらに各点の移動量をセル半分に制限する。階段の振幅はセル半分が上限
    なので、それを超える移動は量子化の除去ではなく経路そのものの変更で
    ある。壁を回り込む角では平滑化は 49 mm 内側を通ろうとするが、地点
    4〜7 の余裕は 57〜59 mm しかない。制限しておけば、直線の斜め区間では
    階段が丸ごと消え、実在する角は削られない。
    """
    count = len(points)
    if count < 3 or spacing <= 0.0 or smoothing <= 0.0:
        return points
    # 1回の [1,2,1]/4 の分散は spacing^2 / 2 なので、長さ尺度 smoothing に
    # 達するのに必要な回数はこれ。
    passes = int(round(2.0 * (smoothing / spacing) ** 2))
    if passes < 1:
        return points
    smoothed = np.array(points, dtype=float, copy=True)
    for _ in range(passes):
        smoothed[1:-1] = (
            0.25 * smoothed[:-2] + 0.5 * smoothed[1:-1] + 0.25 * smoothed[2:])
    offset = smoothed - points
    distance = np.linalg.norm(offset, axis=1)
    limit = 0.5 * spacing
    excess = distance > limit
    if np.any(excess):
        offset[excess] *= (limit / distance[excess])[:, None]
    return points + offset


def path_tangents(points: np.ndarray) -> np.ndarray:
    """中心差分の単位接線。端点は片側差分。"""
    count = len(points)
    if count < 2:
        return np.tile(np.array([1.0, 0.0]), (max(count, 1), 1))
    delta = np.zeros_like(points)
    delta[1:-1] = points[2:] - points[:-2]
    delta[0] = points[1] - points[0]
    delta[-1] = points[-1] - points[-2]
    norms = np.linalg.norm(delta, axis=1)
    norms[norms < 1.0e-9] = 1.0
    return delta / norms[:, None]


def menger_curvature(points: np.ndarray) -> np.ndarray:
    """3点外接円の曲率。端点は隣の値で埋める。"""
    count = len(points)
    curvature = np.zeros(count)
    if count < 3:
        return curvature
    a, b, c = points[:-2], points[1:-1], points[2:]
    area2 = np.abs(
        (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
        - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    )
    denominator = (
        np.linalg.norm(b - a, axis=1)
        * np.linalg.norm(c - b, axis=1)
        * np.linalg.norm(c - a, axis=1)
    )
    safe = denominator > 1.0e-9
    curvature[1:-1] = np.where(
        safe, 2.0 * area2 / np.where(safe, denominator, 1.0), 0.0)
    curvature[0] = curvature[1]
    curvature[-1] = curvature[-2]
    return curvature


def yaw_plan_knots(arclength, spacing: float, cutoff: float):
    """姿勢を最適化するサンプル点の添字と、その最後の点の弧長を返す。

    ``cutoff`` は指定姿勢へ入れ終えたい弧長。返る添字の最後は必ず
    ``arclength <= cutoff`` の中で最大の点であり、第2の戻り値はその弧長で
    ある。呼び手はそれを補間の打ち切りに使うこと。

    ここを分けてあるのは、二つの弧長格子を混ぜると静かに壊れるからである。
    最初の実装は等間隔の目標弧長 ``targets`` から ``searchsorted`` で添字を
    取り、補間は ``cutoff`` で打ち切っていた。``searchsorted`` は目標以上の
    最初の点を返すので最後の添字の弧長は ``cutoff`` を超えることがあり、
    そのとき補間は最後の節点に到達しない。実測（地点1->4）では基準姿勢が
    指定姿勢の 1.29 度手前で平らになった。姿勢の公差は 2.0 度なので、
    何もしていないのに公差の 65% を使っていた。
    """
    arclength = np.asarray(arclength, dtype=float)
    limit = float(np.clip(cutoff, 0.0, float(arclength[-1])))
    count = max(2, int(round(limit / max(spacing, 1.0e-6))) + 1)
    targets = np.linspace(0.0, limit, count)
    # 目標以下で最大の点。最後の節点が cutoff を越えない。
    indices = np.clip(
        np.searchsorted(arclength, targets, side='right') - 1,
        0, len(arclength) - 1)
    # 点間隔より節点間隔が細かいと添字が重複し、補間の x が単調でなくなる。
    indices = np.unique(indices)
    if len(indices) < 2:
        indices = np.array([0, len(arclength) - 1])
    return indices, float(arclength[indices[-1]])


class Trajectory:
    """等間隔の経路に、点ごとの速度上限と姿勢列を載せた時間関数。"""

    def __init__(
        self,
        points: np.ndarray,
        yaws: np.ndarray,
        speed_limits: np.ndarray,
        *,
        acceleration: float,
        lateral_acceleration: float,
        entry_speed: float,
        angular_speed: Optional[float] = None,
        angular_acceleration: Optional[float] = None,
        deceleration: Optional[float] = None,
    ) -> None:
        deceleration = float(acceleration if deceleration is None else deceleration)
        if not all(math.isfinite(v) and v > 0. for v in (acceleration, deceleration)):
            raise ValueError('acceleration and deceleration must be positive and finite')
        self.points = points
        self.yaws = np.asarray(yaws, dtype=float)
        segments = np.linalg.norm(np.diff(points, axis=0), axis=1)
        self.arclength = np.concatenate(([0.0], np.cumsum(segments)))
        self.length = float(self.arclength[-1])
        self.yaw_per_metre = yaw_derivative(self.yaws, self.arclength)

        # A one-pose plan is a position hold with a yaw target. Do not invent
        # a translation just to turn in place; the terminal servo handles it.
        if self.length < 1.0e-9:
            self.speed = np.zeros(len(points))
            self.time = np.zeros(len(points))
            self.duration = 0.0
            return

        # 1) 方向ごとの上限（車輪バジェット＋プロファイル）と曲率の上限
        curvature = menger_curvature(points)
        with np.errstate(divide='ignore'):
            curve_limit = np.where(
                curvature > 1.0e-6,
                np.sqrt(lateral_acceleration / np.maximum(curvature, 1.0e-6)),
                np.inf,
            )
        speed = np.minimum(np.asarray(speed_limits, dtype=float), curve_limit)

        # 1b) 旋回そのものの上限。角速度は yaw'(s) * v、角加速度は
        # yaw''(s) * v^2 + yaw'(s) * a なので、どちらも速度の上限になる。
        # これを入れないと、姿勢列に折れ目がある場所で実現不能な角速度を
        # フィードフォワードし、出口の加速度制限がそれを削る。削られた分は
        # 基準姿勢に対する遅れになるので、「回りながら走る」区間でだけ
        # 追従誤差が出ていた。曲率と横加速度の関係とまったく同じ形。
        rate = np.abs(self.yaw_per_metre)
        if angular_speed is not None and float(angular_speed) > 0.0:
            with np.errstate(divide='ignore'):
                speed = np.minimum(speed, np.where(
                    rate > 1.0e-6,
                    float(angular_speed) / np.maximum(rate, 1.0e-6),
                    np.inf,
                ))
        if angular_acceleration is not None and float(
            angular_acceleration
        ) > 0.0:
            bend = np.abs(yaw_derivative(self.yaw_per_metre, self.arclength))
            with np.errstate(divide='ignore'):
                speed = np.minimum(speed, np.where(
                    bend > 1.0e-6,
                    np.sqrt(float(angular_acceleration)
                            / np.maximum(bend, 1.0e-6)),
                    np.inf,
                ))
        speed = np.maximum(speed, 0.0)
        # Both q and speed are interpolated inside a segment. Checking their
        # product only at vertices misses an interior angular-speed peak.
        segment_rate = np.maximum(rate[:-1], rate[1:])
        if angular_speed is not None and angular_speed > 0.0:
            segment_cap = np.divide(
                float(angular_speed), segment_rate,
                out=np.full_like(segment_rate, np.inf),
                where=segment_rate > 1.0e-9)
            speed[:-1] = np.minimum(speed[:-1], segment_cap)
            speed[1:] = np.minimum(speed[1:], segment_cap)

        # alpha = q'(s) v^2 + q(s) a, q = d(yaw)/ds. Use BOTH terms,
        # without reserving an arbitrary half of alpha for each. With u=v^2
        # and a=(u1-u0)/(2 ds), a conservative bound over the whole segment is
        # B*max(u0,u1) + Q*abs(u1-u0)/(2 ds) <= angular_acceleration,
        # where B=abs(q') and Q=max(abs(q0),abs(q1)). It is linear in the
        # larger endpoint u, so each forward/backward reach is analytic.
        # This allows the unused acceleration budget to support cruise and
        # avoids braking simply because one of the two terms was allocated
        # a fixed 50% share. The path and heading geometry are unchanged.
        bend = np.zeros(len(segments))
        yaw_budget = None
        conservative_accel = None
        if angular_acceleration is not None and angular_acceleration > 0.0:
            yaw_budget = float(angular_acceleration)
            bend = np.abs(np.divide(
                np.diff(self.yaw_per_metre), segments,
                out=np.zeros_like(segments), where=segments > 1.0e-12))
            turning = bend > 1.0e-9
            curve_cap = np.sqrt(np.divide(
                yaw_budget, bend, out=np.full_like(bend, np.inf),
                where=turning))
            speed[:-1] = np.minimum(speed[:-1], curve_cap)
            speed[1:] = np.minimum(speed[1:], curve_cap)
            if np.max(segment_rate, initial=0.) * max(acceleration, deceleration) > yaw_budget:
                # Rotation-dominated short moves need the established reserve:
                # with uncertain gain and 300 ms delay, fully using alpha made
                # terminal yaw settle too late. Select by physical demand,
                # not route IDs or a fitted distance threshold.
                curve_cap = np.sqrt(np.divide(
                    .5*yaw_budget, bend, out=np.full_like(bend, np.inf),
                    where=turning))
                speed[:-1] = np.minimum(speed[:-1], curve_cap)
                speed[1:] = np.minimum(speed[1:], curve_cap)
                remaining_budget = yaw_budget-bend*np.maximum(speed[:-1], speed[1:])**2
                conservative_accel = np.minimum(max(acceleration, deceleration), np.divide(
                    remaining_budget, segment_rate,
                    out=np.full_like(segment_rate, np.inf), where=segment_rate > 1.e-9))

        def reachable_squared(index, known_speed, budget):
            ds = float(segments[index])
            u = float(known_speed)**2
            if conservative_accel is not None:
                return max(0., u + 2.*min(budget, float(conservative_accel[index]))*ds)
            reachable = u + 2.0 * budget * ds
            if yaw_budget is not None and ds > 1.e-12:
                slope = float(segment_rate[index]) / (2.0 * ds)
                denominator = float(bend[index]) + slope
                if denominator > 1.e-12:
                    reachable = min(reachable, (yaw_budget + slope*u) / denominator)
            return max(0., reachable)

        # 2) 前向き掃引: 今の速度から加速できる範囲に抑える
        speed[0] = min(speed[0], max(0.0, entry_speed))
        for index in range(1, len(speed)):
            speed[index] = min(speed[index], math.sqrt(
                reachable_squared(index - 1, speed[index - 1], acceleration)))

        # 3) 後ろ向き掃引: 終端で0に落とせる範囲に抑える
        speed[-1] = 0.0
        for index in range(len(speed) - 2, -1, -1):
            speed[index] = min(speed[index], math.sqrt(
                reachable_squared(index, speed[index + 1], deceleration)))
        self.speed = speed

        # 4) 台形則で時刻を積む
        times = np.zeros(len(speed))
        for index in range(1, len(speed)):
            mean_speed = max(0.5 * (speed[index - 1] + speed[index]), 1.0e-4)
            times[index] = times[index - 1] + segments[index - 1] / mean_speed
        self.time = times
        self.duration = float(times[-1])

    def sample(self, at_time: float):
        """時刻 t の基準 (位置, 世界系速度, 姿勢, 角速度, 弧長) を返す。"""
        if self.length < 1.0e-9:
            return (self.points[-1].copy(), np.zeros(2),
                    wrap(float(self.yaws[-1])), 0.0, 0.0)
        clamped = float(np.clip(at_time, 0.0, self.duration))
        # The forward/backward passes assume constant acceleration per
        # segment. Integrate that same law; interpolating s(t) and then v(s)
        # independently gives contradictory position/velocity references.
        index = min(int(np.searchsorted(self.time, clamped, side='right')) - 1,
                    len(self.time) - 2)
        index = max(0, index)
        dt = float(self.time[index + 1] - self.time[index])
        elapsed = clamped - float(self.time[index])
        accel = ((self.speed[index + 1] - self.speed[index]) / dt
                 if dt > 1.0e-12 else 0.0)
        s = float(np.clip(
            self.arclength[index] + self.speed[index] * elapsed
            + 0.5 * accel * elapsed ** 2,
            self.arclength[index], self.arclength[index + 1]))
        position = np.array([
            np.interp(s, self.arclength, self.points[:, 0]),
            np.interp(s, self.arclength, self.points[:, 1]),
        ])
        speed = max(0.0, float(self.speed[index] + accel * elapsed))
        yaw = float(np.interp(s, self.arclength, self.yaws))
        tangent = self._tangent(s)
        # 角速度は姿勢の弧長微分に速度を掛けたもの。微分は構築時に中心差分で
        # 一度だけ作ってあるので、ここは連続な配列の補間になる。二点差分を
        # その場で取ると、標本間隔をまたぐたびに角速度が飛んでいた。
        # 姿勢列そのものが車輪バジェットと角加速度を考えて作られているので、
        # この角速度は実現可能である。
        yaw_rate = float(
            np.interp(s, self.arclength, self.yaw_per_metre)) * speed
        return position, tangent * speed, wrap(yaw), yaw_rate, s

    def _tangent(self, s: float) -> np.ndarray:
        step = max(1.0e-3, 0.5 * float(np.mean(np.diff(self.arclength))))
        ahead = min(self.length, s + step)
        behind = max(0.0, s - step)
        delta = np.array([
            np.interp(ahead, self.arclength, self.points[:, 0])
            - np.interp(behind, self.arclength, self.points[:, 0]),
            np.interp(ahead, self.arclength, self.points[:, 1])
            - np.interp(behind, self.arclength, self.points[:, 1]),
        ])
        norm = float(np.linalg.norm(delta))
        return delta / norm if norm > 1.0e-9 else np.array([1.0, 0.0])

    def project(self, position: np.ndarray) -> float:
        if self.length < 1.0e-9:
            return 0.0
        # Project onto segments, not the nearest 5 cm vertex. Vertex snapping
        # made the progress watchdog and terminal handoff jump by a full cell.
        delta = np.diff(self.points, axis=0)
        length2 = np.sum(delta * delta, axis=1)
        fraction = np.clip(np.divide(
            np.sum((position - self.points[:-1]) * delta, axis=1), length2,
            out=np.zeros_like(length2), where=length2 > 1.0e-18), 0.0, 1.0)
        nearest = self.points[:-1] + fraction[:, None] * delta
        index = int(np.argmin(np.sum((nearest - position) ** 2, axis=1)))
        return float(self.arclength[index]
                     + fraction[index] * math.sqrt(length2[index]))

    def time_at_arclength(self, arclength: float) -> float:
        """Inverse of the same constant-acceleration clock used by sample."""
        if self.length < 1.0e-9:
            return 0.0
        s = float(np.clip(arclength, 0.0, self.length))
        index = min(int(np.searchsorted(self.arclength, s, side='right')) - 1,
                    len(self.arclength) - 2)
        index = max(0, index)
        ds = s - float(self.arclength[index])
        dt = float(self.time[index + 1] - self.time[index])
        accel = ((self.speed[index + 1] - self.speed[index]) / dt
                 if dt > 1.0e-12 else 0.0)
        speed = math.sqrt(max(0.0, self.speed[index] ** 2 + 2 * accel * ds))
        denominator = float(self.speed[index]) + speed
        elapsed = 2 * ds / denominator if denominator > 1.0e-12 else 0.0
        return float(self.time[index] + elapsed)


def to_body(world_velocity, yaw: float) -> np.ndarray:
    """世界系の速度を、姿勢 yaw の機体系へ落とす。"""
    cosine, sine = math.cos(float(yaw)), math.sin(float(yaw))
    return np.array([
        world_velocity[0] * cosine + world_velocity[1] * sine,
        -world_velocity[0] * sine + world_velocity[1] * cosine,
    ])


def track_progress(distance, best, best_at, now, epsilon):
    """目標までの距離の最良値と、それを更新できていない時間を返す。

    ``(best, best_at, stalled_sec)``。epsilon より縮んだときだけ更新する。
    """
    if distance < best - float(epsilon):
        return float(distance), float(now), 0.0
    return float(best), float(best_at), float(now) - float(best_at)


def same_endpoint(previous, endpoint, tolerance: float = 0.05) -> bool:
    """差し替えた経路が、同じ場所を終点にしているか。"""
    if previous is None:
        return False
    return bool(
        np.linalg.norm(np.asarray(endpoint, dtype=float) - previous) <= tolerance
    )


def terminal_latch(since, remaining_arc: float, zone: float, now: float):
    """終端ゾーンに入った時刻。ゾーンを明確に出たら外す。

    ``since`` が None 以外なら、その時刻からずっとゾーンの中にいる。

    停滞判定（``track_progress``）は終端ゾーンを除外している。目標のそばで
    残り弧長が縮まないのは正常な整定でも起きるからである。そのぶん、ゾーンの
    中を回り続ける破綻はどこにも引っかからない。ここで入った時刻を掛けて
    おき、公差へ入れないまま時間切れになったら止める。

    外すのは ``2 * zone`` を超えたときだけである。半径 0.05 m 前後の円を
    描いていると残り弧長は zone の境界をまたいで往復するので、境界そのもので
    外すと掛け金が周回ごとにリセットされ、時間切れが成立しない。
    """
    if remaining_arc <= zone:
        return float(now) if since is None else since
    if remaining_arc > 2.0 * zone:
        return None
    return since


def direction_gap(commanded, measured, minimum: float = 0.05):
    """指令した進行方向と実際に出た進行方向の角度差（度）。

    どちらかが遅すぎて向きが定まらないときは ``None``。

    これは診断であって制御ではない。指令を機体系へ落とす回転が実際の向きから
    ずれていると（計測輪やミキサの符号違い、姿勢推定のずれ）、位置のP制御は
    誤差を目標方向ではなく横へ倒すので、機体は目標のまわりを回り続ける。
    その「ずれ」をそのまま角度で出しておけば、どの符号を疑えばよいかが
    走行ログだけで分かる。90 度付近ならミキサの x/y 入れ替え、180 度付近なら
    符号反転、緩やかな一定値なら姿勢推定のオフセットである。
    """
    a = float(np.linalg.norm(commanded))
    b = float(np.linalg.norm(measured))
    if a < minimum or b < minimum:
        return None
    cross = float(commanded[0] * measured[1] - commanded[1] * measured[0])
    dot = float(commanded[0] * measured[0] + commanded[1] * measured[1])
    return math.degrees(math.atan2(cross, dot))


def terminal_yaw_reference(yaw, yaw_rate, goal_yaw, remaining, zone):
    """Blend out the moving reference before the final position servo starts."""
    if zone <= 0.0:
        return yaw, yaw_rate
    fraction = float(np.clip((2.0 * zone - remaining) / zone, 0.0, 1.0))
    weight = fraction * fraction * (3.0 - 2.0 * fraction)
    return (wrap(yaw + weight * wrap(goal_yaw - yaw)),
            (1.0 - weight) * yaw_rate)


def terminal_translation(velocity, position, goal, remaining, zone, gain, cap):
    """Continuously hand over to the capped position servo over the last 2 zones."""
    if zone <= 0.0:
        return velocity
    fraction = float(np.clip((2.0 * zone - remaining) / zone, 0.0, 1.0))
    weight = fraction * fraction * (3.0 - 2.0 * fraction)
    target = float(gain) * (goal - position)
    speed = float(np.linalg.norm(target))
    if speed > cap:
        target *= max(0.0, cap) / speed
    return (1.0 - weight) * velocity + weight * target


def entry_time(trajectory: Trajectory, pose) -> float:
    """差し替えた軌道の、いまの機体位置に対応する基準時刻。"""
    if pose is None:
        return 0.0
    start = trajectory.project(np.asarray(pose, dtype=float)[:2])
    return trajectory.time_at_arclength(start)


class TrajectoryTracker(StagedHeadingMixin, Node):
    def __init__(self) -> None:
        super().__init__('trajectory_tracker')
        defaults = {
            'plan_topic': '/plan',
            'active_goal_topic': '/navigation/active_goal',
            'pose_topic': '/localization/pose',
            'odom_topic': '/wheel/odometry',
            'output_topic': '/cmd_vel_nav_smoothed',
            'status_topic': '/trajectory_tracker/status',
            # behavior_server の復帰動作 (Spin / BackUp / DriveOnHeading) の
            # 出口。MPPI モードでは velocity_smoother がこれを読むが、追従器
            # モードでは smoother は cmd_vel_mppi_idle へ張り替わるので、
            # 誰も購読していなかった。BT は詰まると復帰動作を回すが機体は
            # 動かず、進捗チェッカが再び落ちて目標が abort する。ここで
            # 中継する。
            'behavior_topic': '/cmd_vel_nav',
            # RuntimeGuard reports both the motor gate's applied_scale and
            # reference_scale (the healthy planner's clock). A stale output
            # must not prevent the controller from generating its replacement.
            'safety_state_topic': '/system/safety_state',
            # 中継した指令を保持する時間。behavior_server は動作中 20 Hz で
            # 出すので、これを過ぎたら動作は終わっている。
            'behavior_timeout_sec': 0.5,
            'robot_config_file': '',
            'runtime_config_file': '',
            'nav2_config_file': '',
            'routes_config_file': '',
            'field_config_file': '',
            'footprint_config_file': '',
            'footprint_profile': 'NORMAL',
            # collision_monitor の FootprintApproach は、固定バケツ脇では
            # 1.2 秒先までフットプリントを掃引する。100 Hz で指令を渡すと
            # コールバックが滞留し、安全指令が RuntimeGuard の 0.25 秒
            # watchdog に間に合わない。30 Hz へ落とした後も、固定バケツ脇で
            # FootprintApproach が連続すると出力が 0.47〜0.68 秒途切れる例が
            # 残った。20 Hz は Nav2 controller と同じ周期で、1指令あたりの
            # swept-footprint 判定を完了させつつ追従帯域を保つ。
            'control_rate_hz': 20.0,
            'resample_spacing_m': 0.05,
            # グリッド階段を落とす平滑長。planner のセルは 0.05 m なので
            # 階段の周期は最大でも 3 セルほどであり、0.12 m で消える。
            # 大きくすると壁を回り込む区間で内側を削るので、セル数個に
            # とどめる（0 で無効）。
            'path_smoothing_m': 0.12,
            # 姿勢最適化の間隔。細かくすると解が良くなるが、点数 x 候補数
            # だけ余裕評価が増える。0.30 m なら 3.6 m の経路で13点。
            'yaw_plan_spacing_m': 0.30,
            # 姿勢列の角を丸める長さ。動的計画法は離散候補から選ぶので出力は
            # 折れ線で、そのままでは角速度が節点ごとに階段状に飛ぶ。節点間隔
            # より少し短く取ると、折れ目だけが丸まって計画そのものは残る
            # （0 で無効）。角加速度の上限はこの長さで決まる:
            # |yaw''| ~ 節点の角速度差 / この長さ。
            'yaw_smoothing_m': 0.25,
            # 姿勢最適化。system.launch.py の既定は false で、その判断は
            # いまも測定で支持されている（下の「なぜ既定は false か」）。
            # ここが true のときの計画コストは実測 240-280 ms/経路であり、
            # 候補数に比例する。1 Hz の再計画に対して worker スレッドで
            # 走るので制御周期は乱さないが、Jetson の1コアの4分の1を使う。
            # 既定の線形配分は同じ経路で 0.2 ms である。
            #
            # ここは system.launch.py の既定と同じ false にしてある。以前は
            # ノード既定が true・launch 既定が false で食い違っていた。
            # launch は必ず値を渡すので運用上は同じだが、このノードを直接
            # 起動して測ると別のものを測ってしまう。
            'optimize_yaw': False,
            'motion_mode': 'simultaneous',
            'clearance_weight': 1.6,
            'minimum_clearance_m': 0.02,
            # /plan の終端をこの距離以内の実目標へ差し替える。
            # SmacPlanner2D の tolerance 0.20 m より大きく取る。
            'goal_snap_distance_m': 0.40,
            'max_reference_lead_m': 0.35,
            'pose_timeout_sec': 0.30,
            'velocity_timeout_sec': 0.20,
            'plan_timeout_sec': 2.0,
            # Keep correction below the delay-limited hardware loop's gain
            # margin. Feed-forward still supplies the planned cruise speed.
            # See HARDWARE_ROBUSTNESS_20260905.md for delayed-plant settling.
            'position_gain': 0.8,
            'position_damping': 0.10,
            # Use 1.0 while the actuator calibration remains unmeasured.
            # A 1.6 loop tolerated gain error at 120+80 ms delay, but the
            # longer-delay replay still hunted in heading after arrival.
            'yaw_gain': 1.0,
            'lateral_acceleration': 1.2,
            # 終端は純Pの低速サーボへ切り替える。速度プロファイルは
            # 「指令が」目標でちょうど0になるように作るが、駆動系には
            # 実測 120 ms の無駄時間と 80 ms の一次遅れがあるため、実機は
            # そのぶん速い状態で目標に着き、行き過ぎてから戻る。実測で
            # Nav2 が 40 mm 以内と判定した 1 秒後に 145 mm ずれていた。
            # 低速で寄せれば行き過ぎ量は速度に比例して小さくなる。
            #
            # 0.16 m / 0.16 m/s は、行き過ぎの原因が「指令が目標でゼロに
            # なるだけ」だった時期の値である。いまは feedback_delay_sec の
            # 遅れ補償が行き過ぎ自体を消しており、さらに姿勢を終端の手前で
            # 入れ終える（yaw_cutoff）ので、この区間は純粋な並進の整定に
            # なった。同じ 4 区間を同じ機体モデルで掃引した結果:
            #
            #   ゾーン/上限   到着          行き過ぎ  整定    到着時の姿勢誤差
            #   0.16 / 0.16   6.42/4.99/6.61/5.56 s  0.3 mm  0.3 mm  0.02-0.30 deg
            #   0.10 / 0.30   6.28/4.51/5.93/4.91 s  0.3 mm  0.3 mm  0.01-0.92 deg
            #   0    / 0      6.42/4.41/5.93/4.87 s  16.7 mm 16.7 mm 0.01-1.85 deg
            #
            # 精度は変わらず 5〜12% 速い。ゾーンを外すと整定が 16.7 mm へ
            # 崩れるので、低速サーボ自体は要る。
            'terminal_approach_m': 0.10,
            'terminal_speed': 0.30,
            # 旋回指令を、フットプリントが collision_monitor の外挿時間
            # (nav2_next.yaml の time_before_collision) のあいだに実際に
            # 掃ける角度の中へ収める。壁ぎわの発射姿勢の回転余裕は 10〜24 度
            # しかないのに、1.2 s のあいだ角速度を一定と見なす外挿は
            # balanced の 1.30 rad/s で 89 度も回すので、通常の旋回はどれも
            # 「衝突する」と判定され、approach が並進ごと twist 全体を
            # contact_time / 1.2 倍に絞っていた。これが報告された「急に
            # 遅くなって進まない」である。詳細は
            # rl_residual.limit_yaw_rate。
            'limit_yaw_rate_to_clearance': True,
            # 駆動系の無駄時間＋一次遅れの合計。フィードバックはこの
            # 時間だけ先の姿勢に対して誤差を取る。指令側だけを目標で
            # ゼロにすると、機体はまだ動いている状態で目標に着くので
            # 行き過ぎる。実測で 130 mm 行き過ぎ、戻るのに 2 秒かかって
            # いた。地点5の余裕は 59 mm なので接触する量である。
            'feedback_delay_sec': 0.20,
            # Enabled by matching hardware calibration; corridor eligibility
            # applies unless sprint_turn_everywhere is selected.
            'predictive_sprint': False,
            # Release location/direction turn budgets in calibrated sprint.
            'sprint_turn_everywhere': False,
            # 計測輪速度の一次遅れ時定数。速度差分の雑音が予測・制動へ
            # 直接入るのを抑える。0でフィルターを無効にする。
            'velocity_filter_sec': 0.06,
            # 遅れ補償で回す角度の上限。計測輪の角速度が壊れた（配線・符号・
            # スケール）ときに指令の座標系が現実と大きくずれるのを止める。
            # 0.35 rad は yaw_limit 1.75 rad/s * 0.20 s に相当する。
            'max_predicted_yaw_rad': 0.35,
            'terminal_hold_sec': 3.0,
            # 終端ゾーンに入ってから、目標公差へ入れないまま回り続けてよい
            # 時間。基準プロファイルを撃ち切ったかどうかとは別の掛け金に
            # している。指令の座標系が実際の向きからずれていると、終端の
            # P制御は誤差を横へ倒すので機体は目標のまわりを一定半径で回る。
            # そのとき残り弧長は縮まないが、経路は 1 Hz で出し直されるので
            # 基準時刻は毎回入口へ置き直され、duration には届かない。
            # 0.16 m のゾーンを 0.16 m/s の上限で詰めるのに要るのは 1 秒
            # ほどなので、4 秒あれば正常な整定を切ることはない。
            'terminal_timeout_sec': 4.0,
            # 指令した進行方向と実際に出た進行方向の食い違いを、この時定数で
            # 均して状態に出す。0 で無効。
            'flow_filter_sec': 0.5,
            # 目標へ寄っていないのに走り続けない。指令の座標系と機体の実際の
            # 向きがずれていると、位置のP制御は目標へ収束せず一定の円を描く
            # （回転行列が誤差ベクトルを横へ倒すため）。この破綻は自力では
            # 抜けられないので、止めて状態を出す。
            'no_progress_timeout_sec': 4.0,
            'no_progress_epsilon_m': 0.02,
            # 停滞として数えるのは「動いているのに寄らない」ときだけ。障害物
            # で collision monitor に止められている間や非常停止中は、寄らない
            # のは当たり前なので数えない。数えると 4 秒待たされたあと復帰でき
            # なくなる。
            'no_progress_speed_m_s': 0.05,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

        robot_file = str(self.get_parameter('robot_config_file').value)
        runtime_file = str(self.get_parameter('runtime_config_file').value)
        nav2_file = str(self.get_parameter('nav2_config_file').value)
        if not robot_file or not runtime_file or not nav2_file:
            raise ValueError(
                'robot_config_file, runtime_config_file and nav2_config_file '
                'are required')
        robot = load_robot(robot_file)
        self.base_frame = robot['base_frame_id']
        self.odom_frame = robot['measurement_wheels']['odom_frame_id']
        self.envelope = OmniEnvelope(robot['drivetrain'])
        self.default_max_wheel_speed = self.envelope.max_wheel
        self.profile_max_wheel_speeds = robot['drivetrain'].get('profile_max_wheel_speeds', {})
        self.profile_linear_accelerations = robot['drivetrain'].get('profile_linear_accelerations', {})

        with open(runtime_file, encoding='utf-8') as stream:
            runtime = yaml.safe_load(stream)['runtime_guard']['ros__parameters']
        # プロファイルは全部読む。運転中に GUI から切り替わるので、既定値を
        # 起動時に一度だけ焼き付けてはいけない。
        self.profiles = json.loads(runtime['profiles_json'])
        self.profile_name = str(runtime['default_profile'])
        # RuntimeGuard が実際に通している一様な速度倍率。操作卓のスライダ、
        # 学習残差の減速、地点まわりの red zone をすべて含む。
        #
        # ここが無かったのは実害のある穴だった。追従器モードでは
        # velocity_smoother を迂回しているので、指令を作るのはこのノードだけ
        # である。にもかかわらず倍率を知らないので、
        #
        #   * 操作卓でプロファイルを sprint にしても軌道は balanced のまま
        #     作られ、速くならない。
        #   * ACCEPTANCE step 3 の「まず 10% で動かす」では、0.78 m/s の
        #     時間割りに対してゲートは 0.078 m/s しか通さない。
        #   * 学習残差の speed_scale は配備ポリシーで 0.35〜1.0 を取り、
        #     red zone は全地点の 0.95 m 以内で 0.70 を掛ける。つまり目標の
        #     手前ではつねに 30% 速い時間割りを追っていた。
        #
        # 倍率は経路の形を変えない一様なスケールなので、軌道を作り直す必要は
        # ない。基準時刻をその倍率で進め、基準速度と角速度に同じ倍率を掛ける
        # （経路速度スケーリング）。加速度は倍率の2乗で下がるので、常に元の
        # 制約の内側である。
        self.speed_scale = 1.0
        self._apply_profile(self.profile_name)
        # 加速度は整形段 (velocity_smoother) の値を使う。RuntimeGuard の
        # profile はその上に余裕を持たせたゲート値なので、そちらで軌道を
        # 作ると実際には出せない時間割りになる。
        with open(nav2_file, encoding='utf-8') as stream:
            smoother = yaml.safe_load(stream)['velocity_smoother'][
                'ros__parameters']
        self.acceleration = float(smoother['max_accel'][0])
        self.default_acceleration = self.acceleration
        self.deceleration = abs(float(smoother['max_decel'][0]))
        self._apply_profile(self.profile_name)
        self.yaw_acceleration = float(smoother['max_accel'][2])
        self.lateral_acceleration = float(
            self.get_parameter('lateral_acceleration').value)

        # 余裕の計算は姿勢最適化だけのものではない。旋回指令を下流の
        # collision_monitor が通す範囲に収めるのにも使うので、
        # optimize_yaw が false でも読む（以前は true のときだけ読んでいた）。
        self.clearance: Optional[CadClearanceModel] = None
        routes_file = str(self.get_parameter('routes_config_file').value)
        self.fixed_gate_speed = FixedGateSpeed.from_yaml(routes_file) if routes_file else None
        field_file = str(self.get_parameter('field_config_file').value)
        if field_file:
            footprint_file = str(
                self.get_parameter('footprint_config_file').value) or None
            try:
                self.clearance = CadClearanceModel.from_yaml(
                    field_file, footprint_file,
                    str(self.get_parameter('footprint_profile').value))
            except (OSError, ValueError) as error:
                self.get_logger().warning(
                    f'footprint clearance unavailable, planning yaw for speed '
                    f'only and leaving the yaw rate to collision_monitor: '
                    f'{error}')
        # collision_monitor が指令を外挿する時間。配備値をそのまま読む。
        self.monitor_horizon = None
        if bool(self.get_parameter('limit_yaw_rate_to_clearance').value):
            self.monitor_horizon = load_collision_monitor_horizon(nav2_file)

        self.lock = threading.Lock()
        self.stopping = False
        self.pending_plan = None
        self.active_goal = None
        self.plan_event = threading.Event()
        self.trajectory: Optional[Trajectory] = None
        self.reference_time = 0.0
        self.plan_stamp = -math.inf
        self.pose: Optional[np.ndarray] = None
        self.pose_stamp = -math.inf
        self.velocity = np.zeros(3)
        self.velocity_stamp: Optional[float] = None
        self.velocity_source_stamp: Optional[float] = None
        self.odometry_paused = False
        self.command = np.zeros(3)
        self.finished_at: Optional[float] = None
        self.terminal_since: Optional[float] = None
        self.terminal_best_yaw = math.inf
        self.terminal_best_distance = math.inf
        self.command_flow = np.zeros(2)
        self.measured_flow = np.zeros(2)
        self.best_distance = math.inf
        self.best_distance_at = time.monotonic()
        self.tracked_endpoint: Optional[np.ndarray] = None
        self.behavior = np.zeros(3)
        self.behavior_stamp = -math.inf
        self.last_status = ''
        self.plan_build_ms = 0.0
        self.snap_offset = None
        self.snapped = False
        # プロファイルが変わったときに作り直すための、加工前の経路。
        self.last_plan = None
        self._init_staged_heading()

        control_qos = QoSProfile(depth=1)
        control_qos.reliability = ReliabilityPolicy.RELIABLE
        control_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.command_pub = self.create_publisher(
            Twist, str(self.get_parameter('output_topic').value), 5)
        self.status_pub = self.create_publisher(
            String, str(self.get_parameter('status_topic').value), control_qos)
        self.create_subscription(
            Path, str(self.get_parameter('plan_topic').value), self._on_plan, 5)
        self.create_subscription(
            PoseStamped, str(self.get_parameter('active_goal_topic').value),
            self._on_goal, control_qos)
        self.create_subscription(
            PoseWithCovarianceStamped,
            str(self.get_parameter('pose_topic').value), self._on_pose, 5)
        self.create_subscription(
            Odometry, str(self.get_parameter('odom_topic').value),
            self._on_odom, 1)
        self.create_subscription(
            Twist, str(self.get_parameter('behavior_topic').value),
            self._on_behavior, 5)
        self.create_subscription(
            String, str(self.get_parameter('safety_state_topic').value),
            self._on_safety_state, control_qos)

        rate = max(20.0, float(self.get_parameter('control_rate_hz').value))
        self.period = 1.0 / rate
        self.create_timer(self.period, self._tick)
        self.worker = threading.Thread(target=self._plan_worker, daemon=True)
        self.worker.start()
        axis_speed = self.envelope.best_alignment_speed(
            self.speed_limit, self.lateral_limit)
        self.get_logger().info(
            f'Trajectory tracker at {rate:.0f} Hz, profile '
            f'{self.speed_limit:.2f} m/s, body-axis limit {axis_speed:.3f} m/s, '
            f'yaw optimisation='
            f'{bool(self.get_parameter("optimize_yaw").value)}, '
            f'footprint clearance='
            f'{self.clearance is not None}'
        )

    # ------------------------------------------------------------------

    def _apply_profile(self, name: str) -> None:
        """Apply speed and acceleration policy; braking has its own budget."""
        profile = self.profiles.get(name)
        if profile is None:
            return
        self.profile_name = name
        self.speed_limit = float(profile['linear'])
        self.lateral_limit = float(profile['lateral'])
        self.yaw_limit = float(profile['angular'])
        if hasattr(self, 'profile_max_wheel_speeds'):
            self.envelope.max_wheel = self.profile_max_wheel_speeds.get(
                name, self.default_max_wheel_speed)
        if hasattr(self, 'default_acceleration'):
            requested = getattr(self, 'profile_linear_accelerations', {}).get(
                name, self.default_acceleration)
            self.acceleration = min(float(requested), float(profile['linear_accel']))

    def _on_safety_state(self, message: String) -> None:
        """Use the guard's profile and health-gated reference clock.

        reference_scale keeps the reference generating fresh commands after
        downstream silence. The final motor gate still rejects stale commands.
        Old guard versions fall back to applied_scale.
        """
        try:
            state = json.loads(message.data)
        except (TypeError, ValueError):
            return
        if not isinstance(state, dict):
            return
        # A stale downstream command closes the motor gate, but the tracker
        # must still generate a fresh replacement when all health gates pass.
        # Older guards retain the previous applied_scale protocol.
        scale = state.get('reference_scale', state.get('applied_scale'))
        if isinstance(scale, (int, float)) and math.isfinite(float(scale)):
            self.speed_scale = float(np.clip(float(scale), 0.0, 1.0))
        name = state.get('profile')
        if (
            isinstance(name, str)
            and name in self.profiles
            and name != self.profile_name
        ):
            with self.lock:
                self._apply_profile(name)
                # The worker can be building an old speed envelope while this
                # callback changes its limits. Invalidate that commit and any
                # prepared departure, then rebuild the retained source plan.
                self.stage_revision += 1
                self.stage_continuation = None
                # 包絡線の形が変わったので時間割りを作り直す。倍率と違って
                # これは経路上の速度上限そのものを変える。
                self.pending_plan = self.last_plan
            if self.last_plan is not None:
                self.plan_event.set()
            self.get_logger().info(
                f'tracker envelope -> {name} '
                f'({self.speed_limit:.2f}/{self.lateral_limit:.2f} m/s, '
                f'{self.yaw_limit:.2f} rad/s)')

    def _on_pose(self, message: PoseWithCovarianceStamped) -> None:
        if message.header.frame_id != 'map':
            return
        source_ns = message_stamp_nanoseconds(message.header.stamp)
        if source_ns is None:
            return
        pose = np.array([
            message.pose.pose.position.x,
            message.pose.pose.position.y,
            yaw_of(message.pose.pose.orientation),
        ])
        if not np.all(np.isfinite(pose)):
            return
        now = time.monotonic()
        source_stamp = source_ns * 1.e-9
        age = float(self.get_clock().now().nanoseconds) * 1.e-9 - source_stamp
        timeout = float(self.get_parameter('pose_timeout_sec').value)
        if not math.isfinite(age) or age < -.02 or age > timeout:
            return
        with self.lock:
            previous = getattr(self, 'pose_source_stamp', None)
            interrupted = now - self.pose_stamp > timeout
            if not interrupted and previous is not None and source_stamp <= previous:
                return
            self.pose = pose
            self.pose_stamp = now - max(0., age)
            self.pose_source_stamp = source_stamp

    def _on_goal(self, message: PoseStamped) -> None:
        if (message.header.frame_id != 'map'
                or message_stamp_nanoseconds(message.header.stamp) is None):
            return
        goal = np.array([
            message.pose.position.x,
            message.pose.position.y,
            yaw_of(message.pose.orientation),
        ])
        if not np.isfinite(goal).all():
            return
        with self.lock:
            if hasattr(self, 'motion_mode'):
                self._stage_new_goal()
            self.active_goal_stamp = [message.header.stamp.sec, message.header.stamp.nanosec]
            if (self.active_goal is None
                    or np.linalg.norm(goal[:2] - self.active_goal[:2]) > 1.0e-6
                    or abs(wrap(goal[2] - self.active_goal[2])) > 1.0e-6):
                self.finished_at = None
                self.terminal_since = None
                self.terminal_best_yaw = math.inf
                self.terminal_best_distance = math.inf
                self.previous_yaw_error = None
            self.active_goal = goal

    def _on_odom(self, message: Odometry) -> None:
        """計測輪の速度を一次遅れで均してから使う。

        /wheel/odometry の twist は生の1サンプル差分である
        (measurement_wheel_node: ``twist = body_delta / dt``)。フィルタは
        どこにも入っていない。この値は追従器の中で

          位置予測   position_gain * feedback_delay = 0.16
          制動項     position_damping               = 0.10
          姿勢予測   yaw_gain * feedback_delay      = 0.20

        倍されて指令に入るので、エンコーダを微分した雑音がほぼ等倍で
        指令に乗る。合成デモではオドメトリが厳密値だったためこの経路は
        一度も励起されていない。時定数は予測地平 0.20 s に対して十分
        短く取り、予測そのものを鈍らせない。
        """
        if (message.header.frame_id != getattr(self, 'odom_frame', 'odom')
                or message.child_frame_id != getattr(self, 'base_frame', 'base_link')):
            return
        source_ns = message_stamp_nanoseconds(message.header.stamp)
        if source_ns is None:
            return
        sample = np.array([
            message.twist.twist.linear.x,
            message.twist.twist.linear.y,
            message.twist.twist.angular.z,
        ])
        if not np.all(np.isfinite(sample)):
            return
        now = time.monotonic()
        source_stamp = source_ns * 1.e-9
        age = float(self.get_clock().now().nanoseconds) * 1.0e-9 - source_stamp
        timeout = float(self.get_parameter('velocity_timeout_sec').value)
        if not math.isfinite(age) or age < -0.02 or age > timeout:
            return
        tau = float(self.get_parameter('velocity_filter_sec').value)
        with self.lock:
            previous = self.velocity_stamp
            source_previous = self.velocity_source_stamp
            interrupted = previous is None or now - previous > timeout
            if not interrupted and source_previous is not None and source_stamp <= source_previous:
                return
            # Freshness follows acquisition time, not delivery of an old DDS
            # sample. Filter dt follows the sensor clock so batched callbacks
            # do not alternately over-weight and ignore velocity readings.
            self.velocity_stamp = now - max(0.0, age)
            self.velocity_source_stamp = source_stamp
            if interrupted or source_previous is None or tau <= 0.0:
                self.velocity = sample
                return
            dt = max(0.0, source_stamp - source_previous)
            self.velocity = self.velocity + (
                1.0 - math.exp(-dt / tau)) * (sample - self.velocity)

    def _on_behavior(self, message: Twist) -> None:
        """復帰動作の指令を控えておく。流すかどうかは _tick が決める。"""
        sample = np.array([
            message.linear.x, message.linear.y, message.angular.z])
        if not np.all(np.isfinite(sample)):
            return
        with self.lock:
            self.behavior = sample
            self.behavior_stamp = time.monotonic()

    def _relay_behavior(self, now: float) -> bool:
        if getattr(self, 'motion_mode', 'simultaneous') == 'staged_heading':
            # Nav2 recovery Spin/BackUp must not bypass the selected gate.
            return False
        """復帰動作が出ていればそれを流す。流したら True。

        追従できる軌道が無いときにだけ呼ぶこと。追従中に呼ぶと、経路追従の
        指令と復帰動作が同じ出口を奪い合う。
        """
        with self.lock:
            command = self.behavior.copy()
            stamp = self.behavior_stamp
        if now - stamp > float(
            self.get_parameter('behavior_timeout_sec').value
        ):
            return False
        vx, vy, wz = float(command[0]), float(command[1]), float(command[2])
        # 復帰動作は Nav2 側の速度制限しか通っていない。追従器の出口は
        # velocity_smoother を迂回しているので、通常経路と同じ包絡線と
        # 加速度制限をここで掛ける。
        ellipse = math.hypot(
            vx / max(self.speed_limit, 1.0e-6),
            vy / max(self.lateral_limit, 1.0e-6),
        )
        if ellipse > 1.0:
            vx /= ellipse
            vy /= ellipse
        wz = float(np.clip(wz, -self.yaw_limit, self.yaw_limit))
        # Spin も同じ壁に阻まれる。復帰動作は自己位置に依らない機体系の指令
        # なので、姿勢が古いときは絞らずにそのまま流す。
        with self.lock:
            pose = None if self.pose is None else self.pose.copy()
            pose_stamp = self.pose_stamp
        if pose is not None and now - pose_stamp <= float(
            self.get_parameter('pose_timeout_sec').value
        ):
            wz = self._monitor_safe_yaw_rate(wz, pose[:2], float(pose[2]))
        cost = self.envelope.wheel_cost(vx, vy, wz)
        if cost > self.envelope.max_wheel:
            scale = self.envelope.max_wheel / cost
            vx, vy, wz = vx * scale, vy * scale, wz * scale
        vx, vy, wz = self._rate_limit(vx, vy, wz)
        self._publish(vx, vy, wz)
        # 復帰動作が動かしている間は停滞時間を進めない。動作が終わって経路が
        # 戻ってきた瞬間に NO_PROGRESS で止まっては、復帰した意味が無い。
        # 終端ゾーンの掛け金も同じ理由で外す。
        self.best_distance_at = now
        self.terminal_since = None
        return True

    def _on_plan(self, message: Path) -> None:
        """新しい経路を worker へ渡すだけ。ここでは何も計算しない。

        姿勢最適化は経路長に比例して数十 ms かかる。それを購読コールバック
        でやると 20 Hz の制御タイマーがその間止まり、指令が途切れる。
        古い軌道を追い続けたまま裏で作り、出来たら差し替える。
        """
        frame_valid = (message.header.frame_id == 'map'
                       and all(p.header.frame_id == 'map' for p in message.poses))
        source_valid = (message_stamp_nanoseconds(message.header.stamp) is not None
                        and all(message_stamp_nanoseconds(p.header.stamp) is not None
                                for p in message.poses))
        points = np.array(
            [[p.pose.position.x, p.pose.position.y] for p in message.poses])
        geometry_valid = (len(points) > 0 and np.isfinite(points).all()
                          and all(math.isfinite(yaw_of(p.pose.orientation))
                                  for p in message.poses))
        if not frame_valid or not source_valid or not geometry_valid:
            with self.lock:
                self.stage_revision += 1
                self.trajectory = None
                self.pending_plan = None
                self.last_plan = None
                self.stage_continuation = None
                self.stage_blocked = ('PLAN_FRAME_MISMATCH' if not frame_valid
                                      else 'INVALID_GLOBAL_PLAN')
            return
        goal_yaw = yaw_of(message.poses[-1].pose.orientation)
        with self.lock:
            # An action for the previous bucket may finish planning after a
            # new goal has arrived. It must not refresh the watchdog or replace
            # the new route with a route back to the old bucket.
            if (self.active_goal is not None and np.linalg.norm(
                    points[-1] - self.active_goal[:2]) > float(
                        self.get_parameter('goal_snap_distance_m').value)):
                return
            self.plan_received_stamp = time.monotonic()
            self.pending_plan = (
                points, goal_yaw)
            self.last_plan = self.pending_plan
        self.plan_event.set()

    def _plan_worker(self) -> None:
        while not self.stopping:
            if not self.plan_event.wait(timeout=0.2):
                continue
            self.plan_event.clear()
            with self.lock:
                pending = self.pending_plan
                self.pending_plan = None
            if pending is None:
                continue
            try:
                self._build_trajectory(*pending)
            except Exception as error:  # 追従を止めないためここで受ける
                self.get_logger().warning(f'trajectory build failed: {error}')

    def _build_trajectory(self, points: np.ndarray, goal_yaw: float) -> None:
        started = time.monotonic()
        with self.lock:
            staged = getattr(self, 'motion_mode', 'simultaneous') == 'staged_heading'
            if getattr(self, 'reverse_goal', None) is not None:
                return
            revision = getattr(self, 'stage_revision', 0)
            stage = copy.deepcopy(getattr(self, 'heading_stage', None))
            if not getattr(self, 'stage_goal_enabled', True):
                return
            pose = None if self.pose is None else self.pose.copy()
            # 実測速度を初速にする。直前の指令値を使うほうが理屈は通るが、
            # 実測では 4 <-> 5 往復が 14.4-15.3 s から 19.7-21.2 s へ悪化し、
            # 1 区間がタイムアウトした（速い初速でプロファイルが短くなり、
            # 遅れて追う実機に対して基準先行の制限が効きやすくなるため）。
            # 理屈より計測を採る。
            entry_speed = float(np.linalg.norm(self.velocity[:2]))
        if pose is None:
            return
        # Build the departure while the stationary rotation owns the output.
        # Only a goal/revision/profile-matched result may be handed over.
        warm_departure = staged and stage is not None and stage.phase in ('SETTLE', 'ROTATE')
        if warm_departure:
            pose = np.array([*stage.gate, stage.target])
            stage.phase = 'TRANSLATE'
            entry_speed = 0.
        points = remaining_path(points, pose[:2])
        # /plan の終端は目標ではない。SmacPlanner2D の tolerance は 0.20 m で、
        # 目標セルが占有されているとき（地点4〜7は余裕 47〜59 mm）その手前で
        # 終わる経路を返す。経路だけを追うと最大 0.20 m 手前で停まる。実測で
        # 66〜77 mm ずれていた。goal_bridge が出す実目標へ終端を差し替える。
        with self.lock:
            goal_pose = self.active_goal
        snap_offset = None
        snapped = False
        if goal_pose is not None:
            snap_offset = float(np.linalg.norm(goal_pose[:2] - points[-1]))
            if snap_offset <= float(
                self.get_parameter('goal_snap_distance_m').value
            ):
                if snap_offset > 1.0e-3:
                    points = np.vstack((points, goal_pose[:2]))
                goal_yaw = float(goal_pose[2])
                snapped = True
            elif not staged:
                # Recheck in the worker: the goal can change after reception
                # but before this queued plan starts building.
                return
        with self.lock:
            previous = self.trajectory
        turn_everywhere = bool(
            not staged and self.get_parameter('sprint_turn_everywhere').value
            and self.get_parameter('predictive_sprint').value
            and getattr(self, 'profile_name', '') == 'sprint')
        profile_key = (self.speed_limit,self.lateral_limit,self.yaw_limit,
                       self.acceleration,self.lateral_acceleration,self.yaw_acceleration,
                       getattr(self, 'deceleration', self.acceleration))
        if turn_everywhere:
            profile_key += ('sprint_turn_everywhere',)
        if (not warm_departure and previous is not None and snapped
                and getattr(self,'trajectory_profile_key',None) == profile_key
                and (not staged or (stage is not None and stage.phase == 'TRANSLATE'))
                and not getattr(self, 'stage_blocked', None)
                and abs(wrap(float(previous.yaws[-1])-goal_yaw)) < 1.e-6
                and same_remaining_corridor(previous.points,points,pose[:2])):
            with self.lock:
                if self.trajectory is previous and revision == getattr(self,'stage_revision',0):
                    self.plan_stamp = time.monotonic()
                    self.plan_build_ms = 1000.*(time.monotonic()-started)
            return
        spacing = float(self.get_parameter('resample_spacing_m').value)
        points = resample(points, spacing)
        if len(points) == 0:
            return
        # 接線を取る前に階段を落とす。ここを飛ばすと、フィードフォワード
        # 速度の向きがセルごとに振れる（斜め区間で 45 度、±0.53 m/s）。
        points = smooth_path(
            points, spacing,
            float(self.get_parameter('path_smoothing_m').value))
        if staged:
            try:
                if goal_pose is None or snap_offset is None or not snapped:
                    raise ValueError('PLAN_GOAL_MISMATCH')
                if stage is None:
                    stage = HeadingStage(float(pose[2]), goal_yaw)
                if stage.phase not in ('SETTLE', 'ROTATE'):
                    if self.clearance is None or self.clearance.footprint is None:
                        raise ValueError('FOOTPRINT_UNAVAILABLE')
                    headings = [stage.target] if stage.phase == 'TRANSLATE' else [stage.heading, stage.target]
                    points = repair_corridor(self.clearance, points, headings)
                stage, points, fixed_yaw = prepare_stage(
                    self.clearance, points, pose, stage,
                    stopping_distance=(entry_speed*.20 + entry_speed**2/(
                                       2*getattr(self, 'deceleration', self.acceleration))
                                       + .03 if entry_speed > .02 else 0.))
            except ValueError as error:
                with self.lock:
                    if revision == self.stage_revision and not warm_departure:
                        self.trajectory = None
                        self.stage_blocked = str(error)
                return
            if (not warm_departure and previous is not None
                    and getattr(self, 'trajectory_profile_key', None) == profile_key
                    and not getattr(self, 'stage_blocked', None)
                    and abs(wrap(float(previous.yaws[-1])-fixed_yaw)) < 1.e-6
                    and same_remaining_corridor(previous.points, points, pose[:2])):
                with self.lock:
                    if self.trajectory is previous and revision == self.stage_revision:
                        self.plan_stamp = time.monotonic()
                        self.plan_build_ms = 1000.*(time.monotonic()-started)
                return
            # Gate truncation/repair can recreate a two-point rest-to-rest
            # trajectory AFTER the initial resample. Keep an acceleration
            # midpoint even in the final centimetres of a gate approach.
            if len(points) == 2:
                points = np.vstack((points[0], .5*(points[0]+points[1]), points[1]))
        elif (getattr(self, 'clearance', None) is not None
                and self.clearance.footprint is not None
                and not bool(self.get_parameter('optimize_yaw').value)
                and abs(wrap(goal_yaw-float(pose[2]))) < 1.e-6):
            # Fixed-heading routes in simultaneous mode need the same corner
            # repair as staged translation. Smoothing alone can cut into a
            # wall's clearance envelope and provoke downstream stop/restarts.
            points = repair_corridor(self.clearance, points, [goal_yaw])
            _, margins = path_clearances(self.clearance, points, goal_yaw)
            if np.min(margins) < .025:
                # Repair is bounded, not a replacement global planner. Never
                # execute an unchecked connector or move the requested goal.
                with self.lock:
                    if (self.trajectory is previous
                            and revision == getattr(self, 'stage_revision', 0)):
                        self.trajectory = None
                self.get_logger().warning('fixed-heading path clearance blocked')
                return
        tangents = path_tangents(points)
        segments = np.linalg.norm(np.diff(points, axis=0), axis=1)
        arclength = np.concatenate(([0.0], np.cumsum(segments)))

        # 姿勢は「終端の寄せが始まるまでに」入れ終える。以前は目標位置と
        # 同時に入れ終えていたので、最後の低速区間でも基準姿勢が動き続け、
        # 到着判定の瞬間の姿勢誤差はその区間の速度に比例していた。終端ゾーン
        # を速くすると姿勢誤差が増える形で現れる（実測、区間 1->2 で
        # 0.16 m/s のとき 1.30 deg が 0.35 m/s では 1.74 deg、公差 2.0 deg）。
        # 分けておけば終端は純粋な並進の整定になり、旋回はその前に終わって
        # いる。並進側の距離が 0.16 m 短くなるぶん旋回はわずかに速くなるが、
        # 3.5 m の区間では 4.7% で、車輪バジェットには収まる。
        yaw_cutoff = max(
            arclength[-1] - terminal_zone(self, stage),
            0.5 * arclength[-1],
        )
        if staged:
            yaws = np.full(len(points), fixed_yaw)
        elif len(points) == 1:
            yaws = np.array([goal_yaw])
        elif bool(self.get_parameter('optimize_yaw').value):
            yaws = self._plan_yaw(
                points, tangents, arclength, float(pose[2]), goal_yaw,
                yaw_cutoff)
        else:
            # 従来どおり弧長に対して線形に配る。
            fraction = np.clip(arclength / max(yaw_cutoff, 1.0e-6), 0.0, 1.0)
            yaws = pose[2] + wrap(goal_yaw - pose[2]) * fraction

        # When translation starts along a drive axis, a linear yaw ramp
        # spends the acceleration phase turning away from its strongest wheel
        # direction. Ease the ramp to sustain fast translation and rotation
        # together. Diagonal starts retain their useful mid-route alignment.
        # Both heading sweeps must fit the full open-corridor reserve.
        if (not staged and len(points) > 1
                and getattr(self, 'profile_name', '') == 'sprint'
                and self.get_parameter('predictive_sprint').value
                and np.max(np.abs(to_body(tangents[0], float(pose[2])))) >= .95
                and np.all(sprint_turn_clearance(
                    points, yaws, getattr(self, 'clearance', None)))):
            fraction = np.clip(arclength/max(yaw_cutoff, 1.e-6), 0., 1.)
            eased = pose[2]+wrap(goal_yaw-pose[2])*(3*fraction**2-2*fraction**3)
            if np.all(sprint_turn_clearance(points, eased, getattr(self, 'clearance', None))):
                yaws = eased

        planning_envelope = self.envelope
        linear_limit, lateral_limit = self.speed_limit, self.lateral_limit
        limits, _ = direction_speed_limits(
            tangents, yaws, arclength, envelope=planning_envelope,
            ellipse_x=linear_limit, ellipse_y=lateral_limit)
        wheel_limits = np.full(len(points), planning_envelope.max_wheel)
        linear_limits = np.full(len(points), linear_limit)
        lateral_limits = np.full(len(points), lateral_limit)
        sprint_cruise_allowed = turn_everywhere
        sprint_fast_turn_allowed = turn_everywhere
        turn_start_yaw = float(pose[2])
        # Full coupled wheel limits still apply; only the extra geographic
        # turn budget is bypassed. Braking, curvature and guards remain active.
        if not turn_everywhere and max(linear_limit, lateral_limit) > 1.10 and np.ptp(yaws) > .05:
            # Keep the validated budget where heading changes, but release
            # constant-heading sections of the same route for full cruise.
            turning = np.abs(yaw_derivative(yaws, arclength)) > 1.e-6
            turn_speed = 1.05
            turn_wheel_budget = 15.709120382
            if getattr(self, 'profile_name', '') == 'sprint':
                # The full coupled translation/yaw wheel envelope already
                # bounds limits above. Keep it on checked open sprint routes rather
                # than imposing 1.05 m/s merely because yaw is changing.
                turning &= ~sprint_turn_clearance(points, yaws, getattr(self, 'clearance', None))
                sprint_cruise_allowed = bool(np.all(sprint_turn_clearance(
                    points, yaws, getattr(self, 'clearance', None), reserve_m=.025)))
                if (previous is not None
                        and getattr(self, 'trajectory_profile_key', None) == profile_key
                        and same_endpoint(previous.points[-1], points[-1])
                        and abs(wrap(float(previous.yaws[-1])-float(yaws[-1]))) < 1.e-6):
                    # A shortened replan must not erase an earlier narrow
                    # approach and unlock faster turning halfway through it.
                    # Revocation is immediate; promotion waits for a new goal.
                    sprint_cruise_allowed &= getattr(previous, 'sprint_cruise_allowed', False)
                    turn_start_yaw = getattr(previous, 'turn_start_yaw', turn_start_yaw)
                    sprint_fast_turn_allowed = getattr(previous, 'sprint_fast_turn_allowed', False)
                else:
                    sprint_fast_turn_allowed = True
                # Faster turning is validated for straight forward/reverse
                # departures. Lateral starts retain 3 m/s: delayed-drive replay
                # showed arrival regressions there. Preserve the departure frame
                # across replans, and never promote an already restricted goal.
                departure_axis = np.array([math.cos(turn_start_yaw), math.sin(turn_start_yaw)])
                sprint_fast_turn_allowed &= bool(
                    np.all(np.abs(tangents @ departure_axis) >= .999)
                    and np.all(sprint_turn_clearance(
                        points, yaws, getattr(self, 'clearance', None), reserve_m=.15)))
                if np.any(turning) and sprint_cruise_allowed:
                    # Checked corridors can use the 3 m/s turn budget with
                    # command-history prediction; retain the legacy 1.3 m/s
                    # budget when that hardware tuning is disabled.
                    # Keep one budget for the entire remaining route: local
                    # high/low switches regressed delayed-drive tracking.
                    turn_speed = (3.00 if self.get_parameter('predictive_sprint').value else 1.30)
                    if self.get_parameter('predictive_sprint').value and sprint_fast_turn_allowed:
                        turn_speed = 4.00
                    turn_wheel_budget *= turn_speed/1.05
            turning_envelope = copy.copy(self.envelope)
            turning_envelope.max_wheel = min(self.envelope.max_wheel, turn_wheel_budget)
            turn_linear, turn_lateral = min(linear_limit, turn_speed), min(lateral_limit, turn_speed)
            turn_limits, _ = direction_speed_limits(
                tangents, yaws, arclength, envelope=turning_envelope,
                ellipse_x=turn_linear, ellipse_y=turn_lateral)
            limits = np.where(turning, np.minimum(limits, turn_limits), limits)
            wheel_limits[turning] = turning_envelope.max_wheel
            linear_limits[turning] = turn_linear
            lateral_limits[turning] = turn_lateral
        # Cruise through straight lanes at the selected profile speed. Only
        # tight lane-mouth turns need extra preview for the delayed drive.
        # Central bypass gate passage is handled by the Nav2 BT, not a crawl.
        gate_speed = getattr(self, 'fixed_gate_speed', None)
        gate_turns = ()
        turn_limits = None
        if gate_speed is not None:
            turn_limits, gate_turns = gate_speed.plan_limits(
                points, getattr(self, 'deceleration', self.acceleration), curvature=menger_curvature(points),
                arclength=arclength, retained_turns=getattr(previous, 'gate_turns', ()))
            limits = np.minimum(limits, turn_limits)
        # Fixed-heading translation uses the wheel/curvature envelope above.
        # Its delayed braking sweep below checks clearance in the actual motion
        # direction; lateral wall distance is not a forward stopping distance.
        # 終端の低速寄せは速度プロファイルの一部にする。制御側で後から
        # 上限を掛けると、0.553 m/s から 0.16 m/s への段差になり、加速度
        # 制限で落とし切る間に 0.15 m 進んでしまう。プロファイルに入れれば
        # 後ろ向き掃引が滑らかに落としてくれる。
        zone = terminal_zone(self, stage)
        cap = terminal_cap(self)
        if zone > 0.0:
            limits = np.where(
                (arclength[-1] - arclength) <= zone,
                np.minimum(limits, cap), limits)

        try:
            trajectory = Trajectory(
                points, yaws, limits,
                acceleration=self.acceleration,
                deceleration=getattr(self, 'deceleration', self.acceleration),
                lateral_acceleration=self.lateral_acceleration,
                entry_speed=entry_speed,
                angular_speed=self.yaw_limit,
                angular_acceleration=self.yaw_acceleration,
            )
            trajectory.wheel_limit = planning_envelope.max_wheel
            trajectory.motion_limits = np.column_stack(
                (linear_limits, lateral_limits, wheel_limits))
            trajectory.sprint_cruise_allowed = sprint_cruise_allowed
            trajectory.sprint_fast_turn_allowed = sprint_fast_turn_allowed
            trajectory.turn_start_yaw = turn_start_yaw
            trajectory.gate_turns = gate_turns
            trajectory.gate_turn_limits = turn_limits
            trajectory.linear_limit = linear_limit
            trajectory.lateral_limit = lateral_limit
        except (ValueError, FloatingPointError) as error:
            self.get_logger().warning(f'trajectory build failed: {error}')
            return
        with self.lock:
            if revision != getattr(self, 'stage_revision', 0):
                return
            if warm_departure:
                self.stage_continuation = (trajectory, profile_key, revision)
                self.plan_stamp = time.monotonic()
                self.plan_build_ms = 1000.*(time.monotonic()-started)
                return
            if staged:
                if self.heading_stage is None:
                    self.heading_stage = stage
                self.stage_blocked = None
            self.snap_offset = snap_offset
            self.snapped = snapped
            self.trajectory = trajectory
            self.trajectory_profile_key = profile_key
            # 基準時刻を 0 に戻すと、基準は「この経路を計画した時点の機体
            # 位置」へ飛び戻る。BT の replanning は 1 Hz なので、これが走行中
            # 毎秒起きていた。計画と姿勢最適化の間に機体は進んでいるので、
            # 差し替えた瞬間の位置誤差は後ろ向きに 0.1 m 前後になり、P項が
            # そのぶん制動をかける。基準は実時間で進むだけで追いつけないため
            # 誤差は残り、次の経路でまた飛び戻る。1 Hz の前後の脈動と、
            # 経路が曲がっている区間ではその横成分が左右の振れになる。
            # 差し替え時点の機体位置に合わせて基準を置き直す。
            self.reference_time = entry_time(trajectory, self.pose)
            self.plan_stamp = time.monotonic()
            # 進捗の記録は経路の差し替えでは捨てない。BT は 1 Hz で出し直す
            # ので、ここで毎回 inf に戻すと停滞時間は最大 1 秒しか積まれず、
            # no_progress_timeout_sec には決して届かない。つまり「経路は
            # 出続けているのに目標へ寄らない」という、まさに捕まえたい円の
            # 走行だけが検出できなかった。終点が別の場所へ動いたとき（次の
            # 目標）だけ捨てる。
            #
            # 終端の掛け金も同じである。ここで finished_at を毎回 None へ
            # 戻していたので、terminal_hold_sec = 3 s は 1 Hz の差し替えで
            # 必ず折り返され、TERMINAL_HOLD_EXPIRED は一度も成立しなかった。
            # 目標公差 0.04 m に入れないまま終端ゾーンを回り続ける機体は、
            # BT が経路を出し続ける限り永久に走り続ける。これが「小さく円を
            # 描いて止まらない」の止まらない側である。
            if not same_endpoint(self.tracked_endpoint, trajectory.points[-1]):
                self.tracked_endpoint = trajectory.points[-1].copy()
                self.best_distance = math.inf
                self.best_distance_at = time.monotonic()
                self.finished_at = None
                self.terminal_since = None
                self.terminal_best_yaw = math.inf
                self.terminal_best_distance = math.inf
            self.plan_build_ms = 1000.0 * (time.monotonic() - started)

    def _plan_yaw(self, points, tangents, arclength, current_yaw, goal_yaw,
                  cutoff=None):
        """粗いサンプルで姿勢を最適化し、全点へ補間して滑らかにして返す。

        ``cutoff`` は指定姿勢へ入れ終える弧長。それより先は指定姿勢のまま
        保つ（``np.interp`` は範囲外を端の値で返すので自動的にそうなる）。
        """
        spacing = float(self.get_parameter('yaw_plan_spacing_m').value)
        indices, last = yaw_plan_knots(
            arclength, spacing,
            float(arclength[-1] if cutoff is None else cutoff))
        # 姿勢変化の上限は、その区間を最速で通ったときに出せる角度。
        step_lengths = np.diff(arclength[indices])
        fastest = max(self.speed_limit, 1.0e-3)
        max_step = np.maximum(
            self.yaw_limit * step_lengths / fastest, math.radians(5.0))
        coarse = plan_yaw_profile(
            points[indices], tangents[indices],
            envelope=self.envelope,
            ellipse_x=self.speed_limit,
            ellipse_y=self.lateral_limit,
            current_yaw=current_yaw,
            goal_yaw=goal_yaw,
            clearance_model=self.clearance,
            minimum_clearance=float(
                self.get_parameter('minimum_clearance_m').value),
            clearance_weight=float(
                self.get_parameter('clearance_weight').value),
            max_yaw_step=max_step,
        )
        # 打ち切りは最後の節点の弧長。そこで指定姿勢へ入り、以降は保つ
        # （np.interp は範囲外を端の値で返す）。
        planned = np.interp(
            np.minimum(arclength, last), arclength[indices], coarse)
        # 丸めるのは旋回している区間だけにする。平滑長 0.25 m は終端の
        # 平らな区間 (0.10 m) より長いので、配列全体を丸めると角が平らな
        # 区間へにじみ出し、いま作った「終端では姿勢を動かさない」性質を
        # 打ち消す（実測で最後の 0.15 m に 2.76 度が戻っていた）。
        stop = int(indices[-1]) + 1
        smoothed = planned.copy()
        smoothed[:stop] = smooth_yaw_profile(
            planned[:stop], arclength[:stop],
            float(self.get_parameter('yaw_smoothing_m').value))
        return self._clearance_safe_yaw(points, planned, smoothed)

    def _clearance_safe_yaw(self, points, planned, smoothed):
        """角を丸めた姿勢列が余裕を割るなら、割らない範囲まで戻す。

        動的計画法が余裕を評価するのは 0.30 m ごとの節点だけである。節点間の
        姿勢は補間値なので、丸める前も後も一度も検査されていない。地点4〜7の
        余裕は 47〜59 mm しかないので、ここは全点を検査する。

        戻すのは全体を一様に混ぜる形にする。点ごとに混ぜ方を変えると、
        まさにここで作った角速度の連続性が壊れるからである。凸結合なので
        端点（機体の実姿勢と指定姿勢）はどちらの側でも動かない。
        """
        if self.clearance is None or np.array_equal(smoothed, planned):
            return smoothed
        minimum = float(self.get_parameter('minimum_clearance_m').value)
        # 知りたいのは「余裕が minimum 以上か」だけである。cap を minimum に
        # しておくと body_clearance はそれ以上を測らずに返るので、外接半径
        # + minimum より壁が遠い点は多角形の距離計算そのものを飛ばせる。
        # cap を 0.35 で呼ぶと 1 経路あたり数百回の多角形計算になる。
        reach = float(getattr(self.clearance, 'radius', 0.6)) + minimum
        checked = [
            index for index in range(len(points))
            if self.clearance.clearance_and_gradient(points[index])[0] < reach
        ]
        if not checked:
            return smoothed
        blend = 1.0
        for _ in range(4):
            candidate = planned + blend * (smoothed - planned)
            if all(
                self.clearance.body_clearance(
                    points[index], float(candidate[index]),
                    cap=minimum) >= minimum
                for index in checked
            ):
                return candidate
            blend *= 0.5
        return planned

    # ------------------------------------------------------------------

    def _track_flow(self, yaw: float, measured, vx: float, vy: float) -> None:
        """指令と実測の進行方向を世界系で均しておく（診断用）。

        機体系のまま比べると、旋回している間はどちらも同じだけ回るので
        ずれが見えない。世界系へ出してから比べる。
        """
        tau = float(self.get_parameter('flow_filter_sec').value)
        if tau <= 0.0:
            return
        cosine, sine = math.cos(float(yaw)), math.sin(float(yaw))
        commanded = np.array([
            vx * cosine - vy * sine, vx * sine + vy * cosine])
        actual = np.array([
            measured[0] * cosine - measured[1] * sine,
            measured[0] * sine + measured[1] * cosine,
        ])
        alpha = 1.0 - math.exp(-self.period / tau)
        self.command_flow = self.command_flow + alpha * (
            commanded - self.command_flow)
        self.measured_flow = self.measured_flow + alpha * (
            actual - self.measured_flow)
        # This compares current upstream demand with delayed measured motion.
        # Braking and downstream slew limiting can also make this ratio > 1;
        # it is evidence to inspect, not an actuator calibration estimate.
        gain = self._flow_gain()
        if gain is not None and gain > 1.25:
            self.get_logger().warning(
                f'measured/upstream speed ratio={gain:.2f}; '
                'may reflect braking delay, downstream limiting or calibration. '
                'Compare timestamped /cmd_vel_safe and /wheel/odometry with '
                'run.py record-run before changing drive calibration.',
                throttle_duration_sec=10.0)

    def _flow_gap(self):
        gap = direction_gap(self.command_flow, self.measured_flow)
        return None if gap is None else round(gap, 1)

    def _flow_gain(self):
        """指令した速さに対して、実際に出た速さの比。判定不能なら None。

        _track_flow が両方を 0.5 s で均してあるので、巡航中のこの比が駆動系
        ゲイン（指令 1 m/s に対して実際に出る速度）である。1.0 でなければ
        ミキサの較正 auto_units_per_mps が違うということで、

          * フィードフォワードは全区間その倍率だけ外れる
          * 加速度・車輪バジェット・包絡線もすべて同じ倍率だけ嘘になる
          * 姿勢のP項は駆動系ゲインの倍だけ実効ゲインが上がる。
            安定性の限界はゲインだけでなく遅れにも依存する。

        となる。この比はゲート後の指令ではなく追従器の出口を見ているので、
        RuntimeGuard が絞った分は分母に入らない点に注意（絞られた区間では
        1.0 より大きく出る）。巡航中の定常値で読むこと。

        加減速中は 0.5 s の遅れのぶん偏るので、分母が小さいうちは出さない。
        """
        commanded = float(np.linalg.norm(self.command_flow))
        if commanded < 0.15:
            return None
        return round(float(np.linalg.norm(self.measured_flow)) / commanded, 3)

    def _monitor_safe_yaw_rate(self, yaw_rate: float, position, yaw: float):
        """下流の collision_monitor が絞らずに通す角速度まで落とす。

        判断の中身は rl_residual.limit_yaw_rate にある。ここに置く理由は
        車輪バジェットの配分で、rl_policy 側にも同じ制限が入っているのは
        MPPI モードと、この追従器が止まっている間の復帰動作のためである。
        どちらも同じ純粋関数を呼ぶので二重にはならない（同じ上限で二度
        飽和させるだけ）。
        """
        if self.monitor_horizon is None or self.clearance is None:
            return yaw_rate
        return limit_yaw_rate(
            yaw_rate, position=position, yaw=yaw,
            clearance_model=self.clearance, horizon=self.monitor_horizon)

    def _rate_limit(self, vx: float, vy: float, wz: float):
        previous_command = self.command.copy()
        frame = getattr(self, 'prediction_frame', None)
        last_frame = getattr(self, 'last_prediction_frame', None)
        self.last_prediction_frame = frame
        if frame is not None and last_frame is not None:
            angle = wrap(frame[1]-last_frame[1])
            if 0. <= frame[0]-last_frame[0] <= .15 and abs(angle) < .5:
                previous_command[:2] = to_body(previous_command[:2], angle)
        translation = np.array([vx, vy])
        delta = translation - previous_command[:2]
        norm = float(np.linalg.norm(delta))
        dt = getattr(self, 'control_dt', self.period)
        braking = getattr(self, 'deceleration', self.acceleration)
        # Only speed increase along the current travel direction gets the
        # sprint acceleration. Reversal, braking and lateral correction keep
        # the existing budget. Equal budgets reproduce the old circular limit.
        previous_speed = float(np.linalg.norm(previous_command[:2]))
        direction = (previous_command[:2]/previous_speed if previous_speed > 1.e-9
                     else delta/max(norm, 1.e-9))
        forward = max(0., float(np.dot(delta, direction)))
        correction = delta-forward*direction
        cost = math.hypot(forward/(self.acceleration*dt),
                          float(np.linalg.norm(correction))/(braking*dt))
        if cost > 1.:
            translation = previous_command[:2]+delta/cost
        yaw_budget = self.yaw_acceleration * dt
        yaw = self.command[2] + float(
            np.clip(wz - self.command[2], -yaw_budget, yaw_budget))
        return float(translation[0]), float(translation[1]), yaw

    def _publish(self, vx: float, vy: float, wz: float) -> None:
        if ((getattr(self, 'motion_mode', 'simultaneous') == 'staged_heading'
                or getattr(self, 'reverse_goal', None) is not None)
                and (vx or vy or wz)):
            vx, vy, wz = self._stage_safe_command(vx, vy, wz)
        message = Twist()
        message.linear.x = float(vx)
        message.linear.y = float(vy)
        message.angular.z = float(wz)
        self.command_pub.publish(message)
        self.command = np.array([vx, vy, wz])
        record_prediction_command(self, time.monotonic(), self.command)

    def _status(self, state: str, **values) -> None:
        if state != self.last_status:
            self.last_status = state
            self.get_logger().info(f'tracker: {state}')
        payload = {'state': state,
                   'motion_mode': getattr(self, 'motion_mode', 'simultaneous'),
                   'requested_motion_mode': getattr(self, 'requested_motion_mode', 'simultaneous')}
        if self.pose is not None and self.active_goal is not None:
            position_error = float(np.linalg.norm(self.active_goal[:2]-self.pose[:2]))
            yaw_error = abs(wrap(self.active_goal[2]-self.pose[2]))
            fresh = (time.monotonic()-self.pose_stamp <= .3
                     and self.velocity_stamp is not None
                     and time.monotonic()-self.velocity_stamp <= .2)
            payload['arrival'] = dict(goal=self.active_goal.tolist(),
                goal_stamp=getattr(self, 'active_goal_stamp', None),
                ready=bool(fresh and position_error <= .015 and yaw_error <= .015
                           and np.linalg.norm(self.velocity[:2]) <= .025
                           and abs(self.velocity[2]) <= .025))
            if getattr(self, 'reverse_goal', None) is not None:
                # The bridge must not end docking while the reverse servo is
                # still approaching, or mistake a blocked stop for arrival.
                settled = bool(
                    fresh and state == 'REVERSING'
                    and position_error <= POSITION_TOLERANCE and yaw_error <= YAW_TOLERANCE
                    and np.linalg.norm(self.command) <= 1.e-9
                    and np.linalg.norm(self.velocity[:2]) <= .005
                    and abs(self.velocity[2]) <= .01)
                now = time.monotonic()
                previous = getattr(self, 'reverse_arrival_check', None)
                if not settled or previous is None or not 0. <= now-previous <= .2:
                    self.reverse_settled_since = None
                if settled and getattr(self, 'reverse_settled_since', None) is None:
                    self.reverse_settled_since = now
                self.reverse_arrival_check = now
                payload['arrival']['ready'] = bool(
                    settled and now-self.reverse_settled_since >= .3)
        stage = getattr(self, 'heading_stage', None)
        if stage is not None:
            payload.update(phase=stage.phase, heading=stage.target,
                           rotation_gate=None if stage.gate is None else stage.gate.tolist(),
                           rotation_clearance_m=stage.clearance)
        payload.update(values)
        self.status_pub.publish(
            String(data=json.dumps(payload, separators=(',', ':'))))

    def _tick(self) -> None:
        now = time.monotonic()
        self.prediction_frame = None
        previous_tick = getattr(self, 'last_control_tick', None)
        elapsed = self.period if previous_tick is None else now-previous_tick
        self.last_control_tick = now
        # ROS timers are not clocks: polygon work and scheduling can delay a
        # callback. Advancing by a fixed 50 ms silently slows the whole route.
        # Bound catch-up, and update even while paused so recovery cannot jump.
        self.control_dt = (min(elapsed, 2*self.period)
                           if elapsed > 0. else self.period)
        if not getattr(self, 'stage_goal_enabled', True):
            self._publish(0., 0., 0.)
            self._status('IDLE')
            return
        with self.lock:
            trajectory = self.trajectory
            pose = None if self.pose is None else self.pose.copy()
            pose_stamp = self.pose_stamp
            reference_time = self.reference_time
            plan_stamp = self.plan_stamp
            plan_received_stamp = getattr(self, 'plan_received_stamp', None)
            goal = None if self.active_goal is None else self.active_goal.copy()
            velocity_stamp = self.velocity_stamp
        if velocity_stamp is None or now - velocity_stamp > float(
            self.get_parameter('velocity_timeout_sec').value
        ):
            stage = getattr(self, 'heading_stage', None)
            if stage is not None:
                stage.settled_since = None
            self.odometry_paused = True
            self.best_distance_at = now
            self.terminal_since = None
            self.finished_at = None
            self._publish(0.0, 0.0, 0.0)
            self._status('ODOMETRY_STALE')
            return
        if self.odometry_paused:
            with self.lock:
                if trajectory is not None:
                    reference_time = entry_time(trajectory, pose)
                    self.reference_time = reference_time
                self.odometry_paused = False
        if getattr(self, 'reverse_goal', None) is not None:
            self._reverse_tick(now, pose, pose_stamp)
            return
        if trajectory is None:
            if getattr(self, 'motion_mode', 'simultaneous') == 'staged_heading':
                self._publish(0., 0., 0.)
                self._status('STAGED_BLOCKED' if self.stage_blocked else 'STAGED_PLANNING',
                             reason=self.stage_blocked)
                return
            if self._relay_behavior(now):
                self._status('BEHAVIOR')
                return
            self._publish(0.0, 0.0, 0.0)
            return
        if pose is None or not np.isfinite(pose).all() or now - pose_stamp > float(
            self.get_parameter('pose_timeout_sec').value
        ):
            stage = getattr(self, 'heading_stage', None)
            if stage is not None:
                stage.settled_since = None
            # 自己位置が無い状態でフィードフォワードを流すのは危険。復帰動作
            # は機体系の指令で自己位置に依らないので、そちらは流してよい。
            if self._relay_behavior(now):
                self._status('BEHAVIOR', pose_stale=True)
                return
            self._publish(0.0, 0.0, 0.0)
            self.best_distance_at = now
            self.terminal_since = None
            self._status('POSE_STALE')
            return
        # BT は走行中 1 Hz で経路を出し直す。それが途切れたということは、
        # 目標が終わったか中断されたか、planner が失敗したかである。
        # plan_timeout_sec は宣言されているだけで誰も見ておらず、最後の軌道を
        # 永遠に追い続けていた。指令の座標系がずれていれば、その「永遠」は
        # 目標へ寄らない円になる。経路が古くなったら止める。
        # Planner heartbeat and trajectory-build latency are different. Keep
        # following the checked route while a fresh replan is being processed;
        # otherwise slow geometry work fabricates a planner outage. Goal
        # cancellation and live obstacle / pose / wheel gates are independent.
        plan_age = now - max(plan_stamp, plan_received_stamp
                            if plan_received_stamp is not None else plan_stamp)
        # ComputePathThroughPoses can take over two seconds with several CAD
        # gates. Cancellation has its own immediate subscription in staged
        # mode; pose/wheel/live-obstacle watchdogs remain independent.
        plan_timeout = (4.0 if getattr(self, 'motion_mode', 'simultaneous') == 'staged_heading'
                        else float(self.get_parameter('plan_timeout_sec').value))
        final_approach = (now < getattr(self,'final_approach_until',0.)
                          and goal is not None and np.linalg.norm(goal[:2]-pose[:2]) <= .08)
        if plan_age > plan_timeout and not final_approach:
            stage = getattr(self, 'heading_stage', None)
            if stage is not None:
                stage.settled_since = None
            # 経路が途切れる主な理由は、BT が復帰動作へ入ったことである。
            # ここでゼロを出し続けると復帰動作は 1 本も外へ出ない。
            if self._relay_behavior(now):
                self._status('BEHAVIOR', plan_age=round(plan_age, 2))
                return
            self._publish(0.0, 0.0, 0.0)
            # 止めている時間を停滞として数えると、経路が戻ってきた瞬間に
            # NO_PROGRESS へ落ちて動き出せない。
            self.best_distance_at = now
            self.terminal_since = None
            self._status('PLAN_STALE', plan_age=round(plan_age, 2))
            return

        # 駆動系は無駄時間と一次遅れを持つので、いま観測している姿勢は
        # すでに過去のものである。誤差は「この指令が効くころの姿勢」に対して
        # 取る。計測輪の速度（機体系）を世界系へ回して外挿する。標準的な
        # 無駄時間補償で、行き過ぎと定常追従の両方に効く。
        with self.lock:
            measured = self.velocity.copy()
        lag = float(self.get_parameter('feedback_delay_sec').value)
        cosine, sine = math.cos(pose[2]), math.sin(pose[2])
        world_measured = np.array([
            measured[0] * cosine - measured[1] * sine,
            measured[0] * sine + measured[1] * cosine,
        ])
        position = pose[:2] + world_measured * lag
        yaw_lead = float(np.clip(
            float(measured[2]) * lag,
            -float(self.get_parameter('max_predicted_yaw_rad').value),
            float(self.get_parameter('max_predicted_yaw_rad').value),
        ))
        predicted_yaw = pose[2] + yaw_lead
        if (self.get_parameter('predictive_sprint').value
                and getattr(self, 'profile_name', '') == 'sprint'
                and getattr(trajectory, 'sprint_cruise_allowed', False)):
            predictor = getattr(self, 'motion_predictor', None)
            if predictor is None:
                predictor = self.motion_predictor = DelayedMotionPredictor()
            predicted, predicted_velocity = predictor.predict(pose, measured, now)
            lag = predictor.horizon
            position, predicted_yaw = predicted[:2], float(predicted[2])
            cosine, sine = math.cos(predicted_yaw), math.sin(predicted_yaw)
            world_measured = np.array([
                cosine*predicted_velocity[0]-sine*predicted_velocity[1],
                sine*predicted_velocity[0]+cosine*predicted_velocity[1]])
            self.prediction_frame = (now, predicted_yaw)
        observed = pose[:2]

        # 基準時刻を進めるが、実機より前へ出過ぎないよう抑える（バーチャル
        # ビークル）。障害物で止められている間に基準だけ走り去るのを防ぐ。
        #
        # 進める速さは RuntimeGuard が参照軌道へ許可する倍率である（経路速度
        # スケーリング）。経路の形と姿勢列は弧長の関数なので倍率では変わらず、
        # 変わるのは時間割りだけなので、作り直さずにここで掛けられる。倍率が
        # 0（ARM・センサー等の異常）なら基準は機体と一緒に止まり、復帰後は続きを
        # 走る。以前は基準だけ走り去り、max_reference_lead_m が引き戻していた。
        scale = float(np.clip(self.speed_scale, 0.0, 1.0))
        if scale <= 0.0:
            stage = getattr(self, 'heading_stage', None)
            if stage is not None:
                stage.settled_since = None
            # A pause at the goal must not consume the settling watchdogs or
            # leave a residual yaw request waiting downstream on resume.
            self.best_distance_at = now
            self.terminal_since = None
            self.finished_at = None
            self.terminal_best_yaw = math.inf
            self.terminal_best_distance = math.inf
            self._publish(0.0, 0.0, 0.0)
            self._status('PAUSED')
            return
        if getattr(self, 'motion_mode', 'simultaneous') == 'staged_heading':
            if self._stage_tick(now, pose, measured, scale):
                return
            # A successful turn can install the prepared departure in this
            # very tick. Never sample the approach captured before the handoff.
            with self.lock:
                if self.trajectory is not trajectory:
                    trajectory = self.trajectory
                    reference_time = self.reference_time
            if trajectory is None:
                self._publish(0., 0., 0.)
                return
        reference_time = min(
            reference_time + self.control_dt * scale, trajectory.duration)
        lead = float(self.get_parameter('max_reference_lead_m').value)
        actual_s = trajectory.project(position)

        # 経路に沿って前へ進めているか。指令を機体系へ落とす回転が実際の向き
        # からずれていると、位置のP制御は誤差を目標方向ではなく横へ倒すので、
        # 経路上を進まないまま一定の弧を回り続ける（右回りの円）。
        # 見るのは終点までの直線距離ではなく残りの弧長である。壁を回り込む
        # 経路では直線距離は数秒間ふつうに増えるので、直線距離で見ると正常な
        # 走行を停滞と誤判定して止めてしまう。
        remaining_arc = float(trajectory.length - actual_s)
        if float(np.linalg.norm(measured[:2])) < float(
            self.get_parameter('no_progress_speed_m_s').value
        ):
            # 止まっている間は停滞時間を進めない（上のコメントの理由）。
            self.best_distance = min(self.best_distance, remaining_arc)
            self.best_distance_at = now
            stalled = 0.0
        else:
            self.best_distance, self.best_distance_at, stalled = track_progress(
                remaining_arc, self.best_distance, self.best_distance_at, now,
                float(self.get_parameter('no_progress_epsilon_m').value))
        if remaining_arc > terminal_zone(self) and stalled > float(
            self.get_parameter('no_progress_timeout_sec').value
        ):
            # 諦めたのだから、この先はこの軌道ではなく復帰動作が機体を持つ。
            # ここでゼロを出し続けると、progress checker が回した Spin /
            # BackUp は 1 本も外へ出ず、目標は動かないまま abort する。
            if self._relay_behavior(now):
                self._status('BEHAVIOR', after='NO_PROGRESS')
                return
            self._publish(0.0, 0.0, 0.0)
            self._status(
                'NO_PROGRESS',
                remaining_arc=round(remaining_arc, 4),
                best_remaining_arc=round(float(self.best_distance), 4),
                goal_distance=round(float(np.linalg.norm(
                    trajectory.points[-1] - observed)), 4),
                stalled_sec=round(stalled, 2),
                flow_gap_deg=self._flow_gap(),
                flow_gain=self._flow_gain(),
                measured=[round(float(v), 4) for v in measured])
            return

        # 終端ゾーンの掛け金。上の停滞判定はここを除外している（目標のそばで
        # 弧長が縮まないのは正常な整定でも起きる）ので、ゾーンの中で回り続ける
        # 破綻はこちらで切る。ゾーンを明確に出たときだけ外す。半径 0.05 m 前後
        # の円を描いていると弧長は境界をまたいで往復するので、境界そのもので
        # 外すと掛け金が毎周リセットされてしまう。
        self.terminal_since = terminal_latch(
            self.terminal_since, remaining_arc,
            terminal_zone(self), now)
        # Settling is progress in measured position as well as yaw. With an
        # uncalibrated drive, the reference can end before the base has settled;
        # an absolute three-second deadline stopped a still-converging base.
        # Track the best error across replans so a constant-radius orbit or
        # localization jitter cannot refresh this deadline indefinitely.
        if self.terminal_since is not None or self.finished_at is not None:
            distance = float(np.linalg.norm(trajectory.points[-1] - observed))
            if distance < getattr(self, 'terminal_best_distance', math.inf) - .01:
                self.terminal_best_distance = distance
                if self.terminal_since is not None:
                    self.terminal_since = now
                if self.finished_at is not None:
                    self.finished_at = now
        # A short move may finish its translation before a large turn. Count
        # actual improvement in yaw while holding the goal as progress, rather
        # than aborting every in-place 180-degree turn after three seconds.
        # Keep the best error across replans; oscillation cannot reset it.
        if np.linalg.norm(trajectory.points[-1] - observed) <= terminal_zone(self):
            goal_yaw_error = abs(wrap(float(trajectory.yaws[-1]) - pose[2]))
            if goal_yaw_error < getattr(self, 'terminal_best_yaw', math.inf) - 0.02:
                self.terminal_best_yaw = goal_yaw_error
                self.terminal_since = now
                if self.finished_at is not None:
                    self.finished_at = now
        limit = float(self.get_parameter('terminal_timeout_sec').value)
        held = (None if self.terminal_since is None
                else now - self.terminal_since)
        if held is not None and held > limit:
            if self._relay_behavior(now):
                self._status('BEHAVIOR', after='TERMINAL_TIMEOUT')
                return
            self._publish(0.0, 0.0, 0.0)
            self._status(
                'TERMINAL_TIMEOUT',
                held=round(held, 2),
                observed_error=round(float(np.linalg.norm(
                    trajectory.points[-1] - observed)), 4),
                flow_gap_deg=self._flow_gap(),
                flow_gain=self._flow_gain(),
                measured=[round(float(v), 4) for v in measured])
            return

        # 状態を lag 先へ外挿したのだから、基準も同じ時刻から取る。position
        # だけを進めて基準を現在時刻のままにすると、進行方向へ v*lag だけ
        # 常に「行き過ぎている」と見えるので、P項が走行中ずっと制動を
        # かける。0.7 m/s では 32 mm の定常追従遅れになっていた。終端は
        # 基準速度が 0 に落ちるのでこの偏りは消え、到達精度には出ない。
        # 基準時刻は倍率 scale で進むので、実時間 lag 秒ぶんの先読みは
        # 基準時刻では lag * scale である。
        sample_time = min(reference_time + lag * scale, trajectory.duration)
        reference = trajectory.sample(sample_time)
        if reference[4] > actual_s + lead:
            sample_time = trajectory.time_at_arclength(actual_s + lead)
            reference = trajectory.sample(sample_time)
            reference_time = max(0.0, sample_time - lag * scale)
        ref_position, ref_velocity, ref_yaw, ref_yaw_rate, ref_s = reference
        # 基準の速度と角速度も同じ倍率で読む。位置と姿勢はそのまま。
        ref_velocity = ref_velocity * scale
        ref_yaw_rate = ref_yaw_rate * scale

        with self.lock:
            if self.trajectory is not trajectory:
                return
            if sample_time >= trajectory.duration and self.finished_at is None:
                self.finished_at = now
                # A fixed route can pass close to its final goal before a
                # departure waypoint, then return to settle. Its earlier best
                # distance must not hide progress during that return. Seed the
                # settling phase once; same-endpoint replans retain this best
                # and the existing bounded no-progress deadlines.
                self.terminal_best_distance = float(np.linalg.norm(
                    trajectory.points[-1] - observed))
            finished_at = self.finished_at
        # 終端の保持時間を過ぎたら指令を落とす。従来はここで状態文字列が
        # TERMINAL_HOLD_EXPIRED に変わるだけで、サーボは回り続けていた。
        if finished_at is not None and now - finished_at > float(
            self.get_parameter('terminal_hold_sec').value
        ):
            self._publish(0.0, 0.0, 0.0)
            with self.lock:
                if self.trajectory is trajectory:
                    self.reference_time = reference_time
            self._status(
                'TERMINAL_HOLD_EXPIRED',
                held=round(now - finished_at, 2),
                observed_error=round(float(np.linalg.norm(
                    trajectory.points[-1] - observed)), 4),
                flow_gap_deg=self._flow_gap(),
                flow_gain=self._flow_gain())
            return

        error = ref_position - position
        # 制動項は誤差の差分ではなく速度差から作る。d(error)/dt は解析的に
        # 「基準速度 - 実速度」だが、position は 12 Hz の ICP 補正で階段状に
        # 動く信号なので、それを 20 Hz で差分すると補正のたびに
        # 5 mm / 0.033 s = 0.15 m/s のパルスが立ち、経路差し替え直後は
        # previous_error のリセットでさらに大きく跳ねていた。同じ量を
        # 微分せずに取る。
        world_velocity = (
            ref_velocity
            + float(self.get_parameter('position_gain').value) * error
            + float(self.get_parameter('position_damping').value)
            * (ref_velocity - world_measured)
        )
        ref_yaw, ref_yaw_rate = terminal_yaw_reference(
            ref_yaw, ref_yaw_rate, float(trajectory.yaws[-1]),
            remaining_arc, terminal_zone(self))
        yaw_error = wrap(ref_yaw - predicted_yaw)
        previous_yaw_error = getattr(self, 'previous_yaw_error', None)
        # At a half-turn, milliradian pose noise can toggle wrap() between
        # +pi and -pi. Keep the chosen direction inside a narrow hysteresis
        # band so acceleration does not repeatedly brake back to zero.
        if (previous_yaw_error is not None
                and abs(yaw_error) > math.pi-.05
                and abs(previous_yaw_error) > math.pi-.05
                and yaw_error * previous_yaw_error < 0.):
            yaw_error += math.copysign(2.*math.pi, previous_yaw_error)
        self.previous_yaw_error = yaw_error
        yaw_command = ref_yaw_rate + float(
            self.get_parameter('yaw_gain').value) * yaw_error

        # 世界系→機体系の回転にも、いまの姿勢ではなく指令が効くころの姿勢を
        # 使う。ここだけ現在姿勢のままだと、旋回している間じゅう実際に出る
        # 方向が yaw_rate * lag だけ横へずれる。0.8 rad/s なら 9.2 度、
        # 0.55 m/s で 88 mm/s の横成分である。姿勢は速い体軸へ合わせに行く
        # ので走行中つねに旋回しており、直進中の左右のふらつきと、終端で
        # 指定姿勢へ入れる間に妙な方向へ逃げるのはこれが効いている。
        world_velocity = terminal_translation(
            world_velocity, position, trajectory.points[-1], remaining_arc,
            terminal_zone(self),
            float(self.get_parameter('position_gain').value),
            terminal_cap(self) * scale)
        turn_limits = getattr(trajectory, 'gate_turn_limits', None)
        if turn_limits is not None:
            # Feedback may add >0.5 m/s on top of a slow turn reference. Apply
            # the same preview envelope to the complete command, at the
            # OBSERVED position, so a leading reference cannot bypass braking.
            observed_s = trajectory.project(observed)
            index = max(0, int(np.searchsorted(trajectory.arclength, observed_s))-1)
            cap = min(turn_limits[index], turn_limits[min(index+1, len(turn_limits)-1)])*scale
            norm = float(np.linalg.norm(world_velocity))
            if norm > cap:
                world_velocity *= cap/norm
        if max(self.speed_limit, self.lateral_limit) > 1.10:
            # At 2 m/s a time-reference alone can ask for cruise too far into
            # braking while feedback catches up. Bound the requested speed by
            # observed remaining path distance, including response/filter lag.
            # Keep 25% deceleration reserve for response/model uncertainty.
            # The terminal servo retains signed fine corrections and zero.
            observed_remaining = trajectory.length - trajectory.project(observed)
            if observed_remaining > terminal_zone(self):
                cap = max(terminal_cap(self)*scale, stopping_speed(
                    observed_remaining, .75*getattr(self, 'deceleration', self.acceleration),
                    lag + float(self.get_parameter('velocity_filter_sec').value)))
                norm = float(np.linalg.norm(world_velocity))
                if norm > cap:
                    world_velocity *= cap/norm
        vx, vy = to_body(world_velocity, predicted_yaw)
        linear_limit = getattr(trajectory, 'linear_limit', self.speed_limit)
        lateral_limit = getattr(trajectory, 'lateral_limit', self.lateral_limit)
        wheel_limit = getattr(trajectory, 'wheel_limit', self.envelope.max_wheel)
        motion_limits = getattr(trajectory, 'motion_limits', None)
        if motion_limits is not None:
            # Use the observed path interval so an advancing time reference
            # cannot release a turn budget before the robot clears the turn.
            observed_s = trajectory.project(observed)
            index = max(0, int(np.searchsorted(trajectory.arclength, observed_s))-1)
            linear_limit, lateral_limit, wheel_limit = np.minimum(
                motion_limits[index], motion_limits[min(index+1, len(motion_limits)-1)])
        ellipse = math.hypot(
            vx / max(linear_limit, 1.0e-6),
            vy / max(lateral_limit, 1.0e-6),
        )
        if ellipse > 1.0:
            vx /= ellipse
            vy /= ellipse
        # A zero-length route has no path-based angular braking profile. Keep
        # its position-hold turn slow enough for the delayed feedback servo.
        yaw_cap = min(self.yaw_limit, .6) if trajectory.length < 1.e-9 else self.yaw_limit
        yaw_command = float(np.clip(yaw_command, -yaw_cap, yaw_cap))
        # 壁ぎわで通らない旋回はここで落とす。車輪バジェットより前に置くの
        # は、通らない角速度のために並進のバジェットを譲るのが逆だからで
        # ある（下流で twist ごと絞られたうえに、並進も自分で削っていた）。
        requested_yaw_rate = yaw_command
        yaw_command = self._monitor_safe_yaw_rate(
            yaw_command, position, predicted_yaw)
        # 車輪バジェットへ収める。RuntimeGuard も同じことをするが、そこで
        # 初めて削られると軌道の時間割りが狂う。
        cost = self.envelope.wheel_cost(vx, vy, yaw_command)
        # ここは RuntimeGuard の倍率 scale とは別物なので別名にしておく。
        # 同じ名前を使っていたので、車輪バジェットが当たった周期の状態は
        # ゲートの倍率の欄にこの縮小率を出していた。balanced では並進 0.78
        # と旋回 1.30 rad/s で 11.03 + 12.25 = 23.28 と上限 15.709 を超える
        # ので、「回りながら走る」区間ではほぼ毎周期そうなる。蛇行を追って
        # いるときに読む欄が別の量になっているのは実害がある。
        budget_scale = 1.0
        if cost > wheel_limit:
            budget_scale = wheel_limit / cost
            vx *= budget_scale
            vy *= budget_scale
            yaw_command *= budget_scale

        # フィードフォワード成分は構成上加速度制約を満たすが、誤差の P/D 項と
        # 経路切替の段差はそれを破りうる。
        vx, vy, yaw_command = self._rate_limit(vx, vy, yaw_command)
        with self.lock:
            # A geometry worker can replace the trajectory while this tick is
            # evaluating its old reference. Never overwrite the new clock or
            # send that old command after the replacement/cancellation.
            if self.trajectory is not trajectory or not getattr(self, 'stage_goal_enabled', True):
                return
            self._publish(vx, vy, yaw_command)
            self.reference_time = reference_time
            build_ms = self.plan_build_ms
        if getattr(self, 'motion_mode', 'simultaneous') == 'staged_heading':
            vx, vy, yaw_command = map(float, self.command)
        self._track_flow(pose[2], measured, vx, vy)

        if sample_time >= trajectory.duration:
            with self.lock:
                snap_offset = self.snap_offset
                snapped = self.snapped
            self._status(
                'TERMINAL',
                position_error=round(float(np.linalg.norm(error)), 4),
                observed_error=round(float(np.linalg.norm(
                    trajectory.points[-1] - observed)), 4),
                yaw_error=round(yaw_error, 4),
                endpoint=[round(float(v), 4) for v in trajectory.points[-1]],
                goal=(None if goal is None
                      else [round(float(v), 4) for v in goal[:2]]),
                goal_offset=(None if goal is None else round(float(
                    np.linalg.norm(goal[:2] - trajectory.points[-1])), 4)),
                snap_offset=(None if snap_offset is None
                             else round(snap_offset, 4)),
                snapped=snapped,
                remaining=round(float(remaining_arc), 4),
                flow_gap_deg=self._flow_gap(),
                flow_gain=self._flow_gain(),
                commanded=[round(vx, 4), round(vy, 4), round(yaw_command, 4)])
        else:
            self._status(
                'TRACKING',
                progress=round(ref_s / max(trajectory.length, 1.0e-6), 3),
                cross_track=round(float(np.linalg.norm(error)), 4),
                flow_gap_deg=self._flow_gap(),
                flow_gain=self._flow_gain(),
                reference_speed=round(float(np.linalg.norm(ref_velocity)), 3),
                planned_peak_speed=round(float(np.max(trajectory.speed)), 3),
                acceleration_limit_m_s2=round(self.acceleration, 3),
                braking_limit_m_s2=round(getattr(self, 'deceleration', self.acceleration), 3),
                wheel_speed_budget_rad_s=round(self.envelope.max_wheel, 3),
                local_linear_limit_m_s=round(float(linear_limit), 3),
                local_wheel_budget_rad_s=round(float(wheel_limit), 3),
                profile=self.profile_name,
                predictive_sprint_active=self.prediction_frame is not None,
                measured_linear_m_s=round(float(np.linalg.norm(measured[:2])), 3),
                measured_angular_rad_s=round(float(measured[2]), 3),
                speed_scale=round(scale, 3),
                wheel_budget_scale=round(budget_scale, 3),
                yaw_rate=round(float(yaw_command), 3),
                requested_yaw_rate=round(float(requested_yaw_rate), 3),
                plan_build_ms=round(build_ms, 1))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = TrajectoryTracker()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except RuntimeError:
        # SIGINT can invalidate the native subscription while take_message is
        # converting its result. Only suppress that already-stopped context;
        # runtime failures in a live control process must remain visible.
        if rclpy.ok():
            raise
    finally:
        if node is not None:
            node.stopping = True
            node.plan_event.set()
            node.worker.join(timeout=1.0)
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
