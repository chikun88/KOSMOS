"""オムニの姿勢自由度を使い切るための、弧長に対する姿勢計画と速度上限。

45度オムニは方向によって出せる速度が違う。ローラ方向が
225/135/45/-45度なので、機体座標での進行方向が

  0/90/180/270度 (体軸)   -> 各輪コスト 14.14 rad/s per m/s -> 1.111 m/s
  45/135/... 度 (ローラ上) -> 各輪コスト 20.00 rad/s per m/s -> 0.785 m/s

と 1.41 倍違う。ホロノミックなので姿勢は進行方向と独立に選べる。つまり
「どちらを向いて走るか」を選ぶだけで速度が 41% 変わる。

同時に、フットプリント余裕も姿勢の関数である。地点4/5の射撃レーンは
指定姿勢での余裕が 57/59 mm しかなく、数度ずれると無くなる。通過中は
姿勢が自由なので、余裕が広い姿勢で通り、終端でだけ指定姿勢へ入れれば
よい。

この二つは競合する（速い姿勢が狭い姿勢のこともある）ので、弧長方向の
動的計画法で解く。状態は各サンプル点の姿勢候補、終端は指定姿勢に固定する。

コストは「秒」で数える。区間の遷移コストはその区間の通過時間

    ds / speed_limit(進行方向を機体系へ落としたもの, 姿勢変化/ds)

であり、旋回が食う車輪バジェットは speed_limit の中で自動的に速度を
下げる。段コストは余裕の利得を同じ秒へ換算したものだけである。

以前はここが -(速度 + 余裕) + 0.35 * |姿勢変化| という、m/s と rad を
足した式だった。速度は「旋回ゼロ」で評価していたので、姿勢を回して得る
利得は数えるが、そのために食う車輪バジェットは数えていなかった。実測で
1->4 区間の平均上限が 0.707 -> 0.687 m/s とわずかに悪化し、姿勢最適化は
既定で無効にされた（docs/DEPLOYMENT_STATUS.md 2026-08-06）。時間を最小化
すれば、回して得る利得と回して失う速度は同じ単位で釣り合う。
"""
import math

import numpy as np


def wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


class OmniEnvelope:
    """車輪バジェットから、方向つきの速度上限を出す。"""

    def __init__(self, drivetrain):
        self.positions = np.asarray(drivetrain['wheel_positions'], dtype=float)
        angles = np.radians(
            np.asarray(drivetrain['wheel_drive_angles_deg'], dtype=float))
        self.directions = np.column_stack((np.cos(angles), np.sin(angles)))
        self.radius = float(drivetrain['wheel_radius'])
        self.max_wheel = float(drivetrain['max_wheel_speed'])
        signs = np.asarray(
            drivetrain.get('wheel_signs', [1.0] * len(angles)), dtype=float)
        self.signs = signs
        # 旋回1 rad/s あたりの各輪速度（符号つき）
        self.lever = (
            -self.positions[:, 1] * self.directions[:, 0]
            + self.positions[:, 0] * self.directions[:, 1]
        ) * signs / self.radius
        self.axis_x = self.directions[:, 0] * signs / self.radius
        self.axis_y = self.directions[:, 1] * signs / self.radius

    def wheel_cost(self, vx, vy, yaw_rate):
        """この機体速度が要求する各輪速度の最大絶対値[rad/s]。"""
        speeds = self.axis_x * vx + self.axis_y * vy + self.lever * yaw_rate
        return float(np.max(np.abs(speeds)))

    def speed_limits(self, body_x, body_y, yaw_per_metre, ellipse_x, ellipse_y):
        """:meth:`speed_limit` のベクトル版。3引数はブロードキャストされる。

        姿勢計画の動的計画法は「候補 x 候補」の格子ごとに上限を要るので、
        Python ループで呼ぶと1経路あたり2000回近くになる。同じ式を配列で
        一度に解く。
        """
        body_x, body_y, rate = np.broadcast_arrays(
            np.asarray(body_x, dtype=float),
            np.asarray(body_y, dtype=float),
            np.asarray(yaw_per_metre, dtype=float),
        )
        flat_x, flat_y = body_x.reshape(-1), body_y.reshape(-1)
        speeds = (
            self.axis_x[:, None] * flat_x
            + self.axis_y[:, None] * flat_y
            + self.lever[:, None] * rate.reshape(-1)
        )
        cost = np.max(np.abs(speeds), axis=0)
        wheel = np.where(
            cost > 1.0e-9, self.max_wheel / np.maximum(cost, 1.0e-9), np.inf)
        ellipse = np.hypot(
            flat_x / max(float(ellipse_x), 1.0e-6),
            flat_y / max(float(ellipse_y), 1.0e-6),
        )
        profile = np.where(
            ellipse > 1.0e-9, 1.0 / np.maximum(ellipse, 1.0e-9), np.inf)
        return np.minimum(wheel, profile).reshape(body_x.shape)

    def speed_limit(self, direction, yaw_per_metre, ellipse_x, ellipse_y):
        """単位進行方向 direction (機体座標) で出せる速度[m/s]。

        ``yaw_per_metre`` は弧長1 mあたりの姿勢変化[rad]。速度と旋回は
        どちらも v に比例するので、各輪コストも v に比例し、上限は
        割り算で厳密に出る。
        ``ellipse_x/y`` は運用プロファイルの速度上限（駆動系より内側）。
        """
        return float(self.speed_limits(
            float(direction[0]), float(direction[1]), float(yaw_per_metre),
            ellipse_x, ellipse_y))

    def best_alignment_speed(self, ellipse_x, ellipse_y):
        """体軸に沿った最速値。診断表示用。"""
        return self.speed_limit(
            np.array([1.0, 0.0]), 0.0, ellipse_x, ellipse_y)


def yaw_candidates(
    heading, goal_yaw, current_yaw, ramp_yaw=None, step_limit=None,
    extra_step_deg=45.0,
):
    """この点で試す姿勢の候補。

    体軸を進行方向へ合わせる4通り（速度が最大になる姿勢）を必ず含める。
    ここを離散格子で近似すると、7.5度ずれるだけで速度が11%落ちるため、
    格子ではなく進行方向から作る。

    全周格子の間隔は「1区間で回れる角度」(``step_limit``) より細かくする。
    以前はここが 45 度固定で、区間あたりの上限は 19 度だった。つまり格子上の
    どの隣どうしも到達不能なので、緩やかな旋回そのものが探索空間に無い。
    実測（1->2 区間、-90 度から 0 度）では -90 度から -178.6 度へ最初の
    0.199 m で跳び、終端でもう一度 180 度跳ぶ計画が出ていた。要求角加速度は
    105 rad/s^2、使えるのは 2.0 rad/s^2 である。出口の制限がそれを削るので
    機体は基準姿勢に追従できず、2.39 m の区間が 5.03 s から 6.81 s へ延びて
    いた。

    ``ramp_yaw`` は弧長に線形な既定の姿勢配分。各点でこれを候補に入れておく
    と、探索空間が既定の解を必ず含むので、最適化の答えが既定より遅くなる
    ことがない。
    """
    values = [heading - 0.5 * math.pi * k for k in range(4)]
    values.append(goal_yaw)
    values.append(current_yaw)
    if ramp_yaw is not None:
        values.append(ramp_yaw)
    step = math.radians(extra_step_deg)
    if step_limit is not None and float(step_limit) > 0.0:
        step = min(step, float(step_limit))
    count = int(round(2.0 * math.pi / step))
    values.extend(goal_yaw + step * k for k in range(count))
    unique = []
    for value in values:
        wrapped = wrap(value)
        if all(abs(wrap(wrapped - kept)) > math.radians(3.0) for kept in unique):
            unique.append(wrapped)
    return np.asarray(unique, dtype=float)


def plan_yaw_profile(
    points,
    tangents,
    *,
    envelope,
    ellipse_x,
    ellipse_y,
    current_yaw,
    goal_yaw,
    clearance_model=None,
    clearance_cap=0.35,
    minimum_clearance=0.02,
    clearance_weight=1.6,
    max_yaw_step,
    yaw_effort_sec_per_rad=0.05,
):
    """各サンプル点の姿勢を選ぶ。points/tangents は同じ長さ。

    経路の通過時間[s]を最小化する。戻り値は姿勢の配列で、終端は goal_yaw、
    始端は current_yaw に固定する。clearance_model が None なら余裕項なしで
    時間だけを見る。

    ``yaw_effort_sec_per_rad`` は同時間の解が並んだときに旋回の少ない側を
    選ばせるためだけの重み。45度オムニでは進行方向を体軸へ合わせる姿勢が
    4通りあり、どれも同じ速度なので、これが無いと 90 度違う姿勢のどれかが
    任意に選ばれる。
    """
    count = len(points)
    if count == 0:
        return np.zeros(0)
    if count == 1:
        return np.asarray([wrap(goal_yaw)])

    headings = np.arctan2(tangents[:, 1], tangents[:, 0])
    segments = np.linalg.norm(np.diff(points, axis=0), axis=1)
    segments = np.maximum(segments, 1.0e-6)
    arclength = np.concatenate(([0.0], np.cumsum(segments)))
    # 既定の姿勢配分（弧長に線形）。候補に入れて、最適化が既定より遅い解を
    # 返せないようにする。
    ramp = wrap(current_yaw) + wrap(goal_yaw - current_yaw) * (
        arclength / max(float(arclength[-1]), 1.0e-6))
    # 余裕の利得を秒へ換算する係数。元の式は -(v + w*m) を最小化していた
    # ので速度感度は -1 [s^-1 換算なし]、時間の速度感度は d(ds/v)/dv =
    # -ds/v^2 である。同じ釣り合いを保つには w を ds/v^2 倍すればよい。
    reference_speed = max(
        envelope.speed_limit(
            np.array([1.0, 0.0]), 0.0, ellipse_x, ellipse_y),
        1.0e-6,
    )
    seconds_per_metre_of_margin = clearance_weight / reference_speed ** 2
    infeasible = 1.0e6

    # 姿勢ごとのフットプリント余裕は多角形と壁線分の距離なので高い。
    # base_link から最近壁までが「外接半径 + cap」より遠ければ、どの姿勢でも
    # 余裕は cap 飽和なので評価する意味がない。開けた区間はここで落ちる。
    open_field = np.zeros(count, dtype=bool)
    if clearance_model is not None:
        reach = float(getattr(clearance_model, 'radius', 0.6)) + clearance_cap
        for index in range(count):
            base, _ = clearance_model.clearance_and_gradient(points[index])
            open_field[index] = base >= reach

    candidates = []
    stage_cost = []
    for index in range(count):
        if index == 0:
            options = np.asarray([wrap(current_yaw)])
        elif index == count - 1:
            options = np.asarray([wrap(goal_yaw)])
        else:
            options = yaw_candidates(
                headings[index], goal_yaw, current_yaw,
                ramp_yaw=ramp[index],
                step_limit=min(max_yaw_step[index - 1], max_yaw_step[index]))
        # この点が代表する弧長。端点は半分。
        own = 0.5 * (
            (segments[index - 1] if index > 0 else 0.0)
            + (segments[index] if index < count - 1 else 0.0)
        )
        costs = np.zeros(len(options))
        if clearance_model is not None:
            for slot, yaw in enumerate(options):
                if open_field[index]:
                    margin = clearance_cap
                else:
                    margin = clearance_model.body_clearance(
                        points[index], float(yaw), cap=clearance_cap)
                    if margin < minimum_clearance:
                        costs[slot] = infeasible  # 通れない姿勢は選ばない
                        continue
                costs[slot] = -seconds_per_metre_of_margin * margin * own
        candidates.append(options)
        stage_cost.append(costs)

    # 前向きDP
    total = [stage_cost[0].copy()]
    back = [np.zeros(len(candidates[0]), dtype=int)]
    for index in range(1, count):
        previous = candidates[index - 1]
        current = candidates[index]
        step = segments[index - 1]
        signed = np.arctan2(
            np.sin(current[:, None] - previous[None, :]),
            np.cos(current[:, None] - previous[None, :]),
        )
        difference = np.abs(signed)
        # 区間の途中の姿勢で進行方向を機体系へ落とす。速度と旋回は
        # どちらも v に比例するので、上限は割り算で厳密に出る。
        mean_yaw = previous[None, :] + 0.5 * signed
        relative = headings[index - 1] - mean_yaw
        speed = envelope.speed_limits(
            np.cos(relative), np.sin(relative), signed / step,
            ellipse_x, ellipse_y)
        travel = step / np.maximum(speed, 1.0e-6)
        transition = np.where(
            difference <= max_yaw_step[index - 1],
            travel + float(yaw_effort_sec_per_rad) * difference,
            np.inf,
        )
        combined = transition + total[index - 1][None, :]
        choice = np.argmin(combined, axis=1)
        best = combined[np.arange(len(current)), choice]
        if not np.all(np.isfinite(best)):
            # 到達不能な候補は落とすが、全滅なら制約を緩めて連続性を守る。
            # 候補格子が step_limit 間隔で、既定の配分も候補に入っている
            # ので、ここへ来るのは「その区間長では既定の配分すら回り切れない」
            # ときだけである。旋回の少ない側を選び、コストは秒のまま積む。
            if not np.any(np.isfinite(best)):
                choice = np.argmin(difference, axis=1)
                rows = np.arange(len(current))
                best = (
                    total[index - 1][choice]
                    + travel[rows, choice]
                    + float(yaw_effort_sec_per_rad) * difference[rows, choice]
                )
        total.append(best + stage_cost[index])
        back.append(choice)

    yaws = np.zeros(count)
    slot = int(np.argmin(total[-1]))
    for index in range(count - 1, -1, -1):
        yaws[index] = candidates[index][slot]
        slot = int(back[index][slot])
    # 巻き戻しを連続な角度へ展開する（-pi/pi をまたいでも滑らかにする）
    unwrapped = np.zeros(count)
    unwrapped[0] = yaws[0]
    for index in range(1, count):
        unwrapped[index] = unwrapped[index - 1] + wrap(
            yaws[index] - yaws[index - 1])
    return unwrapped


def smooth_yaw_profile(yaws, arclength, smoothing):
    """姿勢列の角を丸めて、角速度を連続にする。端点は厳密に保つ。

    動的計画法は 0.30 m ごとの離散候補から姿勢を選ぶので、出てくる姿勢列は
    折れ線である。角速度は ``d(yaw)/ds * v`` なので、折れ線のままだと節点
    ごとに角速度が階段状に飛ぶ。フィードフォワードにそれを入れると出口の
    加速度制限が毎回それを削り、機体は基準姿勢に対して遅れる。つまり
    「回りながら走る」がいちばん要求される場面でだけ、指令が実現不能に
    なっていた。

    経路の平滑化 (``smooth_path``) と同じ [1,2,1]/4 の反復で、ディリクレ
    境界なので始端（機体の実姿勢）と終端（指定姿勢）はどちらも動かない。
    2次微分が有界になるので、時間割りの側で角加速度を上限に収められる。
    """
    count = len(yaws)
    if count < 3 or smoothing <= 0.0:
        return np.asarray(yaws, dtype=float)
    total = float(arclength[-1]) - float(arclength[0])
    spacing = total / (count - 1)
    if spacing <= 0.0:
        return np.asarray(yaws, dtype=float)
    # 1回の [1,2,1]/4 の分散は spacing^2 / 2 なので、長さ尺度 smoothing に
    # 達するのに必要な回数はこれ。
    passes = int(round(2.0 * (smoothing / spacing) ** 2))
    if passes < 1:
        return np.asarray(yaws, dtype=float)
    smoothed = np.array(yaws, dtype=float, copy=True)
    for _ in range(passes):
        smoothed[1:-1] = (
            0.25 * smoothed[:-2] + 0.5 * smoothed[1:-1] + 0.25 * smoothed[2:])
    return smoothed


def direction_speed_limits(
    tangents, yaws, arclength, *, envelope, ellipse_x, ellipse_y
):
    """姿勢が決まった後の、点ごとの速度上限。旋回コストも含める。"""
    count = len(tangents)
    if count == 0:
        return np.zeros(0), np.zeros(0)
    yaw_per_metre = yaw_derivative(yaws, arclength)
    headings = np.arctan2(tangents[:, 1], tangents[:, 0])
    relative = headings - np.asarray(yaws, dtype=float)
    limits = envelope.speed_limits(
        np.cos(relative), np.sin(relative), yaw_per_metre,
        ellipse_x, ellipse_y)
    return limits, yaw_per_metre


def yaw_derivative(yaws, arclength):
    """弧長についての姿勢の1次微分[rad/m]。中心差分。

    以前は前向き差分で、最後の点は隣の値の写しだった。中心差分にすると
    節点が半セルずれないので、速度上限と角加速度上限が同じ場所を指す。
    """
    yaws = np.asarray(yaws, dtype=float)
    arclength = np.asarray(arclength, dtype=float)
    if len(yaws) < 2:
        return np.zeros(len(yaws))
    steps = np.diff(arclength)
    if float(arclength[-1] - arclength[0]) <= 0.0:
        return np.zeros(len(yaws))
    if np.any(steps <= 0.0):
        # 経路に重複点があると np.gradient は 0 で割る。指令に NaN を出す
        # よりは、間隔を下限で押さえた片側差分で通す。
        safe = np.maximum(steps, 1.0e-6)
        derivative = np.zeros(len(yaws))
        derivative[:-1] = np.diff(yaws) / safe
        derivative[-1] = derivative[-2]
        return derivative
    return np.gradient(yaws, arclength)
