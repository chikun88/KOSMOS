# Omni Autonomy Next

## 2026-10-08 経路追従と3.5 m/s要求の改善

sprintの要求上限を **3.5 m/s** に揃え、曲率・加減速の実行可能な予算と
追従補正後の指令を一致させました。曲がり角の手前から減速し、近接する復路へ
進捗が飛ぶ問題、遅れ予測の数値誤差、車輪履歴外のLiDAR補正、古いUDP返信も修正しています。
CAD距離検査は判定を維持したまま高速化しました。

本番追従器による固定曲線の再生検証を追加しています。
**実機3.5 m/s・全経路の高精度走行は未確認**で、既存の代用モデルには
49経路中3接触予測が残ります。ソフトウェア検証と実機受入を区別してください。
[変更・計測条件・残る確認事項](docs/PATH_TRACKING_20261008.md)を参照してください。
以下の4 m/s設定などは過去の履歴です。

## 2026-10-07 システム監査

停止・通信・センサー鮮度・座標系・目標切替・幾何判定の不具合を修正し、
回帰テストと ROS 2 Jazzy / C++ の自動検証を追加しました。
CAD はパッケージに同梱し、旧PCの絶対パスや実行ディレクトリへの依存を除去しました。
接触予測のある全域高速旋回の制限解除は無効化し、`sprint_turn_everywhere: false`
としました。以下の2026-09-16の記述は変更当時の履歴です。

現行設定のオフラインモデルは **49経路中42到着、4接触予測、2時間切れ、1進捗停止** で、
全経路の受入条件を満たしていません。これは本番軌道追従器の再生や実機試験ではありません。
過去の1200/1200成功ファイルも現行版の合格証拠には使えません。
[監査結果・修正・検証条件](docs/SYSTEM_AUDIT_20261007.md)を参照してください。

ソフトウェアのみの検証は `bash scripts/verify_software.sh`、
オフライン受入条件も含む検証は従来の `python3 run.py verify` です。
ソフトウェア検証の成功だけでは実機走行の受入は完了しません。

`omni_autonomy_next` is a new, independently versioned control system for the
2026 MU3 four-wheel omni robot.  It retains only measured hardware facts and
the proven device transports from the previous systems.  Navigation,
configuration, safety composition, operator interaction, and verification are
organized as a new architecture.

## 実走行に基づく制御補正

### 行先指定後の発進待ちとNav2起動待ちの修正（2026-09-16）

バケツ接続経路の計算を非同期化し、計算によってNav2の状態応答が滞る処理を修正しました。
退避位置→バケツ③の接続計算は、この環境で約0.80秒から約0.14秒へ短縮。
採用経路のCAD検査とNav2の準備確認を維持し、次回起動から反映されます。
実機の発進時間は未検証です。[変更と検証](docs/GOAL_STARTUP_LATENCY_20260916.md)。

### 4 m/s巡航に向けた発進加速の追加調整（2026-09-16）

起動時sprintの加速度を **3.0→3.3 m/s²**、guardを3.5 m/s²へ変更しました。
4 m/s上限と既存の制動0.85 m/s²を使用します。理想16 m直線の計画上の
4 m/s巡航時間は0.978→1.038秒。全移動で4 m/sを維持する変更ではありません。
実応答を外挿した16 m直線モデルの最高速は約3.74 m/sで、実機4 m/sは未確認です。
関連133テスト・ビルド成功。次回起動から反映されます。
[検証と制約](docs/LONGER_CRUISE_20260916.md)。

### 全域での高速回転移動（2026-09-16・当時の設定）

校正済み同時旋回の高速 `sprint` で `sprint_turn_everywhere: true` を有効にしました。
場所のCAD余裕や前後・横・斜めの発進方向に関係なく、回転中も選択した
速度上限（現設定4.0 m/s）の車輪合成予算を使います。予測制御が必要です。
停止距離・加減速・曲率・終点制動・衝突監視・UART上限は引き続き適用されるため、
常に4.0 m/sで走る設定ではありません。通常balancedの設定は従来どおりです。

実機走行は未検証です。過去の一律解除の保存経路再生では壁との接触予測があり、
今回のソフトウェアテストは走行安全性を保証しません。
[過去の比較結果](docs/UNRESTRICTED_TURNING_20260916.md)。
次回起動時に反映されます。`config/robot.yaml` の `calibrated_tracking` 内で
`sprint_turn_everywhere: false` に戻すと、下記の従来の場所・方向制限を使用します。
下記の比較結果はそれぞれの変更当時の記録です。

### 回転中の高速移動を拡張（2026-09-16）

予測制御付きsprintで、前後方向へほぼ直線に発進し、経路全体で15 cmの
CAD余裕（サンプル間の車体移動分を控除後）を確認できる場合、回転中の
速度予算を **3.0→4.0 m/s** に引き上げました。横・斜め方向の発進は既存予算です。
4.0 m/sは指令上限であり、実速度の保証ではありません。
[比較結果と適用条件](docs/FASTER_ROTATING_SPRINT_20260916.md)。次回起動から適用されます。

### 高速設定の保持とバケツ通過の後戻り対策（2026-09-16）

起動時の走行プロファイルを **sprint** に変更し、GUIの起動・再表示で
balancedへ上書きする処理を除去しました。既存の校正済み同時旋回・予測制御が
条件を満たす経路で選ばれます。中継点は連続する新鮮なTFの間で通過した場合も
判定し、目標切替後に遅れて届く旧目的地への経路を除外します。
実機の高速旋回・全バケツ間の安定走行は未確認です。
[変更内容と検証](docs/BUCKET_PASSAGE_STARTUP_20260916.md)。次回起動から適用されます。

### 回転しながら3 m/sへ向けた予測制御（2026-09-16）

指令履歴による遅れ予測、回転座標に対応した加速度制限、開放経路での滑らかな
姿勢変更を実装しました。校正済みハードウェアのsimultaneous・sprintで、
経路余裕を確認できる区間の回転中予算を1.30→3.00 m/sへ引き上げます。
実記録25経路・50条件の再生では、変更前に基準を満たす19条件に新たな退行はありません。
**実機3 m/s・全地点3 m/sは未確認**です。次回起動後のsprintで適用されます。
[実装・モデル検証・適用条件](docs/PREDICTIVE_THREE_MPS_20260916.md)を参照してください。

### 回転移動4 m/sに向けた追加比較（2026-09-16）

回転中予算の1.32〜1.60 m/sへの引き上げと先読み補償を比較しましたが、
一部経路で到着遅延・CAD余裕不足が発生したため、今回の候補は適用していません。
**全地点での高速回転移動・実速度4 m/sは未達成**です。
[比較結果・制約・再現手順](docs/TURN_SPEED_CANDIDATES_20260916.md)を保存しました。

### 発進加速の改善（2026-09-16）

通常balancedの加速度は**1.8 m/s²**、高速sprintは**3.0 m/s²**へ設定しました。
従来0.85 m/s²に対して約2.1倍／3.5倍。次回起動から適用されます。
直近の実機ログは旧加速設定で、最終指令最大0.871 m/sでした。
実記録の経路を使うモデル比較では到着時間が約4〜7%短縮しましたが、実走改善は未確認です。
[解析・設定・比較結果](docs/FAST_ACCELERATION_20260916.md)を参照してください。

### 現在の速度設定（2026-09-16更新）

全自動走行モード（`precision` / `balanced` / `sprint`）のUART各輪上限を
**10000**に揃えました。速度要求は通常2.0 m/s、高速 `sprint` は4.0 m/s、
起動時は `sprint` です。次回起動から適用されます。
実機ログの再解析では、10000上限での純軸速度の外挿は約3.6〜3.8 m/sで、
**実機6 m/sは未達成**です。4 m/s要求ですでに10000に飽和するため、
6 m/s要求に変えても純軸のモーター出力は増えません。
[最新の解析・変更・検証](docs/SPEED_10000_20260916.md)を参照してください。
以下は過去の変更履歴です。

[巡航2 m/sと区間制限の更新](docs/CRUISE_FLOOR_20260915.md)により、低速precisionも
前後・左右の巡航上限を**2.0 m/s**へ変更しました。旋回を含む経路では、
姿勢が変わる区間に追加制限を掛け、姿勢が一定の区間では最高速を許可します。
発進・停止・旋回・障害物回避などでは2 m/s未満になり、実速度の下限保証ではありません。
次回起動から適用されます。変更後の実走確認は未実施です。

追加指定により、狭い入口・外側通路の局所速度上限も**2.0 m/s**へ変更し、
指定地点付近の70%速度倍率を100%へ変更しました。曲率・車輪・障害物制限は
別途掛かります。以前の狭路モデル再生では接触があり、実機での安全性は未確認です。

[実機6 m/s要求の検証](docs/SIX_MPS_20260915.md)では、最新の実機記録3本と
比較用記録1本を再解析しました。6 m/s要求は現行UART上限の約2倍に相当し、
送信時に約49%へ縮小されます。実機6 m/sは未達成で、運用上限は2.0 m/sのままです。
出力・停止距離・検知範囲の条件を報告にまとめています。
追加の24条件のオフライン比較では、既存指令上限を使い切ってもモデル速度は約3 m/s。
上限を仮に2倍以上にした外挿モデルは5.93〜6.18 m/sでしたが、実機の許容出力は未確認で、
これらの候補は運用設定に適用していません。

画像の赤丸をスタート出口と解釈し、その曲がり速度の局所上限を
**3.50 m/s**に設定しました。バケツ入口は追加指定により2.00 m/sです。
全体の2.0 m/s上限や車輪・曲率制限が別途掛かるため、実速度3.50 m/sにはなりません。
この設定での走行検証は未実施です。
以前のオフライン18条件の短縮結果は0.24 m/s設定のもので、1.00 m/sには適用できません。
[設定と検証結果](docs/LOCAL_SPEED_20260915.md)を参照してください。
次回起動から適用されます。変更後の実走確認は未実施です。

2026-09-15の固定バケツ周辺の後戻りは、中継点を通過判定の外側で通り過ぎ、
再計画でその点へ引き返す動きでした。巡航速度を維持するため、中央バケツの
中継点は外側を通過したことでも判定し、減速は曲がる区間に限定しました。
通常・sprintの2.0 m/s設定を保ち、直線ではゲート付近でも一律低速にしません。
[速度を維持する修正と検証](docs/BUCKET_FAST_STABILITY_20260915.md)を参照してください。
次回起動から適用されます。変更後の実走確認は未実施です。

2026-09-15朝の「走行途中で極低速になる」記録から、後方LiDARの継続的な
自己反射による衝突監視の減速を特定し、センサー・角度・距離を限定した補正を
追加しました。[原因と再生検証](docs/SLOWDOWN_20260915.md)を参照してください。
次回起動から適用されます。変更後の実走確認は未実施です。

2026-09-14の実走行2回から過大な速度応答を同定し、通常の実機起動に
並進0.38・旋回0.34の送信補正を設定しました。次回起動から適用されます。
根拠・モデル比較・再走行時の確認事項は
[実走行応答の改善](docs/FIELD_RESPONSE_20260914.md) を参照してください。

補正済み実機の追従ゲインと遅れ補償を調整した
[到達時間の短縮](docs/CALIBRATED_SPEED_20260914.md) も追加しています。
効果はオフライン比較で検証し、変更後の実機速度は未測定です。
さらに[モーター速度指令の引上げ](docs/MOTOR_SPEED_20260914.md)として、通常モードを
前後0.95・左右0.78 m/s、sprintを前後1.00・左右0.85 m/sへ変更しました。
既存の車輪上限内で巡航速度を増やす設定です。到達時間の短縮は経路と遅延に依存します。

2026-09-15の[巡航速度・低速モードの底上げ](docs/NEAR_LIMIT_SPEED_20260915.md)で、
通常モードを前後・左右1.05 m/s、sprintを1.10 m/s、precisionを
前後0.55・左右0.50 m/sへ更新しました。送信側の固定上限も設定と同期します。
既存車輪モデルの上限内での変更であり、実モーターの限界速度は未測定です。
停止・目標付近ではゼロまで減速します。

追加の[2.0 m/s設定](docs/TWO_MPS_20260915.md)により、現在の通常・sprint上限は
前後・左右とも2.0 m/sです。残り距離に応じた制動処理を追加し、
姿勢を大きく変える経路・斜め移動・障害物付近では減速します。実機速度は未測定です。

[3 m/sの実走行データ調査](docs/THREE_MPS_20260915.md)では、2.0 m/s設定の実機記録から
最終指令最大1.095 m/s・計測輪の約0.5秒変位で最大1.063 m/sを確認しました。
3 m/s要求で生じるUART飽和をオフラインモデルにも反映しました。
実機3 m/sは未検証のため、通常設定は2.0 m/sを保持しています。

## 実機走行データの自動記録

通常起動で走行データを自動保存します。保存先は
`~/.ros/omni_autonomy_next/runs/<日時-ID>/` です。
各段の速度指令、計測輪、自己位置、経路・目標、安全状態、モーター通信、
前後LiDAR、設定を時刻付きJSONLに記録します。
`python3 scripts/analyze_run.py <保存フォルダー>` で指令と実測値を比較するCSVを生成できます。
保存容量・記録欠落の確認方法は [走行データ回収](docs/RUN_RECORDING.md) を参照してください。

## What is different

- Nav2 MPPI runs with the holonomic `Omni` model.  It samples simultaneous
  forward, lateral, and yaw motion instead of the former 9 x 3 x 10 DWB grid.
- A cost-aware Smac 2-D global planner and full polygon collision checks are
  used with a dual-LiDAR local costmap.
- Nav2 starts immediately after the CAD localizer reports ready. A guarded
  26-second fallback remains for abnormal starts, without making every normal
  startup wait for the full timeout.
- Numbered goals 4 and 5 use CAD-validated, predetermined approach and
  departure lanes. The loading pose 1 also has a straight departure gate so
  the base clears its 49 mm slot before changing yaw. Nav2 follows these fixed
  lanes instead of choosing a new diagonal on each replan, and still plans
  around live obstacles between the gates.
- Collision Monitor and `RuntimeGuard` form two independent command gates.
  Loss of tracking, stale commands, motor-link loss, disarm, or E-stop always
  produces an immediate zero command.
- The final guard enforces acceleration, jerk, vector-speed, and actual
  four-wheel angular-speed limits.  Yaw and translation are fitted into the
  wheel budget together: `translation_budget_share` in `config/robot.yaml`
  reserves 45% of every wheel's limit for translation whenever translation
  is requested, so goal-yaw regulation can no longer starve progress.  A
  pure in-place turn still receives the whole budget.
- MPPI plans inside the envelope that guard actually executes.  RuntimeGuard
  republishes the operator's profile and speed scale as a Nav2 `SpeedLimit`,
  and the Monte Carlo gate reads its profile from the deployed runtime
  configuration, so no stage can quietly run slower than the one that was
  validated.
- Goal yaw is regulated for the whole route rather than only near the goal, so
  the base retires rotation and translation together instead of arriving and
  then spinning.  The rotation is finished `terminal_approach_m` *before* the
  goal, which makes the final approach a pure translation settle and decouples
  the arrival yaw error from how fast that approach runs.
- The trajectory tracker plans inside the envelope `RuntimeGuard` reports it is
  passing, read from `/system/safety_state`.  A profile change rebuilds the
  trajectory; the speed scale — operator slider, learned residual, and the
  red-zone factor near every pose — rescales the reference clock instead
  (path velocity scaling), so the feed-forward always describes the robot the
  gate will actually allow.  Selecting `sprint` in the panel therefore changes
  how the base drives, not only what the gate permits.
- A PyQt control panel changes profiles and speed scale while running and
  exposes arm/E-stop controls without editing YAML.
- Offline Monte Carlo tests validate the CAD field, all named poses, and the
  image-marked central target area before deployment.
- A safety-constrained tabular Q learner can adapt simulated speed and obstacle
  repulsion.  Its residual actions may only slow the baseline controller or
  increase clearance response.  A saved policy must not regress any route the
  deterministic controller already solved, and must not have collided.
- The offline model uses the exact clearance of the real ten-vertex footprint at
  its actual yaw, plus the deployed nav2 arrival, progress and Collision Monitor
  gates.  Goals 4 to 7 beside the fixed bucket have 47-59 mm of footprint
  clearance against a 40 mm arrival tolerance, so the field-acceptance campaign
  does not pass; see `docs/REINFORCEMENT_LEARNING.md`.

## Repository layout

- `ros2_ws/src/omni_autonomy_interfaces`: repository-owned typed service.
- `ros2_ws/src/omni_autonomy_next`: Jetson ROS 2 package.
- `ros2_ws/src/sllidar_ros2`: vendored Slamtec RPLIDAR driver, built by
  `--packages-up-to omni_autonomy_next`; see its `VENDORED.md`.
- `bacon_gateway`: Raspberry Pi motor gateway, isolated from the old repo.
- `simulation`: deterministic planner/dynamics model, Monte Carlo tuner, and
  offline reinforcement-learning pipeline.
- `docs`: architecture, inherited measurements, and test procedure.

Current host deployment evidence and remaining physical gates are recorded in
`docs/DEPLOYMENT_STATUS.md` and `docs/ACCEPTANCE.md`.

## Quick start

From the repository root, open the unified launcher:

```bash
python3 run.py
```

Choose a function by number.  The menu covers the safe demo, complete robot
system, GUI, RViz, simulation, autotuning, verification, CAD conversion, and
the bacon6 gateway build/test/run tools.  ROS and gateway binaries are built
automatically when they are missing.  Stop a running function with `Ctrl+C`.

Every function also has a short direct command:

```bash
# Safe synthetic demo
python3 run.py demo

# Safe end-to-end RL demo to configured target 4
python3 run.py demo --goal-id 4

# Start from loading-wait pose 0 instead of the match-image pose 1
python3 run.py demo --initial-pose-id 0 --goal-id 1

# Real LiDARs and measurement wheels, with motor output disabled
python3 run.py real

# Full software verification and simulation tools
python3 run.py verify
python3 run.py campaign
python3 run.py autotune
python3 run.py rl-train
python3 run.py rl-evaluate

# Send another configured target to an already running system
python3 run.py goal --goal-id 5

# Save the robot's current localized pose, then return to it by name later
python3 run.py remember --pose-name 作業台前
python3 run.py remembered-goal --pose-name 作業台前

# Individual operator tools
python3 run.py gui
python3 run.py rviz

# Commissioning checks for the drive chain
# Wheel-odometry signs and scale against the LiDAR pose, pushed by hand.
# Needs a running system with motors:=false; no motor moves.
python3 run.py check-odometry

# Record one run and report whether a weave is in the command or the
# response.  Run it during a goal and stop it with Ctrl+C.
python3 run.py record-run

# Measure the whole command chain during a goal: per-stage rate, per-stage
# oscillation, per-stage phase lag, delivered acceleration, and cross-track
# error against the active plan.  This is the tool that localises a weave to
# the stage that generates it.
python3 run.py diagnose-chain --seconds 45

# ACCEPTANCE step 3: 実機で +x / +y / +yaw の物理方向を確定する。
# 計測輪オドメトリを基準に1軸ずつ低速で動かし、逆の軸があれば
# bacon_gateway/include/value.hpp のどの定数を反転するかまで出す。
# 同じ走行で駆動系ゲイン（指令に対して実際に出る速度の比）も測り、
# auto_units_per_mps をいくつへ直すかまで出す。あの値は「フルスティック
# = 0.55 m/s」の仮定から出したもので実測されていない。1.0 から外れて
# いると速度だけでなく姿勢が自励振動する（走行中のフラフラ）。
# モーターが回るので、車輪浮上試験を通してから実行すること。
python3 run.py check-drive-directions --accept-motor-risk
# この機体が物理的に出せる上限（速度・加速度・精度の内訳）を計算する。
# 実機もROSも要らない。速度や精度の議論はまずこれを見てから。
python3 scripts/performance_envelope.py

# MPPI の 20 Hz フィードバックの代わりに、時間パラメータ化した軌道を
# 20 Hz でフィードフォワード追従する。安全ゲートは一切変わらない。
# tracker は既定で true。MPPI へ戻すときだけ tracker:=false を付ける。
ros2 launch omni_autonomy_next system.launch.py demo:=true tracker:=true \
  lidars:=false wheels:=false motors:=false goal_id:=4

# 追従器の姿勢計画と終端の寄せを、実測した駆動系モデル（無駄時間 120 ms +
# 一次遅れ 80 ms）に対して閉ループで比べる。実機もROSも要らない。
#   --rotation  区間ごとの所要時間と、要求角加速度・出口制限の当たり具合
#   --terminal  終端ゾーンの大きさが所要時間と整定精度にどう効くか
python3 scripts/yaw_oscillation_probe.py --rotation
python3 scripts/yaw_oscillation_probe.py --terminal

# 駆動系ゲイン（指令 1 m/s に対して実際に出る速度）の誤差に対して、姿勢の
# ループがどこまで耐えるか。走行中に機体の向きが左右に振れるときはここ。
#   --margin  yaw_gain ごとの余裕。既定 1.6 は x3.2 まで収束する
python3 scripts/yaw_oscillation_probe.py --margin

# bacon6 gateway tools
python3 run.py gateway-build
python3 run.py gateway-test
python3 run.py gateway-monitor

# All commands and options
python3 run.py --help
```

## GUIで地点番号を指定する

通常はリポジトリ直下で `python3 run.py demo`（実機なし）または
`python3 run.py real`（実機センサー、モーター出力なし）を起動します。
表示された「行き先指定」で地点番号を選び、座標・向きを確認して
「選択した番号へ移動」を押してください。

|番号|行き先|
|---:|---|
|0|装填待機|
|1|装填位置|
|2|位置取り待機|
|3|退避位置|
|4|固定バケツ②|
|5|固定バケツ③|
|6|旗上側|
|7|旗下側|

地点4・5を選ぶと、現在位置が目標の上側か下側かに応じて、CADで検証済みの
固定進入ゲートを自動選択します。地点4は `(-0.80, 1.75)` または
`(-0.80, 0.75)`、地点5は `(-0.80, -1.95)` または
`(-0.80, -2.75)` から、各目標まで `x=-0.80 m` のレーンを通ります。
地点4・5から別の目的地へ出発するときも、目的地側の同じゲートまで固定レーンを
逆向きに通ってから通常経路へ合流します。地点4→5では地点4の下側退出ゲートと
地点5の上側進入ゲートを順に通り、地点5→4ではその逆です。
中継点はCADで車体の通過余裕を確認し、接続可能な候補から距離が短い経路を選びます。
バケツ②↔③では中央のバケツ①を `x=-1.55 m`（右側は `+1.55 m`）の
短い通路で回避します。スタートからバケツ③へ向かう場合など、バケツ②も回り込む
必要がある区間では外側の `x=±2.20 m` の候補を使います。
中継点間はNav2が動的障害物を回避し、全区間でCollision MonitorとRuntimeGuardが
有効です。

進入ゲートの通過判定は16 cmです。12 cmでは、わずかに外側を通ったゲートが
未通過のまま残り、戻ってから進む動作が生じたため調整しています。

走行中に別番号を押すと、現在の目標をキャンセルして新しい番号へ切り替えます。
「移動を停止」は待機中・送信中・走行中の目標を取り消します。GUIの移動状態が
「到着」になるまでを完了とし、経路が表示されただけでは到着扱いにしません。

### 現在地を記憶して同じ場所へ戻る

GUIの「現在地を記憶して戻る」で地点名を入力し、
「この名前で現在地を保存」を押すと、現在の `map` 座標と向きが保存されます。
保存済みの名前を選んで「記憶した地点へ移動」を押すと、Nav2の障害物回避と
既存の安全ゲートを使って同じ姿勢へ戻ります。同名で保存すると、確認後に現在地で
選択中のフィールドの位置だけを上書きします。自動装填・各地点ボタン・任意名の地点は、
左／右フィールドそれぞれ独立して調節・保存できます。「走行するフィールド」を選び、
機体を目的の位置・向きに合わせて停止してから、対象地点を保存してください。
未調節の側は従来の設定（右側は鏡映した座標）を使用します。

バケツ②・③の設定地点から30 cm以内の保存地点は、番号指定と同じ固定進入・
迂回経路を使用します。`BAKETU2`・`BAKETU3` や別名で保存した地点にも適用され、
保存した最終座標と向きは変更しません。走行中の行き先切り替えでも現在位置から
進入ゲートを選び直します。

保存先は既定で `~/.ros/omni_autonomy_next/remembered_poses.json` です。
システムを再起動しても残ります。保存操作はARM不要ですが、移動操作には固定地点と
同じARM・非常停止条件が適用されます。局在姿勢を未受信、または1秒以上受信できて
いない場合は、誤った位置を記憶しないよう保存を拒否します。CLIからは次のようにも
操作できます。

```bash
python3 run.py remember --pose-name 作業台前
python3 run.py remembered-goal --pose-name 作業台前
```

ROS 2から直接使う場合、保存要求は `/navigation/remember_pose_request`、移動要求は
`/navigation/remembered_goal_request`（ともに `std_msgs/msg/String`）です。保存一覧と
結果は `/navigation/remembered_poses` にJSONで配信されます。保存先を変更する場合は
launch引数 `remembered_poses_file:=/path/to/poses.json` を指定してください。

実機を動かす場合だけ、`docs/ACCEPTANCE.md` の車輪浮上試験後に
`python3 run.py full --accept-motor-risk` を使用します。初回は速度上限を10%へ
下げてから「自動走行 ARM」を有効にしてください。非常停止中や未ARM時はGUIから
目標を開始できません。`python3 run.py gui` は、すでに起動しているシステムへ
操作パネルだけを追加するときに使います。

Motor-capable functions have an additional safety interlock.  Only after the
wheels-raised tests in `docs/ACCEPTANCE.md` pass, use the interactive menu or:

```bash
# Complete Jetson robot system
python3 run.py full --accept-motor-risk

# Hardware gateway on bacon6
python3 run.py gateway --accept-motor-risk --dashboard
```

## Build on egg8

```bash
cd ~/Desktop/omni_autonomy_next/ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install --packages-up-to omni_autonomy_next
source install/setup.bash
ros2 launch omni_autonomy_next system.launch.py motors:=false rviz:=true gui:=true
```

Motor output is disabled in that command.  Do not set `motors:=true` until the
wheels-raised acceptance tests in `docs/ACCEPTANCE.md` pass.

## Build on bacon6

```bash
cd ~/omni_autonomy_next/bacon_gateway
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build -j2
ctest --test-dir build --output-on-failure
```

The existing repositories are not modified and are not started automatically.
