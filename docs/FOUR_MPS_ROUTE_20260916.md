# 現在の自動走行ルートで4 m/sを目指す改善（2026-09-16）

## 結果

**実機4 m/sは未達成・変更後の実走行は未実施。**
高速モードの車輪予算が追従器へ読み込まれない不具合を修正した。
ただし最新46経路のオフライン比較では速度改善はほぼなく、
この修正だけで現ルートを4 m/sで走れるとは判断していない。

## 採用した変更

`config.load_robot()` が `drivetrain.profile_max_wheel_speeds` を捨てていた。
追従器はこの正規化済み設定を使うため、sprint選択後も車輪予算が
28.284271247 rad/s（純軸2 m/s相当）のままだった。
RuntimeGuardは生YAMLを読み、56.568542494 rad/sへ切り替わっていたため、
追従器と最終ゲートの予算が不一致だった。

- 正規化ローダーでプロファイル別車輪予算を保持し、有限・正の値を検証する。
- 既存設定に項目がない場合は従来の共通車輪予算へ戻す。
- 実際の読み込み経路を通すテストに変更し、sprint→balanced→sprint、
  軸方向・斜め方向、終点速度0、不正設定の拒否を検証した。
- 比較スクリプトも実ローダーとプロファイル適用を通すよう修正した。
  従来の比較には、車輪予算を直接渡して読み込み不具合を隠すものがあった。

UART上限10000、加速度0.85 m/s²、制動予測、旋回制限、
Collision Monitor、RuntimeGuard、計測輪換算は変更していない。
既定プロファイルはbalanced。次回起動後、sprintを選択すると修正を使用する。
インストール済みモジュールでも車輪予算が28.284→56.569→28.284へ
切り替わることを確認した。稼働ノードの再起動・走行指令送信は行っていない。

## 最新実機ログの根拠

対象は日本時間2026-09-16 12:58開始 `837c90a5` と13:22開始 `114477c8`。
[生ログ監査](four_mps_audit_20260916.json)にハッシュ、欠落数、設定、
各制御段の速度と発行時刻基準の応答推定を保存した。

|記録|約0.5秒変位速度の最大|最終指令の最大|UART各輪指令の最大|
|---|---:|---:|---:|
|12:58開始|1.226 m/s|1.190 m/s|4543 / 10000|
|13:22開始|1.083 m/s|1.085 m/s|4080 / 10000|

いずれもsprint使用記録がある。計画速度の最大は1.331 / 1.313 m/s。
記録された経路全体の長さは最大約8.01 mで、直線距離とは異なる。
当該走行では出力10000への飽和が主な速度制限ではない。
記録欠落は66160 / 33005件あり、これらのログで校正値を更新していない。

## 比較検証

[最新ログから抽出した46経路](recorded_sprint_fixture_20260916.json)を
[同じ制御・同じ応答モデルで比較](recorded_sprint_replay_20260916.json)した。
受信順ではなく発行時刻で経路・目標・自己位置を対応付け、
未来の自己位置、300 ms超の古い自己位置、始点/終点の大きな不一致を除外した。
各目標について条件に合う最初の経路を使用する。

- 変更前後の最高速度差は最大でも約0.004 m/s。現ルートでの有意な向上は未確認。
- 曲率、姿勢変更中の約1 m/s制限、終点への加減速が先に効く。
- 元からモデル上のCAD余裕15 mm条件を満たす19経路には、定義した退行がなかった。
- 残る27経路は変更前からCAD余裕条件を満たさない。実ログにある後退・手動操作・
  自己位置更新・モード遷移・ライブ衝突監視をこのモデルは再現しないため、
  全経路安全性の合格とも、実機で衝突した証拠とも扱わない。
- 応答遅れ0.20秒・一次遅れ0.12秒の単一条件。実機改善の測定ではない。

制限を取り落としていた影響は、[空いた仮想直線での比較](sprint_loader_straight_replay_20260916.json)
で確認した。最新2本の応答ゲイン×前後/左右を使い、UART飽和を含める。

|仮の直線距離|修正前のモデル最高速度|修正後のモデル最高速度|
|---|---:|---:|
|8 m|1.98〜2.06 m/s|2.33〜2.40 m/s|
|60 m|1.99〜2.06 m/s|3.64〜3.77 m/s|

これらは障害物のない仮想直線であり、現在の曲がりのあるルートではない。
低速域の応答を高速まで線形外挿しており、4 m/sの実機性能は裏付けない。

旋回中の制限を2 m/s用の予算へ緩める案も、上記19経路でメモリ内だけ変更して
比較したが、14経路で到着時間が0.15秒以上悪化した。
一部は壁との余裕も減少したため採用しなかった。
[棄却した候補の結果](sprint_turning_candidate_rejected_20260916.json)。

## 4 m/sへ向けて残る条件

1. 現行制動モデルでは4 m/sからの停止に14.07 m、静止から加速して停止するまで
   23.48 m必要。約8 mの経路でも、曲がりを無視した加減速モデルで4 m/sへ届かせるには
   加速度約2.88 m/s²・制動減速度約2.16 m/s²が必要となる。
   これは必要条件の計算であり、実機で出せる加減速値ではない。
2. 最新低速ログからの線形外挿では、出力10000で純軸3.65〜3.78 m/s。
   4 m/s相当の指令は約10581〜10970。現在の上限を超えるため、
   指令値だけ増やしても改善しない。実負荷での駆動応答と飽和域を測る必要がある。
3. 次の実走行では修正後のsprintで同じルートを記録し、速度だけでなく
   到着時間・位置誤差・停止距離を比較する。加減速上限を増やす根拠には、
   計測輪の瞬間最大ではなく連続した変位と、既知距離/時間による測定を使用する。

## 検証と再現

- 関連Pythonテスト309件成功。
- `colcon build --symlink-install --packages-up-to omni_autonomy_next`：4パッケージ成功。
- リポジトリ全体テスト・実機高速走行の合格を示すものではない。

```bash
source /opt/ros/jazzy/setup.bash
source ros2_ws/install/setup.bash
OPENBLAS_NUM_THREADS=1 python3 scripts/check_recorded_sprint.py --quick --workers 2
OPENBLAS_NUM_THREADS=1 python3 scripts/check_sprint_loader_replay.py
ROS_DOMAIN_ID=137 OPENBLAS_NUM_THREADS=1 python3 -m pytest -q \
  ros2_ws/src/omni_autonomy_next/test/test_sprint_speed.py \
  ros2_ws/src/omni_autonomy_next/test/test_near_limit_speed.py \
  ros2_ws/src/omni_autonomy_next/test/test_speed_target_replay.py \
  ros2_ws/src/omni_autonomy_next/test/test_trajectory_tracker.py \
  ros2_ws/src/omni_autonomy_next/test/test_runtime_guard.py \
  ros2_ws/src/omni_autonomy_next/test/test_smooth_arrival.py \
  ros2_ws/src/omni_autonomy_next/test/test_fixed_gate_speed.py \
  ros2_ws/src/omni_autonomy_next/test/test_protocol_and_odometry.py \
  ros2_ws/src/omni_autonomy_next/test/test_inherited_hardware.py \
  ros2_ws/src/omni_autonomy_next/test/test_field_response_calibration.py \
  ros2_ws/src/omni_autonomy_next/test/test_speed_target_audit.py
```
