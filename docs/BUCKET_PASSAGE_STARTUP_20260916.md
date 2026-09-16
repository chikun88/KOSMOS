# 高速設定の保持とバケツ通過の後戻り対策

## 適用した変更

- `runtime.yaml` の起動プロファイルを `balanced` から `sprint` へ変更。
  既存の4 m/s要求、3.0 m/s²発進加速度と、校正条件・CAD余裕を満たす
  同時旋回経路の予測制御を起動時から選ぶ。実速度4 m/sの保証ではない。
- GUIは起動時にbalancedを送信しない。RuntimeGuardの実際のプロファイルを
  信号送信なしで表示し、ユーザーが選択を変更した時だけ送信する。
  GUIを再表示しても高速設定や選択済みのprecision/balancedを上書きしない。
- 中継点の通過判定は、現在位置に加えて、同じ目標列に対する連続したTFの
  移動線分も確認する。半径16 cmの領域を計測間隔の間に横切った時も除去し、
  再計画が通過済み地点へ戻ることを防ぐ。判定半径は拡大しない。
  TF間隔150 ms以下、変位が4.4 m/s×間隔+2 cm以内に限定し、
  古いTF・欠落・要求の差し替えでは履歴を破棄する。最終目標は除去しない。
- 新しい目標の受信後に旧バケツへの計画が遅れて到着しても、終点が新目標の
  スナップ許容範囲外なら除外する。無効な計画で経路監視の時刻も更新しない。
  キュー待ちの間に目標が変わる場合に備え、軌道構築時にも再確認する。

車輪出力、衝突監視、停止距離、狭い旋回経路の上限は既存の制限を維持する。
稼働中ノードの再起動・実機への走行指令送信は行っていない。次回起動から適用。

## 検証

- ROS 2の4パッケージをビルド。
- 関連Pythonテスト225件成功。
- 実際のBTプラグインをロードするC++テスト10件成功。
  4 m/s相当の40 cm移動が中継点を挟むケース、判定範囲の外側、
  TF重複・欠落・長い間隔、自己位置の飛び、目標差し替えを確認。
- GUIのオフスクリーン確認：起動時のプロファイル送信ゼロ、
  sprint/balanced/precisionの表示同期、手動変更時の送信を確認。
- Python全体検証は当初894成功・6不合格。今回の目標整合チェックにより
  無効になったテスト用経路1件は、テストの目標と終点を一致させて再検証した。
  残る5件は今回変更していない部分の不整合で、全体合格とはしていない：
  `test_nav2_architecture` のbalanced角加速度とsmootherの大小関係、
  `test_planned_stop` 1件と `test_staged_continuity` 2件の停止待機時間、
  `test_trajectory_continuity` のテスト用パラメータにpredictive_sprintがない問題。

追加候補として微小な姿勢差でも狭路の角補正を行う案を試したが、
記録経路53876の遅れ0.20/0.30秒条件で発進できなくなったため撤回した。
これを速度・安定性の改善として数えていない。

実機の高速旋回、スタート→バケツ③の到達速度、全バケツ間の無振動走行は未確認。
既存の記録経路モデルにはCAD余裕不足のケースも残る。
今回のソフトウェア検証を、全地点での超高速・安定走行の達成とは扱わない。

## 再現

```bash
source /opt/ros/jazzy/setup.bash
cd ros2_ws
colcon build --symlink-install --packages-up-to omni_autonomy_next
source install/setup.bash
ROS_DOMAIN_ID=193 colcon test --packages-select omni_route_bt
colcon test-result --test-result-base build/omni_route_bt/test_results --verbose
cd ..
OPENBLAS_NUM_THREADS=1 python3 -m pytest -q \
  ros2_ws/src/omni_autonomy_next/test/test_navigation_continuity.py \
  ros2_ws/src/omni_autonomy_next/test/test_sprint_turning.py \
  ros2_ws/src/omni_autonomy_next/test/test_smooth_arrival.py \
  ros2_ws/src/omni_autonomy_next/test/test_sprint_speed.py \
  ros2_ws/src/omni_autonomy_next/test/test_yaw_stability.py \
  ros2_ws/src/omni_autonomy_next/test/test_outer_bucket_passage.py
python3 scripts/check_mu3_navigation_gui.py
```
