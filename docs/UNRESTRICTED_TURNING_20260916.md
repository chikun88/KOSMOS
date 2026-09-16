# 回転中固定制限の解除を実経路で比較した結果

**解除案は棄却し、実機用の制御・設定は変更していない。4 m/s達成は未確認。**
最新15:14開始の実記録2経路と、15:00開始の14経路で、回転中の
1.05／1.30 m/s固定予算とCADによる高速許可条件だけを外して比較した。
速度要求はsprintの4 m/sとし、車輪予算、UART上限10000、加減速、曲率、
終点制動は維持した。モーターへの指令送信は一切行っていない。

## 結果

16経路×応答遅れ2条件（0.20／0.30秒）、計32条件。
現行制御で到着しCAD余裕15 mm以上だった13条件のうち、解除後は6条件で
CAD余裕がゼロとなった。到着遅延も含む退行は9条件。
残る19条件は変更前から余裕条件を満たしておらず、合格扱いしていない。

最新記録の具体例：

|経路・遅れ|現行の最高速→解除後|到着時間|最小CAD余裕|
|---|---|---|---|
|14214・0.20秒|1.126→1.956 m/s|9.34→8.99秒|92→26 mm|
|22108・0.20秒|1.065→1.165 m/s|8.52→11.15秒|77→0 mm|

最高速が上がっても、追従を崩して到着が遅くなる経路がある。
最新2経路だけでも一律解除は改善と判断できない。
回転する機体座標に合わせて前周期の指令を変換してから加減速制限する補正案も
メモリ内で試したが、遅れ0.30秒の複数条件で余裕ゼロが残り、採用していない。

CAD余裕ゼロはこのモデル上の壁との接触・交差であり、実機衝突を観測した結果ではない。
低速応答ゲインの外挿と仮定した遅れを使用し、ライブ衝突監視、自己位置の飛び、
手動介入、荷重・滑りは再現しない。したがって安全性の保証にも使えない。

## 保存したもの

- [最新実記録の監査](unrestricted_turn_audit_20260916.json)
- [最新2経路の抽出](unrestricted_turn_routes_20260916.json)
- [最新4条件の比較](unrestricted_turn_latest_replay_20260916.json)
- [直前28条件の比較](unrestricted_turn_previous_replay_20260916.json)
- [座標補正の棄却候補](rotating_frame_candidate_20260916.json)
- [固定制限解除の再現スクリプト](../scripts/check_unrestricted_turning.py)

比較スクリプトは実制御の関数をメモリ内で差し替え、実機用ファイルへ書き戻さない。
退行を検出すると結果を保存して終了コード1を返す。今回の失敗は検証エラーではなく、
変更案が退行条件に該当したことを示す。追加スクリプトの構文チェックも成功した。

```bash
source /opt/ros/jazzy/setup.bash
source ros2_ws/install/setup.bash
OPENBLAS_NUM_THREADS=1 python3 scripts/check_unrestricted_turning.py \
  --fixture docs/unrestricted_turn_routes_20260916.json \
  --audit docs/unrestricted_turn_audit_20260916.json \
  --output docs/unrestricted_turn_latest_replay_20260916.json
OPENBLAS_NUM_THREADS=1 python3 scripts/check_unrestricted_turning.py \
  --fixture docs/latest_speed_retry_routes_20260916.json \
  --audit docs/latest_speed_retry_audit_20260916.json \
  --output docs/unrestricted_turn_previous_replay_20260916.json
```

制限解除の適用には、高速旋回時の追従と遅れへの対応を先に改善する必要がある。
4 m/sそのものについても、現行UART上限と停止距離の制約が残っており、
固定上限を削除するだけでは達成できない。
