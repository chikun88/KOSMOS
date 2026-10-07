# KOSMOS システム監査・修正（2026-10-07）

対象は `chikun88/KOSMOS` の `b567263` を起点とするソースです。ROS 2 Jazzy、Nav2、
自己位置推定、軌道追従、RuntimeGuard、GUI、MU3、UDP/UART gateway、CAD、
オフラインモデル、RL、ビルド・検証経路を調査しました。

発見したソフトウェア上の不具合を修正し、故障入力・競合・停止・起動の回帰検証を
追加しました。実機には接続しておらず、egg8 / bacon6 の稼働サービスを更新していません。
実機の完全動作や全経路の走行受入は未成立です。残っている条件は末尾に記載します。

## 修正した問題

| 領域 | 発生条件と問題 | 修正後の動作 |
| --- | --- | --- |
| gateway の占有 | 複数プロセスが同じUDP/UARTを使用できる | UDPを排他的にbindし、TTYの排他所有を要求。重複起動・無効ポート・UART不在は起動失敗 |
| UDPの送信元・順序 | 不正フレームが送信元ロックや連番を消費する | 検証成功後だけ送信元と連番を更新。期限切れの受信キューを拒否 |
| UDPバースト | E-stopやDISARMが後続ARMで上書きされる | 有効フレームを順番に処理し、停止出力を一度通すまで再始動しない |
| 停止入力 | MU3 PS / legacy PSや明示DISARMが条件によって無視される | 緊急停止をラッチし、明示DISARMを優先。解除時に旧走行指令を再使用しない |
| 未操作の位置軸 | 起動・停止・補完フレームがARM/GM目標0を送る | 軸ごとの明示操作または認可済みv4位置指令まで位置IDを省略。部分フレームでも既存目標を保持 |
| UART・MU3 | 切断、部分書込み、送信詰まりが無制限に待機・誤同期する | 非ブロッキングI/O、上限付き送信、再同期、厳密なhex/COBS検証、エラー時の停止・終了 |
| 終了と送信状態 | 終了時に停止フレーム不足、未送信指令を送信済みと表示 | SIGTERM等で停止フレームを送信。UARTドライバが完全に受理した直近フレームを通知 |
| Guardの入力 | NaN/Infや異常時刻がノードを落とす・状態を汚す | 即座にゼロ出力、無効データを拒否し再認可を要求。無効設定・退化した車輪幾何を起動時に拒否 |
| Guardの制限変更 | 減速・プロファイル変更直後に平滑化出力が新上限を超える | 最終出力を新しい速度・角速度・各輪予算へ再投影し、加速度制限も更新 |
| Guardの鮮度 | 小さい非ゼロ指令やRLの別メッセージが期限切れを隠す | 実際の非ゼロ指令を監視し、RL healthと倍率を独立に期限判定。橋渡し状態のheartbeatを送信 |
| プロファイル整合 | balancedのNav2角加速度2.0がGuard上限1.3を超える | 上流のNav2・smootherを1.2 rad/s²へ下げ、既存Guard内に収める |
| センサー鮮度 | 古い・未来・再送・不正フレームがhealthを更新する | 取得時刻と順序、frame、有限値・共分散・scanメタデータを検証してから採用 |
| scanの取得時刻 | 模擬rear scanをfrontと同じ古い時刻で配信、scan_timeを走査時間と誤解する | 各瞬間scanの取得時刻を使用。time_incrementが0ならscan_timeから架空のbeam時刻を作らない |
| LiDAR切替 | 非同期要求の競合で衝突監視の全入力を無効化する | 要求を直列化し、新入力の有効化を確認してから旧入力を無効化。生存入力がない場合も全解除しない |
| 自己位置推定 | 不正ICP/パラメータ/encoderから不正TFや姿勢を配信する | 有限値・共分散・有効範囲を検証、パラメータを一括検証。読み取り障害や時計逆行でencoder基準を再初期化 |
| 計測輪の較正・原点 | 退化した較正を受理、稼働中の原点resetが移動に見える | 観測可能性を検査。利用中・速度不明・非静止時のresetを拒否。詳細は別文書 |
| デバイス初期化 | LiDAR識別が緩い、DTR・counter/GPIOの例外後に資源が残る | デバイス識別を厳密化し、DTR設定順序、期限付き書込み、失敗時の資源解放を修正 |
| 目標の座標・入力 | 不正な保存名・frame・quaternionが現在の走行状態を変更する | 検証後だけ状態を変更。map座標と正規化した姿勢を使用し、無効経路を停止扱いにする |
| 目標切替の競合 | 旧目標の結果・到着・feedback・cancelが新目標を終了する | 目標時刻・request ID・action handleで現在の要求と照合。同じ名前・座標の再要求も区別 |
| Action取消・監視 | cancel失敗やresult取得失敗で二重の走行要求を出す | handleを保持し取消を再試行。停止確認前に新actionを送らない |
| 軌道計算の競合 | 計算中の速度変更後に旧プロファイルの軌道を採用する | lockとrevisionで計算結果を検証し、旧結果と事前計算出発を破棄 |
| Nav2起動 | action server準備前のlifecycle照会がbringupに干渉する | action server発見後にlifecycleを照会。モーター無効の実ROSデモで起動・目標到達を確認 |
| 最初の固定経路 | 初回要求時のBT構築・action探索で開始が約1.2秒遅れる | 既存の固定経路BTを起動時に準備。実ROSデモの要求→開始は12ms、目標到達も確認 |
| 模擬scanの処理負荷 | 全体接続時にexecutorの処理負荷でscan・姿勢が期限切れとなる | 制御用executorと単一scan workerに変更。計算要求を一つにまとめ、元の取得時刻と停止期限を維持 |
| 経路のCAD検査 | NaN clearance、無効grid・polygon・サンプリング間隔を通す | 不明・不正な幾何を通行許可に使用しない。polygonの向きに依存しない包含判定 |
| 回転・将来軌道 | 離散点間の接触やbody-frame旋回軌道を見落とす | 区間の移動量を使って連続的な余裕を保守的に証明。保持したbody twistは円弧として予測 |
| 遅れ補償 | 指令履歴以前の区間を、実測移動中でも速度0と仮定する | 履歴のない区間に現在の観測運動を使用 |
| RLのhold・health | hold期間が終点保護・無効入力検査を飛ばす | 毎回検証し、終点ではbaselineを優先。無効・古い姿勢でhealthを開かない |
| RLの学習・配備 | 学習の実速度と実行時の指令速度、正規化速度が異なる | 非空方策に一致した観測contextを要求。異なる学習contextのexport・昇格を拒否。実行時profileの鮮度・一致も要求 |
| GUI停止 | ボタン状態により停止時cancelが省略される・古い状態を稼働表示する | E-stopで必ずdisarmと目標cancel。解除時もdisarm。不正statusを拒否し古い状態を表示 |
| CAD配置 | 旧PCの絶対パスやcwd次第で起動できない | STLをROSパッケージに同梱、YAMLからの相対参照を解決し、hashを起動前に検証 |
| ビルド鮮度 | ソース変更後も古いinstall/binaryを再使用する | 設定・ソース・リンク先の指紋で再ビルド判定。ビルド中変更・循環リンクは合格印を付けない |
| 検証・証拠 | 無効な集計や過去設定の成功報告を現行版合格と扱う | 空集計・NaN・不十分な報告を失敗扱い。設定・モデルhashと制約を記録し、現行softwareと経路受入を分離 |

[計測輪resetの契約](MEASUREMENT_WHEEL_RESET.md)、[scan時刻の契約](SYNTHETIC_SCAN_TIMING.md)、
[gatewayの停止条件](../bacon_gateway/README.md)
も参照してください。既存UDP v3/v4およびCOBS triplet形式を維持しています。

`sprint_turn_everywhere` は `false` に変更しました。以前の全域制限解除には
保存経路の接触予測があり、その解除を常時有効にする根拠がありません。
4 m/sの指令上限は実速度・停止性能を保証しません。

## 検証と再現

実ROSの検証には Ubuntu 24.04 の `ros:jazzy-ros-base`、Nav2、tf2、pytest、
NumPy、PyQt5、ament-cmake-gtestを使用しました。ROS 4パッケージをビルドし、
GUI試験は `QT_QPA_PLATFORM=offscreen`、DDSは分離したdomainで実行しました。
github workflowにも同じsoftware検証を追加しました。

最終版で `scripts/verify_software.sh` は終了コード0でした。
**Python 1312件、BTのC++ 10件、gatewayのCTest 8件がすべて成功**しました。
gatewayは `-Werror`・ASan・UBSan構成でも8/8成功しました。
[検証時の構成・ソース指紋・結果](SYSTEM_AUDIT_SOFTWARE_20261007.json)を保存しています。

```bash
bash scripts/verify_software.sh
```

このコマンドはROSビルド、BTのC++試験、Python試験、構文検査、gateway試験を実行します。
gateway試験はloopback socket / PTYを使用し、実際の実行ファイルの停止・終了・
送信内容も検査します。Werror・AddressSanitizer・UndefinedBehaviorSanitizer構成でも
gateway試験を確認しました。

新しい一部のPython故障注入試験は実ソースからcallbackを取り出して、偽時計・
publisherで状態遷移を検証します。DDSスケジューリングを保証する試験ではありません。
ROS環境の全体試験とモーター無効デモも別途実行しています。

ビルド後、リポジトリ外の `/tmp` から起動するデモの例です。

```bash
source /opt/ros/jazzy/setup.bash
source /path/to/KOSMOS/ros2_ws/install/setup.bash
ROS_DOMAIN_ID=198 ros2 launch omni_autonomy_next system.launch.py \
  demo:=true motors:=false lidars:=false wheels:=false gui:=false rviz:=false \
  initial_pose_id:=1 goal_id:=0
```

これは合成scan・模擬運動による起動と目標到達の確認です。
モーター、LiDAR、計測輪の実機I/Oは含みません。

起動ログの `Inflation layer either not found ...` も調査しました。
使用したNav2 1.3.13の
[SmacPlanner2D](https://github.com/ros-navigation/navigation2/blob/1.3.13/nav2_smac_planner/src/smac_planner_2d.cpp#L115)
はradiusモードのfootprintにcost値0を渡し、
[collision checker](https://github.com/ros-navigation/navigation2/blob/1.3.13/nav2_smac_planner/src/collision_checker.cpp#L57)
がmode確認前にその値をERRORとして表示します。現行の外接半径0.588 mに対して
global inflation半径は0.85 mです。この表示を消すための半径変更は行っていません。
radiusモードの大域プランナーに加え、下流のCAD車体検査・衝突監視も必要です。

### MU3から走行・停止までの統合検証

`scripts/check_mu3_navigation_full_chain.py` を `NAV_TEST_SAFETY_ONLY=1` で実行し、
MU3の入力バイト→現行C++gateway core→実motor bridge→Nav2→tracker/Guardを
loopback UDPと合成センサー・模擬運動で接続しました。**終了コード0・FULL_CHAIN_PASS**です。
無効slot拒否、走行中の停止ボタン・radio断によるゼロ指令、停止後の自動再始動防止、
ボタンを保持したまま通信復旧した場合の再始動防止を確認しました。
全保存目的地の到着試験はこのモードに含みません。

最初の3回は自己位置・通信が期限切れになって走行開始できませんでした。
単独実行・記録負荷の除去・実測readiness待ちでも再現し、全体接続時のsimulator
dispatcher負荷を測定した上でworkerを修正しました。固定経路BTの起動時準備も含めた
4回目で成功しています。取得期限・watchdog・scan周期・サンプル数を緩めていません。
試験ハーネスは既存のobserverが記録するため、重複するrun recorderを無効化し、
固定秒数の起動待ちを期限付きの実測readiness待ちへ変更しました。

環境は5論理CPUが見えますが、上位cgroupのCPU予算は4CPU相当です。
CPU測定の時間窓が異なるため、修正前後の厳密な定常CPU削減率は主張していません。
[統合試験の結果・試行履歴・範囲](SYSTEM_AUDIT_INTEGRATION_20261007.json)を参照してください。
このgateway fixtureはmotor UARTを開きません。実機の停止能力の証明にはなりません。

### オフラインモデルの評価

変更前と変更後を同一seed `20261007` の49経路で比較し、別seed `20260808` の
98試行で空のRL方策とbaselineを比較しました。数値・設定・source hash・失敗経路は
以下のJSONに記録しています。

| 評価 | 到着 | 接触予測 | 時間切れ | 進捗停止 |
| --- | ---: | ---: | ---: | ---: |
| 変更前・seed 20261007 | 39/49 | 8 | 1 | 1 |
| 変更後・seed 20261007 | 42/49 | 4 | 2 | 1 |
| 変更後baseline・seed 20260808 | 89/98 | 5 | 1 | 3 |
| 変更後の空RL・seed 20260808 | 89/98 | 5 | 1 | 3 |

変更後には幾何・予測の修正と角加速度2.0→1.2 rad/s²の整合修正を含みます。
個別修正の効果を分離した比較ではありません。**全経路受入・接触ゼロの条件は失敗**です。

- [変更前49経路](SYSTEM_AUDIT_BASELINE_20261007.json)
- [変更後49経路](SYSTEM_AUDIT_MODEL_20261007.json)
- [RL比較98試行](SYSTEM_AUDIT_RL_20261007.json)
- [RL配備contextの制約](RL_DEPLOYMENT_CONTEXT_20261007.md)

モデルはMPPI形式の軽量な代用モデルです。本番の軌道追従器・Nav2を再生するものでは
なく、応答遅れ・加減速・CAD接触にはモデル上の仮定があります。モデルの接触は
実機接触の観測ではありません。再現するには次を実行します。

```bash
python3 -m simulation.run_campaign --episodes 49 --random-episodes 0 \
  --seed 20261007 --output /tmp/kosmos-current-model.json
python3 -m simulation.reinforcement_learning evaluate \
  --deployed-policy ros2_ws/src/omni_autonomy_next/config/rl_policy.yaml \
  --episodes 98 --random-episodes 0 --seed 20260808 \
  --output /tmp/kosmos-current-rl.json
```

現行 `rl_policy.yaml` の `overrides: {}` は学習した動作を適用しません。
baselineと同じ出力であることは確認対象ですが、baseline自身の接触予測を解消する
証拠にはなりません。過去の1200/1200成功は速度・設定・モデルが異なり、現行受入の
証拠に使用できません。[保存結果の位置付け](../simulation/results/README.md)を参照。

全検証の `python3 run.py verify` はsoftware試験の後に、より厳しいモデル受入も
実行します。現行モデルに失敗経路がある状態で、この全検証を成功とは扱いません。

## 残る受入条件

1. **全経路受入は未達。** 現行モデルの失敗経路について、実際の追従器を使う
   モーター無効の全経路ROS検証と、測定した実機応答を使う再評価が必要です。
   49/98試行だけで全場面を検証したことにはなりません。
2. **位置制御軸の物理的停止。** ARM/GM目標保持は新たな原点移動を防ぎますが、
   既に目標へ移動中の軸を停止した証拠にはなりません。MCUのabort/disableまたは
   実測位置の取得、初回指令前の原点合わせ・較正が必要です。
3. **MCU側の独立watchdog。** hostのkill -9、電源断、UARTケーブル断、host停止は
   hostから停止フレームを送れません。MCUで受信期限切れ時の停止を保証する必要があります。
   そのfirmwareはこのリポジトリに含まれていません。
4. **実機限界の測定。** 速度・角速度・荷重・横滑り・制動距離・遅れ・LiDAR有効距離・
   計測輪の較正を現物で確認し、負荷・全フィールド条件を含めて
   [ACCEPTANCE.md](ACCEPTANCE.md)の試験を通す必要があります。

今回の変更を反映することと、上記の実機受入を完了することは別の作業です。
未検証の実機性能を補うための速度上限引上げ、失敗したモデルの合格扱い、
稼働サービスの自動更新は行っていません。
