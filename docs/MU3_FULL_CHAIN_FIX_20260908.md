# MU-3地点移動の修正・検証（2026-09-08）

## 原因と修正

1. Jetsonのmotor_udp_bridgeが明示的なDISARM直後にも待機時自動再ARMを行い、MU-3地点制御の停止確認がDISARM_TIMEOUTになっていた。DISARMを保持し、安全条件を満たす新しいARM要求でのみ解除するよう修正。
2. 到達成功後の不要なキャンセルで、保持される到達結果がCANCELEDに上書きされていた。SUCCEEDEDではキャンセルせずDISARMのみ行う。
3. Jetson GUIにMU-3指示状態、保存地点名、経路点数と受信後経過時間、開始できない理由を表示。経路表示は最後に受信した経路であり、新しい指示の経路と偽らない。

## 実施した検証

- Python既存・回帰テスト465件成功。
- bacon6でC++テスト5件成功（実UDPの再ARM回帰テストを含む）。
- 隔離ROSドメイン94・ループバックUDPによる統合試験。MU-3の7バイト指示→実C++ gateway→実motor_udp_bridge→MU-3地点ノード→実Nav2→実trajectory_tracker→各輪UARTコマンドの生成を確認。
- A: 55点、17.00秒、到達成功。
- BAKETU2: 163点、14.15秒、到達成功。
- BAKETU3: 207点、19.76秒、到達成功。
- hokyuu: 104点、11.83秒、到達成功。
- hokyuu A: 449点、26.13秒、到達成功。
- 走行中の停止ボタン、走行中の無線喪失でDISARM・各輪ゼロ。待機再ARMなし。古いボタン指示を保持した通信復帰でも再発進なし。配備先のインストール済みROSパッケージで再確認。
- GUIの表示・理由表示を隔離ドメイン95で確認。navigation-gui.pngは模擬状態による表示確認画像。

## 適用先と制約

Jetson: /home/egg8/Desktop/omni_autonomy_next

bacon6: /home/bacon6/omni_autonomy_next

双方にbefore-navigation-full-fix-20260908の隣接バックアップを作成。JetsonのROSパッケージを再ビルド。

今回Android APKの変更は不要。MU-3は既存の送信経路を使用し、Androidへ経路や到達結果を返す通信は追加していない。詳細結果はJetson画面で確認する。

統合試験は合成センサーと模擬移動体を使用し、実モーターUARTは開かない。移動体は/cmd_vel_safeで進み、実UDP下流の各輪値も別途検証している。実機の無線伝送・車輪動作・実地到達はこの試験だけでは証明できない。実機走行による全機能正常の断定はしていない。

## 統合試験の再実行（Jetson）

```bash
cd ~/Desktop/omni_autonomy_next/bacon_gateway
g++ -std=c++17 -Iinclude tests/navigation_loopback_gateway.cpp src/udp.cpp src/drive_safety.cpp src/passthrough.cpp src/cobs.cpp -o /tmp/navigation_loopback_gateway
cd ..
source ros2_ws/install/setup.bash
python3 scripts/check_mu3_navigation_full_chain.py
```

記録: full-chain-five-poses.json、full-chain-safety.json。経路点数は各移動中に受信した最大点数。
