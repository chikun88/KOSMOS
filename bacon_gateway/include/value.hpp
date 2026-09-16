#pragma once

#include <cstdint>
#include <termios.h> // ボーレートのマクロ (B19200 / B115200) と speed_t

#include "packet.hpp"

// =====================================================================
//  value.hpp
//  このプロジェクトの「調整可能な設定値」をすべてここに集約する。
//  ・モーターの出力値／倍率
//  ・コントローラーのボタン割り当て
//  ・ロボマス開発ボードへ送るコマンドID
//  ・シリアル/UDPなどのシステム・通信設定
//  コード側（move.cpp / main.cpp / uart.cpp / udp.cpp）に生の数値を
//  書かず、必ずここの定数を参照すること。
// =====================================================================
namespace value
{
    // =================================================================
    //  1. モーター出力・倍率
    // =================================================================

    // --- OMNI (m1〜m4) ---
    constexpr int omni_speed = 4000; // 起動時の最高速度（レンジ初期値）
    constexpr int omni_max   = 8000; // 最高速度の上限
    constexpr int omni_min   = 2000; // 最高速度の下限
    constexpr int omni_range = 2000; // ボタン1回あたりの最高速度の増減幅

    constexpr double omni_deadzone = 0.2; // スティックのデッドゾーン
    constexpr double omni_ratio    = 1.0; // 旋回の速度に対する速度比率
    constexpr double stick_norm    = 128.0; // int8スティック値(-127〜127)を-1〜1へ正規化する除数

    // --- v3自動走行（物理速度指令）の較正 ---
    // Jetsonからのv3コマンド(vx/vy[mm/s], ω[mrad/s])を各輪値へ直接変換する。
    // デッドゾーンなし・手動の速度レンジボタン(ue/shita)非依存の固定スケール。
    // 並進ゲインは従来のフルスティック相当（0.55 m/s → 約3961）を維持する。
    // 45°オムニの旋回項はCAD上の車輪中心半幅 x=y=0.333072 m から
    // (x+y)ω とする。これによりJetson側のwheel budgetと同じ物理モデルになる。
    constexpr double auto_units_per_mps = 7202.0; // 並進 1 m/s あたりの各輪値
    constexpr double auto_rotation_lever_arm_m = 0.666144; // CAD x+y [m]
    constexpr double auto_units_per_radps =
        auto_units_per_mps * auto_rotation_lever_arm_m; // 4797.569088 / (rad/s)
    // 横方向の符号は 2026-08-07 に実機で「左右が行くべき方向と逆」を観測して
    // +1.0（反転なし）に確定した。よって ux = +vy が実機の正しい契約である。
    //
    // -1.0 だった根拠は 2026-08-05 の「左右にふらふらしながら直進し、終端で
    // 逆方向へ寄せる」挙動だが、あれは横符号が逆のときに必ず出る形そのもの
    // である。指令 (vx, vy) が (vx, -vy) で実行されると位置のP制御は
    // -K diag(1,-1) e となり、横方向だけ発散する。MPPI 時代は PathAlignCritic
    // の重み24が毎周期引き戻すので発散せず蛇行に見え、終端では PathAlign が
    // 閾値を抜けて Goal 系が支配するため最終寄せだけが逆に出た。つまりあの
    // 観測は -1.0 の根拠ではなく、-1.0 が誤りであることの証拠だった。
    // 同じ理屈で、追従器に替えたあとも「小刻みに左右へふら付く」が残っていた。
    //
    // 横は反射（鏡映）であって回転ではないので、症状の形からは回転誤差
    // （＝一定半径の円）と区別しにくい。判定は check_drive_directions.py の
    // 実測で行うこと。
    //
    // 注意: main.cpp の legacy/v2 自動経路は lx を素通しするため ux = +vy で
    // あり、これで v3 経路と一致した。
    // 変更する場合は tests/test_omni_velocity.cpp の期待値も、
    // ros2_ws の robomas_uart.py の AUTO_LATERAL_SIGN も同時に直すこと
    // （v4_uart 経路では egg8 側のこの定数が実際に使われる）。
    constexpr double auto_lateral_sign = 1.0;  // wire vy(左+) → ミキサーの横入力(反転なし)
    constexpr double auto_forward_sign = 1.0;  // wire vx(前+) → 前後ミキシング符号
    // 旋回符号は 2026-08-06 に手動運転の実機挙動から導出して確定した。
    //
    // 手動経路 checker_omni() は同じミキサ式を使い、turn = rx（右スティック
    // 横）をそのまま旋回入力に入れる。その手動運転が正常であることを確認済みで、
    // 右スティックを右へ倒すと機体は上から見て時計回りに回る。つまり
    //   ミキサの旋回入力が正 → 機体は時計回り（ROSの ω は負）
    // が実機で確定した契約である。ROSの +wz は反時計回りなので、v3経路は
    // 旋回入力を負にしなければならない。よって -1.0。
    //
    // 同じ導出で前後・横も確認した:
    //   ミキサの並進入力(uy)が正 → 前進 → auto_forward_sign = +1.0（現状で正）
    // 横だけは 2026-08-07 の実測で覆った。ミキサの横入力(ux)が正 → 左であり、
    // ROSの +vy(左)はそのまま渡してよい（auto_lateral_sign = +1.0）。上の
    // 導出は m1..m4 がどの物理コーナーに配線されているかを仮定していたが、
    // それはどこにも記録されていない量である。
    // 矛盾のない実配置の一例: m1=後右@135°, m2=前右@225°, m3=前左@-45°,
    // m4=後左@45°（ローラ角はすべて接線方向）。以前の表と比べると m1↔m2 と
    // m3↔m4 が入れ替わっている。配線がそうなっているという意味である。
    //
    // 2026-08-05 に +1.0 へ変えたのは「一定半径の円」を旋回暗走と解釈した
    // ためだが、その円は車輪バジェット配分が並進とヨーを別倍率で縮小して
    // 曲率を45%ずらしていたこと（2026-08-06に修正）と指令経路の440 msの
    // 遅れで説明がつく。症状の形から符号を推定するのはやめ、静的な契約の
    // 実測で決めること。
    // 変更する場合は tests/test_omni_velocity.cpp の期待値も同時に直すこと。
    constexpr double auto_turn_sign    = -1.0; // wire ω(CCW+) → ミキサの旋回入力(反転あり)
    constexpr int    auto_wheel_limit  = 10000; // 各輪の上限。超過時は全輪比例縮小
    // ダッシュボード等でのint8等価表示に使うフルスケール（v2の較正と同じ）
    constexpr double auto_display_full_mmps  = 550.0;  // 0.55 m/s
    constexpr double auto_display_full_mradps = 1200.0; // 1.2 rad/s

    // --- ARM (m5) : シン・位相制御 ---
    // ボタンを押した瞬間に arm_press_revolutions 周分だけ位相を進める。
    // 1回あたりの送信値 = arm_scale（1周あたりの位相値）× arm_press_revolutions（回転数）
    constexpr int16_t arm_scale             = 36; // M2006/M3508 のシン・位相 1周あたりの位相値
    constexpr int16_t arm_press_revolutions = 4;  // ボタン1回で進める回転数
    constexpr int16_t arm_phase_step        = arm_scale * arm_press_revolutions; // = 144

    // --- UPDOWN (m6) ---
    constexpr int16_t updown_speed = 2000; // 昇降機構の速度

    // --- COLLECT (m7) ---
    constexpr int16_t collect_speed = 2000; // 回収機構の速度

    // --- GM (gm) : シン・位相制御（角度制御） ---
    constexpr double gm_angle_max    = 1000.0; // GM角度の上限
    constexpr double gm_angle_min    = 0.0;    // GM角度の下限
    constexpr double gm_angle_range  = 0.36;   // 1loopあたりの角度変化量
    constexpr double gm_reload_angle = 0.0;    // リロード時のGM角度（※実際の装填角度に要設定）

    // --- GPIO (射出電磁弁など) の出力ビット ---
    // 下位ビットから GPIO4, GPIO5, ... の順（mawarudokusute コマンド254）
    constexpr uint16_t gpio_on  = 0x0001; // GPIO4 = HIGH
    constexpr uint16_t gpio_off = 0x0000; // すべて LOW

    // =================================================================
    //  2. コントローラー配置
    //  （各動作に割り当てるボタンをここで変更できる）
    // =================================================================
    enum class button
    {
        none,                        // 割り当てなし
        batsu, maru, sankaku, shikaku,
        ue, shita, hidari, migi,     // 十字キー
        l1, r1, l2, r2,
        create, option, l3, r3, ps
    };

    // --- OMNI (m1〜m4) ---
    constexpr button omni_speed_up   = button::ue;    // 最高速度アップ
    constexpr button omni_speed_down = button::shita; // 最高速度ダウン

    // --- ARM (m5) ---
    constexpr button arm_up   = button::r2; // R2押下でアーム位相 +arm_phase_step（4回転）
    constexpr button arm_down = button::l1; // L1押下で逆回転（-arm_phase_step）

    // --- GM (gm) ---
    constexpr button gm_up     = button::sankaku; // △：GM角度＋
    constexpr button gm_down   = button::batsu;   // ×：GM角度−
    constexpr button gm_reload = button::r2;      // R2押下中はリロード角へ移動、離すと元角度へ戻る

    // --- UPDOWN (m6) --- （押されている間ON）
    constexpr button updown_on      = button::r2; // R2で昇降ON（リロード動作）
    constexpr button updown_reverse = button::l1; // L1で逆方向

    // --- COLLECT (m7) ---
    constexpr button collect_on_a = button::maru;    // ○：正転
    constexpr button collect_on_b = button::shikaku; // □：逆転

    // --- GPIO (gpio) ---
    constexpr button gpio_toggle = button::r1; // R1：射出GPIO ON（押下中）

    // --- 自動制御 (Jetson連携) ---
    // このボタンを押している間、Jetsonが生存していれば自動制御ONになる
    constexpr button auto_control_on = button::l2;

    // =================================================================
    //  3. ロボマス開発ボードへのコマンドID (mawarudokusute 仕様)
    //  ※C610/620 の ID と GM6020 の ID の重複に注意（仕様書「大前提」参照）
    // =================================================================
    constexpr uint8_t cmd_omni[4] = {0, 1, 2, 3}; // 速度制御 C620 ID1〜4
    constexpr uint8_t cmd_arm     = 44;  // シン・位相制御 M2006/M3508 ID5
    constexpr uint8_t cmd_gm      = 61;  // シン・位相制御 GM6020 ID6
    constexpr uint8_t cmd_updown  = 6;   // 速度制御 M2006 ID7
    constexpr uint8_t cmd_collect = 7;   // 速度制御 M2006 ID8
    constexpr uint8_t cmd_gpio    = 254; // GPIO ON/OFF

    // =================================================================
    //  4. システム・通信設定
    // =================================================================
    namespace sys
    {
        // シリアルポート
        constexpr const char* mu3_device   = "/dev/ttyUSB0"; // MU-3受信機
        constexpr speed_t     mu3_baud     = B19200;
        constexpr const char* motor_device = "/dev/serial0"; // ロボマス開発ボード
        constexpr speed_t     motor_baud   = B115200;

        // Jetson受信用UDP（有線LAN）
        constexpr uint16_t jetson_port = 8888;

        // タイミング（メインループは loop_period_ms 間隔）
        constexpr int loop_period_ms    = 5;   // メインループ周期[ms]（=200Hz）
        constexpr int timeout_threshold = 20;  // MU3受信タイムアウト（loop_period_ms×20=100ms）
        constexpr int dashboard_interval = 20; // ダッシュボード更新間隔（20ループ=100ms/10Hz）
        // Jetson command watchdog.  Both hosts run non-real-time Linux, so a
        // 30/60 ms threshold falsely stopped AUTO during ordinary scheduler
        // stalls even though the wired link had zero packet loss.  Warn at
        // 100 ms, perform a controlled stop at 250 ms, and latch a hard fault
        // at 1000 ms.  A returned link still cannot restart motion by itself:
        // the disarm packet -> auto-request edge handshake remains mandatory.
        constexpr int link_degraded_ms = 100;
        constexpr int controlled_stop_ms = 250;
        constexpr int fault_timeout_ms = 1000;
        constexpr int alive_timeout_ms = controlled_stop_ms;
        // Dev Boardへ送る速度指令単位/秒。実機の制動距離を測定後に確定する
        // 暫定値であり、上げる前に浮上試験と床上停止距離試験を行うこと。
        constexpr double controlled_stop_units_per_sec = 16000.0;
        constexpr int warning_interval_ms = 250; // 警告表示の最短間隔
        constexpr int jetson_status_interval_ms = 1000; // Jetson接続状態の通常表示間隔

        // Jetson連携プロトコル（udp.cpp / v2＋テレメトリ）
        constexpr int telemetry_interval_loops = 2;    // テレメトリ返信間隔（ループ数。2=100Hz）
        constexpr int source_lock_timeout_ms   = 1000; // コマンド送信元ロックの解除時間[ms]
        constexpr int seq_resync_silence_ms    = 1000; // これ以上無通信ならv2シーケンスを再同期
    }

    // value::button で指定されたボタンの押下状態を返す
    inline bool read_button(const Controller_Packet & packet, button b)
    {
        switch (b)
        {
            case button::batsu:   return packet.batsu_state;
            case button::maru:    return packet.maru_state;
            case button::sankaku: return packet.sankaku_state;
            case button::shikaku: return packet.shikaku_state;
            case button::ue:      return packet.ue_state;
            case button::shita:   return packet.shita_state;
            case button::hidari:  return packet.hidari_state;
            case button::migi:    return packet.migi_state;
            case button::l1:      return packet.l1_state;
            case button::r1:      return packet.r1_state;
            case button::l2:      return packet.l2_state;
            case button::r2:      return packet.r2_state;
            case button::create:  return packet.create_state;
            case button::option:  return packet.option_state;
            case button::l3:      return packet.l3_state;
            case button::r3:      return packet.r3_state;
            case button::ps:      return packet.ps_state;
            case button::none:
            default:              return false;
        }
    }
}
