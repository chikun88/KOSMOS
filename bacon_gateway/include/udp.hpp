#pragma once

#include <cstdint>
#include <cstddef>
#include <chrono>
#include <netinet/in.h>
#include <sys/socket.h>

#include "packet.hpp"
#include "drive_safety.hpp"
#include "passthrough.hpp"
#include "value.hpp"
#include "mu3_navigation.hpp"

// =====================================================================
//  Jetson との有線LAN(UDP)通信
//
//  受信（Jetson→Pi）は4形式を判別する。7バイトのlegacyだけ長さで判定し、
//  残りは先頭のmagic(0xB6)とversionバイトで判定する:
//   ・legacy : 7バイト MU3コントローラ互換Raw（従来形式・後方互換）
//   ・v2     : 20バイト int8スティック値＋seq/CRC（下記レイアウト）
//   ・v3     : 24バイト 物理速度指令 mm/s・mrad/s（下記レイアウト）
//              自動走行時はデッドゾーンなし・手動速度レンジ非依存の
//              固定較正で各輪値へ直接変換される。
//   ・v4     : 可変長 UARTパススルー（現行の自動走行経路）。
//              Jetsonが開発ボードへそのまま出せるCOBS済みフレームを
//              組んで送り、Piは中身を書き換えずUARTへ流す。
//
//  v2 コマンド (リトルエンディアン, 計20バイト)
//   [0]      magic   0xB6
//   [1]      version 0x02
//   [2]      flags   bit0:自動要求(L2相当) bit1:非常停止
//   [3..5]   buttons legacy payload[4..6] と同じビット配置
//   [6..9]   lx ly rx ry (int8, デコード後の符号系をそのまま送る)
//   [10..11] seq     uint16 (古い/重複パケットの破棄用)
//   [12..15] t_tx_us uint32 (送信側µs下位32bit, RTT計測用エコー)
//   [16..17] 予約 0
//   [18..19] CRC-16/CCITT-FALSE ([0..17]まで)
//
//  v3 コマンド (リトルエンディアン, 計24バイト)
//   [0]      magic   0xB6
//   [1]      version 0x04
//   [2]      flags   bit0:自動要求 bit1:非常停止
//   [3..5]   buttons legacyと同配置（自動中の機構操作用）
//   [6..7]   vx int16 [mm/s]   機体前+ (ROS body frame)
//   [8..9]   vy int16 [mm/s]   機体左+
//   [10..11] w  int16 [mrad/s] CCW+
//   [12..13] seq u16   [14..17] t_tx_us u32   [18..21] 予約0
//   [22..23] CRC-16/CCITT-FALSE ([0..21]まで)
//
//  v4 コマンド (UARTパススルー, リトルエンディアン, 計 22+uart_len バイト)
//   [0]      magic   0xB6
//   [1]      version 0x06
//   [2]      flags   bit0:自動要求 bit1:非常停止
//   [3..5]   buttons legacyと同配置（将来Jetsonから機構を操作する用。現状0）
//   [6]      uart_len 続くUARTバイト列の長さ [5..64]
//   [7]      予約 0
//   [8..9]   seq u16   [10..13] t_tx_us u32
//   [14..15] vx int16 [mm/s]  ─┐ 参考値。制御には一切使わず、テレメトリの
//   [16..17] vy int16 [mm/s]   │ applied_velocity へそのままエコーして
//   [18..19] w  int16 [mrad/s] ─┘ 既存の診断ツールを動かし続けるためだけ。
//   [20..20+uart_len-1] UARTフレーム（COBS済み、末尾の区切り0x00を含む）
//   [末尾2]  CRC-16/CCITT-FALSE ([0]からUARTバイト列の末尾まで)
//
//  UARTバイト列の中身は [コマンドID, 値上位, 値下位] の3バイト組をCOBSで
//  包んだもの（uart.cpp / passthrough.hpp 参照）。Piはこれを解析するが、
//  中継するのは受信したバイト列そのもので、書き換えは行わない。
//
//  テレメトリ (Pi→Jetson, telemetry_interval_loopsごとに返信)
//   拡張形 計48バイト (version 0x05) を常用。基本形32バイト(0x03)は
//   旧Jetsonブリッジ互換用の定義（Jetson側は両方デコード可能）
//   [0]      magic   0xB6
//   [1]      version 0x03 / 0x05
//   [2]      flags   bit0:自動モード中 bit1:MU3生存 bit2:Jetson生存
//                    bit3:UART開 bit4:非常停止中
//   [3]      flags2  bit0:直近コマンドがv2 bit1:直近コマンドがv3
//                    bit2:リンク劣化 bit3:減速停止 bit4:フォルト
//                    bit5:再ARM待ち bit6:直近コマンドがv4(パススルー)
//   [4..5]   pi_seq uint16
//   [6..7]   last_cmd_seq uint16 (v2/v3受信時のみ)
//   [8..11]  last_cmd_t_tx_us uint32 (t_tx_usエコー)
//   [12..15] hold_us uint32 (コマンド到着→この送信までのPi内滞留時間)
//   [16..19] cmd_count uint32 (受理コマンド累計)
//   [20..21] crc_err uint16   [22..23] stale_drop uint16
//   [24..27] 適用中の lx ly rx ry (int8等価表示値)
//   [28]     直近1秒の受信レート[Hz] (255上限)
//   [29]     予約 0
//   --- 基本形はここでCRC: [30..31] ---
//   [30..31] 適用 vx int16[mm/s]  [32..33] vy  [34..35] w int16[mrad/s]
//   [36..43] 各輪指令値 m1 m2 m3 m4 (int16, UARTへ送った値)
//   [44..45] 予約 0
//   [46..47] CRC-16/CCITT-FALSE ([0..45]まで)
//
//  Jetson側は t_tx_us のエコーと hold_us から、クロック同期なしで
//  純粋な往復ネットワーク遅延を算出できる。
// =====================================================================
class UDP
{
    public:
        static constexpr std::size_t payload_size       = 7;  // legacy Raw
        static constexpr std::size_t v2_command_size    = 20;
        static constexpr std::size_t v3_command_size    = 24;
        static constexpr std::size_t telemetry_size     = 32;
        static constexpr std::size_t telemetry_ext_size = 48;
        static constexpr uint8_t protocol_magic         = 0xB6;
        static constexpr uint8_t v2_command_version     = 0x02;
        static constexpr uint8_t telemetry_version      = 0x03;
        static constexpr uint8_t v3_command_version     = 0x04;
        static constexpr uint8_t telemetry_ext_version  = 0x05;
        static constexpr uint8_t v4_command_version     = 0x06;

        // v4: 固定ヘッダ20バイト + UARTバイト列 + CRC2バイト
        static constexpr std::size_t v4_header_size     = 20;
        static constexpr std::size_t v4_min_uart_bytes  = 5;  // 1コマンドのCOBS
        static constexpr std::size_t v4_max_uart_bytes  =
            UartCommandFrame::max_encoded;
        static constexpr std::size_t v4_max_command_size =
            v4_header_size + v4_max_uart_bytes + 2;

        UDP(uint16_t port);
        ~UDP();
        UDP(const UDP&) = delete;
        UDP& operator=(const UDP&) = delete;

        void update(const Controller_Packet& ctrl);
        void request_estop() { estop_pending_output = true; link_safety.update(false, false, true, 0); auto_mode = false; }
        // A reset cannot hide an emergency event before a stop frame is queued.
        void acknowledge_estop_output() { estop_pending_output = false; }
        Mu3Navigation remote_navigation;

        // 現ループの適用値と機体状態をJetsonへ返信する。
        // 毎ループ呼ぶこと（内部で value::sys::telemetry_interval_loops に間引く）。
        // motors は実際にUARTへ送る各輪値。applied_v* はv3物理経路で
        // 適用した機体速度（比例縮小後）。物理経路でないループでは0を渡す。
        void send_telemetry(const Controller_Packet& applied, bool mu3_alive, bool uart_open,
                            const struct packet& motors,
                            double applied_vx_mps, double applied_vy_mps,
                            double applied_w_radps);

        // UDPソケットの初期化に成功しているか
        bool is_ready() const { return is_initialized; }

        // 受信済みで、motion watchdog未満の正常コマンドリンクならtrue
        bool jetson_alive() const { return is_alive; }

        // 有効な自動要求とDISARM→ARMハンドシェイクが揃っているとき自動
        bool is_auto_mode() const { return auto_mode && !estop_pending_output; }

        // E-stop/通信停止はラッチされる。通信復旧だけでは解除されない。
        bool estop_active() const { return link_safety.estop_active() || estop_pending_output; }
        bool safety_stop_active() const { return link_safety.stop_required() || estop_pending_output; }
        bool fault_latched() const { return link_safety.fault_latched(); }
        bool rearm_required() const { return link_safety.rearm_required() || estop_pending_output; }
        bool link_quality_degraded() const { return link_safety.quality_degraded(); }
        const char* safety_state_name() const { return estop_pending_output ? "ESTOP" : link_safety.state_name(); }

        // 受信したJetsonパケット
        const Jetson_Packet& packet() const { return jetson_packet; }

        // 受信したJetson生パケット（v2/v3受信時はlegacy相当へ再構成した7バイト）
        const uint8_t* raw_payload() const { return last_payload; }

        // 正常に受信したJetsonパケット数
        uint64_t received_count() const { return recv_count; }

        // 直近に受理したコマンドがv2形式か
        bool last_command_was_v2() const { return last_cmd_v2; }

        // 直近に受理したコマンドがv3（物理速度指令）か
        bool velocity_command_active() const { return last_cmd_v3; }

        // 直近に受理したコマンドがv4（UARTパススルー）か
        bool uart_passthrough_active() const { return last_cmd_v4; }

        // v4で受け取ったCOBS済みUARTバイト列。開発ボードへはこれを
        // 1バイトも書き換えずに出す。
        const uint8_t* passthrough_bytes() const { return v4_uart; }
        std::size_t passthrough_size() const { return v4_uart_len; }

        // v4フレームを解析した結果（どのコマンドIDが入っているか）。
        // 中継の可否ではなく、補完フレームと安全停止のために使う。
        const UartCommandFrame& passthrough_commands() const
        {
            return v4_commands;
        }

        // v3コマンドの機体速度（wire座標系: 前+/左+/CCW+）。
        // v4では制御に使われない参考値がそのまま入る（テレメトリ用）。
        double command_vx_mps() const { return v3_vx_mmps / 1000.0; }
        double command_vy_mps() const { return v3_vy_mmps / 1000.0; }
        double command_w_radps() const { return v3_w_mradps / 1000.0; }

        // 表示用プロトコル名
        const char* protocol_name() const
        {
            if (last_cmd_v4) return "v4";
            if (last_cmd_v3) return "v3";
            if (last_cmd_v2) return "v2";
            return "legacy";
        }

        // バインドしたUDPポート（MU3_JETSON_PORT環境変数で変更可能）
        uint16_t port() const { return bound_port; }

        // v2のCRC/ヘッダ不一致数・シーケンス落ちで破棄した数
        uint64_t crc_error_count() const { return crc_err_count; }
        uint64_t stale_drop_count() const { return stale_drops; }

        // 最後にJetsonから受信してからの経過時間[ms]
        int last_receive_age_ms() const;

    private:
        void handle_legacy(const uint8_t* data);
        bool handle_v2(const uint8_t* data);
        bool handle_v3(const uint8_t* data);
        bool handle_v4(const uint8_t* data, std::size_t length);
        // ダッシュボード用に、機体速度からlegacy相当の7バイトを再構成する
        void store_display_payload(const uint8_t buttons[3]);
        // v2/v3/v4共通: CRC検証とシーケンス受理判定（受理時にlast_seq更新）
        bool accept_command_frame(const uint8_t* data, std::size_t crc_offset,
                                  std::size_t seq_offset, uint16_t& seq_out);
        bool accept_source(const struct sockaddr_in& src,
                           const std::chrono::steady_clock::time_point& now);

        int fd;
        bool is_initialized;
        uint16_t bound_port;

        Jetson_Packet jetson_packet;
        uint8_t last_payload[payload_size];
        uint64_t recv_count;
        std::chrono::steady_clock::time_point last_recv_time;

        bool is_alive;
        bool auto_mode;
        bool estop_pending_output;

        // --- v2/v3/v4コマンド共通 ---
        bool last_cmd_v2;
        bool last_cmd_v3;
        bool last_cmd_v4;
        bool v2_auto_request;
        bool last_estop_request;
        bool has_seq;
        uint16_t last_seq;
        uint16_t last_cmd_seq;
        uint32_t last_cmd_t_tx_us;
        int64_t last_cmd_arrival_ns; // CLOCK_REALTIME。hold_us算出用
        uint64_t crc_err_count;
        uint64_t stale_drops;
        uint64_t size_err_count;
        uint64_t foreign_drop_count;

        // --- v3物理速度指令（v4では参考値のエコーとして同じ変数を使う） ---
        int16_t v3_vx_mmps;
        int16_t v3_vy_mmps;
        int16_t v3_w_mradps;

        // --- v4 UARTパススルー ---
        uint8_t v4_uart[v4_max_uart_bytes];
        std::size_t v4_uart_len;
        UartCommandFrame v4_commands;

        // --- 送信元ロック（最初に受信した相手を優先し、混信を防ぐ） ---
        struct sockaddr_in source_addr;
        bool source_locked;

        // --- テレメトリ ---
        uint16_t pi_seq;
        uint64_t telemetry_tick;
        std::chrono::steady_clock::time_point rate_window_start;
        uint32_t rate_window_count;
        uint8_t last_rate_hz;

        std::chrono::steady_clock::time_point last_size_warn;

        DriveLinkSafety link_safety;

        static constexpr std::chrono::milliseconds ALIVE_TIMEOUT{value::sys::alive_timeout_ms};
};
