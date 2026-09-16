#include <iostream>
#include <thread>
#include <chrono>
#include <cstring>
#include <algorithm>
#include <array>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <memory>
#include <string>
#include <termios.h>
#include <unistd.h>
#include <vector>

#include "move.hpp"
#include "drive_safety.hpp"
#include "MU3.hpp"
#include "packet.hpp"
#include "passthrough.hpp"
#include "uart.hpp"
#include "udp.hpp"
#include "value.hpp"

namespace
{
    volatile std::sig_atomic_t shutdown_requested = 0;

    void requestShutdown(int) noexcept
    {
        // Async-signal-safe: the handler only stores a sig_atomic_t flag.
        shutdown_requested = 1;
    }

    bool installShutdownHandlers()
    {
        return std::signal(SIGINT, requestShutdown) != SIG_ERR &&
               std::signal(SIGTERM, requestShutdown) != SIG_ERR;
    }

    bool sendShutdownStopFrames(UART& uart, const packet& last_command)
    {
        // Stop every velocity-controlled motor, but keep the last position
        // targets for ARM (m5) and GM.  Sending an all-zero packet here could
        // command those mechanisms to their zero/home positions during a
        // process shutdown.  GPIO is also held to avoid an unrequested
        // pneumatic/mechanism transition; the Development Board still needs
        // an independent watchdog for cable/power/process failures.
        packet stop_packet = last_command;
        stop_packet.m1 = 0;
        stop_packet.m2 = 0;
        stop_packet.m3 = 0;
        stop_packet.m4 = 0;
        stop_packet.m6 = 0;
        stop_packet.m7 = 0;

        // One leading frame also re-synchronizes a receiver if a prior UART
        // write was partial; the following three repeat the complete stop.
        constexpr int stop_frames = 4;
        constexpr int max_attempts = 12;
        const auto retry_interval =
            std::chrono::milliseconds(value::sys::loop_period_ms);

        int sent = 0;
        for (int attempt = 0; attempt < max_attempts && sent < stop_frames;
             ++attempt)
        {
            if (uart.uart_send(stop_packet))
            {
                ++sent;
            }
            else
            {
                std::cerr << "[SHUTDOWN] stop UART frame attempt failed: "
                          << (attempt + 1) << std::endl;
            }

            if (sent < stop_frames)
            {
                std::this_thread::sleep_for(retry_interval);
            }
        }

        const bool flushed = uart.flush_output();
        const bool success = sent == stop_frames && flushed;
        std::cerr << "[SHUTDOWN] stop UART frames=" << sent << "/"
                  << stop_frames << " flush=" << (flushed ? "OK" : "FAILED")
                  << " result=" << (success ? "OK" : "FAILED") << std::endl;
        return success;
    }
}

// デバッグ用ダッシュボード関数
// --dashboard または RX_DASHBOARD=1 のときだけ表示する。
bool argumentEnabled(int argc, char* argv[], const char* name)
{
    for (int i = 1; i < argc; ++i)
    {
        if (std::strcmp(argv[i], name) == 0)
        {
            return true;
        }
    }

    return false;
}

bool dashboardEnabled(int argc, char* argv[])
{
    const char* env = std::getenv("RX_DASHBOARD");
    if (env != nullptr && std::strcmp(env, "0") != 0 && std::strcmp(env, "false") != 0)
    {
        return true;
    }

    return argumentEnabled(argc, argv, "--dashboard") || argumentEnabled(argc, argv, "--debug");
}

bool jetsonOnlyEnabled(int argc, char* argv[])
{
    return argumentEnabled(argc, argv, "--jetson-only") || argumentEnabled(argc, argv, "--no-hardware");
}

const char* boolColor(bool value)
{
    return value ? "\033[32mON\033[0m" : "\033[90mOFF\033[0m";
}

int axisAbs(int value)
{
    return value < 0 ? -value : value;
}

bool axisActive(int value)
{
    constexpr int motion_deadzone =
        static_cast<int>(value::stick_norm * value::omni_deadzone + 0.5);
    return axisAbs(value) > motion_deadzone;
}

int axisPowerPercent(int value)
{
    return std::min(100, axisAbs(value) * 100 / 127);
}

int movePowerPercent(const Controller_Packet& pkt)
{
    return std::max(axisPowerPercent(pkt.lx_state), axisPowerPercent(pkt.ly_state));
}

int8_t displayAxis(double value, double full_scale)
{
    const int scaled = static_cast<int>(std::lround(value * 127.0 / full_scale));
    return static_cast<int8_t>(std::max(-127, std::min(127, scaled)));
}

// 実際に足回りへ効いている指令を、表示用のスティック軸へ落とす。
//
// v4パススルーと v3 物理速度指令では、足回りは ctrl_packet のスティック軸を
// 通らない。パススルー中は「スティック軸を上書きしない」のが仕様なので、
// ctrl_packet はMU3の中立値のまま残る。それをそのまま applied_cmd として
// 出していたため、egg8 が旋回や並進を出している間もこの欄は「停止」と
// 表示され続けた。運用中に「足回りに指令が入っているか」を見る欄が、
// 正常動作のときに停止と読めるということで、実際に誤読を生んでいる。
// jetson_cmd と同じ換算(handle_v4 の表示軸)を使うので、二つの欄は直接
// 比較できる。減速停止・非常停止では上流で速度が0にされるため、この欄も
// 停止になる。
Controller_Packet appliedDisplayPacket(
    const Controller_Packet& base,
    double vx_mps,
    double vy_mps,
    double w_radps)
{
    Controller_Packet display = base;
    display.lx_state = displayAxis(vy_mps * 1000.0, value::auto_display_full_mmps);
    display.ly_state = displayAxis(vx_mps * 1000.0, value::auto_display_full_mmps);
    display.rx_state = displayAxis(w_radps * 1000.0, value::auto_display_full_mradps);
    return display;
}

std::string motionIntent(const Controller_Packet& pkt)
{
    const int lx = static_cast<int>(pkt.lx_state);
    const int ly = static_cast<int>(pkt.ly_state);
    const int rx = static_cast<int>(pkt.rx_state);

    const bool move_x = axisActive(lx);
    const bool move_y = axisActive(ly);
    const bool turn = axisActive(rx);

    std::string movement;
    if (move_x && move_y)
    {
        movement = std::string(lx > 0 ? "右" : "左") + (ly > 0 ? "前" : "後");
    }
    else if (move_y)
    {
        movement = ly > 0 ? "前進" : "後退";
    }
    else if (move_x)
    {
        movement = lx > 0 ? "右移動" : "左移動";
    }

    std::string rotation;
    if (turn)
    {
        rotation = rx > 0 ? "右旋回" : "左旋回";
    }

    if (movement.empty() && rotation.empty())
    {
        return "停止";
    }
    if (movement.empty())
    {
        return "その場" + rotation;
    }
    if (rotation.empty())
    {
        return movement;
    }

    return movement + " + " + rotation;
}

std::string jetsonCommandText(const UDP& udp_port)
{
    if (udp_port.received_count() == 0)
    {
        return "未受信";
    }

    return motionIntent(udp_port.packet());
}

const char* jetsonLinkStatus(bool udp_ready, bool jetson_alive, uint64_t jetson_count)
{
    if (!udp_ready)
    {
        return "\033[31mUDP SOCKET ERROR\033[0m";
    }
    if (jetson_count == 0)
    {
        return "\033[33mWAITING FIRST UDP\033[0m";
    }
    if (jetson_alive)
    {
        return "\033[32mOK RECEIVING\033[0m";
    }

    return "\033[31mTIMEOUT\033[0m";
}

void printJetsonStatusLine(
    const UDP& udp_port,
    const Controller_Packet& applied_pkt,
    bool jetson_alive,
    bool is_auto,
    bool manual_auto_requested,
    bool jetson_auto_requested,
    int packet_rate_hz,
    uint64_t uart_sent,
    uint64_t uart_skipped,
    bool passthrough_out)
{
    const uint64_t jetson_count = udp_port.received_count();
    const bool udp_ready = udp_port.is_ready();

    const bool terminal_output = isatty(STDOUT_FILENO);

    if (terminal_output)
    {
        std::cout << "\r\033[K";
    }

    std::cout << "[Jetson] "
              << jetsonLinkStatus(udp_ready, jetson_alive, jetson_count)
              << " port=" << udp_port.port()
              << " packets=" << jetson_count
              << " rate=" << packet_rate_hz << "/s"
              << " last=" << udp_port.last_receive_age_ms() << "ms"
              << " auto_req(MU3/Jetson)=" << (manual_auto_requested ? "ON" : "OFF")
              << "/" << (jetson_auto_requested ? "ON" : "OFF")
              << " source=" << (is_auto ? "AUTO" : "MANUAL")
              << " applied_cmd=" << motionIntent(applied_pkt)
              << " jetson_cmd=" << jetsonCommandText(udp_port)
              << " proto=" << udp_port.protocol_name()
              << " safety=" << udp_port.safety_state_name()
              << " crc_err=" << udp_port.crc_error_count()
              << " stale=" << udp_port.stale_drop_count()
              << " uart_tx=" << uart_sent
              << " uart_skip=" << uart_skipped
              << " uart_src="
              << (passthrough_out
                      ? "\033[32mEGG8_PASSTHROUGH\033[0m"
                      : "BACON6_LOCAL");

    if (passthrough_out)
    {
        std::cout << "(" << udp_port.passthrough_commands().count << "cmd/"
                  << udp_port.passthrough_size() << "B)";
    }

    if (udp_port.estop_active())
    {
        std::cout << "  \033[31mESTOP(Jetson)\033[0m";
    }

    if (manual_auto_requested && !jetson_alive)
    {
        std::cout << "  \033[31mAUTO BLOCKED\033[0m";
    }

    if (terminal_output)
    {
        std::cout << std::flush;
    }
    else
    {
        std::cout << std::endl;
    }
}

void printAxisSummary(const char* label, const Controller_Packet& pkt)
{
    std::cout << std::left << std::setw(16) << label
              << " LX:" << std::right << std::setw(4) << static_cast<int>(pkt.lx_state)
              << " LY:" << std::right << std::setw(4) << static_cast<int>(pkt.ly_state)
              << " RX:" << std::right << std::setw(4) << static_cast<int>(pkt.rx_state)
              << " RY:" << std::right << std::setw(4) << static_cast<int>(pkt.ry_state)
              << std::endl;
}

void printButtonSummary(const char* label, const Controller_Packet& pkt)
{
    std::cout << std::left << std::setw(16) << label
              << " L2(auto):" << boolColor(pkt.l2_state)
              << " R1(gpio):" << boolColor(pkt.r1_state)
              << " R2(reload/up):" << boolColor(pkt.r2_state)
              << " L1(reverse):" << boolColor(pkt.l1_state)
              << " O:" << boolColor(pkt.maru_state)
              << " []:" << boolColor(pkt.shikaku_state)
              << std::endl;
}

void printPayloadBytes(const char* label, const uint8_t* payload, std::size_t size)
{
    std::cout << label;
    for (std::size_t i = 0; i < size; ++i)
    {
        std::cout << " [" << i << "]=" << std::setw(3) << static_cast<int>(payload[i]);
    }
    std::cout << std::endl;
}

void printRxDashboard(
    const Controller_Packet& manual_pkt,
    const Jetson_Packet& jetson_pkt,
    const Controller_Packet& applied_pkt,
    const packet& motor_pkt,
    const uint8_t mu3Data[7],
    const uint8_t jetsonData[UDP::payload_size],
    int mu3_count,
    uint64_t jetson_count,
    int jetson_age_ms,
    bool jetson_udp_ready,
    bool jetson_alive,
    bool is_auto,
    bool manual_auto_requested,
    bool jetson_auto_requested,
    const char* jetson_proto,
    uint16_t udp_bind_port,
    uint64_t jetson_crc_err,
    uint64_t jetson_stale_drop,
    bool jetson_estop,
    bool passthrough_out,
    const UartCommandFrame& passthrough)
{
    std::cout << "\033[2J\033[H";
    std::cout << "=== RX DEBUG DASHBOARD ===" << std::endl;
    std::cout << "MU3 Receive Count: " << mu3_count << std::endl;
    std::cout << "Jetson UDP Count : " << jetson_count
              << "  Last Age: " << jetson_age_ms << "ms" << std::endl;

    // Jetsonの通信状態と運転モードを表示
    std::cout << "\n--- Jetson Integration Status ---" << std::endl;
    std::cout << "UDP Socket : " << (jetson_udp_ready ? "\033[32mREADY\033[0m" : "\033[31mERROR\033[0m")
              << "  Port: " << udp_bind_port << std::endl;
    std::cout << "Protocol   : \033[32m" << jetson_proto << "\033[0m"
              << " (v3=物理速度/v2=seq+CRC/legacy=7byte)"
              << "  CRC err: " << jetson_crc_err
              << "  Stale drop: " << jetson_stale_drop
              << (jetson_estop ? "  \033[31mESTOP(Jetson)\033[0m" : "")
              << std::endl;
    std::cout << "Receive Health: " << jetsonLinkStatus(jetson_udp_ready, jetson_alive, jetson_count) << std::endl;
    std::cout << "Last Packet: " << jetson_age_ms << "ms ago"
              << "  OK threshold: <" << value::sys::alive_timeout_ms << "ms" << std::endl;
    std::cout << "Auto Request: MU3 L2=" << boolColor(manual_auto_requested)
              << "  Jetson L2=" << boolColor(jetson_auto_requested) << std::endl;
    std::cout << "Auto Ready: " << (jetson_alive ? "\033[32mYES\033[0m" : "\033[31mNO\033[0m") << std::endl;
    std::cout << "Drive Source: " << (is_auto ? "\033[32m[AUTOMATIC (Jetson)]\033[0m" : "\033[34m[MANUAL (Smartphone)]\033[0m") << std::endl;

    std::cout << "\n--- Auto Control Command (from Jetson UDP) ---" << std::endl;
    printPayloadBytes("Raw UDP bytes:", jetsonData, UDP::payload_size);
    printAxisSummary("Jetson axes", jetson_pkt);
    std::cout << "Jetson Command  "
              << (jetson_count == 0 ? "未受信" : motionIntent(jetson_pkt))
              << "  Move Power:" << std::setw(3) << movePowerPercent(jetson_pkt) << "%"
              << "  Turn Power:" << std::setw(3) << axisPowerPercent(jetson_pkt.rx_state) << "%"
              << std::endl;
    printButtonSummary("Jetson buttons", jetson_pkt);

    std::cout << "\n--- Effective Command ---" << std::endl;
    printAxisSummary("MU3 manual", manual_pkt);
    printAxisSummary("Applied drive", applied_pkt);
    std::cout << "Applied Command "
              << motionIntent(applied_pkt)
              << "  Move Power:" << std::setw(3) << movePowerPercent(applied_pkt) << "%"
              << "  Turn Power:" << std::setw(3) << axisPowerPercent(applied_pkt.rx_state) << "%"
              << std::endl;
    printButtonSummary("Applied buttons", applied_pkt);

    std::cout << "\n--- Motor UART Command ---" << std::endl;
    std::cout << "Frame Source: "
              << (passthrough_out
                      ? "\033[32m[EGG8 PASSTHROUGH (bytes relayed unmodified)]\033[0m"
                      : "\033[34m[BACON6 LOCAL (built from MU3/mixer)]\033[0m")
              << std::endl;
    if (passthrough_out)
    {
        std::cout << "Relayed cmd IDs:";
        for (std::size_t i = 0; i < passthrough.count; ++i)
        {
            std::cout << " " << static_cast<int>(passthrough.command[i])
                      << "=" << passthrough.value[i];
        }
        std::cout << "  (残りはbacon6が補完フレームで送信)" << std::endl;
    }
    std::cout << "OMNI m1:" << std::setw(6) << motor_pkt.m1
              << " m2:" << std::setw(6) << motor_pkt.m2
              << " m3:" << std::setw(6) << motor_pkt.m3
              << " m4:" << std::setw(6) << motor_pkt.m4 << std::endl;
    std::cout << "ARM m5:" << std::setw(6) << motor_pkt.m5
              << " GM:" << std::setw(6) << motor_pkt.gm
              << " UPDOWN m6:" << std::setw(6) << motor_pkt.m6
              << " COLLECT m7:" << std::setw(6) << motor_pkt.m7
              << " GPIO:" << std::setw(6) << motor_pkt.gpio << std::endl;

    std::cout << "\n--- MU3 Raw Received Data (Dec) ---" << std::endl;
    printPayloadBytes("Raw MU3 bytes:", mu3Data, 7);
    
    std::cout << "\n-----------------------------------" << std::endl;
    fflush(stdout);
}



int main(int argc, char* argv[])
{
    if (!installShutdownHandlers())
    {
        std::cerr << "[ERROR] SIGINT/SIGTERM handler installation failed"
                  << std::endl;
        return EXIT_FAILURE;
    }

    const bool dashboard_enabled = dashboardEnabled(argc, argv);
    const bool jetson_only = jetsonOnlyEnabled(argc, argv);

    // 1. MU3初期化
    std::unique_ptr<MU3> mu3;
    if (!jetson_only)
    {
        mu3 = std::make_unique<MU3>(value::sys::mu3_device);
        if (!mu3->is_open())
        {
            std::cerr << "受信側MU-3の初期化に失敗" << std::endl;
            std::cerr << "[WARN] MU3なしで継続します。手動入力は無効、Jetson UDP自動制御は有効です。" << std::endl;
            mu3.reset();
        }
    }
    else
    {
        std::cout << "[INFO] Jetson-only mode: MU3/UARTを開かずにUDPだけ監視します" << std::endl;
    }

    // 2. ロボマス開発ボードへのUART
    // MU3_MOTOR_DEVICE環境変数で出力先を差し替えられる。実機のモーターを
    // 回さずにパススルーのバイト列を検証するため（ptyへ向ける）。
    std::unique_ptr<UART> motor_uart;
    if (!jetson_only)
    {
        std::string motor_device = value::sys::motor_device;
        if (const char* device_env = std::getenv("MU3_MOTOR_DEVICE"))
        {
            if (device_env[0] != '\0')
            {
                motor_device = device_env;
                std::cout << "[INFO] Motor UART device override: "
                          << motor_device << std::endl;
            }
        }
        motor_uart = std::make_unique<UART>(motor_device, value::sys::motor_baud);
    }

    // 3. UDP通信（Jetsonから有線LAN経由で受信）
    // MU3_JETSON_PORT環境変数でポートを上書き可能（本番サービスと並行して
    // 別ポートでテストインスタンスを立てる用途）
    uint16_t jetson_port = value::sys::jetson_port;
    if (const char* port_env = std::getenv("MU3_JETSON_PORT"))
    {
        const int parsed = std::atoi(port_env);
        if (parsed > 0 && parsed < 65536)
        {
            jetson_port = static_cast<uint16_t>(parsed);
        }
        else
        {
            std::cerr << "[WARN] MU3_JETSON_PORT=" << port_env
                      << " は無効のため既定ポートを使用します" << std::endl;
        }
    }
    UDP udp_port(jetson_port);
    std::cout << "[INFO] Jetson UDP port: " << jetson_port << std::endl;

    Controller_Packet ctrl_packet;
    struct packet rm_packet;
    std::memset(&ctrl_packet, 0, sizeof(ctrl_packet));
    std::memset(&rm_packet, 0, sizeof(rm_packet));
    uint8_t last_rx_data[7] = {0};

    // MU3のomniスティック値。有効フレーム受信時だけ更新し、途絶時は
    // 最終値を保持せず安全にゼロへ移す。
    int8_t mu3_lx = 0;
    int8_t mu3_ly = 0;
    int8_t mu3_rx = 0;
    int8_t mu3_ry = 0;

    OMNI   omni_system;
    ARM    ARM_system;
    GM     GM_system;
    UPDOWN  updown_system;
    COLLECT collect_system;
    GPIO   GPIO_system;
    ControlledStopLimiter drive_stop_limiter(
        value::sys::controlled_stop_units_per_sec,
        value::sys::loop_period_ms / 1000.0);

    int timeout_counter = 0;
    const int TIMEOUT_THRESHOLD = value::sys::timeout_threshold; // MU3受信タイムアウト（約100ms）
    int receive_count = 0;            // パケットの正常受信回数カウント
    int loop_count    = 0;            // メインループの回数（ダッシュボード更新の間引き用）
    const int DASHBOARD_INTERVAL = value::sys::dashboard_interval; // ダッシュボード更新間隔

    using namespace std::chrono;
    auto next_loop = steady_clock::now();
    auto last_jetson_status = steady_clock::now() - milliseconds(value::sys::jetson_status_interval_ms);
    uint64_t last_jetson_status_count = 0;
    bool prev_jetson_ready = udp_port.is_ready();
    bool prev_jetson_alive = false;
    bool prev_jetson_waiting = true;
    bool first_jetson_status = true;

    std::cout << "受信待機中..." << std::endl;

    while (!shutdown_requested)
    {
        uint8_t rxData[7] = {0};

        next_loop += milliseconds(value::sys::loop_period_ms);
        // Neither host runs a real-time kernel, so this loop is occasionally
        // descheduled for longer than one period.  Chasing every missed
        // deadline then runs the following iterations back to back with no
        // sleep at all, and each one tries to push another 29-byte COBS frame
        // into a 115200 baud UART that needs 2.5 ms per frame.  The TIOCOUTQ
        // guard in UART::send_data drops all but the first of that burst, so a
        // scheduling stall turned into a run of *stale* drive commands held by
        // the Dev Board.  Resync instead: after a stall the newest command is
        // worth more than a burst of the ones that were missed.
        {
            const auto now_tp = steady_clock::now();
            if (next_loop + milliseconds(value::sys::loop_period_ms) < now_tp)
            {
                next_loop = now_tp;
            }
        }
        std::this_thread::sleep_until(next_loop);

        // パケット受信
        // 受信バッファに溜まっているフレームを全て読み切り、最新の1フレームだけを採用する。
        // （古いフレームを1ループ1個ずつ処理して遅延が蓄積するのを防ぐ）
        int len = 0;
        uint8_t frame[7];
        while (mu3 && mu3->receive(frame, 7) == 7)
        {
            udp_port.remote_navigation.receive(frame);
            std::memcpy(rxData, frame, sizeof(rxData));
            len = 7;
        }

        if (len == 7)
        {
            // アナログスティックの復元
            int16_t lx_raw = static_cast<int16_t>((rxData[0] << 8) - 32768);
            int16_t ly_raw = static_cast<int16_t>((rxData[1] << 8) - 32768);
            int16_t rx_raw = static_cast<int16_t>((rxData[2] << 8) - 32768);
            int16_t ry_raw = static_cast<int16_t>((rxData[3] << 8) - 32768);

            // MU3のomni値は専用変数に保持する（omniの出力元切替は後段で行う）
            mu3_lx = static_cast<int8_t>(std::max(-127, std::min(127, static_cast<int>(-1 * (lx_raw / 256)))));
            mu3_ly = static_cast<int8_t>(std::max(-127, std::min(127, static_cast<int>(-1 * (ly_raw / 256)))));
            mu3_rx = static_cast<int8_t>(rx_raw / 256);
            mu3_ry = static_cast<int8_t>(std::max(-127, std::min(127, static_cast<int>(-1 * (ry_raw / 256)))));

            // ボタンの復元
            ctrl_packet.shita_state   = (rxData[4] >> 7) & 1;
            ctrl_packet.ue_state      = (rxData[4] >> 6) & 1;
            ctrl_packet.batsu_state   = (rxData[4] >> 5) & 1;
            ctrl_packet.sankaku_state = (rxData[4] >> 4) & 1;
            ctrl_packet.l1_state      = (rxData[4] >> 3) & 1;
            ctrl_packet.r1_state      = (rxData[4] >> 2) & 1;
            ctrl_packet.l2_state      = (rxData[4] >> 1) & 1;
            ctrl_packet.r2_state      = (rxData[4] >> 0) & 1;

            ctrl_packet.r3_state      = (rxData[5] >> 7) & 1;
            ctrl_packet.l3_state      = (rxData[5] >> 6) & 1;
            ctrl_packet.option_state  = (rxData[5] >> 5) & 1;
            ctrl_packet.create_state  = (rxData[5] >> 4) & 1;
            ctrl_packet.shikaku_state = (rxData[5] >> 3) & 1;
            ctrl_packet.maru_state    = (rxData[5] >> 2) & 1;
            ctrl_packet.hidari_state  = (rxData[5] >> 1) & 1;
            ctrl_packet.migi_state    = (rxData[5] >> 0) & 1;

            ctrl_packet.ps_state      = (rxData[6] >> 0) & 1;
            
            receive_count++;
            timeout_counter = 0;
            std::memcpy(last_rx_data, rxData, sizeof(last_rx_data));
        }
        else
        {
            timeout_counter++;
            if (timeout_counter >= TIMEOUT_THRESHOLD)
            {
                timeout_counter = TIMEOUT_THRESHOLD;
                udp_port.remote_navigation.timeout();
                mu3_lx = 0;
                mu3_ly = 0;
                mu3_rx = 0;
                mu3_ry = 0;
                // 押しっぱなしの機構・GPIOも保持しない。ARM/GMの位置目標は
                // 各クラス内に残るため、速度機構だけが停止する。
                std::memset(&ctrl_packet, 0, sizeof(ctrl_packet));
            }
        }

        udp_port.update(ctrl_packet);

        const bool jetson_alive = udp_port.jetson_alive();
        const bool is_auto_mode = udp_port.is_auto_mode();
        // L2による自動走行中にegg8がUARTフレーム(v4)を送ってきていれば、
        // 足回りの指令はそのバイト列をそのまま開発ボードへ中継する。
        // egg8が含めなかったコマンドは、下でMU3由来の値から補完する。
        const bool use_passthrough =
            is_auto_mode && udp_port.uart_passthrough_active();
        const UartCommandFrame& passthrough = udp_port.passthrough_commands();
        const bool manual_auto_requested = value::read_button(ctrl_packet, value::auto_control_on);
        const bool jetson_auto_requested = udp_port.packet().l2_state;
        const auto now = steady_clock::now();

        // omniの出力元を明示的に選択する。
        //  ・自動走行中(L2押下 かつ Jetson生存)     : JetsonのUDP値
        //  ・それ以外(手動)                          : 生存中のMU3値
        // 自動リンク途絶時は手動へ瞬時切替せず、下の停止ラッチで減速停止する。
        Controller_Packet manual_packet = ctrl_packet;
        manual_packet.lx_state = mu3_lx;
        manual_packet.ly_state = mu3_ly;
        manual_packet.rx_state = mu3_rx;
        manual_packet.ry_state = mu3_ry;

        // パススルー中はスティック軸を上書きしない。足回りはegg8のバイト列
        // から取るので、ここではMU3の値をそのまま残す。egg8がomniコマンドを
        // 含めなかった場合の退避経路として、また機構のボタン操作を従来通り
        // 効かせるために必要（将来egg8が機構も送るようになれば自動的に
        // egg8側の値が優先される）。
        if (is_auto_mode && !use_passthrough)
        {
            // このlegacy/v2経路は checker_omni() の手動ミキサを通るので、
            // 実機で確定した契約に合わせる必要がある:
            //   横入力が正 -> 左へ, 並進入力が正 -> 前へ, 旋回入力が正 -> 時計回り
            // Jetsonは lx=vy(ROS左+), ly=vx(前+), rx=ω(CCW+) の符号で送るので、
            //   lx: 左が正どうしなので素通し
            //   ly: 前が正どうしなので素通し
            //   rx: ROSのCCW(+ω)を「時計回りが正」の入力へ渡すため反転が要る
            // v3経路の auto_lateral_sign / auto_turn_sign と同じ結論になる。
            // 2026-08-07 に横を実機で測り直して lx の反転をやめた（横入力が正
            // で右へ行くという以前の前提は、スティックの符号についての推測で
            // あって計測ではなかった）。
            ctrl_packet.lx_state = udp_port.packet().lx_state;
            ctrl_packet.ly_state = udp_port.packet().ly_state;
            ctrl_packet.rx_state = static_cast<int8_t>(std::max(-127, std::min(127, -static_cast<int>(udp_port.packet().rx_state))));
            ctrl_packet.ry_state = udp_port.packet().ry_state;
        }
        else
        {
            ctrl_packet = manual_packet;
        }

        // --- 3. 演算 ---
        omni_system.packet_range(ctrl_packet);
        omni_system.checker_omni(ctrl_packet);
        ARM_system.packet_range(ctrl_packet);
        GM_system.packet_range(ctrl_packet);
        updown_system.packet_range(ctrl_packet);
        collect_system.packet_range(ctrl_packet);
        GPIO_system.packet_range(ctrl_packet);
        
        // 足回りの出力元は3通り:
        //  ・v4パススルー(現行の自動走行): egg8が確定した各輪値をそのまま使う
        //  ・v3物理速度指令: デッドゾーンなし・手動速度レンジ非依存の固定較正
        //  ・それ以外（手動 / legacy / v2自動）: 従来のスティックミキシング
        double applied_vx_mps  = 0.0;
        double applied_vy_mps  = 0.0;
        double applied_w_radps = 0.0;
        if (use_passthrough)
        {
            // 中継するバイト列から車輪値を読み出す。中継そのものには不要だが、
            // 減速停止のランプを実際の指令値から始めるため、また各輪値を
            // テレメトリへ載せるために必要。egg8が省いたコマンドについては
            // 手動ミキサの値をそのまま残す。
            rm_packet.m1 = omni_system.motor_speed(0);
            rm_packet.m2 = omni_system.motor_speed(1);
            rm_packet.m3 = omni_system.motor_speed(2);
            rm_packet.m4 = omni_system.motor_speed(3);
            int16_t wheel = 0;
            if (passthrough.find(value::cmd_omni[0], wheel)) rm_packet.m1 = wheel;
            if (passthrough.find(value::cmd_omni[1], wheel)) rm_packet.m2 = wheel;
            if (passthrough.find(value::cmd_omni[2], wheel)) rm_packet.m3 = wheel;
            if (passthrough.find(value::cmd_omni[3], wheel)) rm_packet.m4 = wheel;
            // v4ヘッダの速度は制御に使わない参考値。診断表示のために返す。
            applied_vx_mps  = udp_port.command_vx_mps();
            applied_vy_mps  = udp_port.command_vy_mps();
            applied_w_radps = udp_port.command_w_radps();
        }
        else if (is_auto_mode && udp_port.velocity_command_active())
        {
            int16_t wheels[4];
            const double k = OMNI::speeds_from_velocity(
                udp_port.command_vx_mps(),
                udp_port.command_vy_mps(),
                udp_port.command_w_radps(),
                wheels);
            rm_packet.m1 = wheels[0];
            rm_packet.m2 = wheels[1];
            rm_packet.m3 = wheels[2];
            rm_packet.m4 = wheels[3];
            applied_vx_mps  = udp_port.command_vx_mps() * k;
            applied_vy_mps  = udp_port.command_vy_mps() * k;
            applied_w_radps = udp_port.command_w_radps() * k;
        }
        else
        {
            rm_packet.m1 = omni_system.motor_speed(0);
            rm_packet.m2 = omni_system.motor_speed(1);
            rm_packet.m3 = omni_system.motor_speed(2);
            rm_packet.m4 = omni_system.motor_speed(3);
        }

        const bool mu3_alive_now =
            mu3 && receive_count > 0 && timeout_counter < TIMEOUT_THRESHOLD;
        const bool manual_timed_out = receive_count > 0 && !mu3_alive_now;
        const bool controlled_stop = udp_port.safety_stop_active() ||
            (!is_auto_mode && manual_timed_out);
        const std::array<int16_t, 4> requested_wheels{
            rm_packet.m1, rm_packet.m2, rm_packet.m3, rm_packet.m4};
        const std::array<int16_t, 4> safe_wheels = drive_stop_limiter.apply(
            requested_wheels,
            controlled_stop,
            udp_port.estop_active());
        rm_packet.m1 = safe_wheels[0];
        rm_packet.m2 = safe_wheels[1];
        rm_packet.m3 = safe_wheels[2];
        rm_packet.m4 = safe_wheels[3];
        if (controlled_stop || udp_port.estop_active())
        {
            // The reference is zero while the wheel targets ramp down; actual
            // ramp values remain visible in wheel_commands telemetry.
            applied_vx_mps = 0.0;
            applied_vy_mps = 0.0;
            applied_w_radps = 0.0;
        }
        // 足回りがスティック軸を通らない経路のときは、表示もそちらから作る。
        // ここを ctrl_packet のままにすると applied_cmd が常に「停止」になる。
        const bool velocity_driven_drive =
            use_passthrough || (is_auto_mode && udp_port.velocity_command_active());
        const Controller_Packet applied_packet =
            velocity_driven_drive
                ? appliedDisplayPacket(
                      ctrl_packet, applied_vx_mps, applied_vy_mps, applied_w_radps)
                : ctrl_packet;
        rm_packet.m5 = ARM_system.arm_range();
        rm_packet.gm = GM_system.angle();
        rm_packet.m6 = updown_system.motor_speed();
        rm_packet.m7 = collect_system.motor_speed();
        rm_packet.gpio = GPIO_system.GPIO_state();

        // 機構もegg8が送ってきていれば、そちらが優先される（将来の委譲）。
        // 送ってこないうちは上のMU3由来の値がそのまま補完フレームへ出る。
        if (use_passthrough)
        {
            int16_t mechanism = 0;
            if (passthrough.find(value::cmd_arm, mechanism))     rm_packet.m5 = mechanism;
            if (passthrough.find(value::cmd_gm, mechanism))      rm_packet.gm = mechanism;
            if (passthrough.find(value::cmd_updown, mechanism))  rm_packet.m6 = mechanism;
            if (passthrough.find(value::cmd_collect, mechanism)) rm_packet.m7 = mechanism;
            if (passthrough.find(value::cmd_gpio, mechanism))
            {
                rm_packet.gpio = static_cast<uint16_t>(mechanism);
            }
        }

        // 中継してよいのは、安全側が指令に手を入れていないときだけ。
        // 減速停止・非常停止中はegg8のバイト列を捨て、ランプ済みの値で
        // bacon6が組んだフレームを送る。
        const bool passthrough_out =
            use_passthrough &&
            !controlled_stop &&
            !udp_port.estop_active() &&
            safe_wheels == requested_wheels;

        // --- Jetsonへテレメトリ返信（適用値・各輪値・機体状態・受信統計） ---
        const bool motor_uart_open = motor_uart && motor_uart->is_open();
        udp_port.send_telemetry(ctrl_packet, mu3_alive_now, motor_uart_open,
                                rm_packet,
                                applied_vx_mps, applied_vy_mps, applied_w_radps);

        // ダッシュボードは毎ループ描画すると重いので間引く
        if (dashboard_enabled && loop_count % DASHBOARD_INTERVAL == 0)
        {
            printRxDashboard(
                manual_packet,
                udp_port.packet(),
                applied_packet,
                rm_packet,
                last_rx_data,
                udp_port.raw_payload(),
                receive_count,
                udp_port.received_count(),
                udp_port.last_receive_age_ms(),
                udp_port.is_ready(),
                jetson_alive,
                is_auto_mode,
                manual_auto_requested,
                jetson_auto_requested,
                udp_port.protocol_name(),
                udp_port.port(),
                udp_port.crc_error_count(),
                udp_port.stale_drop_count(),
                udp_port.estop_active(),
                passthrough_out,
                passthrough);
        }
        else if (!dashboard_enabled)
        {
            const bool jetson_waiting = udp_port.received_count() == 0;
            const bool jetson_state_changed =
                first_jetson_status ||
                prev_jetson_ready != udp_port.is_ready() ||
                prev_jetson_alive != jetson_alive ||
                prev_jetson_waiting != jetson_waiting;
            const bool status_due =
                now - last_jetson_status >= milliseconds(value::sys::jetson_status_interval_ms);

            if (jetson_state_changed || status_due)
            {
                const uint64_t current_count = udp_port.received_count();
                const auto elapsed_ms = std::max<int64_t>(
                    1,
                    duration_cast<milliseconds>(now - last_jetson_status).count());
                const int packet_rate_hz = static_cast<int>(
                    ((current_count - last_jetson_status_count) * 1000) / elapsed_ms);

                printJetsonStatusLine(
                    udp_port,
                    applied_packet,
                    jetson_alive,
                    is_auto_mode,
                    manual_auto_requested,
                    jetson_auto_requested,
                    packet_rate_hz,
                    motor_uart ? motor_uart->sent_count() : 0,
                    motor_uart ? motor_uart->busy_skip_count() : 0,
                    passthrough_out);

                last_jetson_status = now;
                last_jetson_status_count = current_count;
                prev_jetson_ready = udp_port.is_ready();
                prev_jetson_alive = jetson_alive;
                prev_jetson_waiting = jetson_waiting;
                first_jetson_status = false;
            }
        }

        // --- 4. UART送信 ---
        if (motor_uart_open)
        {
            // 通常200 Hz経路では待機・再送せず、最新指令を優先する。
            if (passthrough_out)
            {
                // egg8のフレームは1バイトも書き換えずに出す。egg8が送って
                // いないコマンドだけをMU3由来の値で補い、同じwriteで続ける
                // （分割するとUARTの詰まりガードで2本目だけ落ちる）。
                uint8_t out_bytes[UartCommandFrame::max_encoded +
                                  3 * robomas_command_count + 2];
                std::size_t out_len = udp_port.passthrough_size();
                std::memcpy(out_bytes, udp_port.passthrough_bytes(), out_len);

                uint8_t ids[robomas_command_count];
                int16_t values[robomas_command_count];
                const std::size_t total =
                    expand_packet_commands(rm_packet, ids, values);

                uint8_t missing_ids[robomas_command_count];
                int16_t missing_values[robomas_command_count];
                std::size_t missing = 0;
                for (std::size_t i = 0; i < total; ++i)
                {
                    if (!passthrough.contains(ids[i]))
                    {
                        missing_ids[missing] = ids[i];
                        missing_values[missing] = values[i];
                        ++missing;
                    }
                }

                if (missing > 0)
                {
                    const std::vector<uint8_t> complement =
                        encode_robomas_frame(missing_ids, missing_values, missing);
                    std::memcpy(out_bytes + out_len, complement.data(),
                                complement.size());
                    out_len += complement.size();
                }

                (void)motor_uart->uart_send_encoded(out_bytes, out_len);
            }
            else
            {
                (void)motor_uart->uart_send(rm_packet);
            }
        }

        loop_count++;
    }

    std::cerr << "[SHUTDOWN] termination requested; stopping actuators"
              << std::endl;
    bool shutdown_safe = true;
    if (motor_uart && motor_uart->is_open())
    {
        shutdown_safe = sendShutdownStopFrames(*motor_uart, rm_packet);
    }
    else if (!jetson_only)
    {
        shutdown_safe = false;
        std::cerr << "[SHUTDOWN] UART is not open; stop frames were not sent"
                  << std::endl;
    }
    else
    {
        std::cerr << "[SHUTDOWN] Jetson-only mode; UART stop frames skipped"
                  << std::endl;
    }

    return shutdown_safe ? EXIT_SUCCESS : EXIT_FAILURE;
}
