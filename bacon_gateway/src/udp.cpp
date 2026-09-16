#include "udp.hpp"
#include "value.hpp"

#include <algorithm>
#include <arpa/inet.h>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <ctime>
#include <iostream>
#include <netinet/ip.h>
#include <unistd.h>

namespace
{
    int8_t clamp_axis(int value)
    {
        return static_cast<int8_t>(std::max(-127, std::min(127, value)));
    }

    int8_t decode_axis(uint8_t value, bool invert = false)
    {
        int axis = static_cast<int>(value) - 128;
        if (invert)
        {
            axis = -axis;
        }
        return clamp_axis(axis);
    }

    // legacy payload[4..6] / v2 buttons[0..2] 共通のビット配置を展開する
    void decode_buttons(uint8_t b0, uint8_t b1, uint8_t b2, Jetson_Packet& packet)
    {
        packet.shita_state   = (b0 >> 7) & 1;
        packet.ue_state      = (b0 >> 6) & 1;
        packet.batsu_state   = (b0 >> 5) & 1;
        packet.sankaku_state = (b0 >> 4) & 1;
        packet.l1_state      = (b0 >> 3) & 1;
        packet.r1_state      = (b0 >> 2) & 1;
        packet.l2_state      = (b0 >> 1) & 1;
        packet.r2_state      = (b0 >> 0) & 1;

        packet.r3_state      = (b1 >> 7) & 1;
        packet.l3_state      = (b1 >> 6) & 1;
        packet.option_state  = (b1 >> 5) & 1;
        packet.create_state  = (b1 >> 4) & 1;
        packet.shikaku_state = (b1 >> 3) & 1;
        packet.maru_state    = (b1 >> 2) & 1;
        packet.hidari_state  = (b1 >> 1) & 1;
        packet.migi_state    = (b1 >> 0) & 1;

        packet.ps_state      = (b2 >> 0) & 1;
    }

    Jetson_Packet decode_mu3_controller_payload(const uint8_t payload[UDP::payload_size])
    {
        Jetson_Packet packet;
        std::memset(&packet, 0, sizeof(packet));

        packet.lx_state = decode_axis(payload[0]);
        packet.ly_state = decode_axis(payload[1], true);
        packet.rx_state = decode_axis(payload[2]);
        packet.ry_state = decode_axis(payload[3], true);

        decode_buttons(payload[4], payload[5], payload[6], packet);
        return packet;
    }

    // CRC-16/CCITT-FALSE (poly 0x1021, init 0xFFFF, 反転なし)
    // Jetson側 motor_udp_protocol.py の crc16_ccitt と同一であること
    uint16_t crc16_ccitt(const uint8_t* data, std::size_t length)
    {
        uint16_t crc = 0xFFFF;
        for (std::size_t i = 0; i < length; ++i)
        {
            crc = static_cast<uint16_t>(crc ^ (static_cast<uint16_t>(data[i]) << 8));
            for (int bit = 0; bit < 8; ++bit)
            {
                if (crc & 0x8000)
                {
                    crc = static_cast<uint16_t>((crc << 1) ^ 0x1021);
                }
                else
                {
                    crc = static_cast<uint16_t>(crc << 1);
                }
            }
        }
        return crc;
    }

    uint16_t read_u16(const uint8_t* p)
    {
        return static_cast<uint16_t>(p[0] | (static_cast<uint16_t>(p[1]) << 8));
    }

    uint32_t read_u32(const uint8_t* p)
    {
        return static_cast<uint32_t>(p[0])
             | (static_cast<uint32_t>(p[1]) << 8)
             | (static_cast<uint32_t>(p[2]) << 16)
             | (static_cast<uint32_t>(p[3]) << 24);
    }

    void write_u16(uint8_t* p, uint16_t v)
    {
        p[0] = static_cast<uint8_t>(v & 0xFF);
        p[1] = static_cast<uint8_t>((v >> 8) & 0xFF);
    }

    void write_u32(uint8_t* p, uint32_t v)
    {
        p[0] = static_cast<uint8_t>(v & 0xFF);
        p[1] = static_cast<uint8_t>((v >> 8) & 0xFF);
        p[2] = static_cast<uint8_t>((v >> 16) & 0xFF);
        p[3] = static_cast<uint8_t>((v >> 24) & 0xFF);
    }

    // hold_us計測用（カーネル受信タイムスタンプSCM_TIMESTAMPNSと同じCLOCK_REALTIME系）
    int64_t realtime_now_ns()
    {
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        return static_cast<int64_t>(ts.tv_sec) * 1000000000LL + ts.tv_nsec;
    }
}

UDP::UDP(uint16_t port)
    : fd(-1),
      is_initialized(false),
      bound_port(port),
      recv_count(0),
      last_recv_time(std::chrono::steady_clock::now() - ALIVE_TIMEOUT),
      is_alive(false),
      auto_mode(false),
      last_cmd_v2(false),
      last_cmd_v3(false),
      last_cmd_v4(false),
      v2_auto_request(false),
      last_estop_request(false),
      has_seq(false),
      last_seq(0),
      last_cmd_seq(0),
      last_cmd_t_tx_us(0),
      last_cmd_arrival_ns(0),
      crc_err_count(0),
      stale_drops(0),
      size_err_count(0),
      foreign_drop_count(0),
      v3_vx_mmps(0),
      v3_vy_mmps(0),
      v3_w_mradps(0),
      v4_uart_len(0),
      source_locked(false),
      pi_seq(0),
      telemetry_tick(0),
      rate_window_start(std::chrono::steady_clock::now()),
      rate_window_count(0),
      last_rate_hz(0),
      last_size_warn(std::chrono::steady_clock::now() - std::chrono::seconds(10)),
      link_safety(value::sys::link_degraded_ms,
                  value::sys::controlled_stop_ms,
                  value::sys::fault_timeout_ms)
{
    std::memset(&jetson_packet, 0, sizeof(jetson_packet));
    std::memset(last_payload, 0, sizeof(last_payload));
    std::memset(v4_uart, 0, sizeof(v4_uart));
    std::memset(&source_addr, 0, sizeof(source_addr));

    fd = socket(AF_INET, SOCK_DGRAM | SOCK_NONBLOCK, 0);
    if (fd < 0)
    {
        std::cerr << "Failed to create UDP socket: " << std::strerror(errno) << std::endl;
        return;
    }

    const int enabled = 1;
    if (setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &enabled, sizeof(enabled)) < 0)
    {
        std::cerr << "Failed to set SO_REUSEADDR: " << std::strerror(errno) << std::endl;
    }

    // バースト受信（タイマ送信＋即時送信の重なり）でも取りこぼさない容量
    const int receive_buffer_bytes = 65536;
    if (setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &receive_buffer_bytes, sizeof(receive_buffer_bytes)) < 0)
    {
        std::cerr << "Failed to set SO_RCVBUF: " << std::strerror(errno) << std::endl;
    }

#ifdef SO_TIMESTAMPNS
    // カーネル到着時刻でhold_us（Pi内滞留時間）を正確に測る
    if (setsockopt(fd, SOL_SOCKET, SO_TIMESTAMPNS, &enabled, sizeof(enabled)) < 0)
    {
        std::cerr << "Failed to set SO_TIMESTAMPNS: " << std::strerror(errno) << std::endl;
    }
#endif

#ifdef IP_TOS
    const int tos = IPTOS_LOWDELAY;
    if (setsockopt(fd, IPPROTO_IP, IP_TOS, &tos, sizeof(tos)) < 0)
    {
        std::cerr << "Failed to set IP_TOS: " << std::strerror(errno) << std::endl;
    }
#endif

#ifdef SO_PRIORITY
    const int priority = 6;
    if (setsockopt(fd, SOL_SOCKET, SO_PRIORITY, &priority, sizeof(priority)) < 0)
    {
        std::cerr << "Failed to set SO_PRIORITY: " << std::strerror(errno) << std::endl;
    }
#endif

    struct sockaddr_in server_addr;
    std::memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family      = AF_INET;
    server_addr.sin_addr.s_addr = htonl(INADDR_ANY);
    server_addr.sin_port        = htons(port);

    if (bind(fd, reinterpret_cast<struct sockaddr*>(&server_addr), sizeof(server_addr)) < 0)
    {
        std::cerr << "Failed to bind UDP socket on port " << port
                  << ": " << std::strerror(errno) << std::endl;
        close(fd);
        fd = -1;
        return;
    }

    is_initialized = true;
}

UDP::~UDP()
{
    if (fd >= 0)
    {
        close(fd);
    }
}

bool UDP::accept_source(const struct sockaddr_in& src,
                        const std::chrono::steady_clock::time_point& now)
{
    if (source_locked &&
        src.sin_addr.s_addr == source_addr.sin_addr.s_addr &&
        src.sin_port == source_addr.sin_port)
    {
        return true;
    }

    // 未ロック、またはロック相手が一定時間無通信なら新しい送信元を採用する
    const bool lock_expired =
        now - last_recv_time >
        std::chrono::milliseconds(value::sys::source_lock_timeout_ms);
    if (!source_locked || lock_expired)
    {
        source_addr = src;
        source_locked = true;
        has_seq = false; // 送信元が変わったのでシーケンスを再同期
        char ip[INET_ADDRSTRLEN] = "?";
        inet_ntop(AF_INET, &src.sin_addr, ip, sizeof(ip));
        std::cout << "[Jetson] command source locked: "
                  << ip << ":" << ntohs(src.sin_port) << std::endl;
        return true;
    }

    return false;
}

void UDP::handle_legacy(const uint8_t* data)
{
    std::memcpy(last_payload, data, payload_size);
    jetson_packet = decode_mu3_controller_payload(data);
    last_cmd_v2 = false;
    last_cmd_v3 = false;
    last_cmd_v4 = false;
    v4_uart_len = 0;
    v4_commands.count = 0;
    v2_auto_request = false;
    last_estop_request = false;
}

void UDP::store_display_payload(const uint8_t buttons[3])
{
    // ダッシュボード表示用にlegacy相当の7バイトへ再構成する
    last_payload[0] = static_cast<uint8_t>(jetson_packet.lx_state + 128);
    last_payload[1] = static_cast<uint8_t>(128 - jetson_packet.ly_state);
    last_payload[2] = static_cast<uint8_t>(jetson_packet.rx_state + 128);
    last_payload[3] = static_cast<uint8_t>(128 - jetson_packet.ry_state);
    last_payload[4] = buttons[0];
    last_payload[5] = buttons[1];
    last_payload[6] = buttons[2];
}

bool UDP::accept_command_frame(const uint8_t* data, std::size_t crc_offset,
                               std::size_t seq_offset, uint16_t& seq_out)
{
    if (crc16_ccitt(data, crc_offset) != read_u16(data + crc_offset))
    {
        crc_err_count++;
        return false;
    }

    const uint16_t seq = read_u16(data + seq_offset);
    if (has_seq)
    {
        // UDPの追い越し/重複対策。ただし送信側再起動に備えて
        // 一定時間の無通信後は無条件で再同期する。
        const int16_t gap = static_cast<int16_t>(seq - last_seq);
        const bool resync =
            std::chrono::steady_clock::now() - last_recv_time >
            std::chrono::milliseconds(value::sys::seq_resync_silence_ms);
        if (gap <= 0 && !resync)
        {
            stale_drops++;
            return false;
        }
    }
    last_seq = seq;
    has_seq = true;
    seq_out = seq;
    return true;
}

bool UDP::handle_v2(const uint8_t* data)
{
    if (data[0] != protocol_magic || data[1] != v2_command_version)
    {
        crc_err_count++;
        return false;
    }

    uint16_t seq = 0;
    if (!accept_command_frame(data, 18, 10, seq))
    {
        return false;
    }

    const uint8_t flags = data[2];
    v2_auto_request = (flags & 0x01) != 0;
    last_estop_request = (flags & 0x02) != 0;

    Jetson_Packet pkt;
    std::memset(&pkt, 0, sizeof(pkt));
    pkt.lx_state = static_cast<int8_t>(data[6]);
    pkt.ly_state = static_cast<int8_t>(data[7]);
    pkt.rx_state = static_cast<int8_t>(data[8]);
    pkt.ry_state = static_cast<int8_t>(data[9]);
    decode_buttons(data[3], data[4], data[5], pkt);
    jetson_packet = pkt;

    last_cmd_v2 = true;
    last_cmd_v3 = false;
    last_cmd_v4 = false;
    v4_uart_len = 0;
    v4_commands.count = 0;
    last_cmd_seq = seq;
    last_cmd_t_tx_us = read_u32(data + 12);

    store_display_payload(data + 3);
    return true;
}

bool UDP::handle_v3(const uint8_t* data)
{
    if (data[0] != protocol_magic || data[1] != v3_command_version)
    {
        crc_err_count++;
        return false;
    }

    uint16_t seq = 0;
    if (!accept_command_frame(data, 22, 12, seq))
    {
        return false;
    }

    const uint8_t flags = data[2];
    v2_auto_request = (flags & 0x01) != 0;
    last_estop_request = (flags & 0x02) != 0;

    v3_vx_mmps  = static_cast<int16_t>(read_u16(data + 6));
    v3_vy_mmps  = static_cast<int16_t>(read_u16(data + 8));
    v3_w_mradps = static_cast<int16_t>(read_u16(data + 10));

    // ダッシュボード・従来ロジック互換のint8等価スティック値も作る
    // （v2の較正 0.55m/s・1.2rad/s フルスケールに一致させる）
    Jetson_Packet pkt;
    std::memset(&pkt, 0, sizeof(pkt));
    pkt.lx_state = clamp_axis(static_cast<int>(std::lround(
        v3_vy_mmps * 127.0 / value::auto_display_full_mmps)));
    pkt.ly_state = clamp_axis(static_cast<int>(std::lround(
        v3_vx_mmps * 127.0 / value::auto_display_full_mmps)));
    pkt.rx_state = clamp_axis(static_cast<int>(std::lround(
        v3_w_mradps * 127.0 / value::auto_display_full_mradps)));
    decode_buttons(data[3], data[4], data[5], pkt);
    jetson_packet = pkt;

    last_cmd_v2 = false;
    last_cmd_v3 = true;
    last_cmd_v4 = false;
    v4_uart_len = 0;
    v4_commands.count = 0;
    last_cmd_seq = seq;
    last_cmd_t_tx_us = read_u32(data + 14);

    store_display_payload(data + 3);
    return true;
}

bool UDP::handle_v4(const uint8_t* data, std::size_t length)
{
    if (data[0] != protocol_magic || data[1] != v4_command_version)
    {
        crc_err_count++;
        return false;
    }

    // 長さ整合はCRC検証より前に見る。uart_lenを信じてCRCを読むと、
    // 化けた長さでバッファ外を読みかねない。
    const std::size_t uart_len = data[6];
    if (uart_len < v4_min_uart_bytes || uart_len > v4_max_uart_bytes ||
        length != v4_header_size + uart_len + 2)
    {
        crc_err_count++;
        return false;
    }

    uint16_t seq = 0;
    if (!accept_command_frame(data, v4_header_size + uart_len, 8, seq))
    {
        return false;
    }

    // 中継してよいバイト列かをここで確定させる。壊れたフレームを開発ボードへ
    // 流すと、COBSの同期が崩れて後続の正常フレームまで無効になる。
    UartCommandFrame parsed;
    if (!parse_uart_command_frame(data + v4_header_size, uart_len, parsed))
    {
        crc_err_count++;
        return false;
    }

    const uint8_t flags = data[2];
    v2_auto_request = (flags & 0x01) != 0;
    last_estop_request = (flags & 0x02) != 0;

    std::memcpy(v4_uart, data + v4_header_size, uart_len);
    v4_uart_len = uart_len;
    v4_commands = parsed;

    // 参考値。制御には使わず、テレメトリのapplied_velocityへ返すだけ。
    v3_vx_mmps  = static_cast<int16_t>(read_u16(data + 14));
    v3_vy_mmps  = static_cast<int16_t>(read_u16(data + 16));
    v3_w_mradps = static_cast<int16_t>(read_u16(data + 18));

    Jetson_Packet pkt;
    std::memset(&pkt, 0, sizeof(pkt));
    pkt.lx_state = clamp_axis(static_cast<int>(std::lround(
        v3_vy_mmps * 127.0 / value::auto_display_full_mmps)));
    pkt.ly_state = clamp_axis(static_cast<int>(std::lround(
        v3_vx_mmps * 127.0 / value::auto_display_full_mmps)));
    pkt.rx_state = clamp_axis(static_cast<int>(std::lround(
        v3_w_mradps * 127.0 / value::auto_display_full_mradps)));
    decode_buttons(data[3], data[4], data[5], pkt);
    jetson_packet = pkt;

    last_cmd_v2 = false;
    last_cmd_v3 = false;
    last_cmd_v4 = true;
    last_cmd_seq = seq;
    last_cmd_t_tx_us = read_u32(data + 10);

    store_display_payload(data + 3);
    return true;
}

void UDP::update(const Controller_Packet& ctrl)
{
    const auto now = std::chrono::steady_clock::now();
    bool received = false;

    if (is_initialized)
    {
        while (true)
        {
            uint8_t buf[128]; // v4の最大長(20+64+2=86)に余裕を持たせる
            uint8_t cbuf[128];
            struct sockaddr_in src;
            std::memset(&src, 0, sizeof(src));

            struct iovec iov;
            iov.iov_base = buf;
            iov.iov_len  = sizeof(buf);

            struct msghdr msg;
            std::memset(&msg, 0, sizeof(msg));
            msg.msg_name       = &src;
            msg.msg_namelen    = sizeof(src);
            msg.msg_iov        = &iov;
            msg.msg_iovlen     = 1;
            msg.msg_control    = cbuf;
            msg.msg_controllen = sizeof(cbuf);

            const ssize_t n = recvmsg(fd, &msg, MSG_DONTWAIT);
            if (n < 0)
            {
                break;
            }

            // カーネルが記録した到着時刻（無ければ処理時刻で代用）
            int64_t arrival_ns = realtime_now_ns();
#ifdef SCM_TIMESTAMPNS
            for (struct cmsghdr* cmsg = CMSG_FIRSTHDR(&msg); cmsg != nullptr;
                 cmsg = CMSG_NXTHDR(&msg, cmsg))
            {
                if (cmsg->cmsg_level == SOL_SOCKET && cmsg->cmsg_type == SCM_TIMESTAMPNS)
                {
                    struct timespec ts;
                    std::memcpy(&ts, CMSG_DATA(cmsg), sizeof(ts));
                    arrival_ns = static_cast<int64_t>(ts.tv_sec) * 1000000000LL + ts.tv_nsec;
                    break;
                }
            }
#endif

            if (!accept_source(src, now))
            {
                foreign_drop_count++;
                continue;
            }

            // legacyだけ長さで判定する（先頭バイトはスティック値なので
            // magicと衝突しうる）。それ以外はmagic+versionで判定するので、
            // v4のように長さが可変でも取り違えない。
            bool ok = false;
            if (n == static_cast<ssize_t>(payload_size))
            {
                handle_legacy(buf);
                ok = true;
            }
            else if (n >= 4 && buf[0] == protocol_magic &&
                     buf[1] == v2_command_version &&
                     n == static_cast<ssize_t>(v2_command_size))
            {
                ok = handle_v2(buf);
            }
            else if (n >= 4 && buf[0] == protocol_magic &&
                     buf[1] == v3_command_version &&
                     n == static_cast<ssize_t>(v3_command_size))
            {
                ok = handle_v3(buf);
            }
            else if (n >= static_cast<ssize_t>(v4_header_size + v4_min_uart_bytes + 2) &&
                     buf[0] == protocol_magic &&
                     buf[1] == v4_command_version)
            {
                ok = handle_v4(buf, static_cast<std::size_t>(n));
            }
            else
            {
                size_err_count++;
                if (now - last_size_warn >= std::chrono::seconds(1))
                {
                    std::cerr << "Unexpected UDP payload size: " << n
                              << " (total " << size_err_count << ")" << std::endl;
                    last_size_warn = now;
                }
            }

            if (ok)
            {
                recv_count++;
                rate_window_count++;
                received = true;
                last_cmd_arrival_ns = arrival_ns;
            }
        }

        if (received)
        {
            last_recv_time = now;
        }
    }

    const auto elapsed = now - last_recv_time;
    const int age_ms = static_cast<int>(
        std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count());

    const bool manual_auto_requested =
        value::read_button(ctrl, value::auto_control_on);
    const bool structured_command = last_cmd_v2 || last_cmd_v3 || last_cmd_v4;
    const bool jetson_auto_requested =
        jetson_packet.l2_state ||
        (structured_command && v2_auto_request);
    // Always consume DISARM; short-circuiting this call leaves stop_pending latched.
    const bool remote_auto_allowed = remote_navigation.allow_auto(jetson_auto_requested);
    const bool auto_requested = (jetson_auto_requested || manual_auto_requested)
        && remote_auto_allowed;
    link_safety.update(
        received,
        auto_requested,
        structured_command && last_estop_request,
        age_ms);
    // Link health and motion authorization are separate: a healthy disarmed
    // handshake still has a live telemetry link.
    is_alive = is_initialized && age_ms < value::sys::controlled_stop_ms;
    auto_mode = link_safety.motion_allowed() && auto_requested;
}

void UDP::send_telemetry(const Controller_Packet& applied, bool mu3_alive, bool uart_open,
                         const struct packet& motors,
                         double applied_vx_mps, double applied_vy_mps,
                         double applied_w_radps)
{
    if (!is_initialized || !source_locked)
    {
        return;
    }

    telemetry_tick++;
    if (value::sys::telemetry_interval_loops > 1 &&
        (telemetry_tick % value::sys::telemetry_interval_loops) != 0)
    {
        return;
    }

    const auto now = std::chrono::steady_clock::now();
    if (now - rate_window_start >= std::chrono::seconds(1))
    {
        last_rate_hz = static_cast<uint8_t>(std::min<uint32_t>(255, rate_window_count));
        rate_window_count = 0;
        rate_window_start = now;
    }

    // 常に拡張形(48B)で送る: 各輪指令値はv2/手動運転の診断にも有用。
    // 適用速度(vx/vy/w)はv3物理経路でないループでは0になる。
    // 旧Jetsonブリッジ互換が必要な場合のみ基本形(32B)へ戻すこと。
    const bool extended = true;
    const std::size_t frame_size = extended ? telemetry_ext_size : telemetry_size;

    uint8_t p[telemetry_ext_size];
    std::memset(p, 0, sizeof(p));
    p[0] = protocol_magic;
    p[1] = extended ? telemetry_ext_version : telemetry_version;

    uint8_t flags = 0;
    if (auto_mode)       flags |= 0x01;
    if (mu3_alive)       flags |= 0x02;
    if (is_alive)        flags |= 0x04;
    if (uart_open)       flags |= 0x08;
    if (estop_active())  flags |= 0x10;
    p[2] = flags;

    uint8_t flags2 = 0;
    if (last_cmd_v2) flags2 |= 0x01;
    if (last_cmd_v3) flags2 |= 0x02;
    if (link_safety.quality_degraded()) flags2 |= 0x04;
    if (link_safety.stop_required())    flags2 |= 0x08;
    if (link_safety.fault_latched())    flags2 |= 0x10;
    if (link_safety.rearm_required())   flags2 |= 0x20;
    if (last_cmd_v4) flags2 |= 0x40;
    p[3] = flags2;

    const bool structured_command = last_cmd_v2 || last_cmd_v3 || last_cmd_v4;
    write_u16(p + 4, pi_seq++);
    write_u16(p + 6, structured_command ? last_cmd_seq : 0);
    write_u32(p + 8, structured_command ? last_cmd_t_tx_us : 0);

    uint32_t hold_us = 0;
    if (recv_count > 0 && last_cmd_arrival_ns > 0)
    {
        const int64_t d = (realtime_now_ns() - last_cmd_arrival_ns) / 1000;
        if (d > 0)
        {
            hold_us = d > 0xFFFFFFFFLL ? 0xFFFFFFFFu : static_cast<uint32_t>(d);
        }
    }
    write_u32(p + 12, hold_us);
    write_u32(p + 16, static_cast<uint32_t>(recv_count));
    write_u16(p + 20, static_cast<uint16_t>(crc_err_count));
    write_u16(p + 22, static_cast<uint16_t>(stale_drops));
    p[24] = static_cast<uint8_t>(applied.lx_state);
    p[25] = static_cast<uint8_t>(applied.ly_state);
    p[26] = static_cast<uint8_t>(applied.rx_state);
    p[27] = static_cast<uint8_t>(applied.ry_state);
    p[28] = last_rate_hz;
    // Backward-compatible use of reserved bytes; covered by the existing CRC.
    p[29] = 0x80 | remote_navigation.slot;

    if (extended)
    {
        // 適用した機体速度（比例縮小後）と実際にUARTへ送る各輪値
        write_u16(p + 30, static_cast<uint16_t>(static_cast<int16_t>(
            std::lround(applied_vx_mps * 1000.0))));
        write_u16(p + 32, static_cast<uint16_t>(static_cast<int16_t>(
            std::lround(applied_vy_mps * 1000.0))));
        write_u16(p + 34, static_cast<uint16_t>(static_cast<int16_t>(
            std::lround(applied_w_radps * 1000.0))));
        write_u16(p + 36, static_cast<uint16_t>(motors.m1));
        write_u16(p + 38, static_cast<uint16_t>(motors.m2));
        write_u16(p + 40, static_cast<uint16_t>(motors.m3));
        write_u16(p + 42, static_cast<uint16_t>(motors.m4));
        write_u16(p + 44, remote_navigation.sequence);
        write_u16(p + 46, crc16_ccitt(p, 46));
    }
    else
    {
        write_u16(p + 30, crc16_ccitt(p, 30));
    }

    // ノンブロッキング送信。失敗（バッファ満杯等）は次回送信に任せる
    sendto(fd, p, frame_size, MSG_DONTWAIT,
           reinterpret_cast<const struct sockaddr*>(&source_addr),
           sizeof(source_addr));
}

int UDP::last_receive_age_ms() const
{
    const auto elapsed = std::chrono::steady_clock::now() - last_recv_time;
    return static_cast<int>(
        std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count());
}
