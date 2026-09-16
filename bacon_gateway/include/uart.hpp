#pragma once

#include <string>
#include <vector>
#include <cstddef>
#include <cstdint>

struct packet;

// ロボマス開発ボードへ送る完全フレームのコマンド数（omni4 + 機構5）
constexpr std::size_t robomas_command_count = 9;

// packet を UARTフレームの並び順そのままの [コマンドID, 値] 列へ展開する。
// パススルー中に「egg8が送っていないコマンドだけを補完する」ためには、
// 完全フレームの構成を main.cpp からも同じ定義で参照できる必要がある。
// 戻り値は書き込んだ要素数（= robomas_command_count）。
std::size_t expand_packet_commands(const packet& data,
                                   uint8_t ids[robomas_command_count],
                                   int16_t values[robomas_command_count]);

// [コマンドID, 値] 列を [ID, 値上位, 値下位] へ並べてCOBSで包む。
// 開発ボードへ出す1フレームぶんのバイト列（末尾の区切り0x00を含む）。
std::vector<uint8_t> encode_robomas_frame(const uint8_t* ids,
                                          const int16_t* values,
                                          std::size_t count);

class UART
{
    public:
        UART(const std::string& device_name, int baud_rate);
        ~UART();

        bool is_open() const;
        bool uart_send(const packet& data);

        // COBSエンコード済みのバイト列をそのまま送る。
        // 自動走行中はegg8が組んだフレームを1バイトも書き換えずに中継する
        // ため、ここでは内容を解釈しない。補完フレームを続けて出す場合も
        // 呼び出し側で連結して1回のwriteにすること: 下のbusy_skipsガードは
        // フレーム単位で落とすので、分割すると2本目だけが落ちて足回りと
        // 機構でコマンドの世代がずれる。
        bool uart_send_encoded(const uint8_t* data, std::size_t length);

        bool flush_output();

        // A motor frame skipped because the previous one had not finished
        // shifting out leaves the Dev Board holding its last command for
        // another loop period.  That is a silently stale drive command, so the
        // count has to be observable rather than inferred from behaviour.
        uint64_t sent_count() const { return sent_frames; }
        uint64_t busy_skip_count() const { return busy_skips; }

    private:
        int fd;
        bool is_initialized;
        uint64_t sent_frames;
        uint64_t busy_skips;

        bool send_data(const uint8_t* data, std::size_t length);
};
