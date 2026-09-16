#include "cobs.hpp"
#include "packet.hpp"
#include "passthrough.hpp"
#include "uart.hpp"
#include "value.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <utility>
#include <vector>

namespace
{
    void require(bool condition, const char* message)
    {
        if (!condition)
        {
            std::cerr << "test_uart_passthrough: " << message << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }

    // Jetson側 robomas_uart.py が組むのと同じ手順でフレームを作る。
    std::vector<uint8_t> make_frame(
        const std::vector<std::pair<uint8_t, int16_t>>& commands)
    {
        std::vector<uint8_t> ids;
        std::vector<int16_t> values;
        for (const auto& entry : commands)
        {
            ids.push_back(entry.first);
            values.push_back(entry.second);
        }
        return encode_robomas_frame(ids.data(), values.data(), ids.size());
    }
}

int main()
{
    // --- 1. COBSフレームは常に 生バイト数+2 で、末尾が区切りの0 ---
    const std::vector<uint8_t> four_wheels = make_frame({
        {value::cmd_omni[0], 1000}, {value::cmd_omni[1], -1000},
        {value::cmd_omni[2], 0},    {value::cmd_omni[3], 32767}});
    require(four_wheels.size() == 4 * 3 + 2,
            "COBS frame must be the payload size plus two");
    require(four_wheels.back() == 0, "COBS frame must end with the delimiter");
    for (std::size_t i = 0; i + 1 < four_wheels.size(); ++i)
    {
        require(four_wheels[i] != 0,
                "COBS must not leave a zero inside the frame");
    }

    // --- 2. 解析すると元のコマンドID・値へ戻る（値0や負値も含めて） ---
    UartCommandFrame parsed;
    require(parse_uart_command_frame(four_wheels.data(), four_wheels.size(), parsed),
            "a well-formed frame must parse");
    require(parsed.count == 4, "four commands must be recovered");
    int16_t value = 0;
    require(parsed.find(value::cmd_omni[0], value) && value == 1000, "m1 value");
    require(parsed.find(value::cmd_omni[1], value) && value == -1000, "m2 value");
    require(parsed.find(value::cmd_omni[2], value) && value == 0, "m3 value");
    require(parsed.find(value::cmd_omni[3], value) && value == 32767, "m4 value");
    require(!parsed.contains(value::cmd_arm),
            "a wheels-only frame must not claim the ARM command");

    // --- 3. 9コマンドの完全フレームも往復する ---
    struct packet full;
    std::memset(&full, 0, sizeof(full));
    full.m1 = 1; full.m2 = -2; full.m3 = 3; full.m4 = -4;
    full.m5 = 144; full.gm = 500; full.m6 = 2000; full.m7 = -2000;
    full.gpio = value::gpio_on;

    uint8_t ids[robomas_command_count];
    int16_t values[robomas_command_count];
    const std::size_t count = expand_packet_commands(full, ids, values);
    require(count == robomas_command_count, "a full frame carries nine commands");

    const std::vector<uint8_t> full_frame =
        encode_robomas_frame(ids, values, count);
    UartCommandFrame full_parsed;
    require(parse_uart_command_frame(full_frame.data(), full_frame.size(), full_parsed),
            "the full frame must parse");
    require(full_parsed.count == robomas_command_count, "nine commands recovered");
    for (std::size_t i = 0; i < count; ++i)
    {
        require(full_parsed.find(ids[i], value) && value == values[i],
                "every command must round-trip through COBS");
    }

    // --- 4. 補完集合: egg8が送っていないコマンドだけを bacon6 が送る ---
    std::size_t missing = 0;
    for (std::size_t i = 0; i < count; ++i)
    {
        if (!parsed.contains(ids[i]))
        {
            ++missing;
        }
    }
    require(missing == robomas_command_count - 4,
            "a wheels-only passthrough must leave exactly the five mechanisms");

    // egg8がすべて送ってきたら bacon6 は何も足さない（将来の全面委譲）。
    std::size_t missing_when_full = 0;
    for (std::size_t i = 0; i < count; ++i)
    {
        if (!full_parsed.contains(ids[i]))
        {
            ++missing_when_full;
        }
    }
    require(missing_when_full == 0,
            "a complete passthrough frame must need no local complement");

    // --- 5. 壊れたフレームは中継しない ---
    UartCommandFrame rejected;
    require(!parse_uart_command_frame(four_wheels.data(), four_wheels.size() - 1, rejected),
            "a truncated frame must be rejected (no trailing delimiter)");

    // 生4バイト -> COBS 6バイト。3の倍数でないので1コマンドに割り切れない。
    const uint8_t payload_of_four[4] = {0x07, 0x01, 0x02, 0x03};
    uint8_t four_byte_frame[6] = {0};
    const std::size_t four_byte_size =
        encode_cobs(payload_of_four, sizeof(payload_of_four), four_byte_frame);
    require(four_byte_size == sizeof(four_byte_frame), "COBS size for four bytes");
    require(!parse_uart_command_frame(four_byte_frame, four_byte_size, rejected),
            "a payload that is not a whole number of commands must be rejected");

    std::vector<uint8_t> two_frames = four_wheels;
    two_frames.insert(two_frames.end(), four_wheels.begin(), four_wheels.end());
    require(!parse_uart_command_frame(two_frames.data(), two_frames.size(), rejected),
            "two concatenated frames must not be accepted as one");

    require(!parse_uart_command_frame(nullptr, 0, rejected),
            "an empty frame must be rejected");

    // 上限を超える長さは、バッファを読む前に長さだけで弾かれること。
    std::vector<uint8_t> oversized(UartCommandFrame::max_encoded + 1, 0x01);
    oversized.back() = 0;
    require(!parse_uart_command_frame(oversized.data(), oversized.size(), rejected),
            "an oversized frame must be rejected");

    // 未知のコマンドIDは素通しできる（将来egg8が新しい機構を足すため）。
    const std::vector<uint8_t> unknown = make_frame({{200, -1}});
    UartCommandFrame unknown_parsed;
    require(parse_uart_command_frame(unknown.data(), unknown.size(), unknown_parsed),
            "an unknown command id must still relay");
    require(unknown_parsed.count == 1 && unknown_parsed.command[0] == 200,
            "the unknown id must be reported so it is not duplicated locally");

    return EXIT_SUCCESS;
}
