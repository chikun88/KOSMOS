#include "passthrough.hpp"

#include "cobs.hpp"

bool UartCommandFrame::find(uint8_t id, int16_t& out) const
{
    bool found = false;
    for (std::size_t i = 0; i < count; ++i)
    {
        if (command[i] == id)
        {
            out = value[i];
            found = true;
        }
    }
    return found;
}

bool UartCommandFrame::contains(uint8_t id) const
{
    int16_t ignored = 0;
    return find(id, ignored);
}

bool parse_uart_command_frame(const uint8_t* encoded, std::size_t length,
                              UartCommandFrame& out)
{
    out.count = 0;

    // 最小は1コマンド(3バイト)のCOBS = 5バイト。
    if (encoded == nullptr || length < 5 ||
        length > UartCommandFrame::max_encoded)
    {
        return false;
    }

    // encode_cobs は必ず区切りの0x00で終わる。途中に0x00があるフレームは
    // 2個以上のパケットが連結されているので、単一フレームとしては受けない。
    if (encoded[length - 1] != 0)
    {
        return false;
    }
    for (std::size_t i = 0; i + 1 < length; ++i)
    {
        if (encoded[i] == 0)
        {
            return false;
        }
    }

    // decode_cobs は区切りの0x00を含めると失敗するので、手前までを渡す。
    uint8_t decoded[UartCommandFrame::max_encoded];
    const std::size_t decoded_size = decode_cobs(encoded, length - 1, decoded);

    if (decoded_size == 0 || decoded_size % 3 != 0 ||
        decoded_size / 3 > UartCommandFrame::max_commands)
    {
        return false;
    }

    out.count = decoded_size / 3;
    for (std::size_t i = 0; i < out.count; ++i)
    {
        const uint8_t* triplet = decoded + i * 3;
        out.command[i] = triplet[0];
        out.value[i] = static_cast<int16_t>(
            (static_cast<uint16_t>(triplet[1]) << 8) |
             static_cast<uint16_t>(triplet[2]));
    }

    return true;
}
