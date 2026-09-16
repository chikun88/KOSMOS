#include <iostream>
#include <vector>
#include <fcntl.h>
#include <unistd.h>
#include <termios.h>
#include <cstring>
#include <cerrno>
#include <sys/ioctl.h>

#include "cobs.hpp"
#include "packet.hpp"
#include "uart.hpp"
#include "value.hpp"

std::size_t expand_packet_commands(const packet& data,
                                   uint8_t ids[robomas_command_count],
                                   int16_t values[robomas_command_count])
{
    // 並び順は開発ボードへ流すフレームそのもの。ここを変えるときは
    // Jetson側 robomas_uart.py の ROBOMAS_FRAME_ORDER も合わせること。
    const uint8_t frame_ids[robomas_command_count] = {
        value::cmd_omni[0], value::cmd_omni[1],
        value::cmd_omni[2], value::cmd_omni[3],
        value::cmd_arm, value::cmd_gm,
        value::cmd_updown, value::cmd_collect, value::cmd_gpio};
    const int16_t frame_values[robomas_command_count] = {
        data.m1, data.m2, data.m3, data.m4,
        data.m5, data.gm, data.m6, data.m7,
        static_cast<int16_t>(data.gpio)};

    for (std::size_t i = 0; i < robomas_command_count; ++i)
    {
        ids[i] = frame_ids[i];
        values[i] = frame_values[i];
    }
    return robomas_command_count;
}

std::vector<uint8_t> encode_robomas_frame(const uint8_t* ids,
                                          const int16_t* values,
                                          std::size_t count)
{
    if (ids == nullptr || values == nullptr || count == 0)
    {
        return {};
    }

    std::vector<uint8_t> payload;
    payload.reserve(count * 3);
    for (std::size_t i = 0; i < count; ++i)
    {
        payload.push_back(ids[i]);
        payload.push_back(static_cast<uint8_t>((values[i] >> 8) & 0xFF));
        payload.push_back(static_cast<uint8_t>(values[i] & 0xFF));
    }

    std::vector<uint8_t> encoded(payload.size() + 2, 0);
    const std::size_t encoded_size =
        encode_cobs(payload.data(), payload.size(), encoded.data());
    encoded.resize(encoded_size);
    return encoded;
}

UART::UART(const std::string & device_name, int baud_rate)
    : fd(-1), is_initialized(false), sent_frames(0), busy_skips(0)
{
    fd = open(device_name.c_str(), O_RDWR | O_NOCTTY);

    if(fd < 0)
    {
        std::cerr << "エラー: シリアルポート" << device_name << "開けません" << std::strerror(errno) << std::endl;
        return;
    }

    struct termios options;
    std::memset(&options, 0, sizeof(options));

    if (tcgetattr(fd, &options) != 0) 
    {
        std::cerr << "エラー: tcgetattr に失敗: " << std::strerror(errno) << std::endl;
        close(fd);
        fd = -1;
        return;
    }

    cfsetispeed(& options, baud_rate);
    cfsetospeed(& options, baud_rate);

    // これでできるがちょい不安なので書く
    // cfmakeraw(&options);

    options.c_cflag &=  ~PARENB;
    options.c_cflag &= ~CSTOPB;
    options.c_cflag &= ~CSIZE;
    options.c_cflag |= CS8;
    options.c_cflag &= ~CRTSCTS;
    options.c_cflag |= CREAD | CLOCAL;
    
    options.c_lflag &= ~ICANON;
    options.c_lflag &= ~ECHO;
    options.c_lflag &= ~ECHOE;
    options.c_lflag &= ~ECHONL;
    options.c_lflag &= ~ISIG;

    options.c_iflag &= ~(IXON | IXOFF | IXANY);
    options.c_iflag &= ~(IGNBRK | BRKINT | PARMRK | ISTRIP | INLCR | IGNCR | ICRNL);

    options.c_oflag &= ~OPOST;
    options.c_oflag &= ~ONLCR;

    options.c_cc[VMIN] = 0;
    options.c_cc[VTIME] = 0;

    if (tcsetattr(fd, TCSANOW, &options) != 0) 
    {
        std::cerr << "エラー: tcsetattr に失敗: " << std::strerror(errno) << std::endl;
        close(fd);
        fd = -1;
        return;
    }

    is_initialized = true;
    std::cout << "シリアルポート" << device_name << "接続完了" << std::endl;
}

UART::~UART()
{
    if(is_open())
    {
        close(fd);
        fd = -1;
        is_initialized = false;
    }

    std::cout << "シリアルポート切断" << std::endl;
}

bool UART::is_open() const
{
    return is_initialized && fd >= 0;
}

bool UART::uart_send(const packet & data)
{
    if(!is_open())
    {
        return false;
    }

    uint8_t ids[robomas_command_count];
    int16_t values[robomas_command_count];
    const std::size_t count = expand_packet_commands(data, ids, values);

    const std::vector<uint8_t> frame = encode_robomas_frame(ids, values, count);
    return send_data(frame.data(), frame.size());
}

bool UART::uart_send_encoded(const uint8_t* data, std::size_t length)
{
    if (data == nullptr || length == 0)
    {
        return false;
    }
    return send_data(data, length);
}

bool UART::send_data(const uint8_t* data, std::size_t length)
{
    if(!is_open())
    {
        std::cerr << "エラー: シリアルポートが初期化されていません" << std::endl;
        return false;
    }

    int queued_bytes = 0;
    if (ioctl(fd, TIOCOUTQ, &queued_bytes) == 0 && queued_bytes > 0)
    {
        // 115200bpsでは500Hzの全パケットを流し切れないため、古い送信待ちを積まない。
        // 落としたフレームの間、Dev Boardは直前の指令を保持し続けるので、
        // 回数はテレメトリ/状態表示から見えるようにしておく。
        ++busy_skips;
        return false;
    }

    const ssize_t bytes_written = write(fd, data, length);

    if(bytes_written < 0)
    {
        std::cerr << "エラー: データ送信に失敗: "
                  << std::strerror(errno) << std::endl;
        return false;
    }
    if (static_cast<size_t>(bytes_written) != length)
    {
        std::cerr << "エラー: UART部分送信: " << bytes_written
                  << "/" << length << " bytes" << std::endl;
        return false;
    }

    ++sent_frames;
    return true;
}

bool UART::flush_output()
{
    if (!is_open())
    {
        return false;
    }
    if (tcdrain(fd) != 0)
    {
        std::cerr << "エラー: UART出力flushに失敗: "
                  << std::strerror(errno) << std::endl;
        return false;
    }
    return true;
}
