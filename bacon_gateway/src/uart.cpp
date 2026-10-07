#include <iostream>
#include <vector>
#include <fcntl.h>
#include <unistd.h>
#include <termios.h>
#include <cstring>
#include <cerrno>
#include <sys/ioctl.h>
#include <sys/file.h>
#include <chrono>
#include <limits>
#include <thread>

#include "cobs.hpp"
#include "packet.hpp"
#include "uart.hpp"
#include "value.hpp"

std::size_t expand_packet_commands(const packet& data,
                                   uint8_t ids[robomas_command_count],
                                   int16_t values[robomas_command_count],
                                   bool include_arm, bool include_gm)
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

    std::size_t count = 0;
    for (std::size_t i = 0; i < robomas_command_count; ++i)
    {
        if ((!include_arm && frame_ids[i] == value::cmd_arm) ||
            (!include_gm && frame_ids[i] == value::cmd_gm)) continue;
        ids[count] = frame_ids[i];
        values[count] = frame_values[i];
        ++count;
    }
    return count;
}

std::vector<uint8_t> encode_robomas_frame(const uint8_t* ids,
                                          const int16_t* values,
                                          std::size_t count)
{
    if (ids == nullptr || values == nullptr || count == 0)
    {
        return {};
    }

    if (count > (std::numeric_limits<std::size_t>::max() - 2) / 3)
    {
        return {};
    }
    const std::size_t capacity = cobs_encoded_max_size(count * 3);
    if (capacity == 0)
    {
        return {};
    }

    std::vector<uint8_t> payload;
    payload.reserve(count * 3);
    for (std::size_t i = 0; i < count; ++i)
    {
        payload.push_back(ids[i]);
        const uint16_t wire_value = static_cast<uint16_t>(values[i]);
        payload.push_back(static_cast<uint8_t>(wire_value >> 8));
        payload.push_back(static_cast<uint8_t>(wire_value & 0xFF));
    }

    std::vector<uint8_t> encoded(capacity, 0);
    const std::size_t encoded_size =
        encode_cobs_bounded(payload.data(), payload.size(), encoded.data(),
                            encoded.size());
    encoded.resize(encoded_size);
    return encoded;
}

UART::UART(const std::string & device_name, int baud_rate)
    : fd(-1), is_initialized(false), sent_frames(0), busy_skips(0),
      exclusive(false), needs_resynchronization(false)
{
    fd = open(device_name.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK | O_CLOEXEC);

    if(fd < 0)
    {
        std::cerr << "エラー: シリアルポート" << device_name << "開けません" << std::strerror(errno) << std::endl;
        return;
    }

    // The launcher check alone races other startups. flock also protects
    // cooperating privileged processes, which can bypass TIOCEXCL.
    if (flock(fd, LOCK_EX | LOCK_NB) != 0 || ioctl(fd, TIOCEXCL) != 0)
    {
        std::cerr << "エラー: UART排他アクセスに失敗: "
                  << std::strerror(errno) << std::endl;
        close_port();
        return;
    }
    exclusive = true;

    struct termios options;
    std::memset(&options, 0, sizeof(options));

    if (tcgetattr(fd, &options) != 0) 
    {
        std::cerr << "エラー: tcgetattr に失敗: " << std::strerror(errno) << std::endl;
        close_port();
        return;
    }

    if (baud_rate == B0 || cfsetispeed(&options, baud_rate) != 0 ||
        cfsetospeed(&options, baud_rate) != 0)
    {
        std::cerr << "エラー: UARTボーレート設定に失敗" << std::endl;
        close_port();
        return;
    }
    cfmakeraw(&options);

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
        close_port();
        return;
    }

    is_initialized = true;
    std::cout << "シリアルポート" << device_name << "接続完了" << std::endl;
}

UART::~UART()
{
    close_port();
}

void UART::close_port()
{
    if (fd >= 0)
    {
        // On PTYs the master can outlive this slave; release kernel
        // exclusivity explicitly so a subsequent owner can reopen it.
        if (exclusive) (void)ioctl(fd, TIOCNXCL);
        (void)close(fd);
    }
    fd = -1;
    is_initialized = false;
    exclusive = false;
}

bool UART::is_open() const
{
    return is_initialized && fd >= 0;
}

bool UART::uart_send(const packet & data, bool include_arm, bool include_gm)
{
    if(!is_open())
    {
        return false;
    }

    uint8_t ids[robomas_command_count];
    int16_t values[robomas_command_count];
    const std::size_t count = expand_packet_commands(data, ids, values, include_arm, include_gm);

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
    if(!is_open() || data == nullptr || length == 0)
    {
        std::cerr << "エラー: シリアルポートが初期化されていません" << std::endl;
        return false;
    }

    int queued_bytes = 0;
    if (ioctl(fd, TIOCOUTQ, &queued_bytes) != 0)
    {
        if (errno != EINTR) close_port();
        return false;
    }
    if (queued_bytes > 0)
    {
        // 115200bpsでは500Hzの全パケットを流し切れないため、古い送信待ちを積まない。
        // 落としたフレームの間、Dev Boardは直前の指令を保持し続けるので、
        // 回数はテレメトリ/状態表示から見えるようにしておく。
        ++busy_skips;
        return false;
    }

    if (needs_resynchronization)
    {
        const uint8_t delimiter = 0;
        const ssize_t written = write(fd, &delimiter, 1);
        if (written != 1)
        {
            if (written < 0 && errno != EAGAIN && errno != EWOULDBLOCK &&
                errno != EINTR) close_port();
            ++busy_skips;
            return false;
        }
        needs_resynchronization = false;
    }

    // Never wait for space in the driver from the control loop. A partial
    // write is a failed command, and the next command must start at a fresh
    // COBS boundary rather than merge with its incomplete predecessor.
    const ssize_t bytes_written = write(fd, data, length);
    if (bytes_written < 0)
    {
        if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR)
        {
            ++busy_skips;
        }
        else
        {
            std::cerr << "エラー: データ送信に失敗: "
                      << std::strerror(errno) << std::endl;
            close_port();
        }
        return false;
    }
    if (static_cast<std::size_t>(bytes_written) != length)
    {
        needs_resynchronization = bytes_written > 0;
        ++busy_skips;
        return false;
    }

    ++sent_frames;
    return true;
}

bool UART::flush_output(unsigned int timeout_ms)
{
    if (!is_open())
    {
        return false;
    }
    const auto deadline = std::chrono::steady_clock::now() +
        std::chrono::milliseconds(timeout_ms);
    do
    {
        int queued_bytes = 0;
        if (ioctl(fd, TIOCOUTQ, &queued_bytes) != 0)
        {
            if (errno != EINTR) close_port();
            return false;
        }
        if (queued_bytes == 0)
        {
#if defined(TIOCSERGETLSR) && defined(TIOCSER_TEMT)
            // Native UART drivers can expose the hardware shift-register
            // state. An empty software queue alone may leave FIFO bytes on
            // the wire. PTYs and many USB adapters do not provide this ioctl.
            int line_status = 0;
            if (ioctl(fd, TIOCSERGETLSR, &line_status) == 0)
            {
                if (line_status & TIOCSER_TEMT) return true;
            }
            else if (errno == ENOTTY || errno == EINVAL || errno == EOPNOTSUPP)
            {
                return true;
            }
            else
            {
                if (errno != EINTR) close_port();
                return false;
            }
#else
            return true;
#endif
        }
        if (std::chrono::steady_clock::now() >= deadline) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    while (true);
    return false;
}
