#include <iostream>
#include <fcntl.h>
#include <unistd.h>
#include <termios.h>
#include <cstring>
#include <cerrno>
#include <array>
#include <algorithm>
#include <sys/file.h>
#include <sys/ioctl.h>
#include <poll.h>

#include "MU3.hpp"
#include "value.hpp"

namespace
{
    constexpr size_t max_data_length = 127;
    constexpr size_t max_rx_buffer = 4096;

    int hex_digit(char character)
    {
        if (character >= '0' && character <= '9') return character - '0';
        if (character >= 'A' && character <= 'F') return character - 'A' + 10;
        if (character >= 'a' && character <= 'f') return character - 'a' + 10;
        return -1;
    }
}

MU3::MU3(const char* device_path)
    : fd(-1), exclusive(false), needs_resynchronization(false)
{
    if (device_path == nullptr || device_path[0] == '\0') return;
    fd = open(device_path, O_RDWR | O_NOCTTY | O_NONBLOCK | O_CLOEXEC);
    if (fd == -1)
    {
        std::cerr << "[エラー] MU3ポートが開けません: " << device_path
                  << ": " << std::strerror(errno) << std::endl;
        return;
    }
    if (flock(fd, LOCK_EX | LOCK_NB) != 0 || ioctl(fd, TIOCEXCL) != 0)
    {
        std::cerr << "[エラー] MU3排他アクセスに失敗: "
                  << std::strerror(errno) << std::endl;
        close_port();
        return;
    }
    exclusive = true;

    struct termios options;
    if (tcgetattr(fd, &options) != 0)
    {
        std::cerr << "[エラー] MU3 tcgetattrに失敗: "
                  << std::strerror(errno) << std::endl;
        close_port();
        return;
    }
    cfmakeraw(&options);
    options.c_cflag &= ~(CSIZE | PARENB | CSTOPB | CRTSCTS);
    options.c_cflag |= CS8 | CLOCAL | CREAD;
    options.c_cc[VMIN] = 0;
    options.c_cc[VTIME] = 0;
    if (cfsetispeed(&options, value::sys::mu3_baud) != 0 ||
        cfsetospeed(&options, value::sys::mu3_baud) != 0 ||
        tcsetattr(fd, TCSANOW, &options) != 0 ||
        tcflush(fd, TCIOFLUSH) != 0)
    {
        std::cerr << "[エラー] MU3シリアル設定に失敗: "
                  << std::strerror(errno) << std::endl;
        close_port();
    }
}

MU3::~MU3()
{
    close_port();
}

void MU3::close_port()
{
    if (fd >= 0)
    {
        if (exclusive) (void)ioctl(fd, TIOCNXCL);
        (void)close(fd);
    }
    fd = -1;
    exclusive = false;
}

bool MU3::is_open() const
{
    return fd >= 0;
}

int MU3::send(const uint8_t* data, size_t length)
{
    if (!is_open() || (data == nullptr && length != 0) ||
        length > max_data_length) return -1;

    // No receive flush: outgoing traffic must not discard a controller stop
    // packet that arrived at the same time.
    constexpr char hex[] = "0123456789ABCDEF";
    std::array<char, 2 + 5 + 2 * max_data_length + 2> payload{};
    size_t position = 0;
    if (needs_resynchronization)
    {
        payload[position++] = '\r';
        payload[position++] = '\n';
    }
    payload[position++] = '@';
    payload[position++] = 'D';
    payload[position++] = 'T';
    const size_t hex_length = length * 2;
    payload[position++] = hex[hex_length >> 4];
    payload[position++] = hex[hex_length & 0x0F];
    for (size_t i = 0; i < length; ++i)
    {
        payload[position++] = hex[data[i] >> 4];
        payload[position++] = hex[data[i] & 0x0F];
    }
    payload[position++] = '\r';
    payload[position++] = '\n';

    const ssize_t written = write(fd, payload.data(), position);
    if (written == static_cast<ssize_t>(position))
    {
        needs_resynchronization = false;
        return 0;
    }
    if (written > 0) needs_resynchronization = true;
    if (written < 0 && errno != EINTR && errno != EAGAIN &&
        errno != EWOULDBLOCK) close_port();
    return -1;
}

bool MU3::discard_input()
{
    if (!is_open()) return false;
    rx_buffer.clear();
    if (tcflush(fd, TCIFLUSH) != 0)
    {
        if (errno != EINTR) close_port();
        return false;
    }
    return true;
}

int MU3::receive(uint8_t* out_data, size_t expected_length)
{
    if (!is_open() || out_data == nullptr || expected_length == 0 ||
        expected_length > max_data_length) return -1;

    struct pollfd status { fd, POLLIN, 0 };
    const int readiness = poll(&status, 1, 0);
    if ((readiness < 0 && errno != EINTR) ||
        (readiness > 0 && (status.revents & (POLLHUP | POLLERR | POLLNVAL))))
    {
        close_port();
        return -1;
    }

    // Bound work even if a broken radio floods the serial device. Records
    // remain in arrival order: stop, release and new-tap edges must survive
    // a backlog even when the caller ultimately applies the latest axes.
    std::array<char, 256> buffer{};
    for (size_t count = 0; count < max_rx_buffer / buffer.size(); ++count)
    {
        // Do not erase already-received stop/button edges to make room for
        // later traffic. The caller drains these records in arrival order.
        if (rx_buffer.size() > max_rx_buffer - buffer.size()) break;
        const ssize_t received = read(fd, buffer.data(), buffer.size());
        if (received > 0)
        {
            rx_buffer.append(buffer.data(), static_cast<size_t>(received));
            continue;
        }
        if (received < 0 && errno != EAGAIN && errno != EWOULDBLOCK &&
            errno != EINTR)
        {
            close_port();
            return -1;
        }
        break;
    }

    while (!rx_buffer.empty())
    {
        const size_t header = rx_buffer.find("*DR=");
        if (header == std::string::npos)
        {
            // Preserve a split header such as "*DR" across read calls.
            if (rx_buffer.size() > 3) rx_buffer.erase(0, rx_buffer.size() - 3);
            break;
        }
        if (header > 0) rx_buffer.erase(0, header);
        if (rx_buffer.size() < 6) break;

        const int high = hex_digit(rx_buffer[4]);
        const int low = hex_digit(rx_buffer[5]);
        if (high < 0 || low < 0)
        {
            rx_buffer.erase(0, 1);
            continue;
        }
        const size_t hex_length = static_cast<size_t>((high << 4) | low);
        if (hex_length == 0 || hex_length % 2 != 0)
        {
            rx_buffer.erase(0, 1);
            continue;
        }
        const size_t record_length = 6 + hex_length + 2;
        const size_t following_header = rx_buffer.find("*DR=", 4);
        if (following_header != std::string::npos &&
            following_header < record_length)
        {
            rx_buffer.erase(0, following_header);
            continue;
        }
        if (rx_buffer.size() < record_length) break;

        bool valid = hex_length == expected_length * 2 &&
            rx_buffer[record_length - 2] == '\r' &&
            rx_buffer[record_length - 1] == '\n';
        std::array<uint8_t, max_data_length> candidate{};
        if (valid)
        {
            for (size_t i = 0; i < expected_length; ++i)
            {
                const int byte_high = hex_digit(rx_buffer[6 + i * 2]);
                const int byte_low = hex_digit(rx_buffer[7 + i * 2]);
                if (byte_high < 0 || byte_low < 0)
                {
                    valid = false;
                    break;
                }
                candidate[i] = static_cast<uint8_t>((byte_high << 4) | byte_low);
            }
        }
        if (valid)
        {
            rx_buffer.erase(0, record_length);
            std::copy_n(candidate.begin(), expected_length, out_data);
            return static_cast<int>(expected_length);
        }
        else
        {
            // A corrupt length can cover the next frame. Search from the
            // next byte rather than consuming a guessed record length.
            rx_buffer.erase(0, 1);
        }
    }
    return 0;
}
