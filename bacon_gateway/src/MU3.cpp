#include <iostream>
#include <fcntl.h>
#include <unistd.h>
#include <termios.h>
#include <cstring>
#include <cerrno>

#include "MU3.hpp"
#include "value.hpp"

MU3::MU3(const char* device_path)
{
    fd = open(device_path, O_RDWR | O_NOCTTY | O_NDELAY);
    if (fd == -1) 
    {
        std::cerr << "[エラー] MU3ポートが開けません: " << device_path << std::endl;
        return;
    }

    fcntl(fd, F_SETFL, 0);
    struct termios options;

    memset(&options, 0, sizeof(options)); 
    tcgetattr(fd, &options);
    cfmakeraw(&options);
    cfsetispeed(&options, value::sys::mu3_baud);
    cfsetospeed(&options, value::sys::mu3_baud);

    options.c_cflag |= (CS8 | CLOCAL | CREAD);
    options.c_cflag &= ~CRTSCTS; 
    options.c_cc[VMIN] = 0;
    options.c_cc[VTIME] = 0; 

    tcflush(fd, TCIOFLUSH);
    tcsetattr(fd, TCSANOW, &options);
}

MU3::~MU3()
{
    if (fd != -1) close(fd);
}

bool MU3::is_open() const
{
    return fd != -1;
}

int MU3::send(const uint8_t* data, size_t length)
{
    if (!is_open() || (data == nullptr && length != 0) || length > 0x7F) return -1;

    tcflush(fd, TCIFLUSH);
    char payload[256];
    const size_t hex_length = length * 2;
    const size_t frame_length = 5 + hex_length + 2;
    if (frame_length > sizeof(payload)) return -1;
    
    // @DTコマンド + データ長(Hex化するので元の長さの2倍)
    int pos = snprintf(payload, sizeof(payload), "@DT%02X", static_cast<unsigned int>(hex_length));
    if (pos != 5) return -1;
    
    // データをHex文字列("A1B2...")に変換
    for (size_t i = 0; i < length; ++i) 
    {
        const int chars_written = snprintf(payload + pos, sizeof(payload) - pos, "%02X", data[i]);
        if (chars_written != 2) return -1;
        pos += chars_written;
    }
    
    // 即時送信のトリガーとなるターミネータ
    const int terminator_length = snprintf(payload + pos, sizeof(payload) - pos, "\r\n");
    if (terminator_length != 2) return -1;

    const size_t total_length = static_cast<size_t>(pos + terminator_length);
    size_t total_written = 0;
    while (total_written < total_length)
    {
        const ssize_t written = write(fd, payload + total_written, total_length - total_written);
        if (written > 0)
        {
            total_written += static_cast<size_t>(written);
            continue;
        }
        if (written < 0 && errno == EINTR) continue;
        return -1;
    }
    if (tcdrain(fd) != 0) return -1;
    return 0;
}

int MU3::receive(uint8_t* out_data, size_t expected_length)
{
    if (!is_open()) return -1;
    
    char buf[256];
    ssize_t n = read(fd, buf, sizeof(buf));
    if (n > 0)
    {
        rx_buffer.append(buf, n);
    }

    // コマンドモードの受信ヘッダ "*DR=" を探す
    size_t dr_pos = rx_buffer.find("*DR=");
    if (dr_pos != std::string::npos)
    {
        if (dr_pos > 0) rx_buffer.erase(0, dr_pos);

        if (rx_buffer.length() < 6) return 0; // "*DR=XX" の6文字が揃うまで待機

        size_t expected_hex_len = expected_length * 2;
        size_t required_len = 6 + expected_hex_len + 2; // *DR=XX(6) + Hexデータ + \r\n(2)
        
        if (rx_buffer.length() >= required_len)
        {
            if (rx_buffer[6 + expected_hex_len] == '\r' &&
                rx_buffer[6 + expected_hex_len + 1] == '\n')
            {
                try
                {
                    // Hex文字列からバイナリ値に復元
                    for (size_t i = 0; i < expected_length; ++i)
                    {
                        std::string byte_str = rx_buffer.substr(6 + (i * 2), 2);
                        out_data[i] = static_cast<uint8_t>(std::stoul(byte_str, nullptr, 16));
                    }
                }
                catch (...)
                {
                    rx_buffer.erase(0, 3);
                    return 0;
                }

                rx_buffer.erase(0, required_len);
                return static_cast<int>(expected_length);
            }
            else
            {
                // パケット破損時は破棄
                rx_buffer.erase(0, 3);
                return 0;
            }
        }
    }
    else
    {
        if (rx_buffer.length() > 512) rx_buffer.erase(0, rx_buffer.length() - 256);
    }
    return 0; 
}
