#include "MU3.hpp"
#include "cobs.hpp"
#include "packet.hpp"
#include "uart.hpp"

#include <array>
#include <chrono>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <limits>
#include <string>
#include <termios.h>
#include <thread>
#include <type_traits>
#include <unistd.h>
#include <vector>

namespace
{
    void require(bool condition, const char* message)
    {
        if (!condition)
        {
            std::cerr << "test_serial_io: " << message << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }

    struct PseudoTerminal
    {
        int master = -1;
        std::string slave;

        PseudoTerminal()
        {
            master = posix_openpt(O_RDWR | O_NOCTTY | O_NONBLOCK | O_CLOEXEC);
            require(master >= 0, "create PTY");
            require(grantpt(master) == 0 && unlockpt(master) == 0, "unlock PTY");
            const char* name = ptsname(master);
            require(name != nullptr, "find slave PTY");
            slave = name;
        }
        ~PseudoTerminal() { if (master >= 0) close(master); }
        void disconnect() { close(master); master = -1; }

        void input(const std::string& bytes)
        {
            require(write(master, bytes.data(), bytes.size()) ==
                    static_cast<ssize_t>(bytes.size()), "inject serial input");
        }
        std::vector<uint8_t> output()
        {
            std::vector<uint8_t> bytes;
            std::array<uint8_t, 4096> buffer{};
            for (int idle = 0; idle < 5;)
            {
                const ssize_t count = read(master, buffer.data(), buffer.size());
                if (count > 0)
                {
                    bytes.insert(bytes.end(), buffer.begin(), buffer.begin() + count);
                    idle = 0;
                    continue;
                }
                if (count < 0 && errno == EINTR) continue;
                require(count >= 0 || errno == EAGAIN || errno == EWOULDBLOCK,
                        "read serial output");
                ++idle;
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            return bytes;
        }
    };

    int receive_after_delivery(MU3& mu3, uint8_t* output, size_t length)
    {
        for (int attempt = 0; attempt < 20; ++attempt)
        {
            const int received = mu3.receive(output, length);
            if (received != 0) return received;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        return 0;
    }

    void test_cobs()
    {
        // The first long run crosses the 254-byte block boundary that used
        // to overrun encode_robomas_frame's payload+2 allocation.
        for (size_t length : {size_t{0}, size_t{1}, size_t{253}, size_t{254},
                              size_t{255}, size_t{508}, size_t{1024}})
        {
            std::vector<uint8_t> payload(length, 0x7F);
            for (int pattern = 0; pattern < 2; ++pattern)
            {
                if (pattern == 1)
                {
                    for (size_t i = 0; i < length; i += 7) payload[i] = 0;
                }
                const size_t capacity = cobs_encoded_max_size(length);
                std::vector<uint8_t> encoded(capacity + 1, 0xAA);
                const size_t result = encode_cobs_bounded(payload.data(), length,
                                                           encoded.data(), capacity);
                require(result > 0 && result <= capacity, "long COBS frame size");
                require(encoded[result - 1] == 0 && encoded[capacity] == 0xAA,
                        "COBS terminator and capacity canary");
                std::vector<uint8_t> decoded(length + 1, 0xAA);
                require(decode_cobs_bounded(encoded.data(), result - 1,
                                            decoded.data(), length) == length,
                        "COBS round trip length");
                require(std::equal(payload.begin(), payload.end(), decoded.begin()) &&
                        decoded[length] == 0xAA, "COBS round trip data and canary");
            }
        }
        require(cobs_encoded_max_size(std::numeric_limits<size_t>::max()) == 0,
                "COBS capacity overflow rejected");
        std::array<uint8_t, 8> output{};
        output.fill(0xAA);
        const uint8_t payload[] = {1, 2, 3};
        require(encode_cobs_bounded(payload, sizeof(payload), output.data(), 4) == 0 &&
                output[0] == 0xAA, "short encoder destination rejected before writing");
        for (const std::vector<uint8_t>& malformed :
             {std::vector<uint8_t>{0}, {3, 1}, {3, 1, 0}, {2, 1, 0}})
        {
            require(decode_cobs_bounded(malformed.data(), malformed.size(),
                                        output.data(), output.size()) == 0 &&
                    output[0] == 0xAA, "malformed COBS does not mutate destination");
        }
        const uint8_t valid[] = {4, 1, 2, 3};
        require(decode_cobs_bounded(valid, sizeof(valid), output.data(), 2) == 0 &&
                output[0] == 0xAA, "short decoder destination rejected before writing");
        require(encode_cobs(nullptr, 1, output.data()) == 0 &&
                encode_cobs(payload, sizeof(payload), nullptr) == 0 &&
                decode_cobs(nullptr, 1, output.data()) == 0,
                "null COBS arguments rejected");

        std::vector<uint8_t> ids(200, 0x7F);
        std::vector<int16_t> values(200, 0x7F7F);
        const auto frame = encode_robomas_frame(ids.data(), values.data(), ids.size());
        require(frame.size() > ids.size() * 3 + 2, "long motor frame needs extra COBS blocks");
        std::vector<uint8_t> decoded(ids.size() * 3);
        require(decode_cobs_bounded(frame.data(), frame.size() - 1,
                                    decoded.data(), decoded.size()) == decoded.size(),
                "long motor frame decoded safely");
        require(encode_robomas_frame(ids.data(), values.data(),
                                    std::numeric_limits<size_t>::max()).empty(),
                "motor count multiplication overflow rejected");
    }

    void test_uart()
    {
        PseudoTerminal terminal;
        {
            UART uart(terminal.slave, B115200);
            require(uart.is_open(), "motor UART opened");
            UART duplicate(terminal.slave, B115200);
            require(!duplicate.is_open(), "duplicate motor UART rejected");
            MU3 duplicate_radio(terminal.slave.c_str());
            require(!duplicate_radio.is_open(), "UART and radio cannot share one TTY");
            require(!uart.uart_send_encoded(nullptr, 1) &&
                    !uart.uart_send_encoded(nullptr, 0), "null UART frame rejected");
            packet command{};
            command.m1 = 321;
            uint8_t ids[robomas_command_count];
            int16_t values[robomas_command_count];
            expand_packet_commands(command, ids, values);
            const auto expected = encode_robomas_frame(ids, values, robomas_command_count);
            require(uart.uart_send(command), "send normal motor command");
            require(terminal.output() == expected, "motor command byte contract retained");
            require(uart.sent_count() == 1, "complete motor command counted");

            std::vector<uint8_t> large(1024 * 1024, 0x01);
            large.back() = 0;
            const auto before = std::chrono::steady_clock::now();
            require(!uart.uart_send_encoded(large.data(), large.size()),
                    "partial motor command rejected");
            require(std::chrono::steady_clock::now() - before <
                    std::chrono::milliseconds(100), "full UART cannot block control loop");
            const auto partial = terminal.output();
            require(!partial.empty() && partial.size() < large.size() &&
                    partial.back() != 0, "PTY provokes an incomplete motor frame");
            require(uart.sent_count() == 1 && uart.busy_skip_count() > 0,
                    "partial motor command is not counted as sent");
            require(uart.uart_send(command), "send motor command after partial frame");
            auto synchronized = terminal.output();
            require(synchronized.size() == expected.size() + 1 && synchronized[0] == 0,
                    "partial frame boundary terminated before next motor command");
            require(std::equal(expected.begin(), expected.end(), synchronized.begin() + 1),
                    "complete command follows recovery delimiter");
            require(uart.flush_output(10), "PTY output flush completes");
            terminal.disconnect();
            require(!uart.uart_send(command) && !uart.is_open(),
                    "motor disconnect closes failed port");
        }
        PseudoTerminal reusable;
        { UART first(reusable.slave, B115200); require(first.is_open(), "first owner opened"); }
        { UART next(reusable.slave, B115200); require(next.is_open(), "TTY exclusivity released on close"); }
        { UART invalid(reusable.slave, -1); require(!invalid.is_open(), "invalid baud rejected"); }
        { UART zero(reusable.slave, B0); require(!zero.is_open(), "hangup baud rejected"); }
    }

    void test_mu3()
    {
        PseudoTerminal terminal;
        MU3 mu3(terminal.slave.c_str());
        require(mu3.is_open(), "MU3 opened");
        MU3 duplicate(terminal.slave.c_str());
        require(!duplicate.is_open(), "duplicate MU3 rejected");
        std::array<uint8_t, 7> output{};
        output.fill(0xAA);
        require(mu3.receive(nullptr, 7) == -1 &&
                mu3.receive(output.data(), 0) == -1 &&
                mu3.receive(output.data(), std::numeric_limits<size_t>::max()) == -1,
                "invalid MU3 receive arguments rejected");
        terminal.input("noise*D");
        require(receive_after_delivery(mu3, output.data(), output.size()) == 0,
                "split MU3 header retained");
        terminal.input("R=0E010203040506");
        require(receive_after_delivery(mu3, output.data(), output.size()) == 0 &&
                output[0] == 0xAA, "partial MU3 payload does not mutate output");
        terminal.input("07\r\n");
        require(receive_after_delivery(mu3, output.data(), output.size()) == 7 &&
                output == std::array<uint8_t, 7>{1, 2, 3, 4, 5, 6, 7},
                "fragmented MU3 record reconstructed");
        for (const std::string& invalid : std::vector<std::string>{
                "*DR=0E1Z020304050607\r\n",
                "*DR=0E+1020304050607\r\n", "*DR=0D01020304050607\r\n",
                "*DR=0701020304050607\r\n", "*DR=0E01020304050607xx"})
        {
            terminal.input(invalid);
            const auto original = output;
            require(receive_after_delivery(mu3, output.data(), output.size()) == 0 &&
                    output == original, "malformed MU3 record rejected atomically");
        }
        terminal.input("*DR=FEbroken*DR=0E11111111111111\r\n*DR=0E22222222222222\r\n");
        require(receive_after_delivery(mu3, output.data(), output.size()) == 7 &&
                output == std::array<uint8_t, 7>{0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11},
                "broken declared length recovers with first buffered record");
        require(receive_after_delivery(mu3, output.data(), output.size()) == 7 &&
                output == std::array<uint8_t, 7>{0x22, 0x22, 0x22, 0x22, 0x22, 0x22, 0x22},
                "buffered controls retain stop and navigation edge order");
        terminal.input(std::string(3000, 'X') + "*DR=0Eabcdef12345678\r\n");
        require(receive_after_delivery(mu3, output.data(), output.size()) == 7 &&
                output == std::array<uint8_t, 7>{0xAB, 0xCD, 0xEF, 0x12, 0x34, 0x56, 0x78},
                "MU3 recovers after noise and accepts lowercase hex");

        const uint8_t data[] = {0xA1, 0xB2, 0};
        // Sending must retain concurrent incoming commands, formerly lost to
        // tcflush(TCIFLUSH) in MU3::send.
        terminal.input("*DR=0E33333333333333\r\n");
        require(mu3.send(data, sizeof(data)) == 0, "send MU3 command");
        const auto sent = terminal.output();
        const std::string sent_text(sent.begin(), sent.end());
        require(sent_text == "@DT06A1B200\r\n", "MU3 transmit contract retained");
        require(receive_after_delivery(mu3, output.data(), output.size()) == 7 &&
                output[0] == 0x33, "outgoing MU3 does not flush controller input");
        std::array<uint8_t, 127> maximum{};
        require(mu3.send(maximum.data(), maximum.size()) == 0,
                "maximum legal MU3 length no longer exceeds fixed buffer");
        require(terminal.output().size() == 261, "maximum MU3 frame length");
        require(mu3.send(nullptr, 1) == -1 && mu3.send(data, 128) == -1,
                "invalid MU3 send arguments rejected");

        terminal.input("*DR=0E44444444444444\r\n*DR=0E55555555555555\r\n");
        require(receive_after_delivery(mu3, output.data(), output.size()) == 7 &&
                output[0] == 0x44, "first radio record retained before stall");
        terminal.input("*DR=0E66666666666666\r\n");
        require(mu3.discard_input(), "discard stale radio data after control-loop stall");
        require(receive_after_delivery(mu3, output.data(), output.size()) == 0 &&
                output[0] == 0x44, "stall recovery clears both user and kernel radio buffers");
        terminal.input("*DR=0E77777777777777\r\n");
        require(receive_after_delivery(mu3, output.data(), output.size()) == 7 &&
                output[0] == 0x77, "post-flush radio data becomes fresh");
        terminal.disconnect();
        require(mu3.receive(output.data(), output.size()) == -1 && !mu3.is_open(),
                "MU3 disconnect closes failed port");
        require(!mu3.discard_input(), "closed radio cannot discard input");
        MU3 null_device(nullptr);
        require(!null_device.is_open(), "null MU3 device rejected");
    }
}

int main()
{
    static_assert(!std::is_copy_constructible<UART>::value, "UART must own its fd");
    static_assert(!std::is_copy_constructible<MU3>::value, "MU3 must own its fd");
    test_cobs();
    test_uart();
    test_mu3();
    return EXIT_SUCCESS;
}
