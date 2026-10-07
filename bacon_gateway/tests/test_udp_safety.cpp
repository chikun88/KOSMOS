#include "udp.hpp"
#include "uart.hpp"

#include <arpa/inet.h>
#include <unistd.h>
#include <chrono>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {
void require(bool condition, const char* reason) {
    if (!condition) throw std::runtime_error(reason);
}
uint16_t crc(const std::vector<uint8_t>& data, std::size_t length) {
    uint16_t value = 0xffff;
    for (std::size_t i = 0; i < length; ++i) {
        value ^= static_cast<uint16_t>(data[i]) << 8;
        for (int bit = 0; bit < 8; ++bit)
            value = (value & 0x8000) ? (value << 1) ^ 0x1021 : value << 1;
    }
    return value;
}
void checksum(std::vector<uint8_t>& bytes) {
    const auto value = crc(bytes, bytes.size() - 2);
    bytes[bytes.size()-2] = value & 255;
    bytes.back() = value >> 8;
}
std::vector<uint8_t> command(uint16_t seq, uint8_t flags) {
    std::vector<uint8_t> data(UDP::v2_command_size, 0);
    data[0] = UDP::protocol_magic; data[1] = UDP::v2_command_version;
    data[2] = flags; data[10] = seq & 255; data[11] = seq >> 8;
    checksum(data); return data;
}
std::vector<uint8_t> passthrough(uint16_t seq, uint8_t flags,
                                  const std::vector<uint8_t>& frame) {
    std::vector<uint8_t> data(UDP::v4_header_size + frame.size() + 2, 0);
    data[0] = UDP::protocol_magic; data[1] = UDP::v4_command_version;
    data[2] = flags; data[6] = frame.size();
    data[8] = seq & 255; data[9] = seq >> 8;
    std::copy(frame.begin(), frame.end(), data.begin() + UDP::v4_header_size);
    checksum(data); return data;
}
struct Client {
    int fd = socket(AF_INET, SOCK_DGRAM | SOCK_CLOEXEC, 0);
    explicit Client() { require(fd >= 0, "client socket"); }
    ~Client() { close(fd); }
    void send(const UDP& gateway, const std::vector<uint8_t>& bytes) {
        sockaddr_in destination{};
        destination.sin_family = AF_INET;
        destination.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        destination.sin_port = htons(gateway.port());
        require(sendto(fd, bytes.data(), bytes.size(), 0,
                        reinterpret_cast<sockaddr*>(&destination), sizeof(destination)) ==
                        static_cast<ssize_t>(bytes.size()), "loopback send");
    }
};
void drain(UDP& gateway, const Controller_Packet& manual = {}) {
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    gateway.update(manual);
}
}

int main() {
    UDP gateway(0);
    require(gateway.is_ready() && gateway.port() != 0, "ephemeral gateway socket");
    UDP duplicate(gateway.port());
    require(!duplicate.is_ready(), "duplicate command receiver must fail binding");
    Client client, foreign;

    // Malformed first traffic must not claim the source lock.
    foreign.send(gateway, {1,2,3});
    drain(gateway);
    client.send(gateway, command(0, 0));
    client.send(gateway, command(1, 1));
    drain(gateway);
    require(gateway.is_auto_mode(), "DISARM and ARM in one burst must engage");
    require(gateway.received_count() == 2, "only valid source frames counted");

    // An E-stop followed by a normal ARM in the same burst cannot disappear.
    client.send(gateway, command(2, 3));
    client.send(gateway, command(3, 1));
    drain(gateway);
    require(gateway.estop_active() && !gateway.is_auto_mode(), "burst E-stop must latch");
    gateway.acknowledge_estop_output();
    client.send(gateway, command(4, 0));
    client.send(gateway, command(5, 1));
    drain(gateway);
    require(!gateway.estop_active() && gateway.is_auto_mode(), "explicit burst rearm after E-stop");

    // A malformed COBS frame with valid outer CRC must not consume its sequence.
    const uint8_t ids[] = {0,1,2,3};
    const int16_t wheels[] = {1000,-1000,1000,-1000};
    const auto frame = encode_robomas_frame(ids, wheels, 4);
    auto broken_frame = frame;
    broken_frame[0] = 255;
    client.send(gateway, passthrough(6, 1, broken_frame));
    drain(gateway);
    require(gateway.received_count() == 6, "malformed v4 must not count");
    client.send(gateway, passthrough(6, 1, frame));
    drain(gateway);
    require(gateway.received_count() == 7 && gateway.uart_passthrough_active(), "corrected same-sequence v4 accepted");

    const uint8_t duplicate_ids[] = {0,0};
    const int16_t duplicated_values[] = {32767,0};
    client.send(gateway, passthrough(7, 1, encode_robomas_frame(duplicate_ids, duplicated_values, 2)));
    const int16_t excessive[] = {10001,-1000,1000,-1000};
    client.send(gateway, passthrough(7, 1, encode_robomas_frame(ids, excessive, 4)));
    drain(gateway);
    require(gateway.received_count() == 7, "duplicate IDs and excessive wheel targets rejected");
    client.send(gateway, passthrough(7, 1, frame));
    drain(gateway);
    require(gateway.received_count() == 8, "invalid values did not consume sequence");

    client.send(gateway, command(8, 3));
    client.send(gateway, command(9, 0));
    client.send(gateway, command(10, 1));
    drain(gateway);
    require(gateway.estop_active() && !gateway.is_auto_mode(), "E-stop/reset/ARM burst requires an output stop cycle");
    gateway.acknowledge_estop_output();
    require(gateway.is_auto_mode(), "explicit rearm may resume after a complete stop output");

    // Local PS emergency stops even without a new UDP frame.
    Controller_Packet manual{}; manual.ps_state = true;
    gateway.update(manual);
    require(gateway.estop_active(), "local PS must latch with no UDP traffic");
    gateway.acknowledge_estop_output();
    client.send(gateway, command(11, 0));
    client.send(gateway, command(12, 1));
    drain(gateway);
    require(gateway.is_auto_mode(), "local emergency reset requires valid handshake");

    // Do not turn an old Linux receive-queue packet into a fresh motion command.
    client.send(gateway, command(13, 1));
    std::this_thread::sleep_for(std::chrono::milliseconds(value::sys::controlled_stop_ms + 30));
    gateway.update({});
    require(!gateway.is_auto_mode() && gateway.safety_stop_active(), "queued stale frame must not refresh watchdog");
    client.send(gateway, command(13, 1));
    drain(gateway);
    require(!gateway.is_auto_mode(), "fresh ARM alone cannot hide an elapsed watchdog gap");
    client.send(gateway, command(14, 0));
    client.send(gateway, command(15, 1));
    drain(gateway);
    require(gateway.is_auto_mode(), "rearm after command silence");

    // Source replacement cannot inherit an earlier sender's authorization.
    std::this_thread::sleep_for(std::chrono::milliseconds(value::sys::source_lock_timeout_ms + 30));
    foreign.send(gateway, command(100, 1));
    drain(gateway);
    require(!gateway.is_auto_mode(), "new source requires its own disarm");
    foreign.send(gateway, command(101, 0));
    foreign.send(gateway, command(102, 1));
    drain(gateway);
    require(gateway.is_auto_mode(), "new source explicit rearm");
    client.send(gateway, command(16, 3));
    drain(gateway);
    require(!gateway.estop_active(), "foreign packet cannot alter locked source");

    // Restarting at the same address/port must reset sequence and authorization.
    std::this_thread::sleep_for(std::chrono::milliseconds(value::sys::seq_resync_silence_ms + 30));
    foreign.send(gateway, command(0, 1));
    drain(gateway);
    require(!gateway.is_auto_mode(), "same-endpoint restart requires new disarm");
    foreign.send(gateway, command(1, 0));
    foreign.send(gateway, command(2, 1));
    drain(gateway);
    require(gateway.is_auto_mode(), "same-endpoint restart accepts reset sequence and fresh handshake");

    UDP legacy_gateway(0);
    client.send(legacy_gateway, {128,128,128,128,0,0,1});
    drain(legacy_gateway);
    require(legacy_gateway.estop_active(), "legacy PS must be an emergency stop");
    std::cout << "PASS: UDP binding, validation, sequencing, freshness, source rearm and emergency latching\n";
}
