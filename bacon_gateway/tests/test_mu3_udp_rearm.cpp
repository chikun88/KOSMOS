#include "udp.hpp"
#include <arpa/inet.h>
#include <unistd.h>
#include <stdexcept>
#include <iostream>

static void require(bool ok, const char* reason) {
    if (!ok) throw std::runtime_error(reason);
}
static uint16_t crc(const uint8_t* bytes, int count) {
    uint16_t value = 0xffff;
    for (int i = 0; i < count; ++i) {
        value ^= static_cast<uint16_t>(bytes[i]) << 8;
        for (int bit = 0; bit < 8; ++bit)
            value = (value & 0x8000) ? (value << 1) ^ 0x1021 : value << 1;
    }
    return value;
}
int main() {
    // Reserve an ephemeral loopback port. Never open serial or production port 8888.
    const int sender = socket(AF_INET, SOCK_DGRAM, 0);
    require(sender >= 0, "socket");
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    require(bind(sender, reinterpret_cast<sockaddr*>(&address), sizeof(address)) == 0, "bind");
    socklen_t size = sizeof(address);
    require(getsockname(sender, reinterpret_cast<sockaddr*>(&address), &size) == 0, "getsockname");
    const auto port = ntohs(address.sin_port);
    close(sender);
    UDP gateway(port);
    require(gateway.is_ready(), "gateway socket");
    const int client = socket(AF_INET, SOCK_DGRAM, 0);
    uint16_t sequence = 0;
    Controller_Packet manual{};
    auto command = [&](bool arm) {
        uint8_t packet[20] = {0xb6, 2};
        packet[2] = arm ? 1 : 0;
        packet[10] = sequence & 255; packet[11] = sequence >> 8; ++sequence;
        const auto checksum = crc(packet, 18);
        packet[18] = checksum & 255; packet[19] = checksum >> 8;
        require(sendto(client, packet, sizeof(packet), 0,
                       reinterpret_cast<sockaddr*>(&address), sizeof(address)) == 20, "send");
        usleep(1000);
        gateway.update(manual);
    };
    uint8_t radio[7] = {128,128,128,128,0,0,0xc0};
    gateway.remote_navigation.receive(radio);
    command(false);
    radio[6] = 0xc2; gateway.remote_navigation.receive(radio);
    command(true);
    require(gateway.is_auto_mode(), "first ARM");
    // The Android half-second release before the next tap sets stop_pending.
    radio[6] = 0xc0; gateway.remote_navigation.receive(radio);
    command(true);
    require(!gateway.is_auto_mode(), "release must block stale ARM");
    command(false);
    require(!gateway.remote_navigation.stop_pending, "DISARM must execute stop-latch acknowledgement");
    radio[6] = 0xc4; gateway.remote_navigation.receive(radio);
    command(true);
    require(gateway.is_auto_mode(), "second saved goal must engage after DISARM");
    gateway.remote_navigation.timeout();
    command(true);
    require(!gateway.is_auto_mode(), "radio loss must stop");
    gateway.remote_navigation.receive(radio);
    require(gateway.remote_navigation.slot == 0, "restored radio must not replay goal");
    command(false);
    radio[6] = 0xc0; gateway.remote_navigation.receive(radio);
    radio[6] = 0xc6; gateway.remote_navigation.receive(radio);
    command(true);
    require(gateway.is_auto_mode(), "new tap after radio recovery must engage");
    close(client);
    std::cout << "PASS: real UDP ARM/release/DISARM/rearm and radio recovery\n";
}
