// Integration fixture: actual UDP/MU3 decoder, no serial or motor devices.
#include "udp.hpp"
#include "packet.hpp"
#include <arpa/inet.h>
#include <unistd.h>
#include <csignal>
#include <chrono>
#include <stdexcept>

static volatile sig_atomic_t running = 1;
int main() {
    signal(SIGTERM, [](int){ running = 0; });
    signal(SIGINT, [](int){ running = 0; });
    UDP gateway(39400);
    int radio = socket(AF_INET, SOCK_DGRAM | SOCK_NONBLOCK, 0);
    sockaddr_in addr{};
    addr.sin_family = AF_INET; addr.sin_port = htons(39401);
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (!gateway.is_ready() || bind(radio, (sockaddr*)&addr, sizeof(addr)))
        throw std::runtime_error("loopback port unavailable");
    auto last = std::chrono::steady_clock::now() - std::chrono::seconds(1);
    Controller_Packet manual{};
    while (running) {
        uint8_t frame[8];
        while (recv(radio, frame, sizeof(frame), 0) == 7) {
            gateway.remote_navigation.receive(frame);
            last = std::chrono::steady_clock::now();
        }
        bool alive = std::chrono::steady_clock::now() - last < std::chrono::milliseconds(100);
        if (!alive) gateway.remote_navigation.timeout();
        gateway.update(manual);
        packet motors{};
        double vx=0, vy=0, wz=0;
        if (gateway.is_auto_mode()) {
            vx=gateway.command_vx_mps(); vy=gateway.command_vy_mps(); wz=gateway.command_w_radps();
            int16_t m=0;
            if (gateway.passthrough_commands().find(value::cmd_omni[0],m)) motors.m1=m;
            if (gateway.passthrough_commands().find(value::cmd_omni[1],m)) motors.m2=m;
            if (gateway.passthrough_commands().find(value::cmd_omni[2],m)) motors.m3=m;
            if (gateway.passthrough_commands().find(value::cmd_omni[3],m)) motors.m4=m;
        }
        gateway.send_telemetry(manual, alive, true, motors, vx, vy, wz);
        usleep(5000);
    }
    close(radio);
}
