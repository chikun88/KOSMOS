#include "mu3_navigation.hpp"
#include <cassert>
#include <iostream>

int main() {
    Mu3Navigation n;
    uint8_t p[7] = {128,128,128,128,0,0,0xc2};
    n.receive(p); assert(n.slot == 0); // Startup cannot replay a held request.
    p[6] = 0xc0; n.receive(p);
    p[6] = 0xc2; n.receive(p); assert(n.slot == 1 && n.sequence == 1);
    n.receive(p); assert(n.sequence == 1); // repeated radio frames
    p[6] = 0xe2; n.receive(p); assert(n.sequence == 2); // deliberate re-tap
    n.timeout(); assert(n.slot == 0 && n.sequence == 3);
    assert(!n.allow_auto(true));
    n.receive(p); assert(n.slot == 0); // no restart on restored radio
    p[6] = 0xc0; n.receive(p);
    p[6] = 0xc6; n.receive(p); assert(n.slot == 3);
    assert(!n.allow_auto(true)); // new goal must wait for Jetson disarm
    assert(n.allow_auto(false)); assert(n.allow_auto(true));
    p[0] = 160; n.receive(p); assert(n.slot == 0);
    assert(!n.allow_auto(true)); // manual override blocks stale auto output
    p[0] = 128; p[6] = 0; n.receive(p); assert(n.allow_auto(false));
    for (int slot = 1; slot <= 15; ++slot) {
        p[6] = static_cast<uint8_t>(0xc0 | (slot << 1));
        n.receive(p); assert(n.slot == slot);
    }
    p[6] = 0xc3; n.receive(p); assert(n.slot == 0); // PS bit not a command
    std::cout << "MU3 navigation protocol and fail-closed recovery passed\n";
}
