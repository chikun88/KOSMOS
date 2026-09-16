#include "drive_safety.hpp"

#include <array>
#include <cassert>
#include <cstdint>

int main()
{
    DriveLinkSafety safety(30, 60, 200);

    // Startup requires a disarmed packet followed by an auto-request edge.
    safety.update(true, true, false, 0);
    assert(!safety.motion_allowed());
    assert(safety.rearm_required());
    safety.update(true, false, false, 0);
    safety.update(true, true, false, 0);
    assert(safety.motion_allowed());
    assert(safety.state() == DriveLinkState::ACTIVE);

    safety.update(false, true, false, 35);
    assert(safety.state() == DriveLinkState::DEGRADED);
    safety.update(false, true, false, 65);
    assert(safety.stop_required());
    assert(!safety.motion_allowed());
    assert(safety.state() == DriveLinkState::CONTROLLED_STOP);

    // Traffic recovery alone cannot restart the drivebase.
    safety.update(true, true, false, 0);
    assert(!safety.motion_allowed());
    safety.update(true, false, false, 0);
    safety.update(true, true, false, 0);
    assert(safety.motion_allowed());

    // E-stop is immediate and also requires the two-step rearm handshake.
    safety.update(true, true, true, 0);
    assert(safety.estop_active());
    assert(safety.state() == DriveLinkState::ESTOP);
    safety.update(true, false, false, 0);
    assert(!safety.estop_active());
    safety.update(true, true, false, 0);
    assert(safety.motion_allowed());

    ControlledStopLimiter limiter(1000.0, 0.01);  // 10 units per tick
    std::array<int16_t, 4> requested{100, -80, 20, -5};
    assert(limiter.apply(requested, false, false) == requested);
    const std::array<int16_t, 4> first{90, -70, 10, 0};
    assert(limiter.apply({}, true, false) == first);
    for (int i = 0; i < 9; ++i)
    {
        limiter.apply({}, true, false);
    }
    const std::array<int16_t, 4> stopped{0, 0, 0, 0};
    assert(limiter.output() == stopped);

    limiter.apply(requested, false, false);
    assert(limiter.apply({}, false, true) == stopped);
    return 0;
}
