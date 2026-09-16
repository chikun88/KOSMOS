#pragma once

#include <array>
#include <cstdint>

enum class DriveLinkState : uint8_t
{
    DISARMED = 0,
    ACTIVE,
    DEGRADED,
    CONTROLLED_STOP,
    FAULT,
    ESTOP
};

// Jetson command-link state machine.  A stop caused by stale traffic is
// latched: receiving packets again is not enough to resume.  A valid disarmed
// packet followed by a fresh auto-request edge is required.
class DriveLinkSafety
{
    public:
        DriveLinkSafety(int degraded_ms, int stop_ms, int fault_ms);

        void update(bool valid_frame_received,
                    bool auto_requested,
                    bool estop_requested,
                    int command_age_ms);

        bool motion_allowed() const { return state_ == DriveLinkState::ACTIVE || state_ == DriveLinkState::DEGRADED; }
        bool stop_required() const { return stop_latched_ || estop_latched_; }
        bool rearm_required() const { return rearm_required_; }
        bool fault_latched() const { return fault_latched_; }
        bool estop_active() const { return estop_latched_; }
        bool quality_degraded() const { return state_ == DriveLinkState::DEGRADED; }
        DriveLinkState state() const { return state_; }
        const char* state_name() const;

    private:
        int degraded_ms_;
        int stop_ms_;
        int fault_ms_;
        bool active_;
        bool stop_latched_;
        bool fault_latched_;
        bool estop_latched_;
        bool rearm_required_;
        bool disarm_seen_;
        bool previous_auto_request_;
        DriveLinkState state_;
};

// Applies a deterministic ramp to all four wheel target values.  The limiter
// stores no dynamic memory and is called from the 200 Hz loop.
class ControlledStopLimiter
{
    public:
        ControlledStopLimiter(double units_per_second, double period_seconds);

        std::array<int16_t, 4> apply(
            const std::array<int16_t, 4>& requested,
            bool controlled_stop,
            bool immediate_stop);

        const std::array<int16_t, 4>& output() const { return output_; }

    private:
        int step_units_;
        std::array<int16_t, 4> output_;
};
