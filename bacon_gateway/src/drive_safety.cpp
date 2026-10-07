#include "drive_safety.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

DriveLinkSafety::DriveLinkSafety(int degraded_ms, int stop_ms, int fault_ms)
    : degraded_ms_(degraded_ms),
      stop_ms_(stop_ms),
      fault_ms_(fault_ms),
      active_(false),
      stop_latched_(false),
      fault_latched_(false),
      estop_latched_(false),
      rearm_required_(true),
      disarm_seen_(false),
      previous_auto_request_(false),
      state_(DriveLinkState::DISARMED)
{
    if (degraded_ms_ < 0 || stop_ms_ <= degraded_ms_ || fault_ms_ <= stop_ms_)
    {
        throw std::invalid_argument("watchdog thresholds must satisfy 0 <= degraded < stop < fault");
    }
}

void DriveLinkSafety::update(bool valid_frame_received,
                             bool auto_requested,
                             bool estop_requested,
                             int command_age_ms)
{
    const int age_ms = std::max(0, command_age_ms);

    // A local emergency input must work even when the command link is down.
    if (estop_requested)
    {
        estop_latched_ = true;
        active_ = false;
        stop_latched_ = true;
        rearm_required_ = true;
        disarm_seen_ = false;
    }
    else if (valid_frame_received)
    {
        if (!auto_requested)
        {
            // A zero/disarmed frame is the first half of the rearm handshake.
            active_ = false;
            disarm_seen_ = true;
            estop_latched_ = false;
        }
        else if (rearm_required_)
        {
            // Do not clear a stop merely because traffic returned.  The edge
            // must follow a valid disarmed frame from the locked source.
            if (disarm_seen_ && !previous_auto_request_)
            {
                active_ = true;
                stop_latched_ = false;
                fault_latched_ = false;
                rearm_required_ = false;
                disarm_seen_ = false;
            }
        }
        else
        {
            active_ = true;
        }
        previous_auto_request_ = auto_requested;
    }

    if (active_ && age_ms >= stop_ms_)
    {
        active_ = false;
        stop_latched_ = true;
        rearm_required_ = true;
        disarm_seen_ = false;
    }
    if (stop_latched_ && age_ms >= fault_ms_)
    {
        fault_latched_ = true;
    }

    if (estop_latched_)
    {
        state_ = DriveLinkState::ESTOP;
    }
    else if (fault_latched_)
    {
        state_ = DriveLinkState::FAULT;
    }
    else if (stop_latched_)
    {
        state_ = DriveLinkState::CONTROLLED_STOP;
    }
    else if (!active_)
    {
        state_ = DriveLinkState::DISARMED;
    }
    else if (age_ms >= degraded_ms_)
    {
        state_ = DriveLinkState::DEGRADED;
    }
    else
    {
        state_ = DriveLinkState::ACTIVE;
    }
}

void DriveLinkSafety::reset_source()
{
    stop_latched_ = stop_latched_ || active_;
    active_ = false;
    rearm_required_ = true;
    disarm_seen_ = false;
    previous_auto_request_ = false;
    update(false, false, false, 0);
}

const char* DriveLinkSafety::state_name() const
{
    switch (state_)
    {
        case DriveLinkState::DISARMED:        return "DISARMED";
        case DriveLinkState::ACTIVE:          return "ACTIVE";
        case DriveLinkState::DEGRADED:        return "DEGRADED";
        case DriveLinkState::CONTROLLED_STOP: return "CONTROLLED_STOP";
        case DriveLinkState::FAULT:           return "FAULT";
        case DriveLinkState::ESTOP:           return "ESTOP";
    }
    return "UNKNOWN";
}

ControlledStopLimiter::ControlledStopLimiter(
    double units_per_second, double period_seconds)
    : step_units_(0), output_{0, 0, 0, 0}
{
    if (!std::isfinite(units_per_second) || !std::isfinite(period_seconds) ||
        !(units_per_second > 0.0) || !(period_seconds > 0.0) ||
        !std::isfinite(units_per_second * period_seconds))
    {
        throw std::invalid_argument("controlled-stop rate and period must be positive");
    }
    // No int16 target can need a larger step; bound before rounding/casting.
    step_units_ = std::max(1, static_cast<int>(std::lround(std::min(
        32768.0, units_per_second * period_seconds))));
}

std::array<int16_t, 4> ControlledStopLimiter::apply(
    const std::array<int16_t, 4>& requested,
    bool controlled_stop,
    bool immediate_stop)
{
    if (immediate_stop)
    {
        output_.fill(0);
        return output_;
    }
    if (!controlled_stop)
    {
        output_ = requested;
        return output_;
    }

    for (int16_t& value : output_)
    {
        const int current = static_cast<int>(value);
        if (current > 0)
        {
            value = static_cast<int16_t>(std::max(0, current - step_units_));
        }
        else if (current < 0)
        {
            value = static_cast<int16_t>(std::min(0, current + step_units_));
        }
    }
    return output_;
}
