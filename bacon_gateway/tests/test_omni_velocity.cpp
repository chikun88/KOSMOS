#include "move.hpp"
#include "value.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>

namespace
{
    void require(bool condition, const char* message)
    {
        if (!condition)
        {
            std::cerr << "test_omni_velocity: " << message << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }

    void require_wheels(
        const OmniVelocityMixResult& result,
        const std::array<int16_t, 4>& expected,
        const char* message)
    {
        require(result.wheels == expected, message);
        require(std::abs(result.scale - 1.0) < 1.0e-12,
                "single-axis command must not saturate");
    }
}

int main()
{
    static_assert(value::auto_lateral_sign == 1.0,
                  "ROS +vy (left) passes through the mixer boundary unchanged; "
                  "-1.0 drove the base to the wrong side on the real robot");
    static_assert(value::auto_rotation_lever_arm_m == 0.666144,
                  "rotation lever arm must match the CAD wheel-centre x+y");
    require(
        std::abs(value::auto_units_per_radps - 4797.569088) < 1.0e-9,
        "yaw coefficient must be derived from translation gain times CAD lever arm");

    // ROS body +vx is forward. It must match the existing manual forward mix.
    require_wheels(
        OMNI::mix_velocity(0.5, 0.0, 0.0),
        {-3601, -3601, 3601, 3601},
        "+vx must produce the forward wheel signs");

    // ROS body +vy is left and passes through the mixer boundary unchanged.
    // Measured on the robot on 2026-08-07: with the inversion the base went to
    // the wrong side, which a position loop turns into a lateral divergence
    // (-K diag(1,-1) e), seen as the base weaving left and right while it
    // translated.
    require_wheels(
        OMNI::mix_velocity(0.0, 0.5, 0.0),
        {3601, -3601, -3601, 3601},
        "+vy must produce the left wheel signs");

    // ROS body +wz is counter-clockwise and is inverted once at the mixer
    // boundary.  A pure yaw must drive all four wheels the
    // same way; the common sign is negative because the manual mixer's turn
    // input is positive for a *clockwise* turn.  That contract was measured on
    // the robot: manual driving is correct, and pushing the right stick right
    // -- which feeds the turn input positive, unmodified -- turns the base
    // clockwise.  See the derivation in value.hpp.
    require_wheels(
        OMNI::mix_velocity(0.0, 0.0, 1.0),
        {-4798, -4798, -4798, -4798},
        "+wz must produce the counter-clockwise wheel signs");

    // Forward translation plus yaw exceeds the wheel limit. Every wheel must
    // receive one common scale, preserving the geometry-derived twist ratio.
    const OmniVelocityMixResult combined = OMNI::mix_velocity(1.0, 0.0, 1.0);
    const double turn = value::auto_units_per_radps * value::auto_turn_sign;
    const std::array<double, 4> raw{
        -value::auto_units_per_mps + turn,
        -value::auto_units_per_mps + turn,
         value::auto_units_per_mps + turn,
         value::auto_units_per_mps + turn};
    // Which wheel saturates depends on whether the yaw term adds to or
    // subtracts from the forward term, so find the peak instead of assuming
    // index 0; it depends on the sign of auto_turn_sign.
    double max_raw = 0.0;
    std::size_t peak_index = 0;
    for (std::size_t i = 0; i < raw.size(); ++i)
    {
        if (std::abs(raw[i]) > max_raw)
        {
            max_raw = std::abs(raw[i]);
            peak_index = i;
        }
    }
    const double expected_scale =
        static_cast<double>(value::auto_wheel_limit) / max_raw;
    require(std::abs(combined.scale - expected_scale) < 1.0e-12,
            "combined command must report the common saturation scale");
    for (std::size_t i = 0; i < raw.size(); ++i)
    {
        require(
            combined.wheels[i] == static_cast<int16_t>(
                std::lround(raw[i] * expected_scale)),
            "all combined wheel commands must use the common saturation scale");
    }
    const int16_t limit_value = static_cast<int16_t>(
        raw[peak_index] < 0.0 ? -value::auto_wheel_limit
                              : value::auto_wheel_limit);
    require(combined.wheels[peak_index] == limit_value,
            "largest combined wheel must land exactly on the limit");
    const std::size_t twin_index = peak_index ^ 1u;
    require(combined.wheels[twin_index] == limit_value,
            "equal forward+yaw wheels must share the same limit");

    const auto sprint = OMNI::mix_velocity(4.0 * 0.38, 0.0, 0.0, 9000);
    require(sprint.wheels == std::array<int16_t, 4>{-9000, -9000, 9000, 9000},
            "sprint must saturate at 9000");
    require(std::abs(sprint.scale - 9000.0 / (4.0 * 0.38 * 7202.0)) < 1e-12,
            "sprint saturation must preserve the input ratio");
    const auto normal = OMNI::mix_velocity(4.0 * 0.38, 0.0, 0.0);
    require(normal.wheels == std::array<int16_t, 4>{-10000, -10000, 10000, 10000},
            "default transport limit must be 10000");
    require(OMNI::mix_velocity(INFINITY, 0.0, 0.0).wheels == std::array<int16_t,4>{0,0,0,0},
            "nonfinite velocity must produce a safe zero output");
    require(OMNI::mix_velocity(1e308, 0.0, 0.0).wheels == std::array<int16_t,4>{0,0,0,0},
            "overflowing velocity must produce a safe zero output");
    ARM arm;
    Controller_Packet buttons{};
    for (int i = 0; i < 500; ++i) {
        buttons.r2_state = true; arm.packet_range(buttons);
        buttons.r2_state = false; arm.packet_range(buttons);
    }
    require(arm.arm_range() == 32767, "ARM target must saturate without wrapping positive to negative");
    for (int i = 0; i < 600; ++i) {
        buttons.l1_state = true; arm.packet_range(buttons);
        buttons.l1_state = false; arm.packet_range(buttons);
    }
    require(arm.arm_range() == -32768, "ARM target must saturate without wrapping negative to positive");
    return EXIT_SUCCESS;
}
