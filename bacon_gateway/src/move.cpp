#include <iostream>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <initializer_list>

#include "move.hpp"
#include "packet.hpp"
#include "value.hpp"

using value::read_button;

// OMNI
OMNI::OMNI() : current_omni_range(value::omni_speed), prev_ue_button(false), prev_shita_button(false)
{
    for(int i = 0; i < 4; i++)
    {
        omni_speed[i] = 0;
    }
}

inline double OMNI::omni_deadzone(double v, double dz)
{
    if (std::abs(v) < dz)
    {
        return 0.0;
    }

    return (v > 0) ? (v - dz) / (1.0 - dz) : (v + dz) / (1.0 - dz);
}

void OMNI::packet_range(const Controller_Packet & packet)
{
    const bool up_button   = read_button(packet, value::omni_speed_up);
    const bool down_button = read_button(packet, value::omni_speed_down);

    if (up_button && !prev_ue_button)
    {
        if (current_omni_range < value::omni_max)
        {
            current_omni_range += value::omni_range;
        }
    }

    if (down_button && !prev_shita_button)
    {
        if (current_omni_range > value::omni_min)
        {
            current_omni_range -= value::omni_range;
        }
    }

    prev_ue_button    = up_button;
    prev_shita_button = down_button;
}

void OMNI::checker_omni(const Controller_Packet & packet)
{
    double move_x = static_cast<double>(packet.lx_state) / value::stick_norm;
    double move_y = static_cast<double>(packet.ly_state) / value::stick_norm;
    double turn   = static_cast<double>(packet.rx_state) / value::stick_norm;

    move_x = std::clamp(move_x, -1.0, 1.0);
    move_y = std::clamp(move_y, -1.0, 1.0);
    turn   = std::clamp(turn,   -1.0, 1.0);

    move_x = omni_deadzone(move_x, value::omni_deadzone);
    move_y = omni_deadzone(move_y, value::omni_deadzone);
    turn   = omni_deadzone(turn,   value::omni_deadzone);

    double power = std::min(1.0, std::sqrt(move_x * move_x + move_y * move_y));
    double angle = std::atan2(move_y, move_x);

    double vx = power * std::cos(angle);
    double vy = power * std::sin(angle);

    double m1 = vx - vy +  (value::omni_ratio) * turn;
    double m2 = -vx - vy + (value::omni_ratio) * turn;
    double m3 = -vx + vy + (value::omni_ratio) * turn;
    double m4 = vx + vy +  (value::omni_ratio) * turn;

    double maxabs = std::max({std::abs(m1), std::abs(m2), std::abs(m3), std::abs(m4)});

    if(maxabs > 1.0)
    {
        m1 /= maxabs;
        m2 /= maxabs;
        m3 /= maxabs;
        m4 /= maxabs;
    }

    omni_speed[0] = static_cast<int16_t>(std::lround(m1 * current_omni_range));
    omni_speed[1] = static_cast<int16_t>(std::lround(m2 * current_omni_range));
    omni_speed[2] = static_cast<int16_t>(std::lround(m3 * current_omni_range));
    omni_speed[3] = static_cast<int16_t>(std::lround(m4 * current_omni_range));
}

int16_t OMNI::motor_speed(uint8_t index) const
{
    if(index < 4)
    {
        return omni_speed[index];
    }

    return 0;
}

OmniVelocityMixResult OMNI::mix_velocity(
    double vx_mps, double vy_mps, double w_radps, int wheel_limit)
{
    if (!std::isfinite(vx_mps) || !std::isfinite(vy_mps) || !std::isfinite(w_radps))
    {
        return {{0, 0, 0, 0}, 0.0};
    }
    wheel_limit = std::clamp(wheel_limit, 1, value::auto_wheel_limit);
    // 物理速度→各輪値。checker_omni()と同じX配置ミキシング
    // （m1=mx-my+t 系）を、正規化・デッドゾーンを介さず単位換算で行う。
    const double ux = vy_mps  * value::auto_units_per_mps   * value::auto_lateral_sign;
    const double uy = vx_mps  * value::auto_units_per_mps   * value::auto_forward_sign;
    const double ut = w_radps * value::auto_units_per_radps * value::auto_turn_sign;

    double m[4];
    m[0] =  ux - uy + ut;
    m[1] = -ux - uy + ut;
    m[2] = -ux + uy + ut;
    m[3] =  ux + uy + ut;
    for (double wheel : m)
    {
        if (!std::isfinite(wheel)) return {{0, 0, 0, 0}, 0.0};
    }

    const double maxabs = std::max(
        {std::abs(m[0]), std::abs(m[1]), std::abs(m[2]), std::abs(m[3])});

    double k = 1.0;
    if (maxabs > static_cast<double>(wheel_limit))
    {
        k = static_cast<double>(wheel_limit) / maxabs;
    }

    OmniVelocityMixResult result{{0, 0, 0, 0}, k};
    for (std::size_t i = 0; i < result.wheels.size(); ++i)
    {
        result.wheels[i] = static_cast<int16_t>(std::lround(m[i] * k));
    }

    return result;
}

double OMNI::speeds_from_velocity(
    double vx_mps, double vy_mps, double w_radps, int16_t out[4])
{
    const OmniVelocityMixResult result =
        mix_velocity(vx_mps, vy_mps, w_radps);
    std::copy(result.wheels.begin(), result.wheels.end(), out);
    return result.scale;
}



// ARM(m5)
ARM::ARM() : target_phase(0), prev_r2(false), prev_l1(false)
{
}

void ARM::packet_range(const Controller_Packet & packet)
{
    bool current_r2 = read_button(packet, value::arm_up);
    bool current_l1 = read_button(packet, value::arm_down);

    if (current_r2 && !prev_r2)
    {
        target_phase = static_cast<int16_t>(std::min(32767,
            static_cast<int>(target_phase) + value::arm_phase_step));
    }
    // L1が押された瞬間に逆回転（arm_phase_step 分）を減算
    if (current_l1 && !prev_l1)
    {
        target_phase = static_cast<int16_t>(std::max(-32768,
            static_cast<int>(target_phase) - value::arm_phase_step));
    }

    prev_r2 = current_r2;
    prev_l1 = current_l1;

}

int16_t ARM::arm_range() const
{
    return target_phase;
}



// GM
GM::GM() : current_gm_angle(0.0), is_reloading(false)
{
}

void GM::packet_range(const Controller_Packet & packet)
{
    is_reloading = read_button(packet, value::gm_reload);

    if (!is_reloading)
    {
        if(read_button(packet, value::gm_up))
        {
            current_gm_angle += value::gm_angle_range;
        }
        else if(read_button(packet, value::gm_down))
        {
            current_gm_angle -= value::gm_angle_range;
        }

        // Only an explicit operator adjustment creates a new bounded target.
        // An idle cycle must retain a delegated target unchanged.
        if (read_button(packet, value::gm_up) || read_button(packet, value::gm_down))
        {
            current_gm_angle = std::clamp(current_gm_angle, value::gm_angle_min, value::gm_angle_max);
        }
    }
}

int16_t GM::angle() const
{
    int16_t ans = 0;

    if (is_reloading)
    {
        ans = static_cast<int16_t>(value::gm_reload_angle); 
    }
    else
    {
        ans = static_cast<int16_t>(std::lround(current_gm_angle));
    }

    if (ans == -1)
    {
        ans = -2;
    }

    return ans;
}



// UPDOWN
UPDOWN::UPDOWN() : current_updown_speed(0)
{
}

void UPDOWN::packet_range(const Controller_Packet & packet)
{
    if (read_button(packet, value::updown_on))
    {
        current_updown_speed = value::updown_speed;
    }
    else if (read_button(packet, value::updown_reverse))
    {
        current_updown_speed = -value::updown_speed;
    }
    else
    {
        current_updown_speed = 0;
    }
}

int16_t UPDOWN::motor_speed() const
{
    return current_updown_speed;
}



// COLLECT
COLLECT::COLLECT() : current_collect_speed(0)
{
}

void COLLECT::packet_range(const Controller_Packet & packet)
{
    // maru(collect_on_a)で正転、shikaku(collect_on_b)で逆転
    if (read_button(packet, value::collect_on_a))
    {
        current_collect_speed = value::collect_speed;
    }
    else if (read_button(packet, value::collect_on_b))
    {
        current_collect_speed = -value::collect_speed;
    }
    else
    {
        current_collect_speed = 0;
    }
}

int16_t COLLECT::motor_speed() const
{
    return current_collect_speed;
}



// GPIO
GPIO::GPIO() : is_on(false)
{
}

void GPIO::packet_range(const Controller_Packet & packet)
{
    is_on = value::read_button(packet, value::gpio_toggle);
}

uint8_t GPIO::GPIO_state() const
{
    // ON → GPIO4 HIGH / OFF → すべてLOW（value.hpp の gpio_on/gpio_off を参照）
    return is_on ? value::gpio_on : value::gpio_off;
}
