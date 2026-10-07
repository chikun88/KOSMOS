#pragma once

#include "packet.hpp"
#include "value.hpp"

#include <array>
#include <cstdint>
#include <chrono>

struct OmniVelocityMixResult
{
    std::array<int16_t, 4> wheels;
    double scale;
};

// OMNI
class OMNI
{
    public:
        OMNI();
        ~OMNI() = default;

        void packet_range(const Controller_Packet& packet);

        void checker_omni(const Controller_Packet& packet);

        int16_t motor_speed(uint8_t index) const;

        // v3自動走行用: 物理速度指令(wire座標系: vx前+ / vy左+ / ω CCW+)を
        // 各輪値へ直接変換する。デッドゾーンなし・手動速度レンジ非依存の
        // 固定較正(value::auto_units_per_*)。各輪がauto_wheel_limitを超える
        // 場合は方向を保ったまま全輪比例縮小し、その縮小率k(≦1)を返す。
        // 戻り値だけで完結するため、座標契約と飽和を単体テストできる。
        static OmniVelocityMixResult mix_velocity(
            double vx_mps, double vy_mps, double w_radps, int wheel_limit = 10000);

        // 既存呼び出し元との互換ラッパー。
        static double speeds_from_velocity(
            double vx_mps, double vy_mps, double w_radps, int16_t out[4]);

    private:
        int16_t omni_speed[4];
        int current_omni_range;
        bool prev_ue_button;
        bool prev_shita_button;

        inline double omni_deadzone(double v, double dz);
};



// ARM(m5)
class ARM
{
    public:
        ARM();
        ~ARM() = default;

        void packet_range(const Controller_Packet & packet);

        int16_t arm_range() const;
        // Synchronize a commanded target from authorized v4 delegation.
        void set_target(int16_t target) { target_phase = target; }

    private:
        int16_t target_phase;
        bool prev_r2;
        bool prev_l1;
};



// GM
class GM
{
    public:
        GM();
        ~GM() = default;

        void packet_range(const Controller_Packet& packet);

        int16_t angle() const;
        void set_target(int16_t target) { current_gm_angle = target; is_reloading = false; }

    private:
        double current_gm_angle;
        bool is_reloading;
};



// UPDOWN
class UPDOWN
{
    public:
        UPDOWN();
        ~UPDOWN() = default;

        void packet_range(const Controller_Packet& packet);
        int16_t motor_speed() const;

    private:
        int16_t current_updown_speed;
};



// COLLECT
class COLLECT
{
    public:
        COLLECT();
        ~COLLECT() = default;

        void packet_range(const Controller_Packet& packet);
        int16_t motor_speed() const;

    private:
        int16_t current_collect_speed;
};



// GPIO
class GPIO
{
    public:
        GPIO();
        ~GPIO() = default;

        void packet_range(const Controller_Packet& packet);
        uint8_t GPIO_state() const;

    private:
        bool is_on;
};
