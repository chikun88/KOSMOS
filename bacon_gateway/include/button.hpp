#pragma once

#include <cstdint>

// axes
constexpr int AXIS_LX    = 0;
constexpr int AXIS_LY    = 1;
constexpr int AXIS_L2    = 2;
constexpr int AXIS_RX    = 3;
constexpr int AXIS_RY    = 4;
constexpr int AXIS_R2    = 5;
constexpr int AXIS_PAD_X = 6;
constexpr int AXIS_PAD_Y = 7;

// buttons
constexpr int BTN_BATSU   = 0;
constexpr int BTN_MARU    = 1; 
constexpr int BTN_SANKAKU = 2;
constexpr int BTN_SHIKAKU = 3; 
constexpr int BTN_L1      = 4;
constexpr int BTN_R1      = 5;
constexpr int BTN_L2      = 6;
constexpr int BTN_R2      = 7;
constexpr int BTN_CREATE  = 8; 
constexpr int BTN_OPTIONS = 9; 
constexpr int BTN_PS      = 10;
constexpr int BTN_L3      = 11;
constexpr int BTN_R3      = 12;
constexpr int BTN_PAD     = 13;



// enum class button : uint8_t
// {
//     batsu_button       = 0,
//     maru_button        = 1,
//     sankaku_button     = 2,
//     shikaku_button     = 3,
//     l1_button          = 4,
//     r1_button          = 5,
//     l2_button          = 6,
//     r2_button          = 7,
//     create_button      = 8,
//     option_button      = 9,
//     ps5_button         = 10,
//     left_stick_button  = 11,
//     right_stick_button = 12,
//     touch_pad_button   = 13
// };

// enum class axis : uint8_t
// {
//     left_stick_x   = 0,
//     left_stick_y   = 1,
//     l2_trigger     = 2,
//     right_stick_x  = 3,
//     right_stick_y  = 4,
//     r2_trigger     = 5,
//     dpad_x         = 6,
//     dpad_y         = 7
// };