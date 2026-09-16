#pragma once

#include <cstdint>

struct __attribute__((packed)) packet
{
    int16_t m1; // omni
    int16_t m2; // omni
    int16_t m3; // omni
    int16_t m4; // omni
    int16_t m5; // arm
    int16_t gm; // GM6020
    int16_t m6; // updown
    int16_t m7; // collect
    // int16_t m8; // collect(使用しない)
    uint16_t gpio; // GPIO
};

struct __attribute__((packed)) Controller_Packet
{
    bool batsu_state : 1;
    bool maru_state : 1;
    bool sankaku_state : 1;
    bool shikaku_state : 1;

    bool ue_state : 1;
    bool shita_state : 1;
    bool hidari_state : 1;
    bool migi_state : 1;

    bool l1_state : 1;
    bool r1_state : 1;
    bool l2_state : 1;
    bool r2_state : 1;
    
    bool create_state : 1;
    bool option_state : 1;

    bool l3_state : 1;
    bool r3_state : 1;
    
    bool ps_state : 1;

    int8_t lx_state;
    int8_t ly_state;
    int8_t rx_state;
    int8_t ry_state;
};

using Jetson_Packet = Controller_Packet;
