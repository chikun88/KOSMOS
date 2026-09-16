#pragma once
#include <cstdint>

// Dedicated neutral MU3 frames: 80 80 80 80 00 00 (C0 | slot<<1 | nonce<<5).
// Slot 0 releases navigation. F7 is reserved; bit6=0 selects bank 16..31.
// Bit0 (PS/emergency stop) is never accepted as navigation.
class Mu3Navigation {
public:
    uint8_t slot = 0;
    uint16_t sequence = 0;
    bool stop_pending = false;

    void receive(const uint8_t* p) {
        const bool neutral = p[0] == 128 && p[1] == 128 &&
            p[2] == 128 && p[3] == 128 && p[4] == 0 && p[5] == 0;
        const bool command = neutral && (p[6] & 0x81) == 0x80;
        const uint8_t requested = command ? (((p[6] >> 1) & 15) | ((p[6] & 0x40) ? 0 : 16)) : 0;
        if (!requested) {
            release();
            ready = neutral && (p[6] == 0 || command);
            return;
        }
        if (!ready) return; // No restart after loss until an explicit release.
        if (!slot || token != p[6]) {
            slot = requested;
            token = p[6];
            ++sequence;
        }
    }
    void timeout() { release(); ready = false; }
    bool allow_auto(bool jetson_requested) {
        // Hold the Pi gate closed until Jetson has acknowledged disarm.
        if (!jetson_requested) stop_pending = false;
        return !stop_pending;
    }
private:
    bool ready = false;
    uint8_t token = 0;
    void release() {
        if (slot) { slot = 0; ++sequence; stop_pending = true; }
    }
};
