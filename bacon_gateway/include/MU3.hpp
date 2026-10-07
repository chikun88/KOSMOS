#pragma once
#include <cstdint>
#include <cstddef>
#include <string>

class MU3
{
    private:
        int fd;
        std::string rx_buffer;
        bool exclusive;
        bool needs_resynchronization;
        void close_port();

    public:
        MU3(const char* device_path);
        ~MU3();
        MU3(const MU3&) = delete;
        MU3& operator=(const MU3&) = delete;
        MU3(MU3&&) = delete;
        MU3& operator=(MU3&&) = delete;
    
        bool is_open() const;
        // Nonblocking: zero means the complete command was accepted by the
        // driver; -1 means failure (including a full transmit queue).
        int send(const uint8_t* data, size_t length);
        // Discard both buffered and kernel input after a control-loop stall.
        // Serial records have no timestamp: pre-stall bytes cannot prove that
        // the controller is still alive or that its current stick is held.
        bool discard_input();
        // Return one complete, valid record in arrival order. Callers may
        // drain a bounded batch, processing stop/button edges from every
        // record and applying the last stick sample for that control cycle.
        // Incomplete records are retained; malformed records never change
        // out_data. Zero means no valid record, -1 an argument or I/O error.
        int receive(uint8_t* out_data, size_t expected_length);
};
