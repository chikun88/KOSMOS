#pragma once
#include <cstdint>
#include <cstddef>
#include <string>

class MU3
{
    private:
        int fd;
        std::string rx_buffer; // stringに戻す

    public:
        MU3(const char* device_path);
        ~MU3();
    
        bool is_open() const;
        int send(const uint8_t* data, size_t length);
        int receive(uint8_t* out_data, size_t expected_length);
};
