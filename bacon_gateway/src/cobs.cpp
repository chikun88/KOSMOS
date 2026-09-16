#include "cobs.hpp"

size_t encode_cobs(const uint8_t* source, size_t length, uint8_t* destination)
{
    uint8_t* dest_ptr = destination;
    const uint8_t* src_ptr = source;
    uint8_t* code_ptr = dest_ptr++;
    uint8_t code = 1;

    while(length--)
    {
        if(*src_ptr == 0)
        {
            *code_ptr = code;
            code_ptr = dest_ptr++;
            code = 1;
        }
        else
        {
            *dest_ptr++ = *src_ptr;
            code++;

            if(code == 0xFF)
            {
                *code_ptr = code;
                code_ptr = dest_ptr++;
                code = 1;
            }
        }

        src_ptr++;
    }

    *code_ptr = code;
    *dest_ptr = 0;

    return dest_ptr - destination + 1;
}

size_t decode_cobs(const uint8_t* source, size_t length, uint8_t* destination)
{
    size_t src_index = 0;
    size_t dst_index = 0;

    while (src_index < length)
    {
        uint8_t code = source[src_index++];
        if (code == 0)
        {
            return 0;
        }

        for (uint8_t i = 1; i < code; ++i)
        {
            if (src_index >= length)
            {
                return 0;
            }
            destination[dst_index++] = source[src_index++];
        }

        if (code != 0xFF && src_index < length)
        {
            destination[dst_index++] = 0;
        }
    }

    return dst_index;
}
