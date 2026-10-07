#include "cobs.hpp"

#include <limits>

size_t cobs_encoded_max_size(size_t length)
{
    const size_t extra = length / 254;
    const size_t maximum = std::numeric_limits<size_t>::max();
    if (length > maximum - 2 || extra > maximum - length - 2)
    {
        return 0;
    }
    return length + extra + 2;
}

size_t encode_cobs_bounded(const uint8_t* source, size_t length,
                           uint8_t* destination, size_t destination_capacity)
{
    const size_t required = cobs_encoded_max_size(length);
    if (destination == nullptr || (source == nullptr && length != 0) ||
        required == 0 || destination_capacity < required)
    {
        return 0;
    }

    size_t destination_index = 1;
    size_t code_index = 0;
    uint8_t code = 1;
    for (size_t source_index = 0; source_index < length; ++source_index)
    {
        const uint8_t byte = source[source_index];
        if (byte == 0)
        {
            destination[code_index] = code;
            code_index = destination_index++;
            code = 1;
        }
        else
        {
            destination[destination_index++] = byte;
            if (++code == 0xFF)
            {
                destination[code_index] = code;
                code_index = destination_index++;
                code = 1;
            }
        }
    }
    destination[code_index] = code;
    destination[destination_index++] = 0;
    return destination_index;
}

size_t decode_cobs_bounded(const uint8_t* source, size_t length,
                           uint8_t* destination, size_t destination_capacity)
{
    if (source == nullptr || destination == nullptr || length == 0)
    {
        return 0;
    }

    // Validate the complete frame before touching the destination. This also
    // avoids exposing a partly decoded motor command after malformed input.
    size_t source_index = 0;
    size_t decoded_size = 0;
    while (source_index < length)
    {
        const uint8_t code = source[source_index++];
        const size_t block_size = code == 0 ? 0 : static_cast<size_t>(code - 1);
        if (code == 0 || block_size > length - source_index)
        {
            return 0;
        }
        for (size_t i = 0; i < block_size; ++i)
        {
            if (source[source_index + i] == 0)
            {
                return 0;
            }
        }
        source_index += block_size;
        const size_t output_size = block_size +
            ((code != 0xFF && source_index < length) ? 1 : 0);
        if (decoded_size > destination_capacity ||
            output_size > destination_capacity - decoded_size)
        {
            return 0;
        }
        decoded_size += output_size;
    }

    source_index = 0;
    size_t destination_index = 0;
    while (source_index < length)
    {
        const uint8_t code = source[source_index++];
        for (size_t i = 1; i < code; ++i)
        {
            destination[destination_index++] = source[source_index++];
        }
        if (code != 0xFF && source_index < length)
        {
            destination[destination_index++] = 0;
        }
    }
    return decoded_size;
}

size_t encode_cobs(const uint8_t* source, size_t length, uint8_t* destination)
{
    return encode_cobs_bounded(source, length, destination,
                               cobs_encoded_max_size(length));
}

size_t decode_cobs(const uint8_t* source, size_t length, uint8_t* destination)
{
    return decode_cobs_bounded(source, length, destination, length);
}
