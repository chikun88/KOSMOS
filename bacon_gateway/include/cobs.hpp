# pragma once

#include <cstddef>
#include <cstdint>

#ifdef __cplusplus
extern "C"
{
    #endif

    /**
     * @brief
     * @param [in] source
     * @param [in] length
     * @param [out] destination
     * @return size_t
     */
    size_t encode_cobs(const uint8_t* source, size_t length, uint8_t* destination);
    size_t decode_cobs(const uint8_t* source, size_t length, uint8_t* destination);

    #ifdef __cplusplus
};

#endif
