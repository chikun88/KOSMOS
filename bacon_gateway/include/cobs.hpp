#pragma once

#include <cstddef>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// Maximum encoded storage, including the terminating zero. Returns zero if
// the calculation would overflow size_t.
size_t cobs_encoded_max_size(size_t length);

// Checked variants return zero without writing outside destination_capacity.
// The decoder takes encoded bytes WITHOUT the terminating zero and rejects
// malformed blocks and zeros inside those bytes.
size_t encode_cobs_bounded(const uint8_t* source, size_t length,
                           uint8_t* destination, size_t destination_capacity);
size_t decode_cobs_bounded(const uint8_t* source, size_t length,
                           uint8_t* destination, size_t destination_capacity);

// Compatibility entry points. The caller must reserve cobs_encoded_max_size
// bytes for encoding and at least length bytes for decoding. Prefer the
// checked variants for buffers whose capacity is known.
size_t encode_cobs(const uint8_t* source, size_t length, uint8_t* destination);
size_t decode_cobs(const uint8_t* source, size_t length, uint8_t* destination);

#ifdef __cplusplus
}
#endif
