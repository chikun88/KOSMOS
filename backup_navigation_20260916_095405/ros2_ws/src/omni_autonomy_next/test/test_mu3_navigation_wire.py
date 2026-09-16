import struct
import pytest
from omni_autonomy_next.motor_udp_protocol import decode_telemetry, crc16_ccitt, ProtocolError


def frame(marker, sequence):
    data = bytearray(48)
    data[0:2] = bytes([0xB6, 5])
    data[29] = marker
    struct.pack_into('<H', data, 44, sequence)
    struct.pack_into('<H', data, 46, crc16_ccitt(data[:46]))
    return data


@pytest.mark.parametrize('slot', range(16))
def test_gateway_reserved_fields_decode(slot):
    t = decode_telemetry(frame(0x80 | slot, 65535))
    assert t.remote_navigation_slot == slot
    assert t.remote_navigation_sequence == 65535


def test_old_gateway_does_not_claim_command_support():
    t = decode_telemetry(frame(0, 0))
    assert t.remote_navigation_slot is None


def test_command_bit_corruption_is_rejected():
    data = frame(0x81, 4)
    data[29] ^= 2
    with pytest.raises(ProtocolError): decode_telemetry(data)
