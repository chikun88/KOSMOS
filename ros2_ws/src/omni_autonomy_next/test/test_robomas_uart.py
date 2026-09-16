"""egg8が組むUARTフレームと、それを運ぶv4コマンドの検証。

このフレームは bacon6 が1バイトも書き換えずに開発ボードへ流す。つまり
ここでのバイト列が、そのまま実機のモーター指令になる。bacon_gateway 側の
ctest (uart_passthrough / omni_velocity) と対になっているので、片方だけ
直してはいけない。
"""

import pytest

from omni_autonomy_next.motor_udp_protocol import (
    PROTOCOL_MAGIC,
    ProtocolError,
    V4_COMMAND_VERSION,
    V4_HEADER_SIZE,
    decode_v4_command,
    encode_v4_command,
)
from omni_autonomy_next.robomas_uart import (
    AUTO_UNITS_PER_MPS,
    AUTO_UNITS_PER_RADPS,
    AUTO_WHEEL_LIMIT,
    CMD_ARM,
    CMD_GM,
    CMD_OMNI,
    ROBOMAS_FRAME_ORDER,
    UartFrameError,
    build_robomas_frame,
    drive_frame,
    encode_cobs,
    mix_velocity,
    parse_robomas_frame,
)


def test_cobs_frame_is_payload_plus_two_and_zero_terminated():
    # bacon6 の decode_cobs は末尾の区切りを含めると失敗するので、
    # 区切りが1個だけ末尾に付くことがパススルーの前提条件になる。
    payload = bytes([0x00, 0x01, 0xFE, 0x00])
    frame = encode_cobs(payload)
    assert len(frame) == len(payload) + 2
    assert frame[-1] == 0
    assert 0 not in frame[:-1]


def test_uart_frame_round_trips_through_the_wire_format():
    commands = [(CMD_OMNI[0], 1000), (CMD_OMNI[1], -1000),
                (CMD_OMNI[2], 0), (CMD_OMNI[3], 32767)]
    frame = build_robomas_frame(commands)
    assert parse_robomas_frame(frame) == commands


def test_values_are_sent_big_endian_like_the_gateway():
    # bacon_gateway/src/uart.cpp: 上位バイト -> 下位バイトの順。
    frame = build_robomas_frame([(CMD_GM, 0x0102)])
    assert parse_robomas_frame(frame) == [(CMD_GM, 0x0102)]
    # COBSを剥がした生ペイロードが [ID, 上位, 下位] であること
    assert frame[1:4] == bytes([CMD_GM, 0x01, 0x02])


def test_frame_rejects_out_of_range_values():
    with pytest.raises(UartFrameError):
        build_robomas_frame([(CMD_ARM, 40000)])
    with pytest.raises(UartFrameError):
        build_robomas_frame([(300, 0)])
    with pytest.raises(UartFrameError):
        build_robomas_frame([])
    with pytest.raises(UartFrameError):
        # 21コマンド = 生63バイト -> COBS 65バイトで上限64を超える
        build_robomas_frame([(1, 0)] * 21)


def test_full_nine_command_frame_still_fits_the_transport():
    frame = build_robomas_frame([(cmd, 0) for cmd in ROBOMAS_FRAME_ORDER])
    # 将来egg8が機構も送るようになっても1パケットに収まること
    assert len(frame) == len(ROBOMAS_FRAME_ORDER) * 3 + 2


def test_mixer_matches_the_gateway_calibration():
    # bacon_gateway/tests/test_omni_velocity.cpp と同じ期待値。
    # ここがずれると、パススルーした瞬間に走行方向が変わる。
    wheels, scale = mix_velocity(0.5, 0.0, 0.0)
    assert wheels == (-3601, -3601, 3601, 3601)
    assert scale == 1.0

    # +vy は左。2026-08-07 に実機で左右が逆であることを測り、横は反転せずに
    # 渡す（AUTO_LATERAL_SIGN = +1.0）と確定した。
    wheels, scale = mix_velocity(0.0, 0.5, 0.0)
    assert wheels == (3601, -3601, -3601, 3601)
    assert scale == 1.0

    wheels, scale = mix_velocity(0.0, 0.0, 1.0)
    assert wheels == (-4798, -4798, -4798, -4798)
    assert scale == 1.0


def test_saturation_scales_every_wheel_by_one_common_factor():
    wheels, scale = mix_velocity(1.0, 0.0, 1.0)
    raw = [
        -AUTO_UNITS_PER_MPS - AUTO_UNITS_PER_RADPS,
        -AUTO_UNITS_PER_MPS - AUTO_UNITS_PER_RADPS,
        AUTO_UNITS_PER_MPS - AUTO_UNITS_PER_RADPS,
        AUTO_UNITS_PER_MPS - AUTO_UNITS_PER_RADPS,
    ]
    peak = max(abs(component) for component in raw)
    assert scale == pytest.approx(AUTO_WHEEL_LIMIT / peak)
    assert max(abs(wheel) for wheel in wheels) == AUTO_WHEEL_LIMIT
    # 縮小は方向を保つ: 比が保たれていること
    assert wheels[0] == wheels[1]
    assert wheels[2] == wheels[3]


def test_halfway_values_round_away_from_zero_like_std_lround():
    # Pythonのround()は偶数丸めなので、そのまま使うとC++と1カウントずれる。
    # 0.5カウントちょうどになる速度で符号対称を確認する。
    half_mps = 0.5 / AUTO_UNITS_PER_MPS
    positive, _ = mix_velocity(half_mps, 0.0, 0.0)
    negative, _ = mix_velocity(-half_mps, 0.0, 0.0)
    assert positive == (-1, -1, 1, 1)
    assert negative == (1, 1, -1, -1)


def test_drive_frame_carries_only_the_four_wheel_commands():
    # 機構を含めないので、bacon6 が MU3 のボタンから補完し続ける。
    frame, wheels, _scale = drive_frame(0.5, 0.0, 0.0)
    commands = parse_robomas_frame(frame)
    assert [command for command, _ in commands] == list(CMD_OMNI)
    assert [value for _, value in commands] == list(wheels)


def test_v4_command_round_trips_and_carries_the_frame_untouched():
    frame, wheels, _scale = drive_frame(0.25, -0.1, 0.3)
    data = encode_v4_command(
        uart_frame=frame,
        seq=1234,
        t_tx_us=0xDEADBEEF,
        vx_mps=0.25,
        vy_mps=-0.1,
        wz_radps=0.3,
        buttons0=0x02,
        auto_request=True,
    )
    assert data[0] == PROTOCOL_MAGIC
    assert data[1] == V4_COMMAND_VERSION
    assert len(data) == V4_HEADER_SIZE + len(frame) + 2

    relayed, velocity, buttons, flags, seq, t_tx_us = decode_v4_command(data)
    # bacon6 が開発ボードへ出すのは、egg8 が組んだこのバイト列そのもの
    assert relayed == frame
    assert parse_robomas_frame(relayed) == [
        (CMD_OMNI[index], wheels[index]) for index in range(4)
    ]
    assert velocity == (250, -100, 300)
    assert buttons == (0x02, 0, 0)
    assert flags & 0x01
    assert seq == 1234
    assert t_tx_us == 0xDEADBEEF


def test_v4_command_rejects_corruption():
    frame, _wheels, _scale = drive_frame(0.0, 0.0, 0.0)
    data = bytearray(encode_v4_command(
        uart_frame=frame, seq=1, t_tx_us=2,
    ))
    data[V4_HEADER_SIZE] ^= 0xFF
    with pytest.raises(ProtocolError):
        decode_v4_command(bytes(data))

    truncated = encode_v4_command(uart_frame=frame, seq=1, t_tx_us=2)[:-1]
    with pytest.raises(ProtocolError):
        decode_v4_command(truncated)
