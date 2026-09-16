import struct
from dataclasses import dataclass

from .robomas_uart import (
    MAX_UART_FRAME_BYTES,
    MIN_UART_FRAME_BYTES,
)


JETSON_PACKET_SIZE = 7
MU3_CONTROLLER_PACKET_SIZE = 7
AXIS_MIN = -127
AXIS_MAX = 127
BYTE_MIN = 0
BYTE_MAX = 255
MU3_L2_BUTTON_MASK = 0x02

# --- v2/v3 コマンド / テレメトリ（Raspberry Pi側 udp.cpp と同一レイアウト） ---
PROTOCOL_MAGIC = 0xB6
V2_COMMAND_VERSION = 0x02
TELEMETRY_VERSION = 0x03
V3_COMMAND_VERSION = 0x04
TELEMETRY_EXT_VERSION = 0x05
V4_COMMAND_VERSION = 0x06
V2_COMMAND_SIZE = 20
V3_COMMAND_SIZE = 24
TELEMETRY_SIZE = 32
TELEMETRY_EXT_SIZE = 48
# v4は可変長: 固定ヘッダ20B + UARTバイト列 + CRC 2B
V4_HEADER_SIZE = 20
V4_MIN_COMMAND_SIZE = V4_HEADER_SIZE + MIN_UART_FRAME_BYTES + 2
V4_MAX_COMMAND_SIZE = V4_HEADER_SIZE + MAX_UART_FRAME_BYTES + 2
V2_FLAG_AUTO_REQUEST = 0x01
V2_FLAG_ESTOP = 0x02
TELEMETRY_FLAG_AUTO_ENGAGED = 0x01
TELEMETRY_FLAG_MU3_ALIVE = 0x02
TELEMETRY_FLAG_LINK_ALIVE = 0x04
TELEMETRY_FLAG_UART_OPEN = 0x08
TELEMETRY_FLAG_ESTOP_ACTIVE = 0x10
TELEMETRY_FLAG2_CMD_WAS_V2 = 0x01
TELEMETRY_FLAG2_CMD_WAS_V3 = 0x02
TELEMETRY_FLAG2_LINK_DEGRADED = 0x04
TELEMETRY_FLAG2_CONTROLLED_STOP = 0x08
TELEMETRY_FLAG2_FAULT_LATCHED = 0x10
TELEMETRY_FLAG2_REARM_REQUIRED = 0x20
TELEMETRY_FLAG2_CMD_WAS_V4 = 0x40

_JETSON_PACKET = struct.Struct('=bbbbBBB')
# magic, version, flags, buttons0..2, lx..ry, seq, t_tx_us, reserved (CRCは別途付加)
_V2_COMMAND_BODY = struct.Struct('<BBBBBBbbbbHIH')
# magic, version, flags, buttons0..2, vx, vy, w [mm/s, mrad/s], seq, t_tx_us, reserved(4B)
_V3_COMMAND_BODY = struct.Struct('<BBBBBBhhhHII')
# magic, version, flags, buttons0..2, uart_len, reserved, seq, t_tx_us,
# 参考速度 vx/vy/w (制御には使わない。Piがテレメトリへエコーするだけ)
_V4_COMMAND_HEADER = struct.Struct('<BBBBBBBBHIhhh')
_TELEMETRY = struct.Struct('<BBBBHHIIIHHbbbbBBH')
_TELEMETRY_EXT = struct.Struct('<BBBBHHIIIHHbbbbBBhhhhhhhHH')


class ProtocolError(ValueError):
    pass


@dataclass(frozen=True)
class JetsonPacket:
    """Signed drive axes used before encoding for the selected transport."""

    lx_state: int = 0
    ly_state: int = 0
    rx_state: int = 0
    ry_state: int = 0
    reserved0: int = 0
    reserved1: int = 0
    reserved2: int = 0

    @classmethod
    def zero(cls) -> 'JetsonPacket':
        return cls()

    def drivetrain_axes(self) -> list:
        return [
            self.lx_state,
            self.ly_state,
            self.rx_state,
            self.ry_state,
        ]


def _clamp_axis(value: float) -> int:
    rounded = int(round(float(value)))
    return max(AXIS_MIN, min(AXIS_MAX, rounded))


def _clamp_byte(value: int) -> int:
    return max(BYTE_MIN, min(BYTE_MAX, int(value)))


def _scale_axis(value: float, maximum: float, sign: float = 1.0) -> int:
    maximum = float(maximum)
    if maximum <= 0.0:
        raise ProtocolError('axis maximum must be positive')
    return _clamp_axis(float(value) * float(sign) * AXIS_MAX / maximum)


def encode_jetson_packet(packet: JetsonPacket) -> bytes:
    return _JETSON_PACKET.pack(
        _clamp_axis(packet.lx_state),
        _clamp_axis(packet.ly_state),
        _clamp_axis(packet.rx_state),
        _clamp_axis(packet.ry_state),
        0,
        0,
        0,
    )


def _axis_to_mu3_controller_byte(value: float, *, invert: bool = False) -> int:
    axis = _clamp_axis(-float(value) if invert else value)
    return int(axis + 128)


def _mu3_controller_byte_to_axis(value: int, *, invert: bool = False) -> int:
    axis = int(value) - 128
    if invert:
        axis = -axis
    return _clamp_axis(axis)


def encode_mu3_controller_payload(packet: JetsonPacket) -> bytes:
    """Encode axes into the seven raw bytes used by Desktop/MU3 samples."""
    return bytes([
        _axis_to_mu3_controller_byte(packet.lx_state),
        _axis_to_mu3_controller_byte(packet.ly_state, invert=True),
        _axis_to_mu3_controller_byte(packet.rx_state),
        _axis_to_mu3_controller_byte(packet.ry_state, invert=True),
        _clamp_byte(packet.reserved0),
        _clamp_byte(packet.reserved1),
        _clamp_byte(packet.reserved2),
    ])


def encode_mu3_data_frame(payload: bytes) -> bytes:
    if len(payload) > 0xFF:
        raise ProtocolError('MU3 payload length must fit in one byte')
    return b'@DT' + f'{len(payload):02X}'.encode('ascii') + payload + b'\r\n'


def decode_jetson_packet(data: bytes) -> JetsonPacket:
    if len(data) != JETSON_PACKET_SIZE:
        raise ProtocolError(
            f'Jetson packet must be exactly {JETSON_PACKET_SIZE} bytes'
        )
    return JetsonPacket(*_JETSON_PACKET.unpack(data))


def decode_mu3_controller_payload(data: bytes) -> JetsonPacket:
    if len(data) != MU3_CONTROLLER_PACKET_SIZE:
        raise ProtocolError(
            'MU3 controller payload must be exactly '
            f'{MU3_CONTROLLER_PACKET_SIZE} bytes'
        )
    return JetsonPacket(
        lx_state=_mu3_controller_byte_to_axis(data[0]),
        ly_state=_mu3_controller_byte_to_axis(data[1], invert=True),
        rx_state=_mu3_controller_byte_to_axis(data[2]),
        ry_state=_mu3_controller_byte_to_axis(data[3], invert=True),
        reserved0=data[4],
        reserved1=data[5],
        reserved2=data[6],
    )


def twist_to_jetson_packet(
    *,
    linear_x: float,
    linear_y: float,
    angular_z: float,
    max_linear_speed: float,
    max_angular_speed: float,
    linear_x_sign: float = 1.0,
    linear_y_sign: float = 1.0,
    angular_z_sign: float = 1.0,
) -> JetsonPacket:
    return JetsonPacket(
        lx_state=_scale_axis(linear_y, max_linear_speed, linear_y_sign),
        ly_state=_scale_axis(linear_x, max_linear_speed, linear_x_sign),
        rx_state=_scale_axis(angular_z, max_angular_speed, angular_z_sign),
        ry_state=0,
    )


def crc16_ccitt(data: bytes) -> int:
    """CRC-16/CCITT-FALSE (poly 0x1021, init 0xFFFF). Pi側udp.cppと同一."""
    crc = 0xFFFF
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


def encode_v2_command(
    packet: JetsonPacket,
    *,
    seq: int,
    t_tx_us: int,
    auto_request: bool = False,
    estop: bool = False,
) -> bytes:
    """seq/タイムスタンプ/CRC付き20バイトv2コマンドを生成する。

    軸はデコード後の符号系（legacyのオフセット/反転なし）をそのまま載せる。
    buttons[0..2]はlegacy payload[4..6]と同じビット配置（reserved0..2）。
    """
    flags = 0
    if auto_request:
        flags |= V2_FLAG_AUTO_REQUEST
    if estop:
        flags |= V2_FLAG_ESTOP
    body = _V2_COMMAND_BODY.pack(
        PROTOCOL_MAGIC,
        V2_COMMAND_VERSION,
        flags,
        _clamp_byte(packet.reserved0),
        _clamp_byte(packet.reserved1),
        _clamp_byte(packet.reserved2),
        _clamp_axis(packet.lx_state),
        _clamp_axis(packet.ly_state),
        _clamp_axis(packet.rx_state),
        _clamp_axis(packet.ry_state),
        int(seq) & 0xFFFF,
        int(t_tx_us) & 0xFFFFFFFF,
        0,
    )
    return body + struct.pack('<H', crc16_ccitt(body))


def decode_v2_command(data: bytes) -> tuple:
    """テスト/診断用: v2コマンドを(JetsonPacket, flags, seq, t_tx_us)へ復元する."""
    if len(data) != V2_COMMAND_SIZE:
        raise ProtocolError(
            f'v2 command must be exactly {V2_COMMAND_SIZE} bytes'
        )
    (
        magic, version, flags,
        buttons0, buttons1, buttons2,
        lx, ly, rx, ry,
        seq, t_tx_us, _reserved,
    ) = _V2_COMMAND_BODY.unpack(data[:-2])
    if magic != PROTOCOL_MAGIC or version != V2_COMMAND_VERSION:
        raise ProtocolError('v2 command header mismatch')
    (crc,) = struct.unpack('<H', data[-2:])
    if crc != crc16_ccitt(data[:-2]):
        raise ProtocolError('v2 command crc mismatch')
    packet = JetsonPacket(
        lx_state=lx,
        ly_state=ly,
        rx_state=rx,
        ry_state=ry,
        reserved0=buttons0,
        reserved1=buttons1,
        reserved2=buttons2,
    )
    return packet, flags, seq, t_tx_us


def _clamp_i16(value: int) -> int:
    return max(-32767, min(32767, int(value)))


def encode_v3_command(
    *,
    vx_mps: float,
    vy_mps: float,
    wz_radps: float,
    seq: int,
    t_tx_us: int,
    buttons0: int = 0,
    buttons1: int = 0,
    buttons2: int = 0,
    auto_request: bool = False,
    estop: bool = False,
) -> bytes:
    """物理速度指令の24バイトv3コマンドを生成する。

    wire座標系はROS body frame: vx前+ [m/s], vy左+ [m/s], ω CCW+ [rad/s]。
    mm/s・mrad/sのint16へ量子化される（分解能 1mm/s・1mrad/s）。
    Pi側はデッドゾーンなし・手動速度レンジ非依存の固定較正で
    各輪値へ直接変換する（v2のint8スティック値より高精度）。
    """
    flags = 0
    if auto_request:
        flags |= V2_FLAG_AUTO_REQUEST
    if estop:
        flags |= V2_FLAG_ESTOP
    body = _V3_COMMAND_BODY.pack(
        PROTOCOL_MAGIC,
        V3_COMMAND_VERSION,
        flags,
        _clamp_byte(buttons0),
        _clamp_byte(buttons1),
        _clamp_byte(buttons2),
        _clamp_i16(round(float(vx_mps) * 1000.0)),
        _clamp_i16(round(float(vy_mps) * 1000.0)),
        _clamp_i16(round(float(wz_radps) * 1000.0)),
        int(seq) & 0xFFFF,
        int(t_tx_us) & 0xFFFFFFFF,
        0,
    )
    return body + struct.pack('<H', crc16_ccitt(body))


def decode_v3_command(data: bytes) -> tuple:
    """テスト/診断用: v3コマンドを(vx_mmps, vy_mmps, w_mradps, buttons, flags, seq, t_tx_us)へ復元."""
    if len(data) != V3_COMMAND_SIZE:
        raise ProtocolError(
            f'v3 command must be exactly {V3_COMMAND_SIZE} bytes'
        )
    (
        magic, version, flags,
        buttons0, buttons1, buttons2,
        vx_mmps, vy_mmps, w_mradps,
        seq, t_tx_us, _reserved,
    ) = _V3_COMMAND_BODY.unpack(data[:-2])
    if magic != PROTOCOL_MAGIC or version != V3_COMMAND_VERSION:
        raise ProtocolError('v3 command header mismatch')
    (crc,) = struct.unpack('<H', data[-2:])
    if crc != crc16_ccitt(data[:-2]):
        raise ProtocolError('v3 command crc mismatch')
    return (
        vx_mmps, vy_mmps, w_mradps,
        (buttons0, buttons1, buttons2),
        flags, seq, t_tx_us,
    )


def encode_v4_command(
    *,
    uart_frame: bytes,
    seq: int,
    t_tx_us: int,
    vx_mps: float = 0.0,
    vy_mps: float = 0.0,
    wz_radps: float = 0.0,
    buttons0: int = 0,
    buttons1: int = 0,
    buttons2: int = 0,
    auto_request: bool = False,
    estop: bool = False,
) -> bytes:
    """UARTパススルーのv4コマンドを生成する。

    uart_frame は robomas_uart.build_robomas_frame が返す、開発ボードへ
    そのまま出せるCOBS済みバイト列。bacon6 はこれを1バイトも書き換えずに
    UARTへ流す。

    vx/vy/w は制御に一切使われない参考値で、bacon6 がテレメトリの
    applied_velocity へエコーするためだけに載せる（既存の診断ツールが
    そのまま動くようにするため）。実際に効くのは uart_frame の中身だけ。
    """
    if not MIN_UART_FRAME_BYTES <= len(uart_frame) <= MAX_UART_FRAME_BYTES:
        raise ProtocolError(
            f'UART frame must be {MIN_UART_FRAME_BYTES}..{MAX_UART_FRAME_BYTES} '
            f'bytes, got {len(uart_frame)}'
        )
    flags = 0
    if auto_request:
        flags |= V2_FLAG_AUTO_REQUEST
    if estop:
        flags |= V2_FLAG_ESTOP
    header = _V4_COMMAND_HEADER.pack(
        PROTOCOL_MAGIC,
        V4_COMMAND_VERSION,
        flags,
        _clamp_byte(buttons0),
        _clamp_byte(buttons1),
        _clamp_byte(buttons2),
        len(uart_frame),
        0,
        int(seq) & 0xFFFF,
        int(t_tx_us) & 0xFFFFFFFF,
        _clamp_i16(round(float(vx_mps) * 1000.0)),
        _clamp_i16(round(float(vy_mps) * 1000.0)),
        _clamp_i16(round(float(wz_radps) * 1000.0)),
    )
    body = header + bytes(uart_frame)
    return body + struct.pack('<H', crc16_ccitt(body))


def decode_v4_command(data: bytes) -> tuple:
    """テスト/診断用: v4コマンドを分解する。

    戻り値は (uart_frame, (vx_mmps, vy_mmps, w_mradps), buttons, flags,
    seq, t_tx_us)。
    """
    if not V4_MIN_COMMAND_SIZE <= len(data) <= V4_MAX_COMMAND_SIZE:
        raise ProtocolError('v4 command size out of range')
    (
        magic, version, flags,
        buttons0, buttons1, buttons2,
        uart_len, _reserved,
        seq, t_tx_us,
        vx_mmps, vy_mmps, w_mradps,
    ) = _V4_COMMAND_HEADER.unpack(data[:V4_HEADER_SIZE])
    if magic != PROTOCOL_MAGIC or version != V4_COMMAND_VERSION:
        raise ProtocolError('v4 command header mismatch')
    if len(data) != V4_HEADER_SIZE + uart_len + 2:
        raise ProtocolError('v4 command length does not match uart_len')
    (crc,) = struct.unpack('<H', data[-2:])
    if crc != crc16_ccitt(data[:-2]):
        raise ProtocolError('v4 command crc mismatch')
    uart_frame = bytes(data[V4_HEADER_SIZE:V4_HEADER_SIZE + uart_len])
    return (
        uart_frame,
        (vx_mmps, vy_mmps, w_mradps),
        (buttons0, buttons1, buttons2),
        flags, seq, t_tx_us,
    )


@dataclass(frozen=True)
class MotorTelemetry:
    """Raspberry Piから返るテレメトリ（基本32B / v3拡張48B）."""

    flags: int
    flags2: int
    pi_seq: int
    last_cmd_seq: int
    last_cmd_t_tx_us: int
    hold_us: int
    cmd_count: int
    crc_error_count: int
    stale_drop_count: int
    applied_lx: int
    applied_ly: int
    applied_rx: int
    applied_ry: int
    rx_rate_hz: int
    # 拡張形(v3受信中)のみ。基本形ではNone
    applied_vx_mmps: int = None
    applied_vy_mmps: int = None
    applied_w_mradps: int = None
    wheel_commands: tuple = None  # (m1, m2, m3, m4)
    remote_navigation_slot: int = None
    remote_navigation_sequence: int = None

    @property
    def auto_engaged(self) -> bool:
        return bool(self.flags & TELEMETRY_FLAG_AUTO_ENGAGED)

    @property
    def mu3_alive(self) -> bool:
        return bool(self.flags & TELEMETRY_FLAG_MU3_ALIVE)

    @property
    def link_alive(self) -> bool:
        return bool(self.flags & TELEMETRY_FLAG_LINK_ALIVE)

    @property
    def uart_open(self) -> bool:
        return bool(self.flags & TELEMETRY_FLAG_UART_OPEN)

    @property
    def estop_active(self) -> bool:
        return bool(self.flags & TELEMETRY_FLAG_ESTOP_ACTIVE)

    @property
    def cmd_was_v2(self) -> bool:
        return bool(self.flags2 & TELEMETRY_FLAG2_CMD_WAS_V2)

    @property
    def cmd_was_v3(self) -> bool:
        return bool(self.flags2 & TELEMETRY_FLAG2_CMD_WAS_V3)

    @property
    def cmd_was_v4(self) -> bool:
        """直近の受理コマンドがUARTパススルー(v4)だったか。"""
        return bool(self.flags2 & TELEMETRY_FLAG2_CMD_WAS_V4)

    @property
    def link_degraded(self) -> bool:
        return bool(self.flags2 & TELEMETRY_FLAG2_LINK_DEGRADED)

    @property
    def controlled_stop(self) -> bool:
        return bool(self.flags2 & TELEMETRY_FLAG2_CONTROLLED_STOP)

    @property
    def fault_latched(self) -> bool:
        return bool(self.flags2 & TELEMETRY_FLAG2_FAULT_LATCHED)

    @property
    def rearm_required(self) -> bool:
        return bool(self.flags2 & TELEMETRY_FLAG2_REARM_REQUIRED)

    @property
    def has_timestamp_echo(self) -> bool:
        return (
            (self.cmd_was_v2 or self.cmd_was_v3 or self.cmd_was_v4)
            and bool(self.last_cmd_t_tx_us)
        )

    @property
    def protocol_name(self) -> str:
        if self.cmd_was_v4:
            return 'v4'
        if self.cmd_was_v3:
            return 'v3'
        if self.cmd_was_v2:
            return 'v2'
        return 'legacy'


def decode_telemetry(data: bytes) -> MotorTelemetry:
    if len(data) == TELEMETRY_SIZE:
        (
            magic, version, flags, flags2,
            pi_seq, last_cmd_seq,
            last_cmd_t_tx_us, hold_us, cmd_count,
            crc_error_count, stale_drop_count,
            applied_lx, applied_ly, applied_rx, applied_ry,
            rx_rate_hz, _reserved, crc,
        ) = _TELEMETRY.unpack(data)
        if magic != PROTOCOL_MAGIC or version != TELEMETRY_VERSION:
            raise ProtocolError('telemetry header mismatch')
        if crc != crc16_ccitt(data[:-2]):
            raise ProtocolError('telemetry crc mismatch')
        extended = {}
    elif len(data) == TELEMETRY_EXT_SIZE:
        (
            magic, version, flags, flags2,
            pi_seq, last_cmd_seq,
            last_cmd_t_tx_us, hold_us, cmd_count,
            crc_error_count, stale_drop_count,
            applied_lx, applied_ly, applied_rx, applied_ry,
            rx_rate_hz, _reserved0,
            applied_vx_mmps, applied_vy_mmps, applied_w_mradps,
            m1, m2, m3, m4,
            _reserved1, crc,
        ) = _TELEMETRY_EXT.unpack(data)
        if magic != PROTOCOL_MAGIC or version != TELEMETRY_EXT_VERSION:
            raise ProtocolError('telemetry ext header mismatch')
        if crc != crc16_ccitt(data[:-2]):
            raise ProtocolError('telemetry ext crc mismatch')
        extended = {
            'applied_vx_mmps': applied_vx_mmps,
            'applied_vy_mmps': applied_vy_mmps,
            'applied_w_mradps': applied_w_mradps,
            'wheel_commands': (m1, m2, m3, m4),
            'remote_navigation_slot': (_reserved0 & 31) if (_reserved0 & 0xe0) == 0x80 else None,
            'remote_navigation_sequence': _reserved1 if (_reserved0 & 0xe0) == 0x80 else None,
        }
    else:
        raise ProtocolError(
            f'telemetry must be {TELEMETRY_SIZE} or {TELEMETRY_EXT_SIZE} bytes'
        )
    return MotorTelemetry(
        flags=flags,
        flags2=flags2,
        pi_seq=pi_seq,
        last_cmd_seq=last_cmd_seq,
        last_cmd_t_tx_us=last_cmd_t_tx_us,
        hold_us=hold_us,
        cmd_count=cmd_count,
        crc_error_count=crc_error_count,
        stale_drop_count=stale_drop_count,
        applied_lx=applied_lx,
        applied_ly=applied_ly,
        applied_rx=applied_rx,
        applied_ry=applied_ry,
        rx_rate_hz=rx_rate_hz,
        **extended,
    )
