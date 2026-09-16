"""ロボマス開発ボードへそのまま送れるUARTフレームをJetson側で組み立てる。

2026-08-07以降、自動走行(DualSense L2押下中)の足回り指令はここで組んだ
バイト列がそのまま開発ボードへ届く。bacon6 は受け取ったバイト列を
1バイトも書き換えずUARTへ流す（bacon_gateway/include/passthrough.hpp）。

したがってこのモジュールは bacon6 側と完全に一致していなければならない:

  ・フレーム形式  [コマンドID, 値上位, 値下位] の3バイト組をCOBSで包む
                  → bacon_gateway/src/uart.cpp の encode_robomas_frame
  ・コマンドID    → bacon_gateway/include/value.hpp の cmd_*
  ・車輪への換算  → bacon_gateway/src/move.cpp の OMNI::mix_velocity と
                    bacon_gateway/include/value.hpp の auto_* 定数

どれかを変えるときは必ず両方を同時に直し、ctest (uart_passthrough /
omni_velocity) と pytest (test_robomas_uart.py) の両方を通すこと。
"""

import math

# --- ロボマス開発ボードのコマンドID (value.hpp と同一) ---------------------
CMD_OMNI = (0, 1, 2, 3)   # 速度制御 C620 ID1〜4
CMD_ARM = 44              # シン・位相制御 M2006/M3508 ID5
CMD_GM = 61               # シン・位相制御 GM6020 ID6
CMD_UPDOWN = 6            # 速度制御 M2006 ID7
CMD_COLLECT = 7           # 速度制御 M2006 ID8
CMD_GPIO = 254            # GPIO ON/OFF

# bacon6 が完全フレームを組むときの並び順（expand_packet_commands と同一）。
# bacon6 はここに無いIDを「egg8が送っていないコマンド」として補完する。
ROBOMAS_FRAME_ORDER = (
    CMD_OMNI[0], CMD_OMNI[1], CMD_OMNI[2], CMD_OMNI[3],
    CMD_ARM, CMD_GM, CMD_UPDOWN, CMD_COLLECT, CMD_GPIO,
)

# UDPのv4コマンドで運べるUARTバイト列の上限（udp.hpp の v4_max_uart_bytes）
MAX_UART_FRAME_BYTES = 64
MIN_UART_FRAME_BYTES = 5   # 1コマンド(3バイト)のCOBS

# --- 自動走行の較正 (value.hpp の auto_* と同一) ---------------------------
# 並進1 m/s あたりの各輪値。旋回はCAD上の車輪中心半幅 x+y を腕長として使う。
AUTO_UNITS_PER_MPS = 7202.0
AUTO_ROTATION_LEVER_ARM_M = 0.666144
AUTO_UNITS_PER_RADPS = AUTO_UNITS_PER_MPS * AUTO_ROTATION_LEVER_ARM_M
# ROS body frame から手動ミキサの入力へ渡すときの符号。実機挙動から確定済み
# （導出の経緯は value.hpp のコメントに全部ある。推測で反転しないこと）。
# 2026-08-07: 横方向が実機で逆であることを走行で観測したので +1.0 へ。
AUTO_LATERAL_SIGN = 1.0    # wire vy(左+) → ミキサの横入力（反転なし）
AUTO_FORWARD_SIGN = 1.0    # wire vx(前+) → ミキサの並進入力
AUTO_TURN_SIGN = -1.0      # wire ω(CCW+) → ミキサの旋回入力
AUTO_WHEEL_LIMIT = 10000   # 全自動走行プロファイル共通の各輪指令上限。

INT16_MIN = -32768
INT16_MAX = 32767


class UartFrameError(ValueError):
    """フレームを組めない（値域外・長すぎる等）。"""


def _lround(value: float) -> int:
    """C++ の std::lround と同じ丸め（0から遠いほうへ半数丸め）。

    Pythonの round() は偶数丸めなので、そのまま使うと bacon6 の
    OMNI::mix_velocity と1カウントずれる値が出る。
    """
    return int(math.floor(value + 0.5)) if value >= 0.0 else int(math.ceil(value - 0.5))


def encode_cobs(payload: bytes) -> bytes:
    """bacon_gateway/src/cobs.cpp の encode_cobs と同一。

    戻り値は必ず len(payload) + 2 バイト（254連続の非ゼロが無い範囲）で、
    末尾はフレーム区切りの 0x00。
    """
    out = bytearray()
    code_index = 0
    out.append(0)
    code = 1

    for byte in payload:
        if byte == 0:
            out[code_index] = code
            code_index = len(out)
            out.append(0)
            code = 1
        else:
            out.append(byte)
            code += 1
            if code == 0xFF:
                out[code_index] = code
                code_index = len(out)
                out.append(0)
                code = 1

    out[code_index] = code
    out.append(0)
    return bytes(out)


def decode_cobs(encoded: bytes) -> bytes:
    """テスト・診断用。encode_cobs の逆（末尾の区切り0x00は渡さないこと）。"""
    out = bytearray()
    index = 0
    length = len(encoded)
    while index < length:
        code = encoded[index]
        index += 1
        if code == 0:
            raise UartFrameError('COBS code byte must not be zero')
        for _ in range(1, code):
            if index >= length:
                raise UartFrameError('COBS block runs past the end of the frame')
            out.append(encoded[index])
            index += 1
        if code != 0xFF and index < length:
            out.append(0)
    return bytes(out)


def build_robomas_frame(commands) -> bytes:
    """[(コマンドID, 値), ...] を開発ボードへ出せる1フレームへ組む。

    返るのはCOBS済みバイト列（末尾の区切り0x00を含む）。bacon6 はこれを
    そのままUARTへ書き出す。
    """
    entries = list(commands)
    if not entries:
        raise UartFrameError('a UART frame needs at least one command')

    payload = bytearray()
    for command_id, value in entries:
        command_id = int(command_id)
        value = int(value)
        if not 0 <= command_id <= 0xFF:
            raise UartFrameError(f'command id out of range: {command_id}')
        if not INT16_MIN <= value <= INT16_MAX:
            raise UartFrameError(f'command value out of int16 range: {value}')
        payload.append(command_id)
        payload.append((value >> 8) & 0xFF)
        payload.append(value & 0xFF)

    frame = encode_cobs(bytes(payload))
    if not MIN_UART_FRAME_BYTES <= len(frame) <= MAX_UART_FRAME_BYTES:
        raise UartFrameError(
            f'UART frame must be {MIN_UART_FRAME_BYTES}..{MAX_UART_FRAME_BYTES} '
            f'bytes, got {len(frame)}'
        )
    return frame


def parse_robomas_frame(frame: bytes) -> list:
    """テスト・診断用。build_robomas_frame の逆（bacon6の解析と同じ判定）。"""
    if len(frame) < MIN_UART_FRAME_BYTES or len(frame) > MAX_UART_FRAME_BYTES:
        raise UartFrameError('UART frame length out of range')
    if frame[-1] != 0:
        raise UartFrameError('UART frame must end with the COBS delimiter')
    if 0 in frame[:-1]:
        raise UartFrameError('UART frame must contain exactly one COBS packet')

    payload = decode_cobs(frame[:-1])
    if not payload or len(payload) % 3 != 0:
        raise UartFrameError('UART payload must be a whole number of commands')

    commands = []
    for offset in range(0, len(payload), 3):
        raw = (payload[offset + 1] << 8) | payload[offset + 2]
        value = raw - 0x10000 if raw & 0x8000 else raw
        commands.append((payload[offset], value))
    return commands


def mix_velocity(vx_mps: float, vy_mps: float, wz_radps: float, *, wheel_limit=None) -> tuple:
    """物理速度(ROS body frame)を各輪値へ変換する。

    wheel_limitは全モード共通で10000以下。同じ上限を渡した
    bacon_gateway/src/move.cpp の OMNI::mix_velocity と1カウントも違わない
    こと。戻り値は ((m1, m2, m3, m4), scale)。scale は各輪が上限を超えた
    ときの比例縮小率 k (<=1) で、方向は保たれる。
    """
    if wheel_limit is None:
        wheel_limit = AUTO_WHEEL_LIMIT
    elif not isinstance(wheel_limit, int) or not 0 < wheel_limit <= 10000:
        raise ValueError('explicit wheel limit must be an integer in 1..10000')
    ux = float(vy_mps) * AUTO_UNITS_PER_MPS * AUTO_LATERAL_SIGN
    uy = float(vx_mps) * AUTO_UNITS_PER_MPS * AUTO_FORWARD_SIGN
    ut = float(wz_radps) * AUTO_UNITS_PER_RADPS * AUTO_TURN_SIGN

    mixed = (
        ux - uy + ut,
        -ux - uy + ut,
        -ux + uy + ut,
        ux + uy + ut,
    )

    peak = max(abs(component) for component in mixed)
    scale = 1.0
    if peak > float(wheel_limit):
        scale = float(wheel_limit) / peak

    wheels = tuple(_lround(component * scale) for component in mixed)
    return wheels, scale


def drive_commands(vx_mps: float, vy_mps: float, wz_radps: float, *, wheel_limit=None) -> tuple:
    """速度指令から、足回りだけのコマンド列と各輪値・縮小率を返す。

    機構(ARM/GM/昇降/回収/GPIO)はここに含めない。含めなければ bacon6 が
    MU3プロポのボタンから補完するので、自動走行中も手元で機構を操作できる。
    将来Jetsonから機構も出すときは、ここに (CMD_ARM, value) 等を足せば
    bacon6 は自動的にそのコマンドを補完しなくなる。
    """
    wheels, scale = mix_velocity(vx_mps, vy_mps, wz_radps, wheel_limit=wheel_limit)
    commands = [(CMD_OMNI[index], wheels[index]) for index in range(4)]
    return commands, wheels, scale


def drive_frame(vx_mps: float, vy_mps: float, wz_radps: float, *, wheel_limit=None) -> tuple:
    """速度指令から、そのまま開発ボードへ出せるUARTフレームを組む。"""
    commands, wheels, scale = drive_commands(vx_mps, vy_mps, wz_radps, wheel_limit=wheel_limit)
    return build_robomas_frame(commands), wheels, scale
