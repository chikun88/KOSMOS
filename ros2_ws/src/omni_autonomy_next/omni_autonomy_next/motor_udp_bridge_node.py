import json
import math
import os
import socket
import struct
import termios
import time
from typing import Optional

# LinuxのSO_TIMESTAMPNS(_OLD)。Pythonのsocketモジュールに定数がないため直接指定。
# テレメトリ到着のカーネル時刻を取り、RTT計測からポーリング待ちを除くために使う。
_SO_TIMESTAMPNS = getattr(socket, 'SO_TIMESTAMPNS', 35)

import rclpy
from geometry_msgs.msg import Twist
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from std_msgs.msg import Bool, Float32, String
from std_srvs.srv import Trigger

from .motor_udp_protocol import (
    JetsonPacket,
    MU3_L2_BUTTON_MASK,
    ProtocolError,
    TELEMETRY_EXT_SIZE,
    TELEMETRY_SIZE,
    decode_telemetry,
    encode_jetson_packet,
    encode_mu3_controller_payload,
    encode_mu3_data_frame,
    encode_v2_command,
    encode_v3_command,
    encode_v4_command,
    quantize_velocity,
    twist_to_jetson_packet,
)
from .robomas_uart import UartFrameError, drive_frame, mix_velocity


def validate_command_calibration(linear, angular, transport, payload_format):
    for value in (linear, angular):
        if not math.isfinite(value) or not 0.0 < value <= 1.0:
            raise ValueError('command calibration scales must be finite and in (0, 1]')
    if ((linear != 1.0 or angular != 1.0)
            and (transport != 'udp' or payload_format not in ('v3_velocity', 'v4_uart'))):
        raise ValueError('command calibration requires UDP velocity or UART passthrough')


class MotorUdpBridge(Node):
    def __init__(self) -> None:
        super().__init__('motor_udp_bridge')
        self._declare_parameters()
        self.transport = str(self.get_parameter('transport').value)
        if self.transport not in ('mu3_uart', 'udp'):
            raise ValueError('transport must be mu3_uart or udp')
        self.payload_format = str(self.get_parameter('payload_format').value)
        if self.payload_format not in (
            'mu3_controller', 'jetson_axes', 'v2', 'v3_velocity', 'v4_uart'
        ):
            raise ValueError(
                'payload_format must be mu3_controller, jetson_axes, '
                'v2, v3_velocity or v4_uart'
            )
        self.remote_address = (
            str(self.get_parameter('remote_ip').value),
            int(self.get_parameter('remote_port').value),
        )
        self.local_address = (
            str(self.get_parameter('local_ip').value),
            int(self.get_parameter('local_port').value),
        )
        self.mu3_device = str(self.get_parameter('mu3_device').value)
        self.mu3_baud_rate = int(self.get_parameter('mu3_baud_rate').value)
        self.mu3_my_id = str(self.get_parameter('mu3_my_id').value)
        self.mu3_target_id = str(self.get_parameter('mu3_target_id').value)
        self.mu3_response_delay_sec = max(
            0.0, float(self.get_parameter('mu3_response_delay_sec').value)
        )
        self.max_linear_speed = float(
            self.get_parameter('max_linear_speed').value
        )
        self.max_angular_speed = float(
            self.get_parameter('max_angular_speed').value
        )
        if not all(math.isfinite(value) and value > 0.0
                   for value in (self.max_linear_speed, self.max_angular_speed)):
            raise ValueError(
                'max_linear_speed and max_angular_speed must be positive'
            )
        self.linear_x_sign = float(self.get_parameter('linear_x_sign').value)
        self.linear_y_sign = float(self.get_parameter('linear_y_sign').value)
        self.angular_z_sign = float(self.get_parameter('angular_z_sign').value)
        if not all(value in (-1.0, 1.0) for value in
                   (self.linear_x_sign, self.linear_y_sign, self.angular_z_sign)):
            raise ValueError('axis signs must be +1 or -1')
        self.linear_command_scale = float(self.get_parameter('linear_command_scale').value)
        self.angular_command_scale = float(self.get_parameter('angular_command_scale').value)
        validate_command_calibration(self.linear_command_scale, self.angular_command_scale,
                                     self.transport, self.payload_format)
        self.require_enable = bool(self.get_parameter('require_enable').value)
        self.require_healthy_telemetry = bool(
            self.get_parameter('require_healthy_telemetry').value
        )
        self.auto_rearm_when_idle = bool(
            self.get_parameter('auto_rearm_when_idle').value
        )
        self.immediate_send_on_cmd = bool(
            self.get_parameter('immediate_send_on_cmd').value
        )
        status_rate = max(
            0.0, float(self.get_parameter('status_publish_rate_hz').value)
        )
        self.status_publish_period_ns = (
            int(1.0e9 / status_rate) if status_rate > 0.0 else 0
        )
        self.last_status_publish_ns = 0
        # These parameters govern hot-path timers/watchdogs and are fixed for
        # the lifetime of this node.  Cache their converted forms once instead
        # of performing ROS parameter lookups and Duration construction on
        # every 200 Hz send or telemetry packet.
        self.command_timeout = Duration(
            seconds=float(self.get_parameter('command_timeout_sec').value)
        )
        if self.command_timeout.nanoseconds <= 0:
            raise ValueError('command_timeout_sec must be positive')
        latency_rate = max(
            1.0, float(self.get_parameter('latency_publish_rate_hz').value)
        )
        self.latency_publish_period_sec = 1.0 / latency_rate
        telemetry_rate = max(
            0.1, float(self.get_parameter('telemetry_publish_rate_hz').value)
        )
        self.telemetry_publish_period_sec = 1.0 / telemetry_rate
        self.telemetry_timeout_sec = float(
            self.get_parameter('telemetry_timeout_sec').value
        )
        if not math.isfinite(self.telemetry_timeout_sec) or self.telemetry_timeout_sec <= 0.0:
            raise ValueError('telemetry_timeout_sec must be finite and positive')
        self.warn_not_engaged_sec = float(
            self.get_parameter('warn_not_engaged_sec').value
        )
        self.max_telemetry_packets_per_tick = max(
            1,
            int(self.get_parameter('max_telemetry_packets_per_tick').value),
        )

        self.socket: Optional[socket.socket] = None
        self.mu3_fd: Optional[int] = None
        self.latest_twist = Twist()
        self.motion_profile = 'balanced'
        self.latest_command_time = None
        self.latest_command_monotonic: Optional[float] = None
        self.sequence = 0
        self.enabled = not self.require_enable
        self._explicit_disarm = False
        self.estop = False
        # Emergency stop is latched locally.  Clearing the Bool topic must not
        # resume an old command stream: a Trigger reset clears the latch, then
        # a *new* enable=true message explicitly rearms motion.  The separate
        # rearm gate is required even when require_enable is false (the normal
        # Nav2 launch mode), where the ordinary enable gate is bypassed.
        # UDP motion starts disarmed until the Pi has returned a fresh,
        # fault-free status and a *new* enable edge is received.  This avoids
        # a goal sent during boot moving the robot later when Ethernet appears.
        self.rearm_required = self.require_healthy_telemetry
        self._command_was_active = False
        # Always establish the Pi-side disarmed half of the startup handshake,
        # even if /cmd_vel and enable arrive before the first timer callback.
        self._startup_disarm_packets = 3
        # The Pi may restart independently while this node keeps running.  On
        # an AUTO drop, explicitly repeat the disarmed half of the handshake
        # before sending the next auto-request edge.
        self._pi_rearm_disarm_packets = 0
        self.last_state = ''
        # v3_velocity / v4_uart用: 次に送る物理速度指令 (vx, vy, wz)
        self._pending_velocity = (0.0, 0.0, 0.0)
        # v4_uart用: 直近に組んだUARTフレームと、その中の各輪値。
        # Piは中身を書き換えないので、これが開発ボードへ届く指令そのもの。
        self._last_uart_frame = b''
        self._last_wheel_commands = (0, 0, 0, 0)
        self._last_wheel_scale = 1.0

        # --- Piテレメトリ（v2双方向リンク）の状態 ---
        self.telemetry = None
        self.telemetry_monotonic: Optional[float] = None
        self.telemetry_count = 0
        self.telemetry_crc_errors = 0
        self.sent_count = 0
        self.rtt_last_ms: Optional[float] = None
        self.rtt_avg_ms: Optional[float] = None   # EMA (alpha=0.1)
        self.rtt_max_window_ms = 0.0              # JSON出力周期内の最大
        self.loss_pct: Optional[float] = None
        self._loss_window_start = time.monotonic()
        self._loss_window_sent = 0
        self._loss_window_cmd_count: Optional[int] = None
        self._last_link_ok: Optional[bool] = None
        self._last_auto_engaged: Optional[bool] = None
        self._last_latency_pub = 0.0
        self._last_telemetry_json_pub = 0.0
        self._last_bool_refresh = 0.0
        self._not_engaged_since: Optional[float] = None
        self._last_not_engaged_warn = 0.0

        transient_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.status_publisher = self.create_publisher(
            String, str(self.get_parameter('status_topic').value), transient_qos
        )
        self.telemetry_publisher = self.create_publisher(
            String, str(self.get_parameter('telemetry_topic').value), transient_qos
        )
        self.remote_navigation_publisher = self.create_publisher(
            String, '/mu3/navigation_input', 10
        )
        self.latency_publisher = self.create_publisher(
            Float32, str(self.get_parameter('latency_topic').value), 10
        )
        self.auto_engaged_publisher = self.create_publisher(
            Bool, str(self.get_parameter('auto_engaged_topic').value), transient_qos
        )
        self.link_ok_publisher = self.create_publisher(
            Bool, str(self.get_parameter('link_ok_topic').value), transient_qos
        )
        control_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.create_subscription(
            Twist,
            str(self.get_parameter('cmd_vel_topic').value),
            self._twist_callback,
            control_qos,
        )
        self.create_subscription(
            Bool,
            str(self.get_parameter('enable_topic').value),
            self._enable_callback,
            control_qos,
        )
        self.create_subscription(
            Bool,
            str(self.get_parameter('estop_topic').value),
            self._estop_callback,
            control_qos,
        )
        self.create_subscription(String, '/system/profile', self._profile_callback, transient_qos)
        self.create_service(
            Trigger,
            str(self.get_parameter('reset_estop_service').value),
            self._reset_estop_callback,
        )
        rate = max(5.0, float(self.get_parameter('send_rate_hz').value))
        self.create_timer(1.0 / rate, self._timer_callback)
        if self.transport == 'mu3_uart':
            self.get_logger().info(
                f'MU3 UART bridge configured: device={self.mu3_device}, '
                f'baud={self.mu3_baud_rate}, id={self.mu3_my_id}->{self.mu3_target_id}, '
                f'rate={rate:.1f} Hz'
            )
        else:
            self.get_logger().info(
                f'UDP motor bridge configured: {self.local_address} -> '
                f'{self.remote_address} at {rate:.1f} Hz, '
                f'payload_format={self.payload_format}, '
                f'require_enable={self.require_enable}'
            )

    def _declare_parameters(self) -> None:
        defaults = {
            'cmd_vel_topic': '/cmd_vel',
            'enable_topic': '/motion/enable',
            'estop_topic': '/motion/emergency_stop',
            'reset_estop_service': '/motion/reset_emergency_stop',
            'status_topic': '/motor/network_status',
            'require_enable': True,
            # Competition UDP is bidirectional. Never drive before the Pi
            # confirms the UART/fault state. Direct diagnostic UART users may
            # explicitly disable this because there is no Pi return channel.
            'require_healthy_telemetry': True,
            # Keep the drivebase ready at zero velocity. Recovery is allowed
            # only with fresh, armable telemetry and a command that quantizes
            # to zero; E-stop and stale non-zero commands remain latched.
            'auto_rearm_when_idle': True,
            'transport': 'udp',
            # 既定は後方互換のためlegacyのまま。競技構成(system.launch.py)は
            # v4_uart（egg8がUARTフレームを組み、bacon6がパススルー）。
            'payload_format': 'mu3_controller',
            'local_ip': '192.168.50.1',
            'local_port': 8888,
            'remote_ip': '192.168.50.2',
            'remote_port': 8888,
            'mu3_device': '/dev/ttyUSB0',
            'mu3_baud_rate': 19200,
            'mu3_my_id': '01',
            'mu3_target_id': '02',
            'mu3_response_delay_sec': 0.02,
            'send_rate_hz': 100.0,
            'immediate_send_on_cmd': True,
            'status_publish_rate_hz': 10.0,
            'udp_tos': 0x10,
            'udp_priority': 6,
            'udp_send_buffer_bytes': 4096,
            'telemetry_topic': '/motor/telemetry',
            'latency_topic': '/motor/link_latency_ms',
            'auto_engaged_topic': '/motor/auto_engaged',
            'link_ok_topic': '/motor/link_ok',
            'telemetry_timeout_sec': 0.35,
            'telemetry_publish_rate_hz': 10.0,
            'latency_publish_rate_hz': 20.0,
            'warn_not_engaged_sec': 0.5,
            # Bound work in the single-threaded 200 Hz timer.  Normal traffic
            # is one telemetry datagram per tick; 32 still absorbs a generous
            # burst without allowing an unbounded receive loop to starve sends.
            'max_telemetry_packets_per_tick': 32,
            'command_timeout_sec': 0.15,
            'max_linear_speed': 0.55,
            'max_angular_speed': 1.2,
            # Unit conversion for measured drive response, separate from the
            # operator speed scale. Loaded only in the hardware launch path.
            'linear_command_scale': 1.0,
            'angular_command_scale': 1.0,
            'linear_x_sign': 1.0,
            'linear_y_sign': 1.0,
            'angular_z_sign': 1.0,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

    def _open_socket(self) -> bool:
        if self.socket is not None:
            return True
        candidate = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._configure_udp_socket(candidate)
        try:
            candidate.bind(self.local_address)
            candidate.connect(self.remote_address)
        except OSError as error:
            candidate.close()
            self._publish_status('LOCAL_INTERFACE_UNAVAILABLE', error=str(error))
            return False
        # テレメトリ受信をタイマー内で取りこぼしなく行うためノンブロッキング化
        candidate.setblocking(False)
        self.socket = candidate
        bound_ip, bound_port = candidate.getsockname()
        self.get_logger().info(f'UDP motor socket bound to {bound_ip}:{bound_port}')
        return True

    def _configure_udp_socket(self, candidate: socket.socket) -> None:
        # A command source owns this endpoint exclusively. Reusing it permits
        # two bridge processes to interleave sequences and motor commands.
        send_buffer = max(
            0, int(self.get_parameter('udp_send_buffer_bytes').value)
        )
        if send_buffer:
            try:
                candidate.setsockopt(
                    socket.SOL_SOCKET, socket.SO_SNDBUF, send_buffer
                )
            except OSError as error:
                self.get_logger().warning(f'Failed to set SO_SNDBUF: {error}')
        tos = int(self.get_parameter('udp_tos').value)
        if hasattr(socket, 'IP_TOS') and tos >= 0:
            try:
                candidate.setsockopt(socket.IPPROTO_IP, socket.IP_TOS, tos)
            except OSError as error:
                self.get_logger().warning(f'Failed to set IP_TOS: {error}')
        priority = int(self.get_parameter('udp_priority').value)
        if hasattr(socket, 'SO_PRIORITY') and priority >= 0:
            try:
                candidate.setsockopt(
                    socket.SOL_SOCKET, socket.SO_PRIORITY, priority
                )
            except OSError as error:
                self.get_logger().warning(f'Failed to set SO_PRIORITY: {error}')
        try:
            candidate.setsockopt(socket.SOL_SOCKET, _SO_TIMESTAMPNS, 1)
        except OSError as error:
            self.get_logger().warning(
                f'Failed to set SO_TIMESTAMPNS (RTT will include poll wait): {error}'
            )

    def _open_mu3(self) -> bool:
        if self.mu3_fd is not None:
            return True
        try:
            candidate = os.open(
                self.mu3_device,
                os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK,
            )
            self._configure_mu3_serial(candidate)
            self._send_mu3_command(candidate, b'\r\n')
            gi_response = self._send_mu3_command(candidate, b'@GI\r\n')
            self._send_mu3_command(
                candidate, f'@EI{self.mu3_my_id}\r\n'.encode('ascii')
            )
            self._send_mu3_command(
                candidate, f'@DI{self.mu3_target_id}\r\n'.encode('ascii')
            )
        except (OSError, termios.error, ValueError) as error:
            try:
                os.close(candidate)
            except (OSError, UnboundLocalError):
                pass
            self._publish_status('MU3_UART_UNAVAILABLE', error=str(error))
            return False
        self.mu3_fd = candidate
        self.get_logger().info(
            f'MU3 UART opened on {self.mu3_device}; '
            f'GI response={gi_response!r}'
        )
        return True

    def _configure_mu3_serial(self, fd: int) -> None:
        baud_constant = getattr(termios, f'B{self.mu3_baud_rate}', None)
        if baud_constant is None:
            raise ValueError(f'Unsupported MU3 baud rate: {self.mu3_baud_rate}')
        attrs = termios.tcgetattr(fd)
        attrs[0] = 0
        attrs[1] = 0
        attrs[2] &= ~termios.CSIZE
        attrs[2] |= termios.CS8 | termios.CLOCAL | termios.CREAD
        attrs[2] &= ~termios.PARENB
        attrs[2] &= ~termios.CSTOPB
        attrs[2] &= ~getattr(termios, 'CRTSCTS', 0)
        attrs[3] = 0
        attrs[4] = baud_constant
        attrs[5] = baud_constant
        attrs[6][termios.VMIN] = 0
        attrs[6][termios.VTIME] = 0
        termios.tcflush(fd, termios.TCIOFLUSH)
        termios.tcsetattr(fd, termios.TCSANOW, attrs)

    def _send_mu3_command(self, fd: int, command: bytes) -> bytes:
        termios.tcflush(fd, termios.TCIOFLUSH)
        self._write_all(fd, command)
        termios.tcdrain(fd)
        time.sleep(0.1)
        return self._read_available(fd)

    def _write_all(self, fd: int, data: bytes) -> None:
        offset = 0
        while offset < len(data):
            written = os.write(fd, data[offset:])
            if written <= 0:
                raise OSError('serial write returned no progress')
            offset += written

    def _read_available(self, fd: int, max_bytes: int = 256) -> bytes:
        chunks = []
        remaining = max_bytes
        while remaining > 0:
            try:
                chunk = os.read(fd, remaining)
            except BlockingIOError:
                break
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        return b''.join(chunks)

    def _transport_ready(self) -> bool:
        if self.transport == 'mu3_uart':
            return self._open_mu3()
        return self._open_socket()

    def _twist_callback(self, message: Twist) -> None:
        self.latest_twist = message
        self.latest_command_time = self.get_clock().now()
        self.latest_command_monotonic = time.monotonic()
        if self.immediate_send_on_cmd:
            self._send_current_packet(trigger='cmd_vel')

    def _enable_callback(self, message: Bool) -> None:
        if not bool(message.data):
            # Operator DISARM must outlive the idle-recovery path below.
            self._explicit_disarm = True
            self.enabled = False
        if self.estop:
            # An enable received while the latch is active must not count as
            # the post-reset rearm request.
            self.enabled = False
            if self.immediate_send_on_cmd:
                self._send_current_packet(trigger='enable')
            return
        if bool(message.data) and self.rearm_required:
            # The Pi deliberately keeps telemetry.rearm_required asserted
            # after a watchdog/fault recovery until it sees a new auto-request
            # edge.  Requiring that bit to be clear before accepting the edge
            # deadlocks both sides: Jetson waits for Pi to clear it while Pi
            # waits for Jetson's request.  All other return-link and hardware
            # safety conditions must still be healthy here.
            if self._telemetry_armable():
                self.rearm_required = False
            else:
                # Do not remember an enable that arrived before the return
                # link. The operator/mission must issue a new edge after the
                # preflight condition actually becomes true.
                self.enabled = False
                if self.immediate_send_on_cmd:
                    self._send_current_packet(trigger='enable_waiting_health')
                return
        if self.require_enable:
            self.enabled = bool(message.data)
        elif bool(message.data):
            self.enabled = True
        if bool(message.data):
            self._explicit_disarm = False
        if self.immediate_send_on_cmd:
            self._send_current_packet(trigger='enable')

    def _estop_callback(self, message: Bool) -> None:
        if bool(message.data):
            self.estop = True
            self.rearm_required = True
            self.enabled = False
        # A false Bool is deliberately ignored.  Only the reset service may
        # clear the latched stop, so a transient publisher or stale message
        # cannot restart the robot.
        if self.immediate_send_on_cmd:
            self._send_current_packet(trigger='estop')

    def _reset_estop_callback(self, request, response):
        del request
        if not self.estop:
            response.success = True
            response.message = 'Emergency stop is not latched'
            return response
        self.estop = False
        self.rearm_required = True
        self.enabled = False
        if self.immediate_send_on_cmd:
            self._send_current_packet(trigger='estop_reset')
        response.success = True
        response.message = (
            'Emergency stop reset; motion remains disabled until a new '
            'enable=true message is received'
        )
        return response

    def _timer_callback(self) -> None:
        # A current command/watchdog packet has priority over diagnostics.
        # Telemetry is drained afterwards with a finite work budget so an RX
        # burst cannot indefinitely postpone the next 200 Hz send callback.
        self._send_current_packet(trigger='timer')
        self._drain_telemetry()
        self._update_link_state()

    def _send_current_packet(self, trigger: str) -> None:
        if not self._transport_ready():
            return
        now = self.get_clock().now()
        command_received_monotonic = getattr(
            self, 'latest_command_monotonic', None
        )
        if command_received_monotonic is not None:
            command_age_sec = time.monotonic() - command_received_monotonic
            command_fresh = (
                0.0 <= command_age_sec
                <= self.command_timeout.nanoseconds * 1.0e-9
            )
        else:
            # Compatibility for direct-UART diagnostics and old bags/tests.
            # Production callbacks always populate the monotonic timestamp.
            command_age_ns = (
                None if self.latest_command_time is None
                else (now - self.latest_command_time).nanoseconds
            )
            command_fresh = (
                command_age_ns is not None
                and 0 <= command_age_ns <= self.command_timeout.nanoseconds
            )
        command_seen = (
            command_received_monotonic is not None
            or self.latest_command_time is not None
        )
        invalid_command = False
        try:
            latest_velocity = (self._velocity_from_latest_twist()
                               if command_seen else (0.0, 0.0, 0.0))
            latest_packet = (self._packet_from_latest_twist()
                             if command_seen else JetsonPacket.zero())
        except (ProtocolError, ValueError, OverflowError):
            # Reject before int quantization/UART encoding, and send an
            # immediate disarmed zero instead of crashing the executor.
            invalid_command = True
            command_fresh = False
            self.rearm_required = True
            self.enabled = False
            latest_velocity = (0.0, 0.0, 0.0)
            latest_packet = JetsonPacket.zero()
        # Decide idleness from the quantized command that the Pi would
        # actually receive. Tiny planner residuals that quantize to zero must
        # not turn a harmless scheduling gap into a false motion-link fault.
        if self.payload_format == 'v4_uart':
            wire_values, _scale = mix_velocity(*latest_velocity)
        elif self.payload_format == 'v3_velocity':
            wire_values = tuple(quantize_velocity(value) for value in latest_velocity)
        else:
            wire_values = (
                latest_packet.lx_state,
                latest_packet.ly_state,
                latest_packet.rx_state,
                latest_packet.ry_state,
            )
        latest_command_is_zero = all(component == 0 for component in wire_values)
        if getattr(self, '_command_was_active', False) and not command_fresh:
            # A producer stall is a control-link loss, not an ordinary zero
            # command.  Latch rearm so a resumed publisher cannot restart the
            # robot until a new enable event (normally a new goal) is issued.
            self.rearm_required = True
            self.enabled = False
        pi_rearm_disarm = getattr(
            self, '_pi_rearm_disarm_packets', 0
        ) > 0
        if (
            getattr(self, 'auto_rearm_when_idle', False)
            and not getattr(self, '_explicit_disarm', False)
            and self.rearm_required
            and not self.estop
            and not invalid_command
            and command_seen
            and latest_command_is_zero
            and self._telemetry_armable()
            and not pi_rearm_disarm
        ):
            # A zero-speed request completes the Pi's explicit disarm->rearm
            # handshake without permitting an old motion command to restart.
            self.rearm_required = False
            self.enabled = True
        enabled = (self.enabled or not self.require_enable) and not (
            self.rearm_required
        ) and not getattr(self, '_explicit_disarm', False)
        startup_disarm = getattr(self, '_startup_disarm_packets', 0) > 0
        # The arbiter's idle command is exactly zero. Keep the Pi engaged if
        # that harmless heartbeat is briefly delayed by CPU scheduling. A
        # stale non-zero command still disarms and latches rearm above.
        safe_idle_hold = bool(
            enabled
            and command_seen
            and not command_fresh
            and latest_command_is_zero
        )
        local_request = (
            enabled
            and not self.estop
            and (command_fresh or safe_idle_hold)
            and not startup_disarm
            and not pi_rearm_disarm
        )
        telemetry_ready = self._telemetry_base_healthy()
        telemetry_armable = self._telemetry_armable()
        telemetry_required = getattr(
            self, 'require_healthy_telemetry', False
        )
        telemetry = getattr(self, 'telemetry', None)
        pi_engaged = bool(
            not telemetry_required
            or (
                telemetry_ready
                and telemetry is not None
                and telemetry.auto_engaged
            )
        )
        active = (
            local_request and command_fresh and telemetry_ready and pi_engaged
        )
        # Before motion, send only a zero-speed L2/auto request. The Pi must
        # reflect auto_engaged in a subsequent healthy telemetry packet.
        # A zero-speed auto request is the only command allowed while the Pi
        # reports its recoverable rearm_required latch.  Non-zero motion still
        # requires fully healthy telemetry and auto_engaged below.
        arm_request = local_request and telemetry_armable and not pi_engaged
        idle_hold_request = (
            safe_idle_hold and telemetry_ready and pi_engaged
        )
        self._command_was_active = active and not latest_command_is_zero
        self._pending_velocity = (
            latest_velocity if active else (0.0, 0.0, 0.0)
        )
        packet = latest_packet if active else JetsonPacket.zero()
        if active or arm_request or idle_hold_request:
            packet = JetsonPacket(
                lx_state=packet.lx_state,
                ly_state=packet.ly_state,
                rx_state=packet.rx_state,
                ry_state=packet.ry_state,
                reserved0=packet.reserved0 | MU3_L2_BUTTON_MASK,
                reserved1=packet.reserved1,
                reserved2=packet.reserved2,
            )
        try:
            data, drive_payload, response = self._send_packet(packet)
        except (OSError, termios.error) as error:
            self._publish_status('SEND_ERROR', trigger=trigger, error=str(error))
            return
        except (UartFrameError, ProtocolError) as error:
            # 組めないフレームを送るくらいなら何も送らない。Pi側の
            # command watchdog が減速停止まで持っていく。
            self._publish_status(
                'UART_FRAME_ENCODE_FAILED', trigger=trigger, error=str(error)
            )
            return
        if startup_disarm:
            self._startup_disarm_packets -= 1
        if pi_rearm_disarm:
            self._pi_rearm_disarm_packets -= 1
        sent_sequence = self.sequence
        self.sequence = (self.sequence + 1) & 0xFFFFFFFF

        if startup_disarm:
            state = 'STARTUP_DISARM_HANDSHAKE_SENT'
        elif self.estop:
            state = 'EMERGENCY_STOP_PACKET_SENT'
        elif invalid_command:
            state = 'INVALID_COMMAND_ZERO_PACKET_SENT'
        elif pi_rearm_disarm:
            state = 'PI_REARM_DISARM_HANDSHAKE_SENT'
        elif self.rearm_required:
            state = 'REARM_REQUIRED_ZERO_PACKET_SENT'
        elif not enabled:
            state = 'DISABLED_ZERO_PACKET_SENT'
        elif arm_request:
            state = 'AUTO_ARM_REQUEST_ZERO_PACKET_SENT'
        elif idle_hold_request:
            state = 'IDLE_ZERO_HOLD_PACKET_SENT'
        elif not command_fresh:
            state = 'COMMAND_TIMEOUT_ZERO_PACKET_SENT'
        elif not telemetry_ready:
            state = 'TELEMETRY_NOT_READY_ZERO_PACKET_SENT'
        else:
            state = 'JETSON_PACKET_SENT'
        # Check the throttle before constructing byte lists, the nested axes
        # mapping and the remote-description string.  Previously those call
        # arguments were allocated at the 200 Hz packet rate even though the
        # status method discarded them before serialization on 19/20 calls.
        if self._status_publish_due(state, now_ns=now.nanoseconds):
            self._emit_status(
                state,
                sent_sequence=sent_sequence,
                command_calibration={'linear': self.linear_command_scale,
                                     'angular': self.angular_command_scale},
                trigger=trigger,
                transport=self.transport,
                payload_format=self.payload_format,
                remote=self._remote_description(),
                packet_bytes=list(data),
                drive_payload_bytes=list(drive_payload),
                response=list(response),
                axes={
                    'lx_state': packet.lx_state,
                    'ly_state': packet.ly_state,
                    'rx_state': packet.rx_state,
                    'ry_state': packet.ry_state,
                },
                # v4では実際に開発ボードへ届くのはこの各輪値そのもの
                wheel_commands=(
                    list(self._last_wheel_commands)
                    if self.payload_format == 'v4_uart' else None
                ),
                uart_frame_bytes=(
                    list(self._last_uart_frame)
                    if self.payload_format == 'v4_uart' else None
                ),
            )

    def _send_packet(self, packet: JetsonPacket) -> tuple[bytes, bytes, bytes]:
        if self.transport == 'mu3_uart':
            payload = encode_mu3_controller_payload(packet)
            frame = encode_mu3_data_frame(payload)
            assert self.mu3_fd is not None
            termios.tcflush(self.mu3_fd, termios.TCIFLUSH)
            self._write_all(self.mu3_fd, frame)
            termios.tcdrain(self.mu3_fd)
            if self.mu3_response_delay_sec > 0.0:
                time.sleep(self.mu3_response_delay_sec)
            response = self._read_available(self.mu3_fd)
            return frame, payload, response

        data = self._encode_udp_payload(packet)
        assert self.socket is not None
        self.socket.send(data)
        self.sent_count += 1
        return data, data, b''

    def _profile_callback(self, message: String) -> None:
        self.motion_profile = str(message.data)

    def _velocity_from_latest_twist(self) -> tuple:
        vx = float(self.latest_twist.linear.x) * self.linear_x_sign
        vy = float(self.latest_twist.linear.y) * self.linear_y_sign
        wz = float(self.latest_twist.angular.z) * self.angular_z_sign
        if not all(math.isfinite(value) for value in (vx, vy, wz)):
            raise ProtocolError('incoming drive velocity must be finite')
        linear_speed = math.hypot(vx, vy)
        if linear_speed > self.max_linear_speed:
            scale = self.max_linear_speed / linear_speed
            vx *= scale
            vy *= scale
        wz = max(-self.max_angular_speed, min(self.max_angular_speed, wz))
        # Correct physical response after limiting the requested velocity.
        # A common XY factor preserves travel direction. Do not alter the
        # established wire mixer or encoder calibration to hide this gain.
        return (vx * self.linear_command_scale, vy * self.linear_command_scale,
                wz * self.angular_command_scale)

    def _encode_udp_payload(self, packet: JetsonPacket) -> bytes:
        if self.payload_format in ('v2', 'v3_velocity', 'v4_uart'):
            # CLOCK_REALTIMEを使う: 受信側でカーネル到着時刻(SCM_TIMESTAMPNS,
            # 同じくREALTIME系)と直接引き算し、ポーリング待ちを除いた
            # 純ネットワークRTTを得るため。Piはこの値をエコーするだけ。
            t_tx_us = (time.clock_gettime_ns(time.CLOCK_REALTIME) // 1000) & 0xFFFFFFFF
            auto_request = bool(packet.reserved0 & MU3_L2_BUTTON_MASK)
            if self.payload_format == 'v4_uart':
                # 開発ボードへそのまま出せるUARTフレームをここで組む。
                # Piは中身を書き換えずUARTへ流すだけ（パススルー）。
                vx, vy, wz = self._pending_velocity
                # Shared 10000-unit cap, including startup/unknown profiles.
                frame, wheels, scale = drive_frame(vx, vy, wz)
                self._last_uart_frame = frame
                self._last_wheel_commands = wheels
                self._last_wheel_scale = scale
                return encode_v4_command(
                    uart_frame=frame,
                    seq=self.sequence,
                    t_tx_us=t_tx_us,
                    vx_mps=vx,
                    vy_mps=vy,
                    wz_radps=wz,
                    buttons0=packet.reserved0,
                    buttons1=packet.reserved1,
                    buttons2=packet.reserved2,
                    auto_request=auto_request,
                    estop=self.estop,
                )
            if self.payload_format == 'v3_velocity':
                # 物理速度指令(mm/s, mrad/s)。int8量子化なし・
                # Pi側デッドゾーンなしの高精度経路。
                vx, vy, wz = self._pending_velocity
                return encode_v3_command(
                    vx_mps=vx,
                    vy_mps=vy,
                    wz_radps=wz,
                    seq=self.sequence,
                    t_tx_us=t_tx_us,
                    buttons0=packet.reserved0,
                    buttons1=packet.reserved1,
                    buttons2=packet.reserved2,
                    auto_request=auto_request,
                    estop=self.estop,
                )
            return encode_v2_command(
                packet,
                seq=self.sequence,
                t_tx_us=t_tx_us,
                auto_request=auto_request,
                estop=self.estop,
            )
        if self.payload_format == 'mu3_controller':
            return encode_mu3_controller_payload(packet)
        return encode_jetson_packet(packet)

    # ------------------------------------------------------------------
    #  Piテレメトリ受信・リンク品質計測
    # ------------------------------------------------------------------

    def _drain_telemetry(self) -> None:
        if self.socket is None:
            return
        for _ in range(self.max_telemetry_packets_per_tick):
            try:
                data, ancdata, _flags, _addr = self.socket.recvmsg(
                    64, socket.CMSG_SPACE(64)
                )
            except BlockingIOError:
                break
            except OSError:
                # ECONNREFUSED等（Pi側サービス停止中のICMP）は無視する
                break
            if len(data) not in (TELEMETRY_SIZE, TELEMETRY_EXT_SIZE):
                continue
            try:
                telemetry = decode_telemetry(data)
            except ProtocolError:
                self.telemetry_crc_errors += 1
                continue
            # カーネル到着時刻（SCM_TIMESTAMPNS, CLOCK_REALTIME系）。
            # 取れなければ処理時刻で代用（ポーリング待ち分だけRTTが膨らむ）。
            arrival_us = None
            for level, ctype, cdata in ancdata:
                if level == socket.SOL_SOCKET and ctype == _SO_TIMESTAMPNS \
                        and len(cdata) >= 16:
                    sec, nsec = struct.unpack('@qq', cdata[:16])
                    arrival_us = (sec * 1_000_000_000 + nsec) // 1000
                    break
            if arrival_us is None:
                arrival_us = time.clock_gettime_ns(time.CLOCK_REALTIME) // 1000
            self._handle_telemetry(telemetry, arrival_us & 0xFFFFFFFF)

    def _handle_telemetry(self, telemetry, arrival_us: int) -> None:
        now = time.monotonic()
        self.telemetry = telemetry
        self.telemetry_monotonic = now
        self.telemetry_count += 1
        if telemetry.remote_navigation_slot is not None:
            self.remote_navigation_publisher.publish(String(data=json.dumps({
                'slot': telemetry.remote_navigation_slot,
                'sequence': telemetry.remote_navigation_sequence,
                'pi_seq': telemetry.pi_seq,
                'mu3_alive': telemetry.mu3_alive,
                'auto_engaged': telemetry.auto_engaged,
                'link_alive': telemetry.link_alive,
                'uart_open': telemetry.uart_open,
                'estop_active': telemetry.estop_active,
                'fault_latched': telemetry.fault_latched,
            })))

        # RTT算出: 自分が載せた送信時刻(REALTIME µs)のエコーと、Pi内滞留時間
        # hold_us、カーネル到着時刻から、クロック同期なしで純粋な往復
        # ネットワーク遅延だけを取り出す（両端の時刻は全てJetson側クロック）
        if telemetry.has_timestamp_echo:
            rtt_us = (arrival_us - telemetry.last_cmd_t_tx_us - telemetry.hold_us) & 0xFFFFFFFF
            if rtt_us < 5_000_000:  # 5s超は折り返し/再起動とみなして棄却
                rtt_ms = rtt_us / 1000.0
                self.rtt_last_ms = rtt_ms
                self.rtt_avg_ms = (
                    rtt_ms if self.rtt_avg_ms is None
                    else self.rtt_avg_ms * 0.9 + rtt_ms * 0.1
                )
                self.rtt_max_window_ms = max(self.rtt_max_window_ms, rtt_ms)

                if (
                    now - self._last_latency_pub
                    >= self.latency_publish_period_sec
                ):
                    message = Float32()
                    message.data = float(rtt_ms)
                    self.latency_publisher.publish(message)
                    self._last_latency_pub = now

        # パケットロス推定（1秒窓: 自分の送信数 vs Piの受理数の増分）
        if now - self._loss_window_start >= 1.0:
            if self._loss_window_cmd_count is not None:
                sent_delta = self.sent_count - self._loss_window_sent
                recv_delta = (telemetry.cmd_count - self._loss_window_cmd_count) & 0xFFFFFFFF
                if sent_delta > 0 and recv_delta <= sent_delta:
                    self.loss_pct = 100.0 * (1.0 - recv_delta / sent_delta)
            self._loss_window_start = now
            self._loss_window_sent = self.sent_count
            self._loss_window_cmd_count = telemetry.cmd_count

        # Pi側の自動モード係合状態
        previous_auto_engaged = self._last_auto_engaged
        if self._last_auto_engaged != telemetry.auto_engaged:
            self._last_auto_engaged = telemetry.auto_engaged
            message = Bool()
            message.data = telemetry.auto_engaged
            self.auto_engaged_publisher.publish(message)
            self.get_logger().info(
                f'Pi drivebase auto mode: '
                f'{"ENGAGED" if telemetry.auto_engaged else "DISENGAGED"}'
            )
        if previous_auto_engaged is True and not telemetry.auto_engaged:
            self.rearm_required = True
            self.enabled = False
            self._command_was_active = False
            self._pending_velocity = (0.0, 0.0, 0.0)
            self._pi_rearm_disarm_packets = max(
                getattr(self, '_pi_rearm_disarm_packets', 0), 3
            )

        # アクティブ指令中なのにPiが自動モードでない場合の警告
        if self.last_state == 'JETSON_PACKET_SENT' and not telemetry.auto_engaged:
            if self._not_engaged_since is None:
                self._not_engaged_since = now
            if (
                now - self._not_engaged_since >= self.warn_not_engaged_sec
                and now - self._last_not_engaged_warn >= 2.0
            ):
                self.get_logger().warning(
                    'Sending drive commands but Pi has NOT engaged auto mode '
                    '(check MU3 L2 request, estop, or link health)'
                )
                self._last_not_engaged_warn = now
        else:
            self._not_engaged_since = None

        # サマリJSON
        if (
            now - self._last_telemetry_json_pub
            >= self.telemetry_publish_period_sec
        ):
            self._publish_telemetry_json(now)
            self._last_telemetry_json_pub = now
            self.rtt_max_window_ms = 0.0

    def _publish_telemetry_json(self, now: float) -> None:
        telemetry = self.telemetry
        if telemetry is None:
            return
        age_ms = None
        if self.telemetry_monotonic is not None:
            age_ms = (now - self.telemetry_monotonic) * 1000.0
        payload = {
            'rtt_ms': {
                'last': self.rtt_last_ms,
                'avg': self.rtt_avg_ms,
                'max_window': self.rtt_max_window_ms or None,
            },
            'loss_pct': self.loss_pct,
            'pi': {
                'auto_engaged': telemetry.auto_engaged,
                'mu3_alive': telemetry.mu3_alive,
                'link_alive': telemetry.link_alive,
                'uart_open': telemetry.uart_open,
                'estop_active': telemetry.estop_active,
                'cmd_was_v2': telemetry.cmd_was_v2,
                'cmd_was_v3': telemetry.cmd_was_v3,
                'cmd_was_v4': telemetry.cmd_was_v4,
                'protocol': telemetry.protocol_name,
                'link_degraded': telemetry.link_degraded,
                'controlled_stop': telemetry.controlled_stop,
                'fault_latched': telemetry.fault_latched,
                'rearm_required': telemetry.rearm_required,
                'rx_rate_hz': telemetry.rx_rate_hz,
                'cmd_count': telemetry.cmd_count,
                'crc_error_count': telemetry.crc_error_count,
                'stale_drop_count': telemetry.stale_drop_count,
                'applied_axes': {
                    'lx': telemetry.applied_lx,
                    'ly': telemetry.applied_ly,
                    'rx': telemetry.applied_rx,
                    'ry': telemetry.applied_ry,
                },
                'applied_velocity': (
                    None if telemetry.applied_vx_mmps is None else {
                        'vx_mmps': telemetry.applied_vx_mmps,
                        'vy_mmps': telemetry.applied_vy_mmps,
                        'w_mradps': telemetry.applied_w_mradps,
                    }
                ),
                'wheel_commands': (
                    None if telemetry.wheel_commands is None
                    else list(telemetry.wheel_commands)
                ),
                'hold_us': telemetry.hold_us,
            },
            'bridge': {
                'sent_count': self.sent_count,
                'telemetry_count': self.telemetry_count,
                'telemetry_crc_errors': self.telemetry_crc_errors,
                'telemetry_age_ms': age_ms,
            },
        }
        message = String()
        message.data = json.dumps(payload)
        self.telemetry_publisher.publish(message)

    def _telemetry_safety_healthy(
        self,
        now: Optional[float] = None,
        *,
        allow_rearm_required: bool = False,
        allow_recovery_latches: bool = False,
    ) -> bool:
        if not getattr(self, 'require_healthy_telemetry', False):
            return True
        current = time.monotonic() if now is None else float(now)
        telemetry = getattr(self, 'telemetry', None)
        received = getattr(self, 'telemetry_monotonic', None)
        if (
            telemetry is None or received is None
            or current < received
            or current - received > self.telemetry_timeout_sec
        ):
            return False
        protocol_ok = True
        if self.payload_format == 'v4_uart':
            protocol_ok = bool(telemetry.cmd_was_v4)
        elif self.payload_format == 'v3_velocity':
            protocol_ok = bool(telemetry.cmd_was_v3)
        elif self.payload_format == 'v2':
            protocol_ok = bool(telemetry.cmd_was_v2)
        # After a link timeout, the Pi intentionally reports controlled_stop
        # and fault_latched until it receives a fresh auto-request edge.  The
        # edge is a zero-speed packet, so permit those two latches only while
        # the Pi explicitly reports that the rearm handshake is pending.
        # E-stop, stale telemetry, UART and protocol failures remain
        # non-recoverable here.  A DEGRADED report is deliberately *not* a
        # stop condition: the Pi uses it as an early warning while its own
        # tighter command watchdog still permits motion.  Turning that warning
        # into a disarm here made a harmless scheduler gap drop L2; the next
        # Pi packet then quite correctly latched rearm_required, producing the
        # observed AUTO disengage/re-engage loop.
        recovery_latches_ok = bool(
            allow_recovery_latches and telemetry.rearm_required
        )
        return bool(
            protocol_ok
            and telemetry.link_alive
            and telemetry.uart_open
            and not telemetry.estop_active
            and (not telemetry.controlled_stop or recovery_latches_ok)
            and (not telemetry.fault_latched or recovery_latches_ok)
            and (allow_rearm_required or not telemetry.rearm_required)
        )

    def _telemetry_armable(self, now: Optional[float] = None) -> bool:
        """Return whether a zero-speed auto/rearm request may be sent."""
        return self._telemetry_safety_healthy(
            now,
            allow_rearm_required=True,
            allow_recovery_latches=True,
        )

    def _telemetry_base_healthy(self, now: Optional[float] = None) -> bool:
        """Return whether non-zero drive commands may be considered."""
        return self._telemetry_safety_healthy(now)

    def _update_link_state(self) -> None:
        now = time.monotonic()
        # rearm_required is a drive-state handshake, not a lost/unhealthy
        # return link.  Treating it as link loss can race a fresh enable edge:
        # the callback sends the permitted zero-speed auto request, then this
        # timer immediately closes the local gate again before the Pi can
        # acknowledge engagement.
        link_ok = self._telemetry_armable(now)
        if self._last_link_ok is True and not link_ok:
            # Loss of the Pi return path or a non-rearm safety fault is treated
            # as a hard fault.  The command stream immediately becomes
            # disarmed; recovery alone cannot restart the drivebase.
            self.rearm_required = True
            self.enabled = False
            self._command_was_active = False
            self._pending_velocity = (0.0, 0.0, 0.0)
            self._pi_rearm_disarm_packets = max(
                getattr(self, '_pi_rearm_disarm_packets', 0), 3
            )
        refresh_due = now - self._last_bool_refresh >= 0.5
        if link_ok != self._last_link_ok or refresh_due:
            if link_ok != self._last_link_ok:
                self.get_logger().info(
                    f'Pi telemetry link: {"OK" if link_ok else "LOST"}'
                )
            self._last_link_ok = link_ok
            message = Bool()
            message.data = link_ok
            self.link_ok_publisher.publish(message)
            if refresh_due and self._last_auto_engaged is not None:
                engaged = Bool()
                engaged.data = self._last_auto_engaged and link_ok
                self.auto_engaged_publisher.publish(engaged)
            self._last_bool_refresh = now

    def _remote_description(self) -> str:
        if self.transport == 'mu3_uart':
            return (
                f'{self.mu3_device} {self.mu3_baud_rate}bps '
                f'{self.mu3_my_id}->{self.mu3_target_id}'
            )
        return f'{self.remote_address[0]}:{self.remote_address[1]}'

    def _packet_from_latest_twist(self) -> JetsonPacket:
        return twist_to_jetson_packet(
            linear_x=self.latest_twist.linear.x,
            linear_y=self.latest_twist.linear.y,
            angular_z=self.latest_twist.angular.z,
            max_linear_speed=self.max_linear_speed,
            max_angular_speed=self.max_angular_speed,
            linear_x_sign=self.linear_x_sign,
            linear_y_sign=self.linear_y_sign,
            angular_z_sign=self.angular_z_sign,
        )

    def _status_publish_due(self, state: str, now_ns: Optional[int] = None) -> bool:
        """Update state/logging and return whether a status message is due."""
        if now_ns is None:
            now_ns = self.get_clock().now().nanoseconds
        state_changed = state != self.last_state
        if state_changed:
            self.get_logger().info(f'Motor network state: {state}')
            self.last_state = state
        if (
            not state_changed
            and self.status_publish_period_ns > 0
            and now_ns - self.last_status_publish_ns
            < self.status_publish_period_ns
        ):
            return False
        self.last_status_publish_ns = now_ns
        return True

    def _emit_status(self, state: str, **values) -> None:
        """Serialize and publish a status whose throttle check already passed."""
        payload = {'state': state}
        payload.update(values)
        message = String()
        message.data = json.dumps(payload)
        self.status_publisher.publish(message)

    def _publish_status(self, state: str, **values) -> None:
        """Publish infrequent/error statuses through the common throttle."""
        if self._status_publish_due(state):
            self._emit_status(state, **values)

    def destroy_node(self):
        if self.mu3_fd is not None:
            try:
                stop_frame = encode_mu3_data_frame(
                    encode_mu3_controller_payload(JetsonPacket.zero())
                )
                for _ in range(3):
                    self._write_all(self.mu3_fd, stop_frame)
                    termios.tcdrain(self.mu3_fd)
            except (OSError, termios.error):
                pass
            try:
                os.close(self.mu3_fd)
            except OSError:
                pass
            self.mu3_fd = None
        if self.socket is not None:
            try:
                self._pending_velocity = (0.0, 0.0, 0.0)
                for _ in range(3):
                    # v2/v3は重複seqを破棄するため停止パケットごとにseqを進める
                    stop_packet = self._encode_udp_payload(JetsonPacket.zero())
                    self.sequence = (self.sequence + 1) & 0xFFFFFFFF
                    self.socket.sendto(stop_packet, self.remote_address)
            except OSError:
                pass
            self.socket.close()
            self.socket = None
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MotorUdpBridge()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
