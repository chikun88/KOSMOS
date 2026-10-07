"""Safety fault regressions, runnable without ROS or connected hardware.

The node harness executes the actual node class methods extracted with AST,
using fake clocks and publishers. It covers state transitions and wire bytes;
ROS subscriptions, QoS, executor scheduling, and hardware remain integration
checks rather than claims made by these unit tests.
"""
import ast
import json
import math
import errno
from pathlib import Path
from types import SimpleNamespace
import time
import socket

import numpy as np
import pytest

from omni_autonomy_next.motor_kinematics import (
    limit_wheel_speeds, twist_to_wheel_speeds,
)
from omni_autonomy_next.motor_udp_protocol import (
    JetsonPacket, ProtocolError, V4_HEADER_SIZE, crc16_ccitt,
    decode_v3_command, decode_v4_command, encode_v3_command, encode_v4_command,
    quantize_velocity, twist_to_jetson_packet,
)
from omni_autonomy_next.robomas_uart import (
    UartFrameError, build_robomas_frame, drive_frame, mix_velocity,
    parse_robomas_frame, validate_passthrough_frame,
)
from omni_autonomy_next.runtime_guard import GuardHealth, MotionLimits, RuntimeGuard

PACKAGE = Path(__file__).resolve().parents[1] / 'omni_autonomy_next'
POSITIONS = np.array([[.3, -.3], [-.3, -.3], [-.3, .3], [.3, .3]])
ANGLES = np.radians([225., 135., 45., -45.])
ACTIVE = GuardHealth(True, False, True, True, True)


def make_guard(**overrides):
    full = MotionLimits(1., 1., 1., 1., 1., 4., 4.)
    kwargs = dict(profiles={'full': full,
                  'slow': MotionLimits(.1, .1, .1, .5, .5, 2., 2.)},
        hard_limits=full, default_profile='full', command_timeout_sec=.2,
        red_zone_speed_scale=.25, wheel_radius=.05, wheel_positions=POSITIONS,
        wheel_drive_angles_rad=ANGLES, wheel_signs=np.ones(4), max_wheel_speed=20.)
    kwargs.update(overrides)
    return RuntimeGuard(**kwargs)


def step(guard, command, now, **overrides):
    kwargs = dict(now_sec=now, command_age_sec=0., health=ACTIVE,
                  profile='full', user_scale=1., red_zone=False)
    kwargs.update(overrides)
    return guard.step(command, **kwargs)


@pytest.mark.parametrize('change,expected', [
    ({'profile': 'slow'}, .1), ({'user_scale': .1}, .1),
    ({'red_zone': True}, .25), ({'rl_scale': .1}, .1),
])
def test_reduced_envelope_applies_to_output_on_the_first_tick(change, expected):
    guard = make_guard()
    for tick in range(500):
        result = step(guard, (1., 0., 0.), tick*.01)
    assert result.velocity[0] == pytest.approx(1.)
    result = step(guard, (1., 0., 0.), 5., **change)
    assert 0. <= result.velocity[0] <= expected + 1.e-12
    # The following tick must also remain in the new envelope.
    assert step(guard, (1., 0., 0.), 5.01, **change).velocity[0] <= expected + 1.e-12


def test_every_transient_mixed_command_stays_in_the_physical_wheel_budget():
    guard = make_guard(max_wheel_speed=12.)
    rng = np.random.default_rng(87)
    for tick in range(1800):
        request = rng.uniform(-1., 1., 3) if tick % 25 == 0 else request
        result = step(guard, request, tick*.01)
        wheels = twist_to_wheel_speeds(*result.velocity, drive_model='omni4',
            wheel_radius=.05, wheel_positions=POSITIONS, wheel_drive_angles=ANGLES,
            wheel_signs=np.ones(4), track_width=0., wheelbase=0.)
        assert np.max(np.abs(wheels)) <= 12.+1.e-10
        assert np.linalg.norm(result.velocity[:2]) <= 1.+1.e-10


def test_lower_acceleration_profile_bounds_an_existing_ramp_immediately():
    full = MotionLimits(1., 1., 1., 1., 1., 4., 4.)
    slow_ramp = MotionLimits(1., 1., 1., .1, .1, .2, .2)
    guard = make_guard(profiles={'full': full, 'slow_ramp': slow_ramp})
    for tick in range(30):
        result = step(guard, (.8, 0., .8), tick*.01)
    assert result.acceleration[0] > .1
    changed = step(guard, (.8, 0., .8), .3, profile='slow_ramp')
    assert np.linalg.norm(changed.acceleration[:2]) <= .1+1.e-12
    assert abs(changed.acceleration[2]) <= .1+1.e-12


@pytest.mark.parametrize('kwargs', [
    {'now_sec': math.nan}, {'now_sec': math.inf},
    {'command_age_sec': math.nan}, {'command_age_sec': -.01},
])
def test_invalid_timing_stops_without_poisoning_subsequent_valid_commands(kwargs):
    guard = make_guard()
    step(guard, (.5, 0., 0.), 1.)
    result = step(guard, (.5, 0., 0.), 1.01, **kwargs)
    assert result.velocity == (0., 0., 0.)
    assert result.reason == 'INVALID_TIMING'
    assert not result.allowed
    recovered = step(guard, (.5, 0., 0.), 2.)
    assert recovered.allowed
    assert np.all(np.isfinite(recovered.velocity))


def test_time_reversal_and_malformed_command_fail_closed():
    guard = make_guard()
    step(guard, (.5, 0., 0.), 2.)
    assert step(guard, (.5, 0., 0.), 1.).reason == 'INVALID_TIMING'
    for command in ([None, 0., 0.], [[1.], [0., 0.]], ['bad', 0., 0.]):
        result = step(guard, command, 3.)
        assert result.reason == 'INVALID_COMMAND'
        assert result.velocity == (0., 0., 0.)


@pytest.mark.parametrize('name,value', [
    ('command_timeout_sec', math.nan), ('command_timeout_sec', math.inf),
    ('red_zone_speed_scale', math.nan), ('max_wheel_speed', math.nan),
    ('wheel_radius', math.nan), ('wheel_radius', math.inf),
    ('wheel_signs', [1., 1., 0., 1.]),
    ('wheel_positions', np.full((4, 2), math.nan)),
    ('wheel_positions', np.zeros((4, 2))),
    ('wheel_drive_angles_rad', np.zeros(4)),
])
def test_invalid_safety_configuration_is_rejected_at_startup(name, value):
    with pytest.raises(ValueError):
        make_guard(**{name: value})


@pytest.mark.parametrize('value', [math.nan, math.inf, -.1])
def test_invalid_motion_limits_cannot_silently_open_the_gate(value):
    with pytest.raises(ValueError):
        MotionLimits(value, 1., 1., 1., 1., 1., 1.)


def test_zero_forward_limit_does_not_disable_permitted_lateral_motion():
    limits = MotionLimits(0., .5, .5, 1., 1., 4., 4.)
    guard = make_guard(profiles={'full': limits})
    for tick in range(300):
        result = step(guard, (1., .3, 0.), tick*.01)
    assert result.velocity == pytest.approx((0., .3, 0.))


@pytest.mark.parametrize('value', [math.nan, math.inf, -math.inf])
def test_protocol_and_uart_reject_nonfinite_velocities(value):
    with pytest.raises(ProtocolError):
        encode_v3_command(vx_mps=value, vy_mps=0., wz_radps=0., seq=1, t_tx_us=1)
    with pytest.raises(UartFrameError):
        mix_velocity(value, 0., 0.)
    with pytest.raises(ProtocolError):
        twist_to_jetson_packet(linear_x=value, linear_y=0., angular_z=0.,
                               max_linear_speed=1., max_angular_speed=1.)
    with pytest.raises(ValueError):
        limit_wheel_speeds([value], 1.)


def test_extreme_finite_velocity_saturates_wire_range_without_overflow():
    packet = encode_v3_command(vx_mps=1.e308, vy_mps=-1.e308,
                              wz_radps=1.e308, seq=1, t_tx_us=1)
    assert decode_v3_command(packet)[:3] == (32767, -32767, 32767)


@pytest.mark.parametrize('frame', [
    b'\x05\x01\x02\x03\x00',  # COBS block overruns its encoded packet
    build_robomas_frame([(0, 10), (0, 0)]),
    build_robomas_frame([(0, 10001)]),
    build_robomas_frame([(3, -10001)]),
])
def test_v4_rejects_frames_the_gateway_cannot_safely_passthrough(frame):
    with pytest.raises((ProtocolError, UartFrameError)):
        encode_v4_command(uart_frame=frame, seq=1, t_tx_us=1)
    good = encode_v4_command(uart_frame=drive_frame(0., 0., 0.)[0], seq=1, t_tx_us=1)
    # A correct transport CRC must not legitimize unsafe embedded UART bytes.
    body = bytearray(good[:V4_HEADER_SIZE])
    body[6] = len(frame)
    body.extend(frame)
    bad = bytes(body) + crc16_ccitt(body).to_bytes(2, 'little')
    with pytest.raises(ProtocolError):
        decode_v4_command(bad)


def message(data=None):
    return SimpleNamespace(data=data)


def twist(velocity=(0., 0., 0.)):
    return SimpleNamespace(linear=SimpleNamespace(x=velocity[0], y=velocity[1]),
                           angular=SimpleNamespace(z=velocity[2]))


def node_class(filename, class_name):
    """Load actual method bodies while avoiding unavailable ROS imports."""
    tree = ast.parse((PACKAGE / filename).read_text())
    cls = next(item for item in tree.body if isinstance(item, ast.ClassDef)
               and item.name == class_name)
    cls.bases = []
    namespace = dict(json=json, math=math, time=time, socket=socket,
        _SO_TIMESTAMPNS=getattr(socket, 'SO_TIMESTAMPNS', 35), Twist=twist, Bool=message,
        String=message, Float32=message, JetsonPacket=JetsonPacket,
        GuardHealth=GuardHealth, ProtocolError=ProtocolError,
        UartFrameError=UartFrameError, drive_frame=drive_frame,
        mix_velocity=mix_velocity, quantize_velocity=quantize_velocity,
        encode_v3_command=encode_v3_command, encode_v4_command=encode_v4_command,
        MU3_L2_BUTTON_MASK=2)
    exec(compile('from __future__ import annotations\n' + ast.unparse(cls),
                 str(PACKAGE / filename), 'exec'), namespace)
    return namespace[class_name]


@pytest.fixture
def bridge_node(monkeypatch):
    current = [5.]
    monkeypatch.setattr(time, 'monotonic', lambda: current[0])
    cls = node_class('motor_udp_bridge_node.py', 'MotorUdpBridge')
    node = cls.__new__(cls)
    node.latest_twist = twist((.3, 0., 0.))
    node.linear_x_sign = node.linear_y_sign = node.angular_z_sign = 1.
    node.linear_command_scale = node.angular_command_scale = 1.
    node.max_linear_speed = 1.; node.max_angular_speed = 1.8
    node.latest_command_monotonic = current[0]; node.latest_command_time = None
    node.command_timeout = SimpleNamespace(nanoseconds=150000000)
    node.get_clock = lambda: SimpleNamespace(now=lambda:
        SimpleNamespace(nanoseconds=int(current[0]*1.e9)))
    node.payload_format = 'v4_uart'; node.sequence = 0
    node.enabled = True; node.require_enable = False; node.rearm_required = False
    node.estop = False; node.immediate_send_on_cmd = False
    node._explicit_disarm = False; node.auto_rearm_when_idle = True
    node.require_healthy_telemetry = True
    node.telemetry = SimpleNamespace(auto_engaged=True)
    node._transport_ready = lambda: True
    node._telemetry_base_healthy = node._telemetry_armable = lambda: True
    node._packet_from_latest_twist = lambda: twist_to_jetson_packet(
        linear_x=node.latest_twist.linear.x, linear_y=node.latest_twist.linear.y,
        angular_z=node.latest_twist.angular.z, max_linear_speed=1., max_angular_speed=1.8)
    node._status_publish_due = lambda *args, **kwargs: False
    node.frames = []
    def send(packet):
        raw = node._encode_udp_payload(packet)
        node.frames.append(raw)
        return raw, raw, b''
    node._send_packet = send
    return node, current


def zero_v4(frame):
    uart, velocity, _, flags, _, _ = decode_v4_command(frame)
    assert velocity == (0, 0, 0)
    assert all(value == 0 for _, value in parse_robomas_frame(uart))
    assert not flags & 1


def test_explicit_disarm_stops_even_when_enable_is_optional(bridge_node):
    node, _now = bridge_node
    node._send_current_packet('test')
    assert decode_v4_command(node.frames[-1])[1][0] > 0
    node._enable_callback(message(False))
    node._send_current_packet('test')
    zero_v4(node.frames[-1])
    node._enable_callback(message(True))
    node._send_current_packet('test')
    assert decode_v4_command(node.frames[-1])[1][0] > 0


@pytest.mark.parametrize('axis', [0, 1, 2])
@pytest.mark.parametrize('value', [math.nan, math.inf, -math.inf])
def test_invalid_incoming_twist_sends_immediate_stop_and_requires_rearm(bridge_node, axis, value):
    node, _now = bridge_node
    velocity = [.3, 0., 0.]; velocity[axis] = value
    node.latest_twist = twist(velocity)
    node._send_current_packet('test')
    zero_v4(node.frames[-1])
    assert node.rearm_required
    assert not node.enabled


@pytest.mark.parametrize('protocol,velocity', [('v3_velocity', .001), ('v4_uart', .0001)])
def test_low_speed_wire_motion_must_rearm_after_command_timeout(bridge_node, protocol, velocity):
    node, now = bridge_node
    node.payload_format = protocol
    node.latest_twist = twist((velocity, 0., 0.))
    assert node._packet_from_latest_twist().ly_state == 0  # old idle detector
    node._send_current_packet('test')
    assert node._command_was_active
    now[0] += .2
    node._send_current_packet('test')
    assert node.rearm_required
    if protocol == 'v4_uart':
        zero_v4(node.frames[-1])
    else:
        packet = decode_v3_command(node.frames[-1])
        assert packet[:3] == (0, 0, 0)
        assert not packet[4] & 1


def test_stale_exact_zero_heartbeat_holds_engagement_without_motion(bridge_node):
    node, now = bridge_node
    node.latest_twist = twist()
    node._send_current_packet('test')
    now[0] += .2
    node._send_current_packet('test')
    frame = decode_v4_command(node.frames[-1])
    assert frame[1] == (0, 0, 0)
    assert frame[3] & 1
    assert not node.rearm_required


def test_motor_udp_endpoint_rejects_a_competing_bridge_process():
    cls = node_class('motor_udp_bridge_node.py', 'MotorUdpBridge')
    node = cls.__new__(cls)
    node.get_parameter = lambda name: SimpleNamespace(value={
        'udp_send_buffer_bytes': 4096, 'udp_tos': 0, 'udp_priority': 0}[name])
    node.get_logger = lambda: SimpleNamespace(warning=lambda *_: None)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as first, \
            socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as second:
        node._configure_udp_socket(first)
        node._configure_udp_socket(second)
        first.bind(('127.0.0.1', 0))
        first.connect(('127.0.0.1', 9999))
        with pytest.raises(OSError) as duplicate:
            second.bind(first.getsockname())
        assert duplicate.value.errno == errno.EADDRINUSE


@pytest.fixture
def guard_node(monkeypatch):
    now = [5.]
    monkeypatch.setattr(time, 'monotonic', lambda: now[0])
    cls = node_class('runtime_guard_node.py', 'RuntimeGuardNode')
    node = cls.__new__(cls)
    params = {'require_tracking': True, 'require_armed': True,
              'require_motor_link': True, 'require_auto_engaged': True,
              'require_rl_policy': True, 'tracking_heartbeat_timeout_sec': .5,
              'tracking_rejection_grace_sec': .35, 'rl_heartbeat_timeout_sec': .4,
              'motor_heartbeat_timeout_sec': 1.}
    node.get_parameter = lambda name: SimpleNamespace(value=params[name])
    node.command = (.3, 0., 0.); node.command_time = 5.
    node.tracking_ok = node.motor_link_ok = node.auto_engaged = node.armed = True
    node.estop = False; node.rl_healthy = node.rl_scale_valid = True
    node.rl_scale = node.user_scale = 1.; node.red_zone = False; node.profile = 'full'
    node.tracking_message_time = node.tracking_last_ok_time = 5.
    node.motor_link_message_time = node.auto_message_time = 5.
    node.rl_message_time = node.rl_scale_time = 5.
    node.guard = make_guard(); node.last_reason = None; node.tick = 0
    node._publish_speed_limit = lambda now: None
    node.outputs = []; node.output_pub = SimpleNamespace(publish=node.outputs.append)
    node.get_logger = lambda: SimpleNamespace(info=lambda *_: None, warning=lambda *_: None)
    return node, now


def test_rl_health_cannot_hide_invalid_or_missing_scale_stream(guard_node):
    node, now = guard_node
    node._rl_scale_cb(message(math.nan))
    node._rl_health_cb(message(True))
    node._timer_cb()
    assert node.last_reason == 'RL_POLICY_UNHEALTHY'
    assert node.outputs[-1].linear.x == 0.
    node._rl_scale_cb(message(.8))
    node._timer_cb()
    assert node.last_reason == 'ACTIVE'
    now[0] += .41
    node.command_time = node.tracking_message_time = now[0]
    node._rl_health_cb(message(True))
    node._timer_cb()
    assert node.last_reason == 'RL_POLICY_UNHEALTHY'


@pytest.mark.parametrize('stream', ['motor_link_message_time', 'auto_message_time'])
def test_stale_motor_heartbeat_cannot_remain_healthy(guard_node, stream):
    node, now = guard_node
    setattr(node, stream, now[0]-1.01)
    node._timer_cb()
    assert node.last_reason in ('MOTOR_LINK_UNHEALTHY', 'AUTO_NOT_ENGAGED')
    assert node.outputs[-1].linear.x == 0.
