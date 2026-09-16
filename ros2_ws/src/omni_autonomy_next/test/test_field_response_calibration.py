"""Measured-response calibration applies once, with stops and limits intact."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from geometry_msgs.msg import Twist
from omni_autonomy_next.motor_udp_bridge_node import MotorUdpBridge
from omni_autonomy_next.motor_udp_protocol import JetsonPacket, decode_v4_command
from omni_autonomy_next.robomas_uart import drive_frame

ROOT = Path(__file__).resolve().parents[4]
spec = importlib.util.spec_from_file_location('fit_field_response',ROOT/'scripts/fit_field_response.py')
fitting = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fitting)


def bridge(velocity, linear=.38, angular=.34):
    return SimpleNamespace(latest_twist=Twist(),linear_x_sign=1.,linear_y_sign=1.,
        angular_z_sign=1.,estop=False,max_linear_speed=1.,max_angular_speed=1.8,
        linear_command_scale=linear,angular_command_scale=angular)


@pytest.mark.parametrize('velocity',[(.4,.3,.6),(-.4,.3,-.6),(0.,0.,0.),(0.,0.,1.),(2.,2.,3.)])
def test_scales_limited_velocity_once_and_preserves_xy_direction(velocity):
    node=bridge(velocity)
    node.latest_twist.linear.x=float(velocity[0]);node.latest_twist.linear.y=float(velocity[1])
    node.latest_twist.angular.z=float(velocity[2])
    got=np.array(MotorUdpBridge._velocity_from_latest_twist(node))
    xy=np.array(velocity[:2]);xy=xy/max(1.,np.linalg.norm(xy))
    assert got[:2] == pytest.approx(xy*.38)
    assert got[2] == pytest.approx(np.clip(velocity[2],-1.8,1.8)*.34)
    assert np.linalg.norm(got[:2]) <= .38+1e-12
    node._pending_velocity=tuple(got)
    node.payload_format='v4_uart';node.sequence=1
    encoded=MotorUdpBridge._encode_udp_payload(node,JetsonPacket())
    frame,wire,_,_,_,_=decode_v4_command(encoded)
    expected_frame,wheels,_=drive_frame(*got)
    assert frame==expected_frame  # Actual wire frame, not just advertised twist.
    assert wire==pytest.approx(got*1000,abs=.5)
    assert node._last_wheel_commands==wheels


def test_timestamp_unwrap_handles_32bit_rollover_and_rejects_stale():
    tx=(1<<32)-20
    received=((1<<32)+50)*1000
    assert fitting.unwrap_tx_us(tx,received)==tx*1000
    with pytest.raises(ValueError):fitting.unwrap_tx_us(100,20000000000)


def test_fit_recovers_gain_and_lag_with_out_of_window_samples_excluded():
    rng=np.random.default_rng(2)
    times=np.arange(0.,12.,.02)
    velocity=rng.uniform(-.5,.5,(len(times),3))
    cmd=np.column_stack([times,velocity])
    wheel=np.column_stack([times+.1,velocity*np.array([2.6,2.5,2.8])])
    fitted=fitting.fit_axes(cmd,wheel)
    assert [r['gain'] for r in fitted] == pytest.approx([2.6,2.5,2.8],rel=.05)
    assert [r['effective_lag_s'] for r in fitted] == pytest.approx([.1,.1,.1],abs=.0100001)
    x,y=fitting.pairs(cmd,wheel+np.array([20.,0.,0.,0.]),0,.1)
    assert len(x)==len(y)==0


@pytest.mark.parametrize('linear,angular,transport,protocol', [
    (0.,.34,'udp','v4_uart'),(-.1,.34,'udp','v4_uart'),(1.1,.34,'udp','v4_uart'),
    (.38,float('nan'),'udp','v4_uart'),(float('inf'),.34,'udp','v4_uart'),
    (.38,.34,'udp','mu3_controller'),(.38,.34,'mu3_uart','v4_uart')])
def test_invalid_or_unsupported_calibration_is_rejected(linear,angular,transport,protocol):
    from omni_autonomy_next.motor_udp_bridge_node import validate_command_calibration
    with pytest.raises(ValueError):
        validate_command_calibration(linear,angular,transport,protocol)


@pytest.mark.parametrize('fault',['none','estop','stale','disarmed','telemetry'])
def test_calibration_cannot_bypass_final_motor_safety_gate(monkeypatch,fault):
    from omni_autonomy_next import motor_udp_bridge_node as module
    monkeypatch.setattr(module.time,'monotonic',lambda:5.)
    node=bridge((.4,.2,.5))
    node.latest_twist.linear.x=.4;node.latest_twist.linear.y=.2;node.latest_twist.angular.z=.5
    node.payload_format='v4_uart';node.sequence=0
    node.get_clock=lambda:SimpleNamespace(now=lambda:SimpleNamespace(nanoseconds=5000000000))
    node.latest_command_monotonic=4. if fault=='stale' else 5.
    node.latest_command_time=None;node.command_timeout=SimpleNamespace(nanoseconds=150000000)
    node.enabled=fault!='disarmed';node.require_enable=True;node.rearm_required=False
    node.estop=fault=='estop'
    node.require_healthy_telemetry=True;node.telemetry=SimpleNamespace(auto_engaged=True)
    node._transport_ready=lambda:True
    node._velocity_from_latest_twist=lambda:MotorUdpBridge._velocity_from_latest_twist(node)
    node._packet_from_latest_twist=lambda:JetsonPacket(lx_state=20,ly_state=10,rx_state=10)
    node._telemetry_base_healthy=lambda:fault!='telemetry'
    node._telemetry_armable=lambda:fault!='telemetry'
    node._status_publish_due=lambda *args,**kwargs:False
    frames=[]
    def send(packet):
        raw=MotorUdpBridge._encode_udp_payload(node,packet)
        frames.append(raw)
        return raw,raw,b''
    node._send_packet=send
    MotorUdpBridge._send_current_packet(node,trigger='test')
    frame,velocity,_,_,_,_=decode_v4_command(frames[-1])
    expected=(.4*.38,.2*.38,.5*.34) if fault=='none' else (0.,0.,0.)
    assert velocity==pytest.approx(np.array(expected)*1000,abs=.5)
    assert frame==drive_frame(*expected)[0]
