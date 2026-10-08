"""Real subprocess/UDP checks for the synthetic radio, without ROS or hardware."""
from collections import deque
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

import pytest


SOURCE = Path(__file__).resolve().parents[1]/'scripts/mu3_radio_fixture.py'
spec = importlib.util.spec_from_file_location('mu3_radio_fixture', SOURCE)
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)


class RadioProcess:
    def __init__(self, output):
        self.output = output
        self.receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.receiver.bind(('127.0.0.1', 0))
        self.control, child_control = socket.socketpair(socket.AF_UNIX, socket.SOCK_DGRAM)
        code = ('import importlib.util,sys; '
                's=importlib.util.spec_from_file_location("radio",sys.argv[1]); '
                'm=importlib.util.module_from_spec(s); s.loader.exec_module(m); '
                'm.run(int(sys.argv[2]),sys.argv[3],float(sys.argv[4]),'
                'target_port=int(sys.argv[5]))')
        try:
            self.process = subprocess.Popen(
                [sys.executable, '-c', code, str(SOURCE), str(child_control.fileno()),
                 str(output), str(time.monotonic()), str(self.receiver.getsockname()[1])],
                pass_fds=(child_control.fileno(),), start_new_session=True,
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        finally:
            child_control.close()

    def state(self, enabled, token):
        self.control.send(bytes([int(enabled), token]))

    def receive(self, timeout=2):
        self.receiver.settimeout(timeout)
        frame = self.receiver.recv(64)
        assert frame[:6] == bytes([128, 128, 128, 128, 0, 0])
        assert len(frame) == 7
        return frame[6]

    def wait_token(self, token):
        deadline = time.monotonic()+2
        while time.monotonic() < deadline:
            if self.receive() == token:
                return
        raise AssertionError(f'token {token:#x} was not transmitted')

    def drain(self):
        self.receiver.setblocking(False)
        try:
            while True:
                self.receiver.recv(64)
        except BlockingIOError:
            pass

    def stop(self, number=signal.SIGINT):
        if self.process.poll() is None:
            os.killpg(self.process.pid, number)
        _, errors = self.process.communicate(timeout=3)
        assert self.process.returncode == 0, errors.decode()
        return json.loads(self.output.read_text())

    def close(self):
        if self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGKILL)
            self.process.wait(timeout=3)
        if self.process.stderr is not None:
            self.process.stderr.close()
        self.control.close()
        self.receiver.close()


@pytest.fixture
def radio(tmp_path):
    child = RadioProcess(tmp_path/'radio.json')
    try:
        child.wait_token(0xc0)
        yield child
    finally:
        child.close()


def test_release_hold_loss_and_held_recovery_keep_the_exact_radio_state(radio):
    radio.state(True, 0xc4)
    radio.wait_token(0xc4)
    assert [radio.receive() for _ in range(4)] == [0xc4]*4
    radio.state(True, 0xc0)
    radio.wait_token(0xc0)
    radio.state(True, 0xc4)
    radio.wait_token(0xc4)
    radio.state(False, 0xc4)
    # Allow a frame already in flight before the disable command was applied.
    time.sleep(.06)
    radio.drain()
    with pytest.raises(socket.timeout):
        radio.receive(timeout=.25)
    radio.state(True, 0xc4)
    radio.wait_token(0xc4)
    assert radio.receive() == 0xc4
    radio.state(True, 0xc0)
    radio.wait_token(0xc0)
    timing = radio.stop()
    assert timing['intentional_disables'] == 1
    assert timing['control_messages'] == 6
    assert timing['first_gap_over_radio_timeout'] is None
    assert timing['max_send_gap_sec'] < .1
    assert timing['enabled'] and timing['token'] == 0xc0


def test_malformed_datagrams_cannot_change_or_disable_the_state(radio):
    for message in (b'', b'\x00', b'\x02\xc4', b'\x00\xc4extra'):
        radio.control.send(message)
    assert [radio.receive() for _ in range(4)] == [0xc0]*4
    timing = radio.stop(signal.SIGTERM)
    assert timing['invalid_control_messages'] == 4
    assert timing['control_messages'] == 0
    assert timing['enabled'] and timing['token'] == 0xc0


def test_observer_gil_stall_does_not_stall_independent_radio(radio):
    # This reproduces a same-process sender thread's failure without ROS: no
    # other Python thread can acquire this observer's GIL for over100ms.
    old_interval = sys.getswitchinterval()
    try:
        sys.setswitchinterval(1.)
        end = time.monotonic()+.45
        while time.monotonic() < end:
            pass
    finally:
        sys.setswitchinterval(old_interval)
    radio.drain()
    radio.receive()
    timing = radio.stop()
    assert timing['sent_frames'] >= 15
    assert timing['max_send_gap_sec'] < .1
    assert timing['first_gap_over_radio_timeout'] is None


def test_actual_sender_pause_remains_visible_as_a_watchdog_sized_gap(radio):
    # The first UDP packet can arrive before the sender records its initial
    # timestamp. Receiving a second packet establishes that baseline before
    # SIGSTOP; otherwise pausing between sendto() and record_radio_send()
    # measures only a post-resume gap and intermittently misses the injection.
    radio.receive()
    os.killpg(radio.process.pid, signal.SIGSTOP)
    try:
        time.sleep(.18)
        radio.drain()
    finally:
        os.killpg(radio.process.pid, signal.SIGCONT)
    radio.receive()
    timing = radio.stop()
    assert timing['max_send_gap_sec'] >= .15
    assert timing['first_gap_over_radio_timeout']['gap_sec'] >= .15
    assert timing['first_gap_over_radio_timeout'] in timing['recent_send_gaps']


@pytest.mark.parametrize('number', [signal.SIGINT, signal.SIGTERM])
def test_owner_cleanup_reaps_sender_and_writes_final_stats(radio, number):
    radio.receive()
    timing = radio.stop(number)
    assert radio.process.poll() == 0
    assert timing['sent_frames'] >= 2
    assert 0 < len(timing['recent_send_gaps']) <= 100
    assert all(sample['gap_sec'] > 0 for sample in timing['recent_send_gaps'])
    radio.drain()
    with pytest.raises(socket.timeout):
        radio.receive(timeout=.06)


def test_gap_stats_retain_the_first_fault_and_only_last100_intervals():
    timing = {'last_send_monotonic': None, 'max_send_gap_sec': 0.,
              'max_gap_at_sec': None, 'first_gap_over_radio_timeout': None,
              'recent_send_gaps': deque(maxlen=100), 'sent_frames': 0}
    fixture.record_radio_send(timing, 10., 10.)
    fixture.record_radio_send(timing, 10.2, 10.)
    for index in range(110):
        fixture.record_radio_send(timing, 10.22+index*.02, 10.)
    assert timing['sent_frames'] == 112
    assert timing['max_send_gap_sec'] == pytest.approx(.2)
    assert timing['max_gap_at_sec'] == pytest.approx(.2)
    assert timing['first_gap_over_radio_timeout'] == pytest.approx(
        {'t': .2, 'gap_sec': .2})
    assert len(timing['recent_send_gaps']) == 100
    assert timing['recent_send_gaps'][-1]['gap_sec'] == pytest.approx(.02)
    timing['last_send_monotonic'] = None
    fixture.record_radio_send(timing, 50., 10.)
    assert timing['max_send_gap_sec'] == pytest.approx(.2)


def test_cli_is_import_safe_and_validates_required_descriptor(tmp_path):
    # Import above did not start a sender. The actual CLI rejects an invalid
    # descriptor before opening sockets or starting a loop.
    result = subprocess.run([sys.executable, str(SOURCE), '--control-fd', '-1',
                             '--output', str(tmp_path/'unused.json'), '--started', '1'],
                            capture_output=True, timeout=3)
    assert result.returncode == 2
    assert b'finite nonnegative' in result.stderr
    assert not (tmp_path/'unused.json').exists()
