"""Exercise the actual gateway with pseudo terminals; never open robot devices."""
import binascii
import os
import pty
import select
import signal
import socket
import struct
import subprocess
import sys
import time


def cobs_encode(payload):
    out = bytearray([0])
    index, code = 0, 1
    for value in payload:
        if value == 0:
            out[index] = code
            index = len(out)
            out.append(0)
            code = 1
        else:
            out.append(value)
            code += 1
    out[index] = code
    out.append(0)
    return bytes(out)


def decode_frame(encoded):
    payload = bytearray()
    index = 0
    while index < len(encoded):
        code = encoded[index]
        index += 1
        assert code and index + code - 1 <= len(encoded), "invalid UART COBS output"
        payload.extend(encoded[index:index + code - 1])
        index += code - 1
        if code != 255 and index < len(encoded):
            payload.append(0)
    assert len(payload) % 3 == 0, "partial UART command output"
    return {payload[i]: struct.unpack_from(">h", payload, i + 1)[0]
            for i in range(0, len(payload), 3)}


def command(seq, flags, commands):
    payload = b"".join(struct.pack(">Bh", ident, value) for ident, value in commands.items())
    uart = cobs_encode(payload)
    header = struct.pack("<BBBBBBBBHIhhh", 0xb6, 6, flags, 0, 0, 0,
                         len(uart), 0, seq, 0, 0, 0, 0)
    data = header + uart
    return data + struct.pack("<H", binascii.crc_hqx(data, 0xffff))


def main():
    binary = sys.argv[1]
    radio_master, radio_slave = pty.openpty()
    motor_master, motor_slave = pty.openpty()
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    env = dict(os.environ, MU3_DEVICE=os.ttyname(radio_slave),
               MU3_MOTOR_DEVICE=os.ttyname(motor_slave), MU3_JETSON_PORT=str(port))
    os.close(radio_slave)
    os.close(motor_slave)
    os.set_blocking(motor_master, False)
    client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    child = subprocess.Popen([binary], env=env, stdin=subprocess.DEVNULL,
                             stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    pending = bytearray()
    frames = []
    sequence = 0

    def read_frames(timeout=0.02):
        if select.select([motor_master], [], [], timeout)[0]:
            try:
                pending.extend(os.read(motor_master, 8192))
            except (BlockingIOError, OSError):
                return
        while 0 in pending:
            index = pending.index(0)
            frame = bytes(pending[:index])
            del pending[:index + 1]
            if frame:
                frames.append(decode_frame(frame))

    def until(predicate, reason, timeout=1.5, tick=None):
        deadline = time.monotonic() + timeout
        start = len(frames)
        while time.monotonic() < deadline:
            if child.poll() is not None:
                raise AssertionError(f"gateway exited early: {child.stderr.read().decode()}")
            if tick:
                tick()
            read_frames()
            for frame in frames[start:]:
                if predicate(frame):
                    return frame
            start = len(frames)
        raise AssertionError(reason)

    def radio(ps=False, mechanisms=False):
        sample = bytes([128, 80 if mechanisms else 128, 128, 128,
                        5 if mechanisms else 0, 4 if mechanisms else 0, int(ps)])
        os.write(radio_master, b"*DR=0E" + sample.hex().upper().encode() + b"\r\n")

    def udp(flags, commands):
        nonlocal sequence
        client.sendto(command(sequence, flags, commands), ("127.0.0.1", port))
        sequence = (sequence + 1) & 0xffff

    def stopped(frame):
        return all(frame.get(ident) == 0 for ident in (0, 1, 2, 3, 6, 7, 254))

    full = {0: 1000, 1: -1000, 2: 1000, 3: -1000,
            44: 1000, 61: 500, 6: 2000, 7: -2000, 254: 1}
    zero = dict(full)
    for ident in (0, 1, 2, 3, 6, 7, 254):
        zero[ident] = 0
    try:
        startup = until(lambda frame: all(ident in frame for ident in (0,1,2,3,6,7,254)), "gateway startup")
        assert 44 not in startup and 61 not in startup, "startup must omit unrequested position targets"
        assert all(44 not in frame and 61 not in frame for frame in frames), "startup emitted a home target"
        startup_wheels = {ident: value for ident, value in full.items() if ident < 4}
        startup_zero = {ident: 0 for ident in startup_wheels}
        udp(0, startup_zero)
        udp(1, startup_wheels)
        until(lambda frame: frame.get(0) == 1000, "startup wheels-only auto command")
        complement = until(lambda frame: 0 not in frame and 6 in frame,
                           "startup wheels-only complement")
        assert 44 not in complement and 61 not in complement, "complement synthesized unknown position targets"
        assert all(44 not in frame and 61 not in frame for frame in frames), "wheels-only startup moved a position axis"
        udp(0, startup_zero)
        until(lambda frame: 61 in frame and 44 not in frame,
              "explicit GM action must authorize only the GM target",
              tick=lambda: os.write(radio_master, b"*DR=0E80808080100000\r\n"))
        assert all(44 not in frame for frame in frames), "GM-only action authorized an ARM position target"
        # Concrete local PS scenario: wheels, mechanisms and momentary GPIO active.
        active = until(lambda frame: any(frame.get(i) for i in (0, 1, 2, 3))
                       and frame.get(6) == 2000 and frame.get(7) == 2000
                       and frame.get(254) == 1,
                       "manual mechanism/drive command", tick=lambda: radio(mechanisms=True))
        # A paused host must discard old TTY samples instead of refreshing the radio watchdog.
        child.send_signal(signal.SIGSTOP)
        time.sleep(0.13)
        radio(mechanisms=True)
        child.send_signal(signal.SIGCONT)
        until(stopped, "old buffered radio command resumed after a scheduler stall")
        active = until(lambda frame: frame.get(6) == 2000 and frame.get(254) == 1,
                       "fresh radio recovery", tick=lambda: radio(mechanisms=True))
        emergency = until(stopped, "MU3 PS failed to stop all velocity/GPIO outputs",
                          tick=lambda: radio(ps=True, mechanisms=True))
        assert emergency[44] == active[44], "PS must retain the ARM position target"
        # A released PS alone cannot resume; the UDP disarm/ARM handshake is required.
        for _ in range(4):
            radio(mechanisms=True)
            read_frames()
        assert stopped(frames[-1]), "PS release must keep emergency stop latched"
        radio()
        udp(0, zero)
        time.sleep(0.02)
        udp(1, full)
        until(lambda frame: frame.get(0) == 1000 and frame.get(61) == 500,
              "explicit rearm must activate delegated commands")
        udp(1, {ident: value for ident, value in full.items() if ident < 4})
        retained = until(lambda frame: 0 not in frame and frame.get(44) == 1000 and frame.get(61) == 500,
                         "wheels-only frame must retain delegated position targets")
        assert retained[44] == 1000 and retained[61] == 500
        # E-stop delegated mechanisms: retain their actual positions, not local defaults.
        udp(3, full)
        emergency = until(stopped, "UDP E-stop failed to stop delegated mechanisms")
        assert emergency[44] == 1000 and emergency[61] == 500, "E-stop changed position targets"
        udp(1, full)
        for _ in range(4):
            read_frames()
        assert stopped(frames[-1]), "ARM without DISARM resumed after E-stop"
        udp(0, zero)
        udp(1, full)
        until(lambda frame: frame.get(0) == 1000 and frame.get(254) == 1,
              "burst disarm/ARM must restore operation")
        # Force all three packets into one drain; the zero output must precede motion.
        child.send_signal(signal.SIGSTOP)
        time.sleep(0.02)
        output_start = len(frames)
        udp(3, full)
        udp(0, zero)
        udp(1, full)
        child.send_signal(signal.SIGCONT)
        until(stopped, "E-stop/DISARM/ARM burst failed to output a stop frame")
        until(lambda frame: frame.get(0) == 1000 and frame.get(254) == 1,
              "rearmed burst did not resume after the output stop")
        assert any(stopped(frame) for frame in frames[output_start:]), "emergency output was erased by burst reset"
        # Terminate during active motion, then inspect the complete stop repetitions.
        child.send_signal(signal.SIGTERM)
        deadline = time.monotonic() + 2
        while child.poll() is None and time.monotonic() < deadline:
            read_frames()
        assert child.wait(timeout=1) == 0, child.stderr.read().decode()
        read_frames(0)
        assert len(frames) >= 4 and all(stopped(frame) for frame in frames[-4:]), \
            "shutdown must emit four complete actuator stop frames"
        assert all(frame[44] == 1000 and frame[61] == 500 for frame in frames[-4:]), \
            "shutdown changed delegated position targets"
        invalid = subprocess.run([binary, "--no-hardware"],
                                 env=dict(env, MU3_JETSON_PORT="123x"),
                                 stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=2)
        assert invalid.returncode != 0, "invalid port override must fail startup"
        # Disconnected and persistently backpressured UARTs must both fail-stop.
        for failure_mode in ("startup_shutdown", "disconnect", "blocked"):
            failed_master, failed_slave = pty.openpty()
            failed_env = dict(env, MU3_MOTOR_DEVICE=os.ttyname(failed_slave))
            failed = subprocess.Popen([binary], env=failed_env, stdin=subprocess.DEVNULL,
                                      stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            try:
                deadline = time.monotonic() + 1.5
                opened = False
                startup_bytes = bytearray()
                while time.monotonic() < deadline:
                    assert failed.poll() is None, failed.stderr.read().decode()
                    if select.select([failed_master], [], [], 0.02)[0]:
                        startup_bytes.extend(os.read(failed_master, 8192))
                        opened = bool(startup_bytes)
                        if opened:
                            break
                assert opened, "failed-UART fixture did not start"
                if failure_mode == "startup_shutdown":
                    failed.send_signal(signal.SIGTERM)
                    assert failed.wait(timeout=2) == 0, failed.stderr.read().decode()
                    while select.select([failed_master], [], [], 0.02)[0]:
                        startup_bytes.extend(os.read(failed_master, 8192))
                    unknown_frames = [decode_frame(frame) for frame in startup_bytes.split(b"\x00") if frame]
                    assert len(unknown_frames) >= 4 and all(stopped(frame) for frame in unknown_frames[-4:]), \
                        "uncommanded startup shutdown failed to stop velocity outputs"
                    assert all(44 not in frame and 61 not in frame for frame in unknown_frames), \
                        "startup/shutdown synthesized an unrequested ARM/GM position target"
                    continue
                if failure_mode == "blocked":
                    # This pre-opened test-only slave fills the PTY driver;
                    # no real serial device or external process is involved.
                    os.set_blocking(failed_slave, False)
                    os.write(failed_slave, b"\x01" * (1024 * 1024))
                else:
                    os.close(failed_master)
                    failed_master = -1
                assert failed.wait(timeout=2) != 0, "failed motor UART must terminate the gateway"
                assert b"Motor UART command path failed" in failed.stderr.read(), "UART failure was not diagnosed"
            finally:
                if failed.poll() is None:
                    failed.kill()
                    failed.wait(timeout=2)
                failed.stderr.close()
                os.close(failed_slave)
                if failed_master >= 0:
                    os.close(failed_master)
        print("PASS: PTY manual PS, UDP emergency mechanisms, explicit rearm and active-motion shutdown")
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=2)
        child.stderr.close()
        client.close()
        os.close(radio_master)
        os.close(motor_master)


if __name__ == "__main__":
    main()
