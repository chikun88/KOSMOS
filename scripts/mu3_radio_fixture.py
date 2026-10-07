"""Independent loopback MU3 sender; importing this module starts no processes.

The owning harness passes an AF_UNIX/SOCK_DGRAM socket and must signal and reap
this process during cleanup: closing a datagram socket does not communicate EOF.
Control messages contain exactly [enabled (0 or 1), token (one byte)].
"""
import argparse
from collections import deque
import json
import math
import os
from pathlib import Path
import select
import signal
import socket
import tempfile
import time


SEND_INTERVAL = .020
RADIO_TIMEOUT = .100
MAX_CONTROLS_PER_TICK = 64


def record_radio_send(timing, now, started):
    previous = timing['last_send_monotonic']
    timing['last_send_monotonic'] = now
    timing['sent_frames'] += 1
    if previous is None:
        return
    gap = now-previous
    sample = {'t': now-started, 'gap_sec': gap}
    timing['recent_send_gaps'].append(sample)
    if gap > RADIO_TIMEOUT and timing['first_gap_over_radio_timeout'] is None:
        timing['first_gap_over_radio_timeout'] = sample
    if gap > timing['max_send_gap_sec']:
        timing['max_send_gap_sec'] = gap
        timing['max_gap_at_sec'] = now-started


def _write_output(output, timing):
    """Readers receive one complete final JSON object, including on worker error."""
    output = Path(output)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8',
                                         dir=output.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(timing, stream)
        os.replace(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def run(control_fd, output, started, *, target_port=39401):
    """Run in the subprocess main thread; target_port permits isolated UDP tests."""
    if not math.isfinite(started) or started < 0:
        raise ValueError('started must be a finite monotonic timestamp')
    if not isinstance(target_port, int) or not 1 <= target_port <= 65535:
        raise ValueError('target_port must be a UDP port')
    timing = {'last_send_monotonic': None, 'max_send_gap_sec': 0.,
              'max_gap_at_sec': None, 'first_gap_over_radio_timeout': None,
              'recent_send_gaps': deque(maxlen=100), 'sent_frames': 0,
              'control_messages': 0, 'invalid_control_messages': 0,
              'intentional_disables': 0, 'enabled': True, 'token': 0xc0}
    running = True

    def stop(_signal, _frame):
        nonlocal running
        running = False

    old_handlers = {number: signal.signal(number, stop)
                    for number in (signal.SIGINT, signal.SIGTERM)}
    try:
        with socket.socket(fileno=control_fd) as control, \
                socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
            if control.family != socket.AF_UNIX or control.type != socket.SOCK_DGRAM:
                raise ValueError('control_fd must be an AF_UNIX/SOCK_DGRAM socket')
            control.setblocking(False)
            sender.setblocking(False)
            deadline = time.monotonic()
            while running:
                # One complete state per datagram; bounded draining cannot
                # indefinitely starve a scheduled radio transmission.
                for _ in range(MAX_CONTROLS_PER_TICK):
                    try:
                        frame = control.recv(3)
                    except BlockingIOError:
                        break
                    if len(frame) != 2 or frame[0] not in (0, 1):
                        timing['invalid_control_messages'] += 1
                        continue
                    timing['control_messages'] += 1
                    enabled, token = bool(frame[0]), frame[1]
                    if not enabled:
                        if timing['enabled']:
                            timing['intentional_disables'] += 1
                        # Loss and recovery are deliberate test inputs, rather
                        # than sender scheduling gaps, even if queued together.
                        timing['last_send_monotonic'] = None
                    timing['enabled'], timing['token'] = enabled, token
                if not running:
                    break
                now = time.monotonic()
                if now >= deadline:
                    if timing['enabled']:
                        sender.sendto(bytes([128, 128, 128, 128, 0, 0, timing['token']]),
                                      ('127.0.0.1', target_port))
                        record_radio_send(timing, time.monotonic(), started)
                    deadline += SEND_INTERVAL
                    # Coalesce missed ticks instead of manufacturing a burst
                    # that could conceal an actual scheduling delay.
                    if deadline <= now:
                        deadline = now+SEND_INTERVAL
                select.select([control], [], [], max(0., deadline-time.monotonic()))
    except BaseException as error:
        timing['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        for number, handler in old_handlers.items():
            signal.signal(number, handler)
        timing['recent_send_gaps'] = list(timing['recent_send_gaps'])
        _write_output(output, timing)
    return timing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control-fd', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--started', type=float, required=True)
    args = parser.parse_args()
    if args.control_fd < 0 or not math.isfinite(args.started) or args.started < 0:
        parser.error('control-fd and started must be finite nonnegative values')
    run(args.control_fd, args.output, args.started)


if __name__ == '__main__':
    main()
