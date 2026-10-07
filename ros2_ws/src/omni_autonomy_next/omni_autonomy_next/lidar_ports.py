"""Resolve which serial device each LiDAR is actually attached to.

The two A2M8 units on this robot hang off the same USB hub through CP2102
bridges that both report the USB serial string ``0001``, so ``/dev/serial/by-id``
cannot tell them apart and ``robot.yaml`` has to name a ``by-path`` entry
instead.  A ``by-path`` entry encodes the *hub port* the cable sits in, so
re-plugging a LiDAR into a neighbouring port (or a flaky port forcing a move)
silently invalidates the config: the path simply stops existing and
``sllidar_node`` respawns forever on a device node that is not there.

The LiDAR itself does carry a unique identity -- the 16-byte serial number in
its ``GET_DEVICE_INFO`` reply -- so this module probes every candidate tty and
matches it against the ``device_serial`` prefix recorded in ``robot.yaml``.
That is stable across hub ports, cables and reboots.

A LiDAR whose electronics are dead answers nothing, so identification alone
would leave it unassigned.  When exactly one LiDAR and one candidate tty are
left over they are paired anyway, which lets ``sllidar_node`` surface the real
"no reply from the device" error instead of a misleading missing-port one.
"""

from __future__ import annotations

import binascii
import glob
import math
import os
import time

# GET_DEVICE_INFO: request A5 50, reply A5 5A 14 00 00 00 04 + 20 payload bytes.
_DEVICE_INFO_REQUEST = b'\xa5\x50'
_DEVICE_INFO_RESPONSE_LENGTH = 27
_DEVICE_INFO_DESCRIPTOR = b'\xa5\x5a\x14\x00\x00\x00\x04'
_SERIAL_NUMBER_SLICE = slice(11, 27)


def candidate_ports():
    """Return the serial devices that could be a LiDAR, in a stable order.

    Only CP210x bridges are considered so the Contec I/O unit, the wireless
    keyboard and any USB console adapter never get probed with LiDAR commands.
    """
    candidates = []
    for device in sorted(glob.glob('/dev/ttyUSB*')):
        driver = os.path.realpath(
            f'/sys/class/tty/{os.path.basename(device)}/device/driver'
        )
        if os.path.basename(driver) != 'cp210x':
            continue
        candidates.append(device)
    return candidates


def probe_device_serial(port, baudrate=115200, timeout=1.0):
    """Return the LiDAR serial number as uppercase hex, or None if silent."""
    if not math.isfinite(float(timeout)) or timeout <= 0.0:
        raise ValueError('LiDAR probe timeout must be finite and positive')
    try:
        import serial
    except ImportError:  # pragma: no cover - pyserial ships with the ROS image
        return None
    try:
        # Set the modem-control line before open, avoiding pyserial's default
        # DTR pulse on connection. Probing identity must not start the motor.
        with serial.Serial(port=None, baudrate=baudrate, timeout=timeout,
                           write_timeout=timeout) as link:
            # The A-series adapter drives the scan motor from DTR; keeping it
            # deasserted means a probe never spins a LiDAR up just to identify it.
            link.dtr = False
            link.port = port
            link.open()
            time.sleep(0.05)
            link.reset_input_buffer()
            if link.write(_DEVICE_INFO_REQUEST) != len(_DEVICE_INFO_REQUEST):
                return None
            reply = link.read(_DEVICE_INFO_RESPONSE_LENGTH)
    except (OSError, ValueError):
        return None
    if len(reply) != _DEVICE_INFO_RESPONSE_LENGTH:
        return None
    if not reply.startswith(_DEVICE_INFO_DESCRIPTOR):
        return None
    return binascii.hexlify(reply[_SERIAL_NUMBER_SLICE]).decode('ascii').upper()


def resolve_lidar_ports(lidars, probe=probe_device_serial, ports=None):
    """Map each LiDAR name to the tty it is attached to right now.

    ``lidars`` are the entries from ``robot.yaml``.  Each may carry a
    ``device_serial`` prefix used for identification; the configured
    ``serial_port`` stays as the fallback so a machine without the hardware
    attached (or a config without ``device_serial``) behaves as before.

    Returns ``(resolved, responsive, notes)`` where ``resolved`` maps name ->
    device path, ``responsive`` is the set of LiDARs that actually answered the
    probe, and ``notes`` is a list of human-readable lines for the launch log.
    """
    if ports is None:
        ports = candidate_ports()
    ports = list(ports)

    observed = {port: probe(port) for port in ports}
    resolved = {}
    notes = []
    for port, serial_number in observed.items():
        notes.append(
            f'{port}: {"S/N " + serial_number if serial_number else "no reply"}'
        )

    unclaimed = list(ports)
    pending = []
    responsive = set()
    for lidar in lidars:
        name = str(lidar['name'])
        wanted = str(lidar.get('device_serial', '')).strip().upper()
        matches = [
            port for port in unclaimed
            if wanted and (observed[port] or '').startswith(wanted)
        ]
        if len(matches) != 1:
            if len(matches) > 1:
                notes.append(f'{name}: ambiguous S/N prefix {wanted}; refusing to guess')
            pending.append(lidar)
            continue
        match = matches[0]
        unclaimed.remove(match)
        resolved[name] = match
        responsive.add(name)
        notes.append(f'{name}: matched S/N {wanted}* on {match}')

    # A LiDAR that never answered cannot be identified, but if it is the only
    # one left and one candidate tty is also left, that pairing is unambiguous.
    if (len(pending) == 1 and len(unclaimed) == 1
            and observed[unclaimed[0]] is None):
        lidar = pending.pop()
        port = unclaimed.pop()
        resolved[str(lidar['name'])] = port
        notes.append(
            f'{lidar["name"]}: no reply from any port; assigned the only '
            f'unclaimed CP210x device {port} by elimination'
        )

    for lidar in pending:
        name = str(lidar['name'])
        configured = str(lidar.get('serial_port', '')).strip()
        resolved[name] = configured
        notes.append(
            f'{name}: could not be identified; falling back to the configured '
            f'{configured}'
            + ('' if os.path.exists(configured) else ' (which does not exist)')
        )

    # Nothing answered at all: either every LiDAR really is dead, or this
    # machine cannot probe (no pyserial, no permission, ports already opened by
    # a running stack).  Those are indistinguishable from here, so report every
    # LiDAR as responsive.  Callers use this set to switch off safety inputs,
    # and guessing "all dead" would switch off the collision monitor entirely.
    if not responsive:
        responsive = {str(lidar['name']) for lidar in lidars}
        notes.append(
            'no LiDAR answered the probe; assuming all are present rather than '
            'disabling scan inputs on what may be a probe failure'
        )
    return resolved, responsive, notes
