# bacon6 gateway

This target is the hardware boundary for Omni Autonomy Next.  It accepts the
existing CRC16/sequence-protected v3 physical velocity datagram on UDP 8888,
fits the twist to the four motor limits, and writes the established COBS UART
frames to the RoboMaster development board.

The transport code is intentionally derived from the deployed
`/home/bacon6/MU3_robomas` implementation.  Changing this byte contract at the
same time as replacing localization and navigation would add avoidable motor
risk.  The new binary name and repository permit side-by-side build and
rollback; the old service is not overwritten.

Start only one gateway process at a time.  Startup requires a disarmed packet
followed by a fresh auto-request edge.  Link timeout, E-stop, and process exit
all command zero drive velocity.


## Boot service

The service state described below is the repository's deployment record.
The 2026-10-07 audit changed source only and did not inspect or update the
live bacon6 host. See [current audit scope](../docs/SYSTEM_AUDIT_20261007.md).

`omni-gateway-next.service` starts `auto_run_gateway.sh` at boot as user
`bacon6`, with the same real-time scheduling and Energy-Efficient-Ethernet-off
pre-start the previous receiver used.  It replaces
`mu3-robomas-receiver.service`, which is now stopped and disabled but still
present under `/home/bacon6/MU3_robomas` for rollback.

```bash
sudo systemctl status omni-gateway-next
journalctl -u omni-gateway-next -f

# roll back to the previous receiver
sudo systemctl disable --now omni-gateway-next
sudo systemctl enable --now mu3-robomas-receiver
```

The launcher refuses to start while the configured UDP port or motor UART is
already held by another process. The binary also binds UDP exclusively and
requests exclusive TTY ownership, so a second gateway fails at startup.

Every valid UDP frame is processed in order, including E-stop and DISARM/ARM
edges in a single burst. A reset/ARM after an E-stop still requires one complete
zero velocity/GPIO frame to be queued before motion can resume. Invalid frames
cannot acquire the source lock or consume a sequence number. Commands that
spent 250 ms in the Linux receive
queue are rejected; sender replacement requires a new DISARM/ARM handshake.
v4 frames reject duplicate command IDs and wheel targets beyond ±10000.

Local frames omit ARM/GM position command IDs until an explicit operator
button action or an authorized v4 position target has been received. Shutdown
includes only targets previously queued during this process. A wheels-only v4
frame cannot synthesize position
targets in its complement; once a position target has been commanded, later
partial frames retain that target. The variable triplet/COBS wire format is
unchanged. The first explicit position target still requires the mechanisms
to have been homed/calibrated to the command origin: the gateway has no measured
position feedback and cannot infer a safe physical starting position.

The MU3 PS button and the legacy PS bit latch an emergency stop even without
new UDP traffic. Emergency and link stops halt velocity mechanisms and release
momentary GPIO while retaining the last ARM/GM position targets. Shutdown sends
four complete stop frames and reports failure if they cannot be sent/drained.
Retaining a position target avoids commanding a new home position; it does not
prove that an ARM/GM axis already moving toward that target has stopped. A
physical emergency stop for those axes requires firmware abort/disable support
or measured positions, neither of which this gateway currently provides.
After a control-loop stall exceeding the 100 ms radio watchdog, the gateway
discards buffered MU3 input and requires newly received samples. UART failures
terminate the gateway; a persistently blocked UART is given 250 ms before stop
and shutdown attempts. Telemetry reports the last complete frame accepted by
the UART driver, rather than an attempted frame that was skipped. This is not
an acknowledgement from the board or evidence of physical motor motion.
The development board still needs an independent watchdog for process death,
power loss, a broken UART cable, or a blocked host.

Offline checks (loopback sockets and pseudo terminals only):

```bash
cmake -S bacon_gateway -B /tmp/kosmos-gateway -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build /tmp/kosmos-gateway -j2
ctest --test-dir /tmp/kosmos-gateway --output-on-failure
```

`MU3_DEVICE`, `MU3_MOTOR_DEVICE`, and `MU3_JETSON_PORT` permit isolated test
endpoints. A missing motor UART, busy UDP port, or invalid port override fails
startup; `--no-hardware` monitors UDP without opening either serial device.

Because this unit is enabled, the motor command path comes up at boot.  The
`ACCEPTANCE.md` wheels-raised tests still gate physical motion; disable the unit
or keep the wheels raised until they pass.
