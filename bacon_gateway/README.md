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

The launcher refuses to start while UDP 8888 or `/dev/serial0` is already held
by another process.  That check is necessary because the gateway socket sets
`SO_REUSEADDR`: a duplicate bind on 8888 succeeds rather than failing, and two
receivers would then split the command stream and both write to the same UART.

Because this unit is enabled, the motor command path comes up at boot.  The
`ACCEPTANCE.md` wheels-raised tests still gate physical motion; disable the unit
or keep the wheels raised until they pass.
