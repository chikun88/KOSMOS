# Simulation report — 2026-08-01

## Learning campaign

Thirty-six profiles were compared under identical CAD routes and noise seeds.
The selected balanced profile was:

- translation 0.78 m/s;
- lateral translation 0.702 m/s;
- yaw 1.30 rad/s;
- translation acceleration 0.85 m/s²;
- angular acceleration 2.0 rad/s²;
- 0.55 m lookahead and position gain 2.1 in the validation model.

The lower 0.85 m/s² acceleration beat 1.05 and 1.25 m/s² alternatives on
clearance and final error while retaining the best travel-time score. The ROS
runtime copies the physical speed/acceleration limits; MPPI owns its internal
path critic gains.

## Held-out gate

Seed `20260801`, conservative 5 cm CAD distance field with 5 mm interpolation
allowance and a tightened 40 mm arrival gate:

- 1,200 / 1,200 successful;
- 0 collisions and 0 timeouts;
- 1,080 / 1,080 critical red-region route repetitions successful;
- 120 / 120 random, connected, same-field-side routes successful;
- overall 95th-percentile position error: 0.03974 m;
- critical-route 95th-percentile position error: 0.03977 m;
- critical-route 95th-percentile yaw error: 0.1160 degrees;
- minimum simulated centerline-to-CAD-wall clearance: 0.42549 m.

Machine-readable evidence is in `simulation/results/autotune.json` and
`simulation/results/campaign_heldout.json`.

## ARM64 on-target software gate

The complete ROS 2 stack was also built and exercised on egg8PC in safe demo
mode (`motors:=false`, synthetic odometry and dual-LiDAR scans).  From the
field start pose, the image-marked critical target pose 4 completed with Nav2
`SUCCEEDED` under the tightened 40 mm / 2 degree goal checker.  The run took
33 seconds and recorded zero invalid commands, controller deadline misses,
progress-checker failures, TF failures, process deaths, or shutdown
tracebacks.  This test covers the real ARM64 ROS graph and command pipeline;
it is not a substitute for the physical acceptance procedure.

This is a deterministic software model with injected localization, slip, and
actuation noise. It demonstrates internal consistency and regression safety;
it cannot certify wheel-floor friction, chassis flex, mechanism envelope,
electrical faults, opponents, or errors in the source CAD. The physical tests
in `ACCEPTANCE.md` remain mandatory.
