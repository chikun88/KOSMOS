# Deployment status — 2026-09-08

## 2026-10-07 repository audit status

The audit changes are repository changes, not evidence of deployment to egg8PC
or bacon6. See [SYSTEM_AUDIT_20261007.md](SYSTEM_AUDIT_20261007.md) for current
software verification and unresolved acceptance gates. Historical deployment
and campaign statements below apply to their recorded versions only.

## 2026-09-08 - the Android saved-pose buttons became a seven-point map and gained a field side

The Android app now shows an overhead half-field and the operator taps a
destination on it instead of pressing a row of raw pose names. Seven points are
exposed, mixing poses that were already operator-recorded with four that were
only ever fixed goals:

| Android | saved pose | left slot | right slot | seeded from |
|---|---|---|---|---|
| 自動装填 | `A` | 1 | 9 | goal 1 |
| バケツ2 | `BAKETU2` | 2 | 10 | goal 4 |
| バケツ3 | `BAKETU3` | 3 | 11 | goal 5 |
| 旗下 | `旗上側` | 4 | 12 | goal 6 |
| 旗上 | `旗下側` | 5 | 13 | goal 7 |
| 退避位置 | `退避位置` | 6 | 14 | goal 3 |
| スタート位置 | `装填位置` | 7 | 15 | goal 1 |

### Every point is now adjustable in place, including the four that were fixed

`旗上側`, `旗下側`, `退避位置` and `装填位置` were configured goals, so the only
way to correct where the robot stopped was to edit `field_poses.yaml` and
rebuild. They are saved poses now, which means the panel's existing
"現在地を記憶" path adjusts them. `config/field_poses.yaml` gained
`remembered_pose_defaults`, mapping each saved-pose name to the configured goal
it starts from; `goal_bridge` uses the recorded pose when one exists and the
seed otherwise, and publishes a `defaulted` list so the panel can mark the
points nobody has measured yet. A seed naming an unknown goal id is a startup
error — a slot with no pose would otherwise only be discovered when it is
tapped during a match.

The panel also gained one button per Android slot, so re-measuring a point does
not require retyping a name the radio protocol matches exactly.

### The right field is the left field reflected, not a second set of numbers

`field_poses.yaml` had carried `field_side: left` since it was written and
nothing read it. Every stored coordinate — configured poses, fixed departures,
fixed approach lanes and recorded poses — is now explicitly the left-field
value, and `field_side.py` reflects them about the divider at `x=0`
(`field_layout.yaml`: the divider occupies `x=[-0.300, +0.300]`). The reflection
keeps y, so `split_y` and the upper/lower lane roles survive it unchanged, and
the CAD map already spans both halves so no second map is needed.

Two consequences worth stating, because they are the reason for doing it this
way rather than storing both fields:

- Recording on the right field stores the reflected pose. An adjustment made on
  either field applies to both, so the points are measured once.
- A side change while a goal is running cancels it. The active route was
  planned through the other field's throat; continuing it would drive across
  the divider. Each `GoalRequest` therefore carries the side it was planned
  for, so a late switch cannot re-target a goal already in flight.

### The field travels in the same message as the point

`/navigation/remembered_goal_request` accepts `{"name": ..., "side": ...}` as
well as a bare name. The radio bridge uses the object form. A separate
`/navigation/field_side` topic would have been two messages on two topics with
no ordering guarantee between them, and the failure mode of losing that race is
the robot driving into the other team's half. The panel still publishes
`/navigation/field_side` for the operator's own selection, latched.

### No gateway change

The radio frame is unchanged. The 4-bit slot already carried 1..15 and
`test_mu3_navigation_wire.py` already covered all 16 values, so slots 9..15
reach `mu3_navigation` through the existing `byte29 = 0x80 | slot`. Slot 8
decodes to point 0 and is rejected. An older Jetson build would reject 9..15 as
`UNKNOWN_SAVED_POSE` and stop, rather than driving somewhere unintended.

### Verified

`colcon build`, 491 package pytest cases and 21 simulation cases on egg8PC.
New coverage: `test_field_side.py` (reflection is an involution, mirrored lanes
keep their upper/lower roles, every MU-3 slot has a seed, a seed naming an
unknown goal id fails at startup) and the extended `test_mu3_navigation.py`
(slot 1..7 vs 9..15 decoding, slot 8 rejected without emitting a goal).

`scripts/check_mu3_navigation_full_chain.py` on ROS domain 94 with loopback UDP,
extended to the seven points and both fields — MU-3 bytes through the real C++
gateway, the real `motor_udp_bridge`, `mu3_navigation`, real Nav2 and the real
tracker, down to non-zero per-wheel UART commands:

- Left field, slots 1..7, all arrived: `A` 55 plan points / 16.73 s,
  `BAKETU2` 163 / 13.79, `BAKETU3` 208 / 21.18, `旗上側` 188 / 13.91,
  `旗下側` 142 / 11.28, `退避位置` 195 / 12.28, `装填位置` 267 / 16.50. The last
  four ran from their seeds, so the seeding path is covered end to end.
- Right field, slots 9..15, goal x reaching Nav2: 2.007, 0.748, 0.801, 0.810,
  0.850, 3.880, 1.800 — all mirrored across the divider, `A` including its
  0.25 m reverse gate.
- Slot 8 refused as `UNKNOWN_SAVED_POSE`, no goal emitted.
- Stop button and radio loss during motion still DISARM with zero wheels, no
  idle re-arm, and no restart when the link returns with the button held.

bacon6: 5 C++ gateway tests pass. The gateway was not modified.

Android: APK build, 8 JVM cases on the wire encoding, and an emulator check of
the map, the mirror and the refusal to send while the MU-3 link is down.

Not verified: the robot starts on the left field in this harness, so the right
field is a command-path and target-coordinate check, not an arrival test. No
MU-3 radio and no motor UART were in the loop.

## 2026-09-04 - yaw wobble on hardware: the yaw loop had no margin against an unmeasured drivebase gain

The field report was that the base wanders in yaw while it drives to a goal.
Nothing in the offline model reproduced it, so the first work was to find out
why the model could not see it.

### The replay was measuring a controller that is no longer deployed

`scripts/yaw_oscillation_probe.py` replayed the control law at its plant step,
100 Hz.  The tracker has not run at 100 Hz since 2026-09-01: it was cut to
30 Hz and then to 20 Hz so `collision_monitor`'s swept-footprint check would
fit its budget.  The probe now integrates the plant at 100 Hz and runs the
control law at `CONTROL_HZ = 20`, holding the command in between, and filters
the measurement-wheel twist once per odometry sample rather than once per
control tick, which is where `_on_odom` actually does it.

The rate itself is not the fault.  Measured on the same three legs, 100 Hz
against 20 Hz: yaw peak-to-peak 0.45 -> 0.49 deg, cross-track 18.2 -> 3.8 mm,
output limiter engaged 11.5% -> 11.3% of ticks.  Measurement-wheel velocity
noise is not the fault either; at 20 Hz the loop is *less* sensitive to it
(limiter 59.7% -> 15.7% of ticks at 0.06 m/s of noise).  Both hypotheses are
dead, and the probe now agrees with the deployed controller.

### The gain from a command to actual motion has never been measured

The plant model assumed the base delivers exactly the commanded velocity.  What
sets that in the real chain is `auto_units_per_mps = 7202` in
`bacon_gateway/include/value.hpp`, copied into `robomas_uart.py`, which the v4
UART passthrough now uses to build the wheel frame on the Jetson.  Its
provenance is in the comment beside it: the translation gain was carried over
from full manual stick, *declared* to be 0.55 m/s.  It was never measured.  So
it is an unknown of the deployed system, not a property of it.

Three independent lines put it near x2.9:

- The one armed hardware run recorded on the chain
  (`logs/diagnose-chain-20260806-204616.log`): final command peaks of
  0.138 m/s lateral and 0.197 m/s forward against measurement-wheel response
  peaks of 0.400 and 0.623, so x2.9 and x3.2.  That run was degraded
  (`STALE_COMMAND` for half its samples), so treat it as an indication.  The
  odometry it is measured against is trustworthy: `robot.yaml` carries a
  calibration matrix identified against the LiDAR pose that agrees with the
  wheel geometry to within 6% on all three axes.
- If the development board's speed unit is motor rpm, which `omni_max = 8000`
  sitting exactly at the M3508's free-run limit suggests, then through the
  C620's 19.2:1 gearbox at r = 0.05 m, 8000 units is 3.09 m/s along a body
  axis, and 7202 units/(m/s) is x2.78 too large.
- The symptom itself, below.

### A gain error is a loop-gain error, and the yaw loop was the part with no margin

A drivebase gain multiplies the effective loop gain directly.  The loop is
delay limited (120 ms dead time, 80 ms first-order lag, plus the 20 Hz hold),
so past some gain it self-oscillates.  Yaw peak-to-peak against the reference,
and the share of ticks where the output angular-acceleration limiter is
saturated, on leg 4->5 with yaw held:

| `yaw_gain` | x1.0 | x2.0 | x2.5 | x2.8 | x3.2 |
| ---: | --- | --- | --- | --- | --- |
| 2.6 (was) | 0.7 / 0% | 1.3 / 0% | 12.3 / 66% | 17.4 / 84% | 28.4 / 91% |
| 2.0 | 0.6 / 0% | 1.0 / 0% | 1.2 / 0% | 1.8 / 0% | 13.2 / 61% |
| **1.6 (now)** | 0.6 / 0% | 0.9 / 0% | 1.2 / 0% | 1.3 / 0% | 1.4 / 0% |

The deployed 2.6 breaks into a sustained oscillation between x2.2 and x2.5,
which is below where the hardware evidence puts the gain.  That is the reported
symptom: yaw peak-to-peak of 17-25 deg with the angular-acceleration limiter
saturated on 84% of ticks is a base visibly hunting left and right in heading
while it translates.  The translation loop does not have this problem; a gain
error costs it cross-track error (3 -> 14 mm at x1.15) but it stays stable, and
lowering `position_gain` makes it worse, so it is unchanged at 2.4.

`yaw_gain` is therefore not a tracking-quality number at all.  Yaw is carried
by the feed-forward reference rate and this term only removes the residual, so
what the gain actually buys is margin.  It is now 1.6.  At the nominal gain the
two are indistinguishable over four legs: arrival within 0.01 s, final position
error identical at 1.1-1.3 mm, and yaw error at the accepted arrival slightly
better (0.01-0.56 deg against 0.02-0.67 deg).  The one thing that gets worse is
disturbance rejection: against a sustained 0.15 rad/s external yaw disturbance,
yaw peak-to-peak goes 3.4 -> 5.2 deg.  Wobbling 17-25 deg from a calibration
error is the larger harm.

`scripts/yaw_oscillation_probe.py --margin` reproduces the table, and
`test_the_yaw_loop_keeps_margin_against_a_drivebase_gain_error` pins it with a
self-contained yaw-axis replay that locates the same boundary (2.6 sustains
3.4 deg at x2.9 where 1.6 sustains none, and 1.6 still converges at x3.2).

### The gain is now measured rather than assumed

Lowering the gain buys margin; it does not make the calibration right.  A
x2.9 base still executes every feed-forward profile, acceleration limit, wheel
budget and envelope at x2.9, so the number has to be measured and written down.
It was invisible before, and is now reported in three places:

- `trajectory_tracker` publishes `flow_gain` in its status.  `_track_flow`
  already low-passes commanded and measured travel direction in the world
  frame; the ratio of their norms is the delivered gain.
- `scripts/diagnose_chain.py` reports it per axis from one 45 s run, lag
  compensated by the cross-correlation peak it already computes, and prints the
  `auto_units_per_mps` the measurement implies.
- `scripts/check_drive_directions.py` (ACCEPTANCE step 3) already drives one
  axis at a time at a known speed for a known time, so the expected
  displacement is known and the ratio falls out of the same run.  It now prints
  the gain per axis and the corrected constants for all three files.  Its
  clearance warning was also wrong for an unverified gain: it promised 15 cm
  per axis, and at x2.9 the base travels 42 cm.

Software verification is 138 package tests and 21 simulation tests.  The probe
numbers in the 2026-09-01 entries shift by a few percent because the control
rate is now modelled correctly; the conclusions there are unchanged.

Remaining, and required before any full-speed running:
`python3 run.py check-drive-directions --accept-motor-risk` on a raised or
cleared base, then set `auto_units_per_mps` (value.hpp), `AUTO_UNITS_PER_MPS`
(`robomas_uart.py`) and the derived `max_wheel_speed` (`robot.yaml`,
`8000 / units * cos 45 deg / r`) from the measurement.  The operating profiles
should not be raised in the same change.

## 2026-09-04 - start Nav2 when localization is ready, not after 26 seconds

Nav2 was deliberately behind a fixed 26 s timer because starting it while the
CAD wall lookup was consuming the Jetson had previously made lifecycle
bringup fail. The timer prevented that race, but it also held every automatic
goal in `WAITING_FOR_NAV2` long after a normal lookup had finished.

`system.launch.py` now watches `wall_localizer` process output on both stdout
and stderr. When the localizer reports its loaded walls, LiDARs, and initial
pose, a start-once gate launches Nav2 immediately. The existing 26 s value is
retained only as a fallback deadline when readiness output is unavailable.
Both paths share the same latch, so the fallback cannot launch a second Nav2
instance after a successful readiness start.

With the command line still passing `navigation_delay:=26.0`, the integrated
synthetic run measured:

- localizer ready and Nav2 process launch: 7.64 s after process startup;
- all managed Nav2 nodes active: 12.12 s;
- automatic goal submitted: 12.39 s, versus 30.43 s before this change;
- pose 1 to pose 2 succeeded on its first attempt.

The process was kept alive beyond the 26 s fallback deadline and no duplicate
Nav2 nodes were launched.

## 2026-09-04 - prevent crawl when starting at loading pose 1

The remaining location-dependent crawl was reproduced geometrically at the
default match start, pose 1. Its -90 degree NORMAL footprint has 48.6 mm of
CAD clearance, but the old direct 1-to-2 plan distributed the 90 degree yaw
change from the first path sample. Intermediate footprints overlap the nearby
fixed structure, so Collision Monitor's 1.2 s `FootprintApproach` prediction
correctly scaled the complete twist toward zero.

`routes.yaml` now supplies a mandatory loading-bay departure gate at
`(-1.80, 4.10, -90 deg)`. `goal_bridge` submits that gate before any configured
or RViz destination whenever localization starts within 0.20 m of pose 1. The
position path therefore first moves 0.65 m toward the opening instead of
cutting directly along the wall. Exact CAD regression checks establish at
least 40 mm over the complete combined path with the tracker's yaw ramp,
60 mm over the complete yaw sweep at the gate, and 60 mm on the
gate-to-pose-2 transition. Selecting pose 1 while already there remains a
no-op. A retry while the base is partway down the departure line keeps the same
gate instead of reverting to a direct plan.

Software verification passes 125 ROS package tests and 21 simulation tests.
An integrated synthetic run selected `route=loading_bay_exit` and completed
pose 1 to pose 2 on its first attempt in 7.27 s. Collision Monitor reported
only the configured 80% `SlowZone`; no `FootprintApproach` scaling occurred
during the active leg.
This change still requires a low-speed physical-field replay before the
hardware gate can be considered complete.

## 2026-09-02 - remove intermittent stops beside fixed buckets 2 and 3

The bidirectional fixed routes now reach every tested goal, but the latest full
GUI demo still recorded two visible command interruptions while leaving the
fixed-bucket area. RuntimeGuard entered `STALE_COMMAND` for 0.47 s and 0.68 s
during active navigation. Both events coincided with a dense sequence of
Collision Monitor `FootprintApproach` checks; there were no invalid LiDAR
sources, collisions, localization failures, or Nav2 action failures. All five
goals in that run succeeded on their first action attempt.

The remaining overload was the 30 Hz trajectory output feeding a complete
ten-vertex swept-footprint simulation 30 times per second. The tracker now runs
at 20 Hz, equal to Nav2's controller frequency. This removes one third of those
checks while preserving the controller's command bandwidth. The 1.2 s
collision horizon, 0.05 s collision simulation step, exact footprint, 0.25 s
RuntimeGuard watchdog, and every stop condition are unchanged; the fix does not
hide stale data or weaken obstacle detection.

Package verification is 121/121 tests. After a clean restart, a full operator
replay with the GUI active completed all five legs on their first action:
`1->4`, `4->5`, `5->4`, `4->5`, and `5->3`. Every leg selected its configured
fixed approach/departure route. RuntimeGuard recorded zero `STALE_COMMAND`
transitions during active navigation. The largest tracker command gap was
61 ms and the largest post-collision-monitor gap was 98 ms, both comfortably
inside the unchanged 250 ms watchdog.

## 2026-09-01 - predetermined approaches and departures for goals 4 and 5

Goals 4 and 5 were previously submitted as single `NavigateToPose` targets.
Although the CAD planner could reach both, every replan remained free to choose
a different diagonal beside the fixed bucket. The route names in `routes.yaml`
did not contain geometry and were not consumed by the goal executor.

Both goals now have CAD-validated upper and lower gates. `goal_bridge` selects
the entry gate from the robot's localized side and submits `[gate, goal]` to
`NavigateThroughPoses`. The fixed segments are:

| goal | upper gate | lower gate | final goal |
| ---: | --- | --- | --- |
| 4 | `(-0.80, 1.75)` | `(-0.80, 0.75)` | `(-0.81, 1.34)` |
| 5 | `(-0.80, -1.95)` | `(-0.80, -2.75)` | `(-0.81, -2.31)` |

All four final segments retain at least 40 mm of exact NORMAL-footprint CAD
clearance. A custom through-poses behavior tree replans at 1 Hz and does not
drop a gate until the base is within 0.12 m. Collision Monitor, RuntimeGuard,
dynamic planning to the gate, cancellation, and retries remain active.

Departure uses the same validated segment in reverse. The side of the next
destination chooses the exit: 4-to-5 is
`fixed_bucket_2_to_lower -> fixed_bucket_3_from_upper`, and 5-to-4 is
`fixed_bucket_3_to_upper -> fixed_bucket_2_from_lower`. Configured goals and
RViz goals both receive the departure gate. Gate selection is rebuilt from the
latest localized pose immediately before action submission and every retry;
selecting the same numbered goal while already there remains a no-op instead
of driving out and back.

The installed stack selected `fixed_bucket_2_from_lower` from goal 5 and Nav2
accepted two poses; the live `/plan` ended on the configured vertical lane.
Earlier runs from above selected the upper variants and reached goals 4 and 5
on the first action attempt. Package verification is 121/121 tests. Physical
field clearance and low-speed hardware arrival still require acceptance work.

An isolated runtime probe of the departure extension produced exactly two
intermediate poses in both directions. From goal 4 to goal 5 the route ID was
`fixed_bucket_2_to_lower+fixed_bucket_3_from_upper`; from goal 5 to goal 4 it
was `fixed_bucket_3_to_upper+fixed_bucket_2_from_lower`. This verifies that
neither shuttle can begin with a free planner-selected diagonal beside its
origin bucket.

## 2026-09-01 - fixed-bucket goals 4 and 5 reach successfully

Goals 4 (`fixed_bucket_2_upper`) and 5 (`fixed_bucket_3_lower`) failed only in
the full operator configuration. The same paths passed without the speed GUI,
which isolated the failure from the goal coordinates and planner geometry.

The tracker was publishing `/cmd_vel_nav_smoothed` at 100 Hz. Beside a fixed
bucket, `collision_monitor`'s `FootprintApproach` sweeps the full footprint for
1.2 seconds of predicted motion for every command. Under the full load those
callbacks accumulated: the tracker input remained near 99 Hz, but
`/cmd_vel_collision_safe` fell to 15-24 Hz with a measured maximum gap of
2.225 s. The 0.25 s `RuntimeGuard` command watchdog correctly entered
`STALE_COMMAND`, stopped the base, and Nav2 then aborted with `Failed to make
progress`.

The tracker command rate is now 30 Hz, matching the existing
`velocity_smoother` rate. The collision prediction horizon, footprint,
clearances, command watchdog, and every downstream safety gate are unchanged.

Replayed with the GUI active, the same 120 ms command delay and 80 ms plant
lag, and the route that previously reproduced the failure:

| leg | result | elapsed |
| --- | --- | ---: |
| arbitrary RViz point `(-3.375, 0.919)` -> goal 4 | succeeded, first attempt | 9.49 s |
| goal 4 -> goal 5 | succeeded, first attempt | 15.63 s |

During goal 5, `/cmd_vel_collision_safe` held 30.08 Hz with a 56 ms maximum
gap. There were no `STALE_COMMAND` transitions or progress-checker failures
while either goal was active. A stale transition after each completed goal is
expected: the tracker deliberately stops publishing after `PLAN_STALE`.

## 2026-09-01 - smooth rotation while translating, and 2-12% shorter legs

The request was for the robot to reach a goal quickly and efficiently while
rotating smoothly. Measured on the same plant model as the 2026-08-06 and
-08-07 entries (120 ms dead time, 80 ms first-order lag, sensors at the rates
the real nodes publish, measurement-wheel quantisation) over four real legs.
`scripts/yaw_oscillation_probe.py --rotation` and `--terminal` reproduce every
number here.

### `optimize_yaw` was not merely unprofitable, it was a trap

The launch has exposed `optimize_yaw` since the tracker was written, defaulting
to false because the 2026-08-06 measurement found it slightly *slower*. What
that measurement did not show is that the plans it produced were not executable
at all.

Leg 1->2, yaw -90 to 0 degrees over 2.39 m, the plan it returned:

    knot        0     1     2 ...  10     11     12
    yaw [deg] -90 -178.6 -178.6  -178.6 -358.6 -360.0

88.6 degrees inside the first 0.199 m, then 180 more in the last segment. That
is 7.76 rad/m, and at any usable speed it demands about **105 rad/s^2 of the
2.00 rad/s^2** the chain can deliver. The output rate limiter refuses it, so the
base cannot follow its own reference yaw; the leg took 6.81 s against 5.03 s for
the plain linear ramp, and the yaw tracking error peaked over 100 degrees.

**Root cause: the candidate grid was coarser than the feasible step.**
`yaw_candidates` built a fixed 45-degree grid, while one 0.199 m knot at
1.30 rad/s and 0.78 m/s can turn 19 degrees. Every transition between adjacent
grid points was therefore infeasible, the "all candidates unreachable" fallback
fired at every knot, and that fallback ignored the step limit by construction.
A gradual turn did not exist anywhere in the search space.

Four changes make the option honest:

- **The grid spacing is now `min(45 deg, step_limit)`**, so adjacent candidates
  are always reachable and a gradual turn is representable.
- **The linear ramp is a candidate at every knot.** The search space now
  contains the deployed default, so the optimum cannot be worse than it.
- **The objective is traversal time in seconds**, `ds / speed_limit(direction,
  dyaw/ds)`. It was `-(speed + clearance) + 0.35 * |dyaw|`, which adds m/s to
  radians and evaluates the speed at *zero* yaw rate: it counted the gain from
  turning and not the wheel budget the turn spends. Time counts both in one
  unit. Clearance stays a preference, converted to seconds at the same ratio.
- **The unreachable-candidate fallback keeps its units** and prefers the least
  rotation instead of silently breaking the rate limit.

Same leg after the fix: a monotone 7.5 deg per knot, every step inside the
19 deg budget.

### Rotation is now bounded where it is planned, not clipped at the output

Two mechanisms, because a discrete optimiser produces a polyline:

- `smooth_yaw_profile` rounds the knot corners with the same [1,2,1]/4
  Dirichlet iteration `smooth_path` uses, so the start (measured yaw) and the
  end (commanded yaw) are exact while the second derivative becomes finite.
  Measured 6.3x reduction on a kinked profile.
- `Trajectory` now folds the angular rate and angular acceleration into the
  speed profile, exactly as it already did for curvature and lateral
  acceleration: `v <= wz_max / |yaw'(s)|` and `v <= sqrt(az_max / |yaw''(s)|)`.
  The feed-forward yaw rate is therefore executable instead of being clipped
  afterwards, and the clipped part no longer shows up as reference-yaw lag.
  `sample()` reads a precomputed centred derivative rather than a two-point
  difference, so the commanded yaw rate is continuous.

### The yaw is retired before the final approach, not at the goal

The yaw used to reach its target at the goal, so the reference yaw was still
moving during the low-speed terminal approach and the yaw error at the arrival
instant was proportional to the speed of that approach. It showed up as a
coupling: raising the terminal cap from 0.16 to 0.35 m/s degraded the arrival
yaw error on leg 1->2 from 1.30 to 1.74 degrees, against a 2.00 degree
tolerance. Yaw now finishes `terminal_approach_m` before the goal and holds,
which makes the last stretch a pure translation settle. Same comparison
afterwards: 0.30 -> 0.37 degrees.

That decoupling is what made the next change safe.

### The low-speed terminal approach was most of what a leg cost

`terminal_approach_m` / `terminal_speed` were 0.16 m / 0.16 m/s, chosen when
the overshoot they existed to suppress was still real. Latency compensation
(2026-08-06) removed the overshoot itself, so the crawl was paying for
something already paid for. Swept over the four legs:

| zone / cap | 1->4 | 1->2 | 4->5 | 3->6 | overshoot | settled | yaw at arrival |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.16 / 0.16 | 6.42 s | 4.99 s | 6.61 s | 5.56 s | 0.3 mm | 0.3 mm | 0.02-0.30 deg |
| **0.10 / 0.30** | **6.28 s** | **4.51 s** | **5.93 s** | **4.91 s** | 0.3 mm | 0.3 mm | 0.01-0.92 deg |
| 0 (no servo) | 6.42 s | 4.41 s | 5.93 s | 4.87 s | 16.7 mm | 16.7 mm | 0.01-1.85 deg |

**2.2 to 11.7% shorter legs at the same 0.3 mm settled accuracy and no
overshoot.** Removing the servo entirely gives nothing more and breaks settling
by 50x, so the low-speed stage is required — only its size was wrong.

### The tracker now plans inside the envelope the guard actually passes

This was a live defect, not an optimisation. In tracker mode
`velocity_smoother` is bypassed, so `trajectory_tracker` is the only stage that
shapes a command — and it read the speed profile once, at startup, from
`default_profile`. It never saw:

- the operator's profile selection, so choosing `sprint` in the GUI changed the
  guard's limits and nothing else. The base did not speed up.
- the operator's speed scale, so `docs/ACCEPTANCE.md` step 3 ("drag the slider
  to 10% before the first armed motion") had the tracker feeding forward
  0.78 m/s trajectories into a gate passing 0.078 m/s.
- the learned residual's `speed_scale`, which the deployed policy sets to
  0.35, 0.5, 0.7 or 1.0 by state, and the `red_zone` factor of 0.70 that
  applies within 0.95 m of every configured pose. **Both are on the operating
  path all the time**, so near every goal the tracker was following a time
  parameterisation about 30% faster than the gate would pass.

`trajectory_tracker` now subscribes to `/system/safety_state`, which
`runtime_guard` already publishes at 10 Hz with `profile` and `applied_scale`.
The guard stays the only authority; the tracker only reads what it reports.

- A **profile** change rebuilds the trajectory, because it changes the shape of
  the translation envelope and therefore the per-point speed limit.
- The **scale** is a single uniform factor, so it needs no rebuild: the
  reference clock advances at that rate and the reference velocity and yaw rate
  are read at the same factor (path velocity scaling). The geometry and the yaw
  profile are functions of arc length and do not move. Acceleration scales with
  the square of the factor, so it is always inside the original bound.
- Scale 0 (disarmed, E-stop, tracking or motor-link loss) now freezes the
  reference on the robot instead of letting it run away for
  `max_reference_lead_m` to claw back, and the stall and terminal watchdogs do
  not accumulate while the gate is shut.

If the topic never arrives the tracker keeps scale 1.0 and its startup profile,
which is exactly the previous behaviour. A stale value is kept rather than
reset to 1.0, because keeping it only ever slows the plan down.

### A bug this work introduced and the smoke test caught

Retiring the yaw early mixed two arc-length grids. The knot indices came from
`searchsorted(arclength, targets)`, which returns the first point *at or after*
each target, so the last knot could sit past the cutoff and the interpolation
never reached it. The reference yaw went flat 1.29 degrees short of the
commanded yaw — 65% of the 2.0 degree tolerance, spent on nothing. The index
and cutoff arithmetic is now one tested function, `yaw_plan_knots`, and the
smoothing is applied only up to the last knot so the 0.25 m kernel cannot smear
the flat terminal stretch (it had put 2.76 degrees back into the last 0.15 m).

### `optimize_yaw` stays false, and now that is a measurement rather than a trap

| leg | linear ramp | yaw optimisation | sprint + optimisation |
| --- | --- | --- | --- |
| 1->4, 90 deg | 6.28 s | 6.08 s | 5.73 s |
| 1->2, 90 deg | 4.51 s | 4.68 s | 4.67 s |
| 4->5, 0 deg | 5.93 s | 5.93 s | 6.95 s |
| 3->6, 11 deg | 4.91 s | 4.91 s | 5.24 s |
| yaw accel demanded | 2.5-8.5 rad/s^2 | 2.8-41 rad/s^2 | 17-61 rad/s^2 |
| output limiter engaged | 0.5-1.2 % | 0.7-16.3 % | 3.4-8.8 % |

The ramp wins or ties on time and wins clearly on smoothness. The reason is
structural and worth stating: the largest circle inside the omni diamond has
radius `1.111 / sqrt(2) = 0.786 m/s`, and `balanced` at 0.78 is already at that
ceiling, so there is almost no direction-dependent speed left to exploit.
Reaching the 1.111 m/s body axis requires an anisotropic profile plus turning
the base into the direction of travel, and yaw and translation share one
15.709 rad/s wheel budget: at 0.95 m/s only 0.24 rad/s of yaw is left, so a
90 degree alignment costs over 3 s and the legs here are 2.4-3.7 m. It does not
pay back.

The remaining roughness in the optimised mode is the optimiser itself running
inside the 1 Hz replan: a discrete solver can flip branches between rebuilds,
and the reference yaw rate steps when it does. Located precisely - on leg 1->4
the peak demand starts at t = 4.00 s, a replan instant, and lasts the 0.12 s
the rate limiter needs to catch up. Fixing that would need hysteresis toward
the previous solution, which is not worth building for a mode that is not
faster.

Enabling it costs 240-280 ms per plan against 0.2 ms for the ramp, on the
worker thread, so it does not disturb the 30 Hz loop.

### Not established

Nothing here has run on the real drivebase, and per the 2026-08-07 evening
entry the drivebase does not respond to MU3 either, so it cannot yet. Every
number is against the same 120 ms / 80 ms model. What the model cannot decide:

- whether 0.30 m/s over the last 0.10 m is acceptable beside a fixture with
  47-59 mm of footprint clearance. The model says the approach is monotone with
  no overshoot; the field has not said anything.
- whether the drivebase can deliver the 0.85 m/s^2 the trajectory is built
  with. The chain once measured 0.373 m/s^2. The guard's `balanced` profile
  allows 1.30, and in tracker mode `velocity_smoother` is not in the path at
  all, so planning at the shaping stage's number is now a conservative choice
  rather than a necessary one. That is the next measurable item and it is worth
  more than anything above: two acceleration ramps are about 1.6 s of a 6 s leg.

117 package tests and 21 simulation tests pass.

## Older entries — 2026-08-07 and before

## 2026-08-07 - lateral sign was inverted, and the terminal zone had no way to stop

Two separate faults produced one symptom: the base wobbled left and right while
tracing a small circle, and never stopped.

**The circle: `auto_lateral_sign` was `-1.0`.** Measured on the robot: left and
right go the wrong way. A lateral inversion is a reflection, not a rotation, so
the position loop becomes `-K diag(1,-1) e` and the lateral axis *diverges*
rather than converging. Saturation and the 1 Hz replan bound the divergence, so
what it looks like from outside is a weave, not a runaway.

The value was set to `-1.0` on the strength of the 2026-08-05 observation
"wove left-right while translating and reversed the final approach". That is
exactly the signature of the inversion it was introduced to fix. Under MPPI,
PathAlignCritic (weight 24) pulled the base back every cycle so the divergence
read as a weave; near the goal PathAlign fell below its threshold, the Goal
critics took over, and the final approach ran the other way. The observation
was evidence *against* `-1.0`, not for it.

| axis | contract measured on the robot | constant | was | now |
| --- | --- | --- | --- | --- |
| forward | mixer translation input positive -> forward | `auto_forward_sign` | +1.0 | +1.0 (unchanged) |
| lateral | mixer lateral input positive -> **left** | `auto_lateral_sign` | -1.0 | **+1.0** |
| yaw | mixer turn input positive -> clockwise | `auto_turn_sign` | -1.0 | -1.0 (unchanged) |

The consistent wheel-to-corner assignment moves with it: `m1` and `m2` swap, and
so do `m3` and `m4` (`m1`=rear-right@135, `m2`=front-right@225,
`m3`=front-left@-45, `m4`=rear-left@45). Which corner is wired to ID 1 is
recorded nowhere, which is why the earlier derivation could reach the opposite
answer while staying internally consistent. `scripts/e2e_chain_check.py`
carries the same table for its independent geometric cross-check and was
updated with it; mixer and geometry now agree to one common ratio (509.26
units per rad/s) on every wheel of every case.

Three code paths apply this sign and all three were changed together:

- `bacon_gateway/include/value.hpp` (v3 velocity path)
- `ros2_ws/.../robomas_uart.py` (**the deployed one** — `payload_format` is
  `v4_uart`, so egg8 builds the UART frame and bacon6 passes it through)
- `bacon_gateway/src/main.cpp` legacy/v2, which negated `lx` to match the old
  contract and now passes it through

**Never stopping: two holes in the terminal logic.**

- `_build_trajectory` reset `finished_at = None` on every plan. The BT replans
  at 1 Hz, so `terminal_hold_sec = 3.0` was folded back every second and
  `TERMINAL_HOLD_EXPIRED` could never be reached. The latch now survives a
  replan onto the same endpoint, exactly like the progress record.
- The stall watchdog excludes the terminal zone, because remaining arc length
  legitimately stops shrinking while the base settles. A base orbiting inside
  the 0.16 m zone without reaching the 0.04 m goal tolerance therefore hit
  neither guard, and the BT kept feeding it paths forever. `terminal_latch` +
  `terminal_timeout_sec` (4.0 s) closes it, releasing only when the base leaves
  2x the zone so an orbit crossing the boundary cannot keep resetting it.

`NO_PROGRESS` and `TERMINAL_TIMEOUT` now relay `behavior_server` output like
the other give-up paths; publishing zeros there meant the progress checker's
Spin/BackUp never reached the wheels.

**New in `/trajectory_tracker/status`: `flow_gap_deg`**, the angle between the
commanded and the measured direction of travel in the world frame, smoothed
over 0.5 s. A steady large angle is a rotation error (yaw estimate or turn
sign); a gap whose sign flips with the direction of travel is a reflection,
i.e. this lateral fault. Either way the log now names the axis to suspect
instead of only recording that the base failed to converge.

## 2026-08-07 evening - the drivebase does not respond to MU3 either, so this is not an autonomy fault

On the field, on the ground, the robot does not move. It does not move under
autonomy and it does not move under MU3 manual control, which does not involve
egg8 at all. What bacon6's own record shows, with no Jetson connected
(`jetson_cmd=未受信`, `source=MANUAL`), is the gateway reading the sticks and
producing drive intent:

    applied_cmd=その場右旋回 / 前進 / 前進 + 右旋回 / 左移動 + 左旋回
    uart_tx=1805 -> 2006 -> 2206 -> 2407 (rising)

So MU3 reception, the mixer and the UART writes are all alive. Three software
paths that could still zero the wheels were checked and are all clear:

- `estop_latched_` — `ESTOP(Jetson)` appears 0 times in 18990 status lines
  since the 18:12 boot, and `AUTO BLOCKED` 0 times.
- `stop_latched_` — would display as `CONTROLLED_STOP`, which takes priority
  over `DISARMED` in `DriveLinkSafety::update`. The line reads `DISARMED`, so
  it is not latched, and `stop_required()` is therefore false and
  `ControlledStopLimiter` is passing the manual wheel values through.
- Frame format — `passthrough_wire_check.py` decodes the emitted bytes cleanly
  into the expected nine `[id, value]` commands under COBS.

`/dev/serial0 -> ttyS0` was checked and is not a regression: on a Pi 4 with
Bluetooth enabled that is the normal mapping, `enable_uart=1` is set, and
`/boot/firmware/config.txt` has not been modified since 2026-03-15, well before
the 18:12 reboot.

That places the fault downstream of bacon6's UART output — the wiring to the
Development Board, the board itself, the CAN bus to the C620s, or motor power.
No change to the egg8 autonomy stack can move the robot while MU3 cannot.

## 2026-08-07 - none of the tracker work had ever run: the launcher could not ask for it

Three real runs today failed the same way. Goal 3 (`retreat`, -3.88, -0.55)
was requested from the loading pose and aborted every time — 180 s at 12:40,
180 s at 14:19, 198 s at 16:07 — with `controller_server` reporting
"Failed to make progress" every eight seconds and the robot ending within
0.3 m of where it started. The Spin and BackUp recoveries also timed out:
a 90-degree spin did not complete inside its 10 s allowance, four separate
times.

The cause is not in the control law. It is that the control law was not
running. `trajectory_tracker` only starts under `IfCondition(tracker)`,
`system.launch.py` declared `tracker` with `default_value='false'`, and
`run.py` — the only launcher used in operation — never passed the argument at
all. Every operator run therefore fell back to the 20 Hz MPPI feedback loop,
including every run in the two entries below, whose measurements describe the
tracker.

`scripts/diagnose_chain.py` has recorded both stages, but **not on the same
plant**, and the difference matters when reading these numbers. The 16:31
tracker run is launch `2026-08-06-16-30-56`, which starts `synthetic_scans`
and no `sllidar_node`, no `measurement_wheel` and no `motor_udp_bridge`: it is
the demo. The 20:46 MPPI run is launch `2026-08-06-20-45-29`, which starts all
four: it is hardware. Across the whole log history only 2 of the 16 launches
that ever started `trajectory_tracker` had a real LiDAR attached, and both are
from 2026-08-07 after this entry was written.

| | tracker, 16:31 (**demo**) | MPPI, 20:46 (**hardware**) |
| --- | --- | --- |
| cross-track RMS to `/plan` | 0.039 m | 0.463 m |
| cross-track mean | -0.001 m | **+0.365 m** |
| cross-track p95 | 0.087 m | 0.657 m |
| `/cmd_vel_safe` speed p95 | 0.706 m/s | 0.203 m/s |
| RuntimeGuard ACTIVE : STALE_COMMAND | 515 : 1 | 131 : **146** |
| dominant vy oscillation | none reported | **1.59 Hz** |
| measured lag, MPPI output to `/cmd_vel_safe` | — | vx **520 ms**, vy 180 ms |

So the right-hand column is what this robot has been doing, and the left-hand
column is not a control against it. What the right-hand column does establish
on its own is that the MPPI path carries about half a second of chain lag,
weaves at 1.59 Hz a third of a metre off one side of the path, and runs at a
quarter of the profile speed. The `tracker=1` launches on 2026-08-06 at 16:25
and 16:31 were demos started by hand; every `run.py` launch before and since
carried `tracker=0`.

### Fixed

- **`run.py` passes `tracker:=`, and `--no-tracker` falls back to MPPI.** The
  launcher had no way to ask for the tracker at all, so the comparison this
  needs could not be run without hand-writing a `ros2 launch` line.
  `test_the_launcher_can_actually_ask_for_the_tracker` pins it. The default is
  `true`, restored at the operator's request so the stack matches the 16:59
  configuration of 2026-08-07; the argument's comment in `system.launch.py`
  records what is and is not established about that choice.
- **`run.py` no longer overrides `navigation_delay` with 18.0.** The launch
  file raised it to 26.0 after measuring that 18.0 times out the lifecycle
  bringup, and `test_navigation_start_is_sequenced_after_map_and_localization_setup`
  already asserted >= 24.0 — but `run.py` passed 18.0 on every operator
  launch, so the measured value had never been used. The 16:07 run shows the
  documented symptom: `goal_bridge` logging "waiting for not active:
  bt_navigator, collision_monitor, controller_server, planner_server,
  velocity_smoother".

Synthetic demo, goal 3 from the loading pose, which no earlier entry had run
(all previous demo numbers come from the goal 4 <-> 5 shuttle):

| | time to `Goal succeeded` |
| --- | --- |
| MPPI (`tracker:=false`) | 17.4 s |
| tracker (`tracker:=true`) | 11.3 s |

Both reach the goal. This is the demo, so it says nothing about the hardware
question above; what it does do is exercise the whole 2026-08-07 entry below —
`smooth_path`, the reference-at-`t + lag` sampling, the velocity filter and the
new `PLAN_STALE` / `NO_PROGRESS` / `TERMINAL_HOLD_EXPIRED` gates — inside a
live system rather than in unit tests. 106 package tests pass.

### The runs from 16:59 on are wheels-raised, and must not be read as failures

Launches `2026-08-07-16-59-57` (tracker) and `2026-08-07-17-33-41` (MPPI) both
have the full hardware node set, but the robot was up on blocks for both. Read
naively they look like the failures above — 35 "Failed to make progress" in the
17:33 run, Spin and BackUp timing out, `LOCALIZATION_UNHEALTHY` 21 times — and
none of that means anything: a robot that cannot translate cannot satisfy a
0.04 m progress check, and the operator moved the estimate around with
`Pose reset from RViz` while the LiDARs kept seeing wherever the robot was
actually propped, so ICP rejects and the guard opens and closes. Nothing in a
raised run distinguishes a working stack from a broken one by outcome. What it
can show is whether the chain carries a command, and that is what to read.

**The chain does carry commands.** From bacon6 over the 17:33 run, 481 status
samples: `proto=v4`, `crc_err=0`, `stale=0`, `uart_src=EGG8_PASSTHROUGH
(4cmd/14B)` on 422 of them, `safety=ACTIVE` 422 / `CONTROLLED_STOP` 2 /
`DISARMED` 57, and `jetson_cmd` reading `その場右旋回` 13 times and `左前`
twice — egg8 asking for motion above the display deadband, relayed byte for
byte. The `navigation_delay` fix is also visible: 26.0 s between the first node
and `controller_server`, against 18.0 s before.

Two measurements say where to look next.

**The commanded speed never leaves a crawl, on either controller.** bacon6's
gateway classifies the Jetson's own reference velocity, and it read `停止` on
every one of the 540 samples of the 16:59 tracker run and every one of the 321
samples of the 16:07 MPPI run. That classifier's deadband is
`value::omni_deadzone` on a `auto_display_full_mmps` scale, i.e. 0.113 m/s and
0.246 rad/s, so what it establishes is that `/cmd_vel_safe` stayed below those
for both entire runs — consistent with the 0.203 m/s p95 measured at 20:46, and
with the 4 m the robot did cover between 699 and 785 s at an average of
0.05 m/s. The first thing to check on the next run is RuntimeGuard's
`applied_scale` and the `/speed_limit` percentage it publishes: the 20:46
diagnostic ends with `speed_limit_pct=31` and `applied_scale=0.0`, and
`docs/ACCEPTANCE.md` step 3 has the operator start the GUI slider at 10%.

The operator has since confirmed the slider was left at 10-30 % for these
runs. `0.31 * 0.78 = 0.242 m/s` accounts for the 0.203 m/s p95 at 20:46
directly, so the low speed is the speed scale and not the command chain, and
the MPPI column above cannot be read as evidence against MPPI. What that does
not explain is 0.29 m in 198 s on 2026-08-07 16:07: even at 10 % of profile the
progress checker's 0.04 m in 8.0 s has 0.078 m/s * 8 s = 0.62 m of headroom, so
something downstream is still removing most of an already small command.
`FootprintApproach` is the first suspect and `scripts/diagnose_chain.py`
separates it — it reports max |vx| at `/cmd_vel_rl`, `/cmd_vel_collision_safe`
and `/cmd_vel_safe` side by side.

**`PLAN_STALE` freezes the robot for as long as Nav2 stays in recovery.** In
the 16:59 run the tracker sat in `PLAN_STALE` — publishing zeros — for 13.3,
13.2, 9.2, 13.8, 13.7, 13.8, 9.5 and 13.8 s. The gate itself is correct: a
tracker must not keep following a trajectory nobody is refreshing. But combined
with the `cmd_vel_nav` gap below it means that in tracker mode a robot in
recovery does not move at all for the whole recovery, whereas in MPPI mode the
behaviors at least command something. That is one of the reasons the default
stays on MPPI for now.

### Fixed: `applied_cmd` was meaningless on the v4 path

`printJetsonStatusLine` was handed `ctrl_packet` as the applied command, and in
passthrough mode `ctrl_packet` deliberately keeps the MU3 stick axes untouched
— the drivebase comes from the relayed bytes, not from those axes. So the one
field an operator reads to answer "is anything being commanded?" printed
`停止` on all 481 samples of the 17:33 run while `jetson_cmd` was showing
`その場右旋回`. It cost two wrong readings during this investigation before the
call site was checked. It now derives from the velocities that actually drive
the wheels, on the same scale as `jetson_cmd` so the two fields compare
directly, and it still reads `停止` under a controlled stop or E-stop because
those zero the reference upstream. 3 ctest cases pass.

### MPPI cannot hold its configured rate on this Jetson

Not new, but it is now measured on three separate days and it is not a
consequence of anything above. `controller_server` logs "Control loop missed
its desired rate of 20.0000 Hz" at 9.68 Hz (2026-08-06 20:21), 13.1-22.3 Hz
(2026-08-07 16:07) and 9.78-19.2 Hz fourteen times in the 17:33 run. `model_dt`
is pinned to 1/`controller_frequency` = 0.05 s, so every rollout is integrated
on a grid the loop does not keep and the optimizer's own prediction runs fast
by the same ratio. The comment on `batch_size` in `nav2_next.yaml` already
warns about this. It has to be re-timed on an idle host before any conclusion
about MPPI's tracking on the ground is worth drawing.

### Not established

The 2026-08-07 tracker changes have run on hardware once, wheels raised, so
nothing about their effect on the real drivebase is established either way.
Neither is anything about MPPI's, beyond the 16:07 run's 0.29 m in 198 s.

### The 2026-08-06 20:21 run: the front LiDAR stopped publishing 15 times

Launch `2026-08-06-20-21-16`, hardware, MPPI. This one has a separate and much
more direct fault, and it is the run the operator filmed.

`/scan_front` stopped reaching every one of its subscribers fifteen times, for
0.26 s to 5.5 s each, spread over the whole four minutes. Both
`scan_source_supervisor` (watching `/scan_front_filtered`) and `wall_localizer`
(watching the raw `/scan_front`) saw the same gaps at the same moments, so the
driver stopped publishing rather than the filter stopping. The rear unit never
missed a revolution. What the collision monitor logged:

| | 20:21 run | 16:07 run |
| --- | --- | --- |
| `[scan_front]: ... Ignoring the source` | **395** | 0 |
| `Robot to stop due to invalid source` | **32** | 0 |
| `Robot to approach ...` | 197 | 180 |

Those 32 entries are full stops: the monitor holds the robot at zero velocity
while an enabled source has no fresh data, which is the correct behaviour. They
cluster at t+24..36, t+89..90, t+100..101, t+108, t+124..131 relative to the
goal, and the first "Failed to make progress" lands at t+31, inside the first
cluster. `scan_source_supervisor` cannot prevent them as configured: its
`scan_timeout_sec` is 0.6 s while the monitor's `source_timeout` is 0.45 s, so
the monitor always notices first — the opposite of what the parameter's own
comment says it is for. Reversing that is not obviously right either, because
several of these outages are seconds long and braking is the correct response
to a 5.5 s blackout. The fix is the LiDAR, not the timeout.

`journalctl -k` on egg8 covers this window (the journal is persistent back to
2026-06-06) and records no USB, cp210x or xhci event at all during the run, so
the link never dropped at the kernel level. The host was heavily loaded at the
time — the MPPI control loop fell to 9.68 Hz against 20 Hz nominal, and the
planner to 1.15 Hz against 5 Hz — and the runs that showed no dropouts held
13-22 Hz. That correlation is suggestive, not established.

Two further things in these logs are unexplained and not addressed here. The
collision monitor's `FootprintApproach` is engaged almost continuously on
hardware (180 transitions in the 16:07 run, 197 at 20:21) and never once in the
synthetic demo, so `min_points: 1` against real scan noise deserves its own
measurement. And in the 12:40 run the pose reported by `bt_navigator` jumped
between (-2.50, 3.96), (3.06, -0.07), (-6.77, -1.71) and (1.61, 7.48) within
seconds, which is a localization failure independent of the controller.

bacon6's journal was not persistent, so the gateway's own record of the
2026-08-06 runs is gone for good. The cause was
`/usr/lib/systemd/journald.conf.d/40-rpi-volatile-storage.conf`, a Raspberry Pi
OS default that sets `Storage=volatile` to spare the SD card;
`/var/log/journal` existed but was never written to. Fixed with
`/etc/systemd/journald.conf.d/50-persistent-storage.conf` (`Storage=persistent`,
`SystemMaxUse=200M`, `SystemMaxFileSize=20M`) — capped so the card is neither
filled nor worn out by it. Verified by rebooting bacon6: `journalctl
--list-boots` now lists the previous boot, and the 17:33 run's gateway status
lines are queryable from it. egg8's journal was already persistent back to
2026-06-06 and is untouched.

A third gap is in the tracker wiring itself and predates this entry, but now
that tracker mode is the default it is on the operating path. `behavior_server`
publishes recovery motion to `cmd_vel_nav` unconditionally, and in tracker mode
`velocity_smoother` is remapped away to `cmd_vel_mppi_idle`, so nothing
subscribes to `cmd_vel_nav`: Spin, BackUp and DriveOnHeading command a topic
with no consumer. A robot that does get stuck therefore cannot recover — the
progress checker fires, the BT runs a behavior, the behavior moves nothing, and
the goal aborts. Routing it is not a one-line remap, because the tracker
publishes `/cmd_vel_nav_smoothed` at 100 Hz continuously (zeros while
`PLAN_STALE`) and would fight a second publisher on the same topic; the tracker
needs to relay the behavior's twist while a behavior is active. That is a change
to the command chain and should be measured, not assumed.

## 2026-08-07 - the weave during translation: the planner's grid staircase reached the feedforward

The base oscillated in yaw and swung left and right while translating. The yaw
*reference* was not the cause: on a straight leg the planned yaw profile is
flat, and the yaw loop has a wide phase margin against the 120 ms + 80 ms
drivebase. The excitation came from the translation feedforward.

`SmacPlanner2D` returns cell centres on a 0.05 m grid. `resample_spacing_m` is
also 0.05 m, so resampling preserved the staircase exactly, and
`path_tangents` central-differenced it. Measured on the deployed legs:

| leg | heading swing | direction reversals | commanded lateral at 0.7 m/s |
| --- | --- | --- | --- |
| 4 -> 5 (parallel to the grid axis) | 0.0 deg | 0 | 0.00 m/s |
| 3 -> 6 diagonal | 45.0 deg | 31 | 0.53 m/s |
| 2 -> 4 diagonal | 45.0 deg | 44 | 0.54 m/s |
| 1 -> 4 diagonal | 45.1 deg | 40 | 0.53 m/s |

**Every leg except 4 <-> 5 was commanding a half-metre-per-second lateral
square wave.** The 100 Hz acceleration limiter passes 0.0085 m/s per tick, so
it could not follow: it sat saturated for 99.9 % of ticks on a turning leg and
low-passed the square wave into a weave. A saturated rate limiter inside a
200 ms delayed loop is a limit cycle, and it behaved like one — the ring
frequency stayed at 1.29 Hz while its amplitude scaled with disturbance.

**This is why the synthetic demo never showed it.** Every tracker number in the
2026-08-06 entry above came from the goal 4 <-> 5 shuttle, the one leg that is
parallel to the grid axis and therefore the one leg with no staircase.

### Fixed

- **`smooth_path` before the tangents.** Endpoint-pinned `[1,2,1]/4` passes, so
  the start (robot position) and the end (the goal the tracker snaps to) are
  exact. Each point may move at most half a cell: the staircase amplitude is
  bounded by half a cell, so a larger correction is not removing quantisation,
  it is rewriting the path. Without that bound a 90-degree wall corner is cut
  by 49 mm, and goals 4-7 only have 57-59 mm of footprint clearance.
- **The reference is read at `t + feedback_delay`.** The state was extrapolated
  forward but the reference was still sampled at the current time, so the base
  looked `v * lag` too far along and the P term braked for the whole leg.
- **The measurement-wheel twist is low-passed before use.** `/wheel/odometry`
  publishes a raw one-sample difference (`twist = body_delta / dt`), and the
  tracker multiplies it by 0.48 (position prediction), 0.25 (damping) and 0.52
  (yaw prediction). Encoder differentiation noise was reaching the command at
  roughly unit gain. The synthetic demo had exact odometry, so this path had
  never been excited.

Replaying the same legs against the same plant model:

| | before | after |
| --- | --- | --- |
| cross-track, turning leg | 50.2 mm | 13.0 mm |
| along-track lag | 32.4 mm | 4.6 mm |
| accel limiter saturated | 99.9 % of ticks | 10.6 % |
| feedforward heading swing, diagonal | 45.0 deg | 4.5-10.7 deg |

`scripts/yaw_oscillation_probe.py` reproduces both columns and carries the
ablations, including the two hypotheses that were investigated and rejected
(carrying the yaw ramp across replans, and rate-limiting in the world frame).
104 package tests pass.

### Not established

None of this has run on the real drivebase yet — it is measured against the
same 120 ms / 80 ms model as the 2026-08-06 entry. What the model cannot tell
us is the real measurement-wheel noise level, which sets how much the velocity
filter is worth; the sweep in the probe brackets it.

## 2026-08-06 - accurate arrival at goals 4 and 5: latency compensation and a corrected measurement method

Measured on egg8 in the synthetic demo with a drivebase model of 120 ms dead
time and an 80 ms velocity time constant, host left idle, four legs of the
goal 4 <-> 5 round trip. Settled error is read 2.5 s after Nav2 declares
success; peak is the worst value in that window, which is the number that
decides whether the footprint touches the fixture.

| leg | MPPI time | MPPI peak | MPPI settled | MPPI yaw | tracker time | tracker peak | tracker settled | tracker yaw |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| -> 5 | 16.1 s | 29 mm | 22 mm | 0.50 deg | 16.4 s | 33 mm | **1 mm** | **0.00 deg** |
| -> 4 | 22.1 s | 39 mm | 30 mm | 0.02 deg | 21.7 s | 33 mm | **1 mm** | **0.01 deg** |
| -> 5 | 15.9 s | 28 mm | 19 mm | 0.57 deg | 14.8 s | 33 mm | **1 mm** | **0.00 deg** |
| -> 4 | 20.2 s | 38 mm | 30 mm | 0.00 deg | 21.8 s | 33 mm | **1 mm** | **0.00 deg** |

**Time is the same within run-to-run spread. Settled position error improves
20-30x, from 19-30 mm to 1 mm, and yaw settles to 0.00-0.01 deg every time.**
That matters here specifically: goals 4 and 5 have 57 and 59 mm of footprint
clearance at their commanded firing yaw, so a 30 mm resting error is half the
margin, and the yaw component of the arrival tolerance moves the outermost
vertex by another 20 mm at 2 degrees.

### What made the difference

**Latency compensation.** The velocity profile brings the *command* to zero
exactly at the goal, but the drivebase has dead time and lag, so the machine is
still moving when the command reaches zero. Traced at 0.25 s intervals through
an arrival at goal 5, before the fix:

    +0.25 s   60 mm   cmd_vel_safe +0.037,-0.167   still driving past the goal
    +0.75 s  130 mm   cmd_vel_safe -0.016,+0.111   peak overshoot, turning back
    +2.00 s    0 mm
    +4.00 s    0 mm   settled at 0.3 mm

A 130 mm excursion past a pose with 59 mm of clearance is a contact. The
tracker now takes its error against the pose the robot will be in after
`feedback_delay_sec` (0.20 s), extrapolated from the measurement-wheel velocity
rotated into the map frame. The same trace afterwards decays monotonically -
27, 19, 14, 9, 7, 5, 4, 2, 1, 0 mm - with no overshoot at all. The terminal
low-speed approach was also folded into the velocity profile rather than
applied as a cap in the controller, so there is no 0.553 -> 0.16 m/s step for
the acceleration limiter to work off.

**The trajectory was converging to the plan end, not to the goal.**
`SmacPlanner2D` runs with `tolerance: 0.20`, so when the goal cell is occupied -
the normal case at goals 4-7 - the returned path stops short. MPPI does not
care because `GoalCritic` scores against the real goal. Measured 66-77 mm of
terminal error before the fix. `goal_bridge` now publishes the executing goal
on `/navigation/active_goal` and the tracker snaps its endpoint and terminal
yaw to it; the status now reports `endpoint`, `goal`, `goal_offset` and
`snapped` so this cannot regress unnoticed.

### Two defects in the launch, both found by trying to measure

`navigation_delay` was 18.0 s against a CAD wall-lookup build that measurably
takes about 15 s. Under any load the lifecycle manager logged
`bt_navigator/get_state service client: async_send_request failed` and then
`Failed to bring up all requested nodes. Aborting bringup.`, leaving
`collision_monitor` and `velocity_smoother` inactive. Every goal then queued
forever. Raised to 26.0 s. The test now pins the *invariant* (comfortably above
the measured build time) rather than the literal number.

This failure was only diagnosable because the goal_bridge fix from earlier
today names the blocking nodes. Without it the panel simply shows a goal that
never starts.

### The measurement method itself was wrong, and earlier timings in this file are not trustworthy

Every teardown in this session's ad-hoc scripts used
`pkill -f "install/omni_autonomy_next/lib"` and
`pkill -f "/opt/ros/jazzy/lib/nav2"`. Neither matches
`/opt/ros/jazzy/lib/tf2_ros/static_transform_publisher`, so **every launch left
two orphaned static TF publishers behind**. After eight runs there were sixteen
of them, each burning about 2% of a core and all publishing the same
`base_link -> lidar_*` transforms into one TF tree. Host load average reached
**20-22 on six cores**.

Consequences, stated plainly:

- The A/B timings taken during that period swung by a factor of two for an
  identical configuration - 14.4 s and 29.5 s for the same code - and led to two
  wrong conclusions that were then reverted: that raising the lateral speed
  limit slowed the robot, and that seeding the velocity profile from the last
  command slowed it. Neither is established. Both were measured on an
  overloaded host.
- The 100-145 mm "terminal error" reported earlier was an artefact of sampling
  the overshoot 1.0 s after arrival, which is close to its peak. The tracker
  was settling to 0.3 mm the whole time.
- `config/nav2_next.yaml` already carried a warning about exactly this trap
  for `batch_size`: "the host was not idle during those runs". The same mistake
  was repeated.

Teardown in the measurement scripts now kills `/opt/ros/jazzy/lib/` as a whole.
The findings that survive are the ones that do not depend on wall-clock timing:
the arrival trace and its mechanism, the settled-accuracy comparison above
(1 mm against 19-30 mm appeared in every tracker run at every load), the omni
diamond envelope, and the `sprint` profile being outside it.

### Status

`tracker:=true` is still not the default. What is now demonstrated is accurate
arrival - 1 mm and 0.00 deg, repeatably, with no overshoot - at the same time
cost as MPPI. What is not demonstrated is any speed improvement, and nothing has
run on the real drivebase. 72 package tests and 21 simulation tests pass.

## 2026-08-06 - omni envelope measured, one real profile defect fixed, yaw optimisation does not pay yet

The request was to exploit the omni wheels for a faster and more accurate
arrival at and departure from goals 4 and 5. This entry records what the
measurements actually support, including two ideas that were built, measured and
then *not* carried.

### The exact translation envelope is a diamond, not an ellipse

For a 45-degree omni the wheel command is `|d_i . v| / r` with the roller
directions at +-45 degrees, so the peak over the four wheels is
`(|vx| + |vy|) / (sqrt(2) r)`. The feasible set is therefore

    |vx| + |vy| <= sqrt(2) * r * max_wheel_speed = 1.111 m/s

which reaches **1.111 m/s along either body axis and only 0.785 m/s at
45 degrees**. `scripts/performance_envelope.py` prints this, and
`omni_autonomy_next/omni_yaw.py` provides `OmniEnvelope.speed_limit`, which
returns the exact limit for a travel direction *and* a yaw rate, because both
consume the same budget linearly.

The consequence for any profile: its translation envelope is an ellipse with
semi-axes (linear, lateral), and the largest `|vx| + |vy|` on that ellipse is
`sqrt(linear^2 + lateral^2)`. That has to stay inside 1.111.

### Defect found: the `sprint` profile was outside the envelope

`sprint` was 0.95 / 0.80, giving `sqrt(0.95^2 + 0.80^2) = 1.242` - **12%
outside the diamond**. A diagonal sprint command would have been scaled down by
RuntimeGuard with the controller never knowing, which is exactly the hidden
governor `docs/ARCHITECTURE.md` forbids. Corrected to 0.95 / 0.57
(`sqrt = 1.108`), which is what sprint is useful as: a fast *forward* profile.
`balanced` (1.049) and `precision` (0.474) were already feasible.
`test_translation_envelope_fits_the_omni_diamond` now pins this for every
profile.

### Built and measured: yaw as a free variable. It does not pay yet.

Body yaw is independent of travel direction on a holonomic base, so it can be
chosen to (a) align travel with a body axis, worth up to 1.41x on the
drivetrain envelope, and (b) widen footprint clearance, which is yaw-dependent
and only 57/59 mm at the commanded firing yaw of goals 4/5.
`omni_yaw.plan_yaw_profile` solves this as a dynamic program over arc length:
states are yaw candidates per sample, stage cost is
`-(direction speed + weight * footprint clearance)` with impassable yaws
rejected outright, transition cost bounds the yaw step by what the yaw rate can
deliver, and the ends are pinned to the current and goal yaw. Candidates are
generated from the travel heading (`heading - k*90deg`, the exactly-fast
alignments) rather than from a fixed grid, because a 7.5 degree misalignment
already costs 11% of the speed.

Measured, and the answer is negative at the deployed profile:

- 1 -> 4 diagonal: mean speed limit **0.707 m/s with yaw fixed against
  0.687 m/s with yaw optimised**. Slightly worse.
- departure north out of goal 4: **0.702 m/s and 57 mm clearance either way**;
  the optimiser declines to rotate at all.

The reason is that the 1.41x exists on the *drivetrain* diamond, not inside the
deployed `balanced` ellipse. Within 0.78 / 0.702 the spread between the best
and worst direction is only about 11%, and rotating to collect it costs wheel
budget and time. The DP is behaving correctly by refusing. It is therefore
**disabled by default** (`optimize_yaw:=false`) and becomes worth enabling only
if the profile is raised toward the diamond.

The clearance half is also bounded by geometry rather than by control: the
straight x = -0.81 line from goal 4 to goal 5 measures **0.000 m footprint
clearance at every yaw**, which is the already-documented fixed bucket 1 sitting
in the firing lane with a 0.40 m gap against a 0.90 m robot. No yaw passes
there; the planner has to detour, and it does.

### Tried and reverted: making the lateral limit symmetric

0.702 lateral has no physical basis, and the 4 <-> 5 transit is a pure lateral
move at the commanded firing yaw, so raising it to 0.78 looked like a direct
11% win. It is feasible. It produced no gain. Measured 4 <-> 5 round trip with
MPPI: **15.8 / 20.2 / 15.5 / 20.3 s at 0.702 against 16.3 / 21.4 / 16.3 /
22.2 s at 0.78**. At 0.78 the 45-degree worst case consumes 99.3% of the wheel
budget, so nothing is left for yaw and the guard's uniform scaling slows the
whole twist instead. The difference is inside run-to-run spread in both
directions, so the speed increase is not carried. The finding and the exact
one-line change are recorded in `config/runtime.yaml`.

### Fixed in the tracker: it was converging to the plan end, not to the goal

`SmacPlanner2D` runs with `tolerance: 0.20`, so when the goal cell is occupied -
which is the normal case at goals 4-7 with their 47-59 mm clearance - the
returned path ends short of the goal. MPPI does not care because `GoalCritic`
scores against the real goal; a controller that only reads `/plan` stops
wherever the path stops. Measured: **66-77 mm of terminal error**.

`goal_bridge` now publishes the executing goal pose on
`/navigation/active_goal` (transient local), and the tracker snaps its
trajectory endpoint and terminal yaw to it when the plan ends within
`goal_snap_distance_m` (0.40 m).

A terminal low-speed servo was also added. The velocity profile brings the
*command* to zero exactly at the goal, but the drivebase has 120 ms of dead
time and an 80 ms lag, so the machine arrives faster than commanded and
overshoots. Inside `terminal_approach_m` (0.18 m) the feed-forward is dropped
and the residual is closed by pure P capped at `terminal_speed` (0.14 m/s).

Trajectory building also moved to a worker thread. The yaw DP costs 37-59 ms
for a 3.6 m path, and running that in the plan callback stalled the 100 Hz
control timer for that long. The old trajectory keeps being tracked while the
new one is built.

### Still not ready: the tracker leaves terminal error on the leg into goal 5

Measured 4 <-> 5 round trip, same drivebase model (120 ms dead time, 80 ms
lag):

| leg | MPPI time | MPPI error | tracker time | tracker error |
| --- | --- | --- | --- | --- |
| -> 5 | 16.3 s | 22 mm | 15.9 s | **103 mm** |
| -> 4 | 21.4 s | 27 mm | timeout | - |
| -> 5 | 16.3 s | 19 mm | 15.6 s | **100 mm** |
| -> 4 | 22.2 s | 28 mm | 26.9 s | 4 mm |

Yaw error is far better with the tracker (0.01-0.02 deg against 0.18-0.71 deg)
and the legs into goal 5 are marginally faster, but a repeatable ~100 mm
position error on that leg and one timeout are not acceptable. Note that Nav2
declares success only inside 40 mm, so the robot is *moving away after arrival*:
the error is measured 1 s later. The remaining suspects, in order, are the
terminal servo continuing to drive toward a trajectory endpoint that the goal
snap did not reach on that leg, and Collision Monitor's `FootprintApproach`
blocking the last push into a pose that has 59 mm of clearance.

The tracker therefore stays `tracker:=false` by default. The earlier 1 -> 4 /
4 -> 5 / 5 -> 2 route still measures 8.8 s against MPPI's 14.4 s on the first
leg, so the approach is sound; the terminal behaviour into the tight firing
poses is what is unfinished.

72 package tests and 21 simulation tests pass.

## 2026-08-06 - feed-forward trajectory tracking added as an alternative to MPPI

The question was whether this system can reach the speed and accuracy that the
strongest NHK Robocon teams achieve. It cannot, and the reason is measurable
rather than a matter of tuning. `scripts/performance_envelope.py` computes the
ceiling from the deployed configuration:

- Maximum forward speed is **1.111 m/s**, and only **0.785 m/s** along the
  45-degree diagonal, which is the worst direction for this wheel layout. The
  number comes from the `omni_max = 8000` command ceiling in `value.hpp`
  divided by the measured 7202 units per m/s. Top teams run 2-4 m/s.
- A 3.55 m leg therefore cannot be driven faster than 4.50 s even with a
  perfect controller; at 2.5 m/s it would take 2.25 s.
- The LiDAR correction is capped at 4 Hz x 40 mm, so it can only absorb pose
  error accumulating slower than **0.16 m/s**. The measurement-wheel
  calibration residual is 5.8% on y, which is 174 mm over 3 m of pure dead
  reckoning.
- At 20 Hz the base travels 39 mm per control cycle and 78 mm inside the
  measured 100 ms command latency.

The architectural gap is that the strong teams run a *time-parameterized*
trajectory with feed-forward velocity at 500 Hz - 1 kHz on an MCU, correcting
only the residual, with heading from a 1 kHz gyro. This system solves a
sampling MPC at 20 Hz on Linux against a 4 Hz ICP. That is a factor of 25-50 in
loop rate, and no critic weight changes it.

What *is* addressable in software is the first half of that: plan the motion in
time and feed it forward, so the latency does not appear as a tracking error on
the nominal motion.

### trajectory_tracker

`omni_autonomy_next/trajectory_tracker_node.py` takes the Smac path from
`/plan`, resamples it to 50 mm, limits speed by curvature against a lateral
acceleration bound, runs the standard forward/backward sweeps for the
longitudinal acceleration bound, and integrates to a time-indexed reference.
At 100 Hz it samples that reference and outputs

    v_command = v_reference(t) + Kp * position_error + Kd * d(error)/dt
    w_command = w_reference(t) + Kyaw * yaw_error

Goal yaw is distributed along arc length, so rotation retires together with
translation instead of after arrival, matching the intent already documented
for `GoalAngleCritic`.

Two details matter for robustness. The reference time is governed: it may not
run more than `max_reference_lead_m` (0.35 m) ahead of the robot's projection
onto the path, so a Collision Monitor stop cannot let the reference walk away
and produce a large command on release. And a new plan is time-parameterized
from the *current measured speed*, so the frequent replans do not each force a
deceleration.

Acceleration limiting is applied to the tracker's own output. The feed-forward
term satisfies the bound by construction, but the error terms and the step at a
plan change do not; leaving that to RuntimeGuard would put the shaping back
into the gate and break the trajectory's timing.

**Nothing gains stopping authority and nothing loses it.** The tracker publishes
on `/cmd_vel_nav_smoothed`, the same topic `velocity_smoother` uses, so
`rl_policy -> collision_monitor -> runtime_guard` is unchanged. `controller_server`
still owns the `NavigateToPose` lifecycle, the goal checker, the progress
checker and recovery; only its MPPI search is cut to a token size and its
command output is dead-ended, so it does not spend the Jetson's cores
optimising a command nobody consumes. If the tracker dies, the command stream
stops and RuntimeGuard reports `STALE_COMMAND` and zeroes, as it already does
for any other producer stall.

Enable with `tracker:=true`; the default is `false`, so an unmodified launch
still runs MPPI.

### Measured, same route and same drivebase model (120 ms dead time, 80 ms tau)

| | MPPI | trajectory_tracker |
| --- | --- | --- |
| leg 1 -> 4 (3.55 m) | 14.4 s | **8.8 s** |
| leg 4 -> 5 (3.65 m) | 15.8 s | **12.9 s** |
| leg 5 -> 2 | not reached in the window | 15.6 s |
| goals reached in 56 s | 2 | **3** |
| cross-track RMS, settled | 0.015 m | 0.019 m |
| cross-track p95, settled | 0.029 m | 0.045 m |

The 1 -> 4 leg is 39% faster. The theoretical floor for that leg at the
`balanced` profile is 5.47 s, so MPPI ran at 2.6x the floor and the tracker at
1.6x; the rest is the 0.70 red-zone speed scale near the goals and the path not
being straight. Cross-track is comparable, and the comparison favours MPPI
slightly here only because it was slower and had longer to settle - an earlier
run at identical settings measured 0.011 m RMS for the tracker against 0.034 m
for MPPI. Run-to-run spread on this metric is larger than the difference, so
the honest claim is *comparable tracking at 39% less time*, not better
tracking.

69 package tests and 21 simulation tests pass.

### Not validated on hardware

The tracker has never driven the real drivebase. It is measured only in the
synthetic demo with a modelled 120 ms dead time and 80 ms velocity lag. Before
using it on the field, run `check-drive-directions`, then a first goal at 10%
speed scale with the wheels raised, and compare `run.py diagnose-chain` against
the numbers above.

### What would actually close the remaining gap

1. **The 1.111 m/s ceiling.** Establish what `omni_max = 8000` means to the
   RoboMaster Dev Board: whether it is the board's own maximum speed command or
   a chosen limit. If there is headroom, the drivetrain, not the controller, is
   what unlocks the time. This has to be measured, and traction and braking
   distance re-validated with it.
2. **Heading.** A 1 kHz gyro fused into the odometry would remove the dominant
   dead-reckoning error and stop the 4 Hz ICP from being the accuracy floor.
3. **Loop rate at the actuator.** The final 100 ms is Linux plus the 100 Hz
   UDP/200 Hz Pi/115200 baud UART chain. Closing a fast loop on the Dev Board
   itself is the only way past it.

## 2026-08-06 - auto_turn_sign settled by derivation from working manual control

The operator confirmed that manual MU-3 driving is correct on every axis, and
answered the two questions that pin the mixer's physical contract:

- pushing the right stick **right** turns the base **clockwise** seen from above
- pushing the left stick **right** strafes the base to its **right**

That is enough to derive the auto signs rather than guess them, because the
manual path and the v3 auto path share the same mixer. `checker_omni()` feeds
`turn = rx` straight into the mixer's yaw input and `move_x = lx` into its
lateral input, with no inversion. So the measured contract is:

    mixer translation input > 0  ->  forward   (ROS +x)
    mixer lateral input     > 0  ->  right     (ROS -y)
    mixer yaw input         > 0  ->  clockwise (ROS -wz)

Applying it to `OMNI::mix_velocity`:

| axis | needed for the ROS convention | constant | was | now |
| --- | --- | --- | --- | --- |
| forward | ROS +x -> forward, both positive | `auto_forward_sign` | +1.0 | +1.0 (unchanged) |
| lateral | ROS +y is left, mixer positive is right | `auto_lateral_sign` | -1.0 | -1.0 (unchanged) |
| yaw | ROS +wz is CCW, mixer positive is CW | `auto_turn_sign` | **+1.0** | **-1.0** |

**`auto_turn_sign` was wrong.** Autonomous driving has been commanding
rotation in the opposite direction to the operator's own verified controller
since it was changed to `+1.0` on 2026-08-05. That entry read a
constant-radius circle as a yaw runaway; the circle is better explained by the
wheel-budget allocator, which scaled translation and yaw by different factors
and distorted the commanded curvature by 45% (fixed in the entry above),
together with the 440 ms of command-path latency measured the same day.
Inferring a sign from the *shape of a symptom* is what produced four wrong
rounds; this one is measured from a static contract instead.

A consistent physical layout exists and is not contradictory - an earlier claim
during this session that the two dashboard labels could not both hold was
wrong. Taking all four motor polarities positive, the wiring is

    m1 = front-right (+0.333072, -0.333072), roller 225 deg
    m2 = rear-right  (-0.333072, -0.333072), roller 135 deg
    m3 = rear-left   (-0.333072, +0.333072), roller  45 deg
    m4 = front-left  (+0.333072, +0.333072), roller -45 deg

Every roller axis is tangential, a pure translation drives two wheels each way
and a pure yaw drives all four the same way, exactly as the mixer requires.
`config/robot.yaml` now carries this order instead of the previous
`[front_left, front_right, rear_left, rear_right]` guess. The Jetson uses that
model only for the wheel-speed budget, whose magnitudes are invariant under a
permutation, and the numbers are unchanged: a `balanced` request of
(0.78, 0, 1.30) still allocates (0.5264, 0, 0.8773). It is corrected because a
model that disagrees with the hardware is a trap for the next sign
investigation, not because the arithmetic moved.

Note that this layout is *a* consistent solution, not a proven one: an inverted
motor polarity would admit another. The three signs, however, follow from the
measured contract alone and do not depend on which layout is real.

### Also fixed: the legacy/v2 auto path disagreed with v3 on the lateral axis

`main.cpp` passed the Jetson's `lx` through unmodified while inverting `rx`.
The Jetson sends `lx` with the sign of ROS `vy` (left positive), so v2 executed
`lateral input = +vy`, i.e. it strafed right when commanded left - the opposite
of the v3 contract, and wrong against the manual contract too. v2's `rx`
inversion was already correct. `lx` is now inverted as well, so the manual, v2
and v3 paths finally agree on all three axes. v2 is not used for autonomous
driving, but two disagreeing paths in one binary is a defect on its own.

### Verified on the wire after the change

`scripts/e2e_chain_check.py` against a `--jetson-only` gateway on port 8889
(UART never opened, real Dev Board unreachable), 0 CRC errors, 0 stale drops,
0% loss, 0.21 ms mean RTT, all nine cases matching:

| command | wheel values m1..m4 |
| --- | --- |
| +wz 1.00 rad/s (CCW / left) | -4798, -4798, -4798, -4798 |
| -wz 1.00 rad/s (CW / right) | 4798, 4798, 4798, 4798 |
| +vx 0.50 m/s | -3601, -3601, 3601, 3601 |
| +vy 0.50 m/s (left) | -3601, 3601, 3601, -3601 |
| 0.78, 0, 1.30 (saturating) | -8000, -8000, -418, -418 (scale 0.675) |

The check now reads `auto_*_sign`, the gain and the wheel limit out of
`value.hpp` rather than restating them, so it cannot pass against a stale copy
of the constants. Both gateway ctest cases pass, 60 package tests and 21
simulation tests pass, and `omni-gateway-next.service` was restarted on the
rebuilt binary.

**Confirm before the first autonomous run.** This is a derivation from a
measured contract, which is much stronger than the symptom-shape reasoning it
replaces, but it has not been executed on the drivebase. Run

    python3 run.py check-drive-directions --accept-motor-risk

after the wheels-raised tests. All three axes must report OK. That closes
`ACCEPTANCE.md` step 3.

## 2026-08-06 - end-to-end command path verified numerically, egg8 to the UART wire

`/cmd_vel_safe` -> UDP v3 -> bacon6 decode -> `OMNI::mix_velocity` -> the bytes
written to the UART was measured end to end and matched at every step. Two new
tools do it, and neither can move a motor.

**`scripts/e2e_chain_check.py`** drives known twists and reads back what the Pi
reports it applied. It runs against a second gateway instance started with
`--jetson-only` on `MU3_JETSON_PORT=8889`, which never opens `/dev/serial0`, so
the real Dev Board is physically unreachable while the whole Ethernet, decode
and mixer path is exercised for real. Measured: `proto=v3`, `auto_engaged`,
162 Hz receive rate, 0 CRC errors, 0 stale drops, 0% loss, 0.19 ms mean RTT.
Nine cases, all matching:

| command (vx, vy, wz) | wheel values m1..m4 | scale |
| --- | --- | --- |
| 0.50, 0, 0    | -3601, -3601, 3601, 3601 | 1.000 |
| -0.50, 0, 0   | 3601, 3601, -3601, -3601 | 1.000 |
| 0, 0.50, 0    | -3601, 3601, 3601, -3601 | 1.000 |
| 0, -0.50, 0   | 3601, -3601, -3601, 3601 | 1.000 |
| 0, 0, 1.00    | 4798, 4798, 4798, 4798 | 1.000 |
| 0, 0, -1.00   | -4798, -4798, -4798, -4798 | 1.000 |
| 0.35, 0.35, 0 | -5041, 0, 5041, 0 | 1.000 |
| 0.60, 0, 0.60 | -1443, -1443, 7200, 7200 | 1.000 |
| 0.78, 0, 1.30 | 418, 418, 8000, 8000 | 0.675 |

Each row is checked four independent ways: against the mixer formula, by
inverting the four wheel values back to a twist, against the wheel speeds
computed from the CAD geometry (wheel corners and roller angles) rather than
from the mixer, and against the 8000-unit ceiling. The saturating last row
scales the *whole* twist by 0.675, so the executed curvature equals the
commanded curvature - the property the 2026-08-06 allocator change was made
for, now confirmed on the wire rather than only in the Jetson.

**`scripts/uart_wire_check.py`** opens a pty, has the real `UART` class write to
it, and decodes the raw bytes. Confirmed: 27 payload bytes as nine
`[command id, high byte, low byte]` triples with ids `0,1,2,3,44,61,6,7,254`,
big-endian int16, COBS-encoded to 29 bytes with exactly one zero at the end and
no zero left in the body. Verified for forward, lateral, rotation, the
saturating case and the all-zero stop frame.

The gateway now also reports `uart_tx` and `uart_skip` on its status line.
Measured over 730,856 frames at its 200 Hz loop: **zero skipped frames**. The
115200 baud link carries the 29-byte frame in 2.5 ms against a 5 ms period, so
it has the headroom, and a skip is now visible instead of inferred.

### What this does and does not establish

It establishes that a twist leaving the Jetson arrives at the UART as the
correct wheel values for the drivetrain geometry in `config/robot.yaml`.

It does **not** establish which physical corner receives `m1`. That is a wiring
fact recorded nowhere, and `auto_lateral_sign` -1.0 and +1.0 are both
geometrically valid solutions differing only in that assignment. No amount of
software reasoning settles it.

`scripts/check_drive_directions.py` (`python3 run.py check-drive-directions
--accept-motor-risk`) closes that gap in about three minutes. It starts only
`measurement_wheel` and `motor_udp_bridge` - no Nav2, no LiDAR - commands
+x, +y and +yaw one at a time at 0.12 m/s / 0.35 rad/s for 1.2 s each, and
reads the resulting `/odom` increment. The measurement wheels are the right
reference because they were independently confirmed correct in sign and scale
on 2026-08-05. A negative increment on an axis names the exact constant to
invert in `bacon_gateway/include/value.hpp`. It also warns when a push shows up
strongly on an axis it was not applied to, which is the signature of an m1..m4
wiring permutation rather than a single sign.

The tool carries the same `--accept-motor-risk` interlock as `full` and
`gateway`. Run the wheels-raised tests in `ACCEPTANCE.md` first.

## 2026-08-06 - weave traced by measurement to command-path latency and a shape-changing wheel allocator

Reported symptom, unchanged by every sign flip so far: part-way through a run
the base weaves left and right while going straight, then moves the wrong way
at the end.

The previous rounds each changed one sign and re-ran. This round measured the
chain instead. `scripts/diagnose_chain.py` was added: it records `/cmd_vel_nav`,
`/cmd_vel_nav_smoothed`, `/cmd_vel_rl`, `/cmd_vel_collision_safe`,
`/cmd_vel_safe`, `/odom`, `/localization/pose` and `/plan` together and reports
per-stage rate, per-stage sign-reversal rate, per-stage phase lag by
cross-correlation, delivered acceleration, and cross-track error against the
plan that was active at each sample. Run it during a goal:
`python3 run.py diagnose-chain --seconds 45`.

**Measured on egg8, synthetic demo, before any change:**

- `/cmd_vel_safe` lagged `/cmd_vel_nav` by **140 ms**, of which RuntimeGuard
  alone contributed **90 ms**.
- The chain delivered **0.373 m/s^2** against the 0.85 m/s^2 configured
  everywhere in the stack: **44%**.
- `/cmd_vel_nav` was already reversing `wz` 0.29 times per second.
- The Ethernet link is not involved: 300 packets egg8 -> bacon6 at 5 ms
  intervals gave 0% loss, 0.082 ms mean and 0.349 ms maximum RTT, with zero
  interface errors. The gateway's own UART is not involved either: it reports
  `uart_tx=201/s uart_skip=0` at its 200 Hz loop.

Four independent defects produced that, and none of them is a sign.

### 1. The wheel-budget allocator changed the shape of the commanded twist

`allocate_omni4_wheel_budget` scaled translation and yaw by *different*
factors. A `balanced` request of vx 0.78 m/s with wz 1.30 rad/s needs
11.03 + 12.25 = 23.28 of the 15.709 rad/s wheel limit, and the split
allocation executed (0.500, 0.917) - a **45% curvature error** against the
trajectory MPPI had just scored. The base therefore left the planned path on
its own, the controller corrected, the allocator distorted the correction in
turn, and the loop hunted. It also disagreed with the deployed Pi mixer, which
already scales all four wheel commands by one common factor.

It now scales the whole twist by one factor. Direction of travel, the
translation-to-yaw ratio and therefore the instantaneous curvature are
preserved exactly; a uniform slowdown is a pure time rescaling that MPPI
absorbs by replanning from the measured state. The same request now executes
(0.527, 0.877), which is **faster** than the split allocation as well as
correctly shaped, and far better than the yaw-absolute-priority behaviour
before it, which left 0.245 m/s. `translation_budget_share` is retained for
configuration compatibility and validated, but a uniform scale needs no
reservation.

### 2. RuntimeGuard was a second shaping filter, not a gate

The guard carried the same acceleration numbers as `velocity_smoother`
(0.85 / 2.00) plus a jerk limit of 2.8 m/s^3. Reaching 0.85 m/s^2 through that
jerk limit costs 0.30 s, so the guard re-limited an already-limited command on
every tick and its jerk term became an unmodelled second-order lag inside the
20 Hz control loop. MPPI integrates its rollouts with acceleration bounds only,
so its predicted response was faster than the chain could produce - which is
the textbook way to under-damp a cross-track loop.

The guard profiles now hold acceleration and jerk strictly **above**
`velocity_smoother`, and `hard_max_*_jerk` was raised to match. The required
ordering, now pinned by
`test_guard_dynamics_are_a_gate_around_the_shaping_stage_not_a_second_one`:

    MPPI = velocity_smoother < RuntimeGuard profile <= hard_max_*

The guard still clamps unconditionally and still zeroes immediately on any
health failure. It simply stops filtering traffic that is already inside its
envelope.

### 3. velocity_smoother ran CLOSED_LOOP in a chain where its output is not executed

`CLOSED_LOOP` recomputes each acceleration step from the speed measured on
`/wheel/odometry`, which assumes this node's output is what the base executes.
`rl_policy`, `collision_monitor` and `runtime_guard` all sit downstream. The
ramp therefore chased RuntimeGuard's output: the target stayed about one
acceleration step above measured, the guard reached it within a single
smoothing period, and the guard's overshoot clamp reset its acceleration state
on nearly every tick. Neither stage could sustain its nominal acceleration.

This is **independent of odometry calibration**, and the 2026-08-05 entry below
is wrong on two counts that are corrected here:

- A `calibration_matrix` *is* identified and deployed in `config/robot.yaml`.
  Checked numerically: `C @ A` is the identity to 1.3% (x), 5.8% (y) and 2.9%
  (yaw) against the corrected geometry.
- The measurement wheels were never dead. The 2026-08-05 log that reported
  "計測輪が無反応" contains, in the same `/wheel/status` payloads, a pose that
  advanced 0.487 m for a 0.5 m forward push, +0.482 m for a 0.5 m left push and
  +1.573 rad for a 90 degree turn. All three axes were correct in sign and
  scale. `scripts/check_odometry_signs.py` bracketed its measurement window
  with two `input()` calls, so the window closed before the operator moved the
  robot and every axis read zero. That tool, not the odometry, was at fault,
  and a wrong tool output sent a whole debugging round after a non-existent
  calibration problem.

`check_odometry_signs.py` now detects the start and end of motion from the raw
counts and the reference pose, needs no `Enter` at all, distinguishes "no
counts" from "counts but no odometry", and warns when a push shows up strongly
on an axis it was not applied to.

### 4. goal_bridge could deadlock and silently queue every goal forever

`_poll_lifecycle_states` issued `GetState` requests with `call_async` and only
freed the slot when the future completed. An unanswered rclpy service future
stays pending forever, and a Nav2 lifecycle node that is busy transitioning
drops the response - Nav2 logs `client will not receive response`. The slot for
that node was then never freed, it was never polled again,
`_navigation_ready()` stayed false, and every goal the operator selected sat in
the queue for the rest of the session with no error.

Reproduced on egg8 during this work: `velocity_smoother` timed out one
`get_state` during bringup and goals 4, 5 and 2 were all logged as
`Replacing queued goal` and never executed, for the full 56 s window. A
`lifecycle_request_timeout_sec` (default 1.0 s) now abandons and reissues the
request, and a throttled warning names the node that is blocking, because a
goal that silently stays queued is the hardest failure to diagnose from the
operator panel.

### Verification

`synthetic_scan_node` gained `command_delay_sec` and
`velocity_time_constant_sec` (both default 0.0, the ideal integrator it has
always been). They model what the real chain adds beyond the ROS graph: the UDP
hop, the Pi's 200 Hz loop, the 115200 baud UART frame and the Dev Board period
as dead time, and the C620 current loop, gearbox and wheel inertia as a
first-order velocity lag. Dead time is what destabilizes a cross-track loop, so
this turns the demo into a stability-margin measurement instead of a kinematic
replay. Exposed as `sim_command_delay` and `sim_velocity_tau` on
`system.launch.py`.

A/B over the same two-leg route (pose 1 -> 4, 4 -> 5, 5 -> 2) with a realistic
drivebase model of 120 ms dead time and an 80 ms velocity time constant:

| measurement                        | before   | after    |
| ---------------------------------- | -------- | -------- |
| speed \|v\| p95                     | 0.456 m/s | **0.765 m/s** |
| delivered acceleration \|a\| p95    | 0.312 m/s^2 | **1.089 m/s^2** |
| command -> measured lag (vy)        | 440 ms   | **250 ms** |
| RuntimeGuard's own lag (vy)         | 230 ms   | **80 ms** |
| `/cmd_vel_nav` wz reversals         | 0.30 /s  | **0.14 /s** |
| cross-track RMS, settled            | 0.051 m  | **0.027 m** |
| cross-track p95, settled            | 0.127 m  | **0.025 m** |

The settled figures exclude 2.0 s after each new goal's first plan, where the
robot is legitimately off a path it has only just been given. Cross-track p95
improved five-fold **while running 1.7x faster**, which is the combination a
latency fix produces and a gain reduction does not.

60 package tests and 21 simulation tests pass. `simulation/run_campaign.py`
now reads acceleration from `velocity_smoother` rather than from the guard
profile, since the guard's headroom is deliberately no longer the number the
chain commands.

### Still open

- `docs/ACCEPTANCE.md` step 3 (command `+x`, `+y`, `+yaw` separately with the
  wheels raised and confirm the physical directions) has still never been run.
  The lateral sign remains a wiring fact that no amount of software reasoning
  settles: `auto_lateral_sign = -1.0` and `+1.0` are both geometrically valid
  and differ only in which corner is C620 ID 1-4, which is recorded nowhere.
  Everything in this entry is orthogonal to that choice.
- No LiDAR or counter board was attached to egg8 during this work
  (`/dev/serial/by-path` is empty, `mu3_alive: false` on bacon6), so all of the
  above is measured on the real ROS graph with the synthetic drivebase, not on
  the field.
- The real robot's dead time and velocity time constant have not been measured.
  Run `python3 run.py diagnose-chain` during a field goal and compare the
  `/odom` lag against the 250 ms measured here; if it is much larger, the
  remaining margin is in the Pi/Dev Board path rather than in the ROS graph.

## 2026-08-05 - weave: uncalibrated odometry removed from the velocity loop

Reported symptom, unchanged by flipping the lateral sign in either direction:
part-way through a run the base weaves left and right while going straight, then
moves the wrong way at the end.

- The lateral sign is therefore not the cause. Both `+1.0` and `-1.0` produce the
  same weave, so the oscillation is generated upstream of the mixer boundary.
  `auto_lateral_sign` is left at its original `-1.0`.
- `velocity_smoother` ran with `feedback: CLOSED_LOOP`, which recomputes every
  acceleration step from the speed measured on `/wheel/odometry`. That odometry is
  still the uncalibrated geometric model: `count_signs` is the untested
  `[1, 1, 1, 1]` and no `calibration_matrix` has been identified, so its scale and
  per-channel signs are unverified. A wrong measured speed makes the smoother ramp
  against the controller on every tick, which is exactly a left-right weave on a
  straight leg plus a wrong fine correction where the final metre needs precision.
  The same parameter already has one documented pathology in this file: it once
  locked a stationary robot at zero.
- `feedback` is now `OPEN_LOOP`, so the smoother ramps from its own last command
  and the uncalibrated odometry is out of the velocity loop. This is config-only
  and reversible; re-pin `CLOSED_LOOP` after `ACCEPTANCE.md` step 4.
  `test_closed_loop_smoother_can_leave_rest_on_every_axis` was renamed to
  `test_smoother_can_leave_rest_on_every_axis` and now pins `OPEN_LOOP` while
  keeping the realizable-first-step invariant.
- Verified: MPPI is genuinely holonomic (`motion_model: Omni`, `vy_max: 0.702`),
  `velocity_smoother` allows `0.702` on y, and `rl_policy`, `runtime_guard` and
  the bridge all carry `linear.y` through with `linear_y_sign: 1.0`. The lateral
  axis is not being zeroed anywhere in the chain, and `/wheel/odometry` matches
  the topic `measurement_wheel` publishes.
- 56 package tests and 21 simulation tests pass.

Two tools were added so the next run settles this by measurement instead of by
another sign flip:

- `scripts/check_odometry_signs.py` - push and turn the robot by hand with
  `motors:=false` and it compares `/odom` against `/localization/pose` per axis,
  reporting a ratio and a verdict of OK / inverted / scale error. No motor moves.
  Run this first: if an axis is inverted, fix `count_signs`; if the scale is off,
  identify `calibration_matrix` with the `wheel_calibration` service.
- `scripts/record_run.py` - run it during an armed goal and Ctrl+C. It reports the
  `/cmd_vel_safe` peaks, what fraction of moving samples carried lateral velocity,
  and the sign-reversal counts for `vy` and `wz` in both the command and the
  `/odom` response. A weave present in the command is an MPPI tuning problem
  (`PathAlignCritic` 24.0 against `PathFollowCritic` 18.0 on a 400-sample search);
  a smooth command with an oscillating response is execution or odometry.

## 2026-08-05 - lateral sign settled: weave and reversed final approach

Reported symptom after the yaw fix: the circling stopped. The base now drove
roughly straight but wove left and right part-way through the run, then moved the
wrong way during the final approach.

- The circling ending confirms `auto_turn_sign = 1.0` was the right correction.
  What remained is the lateral axis alone.
- A bounded left-right weave rather than a divergence is what an inverted lateral
  axis looks like under this MPPI tuning: `PathAlignCritic` runs at weight 24.0
  and pulls the base back onto the global path hard enough to keep the mirrored
  correction from running away, so the loop settles into a weave. Below
  `PathAlignCritic`'s `threshold_to_consider: 0.55` that term drops out and
  `GoalCritic`/`GoalAngleCritic` own the final metre, where fine positioning is
  mostly lateral - so only the final approach visibly went the wrong way.
- `auto_lateral_sign` is therefore back to `-1.0`. This was its original value;
  it had been flipped to `+1.0` while the yaw runaway was still masking every
  other axis, and that flip is what produced this signature.
  `tests/test_omni_velocity.cpp` pins `+vy 0.5 m/s` back to wheels
  `{-3601, 3601, 3601, -3601}`.
- Signs are now measured rather than inherited: forward `+1.0`, lateral `-1.0`,
  turn `+1.0`. Both gateway tests pass, the binary was rebuilt, and
  `omni-gateway-next.service` was restarted.
- Known remaining inconsistency: the legacy/v2 auto path in
  `bacon_gateway/src/main.cpp` passes `lx` through, so it executes `ux = +vy` and
  still disagrees with this v3 contract. v2 is not used for autonomous driving;
  if it ever is, invert `lx` there to match.
- If a residual weave remains once the signs are right, it is a tuning question
  rather than a sign one: `PathAlignCritic` at 24.0 against `PathFollowCritic` at
  18.0 is an aggressive pair for a 400-sample holonomic search, and the
  measurement-wheel `count_signs` are still the unverified `[1, 1, 1, 1]` with no
  `calibration_matrix`. Run `scripts/check_odometry_signs.py` before retuning.

## 2026-08-05 - one-directional circle: yaw sign corrected

Reported symptom: after the wheel-budget fix the base no longer span in place,
but part-way through a run it began driving a constant-radius circle to the
right and never stopped.

- A circle held in one direction forever is a yaw runaway. If the commanded yaw
  executes backwards the yaw error grows monotonically to the profile limit, and
  because `translation_budget_share` now guarantees forward speed, the result is
  constant `vx` with a constant saturated `wz` of one sign - a fixed-radius
  circle. The earlier "only rotates, never advances" report was the same runaway
  before translation had any budget, which is why the symptom changed shape
  rather than disappearing when the budget was fixed.
- `auto_turn_sign` was `-1.0` on the inherited justification "matches the
  previous Jetson yaw-inversion spec", not on a measurement. It is now `1.0`.
  `tests/test_omni_velocity.cpp` pins the new contract: `+wz 1.0 rad/s` gives
  wheels `{4798, 4798, 4798, 4798}`. The combined forward+yaw saturation check no
  longer assumes wheel 0 is the peak, since which wheel saturates depends on
  whether the yaw term adds to or subtracts from the forward term.
- `auto_lateral_sign` stays at `1.0` from the previous entry. Flipping it did not
  change the circling, so lateral was not the cause; it is kept because the v2
  and v3 paths must not disagree, which is an independent defect.
- Both gateway tests pass, the binary was rebuilt, and
  `omni-gateway-next.service` was restarted. Current signs: forward `+1.0`,
  lateral `+1.0`, turn `+1.0`.
- `scripts/check_odometry_signs.py` was added to separate the two possible causes
  of a yaw runaway without moving a motor. It reads `/odom` while the robot is
  pushed and turned by hand: correct `dx`/`dy`/`dyaw` signs mean the fault is on
  the command side, an inverted `dyaw` means the measurement-wheel `count_signs`
  are wrong. Those signs are still the unverified `[1, 1, 1, 1]` with no
  `calibration_matrix`.

## 2026-08-05 - circling limit cycle: lateral sign aligned with the tuned path

Reported symptom: autonomous driving started normally and then settled into a
periodic "keeps driving in a circle" orbit instead of arriving.

- That is the signature of a mirrored lateral axis in a holonomic closed loop.
  With `vy` executed backwards the cross-track correction becomes positive
  feedback, so the base cannot null lateral error and orbits the goal at roughly
  constant `vx` and `wz`. It appears part-way through because the early route is
  mostly forward; the loop only becomes unstable once the commanded `vy` grows.
- `bacon_gateway/include/value.hpp` had `auto_lateral_sign = -1.0`, so the v3
  auto path executed `ux = -vy`. The legacy/v2 auto path in
  `bacon_gateway/src/main.cpp` passes `lx` through and inverts only `rx`, so its
  field-corrected contract is `ux = +vy`. The two paths therefore drove opposite
  lateral motion for the same ROS twist; forward and yaw already agreed.
- `auto_lateral_sign` is now `1.0`, matching the tuned path.
  `tests/test_omni_velocity.cpp` pins the new contract: `+vy 0.5 m/s` must give
  wheels `{3601, -3601, -3601, 3601}` (previously `{-3601, 3601, 3601, -3601}`).
  Both gateway tests pass and `omni-gateway-next.service` was restarted on the
  rebuilt binary.
- The mixer output is verified by `ctest`, not over the live link: while the Pi is
  disarmed it correctly reports zero wheel commands, so a wire-level read cannot
  confirm the sign without arming the motors.
- If a wheels-raised test shows `+y` still moving right, set the constant back to
  `-1.0` and restore the previous expected wheels in the same test; nothing else
  depends on it. `ACCEPTANCE.md` step 3 remains the authority.

## 2026-08-05 — spin-without-progress traced to the wheel budget

Reported symptom: under autonomous control the base only rotated and never
translated to the goal.

- `RuntimeGuard` gave yaw absolute priority over the wheel budget. Measured with
  the deployed numbers: yaw alone needs 9.421 wheel rad/s per commanded rad/s,
  so the `balanced` profile's 1.30 rad/s consumes 12.25 of the 15.709 rad/s
  wheel limit and leaves **0.245 m/s of the 0.78 m/s** forward limit (31%).
- `GoalAngleCritic` uses `threshold_to_consider: 12.0`, a field-sized value that
  means "always regulate goal yaw". With a large starting yaw error MPPI
  therefore requests yaw near the profile limit from the first tick, the guard
  honours it in full, and the base spins at about 75 deg/s while creeping. That
  is the reported behaviour, and it needs no hardware fault to explain.
- Fix: `allocate_omni4_wheel_budget` takes `translation_budget_share`
  (`config/robot.yaml`, default 0.45). While translation is requested, yaw may
  consume at most 55% of each wheel's limit, so forward speed never drops below
  0.500 m/s and yaw is capped at 0.917 rad/s instead of translation collapsing.
  A pure in-place rotation request still gets the full budget (1.30 -> 1.30).
- Separately, `wheel_drive_angles_deg` was `[-45, 45, 45, -45]`, which made a
  pure `+vx` drive all four wheels the same way and a pure `+wz` drive them
  alternating. That is the inverse of the deployed Pi mixer in
  `bacon_gateway/src/move.cpp` (`+vx` two each way, `+wz` all the same way), and
  no motor order or polarity can reconcile the two. Corrected to the tangential
  `[135, 45, 225, -45]`. The wheel-budget magnitudes are unchanged by this, so it
  was a latent modelling error rather than the cause of the symptom.
- 56 package tests and 21 simulation tests pass after the change.
  `test_drive_geometry_and_full_footprint_are_preserved` now pins the kinematic
  structure (yaw all-same-sign, translation two-each-way) instead of the literal
  angle list.

**Still unverified, and required before trusting goal arrival:** `ACCEPTANCE.md`
step 3 (command `+x`, `+y`, `+yaw` separately and confirm the physical
directions) has never been run. The lateral sign in particular is contradictory
in the code: the legacy/v2 auto path in `bacon_gateway/src/main.cpp` passes `lx`
through, so it executes `ux = +vy`, while the v3 path used for autonomous
driving applies `auto_lateral_sign = -1.0` and executes `ux = -vy`. Both are
geometrically valid omni solutions and they differ only in which corner is C620
ID 1-4, which is recorded nowhere. The measurement-wheel `count_signs` are still
the unverified `[1, 1, 1, 1]` with no `calibration_matrix`, so step 4 is open
too.

## 2026-08-05 — egg8PC <-> bacon6 link verified, gateway promoted to the boot service

- Measured on the 192.168.60.0/24 robot link with `motor_udp_bridge`
  (`payload_format: v3_velocity`, 100 Hz, motors disarmed): 14,169 v3 command
  datagrams delivered, 0% loss, 0 CRC errors, 0 stale drops, and a 0.21 ms mean
  echo-timestamp RTT.  The Pi reported `proto=v3`, `link_alive`, `uart_open` and
  `safety=DISARMED` throughout, and stopping the bridge produced `TIMEOUT` with
  the drive command held at zero.
- The earlier claim that no service was enabled on bacon6 was wrong.
  `mu3-robomas-receiver.service` was enabled and running, holding UDP 8888 and
  `/dev/serial0`.  Because the gateway socket sets `SO_REUSEADDR`, a second bind
  on 8888 succeeds instead of failing, so two receivers would have split the
  command stream with no error: running concurrent senders during this test
  raised the Pi's `stale_drop_count` by 21,119 within a few minutes.  Both
  receivers also open the same `/dev/serial0`.
- `mu3-robomas-receiver.service` is now stopped and disabled.
  `omni-gateway-next.service` is installed and enabled; it runs
  `bacon_gateway/auto_run_gateway.sh` and inherits the old unit's real-time
  scheduling and Energy-Efficient-Ethernet-off pre-start.  It declares
  `Conflicts=mu3-robomas-receiver.service rx_run.service`, and the launcher
  refuses to start while UDP 8888 or `/dev/serial0` is already held.  The old
  `/home/bacon6/MU3_robomas` tree is unchanged and can be restored with
  `systemctl enable --now mu3-robomas-receiver`.
- `/dev/ttyUSB0` is absent, so the MU-3 receiver is not connected and the Pi
  reports `mu3_alive: false`.  That flag is not part of the bridge's arming gate
  (`_telemetry_safety_healthy`), so it does not block autonomous commands, but
  manual MU-3 control is unavailable until the receiver is plugged in.
- **The motor command path now starts at boot, before the `ACCEPTANCE.md`
  wheels-raised tests have passed.**  The gateway stays disarmed until it sees a
  Jetson disarm packet followed by a fresh auto-request edge, and commands zero
  on link timeout, E-stop and process exit.  Keep the wheels raised, or run
  `systemctl disable --now omni-gateway-next`, until those tests pass.

## 2026-08-04 — offline model rebuilt on the real footprint

The reported fixed-bucket failure was traced to the offline model, not to the
learned policy. That model treated the robot as a disc of radius 0.39 m, near
the *inscribed* radius of the deployed ten-vertex footprint, and applied its own
arrival test. It reported 96/96 success at a 7.93 s mean on the pose 4 -> 5 and
5 -> 4 transits that this same document records as taking 26.66 s and 26.01 s on
the field, and the minimum clearance of those "collision-free" passes was
0.473 m, under the 0.510 m nearest vertex of the real outline. Every one of them
is a contact. The promoted table had therefore been pruned to five overrides
whose clearance bins all sit away from the bucket, with a mean time 6 ms from
the baseline: a measurable no-op.

- The offline model now uses the exact distance from the rotated `NORMAL`
  footprint to the CAD walls, and takes the arrival tolerance, progress
  allowance and Collision Monitor behaviour from `nav2_next.yaml` instead of
  restating them. An episode costs about 1.4 s rather than 0.05 s.
- Measured geometry: goals 4, 5, 6 and 7 have 0.057, 0.059, 0.057 and 0.047 m of
  footprint clearance at their commanded yaw, while `goal_checker` accepts
  0.04 m and 0.035 rad. The tolerance ball is larger than the margin, so an
  in-tolerance arrival at goal 4 or 7 can already be touching the field. Fixed
  bucket 1 at `(-0.8365, -0.55)` sits inside the `x≈-0.81` firing lane, and its
  gap to the divider is 0.40 m against a 0.90 m robot, so upper-to-lower
  transits must detour.
- Deterministic baseline over the 98 configured-goal transits, seed 20260808:
  94/98 arrivals, zero collisions, zero timeouts, **4 progress aborts**, 13.77 s
  mean, 2.8 mm minimum footprint clearance. The failure signature is now the
  field's `Failed to make progress` rather than a simulated collision.
- The offline obstacle response was calibrated against the only two recorded
  field timings. It is knife-edge: 0.22 repulsion authority makes every
  bucket-lane transit arrive in 13-19 s, contradicting both timings and the
  reported behaviour, while 0.30 reproduces frequent failure at comparable
  duration. That sensitivity is itself the finding. No rosbag of a failing run
  survives in the previous `ROS2` workspace — only colcon build logs — so this
  calibration rests on those two timings and the recorded failure descriptions.
- Promotion is now a paired per-scenario comparison. The previous rule rejected
  any candidate over 0.5% slower in aggregate mean time, which against the
  rebuilt model rejects the fix outright, because the deterministic controller
  does not arrive on the bucket-lane transits at all and anything that does is
  slower. A deployed policy must not regress a route that already worked and
  must not have collided; it need not pass the safety campaign, which the
  deterministic controller also fails and which is reported separately.
- `rl_policy_node` computes the same footprint clearance at runtime from the
  same CAD and footprint files, and reads the goal clearance from the
  orientation of the last `/plan` pose. The learned state gained
  `goal_clearance_margin` and `yaw_error`; the runtime table format is
  version 2 and a version 1 file is refused.
- Remaining limit: this is geometric. No controller makes a 47 mm margin robust
  against a 40 mm arrival tolerance. Re-surveying poses 4 to 7 for clearance is
  a separate change and is not done here.

## 2026-08-04 — runtime reinforcement-learning integration

- The supplied match image is now the startup-pose authority: pose 1 is
  `x=-1.80 m, y=4.75 m, yaw=-90 deg` and is the default for both the synthetic
  truth source and wall localizer. `--initial-pose-id` selects a different
  named pose without letting those two initial states diverge.
- The image-marked central-lane failure was reproduced in the motor-disabled
  ROS graph. A bucket-corner return inside the former static StopZone stopped
  every command direction, including a safe escape and Nav2 recovery, causing
  repeated `Failed to make progress`. Collision Monitor now uses the dynamic
  ten-vertex footprint and a 1.2 s directional approach check: motion toward a
  detected obstacle is stopped, while motion away remains possible. The 0.15 m
  SlowZone remains active.
- After the change, the exact red-lane round trip pose 4 -> pose 5 -> pose 4
  completed with `Goal succeeded` in the motor-disabled ROS graph. With the
  final critic weights, 4 -> 5 took 26.66 s and 5 -> 4 took 26.01 s, with no
  progress failure, localization dropout, or controller-rate miss.
- A second failure mode was then reproduced on the 5 -> 4 leg: lateral drift
  placed the rear LiDAR inside a fixture's 0.15 m blind zone, an all-infinite
  revolution was mistaken for a missing scanner, and RuntimeGuard held zero
  velocity. MPPI now tracks the footprint-safe lane more strongly, the swept
  footprint approach gate reacts to the first corner beam, and the localizer
  can continue with one geometrically sufficient scan while retaining a
  bounded 60 Hz health heartbeat. The same motor-disabled 5 -> 4 leg then
  completed without localization dropout or progress failure.
- The PyQt panel now selects configured goal IDs 0--7, displays their pose and
  navigation status, switches goals with cancel-before-replace semantics, and
  provides an explicit navigation stop. A live motor-disabled test canceled
  goal 5, started goal 0, and then canceled goal 0.
- `rl_policy_node` is in the live command path between velocity smoothing and
  Collision Monitor. RuntimeGuard requires its heartbeat and composes the
  learned scale into both Nav2 planning and final clamping.
- The deployed table was pruned to five full-speed clearance overrides. On the
  same 96-route seed 20260808 it passed 96/96 with no collision/timeout and a
  7.926 s mean versus the deterministic controller's 7.932 s. Promotion now
  rejects policies that are safe but slower.
- The ARM64 safe synthetic stack (`motors:=false`) executed configured goals
  with only 1.00-scale learned clearance actions, mandatory baseline fallback
  in the final 0.35 m, Nav2 `Goal succeeded`, and goal-executor `SUCCEEDED`.

## egg8PC

- New repository: `/home/egg8/Desktop/omni_autonomy_next`
- ROS 2 Jazzy packages build; all 48 package tests and all 14 simulation tests
  pass on ARM64.
- The 1,200-episode held-out campaign passes with 1,200/1,200 successes, zero
  collisions and zero timeouts. It now derives its profile from
  `config/runtime.yaml` rather than a copied constant.
- Speed-chain rework of 2026-08-03: `default_speed_scale` 0.65 -> 1.0, the
  Collision Monitor slow zone narrowed from a 0.50 m belt to 0.15 m at 0.80x,
  MPPI/velocity_smoother limits pinned to the executing `balanced` profile,
  goal yaw regulated for the whole route, and RuntimeGuard feeding
  `/speed_limit` back to `controller_server`.  Measured against the validated
  profile in the offline model, the previous configuration turned a 7.6 s route
  into 11.0 s, or 21.1 s wherever the wide slow zone was satisfied.
- End-to-end synthetic-demo evidence for the omni behaviour, pose-0 to pose-4
  leg (`demo:=true`, no GUI/RViz, motors off): goal succeeded with 39 mm /
  0.47 deg final error, peak `vy` 0.222 m/s against peak `vx` 0.452 m/s,
  lateral velocity present in 92% of moving samples, and the goal yaw already
  retired to 0.47 deg by the end of translation instead of after arrival.
  `/system/safety_state` reported `planner_speed_limit_pct: 70.0` matching
  `applied_scale: 0.7` inside the red zone, so the controller plans at the
  guard's envelope.
- Observed but **not** cleanly measured: `batch_size: 800` produced repeated
  `Control loop missed its desired rate of 20 Hz` (down to 9.7 Hz),
  RuntimeGuard `STALE_COMMAND` and `Failed to make progress`, and left a route
  unfinished.  The host was not idle during those runs, so the value is kept at
  the long-standing Jetson-safe 400 and the loop timing must be re-measured on
  an idle host before it is raised.  Chain probing in the same conditions
  showed MPPI's own command (not the safety chain) as the binding speed
  constraint, peaking near 0.45-0.52 m/s against the 0.78 m/s profile with the
  Collision Monitor publishing no slowdown state; re-check on the real field
  during acceptance step 5.
- No service is enabled and the old Desktop ROS 2 repository is unchanged.

## bacon6

- New repository: `/home/bacon6/omni_autonomy_next`
- The gateway builds natively with warnings treated as errors, and both host
  tests pass on bacon6's Debian 13 ARM64 environment.
- bacon6 is reached through egg8PC's robot network at `192.168.60.2`.
- `omni-gateway-next.service` is enabled and running the gateway from
  `build/omni_gateway_next`; the superseded `mu3-robomas-receiver.service`
  is stopped and disabled, and `/home/bacon6/MU3_robomas` is unchanged.
  See the 2026-08-05 entry for the link measurement and the motor-path
  caveat.

Physical commissioning remains pending.  Follow `ACCEPTANCE.md`; never enable
motor output merely because the software and simulation gates pass.
