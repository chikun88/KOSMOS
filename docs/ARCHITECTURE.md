# Architecture

## Command path

`NavigateToPose / NavigateThroughPoses -> MPPI -> velocity_smoother -> RL residual -> Collision Monitor -> RuntimeGuard -> wheel mixer -> UART frame -> UDP v4 -> bacon gateway (relay) -> RoboMaster board`

Each arrow is a topic boundary with one publisher.  The final two gates are
independent: Collision Monitor evaluates fresh LiDAR points; RuntimeGuard
evaluates authorization, localization, communication health, timing, dynamics,
and wheel feasibility.

### Drivebase transport: egg8 builds the UART frame, bacon6 relays it

Since 2026-08-07 the wheel mixing happens on egg8, not on bacon6.  The Jetson
converts the twist into the four wheel commands and packs them into exactly the
byte string the Development Board expects — `[command id, value hi, value lo]`
triplets wrapped in COBS — then ships that string over Ethernet in a v4 UDP
command.  bacon6 writes those bytes to `/dev/serial0` without modifying a single
one.  `omni_autonomy_next/robomas_uart.py` and `bacon_gateway/src/uart.cpp` are
therefore two implementations of one wire format and must be changed together.

Three properties of the split are deliberate:

**bacon6 stays the manual driver.**  MU3 handheld control is unchanged.  The
relay only replaces the drivebase commands, and only while DualSense L2 requests
auto mode with a live link.  Everything else — and every command id egg8 did not
put in its frame — is still computed on bacon6 from the MU3 buttons and appended
as a second COBS packet in the same `write()`.  Today egg8 sends the four omni
commands and bacon6 supplies ARM, GM, up/down, collect and GPIO, so mechanisms
remain operable by hand during an autonomous run.  When egg8 starts sending a
mechanism command, bacon6 stops emitting that one automatically; nothing needs
to be switched over.

**Relaying is not the same as forwarding blindly.**  bacon6 decodes the frame it
received before relaying it.  A frame that fails CRC, whose length disagrees
with its header, or whose payload is not a whole number of commands is dropped
rather than pushed at the board — a corrupt COBS packet would desynchronize the
board's parser and take valid frames down with it.  Decoding also gives bacon6
the wheel values it needs for the next property.

**The safety stop stayed on bacon6.**  `DriveLinkSafety` and
`ControlledStopLimiter` are unchanged and still authoritative.  While the link
is healthy the limiter passes the relayed wheel values through untouched, so the
bytes on the wire are egg8's.  The moment a controlled stop, an E-stop or a
command timeout applies, bacon6 stops relaying and emits its own full nine
command frame with the wheels ramped down from the value it last relayed.  A
Jetson that stops transmitting therefore decelerates the robot instead of
leaving the board holding the last command.  Measured on the bench: relay frames
are byte-identical to the ones egg8 built, and an aborted command stream ramps
each wheel down in 80-unit steps per 5 ms loop.

`scripts/passthrough_wire_check.py` (bacon6) and
`scripts/passthrough_send_probe.py` (egg8) reproduce those measurements without
the Development Board attached: the gateway's motor output is pointed at a pty
via `MU3_MOTOR_DEVICE` and the captured bytes are decoded back into commands.

Collision Monitor uses a slowdown belt plus a 1.2 s `FootprintApproach` check
over the complete dynamic footprint. The approach check is directional: it
blocks a commanded swept footprint that reaches an obstacle, but does not turn
a corner return beside the robot into an unconditional stop that also forbids
the safe escape direction. This distinction is required in the image-marked
central lanes around poses 4 and 5.

There is one deliberate feedback edge: RuntimeGuard publishes the operator's
profile and speed scale back to `controller_server` as a `nav2_msgs/SpeedLimit`
percentage on `/speed_limit`.  Without it the guard was a hidden governor —
MPPI planned and scored rollouts inside an envelope the guard then silently
reduced, so the controller's predicted motion, and therefore its critic costs,
described a robot that did not exist.  The guard still clamps unconditionally;
the feedback only stops the controller from asking for the impossible.

## State estimation

Four passive measurement wheels propagate `odom -> base_link` at high rate.
The inherited calibrated 3 x 4 count-to-twist matrix is the sole wheel model;
the nominal geometry is retained only as documentation and fallback.  Dual
diagonal A2M8 scans are deskewed with wheel odometry, matched against the
multi-height CAD wall model, and provide bounded `map -> odom` corrections.

`wall_localizer` substitutes the last known `odom -> base_link` whenever no
wheel odometry has arrived within `wheel_odom_tf_timeout_sec`.  A missing
counter board used to leave `base_link` with no parent, which silently removed
the robot, both LiDAR point clouds, and the odom-frame local costmap from RViz
while the field CAD in `map` still rendered.  The substitute reuses the last
wheel transform, so the `odom` frame does not jump if the counters drop out
mid-run, and it stops as soon as `/wheel/odometry` resumes.  Localization is
then LiDAR-only: scans are no longer deskewed and Nav2 still needs
`/wheel/odometry` for controller feedback.

Nav2 lifecycle processes are started from `wall_localizer`'s ready output,
after its one-time CAD lookup grid build has completed. This avoids competing
with that CPU-heavy build but does not impose the former unconditional 26 s
delay. The 26 s timer remains as a start-once fallback if process output is
unavailable; the event and timer share one latch, so Nav2 cannot be launched
twice.

Scan ingestion, ICP, wheel propagation, and the 60 Hz pose/health publisher use
separate callback groups. A fresh all-infinite revolution is recorded as an
empty observation rather than mistaken for a dead sensor, so one temporarily
blind LiDAR does not discard hundreds of valid wall points from the opposite
LiDAR. The health heartbeat retains only a bounded, one-second-old accepted
ICP solution; missing scans, a rejected solution, or a stuck solve still closes
RuntimeGuard.

## Planning and control

Smac 2-D supplies cost-aware global paths for arbitrary field goals.  MPPI uses
the `Omni` motion model, the complete ten-vertex footprint, and independent
`vx`, `vy`, and `wz` samples.  This is intentionally holonomic: body yaw can
track the mechanism target while translation follows the best clearance
direction.

Two settings decide whether that freedom is actually used.  `batch_size` must
cover a three-dimensional control space rather than the two a differential base
needs, and `GoalAngleCritic.threshold_to_consider` must stay large enough to
regulate goal yaw for the whole route.  Gating that critic near the goal turns a
holonomic move into translate-then-spin, which is also what makes the online
controller disagree with the Monte Carlo model that certifies it — that model
has always driven yaw toward the goal from the first tick.

One stage shapes, the next stage gates.  `velocity_smoother` owns acceleration
shaping and MPPI's `ax/ay/az` bounds are pinned to it.  RuntimeGuard's profile
holds acceleration and jerk strictly *above* those, so it passes an
already-limited command through unchanged and only bites when something
upstream exceeds it.  When the two carried the same numbers the guard
re-limited every command and its jerk term became an unmodelled second-order
lag inside the 20 Hz loop: measured, 90 ms of added latency and 44% of the
configured acceleration actually delivered.  The ordering to preserve is
`MPPI = velocity_smoother < RuntimeGuard profile <= hard_max_*`.

Fitting a twist into the four-wheel speed budget scales the whole twist by one
factor.  Direction of travel, the translation-to-yaw ratio and therefore the
instantaneous curvature are preserved, so a saturating command becomes the
planned trajectory executed more slowly - a pure time rescaling the optimizer
absorbs by replanning from the measured state.  Scaling translation and yaw
separately instead changes the curvature the base performs, which takes it off
the planned path under its own power and makes the tracking loop hunt.  The
deployed Pi mixer already scales all four wheel commands by one common factor,
so this is also the only way the two allocators in the command path agree.

MPPI's velocity and acceleration constraints are held equal to the RuntimeGuard
profile that executes, because the optimizer integrates its rollouts with those
constraints before the critics score them.  Braking is the case that matters:
an optimistic `ax_min` lets CostCritic accept gaps the drivebase cannot stop
inside.

Named competition goals are overlays on the same general planner.  Poses 0,
2, 3, 4, 5, 6, and 7 are marked as critical Monte Carlo targets because they
come from the red-marked regions in the supplied field image.  Poses 4-7 also
activate the precision arrival profile automatically.

Goals 4 and 5 additionally have deterministic bidirectional lanes in
`config/routes.yaml`. On arrival, `goal_bridge` selects the upper or lower
entry from the current localized Y position. When the current pose is at goal
4 or 5, it selects the departure gate on the side of the next destination.
The action pose list is `[departure gate, arrival gate, final goal]` when both
contracts apply, for example on the 4-to-5 shuttle. The list is refreshed from
the latest localization immediately before every send or retry, so a startup
or cancellation delay cannot preserve a stale side selection.

Pose 1 has a mandatory departure gate for a different geometric reason. At
the loading pose the commanded -90 degree footprint has 48.6 mm of CAD
clearance, but intermediate yaws on the direct path to pose 2 overlap the
fixed structure. `goal_bridge` forces the position path through a gate 0.65 m
toward the opening. The tracker's yaw ramp over the combined gate path retains
at least 48 mm clearance, and at the gate the complete -90-to-0-degree yaw
sweep retains at least 60 mm. This keeps `FootprintApproach` from scaling all
three twist components to a crawl while the base is still inside the slot.

The custom `NavigateThroughPoses` behavior tree replans at 1 Hz and retains
each gate until the robot is within 0.12 m; this prevents a replan from cutting
the narrow fixed-bucket lane early. Smac still chooses the route between fixed
gates, so live obstacle avoidance is retained. An arbitrary RViz destination
also receives a fixed departure when starting at pose 1 or goal 4 or 5. Other
motion continues to use `NavigateToPose`.

The GUI publishes numbered requests on a volatile topic so an old selection is
never replayed after restart. `goal_bridge` waits until all required Nav2
lifecycle nodes are active, then executes the request. Selection is
latest-wins: a new number first cancels the active action and only then sends
the replacement. The separate cancel request also clears a queued goal.

## Offline reinforcement learning

`simulation/reinforcement_learning.py` trains tabular Q-learning as a bounded
residual controller around the deterministic path tracker. The promoted greedy
table is loaded by `rl_policy_node.py` between velocity smoothing and Collision
Monitor. The state contains remaining distance, wall-clearance margin, turn
error, and current speed. Every 0.5 seconds the policy chooses a speed scale
and an obstacle-repulsion multiplier, avoiding the former quarter-second
action chatter.

The action contract is intentionally asymmetric: speed scale is in `(0, 1]`
and clearance push is in `[1, 2]`.  Loading a model outside that envelope is an
error, and the simulator checks the same contract on every action.  Thus the
learner cannot raise configured speed/acceleration limits or weaken the
baseline wall response.  The classical controller, collision checks, and
RuntimeGuard remain authoritative.

Training and held-out evaluation run only in the CAD simulator. A candidate
must pass both the safety gate and a paired performance non-regression gate
against the deterministic controller before it may be promoted to
`config/rl_policy.yaml`; a failed or slower model cannot replace the runtime
policy. RuntimeGuard requires the policy heartbeat and
composes its learned speed scale into both the Nav2 `SpeedLimit` and the final
hard clamp. See `docs/REINFORCEMENT_LEARNING.md` for the workflow.

## Configuration authority

Hard physical limits live in `config/robot.yaml` and cannot be raised by the
GUI.  Runtime profiles in `config/runtime.yaml` may only reduce those limits.
The GUI publishes a transient-local profile and scale request; RuntimeGuard
clamps and reports the accepted values.

The profile is the speed policy and `default_speed_scale` is the operator's live
override, so the default scale is 1.0.  A default below 1.0 multiplies every
profile invisibly, and no offline gate sees it; `simulation/run_campaign.py`
therefore builds its profile from `config/runtime.yaml` instead of holding its
own copy, so the campaign always measures the robot that will drive.
