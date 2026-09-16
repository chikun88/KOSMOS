# Safety-constrained reinforcement learning

## Runtime architecture

The learned controller is part of the ROS 2 command path:

```text
NavigateToPose
  -> MPPI
  -> velocity_smoother
  -> RL residual
  -> Collision Monitor
  -> RuntimeGuard
  -> motor gateway
```

`rl_policy_node` observes the global path, localized map pose, current smoothed
command, and CAD wall clearance. Every 0.5 seconds it selects a learned speed
scale and clearance response. RuntimeGuard publishes the composed scale back to MPPI as a
Nav2 `SpeedLimit` and independently clamps the final command.

The policy cannot output an arbitrary velocity. Its speed scale is at most
`1.0`, its wall correction cannot weaken the baseline response, and correction
never increases the input translation norm. Collision Monitor and RuntimeGuard
remain downstream. Missing localization, invalid policy output, node failure,
or a stale RL heartbeat produces a zero command.

## Configured goals

Goal IDs `0` through `7` are defined in `config/field_poses.yaml`. A goal is
validated, sent through `NavigateToPose`, monitored to completion, and retried
up to two times when Nav2 rejects or aborts it. Status is published on
`/navigation/goal_status`.

The main GUI lists all eight IDs with their Japanese names and exact configured
pose. Selecting another number during travel cancels the current Nav2 action
before sending the replacement;「移動を停止」cancels both queued and active
requests. Goal requests are volatile, so restarting the executor cannot replay
an old GUI click.

The system start pose is read from that same file.  The default is pose `1`,
the supplied match-image observation `x=-1.80 m`, `y=4.75 m`,
`yaw=-90 degrees`; use `--initial-pose-id 0` only when beginning from the
loading-wait location outside the start zone.  The synthetic robot and wall
localizer receive the same pose, preventing simulation/localization drift from
duplicated hard-coded values.

Run the complete stack with synthetic sensors and no motor output:

```bash
python3 run.py demo --goal-id 4
```

Keep the system running and send another configured goal from a second shell:

```bash
python3 run.py goal --goal-id 5
```

For hardware sensing with motor output disabled:

```bash
python3 run.py real --goal-id 4
```

Only after every physical acceptance step passes may the motor-capable system
be started. Start it without an automatic goal, lower the GUI scale to 10%,
complete the arm/preflight checks, and only then send the goal from a second
shell:

```bash
python3 run.py full --accept-motor-risk
python3 run.py goal --goal-id 4
```

No software can guarantee physical arrival under wheel slip, mechanism
interference, people, opponents, electrical faults, or incorrect CAD. The
action result, safety topics, and physical acceptance procedure remain
mandatory evidence.

## Why the offline model was rebuilt

The learned controller could not improve the fixed-bucket approach while the
offline model did not contain the failure. That model treated the robot as a
disc of radius 0.39 m, close to the *inscribed* radius of the deployed
ten-vertex footprint, and used its own arrival test. It reported 96/96 success
at a 7.93 s mean on exactly the routes the field log records as taking 26.66 s
and 26.01 s, and its "collision-free" bucket-lane passes had a minimum
clearance of 0.473 m, below the 0.510 m nearest vertex of the real outline.
Every such pass is a contact on the field. Trained in that model, the promoted
table shrank to five overrides whose clearance bins all lie away from the
bucket, and its mean time matched the baseline to 6 ms: a measurable no-op.

The model now uses the exact distance from the rotated `NORMAL` footprint to
the CAD wall segments, and the gates the command actually passes, read from
`nav2_next.yaml` rather than restated here:

| Modelled | Source |
| --- | --- |
| Ten-vertex footprint clearance, yaw-dependent | `competition_footprints.yaml` |
| `xy_goal_tolerance` 0.04 m, `yaw_goal_tolerance` 0.035 rad | `goal_checker` |
| `PoseProgressChecker` 0.04 m within 8.0 s, abort on failure | `progress_checker` |
| `FootprintApproach` 1.2 s swept-footprint approach scaling | `collision_monitor` |
| `SlowZone` 0.8 proximity brake | `collision_monitor` |
| 0.10 s command latency through smoother, residual, monitor and UART | measured |

`simulation/body_model.py` holds the clearance model. It is exact, never
reports more clearance than an independent dense-boundary check, and is the
authority for contact. Planning uses an eroded field at 1-degree yaw bins,
which is a few millimetres pessimistic between nodes and so safe for a planner
but cannot see a bucket facet swallowed whole by the outline; only the exact
model is used to decide contact.

## The geometry this exposes

| Goal | Name | Footprint clearance at its commanded yaw |
| --- | --- | --- |
| 4 | 固定バケツ② | 0.057 m |
| 5 | 固定バケツ③ | 0.059 m |
| 6 | フラッグ上 | 0.057 m |
| 7 | フラッグ下 | 0.047 m |

`goal_checker` accepts a 0.04 m position and 0.035 rad yaw error, so the
tolerance ball is larger than the margin: an in-tolerance arrival at goal 7 or
goal 4 can already be touching the field, and 10 degrees of yaw error at goal 4
leaves 0.009 m. Fixed bucket 1 sits at `(-0.8365, -0.55)`, inside the
`x≈-0.81` firing lane, so every transit between an upper goal (4, 6) and a
lower one (5, 7) must thread past it; the divider-to-bucket gap is 0.40 m and
the robot is 0.90 m wide, so the detour is forced.

This is a geometric limit, not a tuning one. No controller makes a 47 mm margin
robust against a 40 mm tolerance. The learned residual reduces how often the
lane is lost; it cannot remove the cause. Re-surveying poses 4 to 7 for more
clearance, or tightening the arrival tolerance below the available margin, is a
separate and larger change.

## Training and promotion

Training is offline and dependency-free tabular Q-learning. The observation is
discretized from remaining goal distance, **footprint** clearance, path-turn
error, speed fraction, the **clearance of the goal pose itself**, and **yaw
error**. The last two are what let one table hold different behaviour for a
tight firing pose and an open one. The reward combines progress, elapsed time,
clearance, arrival, timeout, collision, progress abort, and time spent under
the collision brake.

The decisive action is a slower command, not a stronger push. The Collision
Monitor projects the commanded velocity over 1.2 s, so a slower command is less
likely to predict contact and be zeroed; stop-and-go is what trips the progress
checker. The action set therefore reaches down to a 0.35 speed scale.

```bash
python3 run.py rl-train
```

Promotion is a **paired, per-scenario** comparison, not an aggregate one. The
old gate rejected any candidate more than 0.5% slower in mean time. Against the
rebuilt model that rule rejects the fix, because the deterministic controller
does not arrive at all on the bucket-lane transits and any policy that does
arrive there is slower. The gate now requires, for every scenario the
deterministic controller already solved, that the candidate still arrives and
takes no more than 1.10x its time plus 0.75 s; scenarios the baseline fails may
be traded freely. A candidate must also never have collided.

A deployed policy therefore carries `paired_regression_passed` and
`evaluation_collisions: 0`. It does **not** have to pass the safety campaign,
because the deterministic controller does not either; that campaign remains the
separate system-level acceptance gate and is reported independently. An
unsuccessful candidate is saved as `rl_policy.rejected.json` and cannot
overwrite the deployed policy.

Re-evaluate the accepted full model on another seed with:

```bash
python3 run.py rl-evaluate \
  --episodes 120 --random-episodes 24 --seed 20260805
```

The compact runtime policy contains only sufficiently visited greedy decisions.
All unknown or weakly sampled states fall back to the unmodified deterministic
MPPI controller. Inside the final 0.35 m goal-convergence region the same
fallback is mandatory; learning cannot slow or redirect the final arrival, and
that shield now applies during training too, so the learner cannot value
behaviour the deployed policy will never take.

`rl_policy_node` computes the same footprint clearance at runtime, from the same
CAD and footprint files, and takes the goal clearance from the orientation of
the last pose of `/plan`. A base_link distance would not do: every vertex of the
outline is at least 0.510 m from base_link, so base_link clearance says nothing
about whether the body fits.
