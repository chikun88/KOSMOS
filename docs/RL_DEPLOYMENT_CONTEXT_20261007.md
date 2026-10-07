# RL deployment context

The deployed policy has an empty override table. It supplies the deterministic
baseline and does not apply a learned action. Its historical certification is
not a current physical acceptance result.

The 2026-10-07 audit found a feature mismatch: the offline learner normalizes
model **actual velocity** by its selected runtime profile speed, while the ROS
adapter observes the **smoothed command** and a configured fixed reference speed
(historically 0.78 m/s). Those signals differ during acceleration, latency and
braking. Matching bin names alone does not make a learned table compatible.

Nonempty tables therefore require an `observation_context` declaring
`velocity_source`, `reference_speed_mps` and `reference_profile`. The runtime
loader must receive exactly the same context; missing or different contexts
raise a clear error. The ROS adapter binds its actual smoothed-command source,
configured reference speed and explicitly configured `policy_reference_profile`.
It also requires a fresh matching profile from RuntimeGuard before accepting a
command under a nonempty policy. Empty tables require no learned-context binding.

CAD observations also require map-frame feedback with a positive acquisition
timestamp. Pose callbacks reject replayed/out-of-order, expired or future source
samples, and receipt time retains the source sample's remaining lifetime.
Freshness is checked against both ROS acquisition time and monotonic receipt
time; ordering can restart after an observed ROS clock rollback. Path headers
and every pose must carry map-frame, valid timestamps and finite normalized yaw.

The current simulator records `model_actual_velocity`. Automatic promotion and
export refuse a nonempty table from that context. Setting a profile parameter
does not repair the training-data mismatch: training and evaluation must first
use the same observation signal and speed reference as deployment, and pass the
paired and contact gates. No ungrounded retraining or reference-speed change was
used to bypass this restriction.

[The current paired check](SYSTEM_AUDIT_RL_20261007.json) evaluates 98 configured
scenarios using seed 20260808. Baseline and the empty policy are identical:
92 arrivals, 6 model contacts, no timeouts or progress aborts. Paired
non-regression passes; the combined promotion gate fails. These are results from
an MPPI-style surrogate, not observed hardware collisions or physical acceptance.
The [pre-gradient result](SYSTEM_AUDIT_RL_PRE_GRADIENT_20261007.json) is retained
separately: 89 arrivals, 5 contacts, 1 timeout and 3 progress aborts. The changed
geometry fixes the demonstrated lane-correction sign error, but the paired
contact count rose from 5 to 6. It does not establish uniform safety improvement.
