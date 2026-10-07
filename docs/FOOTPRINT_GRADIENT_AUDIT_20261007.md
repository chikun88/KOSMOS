# Footprint-gradient correction and model evidence

The model v3 evidence from frozen commit `7dddbb5` is preserved byte-for-byte:

- `SYSTEM_AUDIT_MODEL_PRE_GRADIENT_20261007.json`, SHA-256 `1447ac3977016b113f5c81137c74cee4ec1175ec4d11b5eafcb9a998659f09f2`: 42/49 arrivals, 4 contacts, 2 timeouts, 1 progress abort.
- `SYSTEM_AUDIT_RL_PRE_GRADIENT_20261007.json`, SHA-256 `84c23197e03a2fbb2f61c16f39337964a6b43b1b79efbf017d5f91e35e475a0d`: 89/98 arrivals, 5 contacts, 1 timeout, 3 progress aborts; empty RL matches baseline.

These are prior-revision results. Fresh model v4 evaluates the same settings and
seeds after the repulsion-direction correction:

- `SYSTEM_AUDIT_MODEL_20261007.json`: 46/49 arrivals, 3 model contacts, no timeouts or progress aborts; acceptance fails.
- `SYSTEM_AUDIT_RL_20261007.json`: baseline and empty RL both 92/98 arrivals, 6 model contacts, no timeouts or progress aborts; paired non-regression passes, combined promotion fails.

The three prior pose-4 stalls now arrive in the 49-scenario run. Its prior
7→2 contact also arrives, but 7→2 still contacts in one of the paired run's two
trials. The paired contact count increased from 5 to 6 even as arrivals rose
from 89 to 92; these results do not support a uniform safety-improvement claim.
Generated v4 reports fingerprint executable model sources, including the
packaged gradient helper shared with the runtime adapter. Passing unit tests or
improving arrival counts does not establish model or physical acceptance.

The later tracker-only combined XY/yaw swept-path validation and bounded repair
are outside this surrogate's execution dependencies. The reports' current
revision scope is the declared offline model inputs; neither report evaluates
the production trajectory tracker. Their runtime-file hashes provide reference
provenance for the residual and source-validation code, rather than evidence of
live ROS execution. The unchanged numerical source/configuration hashes keep
the 46/49 and 92/98 results applicable to that same surrogate after the tracker
edit; tracker route and software evidence must identify its own tested revision.

The `rl_policy.yaml` header's 89/98 "final paired replay" and its repulsion/
calibration commentary are retained historical pre-gradient statements; they
do not describe the current 92/98 v4 result. Likewise, `SimProfile` commentary
about matching two recorded aggregate round-trip timings predates this geometry
correction and does not identify or validate the current physical plant. No
policy table, parameter, gain, pose or physical limit was changed to obtain the
new results.

An empty policy table still passes through the runtime's configured deterministic
footprint correction and yaw limits. Exact paired equality here compares the
empty learned action with the same surrogate baseline stages; the policy-header
"bit-identical" wording must be read in that paired-action context.

At the saved stalled position `(-0.825, 0.825)`, yaw zero, the chassis outline
is 23.558 mm from bucket 2. Translating in map +X increases that clearance.
The nearest wall to base_link is instead the central wall `x=-0.3`, and its
normal is map -X. The old surrogate therefore pushed toward the bucket. The
runtime adapter used that same centre normal for its tracker baseline: +0.1 m/s
of helpful correction became +0.081779 m/s, while -0.1 m/s toward the closest
body obstacle passed unchanged. This is a demonstrated production geometry bug,
although the surrogate's active push and runtime's attenuation differ.

The shared analytic helper now uses the closest outline/wall features at the
current map yaw. Vertex-to-wall and wall-endpoint-to-outline-edge normals both
point from the wall toward the outline. Duplicate facets do not weight the
answer. A tied-feature mean is retained only when it does not approach any
equally close feature; conflicting ties, contact and capped open clearance have
no invented direction. The runtime baseline still attenuates only the inward
component; gains, thresholds, speed bounds and downstream collision gates are
unchanged. Nonfinite residual inputs raise an error and the adapter publishes
zero with unhealthy status.

The prior failure signatures locate the remaining evidence as follows:

| Route | Prior outcome and location | Interpretation / next evidence |
|---|---|---|
| 2→4 | Contact at 4.20 s, centre `(-0.751, 1.703)`, central wall `cad_surface_12_00` (`x=-0.3`). | Final sampled speed about 0.50 m/s with only 16.7 mm preceding actual clearance. Compare deployed tracker braking/prediction on this approach. |
| 4→7 | Contact at 5.80 s, `(-2.769, 0.421)`, high CAD facets `cad_z1500_02_00…02` near `x=-3.084`, top `y=0.05`. | Final sampled speed about 1.14 m/s and preceding clearance 14.7 mm. Check full-body path/turn sweep and delayed braking, retaining height/outline assumptions until surveyed. |
| 6→3 | Contact at 4.60 s, `(-4.699, 0.025)`, `cad_surface_04_01/02` corner near `(-4.81,-0.37)`. | Final sampled speed about 0.52 m/s and preceding clearance 13.0 mm. Compare the actual gate/trajectory path and approach velocity. |
| 7→2 | Contact at 8.20 s, `(-4.252,4.827)`, upper boundary `cad_surface_12_01` (`y=5.25`). | Final sampled speed about 0.47 m/s and preceding clearance 6.2 mm. Inspect late braking and pose-estimation reserve. |
| 3→4 | Progress abort at 16.65 s, about 0.526 m short of goal, centre `(-0.831,0.815)`. | Bucket-lane gradient mismatch is directly applicable; rerun without changing checker limits. |
| 5→4 | Timeout at 28.40 s, about 0.515 m short, centre `(-0.825,0.825)`, 43 command reversals. | Same incorrect body-obstacle classification; timeout is an offline budget, not a deployed bridge deadline. |
| 7→4 | Timeout at 26.70 s, about 0.512 m short, centre `(-0.837,0.829)`, 34 reversals. | Same bucket-mouth stagnation; compare the checked fixed `x=-0.80` approach after the gradient fix. |

The speeds above are differences between the last two saved actual positions
divided by the model's 0.05 s step. They are sampled displacement rates, not
measured hardware speeds or exact obstacle-normal closing velocities.

The surrogate uses its own A* route, lookahead, repulsion, fixed 0.10 s command
queue and symmetric 0.85 m/s² **plant** acceleration. It omits production fixed
approach gates and the full trajectory/predictor behavior. The deployed tracker
instead plans forward command acceleration at 3.30 m/s² in sprint, keeps a
separate 0.85 m/s² braking/correction budget, and uses measured velocity with
delay compensation. Raising model acceleration to the tracker value would not
identify the physical plant or demonstrate safe behavior.

The v4 49-scenario contacts remain 2→4, 4→7 and 6→3. Their final preceding
clearances are respectively 16.7, 9.0 and 13.0 mm; their final sampled speeds
are about 0.50, 1.17 and 0.52 m/s. Contact at 4→7 occurs at 5.75 s rather than
the prior 5.80 s. All three remain failed scenarios, rather than geometry-test
false positives repaired by the gradient change.

The feasible next sequence is paired publisher-free production replay of the
named failures using the same stored route/observation stream, followed by
motor-disabled ROS route evaluation.
Record requested/limited velocities, measured versus predicted pose and body
clearance around each failure. Physical stopping distance, latency under load,
mechanism outlines and the provisional image/CAD pose registration still need
measurement: the configured 47–59 mm firing margins do not validate coordinates
documented with approximately ±0.10 m survey uncertainty.
