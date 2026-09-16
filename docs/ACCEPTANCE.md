# Acceptance procedure

Software tests reduce risk; they do not replace physical commissioning.

1. Run `scripts/verify_all.sh` on egg8 and require every unit/integration test
   and the learned-residual non-regression gate to pass.

   The final **field-acceptance campaign** is expected to fail. Since the
   offline model was rebuilt on the real ten-vertex footprint it reports that
   the deterministic controller does not arrive on every configured-goal
   transit: 94 of 98 at seed 20260808, the remainder giving up with
   `Failed to make progress` in the fixed-bucket lane. That is a geometric
   limit — goals 4 to 7 have 47-59 mm of footprint clearance against a 40 mm
   arrival tolerance — not a software regression. Read `failing_routes` in
   `simulation/results/campaign.json` for the goal-number pairs, and
   `docs/REINFORCEMENT_LEARNING.md` for the measurements. Do not treat this
   step as passing until those poses are re-surveyed.

   Then run
   `python3 run.py demo --initial-pose-id 1 --goal-id 4` and require both Nav2
   and `goal_bridge` to report success. Pose 1 must report the image-derived
   initial state `x=-1.80 m, y=4.75 m, yaw=-90 deg`. While the synthetic system
   remains up, send goals 5, 6, and 7 with
   `python3 run.py goal --goal-id ID` from a second shell. Require
   `/rl/state` to stay healthy and `/navigation/goal_status` to report
   `SUCCEEDED` for each goal.
   Repeat once through the GUI: select 5, while moving select 0 and require the
   state to show goal switching, then press「移動を停止」and require
   `CANCELED`. The robot must never execute the old and new goals concurrently.
2. With drive wheels raised, start with `motors:=true`, keep the GUI disarmed,
   and verify fresh dual scans, wheel odometry, tracking, and motor telemetry.
3. The GUI speed slider now starts at 100%, meaning the `balanced` profile as
   tuned and validated.  **Drag it down to 10% before the first armed motion.**
   Then arm and command `+x`, `+y`, and `+yaw` separately.  Confirm the
   physical directions and that releasing arm, stale localization, unplugging
   Ethernet, and E-stop each yield zero motion.  Also confirm that the panel's
   E-stop release restores motion after a re-arm: it must call
   `/system/reset_motor_estop`, because the motor bridge ignores a cleared
   `emergency_stop` topic by design.
   Also verify that stopping `rl_policy` or withholding its heartbeat changes
   RuntimeGuard to `RL_POLICY_UNHEALTHY` and produces immediate zero motion.
4. On the floor, validate 0.25 m straight, strafe, diagonal, and one-turn tests.
   Recalibrate the count matrix if translation error exceeds 20 mm or yaw error
   exceeds 1 degree.  `config/robot.yaml` now declares the passive wheel's
   25.4 mm radius (the part is 50.8 mm across) with `odometry_scale: [1,1,1]`;
   a re-fitted matrix must still satisfy
   `test_geometric_odometry_matches_the_on_robot_calibration`.
5. Run every named pose at 20%, then 40%, then 100% of the balanced profile.
   RuntimeGuard republishes the slider and profile as a `nav2_msgs/SpeedLimit`
   percentage, so MPPI plans at each of those speeds instead of being clipped
   afterwards; check `/system/safety_state`'s `planner_speed_limit_pct` tracks
   the slider.  Require final error <= 40 mm and <= 2 degrees for poses 4-7, no
   footprint contact, and no tracking dropout.
   Send poses from the GUI number selector (the CLI
   `python3 run.py goal --goal-id ID` remains available); do not treat a
   planned path as arrival. Require `/navigation/goal_status` to say
   `SUCCEEDED` and independently measure the physical final pose.
   At each speed, explicitly run the red-lane round trip 4 -> 5 -> 4. Require
   both legs to complete without `Failed to make progress`; confirm that
   `FootprintApproach` stops commands toward a real obstacle but permits a
   commanded retreat from it. Keep a spotter at the E-stop and stop the test
   immediately on footprint contact.
6. While climbing through step 5, watch `controller_server` for `Control loop
   missed its desired rate` and RuntimeGuard for `STALE_COMMAND`.  `batch_size`
   stays at 400 because 800 was seen halving the loop rate; that observation was
   made on a busy host, so if more holonomic search is wanted, re-time the loop
   on an idle Jetson first.  Note the real robot replaces the synthetic scan
   node with the LiDAR drivers and the measurement-wheel node.
7. Only after three consecutive clean full-field runs may the sprint profile be
   enabled.  A profile change never bypasses RuntimeGuard's hard limits, but
   sprint is capped by the static MPPI and velocity_smoother limits in
   `config/nav2_next.yaml`; raise both together with the profile, and re-run
   the campaign gate so it measures the speeds you intend to drive.

No claim of competition-ready perfection is valid until these physical tests
pass on the assembled robot and actual field.
