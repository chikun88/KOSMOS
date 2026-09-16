# Inherited measured hardware facts

These values were copied unchanged from the deployed egg8 configuration on
2026-07-31.  `config/robot.yaml` is the machine-readable authority.

- Drive: four 50 mm omni wheels at `(+-0.333072, +-0.333072)` m; roller drive
  angles `[-45, +45, +45, -45]` degrees; maximum `15.709120382 rad/s` per
  wheel.
- Passive odometry: four 50.8 mm *outside diameter* wheels (25.4 mm radius),
  AMT102-V 2048 counts/revolution, mounted at `(0.2154,0)`, `(0,-0.2154)`,
  `(-0.2154,0)`, `(0,0.2154)` m.  The copied configuration held the diameter
  in `measurement_wheels.wheel_radius`, which made the geometric fallback
  report 2.02x the real translation whenever the calibration matrix below was
  absent or being re-fitted; `odometry_scale: [1, 1, 0.4977]` cancelled the
  same factor on the yaw axis only, so rotation-only checks looked correct.
  Both are corrected in `config/robot.yaml` and are now held to the hardware
  by `test_geometric_odometry_matches_the_on_robot_calibration`.
- Passive-wheel calibrated count-to-body-increment matrix.  Its effective
  gains agree with the corrected geometry above to within 1.3% (x), 5.8% (y)
  and 2.9% (yaw), which is the independent evidence for the 25.4 mm radius:

  ```text
  [-3.580812710e-05,  7.774156689e-05, -3.518553360e-05,  7.877766435e-07]
  [ 2.772848508e-04, -2.360042453e-04,  2.038839814e-04, -2.399216352e-04]
  [ 1.076063798e-04,  6.826634990e-05,  1.073162177e-04,  6.804585874e-05]
  ```

- Front LiDAR: `(x,y,z,yaw) = (0.400137729,-0.374424623,0.13,0.799360797)`.
- Rear LiDAR: `(x,y,z,yaw) = (-0.400137729,0.374424623,0.13,-2.321287905)`.
- Ten-vertex CAD footprint and persistent USB by-path device names are copied
  verbatim into `config/robot.yaml`.

The low-level Pi transport is derived from `/home/bacon6/MU3_robomas` because
its UART/COBS/MU3 behavior is a hardware protocol, not a replaceable planning
choice.  It lives in this separate repository, builds to a different binary,
and preserves the v3 CRC/sequence/re-arm contract required by egg8.

