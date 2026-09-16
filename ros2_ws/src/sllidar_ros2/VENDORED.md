# Vendored third-party package

This directory is an unmodified copy of the Slamtec RPLIDAR ROS 2 driver.

- Upstream: https://github.com/Slamtec/sllidar_ros2.git
- Commit: `3430009` ("bugfix:set the minimum distance to 5cm")
- License: BSD 2-Clause, see `LICENSE`.

`system.launch.py` runs `sllidar_ros2/sllidar_node` once per LiDAR listed in
`config/robot.yaml`, so `omni_autonomy_next` declares it as an `exec_depend`.
The driver is vendored rather than referenced from a sibling workspace so that
`colcon build --packages-up-to omni_autonomy_next` produces a complete runtime
on every deployment host.

Do not edit these sources.  Re-vendor from upstream if the driver must change.
