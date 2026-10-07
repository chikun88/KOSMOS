"""A slow LiDAR calculation must not fabricate a wheel sensor outage."""
from pathlib import Path
import threading
import time

import numpy as np
import pytest
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

from omni_autonomy_next import synthetic_scan_node as simulation


@pytest.mark.parametrize('isolated_scan', [False, True])
def test_slow_scan_keeps_receiving_commands_and_publishing_wheels(monkeypatch, isolated_scan):
    config = Path(__file__).resolve().parents[1] / 'config'
    rclpy.init(args=['--ros-args',
                    '-p', f'field_config_file:={config / "field_planning.yaml"}',
                    '-p', f'robot_config_file:={config / "robot.yaml"}',
                    '-p', 'publish_wheel_tf:=false'])
    entered, release = threading.Event(), threading.Event()
    origins = []

    def slow_raycast(origin, directions, walls):
        origins.append(origin.copy())
        if len(origins) == 1:
            entered.set()
            release.wait(2.)
        return np.full(len(directions), 5.)

    monkeypatch.setattr(simulation, 'raycast_segments', slow_raycast)
    if not isolated_scan:
        # Reproduce synchronous raycasting on the control executor under the
        # same load, to ensure this regression exercises worker isolation.
        monkeypatch.setattr(simulation.SyntheticScanNode, '_request_scan',
                            simulation.SyntheticScanNode._publish)
    node = simulation.SyntheticScanNode()
    source = Node('scan_continuity_test')
    publisher = source.create_publisher(Twist, '/cmd_vel', 10)
    samples = []
    source.create_subscription(
        Odometry, '/wheel/odometry',
        lambda msg: samples.append((time.monotonic(), msg.twist.twist.linear.x)), 20)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    executor.add_node(source)
    thread = threading.Thread(target=executor.spin)
    thread.start()
    try:
        assert entered.wait(2.)
        start = time.monotonic()
        command = Twist()
        command.linear.x = .3
        while time.monotonic() - start < .35:
            publisher.publish(command)
            time.sleep(.02)
        moving = [t for t, vx in samples if t >= start and vx > .25]
        if isolated_scan:
            assert len(moving) >= 5, samples
            assert max(np.diff(moving)) < .2
        else:
            assert moving == []
        assert not release.is_set()  # The scan calculation is still blocked.
    finally:
        release.set()
        executor.shutdown()
        thread.join(2.)
        node.destroy_node()
        source.destroy_node()
        rclpy.try_shutdown()
