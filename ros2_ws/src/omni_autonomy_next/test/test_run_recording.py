"""Recorded signals retain sample clocks and use the controller's odometry."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import rclpy
from rclpy.parameter import Parameter
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

ROOT = Path(__file__).resolve().parents[4]
spec = importlib.util.spec_from_file_location('field_recorder', ROOT/'scripts/record_run.py')
recorder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recorder)


def test_recorder_captures_all_control_stages_with_sample_timestamps(tmp_path, monkeypatch):
    rclpy.init()
    node = None
    try:
        node = recorder.Recorder(parameter_overrides=[
            Parameter('log_directory', value=str(tmp_path))])
        topics = {s.topic_name: s.callback for s in node.subscriptions}
        monkeypatch.setattr(recorder.Recorder, 'get_clock',
            lambda self: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=9876543210)))
        twist = Twist()
        twist.linear.x = .2
        commands = ['/cmd_vel_nav', '/cmd_vel_nav_smoothed', '/cmd_vel_rl',
                    '/cmd_vel_collision_safe', '/cmd_vel_safe']
        for topic in commands:
            topics[topic](twist, dict(source_timestamp=9876500000, received_timestamp=9876510000))
        message = Odometry()
        message.header.stamp.sec = 9
        message.header.stamp.nanosec = 123
        message.twist.twist.linear.x = .3
        topics['/wheel/odometry'](message, dict(source_timestamp=9876500000, received_timestamp=9876510000))
        node.close()
        rows = [json.loads(line) for path in sorted(node.directory.glob('samples-*.jsonl'))
                for line in path.read_text().splitlines()]
        assert [r['topic'] for r in rows] == commands + ['/wheel/odometry']
        assert all(r['published_unix_ns'] == 9876500000 for r in rows)
        assert all(r['dds_received_unix_ns'] == 9876510000 for r in rows)
        assert rows[-1]['source_ros_ns'] == 9000000123
        assert all(r['received_ros_ns'] == 9876543210 for r in rows)
        assert rows[-1]['monotonic_ns'] >= rows[0]['monotonic_ns']
        assert [r['sequence'] for r in rows] == list(range(1, 7))
        assert rows[-1]['value']['twist']['twist']['linear']['x'] == .3
        assert '/odom' not in topics
        assert '/trajectory_tracker/status' in topics and '/system/safety_state' in topics
    finally:
        if node is not None:
            node.close()
            node.destroy_node()
        rclpy.try_shutdown()
