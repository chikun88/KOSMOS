"""Storage failure isolation, actual DDS subscriptions, and offline pairing."""
import importlib.util
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import pytest
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, DurabilityPolicy, qos_profile_sensor_data
from geometry_msgs.msg import Twist, PoseStamped, PolygonStamped, Point32
from nav2_msgs.msg import CollisionMonitorState
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

from omni_autonomy_next import run_recorder_node as recording

ROOT = Path(__file__).resolve().parents[4]
spec = importlib.util.spec_from_file_location('analyze_run', ROOT/'scripts/analyze_run.py')
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def read_rows(directory):
    return [json.loads(line) for f in sorted(directory.glob('samples-*.jsonl'))
            for line in f.read_text().splitlines()]


def row(n):
    return dict(schema=2, topic='/test', monotonic_ns=n, value={'number': n})


def test_rotation_and_shutdown_drain(tmp_path):
    writer = recording.RunWriter(tmp_path/'run', segment_bytes=170, max_segments=20, reserve_bytes=0)
    for i in range(10):
        writer.write(row(i))
    writer.close()
    assert [r['value']['number'] for r in read_rows(writer.directory)] == list(range(10))
    assert len(list(writer.directory.glob('samples-*.jsonl'))) > 1
    summary = json.loads((writer.directory/'recording-summary.json').read_text())
    assert summary['closed'] and summary['written'] == 10
    assert summary['dropped'] == 0 and summary['pending'] == 0 and summary['error'] is None
    assert all(f.stat().st_size <= 170 for f in writer.directory.glob('samples-*.jsonl'))


def test_size_limit_preserves_earlier_data(tmp_path):
    writer = recording.RunWriter(tmp_path/'run', segment_bytes=85, max_segments=1, reserve_bytes=0)
    for i in range(5):
        writer.write(row(i))
    writer.close()
    assert 'size limit' in writer.error
    assert len(read_rows(writer.directory)) == 1
    writer.write(row(6))
    assert writer.dropped == 1


def test_disk_reserve_stops_without_affecting_caller(tmp_path, monkeypatch):
    monkeypatch.setattr(recording.shutil, 'disk_usage', lambda _: SimpleNamespace(free=5))
    writer = recording.RunWriter(tmp_path/'run', reserve_bytes=10)
    writer.write(row(1))
    writer.close()
    assert 'reserve' in writer.error
    assert writer.written == 0


def test_full_queue_drops_without_blocking(tmp_path, monkeypatch):
    gate = threading.Event()
    original = recording.RunWriter._run
    def delayed(self):
        gate.wait(3.)
        original(self)
    monkeypatch.setattr(recording.RunWriter, '_run', delayed)
    writer = recording.RunWriter(tmp_path/'run', queue_size=1, reserve_bytes=0)
    try:
        writer.write(row(1))
        writer.write(row(2))
        assert writer.dropped == 1
    finally:
        gate.set()
        writer.close()
    assert writer.written == 1


def test_ros_capture_durable_goals_sensors_and_parameter_snapshot(tmp_path):
    rclpy.init()
    publisher = Node('recorder_test_source')
    executor = SingleThreadedExecutor()
    executor.add_node(publisher)
    recorder = None
    try:
        durable = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        goal_pub = publisher.create_publisher(PoseStamped, '/navigation/active_goal', durable)
        goal_pub.publish(PoseStamped())  # Must be captured by the late subscriber.
        odom_pub = publisher.create_publisher(Odometry, '/wheel/odometry', qos_profile_sensor_data)
        cmd_pub = publisher.create_publisher(Twist, '/cmd_vel_safe', 10)
        scan_pub = publisher.create_publisher(LaserScan, '/scan_front', qos_profile_sensor_data)
        filtered_pub = publisher.create_publisher(
            LaserScan, '/scan_front_filtered', qos_profile_sensor_data)
        collision_pub = publisher.create_publisher(
            CollisionMonitorState, '/collision_monitor/state', 10)
        footprint_pub = publisher.create_publisher(
            PolygonStamped, '/local_costmap/published_footprint', 10)
        publisher.declare_parameter('test_gain', 1.75)
        config = tmp_path/'robot.yaml'
        config.write_text('gain: 1.75\n')
        recorder = recording.Recorder(parameter_overrides=[
            Parameter('log_directory', value=str(tmp_path)),
            Parameter('config_files', value=[str(config)]),
            Parameter('operation_mode', value='demo')])
        executor.add_node(recorder)
        scan = LaserScan(ranges=[1., float('inf'), float('nan'), float('-inf')])
        odom = Odometry()
        odom.header.stamp.sec = 42
        odom.header.frame_id = 'odom'
        odom.pose.covariance[0] = .23
        command = Twist()
        command.linear.x = .4
        collision = CollisionMonitorState(action_type=3, polygon_name='FootprintApproach')
        footprint = PolygonStamped()
        footprint.header.frame_id = 'odom'
        footprint.header.stamp.sec = 43
        footprint.polygon.points = [Point32(x=.4, y=.3), Point32(x=-.4, y=.3),
                                    Point32(x=0., y=-.3)]
        until = time.monotonic()+4.
        while time.monotonic() < until:
            cmd_pub.publish(command)
            odom_pub.publish(odom)
            scan_pub.publish(scan)
            filtered_pub.publish(scan)
            collision_pub.publish(collision)
            footprint_pub.publish(footprint)
            recorder._snapshot_parameters()
            for _ in range(20):
                executor.spin_once(timeout_sec=.01)
            if (recorder.counts['/navigation/active_goal'] and recorder.counts['/scan_front']
                    and recorder.counts['/scan_front_filtered']
                    and recorder.counts['/collision_monitor/state']
                    and recorder.counts['/local_costmap/published_footprint']
                    and '/recorder_test_source' in recorder.parameter_snapshot_times):
                break
        recorder.close()
        rows = read_rows(recorder.directory)
        topics = {r['topic'] for r in rows}
        assert {'/navigation/active_goal', '/wheel/odometry', '/cmd_vel_safe',
                '/scan_front', 'recorder/parameter_snapshot'} <= topics
        sample = next(r for r in rows if r['topic'] == '/wheel/odometry')
        assert sample['published_unix_ns'] > 0
        assert sample['dds_received_unix_ns'] >= sample['published_unix_ns']
        assert sample['source_ros_ns'] == 42000000000
        assert sample['value']['pose']['covariance'][0] == .23
        assert sample['value']['header']['frame_id'] == 'odom'
        sample = next(r for r in rows if r['topic'] == '/scan_front')
        assert sample['value']['ranges'] == [1., 'Infinity', 'NaN', '-Infinity']
        filtered = next(r for r in rows if r['topic'] == '/scan_front_filtered')
        assert filtered['value']['ranges'] == sample['value']['ranges']
        trigger = next(r for r in rows if r['topic'] == '/collision_monitor/state')
        assert trigger['value'] == {'action_type': 3, 'polygon_name': 'FootprintApproach'}
        outline = next(r for r in rows if r['topic'] == '/local_costmap/published_footprint')
        assert outline['source_ros_ns'] == 43_000_000_000
        assert outline['value']['header']['frame_id'] == 'odom'
        assert len(outline['value']['polygon']['points']) == 3
        snapshots = [r for r in rows if r['topic'] == 'recorder/parameter_snapshot']
        params = next(r['value']['parameters'] for r in snapshots
                      if r['value']['node'] == '/recorder_test_source')
        assert params['test_gain']['double_value'] == 1.75
        manifest = json.loads((recorder.directory/'manifest.json').read_text())
        assert manifest['settings']['operation_mode'] == 'demo'
        assert manifest['configs'][0]['text'] == 'gain: 1.75\n'
        assert manifest['configs'][0]['sha256']
        # The recorder has no command publishers.
        assert not any(p.topic_name.startswith('/cmd_vel') for p in recorder.publishers)
    finally:
        executor.shutdown()
        if recorder is not None:
            recorder.close()
            recorder.destroy_node()
        publisher.destroy_node()
        rclpy.try_shutdown()


def test_offline_pairing_rejects_stale_and_future_commands(tmp_path):
    (tmp_path/'manifest.json').write_text(json.dumps(dict(session_id='test', settings=dict(operation_mode='hardware'))))
    def velocity(v):
        return {'linear': {'x': v, 'y': 0.}, 'angular': {'z': 0.}}
    def sample(topic, ns, value):
        return dict(schema=2, topic=topic, monotonic_ns=ns, value=value)
    rows = [sample('/wheel/odometry', 0, {'twist': {'twist': velocity(.1)}}),
            sample('/cmd_vel_safe', 100000000, velocity(.4)),
            sample('/wheel/odometry', 200000000, {'twist': {'twist': velocity(.3)}}),
            sample('/wheel/odometry', 500000000, {'twist': {'twist': velocity(.3)}})]
    (tmp_path/'samples-0000.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows)+'{"schema":')
    result = analysis.analyze(tmp_path)
    assert result['paired_odometry_samples'] == 1
    assert result['unpaired_odometry_samples'] == 2
    assert result['velocity_rmse']['vx_m_s'] == pytest.approx(.1)
    assert result['topics']['recorder/truncated_line'] == 1
    assert (tmp_path/'control.csv').exists()
    assert result['receipt_alignment_warning'] is None
    rows[2].update(source_ros_ns=200000000, received_ros_ns=600000000)
    (tmp_path/'samples-0000.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    delayed = analysis.analyze(tmp_path)
    assert delayed['max_source_to_callback_age_s']['/wheel/odometry'] == pytest.approx(.4)
    assert 'Do not infer drive calibration' in delayed['receipt_alignment_warning']


def test_default_recorder_constructor(tmp_path):
    rclpy.init()
    node = None
    try:
        node = recording.Recorder(parameter_overrides=[Parameter('log_directory', value=str(tmp_path))])
        assert node.directory.exists()
    finally:
        if node:
            node.close()
            node.destroy_node()
        rclpy.try_shutdown()
