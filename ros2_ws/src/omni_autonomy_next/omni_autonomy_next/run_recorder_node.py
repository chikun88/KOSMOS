"""Read-only, bounded recording of the control chain in an independent process."""
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import shutil
import signal
import subprocess
import threading
import time
import uuid
from collections import Counter, deque

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.parameter_client import AsyncParameterClient
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from rcl_interfaces.msg import ParameterEvent, Log
from geometry_msgs.msg import PolygonStamped, PoseStamped, PoseWithCovarianceStamped, Twist
from nav_msgs.msg import Odometry, Path as NavPath
from nav2_msgs.msg import CollisionMonitorState, SpeedLimit
from diagnostic_msgs.msg import DiagnosticArray
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String, Float32, Bool, Float64MultiArray
from tf2_msgs.msg import TFMessage
from rosidl_runtime_py.convert import message_to_ordereddict


def yaw_of(q):
    return math.atan2(2.*(q.w*q.z+q.x*q.y), 1.-2.*(q.y*q.y+q.z*q.z))


def json_safe(value):
    """Keep scan no-return/invalid distinctions in standards-compliant JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return 'NaN' if math.isnan(value) else ('Infinity' if value > 0 else '-Infinity')
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def save_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(json_safe(value), ensure_ascii=False,
                                    indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


class RunWriter:
    """Bound memory/disk use; serialization and filesystem I/O stay off callbacks."""
    def __init__(self, directory, segment_bytes=64*1024*1024,
                 max_segments=32, queue_size=4096, reserve_bytes=512*1024*1024):
        if min(segment_bytes, max_segments, queue_size) <= 0 or reserve_bytes < 0:
            raise ValueError('invalid recording limits')
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.segment_bytes = segment_bytes
        self.max_segments = max_segments
        self.reserve_bytes = reserve_bytes
        self.queue = queue.Queue(maxsize=queue_size)
        self.dropped = 0
        self.error = None
        self.written = 0
        self.bytes_written = 0
        self.topic_counts = Counter()
        self.closed = False
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def write(self, row):
        if self.closed or self.error:
            self.dropped += 1
            return
        try:
            self.queue.put_nowait(row)
        except queue.Full:
            self.dropped += 1

    def _run(self):
        stream = None
        try:
            self._summary(False)
            index, size = 0, 0
            last_flush = time.monotonic()
            while not self.stop.is_set() or not self.queue.empty():
                try:
                    row = self.queue.get(timeout=.2)
                except queue.Empty:
                    row = None
                if row is not None:
                    data = dict(row)
                    if 'message' in data:
                        data['value'] = message_to_ordereddict(data.pop('message'))
                    line = (json.dumps(json_safe(data), separators=(',', ':'),
                                       ensure_ascii=False, allow_nan=False)+'\n').encode('utf-8')
                    if len(line) > self.segment_bytes:
                        self.dropped += 1
                        continue
                    if stream is None or size + len(line) > self.segment_bytes:
                        if stream is not None:
                            stream.flush()
                            os.fsync(stream.fileno())
                            stream.close()
                        if index >= self.max_segments:
                            raise OSError('session size limit reached; recording stopped')
                        stream = (self.directory / f'samples-{index:04d}.jsonl').open('xb')
                        index += 1
                        size = 0
                    # Check every record, including writes within a segment.
                    if shutil.disk_usage(self.directory).free < self.reserve_bytes + len(line):
                        raise OSError('disk free space reserve reached; recording stopped')
                    stream.write(line)
                    size += len(line)
                    self.bytes_written += len(line)
                    self.written += 1
                    self.topic_counts[data['topic']] += 1
                if stream is not None and time.monotonic()-last_flush >= 1.:
                    stream.flush()
                    os.fsync(stream.fileno())
                    self._summary(False)
                    last_flush = time.monotonic()
        except Exception as error:
            self.error = f'{type(error).__name__}: {error}'
        finally:
            if stream is not None and not stream.closed:
                try:
                    stream.flush()
                    os.fsync(stream.fileno())
                    stream.close()
                except OSError as error:
                    self.error = str(error)
            self._summary(self.stop.is_set())

    def _summary(self, closed):
        try:
            save_json(self.directory/'recording-summary.json', dict(
                schema=2, closed=closed, updated_utc=datetime.datetime.now(
                    datetime.timezone.utc).isoformat(), written=self.written,
                dropped=self.dropped, pending=self.queue.qsize(), error=self.error,
                bytes_written=self.bytes_written, topics=dict(self.topic_counts)))
        except OSError as error:
            self.error = str(error)

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.stop.set()
        self.thread.join(timeout=5.)
        if self.thread.is_alive():
            self.error = 'writer did not drain within shutdown deadline'
        else:
            self._summary(True)


# (type, topics, transient-local). High-rate streams use best-effort readers.
TOPIC_GROUPS = [
    (Twist, ['/cmd_vel_nav', '/cmd_vel_nav_smoothed', '/cmd_vel_rl',
             '/cmd_vel_collision_safe', '/cmd_vel_safe', '/cmd_vel'], False),
    (Odometry, ['/wheel/odometry'], False),
    (PoseWithCovarianceStamped, ['/localization/pose', '/initialpose'], False),
    (Float64MultiArray, ['/wheel/counts', '/wheel/delta_counts'], False),
    (NavPath, ['/plan'], False),
    (PoseStamped, ['/navigation/active_goal'], True),
    (PoseStamped, ['/goal_request'], False),
    (String, ['/trajectory_tracker/status', '/system/safety_state', '/rl/state',
              '/navigation/goal_status', '/navigation/remembered_poses',
              '/navigation/motion_mode', '/navigation/field_side', '/system/profile',
              '/motor/network_status', '/motor/telemetry', '/mu3/navigation_status'], True),
    (String, ['/wheel/status', '/navigation/goal_id_request',
              '/navigation/remembered_goal_request', '/navigation/remember_pose_request',
              '/mu3/navigation_input'], False),
    (Bool, ['/system/armed', '/system/emergency_stop', '/rl/healthy',
            '/motor/auto_engaged', '/motor/link_ok'], True),
    (Bool, ['/localization/tracking_ok', '/navigation/cancel_request',
            '/motion/enable', '/motion/emergency_stop'], False),
    (Float32, ['/system/speed_scale', '/rl/speed_scale'], True),
    (Float32, ['/motor/link_latency_ms'], False),
    (SpeedLimit, ['/speed_limit'], True),
    (CollisionMonitorState, ['/collision_monitor/state'], False),
    (PolygonStamped, ['/local_costmap/published_footprint'], False),
    (DiagnosticArray, ['/diagnostics'], False),
    (TFMessage, ['/tf'], False),
    (TFMessage, ['/tf_static'], True),
    (ParameterEvent, ['/parameter_events'], False),
    (Log, ['/rosout'], False),
]


class Recorder(Node):
    def __init__(self, **kwargs):
        super().__init__('run_recorder', **kwargs)
        defaults = dict(log_directory=str(Path.home()/'.ros/omni_autonomy_next/runs'),
                        operation_mode='unknown', demo=False, motors=False, wheels=False,
                        lidars=False, motion_mode='unknown', tracker=False,
                        segment_mb=64, max_segments=32, queue_size=4096, reserve_mb=512)
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        self.declare_parameter('config_files', Parameter.Type.STRING_ARRAY)
        self.declare_parameter('scan_topics', ['/scan_front', '/scan_rear'])
        self.declare_parameter('collision_scan_topics',
                               ['/scan_front_filtered', '/scan_rear_filtered'])
        p = lambda name: self.get_parameter(name).value
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
        self.directory = Path(p('log_directory')).expanduser()/(stamp+'-'+uuid.uuid4().hex[:8])
        self.writer = RunWriter(self.directory, int(p('segment_mb'))*1024*1024,
                                int(p('max_segments')), int(p('queue_size')),
                                int(p('reserve_mb'))*1024*1024)
        self.cmd, self.odom, self.pose = (deque(maxlen=100000) for _ in range(3))
        self.counts = Counter()
        self.sequence = 0
        self.parameter_clients = {}
        self.parameter_futures = {}
        self.parameter_snapshot_times = {}
        self.last_warning = None
        try:
            metadata = dict(schema=2, session_id=self.directory.name,
                            start_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                            start_monotonic_ns=time.monotonic_ns(),
                            start_ros_ns=self.get_clock().now().nanoseconds,
                            settings={k: p(k) for k in defaults},
                            ros_distro=os.environ.get('ROS_DISTRO', ''),
                            ros_domain_id=os.environ.get('ROS_DOMAIN_ID', '0'), configs=[])
            for filename in self.get_parameter_or('config_files').value or []:
                entry = dict(path=filename)
                try:
                    raw = Path(filename).read_bytes()
                    entry.update(sha256=hashlib.sha256(raw).hexdigest(), text=raw.decode('utf-8'))
                except (OSError, UnicodeError) as error:
                    entry['error'] = str(error)
                metadata['configs'].append(entry)
            source = Path(__file__).resolve().parent
            metadata['source_hashes'] = {f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                                         for f in sorted(source.glob('*.py'))}
            try:
                metadata['git_revision'] = subprocess.check_output(
                    ['git', '-C', str(source), 'rev-parse', 'HEAD'],
                    stderr=subprocess.DEVNULL, timeout=2, text=True).strip()
                metadata['git_status'] = subprocess.check_output(
                    ['git', '-C', str(source), 'status', '--porcelain'],
                    stderr=subprocess.DEVNULL, timeout=2, text=True)
            except (OSError, subprocess.SubprocessError):
                metadata['git_revision'] = None
            scan_topics = list(dict.fromkeys(
                list(p('scan_topics')) + list(p('collision_scan_topics'))))
            groups = TOPIC_GROUPS + [(LaserScan, scan_topics, False)]
            metadata['topics'] = {}
            for msg_type, topics, durable in groups:
                qos = QoSProfile(depth=100 if msg_type is not LaserScan else 5,
                                 # Reliable is required for historical delivery of latched state.
                                 reliability=(ReliabilityPolicy.RELIABLE if durable
                                              else ReliabilityPolicy.BEST_EFFORT),
                                 durability=(DurabilityPolicy.TRANSIENT_LOCAL if durable
                                             else DurabilityPolicy.VOLATILE))
                for topic in topics:
                    type_name = msg_type.__module__.split('.')[0]+'/msg/'+msg_type.__name__
                    metadata['topics'][topic] = dict(type=type_name, transient_local=durable,
                                                    reliability='reliable' if durable else 'best_effort')
                    self.create_subscription(msg_type, topic,
                        lambda m, info, t=topic, mt=type_name: self._capture(t, mt, m, info), qos)
            save_json(self.directory/'manifest.json', metadata)
            self.create_timer(5., self._health)
            self.create_timer(10., self._snapshot_parameters)
            self.get_logger().info(f'Run recording: {self.directory}')
        except Exception:
            self.writer.close()
            self.destroy_node()
            raise

    def _capture(self, topic, msg_type, message, info=None):
        mono = time.monotonic_ns()
        self.sequence += 1
        self.counts[topic] += 1
        row = dict(schema=2, sequence=self.sequence, topic=topic, type=msg_type,
                   monotonic_ns=mono, received_ros_ns=self.get_clock().now().nanoseconds,
                   received_unix_ns=time.time_ns(), message=message)
        if info is not None:
            # Callback receipt can lag seconds under load. DDS source time
            # also timestamps headerless Twist/String messages at publication.
            row['published_unix_ns'] = info['source_timestamp']
            row['dds_received_unix_ns'] = info['received_timestamp']
        if hasattr(message, 'header'):
            stamp = message.header.stamp
            row['source_ros_ns'] = stamp.sec*1000000000+stamp.nanosec
        self.writer.write(row)
        if topic == '/cmd_vel_safe':
            self.cmd.append((message.linear.x, message.linear.y, message.angular.z))
        elif topic == '/wheel/odometry':
            v = message.twist.twist
            self.odom.append((v.linear.x, v.linear.y, v.angular.z))
        elif topic == '/localization/pose':
            p = message.pose.pose
            self.pose.append((p.position.x, p.position.y, yaw_of(p.orientation)))

    def _health(self):
        state = (self.writer.error, self.writer.dropped)
        if state != self.last_warning and (state[0] or state[1]):
            self.get_logger().error(f'Run recording incomplete: error={state[0]}, dropped={state[1]}')
        self.last_warning = state

    def _snapshot_parameters(self):
        """Periodically read effective parameters, including nodes started later."""
        if self.writer.error:
            return
        now = time.monotonic()
        for name, namespace in self.get_node_names_and_namespaces():
            remote = namespace.rstrip('/')+'/'+name
            if remote == self.get_fully_qualified_name():
                continue
            if remote in self.parameter_futures:
                future, started = self.parameter_futures[remote]
                if now - started > 10.:
                    future.cancel()
                    del self.parameter_futures[remote]
                else:
                    continue
            if now - self.parameter_snapshot_times.get(remote, -math.inf) < 60.:
                continue
            client = self.parameter_clients.get(remote)
            if client is None:
                client = AsyncParameterClient(self, remote)
                self.parameter_clients[remote] = client
            if not client.services_are_ready():
                continue
            future = client.list_parameters()
            self.parameter_futures[remote] = (future, now)
            future.add_done_callback(lambda f, r=remote: self._parameter_names(r, f))

    def _parameter_names(self, remote, future):
        if future.cancelled() or future.exception():
            return
        names = future.result().result.names
        read = self.parameter_clients[remote].get_parameters(names)
        self.parameter_futures[remote] = (read, time.monotonic())
        read.add_done_callback(lambda f, r=remote, n=names: self._parameter_values(r, n, f))

    def _parameter_values(self, remote, names, future):
        if future.cancelled() or future.exception():
            return
        self.parameter_futures.pop(remote, None)
        self.parameter_snapshot_times[remote] = time.monotonic()
        self.writer.write(dict(schema=2, topic='recorder/parameter_snapshot',
            monotonic_ns=time.monotonic_ns(), received_ros_ns=self.get_clock().now().nanoseconds,
            value=dict(node=remote, parameters={name: message_to_ordereddict(value)
                for name, value in zip(names, future.result().values)})))

    def close(self):
        self.writer.close()
        self._health()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = Recorder()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        # launch may forward SIGINT after the terminal already sent it to us.
        # Let the writer finish draining even when shutdown is requested twice.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        if node is not None:
            node.close()
            node.destroy_node()
        rclpy.try_shutdown()
