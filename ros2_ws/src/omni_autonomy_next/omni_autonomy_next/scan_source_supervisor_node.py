"""Keep autonomy available when a LiDAR dies, without blinding the robot.

Nav2's collision monitor holds the robot at zero velocity for as long as any
enabled observation source has no fresh data ("Robot to stop due to invalid
source").  That is the right default -- a stale LiDAR must not be mistaken for
a clear path -- but it means a single broken unit takes autonomy away entirely
rather than degrading it.  The rear A2M8 on this robot did exactly that.

So this node watches every filtered scan the monitor consumes and switches only
the silent sources off, leaving the live ones braking as usual.  Two properties
make that safe:

* At least one source always stays enabled.  If every LiDAR goes quiet the
  sources are left as they are, so the monitor still stops the robot -- driving
  with no obstacle input at all is exactly what must not happen.
* A source is re-enabled as soon as its scans come back, so a LiDAR that is
  re-plugged mid-session rejoins the safety layer without a restart.

Deciding this at launch time instead would freeze a one-off snapshot taken while
the USB bus may be mid-enumeration, which on this hardware is usually wrong.
"""

import math
import time
from typing import Dict, Set

import rclpy
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
import yaml

from .scan_freshness import scan_metadata_valid, timestamp_is_fresh


def sources_to_change(
    enabled: Dict[str, bool],
    live: Set[str],
    pending: Set[str] = frozenset(),
) -> Dict[str, bool]:
    """Return the source-enable changes to request, if any.

    A source should be enabled exactly when its LiDAR is publishing -- except
    when nothing is publishing at all, where the answer is to change nothing so
    the collision monitor keeps holding the robot.
    """
    # SetParameters is not an atomic transaction and replies can arrive in a
    # different order from submissions. A LiDAR handoff must first confirm the
    # replacement enable, then disable the old input in a later request. Never
    # overlap requests: a delayed disable could otherwise turn off the last
    # input just after a replacement request has been submitted.
    live = live.intersection(enabled)
    if pending:
        return {}
    if not live:
        # Recover a fail-open state conservatively if the monitor was started
        # with every source disabled. An enabled stale source makes it stop.
        if enabled and not any(enabled.values()):
            return {source: True for source in enabled}
        return {}
    enables = {
        source: True for source, is_enabled in enabled.items()
        if source in live and not is_enabled
    }
    if enables:
        return enables
    return {
        source: source in live
        for source, is_enabled in enabled.items()
        if is_enabled != (source in live)
    }


class ScanSourceSupervisor(Node):
    def __init__(self) -> None:
        super().__init__('scan_source_supervisor')
        self.declare_parameter('robot_config_file', '')
        self.declare_parameter('collision_monitor_node', '/collision_monitor')
        self.declare_parameter('output_topic_suffix', '_filtered')
        # Must exceed the monitor's own source_timeout so the monitor is never
        # the first to notice; otherwise it brakes before the source is off.
        self.declare_parameter('scan_timeout_sec', 0.6)
        self.declare_parameter('check_rate_hz', 4.0)
        self.declare_parameter('service_timeout_sec', 5.0)

        robot_file = str(self.get_parameter('robot_config_file').value)
        if not robot_file:
            raise ValueError('robot_config_file is required')
        with open(robot_file, encoding='utf-8') as stream:
            robot = yaml.safe_load(stream)['robot']

        suffix = str(self.get_parameter('output_topic_suffix').value)
        self.scan_timeout_sec = float(self.get_parameter('scan_timeout_sec').value)
        if not 0.0 < self.scan_timeout_sec < float('inf'):
            raise ValueError('scan_timeout_sec must be finite and positive')
        self.service_timeout_sec = float(self.get_parameter('service_timeout_sec').value)
        if not math.isfinite(self.service_timeout_sec) or self.service_timeout_sec <= 0.0:
            raise ValueError('service_timeout_sec must be finite and positive')
        # The monitor names each source after the topic's LiDAR, matching
        # nav2_next.yaml's observation_sources entries.
        self.sources: Dict[str, str] = {}
        self.last_scan_sec: Dict[str, float] = {}
        self.last_scan_stamp_ns: Dict[str, int] = {}
        for lidar in robot['lidars']:
            name = str(lidar['name'])
            source = f'scan_{name}'
            self.sources[source] = f'{lidar["topic"]}{suffix}'
            self.create_subscription(
                LaserScan,
                self.sources[source],
                lambda message, key=source: self._scan_callback(key, message),
                qos_profile_sensor_data,
            )
        self.enabled: Dict[str, bool] = {source: True for source in self.sources}
        self.pending: Set[str] = set()
        self.pending_started = None

        monitor = str(self.get_parameter('collision_monitor_node').value).rstrip('/')
        self.client = self.create_client(SetParameters, f'{monitor}/set_parameters')
        self.create_timer(
            1.0 / max(0.5, float(self.get_parameter('check_rate_hz').value)),
            self._review,
        )
        self.get_logger().info(
            'Scan source supervisor watching '
            + ', '.join(f'{source} <- {topic}' for source, topic in self.sources.items())
            + f' (timeout {self.scan_timeout_sec:.2f} s)'
        )

    def _scan_callback(self, source: str, message: LaserScan) -> None:
        if not scan_metadata_valid(message):
            return
        now = self._now_sec()
        stamp_ns = (
            int(message.header.stamp.sec) * 1_000_000_000
            + int(message.header.stamp.nanosec)
        )
        if stamp_ns:
            if not timestamp_is_fresh(
                round(now * 1.0e9), stamp_ns,
                round(self.scan_timeout_sec * 1.0e9),
                future_tolerance_ns=50_000_000,
            ):
                return
            previous = self.last_scan_stamp_ns.get(source)
            if previous is not None and stamp_ns <= previous:
                # Identical replayed scans must not keep a failed LiDAR alive.
                return
            self.last_scan_stamp_ns[source] = stamp_ns
        self.last_scan_sec[source] = min(now, stamp_ns * 1.0e-9) if stamp_ns else now

    def _now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1.0e-9

    def _review(self) -> None:
        now = self._now_sec()
        if (self.pending and self.pending_started is not None
                and time.monotonic() - self.pending_started > self.service_timeout_sec):
            # Canceling a client future does not cancel an in-flight server
            # mutation. Keep the handoff blocked until its acknowledgement;
            # retrying new disables could race a late reply and blind the robot.
            self.get_logger().warning(
                'Collision-source parameter update has no acknowledgement; '
                'holding source handoffs to preserve the enabled safety input',
                throttle_duration_sec=5.0,
            )
        live = {
            source
            for source in self.sources
            if 0.0 <= now - self.last_scan_sec.get(source, -1.0e9)
            <= self.scan_timeout_sec
        }
        if not live:
            self.get_logger().warning(
                'No LiDAR is publishing; leaving every collision-monitor source '
                'as it is so the monitor keeps the robot stopped',
                throttle_duration_sec=5.0,
            )
        changes = sources_to_change(self.enabled, live, self.pending)
        if changes:
            self._apply(changes)

    def _apply(self, changes: Dict[str, bool]) -> None:
        if not self.client.service_is_ready():
            self.get_logger().warning(
                f'{self.client.srv_name} is not available yet; retrying',
                throttle_duration_sec=5.0,
            )
            return
        request = SetParameters.Request()
        request.parameters = [
            Parameter(
                name=f'{source}.enabled',
                value=ParameterValue(
                    type=ParameterType.PARAMETER_BOOL, bool_value=enabled
                ),
            )
            for source, enabled in changes.items()
        ]
        self.pending.update(changes)
        self.pending_started = time.monotonic()
        try:
            future = self.client.call_async(request)
        except Exception as error:  # noqa: BLE001 - a failed submission must retry
            self.pending.difference_update(changes)
            self.pending_started = None
            self.get_logger().error(f'Failed to submit collision sources: {error}')
            return
        future.add_done_callback(
            lambda done, changes=dict(changes): self._applied(done, changes)
        )

    def _applied(self, future, changes: Dict[str, bool]) -> None:
        self.pending.difference_update(changes)
        self.pending_started = None
        try:
            response = future.result()
        except Exception as error:  # noqa: BLE001 - a failed call must only retry
            self.get_logger().error(f'Failed to update collision sources: {error}')
            return
        results = list(response.results) if response is not None else []
        for index, (source, enabled) in enumerate(changes.items()):
            ok = index < len(results) and results[index].successful
            if not ok:
                reason = results[index].reason if index < len(results) else 'no result'
                self.get_logger().error(
                    f'Could not set {source}.enabled={enabled}: {reason}'
                )
                continue
            self.enabled[source] = enabled
            if enabled:
                self.get_logger().info(
                    f'{source} is publishing again; re-enabled it as a '
                    'collision-monitor source'
                )
            else:
                self.get_logger().warning(
                    f'{self.sources[source]} went silent; disabled {source} so '
                    'autonomy continues on the remaining LiDAR(s)'
                )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = ScanSourceSupervisor()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
