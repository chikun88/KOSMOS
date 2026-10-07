import math
from typing import Optional, Sequence

from .source_freshness import message_stamp_nanoseconds


def timestamp_is_fresh(
    now_ns: int,
    stamp_ns: int,
    timeout_ns: int,
    *,
    future_tolerance_ns: int = 0,
) -> bool:
    """Reject old data and clock rollbacks instead of renewing its lease."""
    age_ns = int(now_ns) - int(stamp_ns)
    return -int(future_tolerance_ns) <= age_ns <= int(timeout_ns)


def scan_metadata_valid(message) -> bool:
    """Validate scan geometry/timing before using its rays or heartbeat.

    Infinite *ranges* are legitimate no-return observations. Non-finite
    geometry or timing cannot describe a usable scan, including an all-inf one.
    """
    values = (
        message.angle_min, message.angle_increment,
        message.range_min, message.range_max,
        message.time_increment, message.scan_time,
    )
    if not all(math.isfinite(float(value)) for value in values):
        return False
    return bool(
        len(message.ranges) > 0
        and (len(message.ranges) == 1 or float(message.angle_increment) != 0.0)
        and 0.0 <= float(message.range_min) < float(message.range_max)
        and float(message.time_increment) >= 0.0
        and float(message.scan_time) >= 0.0
        and message_stamp_nanoseconds(message.header.stamp) is not None
    )


def scan_generations_changed(
    current: Sequence[int],
    previous: Optional[Sequence[int]],
) -> bool:
    """Return whether at least one LiDAR supplied a new usable scan."""
    return previous is None or tuple(current) != tuple(previous)
