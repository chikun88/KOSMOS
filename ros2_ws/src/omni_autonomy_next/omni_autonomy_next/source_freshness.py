"""Validate acquisition timestamps shared by ROS feedback consumers."""
from typing import Optional


def message_stamp_nanoseconds(stamp) -> Optional[int]:
    """Return a positive acquisition time, or None for malformed/zero stamps."""
    try:
        if isinstance(stamp.sec, bool) or isinstance(stamp.nanosec, bool):
            return None
        seconds = int(stamp.sec)
        nanoseconds = int(stamp.nanosec)
    except (AttributeError, TypeError, ValueError, OverflowError):
        return None
    if (seconds != stamp.sec or nanoseconds != stamp.nanosec
            or seconds < 0 or not 0 <= nanoseconds < 1_000_000_000):
        return None
    value = seconds * 1_000_000_000 + nanoseconds
    return value if value > 0 else None
