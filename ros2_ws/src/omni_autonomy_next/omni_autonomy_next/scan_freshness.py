from typing import Optional, Sequence


def scan_generations_changed(
    current: Sequence[int],
    previous: Optional[Sequence[int]],
) -> bool:
    """Return whether at least one LiDAR supplied a new usable scan."""
    return previous is None or tuple(current) != tuple(previous)
