"""Explicit sensor-frame angle AND range windows for surveyed self returns."""
import numpy as np


def reflection_windows(value):
    """Validate configuration; an omitted list never rejects observations."""
    if not isinstance(value, list):
        raise ValueError('self_reflection_windows must be a list')
    result = []
    for window in value:
        if not isinstance(window, dict):
            raise ValueError('self reflection window must be a mapping')
        try:
            bounds = [float(window[key]) for key in (
                'angle_min_deg', 'angle_max_deg', 'range_min_m', 'range_max_m')]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError('self reflection window needs four numeric bounds') from error
        a, b, near, far = bounds
        if not np.isfinite(bounds).all() or not (-180. <= a < b <= 180. and 0. < near < far):
            raise ValueError('self reflection window has invalid angle/range bounds')
        if not str(window.get('source', '')).strip():
            raise ValueError('self reflection window needs a measurement source')
        result.append(dict(zip(
            ('angle_min_deg', 'angle_max_deg', 'range_min_m', 'range_max_m'), bounds),
            source=str(window['source'])))
    return result


def self_reflection_mask(ranges, angle_min, angle_increment, windows):
    """Keep closer/farther returns, other bearings, and invalid readings intact.

    Angles are LaserScan angles, before applying the sensor mount transform.
    Masked returns must become NaN, never free-space clearing rays (+inf).
    """
    ranges = np.asarray(ranges, dtype=float)
    mask = np.zeros(ranges.shape, dtype=bool)
    if not windows:
        return mask
    angles = np.rad2deg(float(angle_min) + np.arange(ranges.size)*float(angle_increment))
    # Equal physical bearings must agree for clockwise/counterclockwise scan
    # layouts, including a bound hit subject to floating-point roundoff.
    angles = np.round((angles + 180.) % 360. - 180., 9)
    for window in windows:
        mask |= ((angles >= window['angle_min_deg'])
                 & (angles <= window['angle_max_deg'])
                 & (ranges >= window['range_min_m'])
                 & (ranges <= window['range_max_m']))
    return mask
