"""Time-contiguous measurements of simultaneous translation and rotation."""
import numpy as np


def simultaneous_motion_metrics(times, body_velocities, *, speed_m_s=3., yaw_rad_s=.05,
                                max_gap_sec=.1):
    """Count only intervals whose two endpoints satisfy both thresholds.

    Invalid samples, clock reversals and recording gaps break a continuous run.
    A peak or a sum of disconnected bursts is not sustained rotating travel.
    The caller identifies whether input is measured feedback or model truth.
    """
    times = np.asarray(times, dtype=float)
    velocities = np.asarray(body_velocities, dtype=float)
    if times.ndim != 1 or velocities.shape != (len(times), 3):
        raise ValueError('expected N timestamps and N by 3 body velocities')
    if not np.isfinite([speed_m_s, yaw_rad_s, max_gap_sec]).all() or min(
            speed_m_s, yaw_rad_s, max_gap_sec) <= 0.:
        raise ValueError('thresholds must be finite and positive')
    valid = np.isfinite(times) & np.isfinite(velocities).all(axis=1)
    speeds = np.linalg.norm(velocities[:, :2], axis=1)
    rotating = valid & (np.abs(velocities[:, 2]) >= yaw_rad_s)
    qualifying = rotating & (speeds >= speed_m_s)
    longest = total = run = run_distance = longest_distance = 0.
    for i in range(1, len(times)):
        dt = times[i]-times[i-1]
        if qualifying[i-1] and qualifying[i] and 0. < dt <= max_gap_sec:
            run += dt
            total += dt
            run_distance += .5*(speeds[i-1]+speeds[i])*dt
            if run > longest:
                longest, longest_distance = run, run_distance
        else:
            run = run_distance = 0.
    return dict(target_speed_m_s=float(speed_m_s), minimum_yaw_rad_s=float(yaw_rad_s),
                rotating_peak_speed_m_s=float(np.max(speeds[rotating], initial=0.)),
                simultaneous_total_sec=float(total), simultaneous_longest_sec=float(longest),
                simultaneous_longest_distance_m=float(longest_distance))
