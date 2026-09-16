import numpy as np
import pytest
from omni_autonomy_next.motion_metrics import simultaneous_motion_metrics


def test_disconnected_peaks_do_not_count_as_continuous_motion():
    times = np.arange(9)*.05
    velocity = np.tile([3.1, 0., .1], (9, 1))
    velocity[3:5, 2] = 0.
    result = simultaneous_motion_metrics(times, velocity)
    assert result['simultaneous_longest_sec'] == pytest.approx(.15)
    assert result['simultaneous_total_sec'] == pytest.approx(.25)
    assert result['simultaneous_longest_distance_m'] == pytest.approx(.465)


@pytest.mark.parametrize('axis,sign', [(0,1),(0,-1),(1,1),(1,-1)])
def test_reverse_and_lateral_translation_with_either_rotation_direction(axis, sign):
    velocity = np.zeros((31,3))
    velocity[:,axis] = sign*3.
    velocity[:,2] = sign*.1
    result = simultaneous_motion_metrics(np.arange(31)*.05, velocity)
    assert result['simultaneous_longest_sec'] == pytest.approx(1.5)
    assert result['simultaneous_longest_distance_m'] == pytest.approx(4.5)


@pytest.mark.parametrize('times', [[0.,.05,.4,.45], [0.,.05,.01,.06], [0.,.05,np.nan,.15]])
def test_gaps_and_clock_errors_break_runs(times):
    result = simultaneous_motion_metrics(times, np.tile([3.,0.,.1], (4,1)))
    assert result['simultaneous_longest_sec'] == pytest.approx(.05)


def test_bad_velocity_and_stopped_translation_are_not_success():
    result = simultaneous_motion_metrics([0.,.05,.1,.15],
        [[0.,0.,1.],[float('nan'),0.,1.],[4.,0.,0.],[0.,0.,0.]])
    assert result['simultaneous_total_sec'] == 0.
    assert simultaneous_motion_metrics([],np.empty((0,3)))['rotating_peak_speed_m_s'] == 0.


@pytest.mark.parametrize('kwargs', [{'speed_m_s':0.},{'yaw_rad_s':float('nan')},{'max_gap_sec':-1.}])
def test_invalid_thresholds_are_rejected(kwargs):
    with pytest.raises(ValueError):
        simultaneous_motion_metrics([],np.empty((0,3)),**kwargs)
