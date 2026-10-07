"""Bound self-return rejection without hiding neighbouring obstacle returns."""
import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from sensor_msgs.msg import LaserScan

from omni_autonomy_next.config import load_robot
from omni_autonomy_next.scan_self_reflections import reflection_windows, self_reflection_mask
from omni_autonomy_next.scan_footprint_filter_node import ScanFootprintFilter

ROOT = Path(__file__).resolve().parents[4]
CONFIG = ROOT/'ros2_ws/src/omni_autonomy_next/config/robot.yaml'


def windows():
    return load_robot(str(CONFIG))['lidars'][1]['self_reflection_windows']


def test_only_the_measured_angle_and_distance_are_rejected():
    ranges = np.array([.26, .27, .28, .292, .30, 1., np.nan, np.inf])
    mask = self_reflection_mask(ranges, math.radians(25), 0., windows())
    assert mask.tolist() == [False, True, True, True, False, False, False, False]
    for bearing in (22.9, 30.1, -25., 120.):
        assert not self_reflection_mask([.28], math.radians(bearing), 0., windows()).any()
    assert not self_reflection_mask([.28], math.radians(25), 0., []).any()


def test_angle_layout_does_not_depend_on_beam_indices():
    a = self_reflection_mask(np.full(721, .28), -math.pi, math.pi/360, windows())
    b = self_reflection_mask(np.full(721, .28), math.pi, -math.pi/360, windows())
    np.testing.assert_array_equal(a, b[::-1])
    assert a.any()


@pytest.mark.parametrize('change', [
    {'range_max_m': .26}, {'range_min_m': 0.}, {'angle_max_deg': 181.},
    {'angle_min_deg': 35.}, {'range_max_m': float('nan')}, {'source': ''},
])
def test_invalid_windows_fail_configuration(change):
    value = windows()
    value[0].update(change)
    with pytest.raises(ValueError):
        reflection_windows(value)


@pytest.mark.parametrize('frame,expect_rejection', [('lidar_rear', True), ('lidar_front', False)])
def test_scan_callback_applies_sensor_window_as_nan_without_clearing(frame, expect_rejection):
    robot = load_robot(str(CONFIG))
    lidar = robot['lidars'][1]
    output = []
    state = dict(origin=np.array([lidar['pose']['x'], lidar['pose']['y']]),
                 yaw=lidar['pose']['yaw'], frame_id='lidar_rear',
                 self_reflection_windows=lidar['self_reflection_windows'],
                 cache_key=None, cutoffs=None, directions=None,
                 publisher=SimpleNamespace(publish=lambda m: output.append(copy.deepcopy(m))),
                 nowalls_publisher=None)
    node = SimpleNamespace(lidar_states={'/scan_rear': state},
                           footprint=robot['footprint'], padding=.05)
    node._cutoffs_for = lambda msg, st: ScanFootprintFilter._cutoffs_for(node, msg, st)
    msg = LaserScan()
    msg.header.stamp.sec = 10
    msg.header.frame_id = frame
    msg.angle_min = math.radians(25.)
    msg.angle_increment = math.radians(1.)
    msg.range_min = .1
    msg.range_max = 12.
    msg.ranges = [.28, .32, 1.]
    ScanFootprintFilter._scan_callback(node, msg, '/scan_rear')
    assert math.isnan(output[0].ranges[0]) == expect_rejection
    assert output[0].ranges[1:] == pytest.approx([.32, 1.])


def test_recorded_body_fixed_returns_are_removed_only_from_rear_sensor():
    fixture = json.loads((ROOT/'docs/slowdown_scan_fixture_20260915.json').read_text())
    robot = load_robot(str(CONFIG))
    for sample in fixture['samples']:
        for lidar in robot['lidars']:
            scan = sample[lidar['topic']+'_filtered']['value']
            ranges = np.asarray(scan['ranges'], float)
            mask = self_reflection_mask(ranges, scan['angle_min'], scan['angle_increment'],
                                        lidar['self_reflection_windows'])
            if lidar['name'] == 'front':
                assert not mask.any()
            else:
                assert 5 <= mask.sum() <= 15
                assert np.all((ranges[mask] >= .27) & (ranges[mask] <= .292))
