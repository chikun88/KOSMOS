"""Sensor safety contracts runnable without ROS or attached hardware.

The callback harness compiles the actual methods from their source, omitting
only ROS imports and node construction. It exercises packet validation and
state transitions; DDS scheduling and real devices still need ROS integration
and hardware tests.
"""
import ast
import math
import threading
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from omni_autonomy_next.contec_cnt import ContecCounter, ContecCounterError
from omni_autonomy_next.geometry import points_in_polygon, relative_pose, compose_pose
from omni_autonomy_next.measurement_wheel_odometry import (
    MeasurementWheelKinematics, counter_delta, counts_to_wheel_displacements,
    solve_calibration_matrix,
)
from omni_autonomy_next.measurement_wheel_power import (
    MeasurementWheelPowerSwitch, PowerControlConfig, PowerControlError,
)
from omni_autonomy_next.scan_freshness import scan_metadata_valid, timestamp_is_fresh
from omni_autonomy_next.source_freshness import message_stamp_nanoseconds as source_stamp_nanoseconds
from omni_autonomy_next import lidar_ports

SOURCE = Path(__file__).resolve().parents[1] / 'omni_autonomy_next'


def callbacks(filename, names, **extra):
    tree = ast.parse((SOURCE / filename).read_text(encoding='utf-8'))
    functions = [node for node in ast.walk(tree)
                 if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in functions} == set(names)
    for node in functions:
        node.decorator_list = []
    module = ast.Module(body=[ast.ImportFrom(module='__future__',
        names=[ast.alias(name='annotations')], level=0), *functions], type_ignores=[])
    scope = dict(np=np, math=math, scan_metadata_valid=scan_metadata_valid,
                 timestamp_is_fresh=timestamp_is_fresh,
                 source_stamp_nanoseconds=source_stamp_nanoseconds,
                 points_in_polygon=points_in_polygon, relative_pose=relative_pose,
                 compose_pose=compose_pose, ContecCounterError=ContecCounterError)
    scope.update(extra)
    exec(compile(ast.fix_missing_locations(module), str(SOURCE / filename), 'exec'), scope)
    return SimpleNamespace(**{name: scope[name] for name in names})


class ClockTime:
    def __init__(self, seconds):
        self.nanoseconds = round(seconds * 1.e9)

    def __sub__(self, other):
        return SimpleNamespace(nanoseconds=self.nanoseconds - other.nanoseconds)

    def to_msg(self):
        return SimpleNamespace(sec=self.nanoseconds // 1_000_000_000,
                               nanosec=self.nanoseconds % 1_000_000_000)


def scan(stamp=10., **changes):
    values = dict(
        header=SimpleNamespace(frame_id='lidar_front', stamp=SimpleNamespace(
            sec=int(stamp), nanosec=round((stamp - int(stamp)) * 1.e9))),
        angle_min=-math.pi, angle_increment=.01, range_min=.1,
        range_max=12., time_increment=.001, scan_time=.1,
        ranges=[math.inf, math.inf, math.inf],
    )
    values.update(changes)
    return SimpleNamespace(**values)


def logger():
    return SimpleNamespace(warning=lambda *a, **k: None, error=lambda *a, **k: None,
                           info=lambda *a, **k: None)


def test_source_handoff_confirms_replacement_before_disabling_old_input():
    decide = callbacks('scan_source_supervisor_node.py', ['sources_to_change']).sources_to_change
    enabled = {'scan_front': False, 'scan_rear': True}
    assert decide(enabled, {'scan_front'}) == {'scan_front': True}
    assert decide(enabled, {'scan_front'}, {'scan_rear'}) == {}
    enabled['scan_front'] = True
    assert decide(enabled, {'scan_front'}) == {'scan_rear': False}
    # Replay the old overlapping-request race: rear-disable is pending when
    # the rear becomes the only live input. Front must remain enabled.
    assert decide(enabled, {'scan_rear'}, {'scan_rear'}) == {}
    enabled['scan_rear'] = False
    assert decide(enabled, {'scan_rear'}) == {'scan_rear': True}


def test_no_live_inputs_never_leaves_all_collision_sources_disabled():
    decide = callbacks('scan_source_supervisor_node.py', ['sources_to_change']).sources_to_change
    assert decide({'front': True, 'rear': False}, set()) == {}
    assert decide({'front': False, 'rear': False}, set()) == {'front': True, 'rear': True}
    assert decide({'front': True}, {'unknown'}) == {}


def test_scan_replay_invalid_metadata_and_stale_headers_do_not_refresh_source():
    callback = callbacks('scan_source_supervisor_node.py', ['_scan_callback'])._scan_callback
    node = SimpleNamespace(last_scan_sec={}, last_scan_stamp_ns={},
                           scan_timeout_sec=.6, _now_sec=lambda: 10.)
    callback(node, 'front', scan(9.9))
    assert node.last_scan_sec == pytest.approx({'front': 9.9})
    node._now_sec = lambda: 10.2
    callback(node, 'front', scan(9.9))
    callback(node, 'front', scan(8.))
    callback(node, 'front', scan(20.))
    callback(node, 'front', scan(10.2, angle_increment=math.nan))
    assert node.last_scan_sec == pytest.approx({'front': 9.9})
    callback(node, 'front', scan(10.2))
    assert node.last_scan_sec == pytest.approx({'front': 10.2})


@pytest.mark.parametrize('changes', [
    dict(angle_min=math.nan), dict(angle_increment=math.inf),
    dict(range_max=math.nan), dict(range_max=.01),
    dict(time_increment=-.01), dict(scan_time=math.inf), dict(ranges=[]),
])
def test_invalid_scan_metadata_is_not_a_sensor_heartbeat(changes):
    assert not scan_metadata_valid(scan(**changes))


def test_all_infinite_ranges_are_still_a_valid_scan_and_clock_rollback_expires_data():
    assert scan_metadata_valid(scan())
    assert timestamp_is_fresh(10_000, 9_500, 600)
    assert not timestamp_is_fresh(9_000, 9_500, 600)


def test_localizer_scan_callback_accepts_empty_return_cloud_but_not_replay():
    methods = callbacks('wall_localizer_node.py',
                        ['_scan_callback', '_scan_point_stamps', 'message_stamp_nanoseconds'])
    now = [10.]
    node = SimpleNamespace(
        get_clock=lambda: SimpleNamespace(now=lambda: ClockTime(now[0])),
        get_logger=logger, scan_timeout=SimpleNamespace(nanoseconds=350_000_000),
        wheel_stamp_tolerance_ns=80_000_000, latest_scans={},
        scan_generations={'front': 0}, frame_warning_sent=set(),
        beam_stride=1, min_range=.1, max_range=12., max_points=100,
        footprint=np.array([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]]),
        _scan_point_stamps=methods._scan_point_stamps,
    )
    lidar = dict(name='front', frame_id='lidar_front', pose=dict(x=0., y=0., yaw=0.))
    methods._scan_callback(node, scan(10.), lidar)
    assert node.scan_generations == {'front': 1}
    assert node.latest_scans['front']['points'].shape == (0, 2)
    now[0] = 10.2
    methods._scan_callback(node, scan(10.), lidar)
    methods._scan_callback(node, scan(9.), lidar)
    methods._scan_callback(node, scan(20.), lidar)
    assert node.scan_generations == {'front': 1}


@pytest.mark.parametrize('invalid', ['zero_stamp', 'negative_stamp', 'invalid_nanosec',
                                      'wrong_frame', 'empty_frame'])
def test_invalid_scan_identity_or_timestamp_does_not_refresh_localization(invalid):
    methods = callbacks('wall_localizer_node.py',
                        ['_scan_callback', '_scan_point_stamps', 'message_stamp_nanoseconds'])
    now = [10.]
    node = SimpleNamespace(
        get_clock=lambda: SimpleNamespace(now=lambda: ClockTime(now[0])),
        get_logger=logger, scan_timeout=SimpleNamespace(nanoseconds=350_000_000),
        wheel_stamp_tolerance_ns=80_000_000, latest_scans={},
        scan_generations={'front': 0}, frame_warning_sent=set(),
        beam_stride=1, min_range=.1, max_range=12., max_points=100,
        footprint=np.array([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]]),
        _scan_point_stamps=methods._scan_point_stamps,
    )
    lidar = dict(name='front', frame_id='lidar_front', pose=dict(x=0., y=0., yaw=0.))
    methods._scan_callback(node, scan(10.), lidar)
    accepted = node.latest_scans['front']
    now[0] = 10.2
    message = scan(10.2)
    if invalid == 'zero_stamp':
        message.header.stamp.sec = message.header.stamp.nanosec = 0
    elif invalid == 'negative_stamp':
        message.header.stamp.sec = -1
    elif invalid == 'invalid_nanosec':
        message.header.stamp.nanosec = 1_000_000_000
    else:
        message.header.frame_id = 'lidar_rear' if invalid == 'wrong_frame' else ''
    methods._scan_callback(node, message, lidar)
    assert node.scan_generations == {'front': 1}
    assert node.latest_scans['front'] is accepted
    assert accepted['received'].nanoseconds == 10_000_000_000


@pytest.mark.parametrize('invalid', ['zero_stamp', 'negative_stamp', 'invalid_nanosec',
                                      'wrong_frame', 'empty_frame', 'wrong_child', 'empty_child'])
def test_invalid_wheel_identity_or_timestamp_cannot_move_pose_or_renew_health(invalid):
    methods = callbacks('wall_localizer_node.py',
                        ['_wheel_odom_callback', 'message_stamp_nanoseconds', 'yaw_from_quaternion'])
    history = []
    node = SimpleNamespace(
        odom_frame='odom', base_frame='base_link', get_logger=logger,
        get_clock=lambda: SimpleNamespace(now=lambda: ClockTime(10.)),
        params={'wheel_odom_freshness_sec': .3}, wheel_stamp_tolerance_ns=80_000_000,
        _pose_lock=threading.RLock(), pose=np.array([3., 4., .5]),
        previous_wheel_pose=np.zeros(3), latest_wheel_pose=np.zeros(3),
        previous_wheel_stamp_ns=9_900_000_000, latest_wheel_stamp_ns=9_900_000_000,
        latest_wheel_received_ns=9_900_000_000, wheel_generation=3,
        _append_wheel_history=lambda stamp, pose: history.append((stamp, pose.copy())),
    )
    message = SimpleNamespace(
        header=SimpleNamespace(frame_id='odom', stamp=SimpleNamespace(sec=10, nanosec=0)),
        child_frame_id='base_link', pose=SimpleNamespace(pose=SimpleNamespace(
            position=SimpleNamespace(x=.1, y=0.),
            orientation=SimpleNamespace(x=0., y=0., z=0., w=1.))),
    )
    if invalid == 'zero_stamp':
        message.header.stamp.sec = 0
    elif invalid == 'negative_stamp':
        message.header.stamp.sec = -1
    elif invalid == 'invalid_nanosec':
        message.header.stamp.nanosec = 1_000_000_000
    elif invalid in ('wrong_frame', 'empty_frame'):
        message.header.frame_id = 'unrelated_odom' if invalid == 'wrong_frame' else ''
    else:
        message.child_frame_id = 'other_base' if invalid == 'wrong_child' else ''
    methods._wheel_odom_callback(node, message)
    np.testing.assert_array_equal(node.pose, [3., 4., .5])
    assert node.odom_frame == 'odom'
    assert node.latest_wheel_received_ns == 9_900_000_000
    assert node.latest_wheel_stamp_ns == 9_900_000_000
    assert node.wheel_generation == 3
    assert history == []


def test_valid_configured_wheel_frame_updates_motion_but_replay_does_not_refresh_health():
    methods = callbacks('wall_localizer_node.py',
                        ['_wheel_odom_callback', 'message_stamp_nanoseconds', 'yaw_from_quaternion'])
    now = [10.]
    history = []
    node = SimpleNamespace(
        odom_frame='custom_odom', base_frame='robot_base', get_logger=logger,
        get_clock=lambda: SimpleNamespace(now=lambda: ClockTime(now[0])),
        params={'wheel_odom_freshness_sec': .3}, wheel_stamp_tolerance_ns=80_000_000,
        _pose_lock=threading.RLock(), pose=np.array([3., 4., 0.]),
        previous_wheel_pose=np.zeros(3), latest_wheel_pose=np.zeros(3),
        previous_wheel_stamp_ns=9_900_000_000, latest_wheel_stamp_ns=9_900_000_000,
        latest_wheel_received_ns=9_900_000_000, wheel_generation=3,
        _append_wheel_history=lambda stamp, pose: history.append((stamp, pose.copy())),
    )
    message = SimpleNamespace(
        header=SimpleNamespace(frame_id='custom_odom', stamp=SimpleNamespace(sec=10, nanosec=0)),
        child_frame_id='robot_base', pose=SimpleNamespace(pose=SimpleNamespace(
            position=SimpleNamespace(x=.1, y=0.),
            orientation=SimpleNamespace(x=0., y=0., z=0., w=1.))),
    )
    methods._wheel_odom_callback(node, message)
    np.testing.assert_allclose(node.pose, [3.1, 4., 0.])
    assert node.latest_wheel_received_ns == 10_000_000_000
    assert node.wheel_generation == 4
    now[0] = 10.2
    methods._wheel_odom_callback(node, message)
    assert node.latest_wheel_received_ns == 10_000_000_000
    assert node.wheel_generation == 4
    assert len(history) == 1


def test_instantaneous_beam_timestamps_are_finite_and_cannot_wrap_int64():
    method = callbacks('wall_localizer_node.py', ['_scan_point_stamps'])._scan_point_stamps
    np.testing.assert_array_equal(method(scan(time_increment=0., scan_time=.1),
        np.array([0, 2]), 10_000_000_000, 3), [10_000_000_000, 10_000_000_000])
    for increment in (math.nan, -1., 1.e300):
        with pytest.raises(ValueError):
            method(scan(time_increment=increment), np.array([0, 2]), 10_000_000_000, 3)


def test_clock_rollback_does_not_claim_wheel_tf_owner_is_alive():
    method = callbacks('wall_localizer_node.py', ['_wheel_odometry_alive'])._wheel_odometry_alive
    node = SimpleNamespace(use_wheel_odometry=True, wheel_odom_tf_timeout_ns=1_500_000_000,
        get_clock=lambda: SimpleNamespace(now=lambda: ClockTime(5.)))
    assert not method(node, 10_000_000_000)
    assert method(node, 4_000_000_000)


def test_localizer_rejects_invalid_parameters_without_partial_cache_change():
    methods = callbacks('wall_localizer_node.py',
        ['_cached_parameters_callback', 'validate_cached_parameters'],
        SetParametersResult=SimpleNamespace)
    node = SimpleNamespace(params={'lidar_correction_gain': .35, 'max_iterations': 18})
    result = methods._cached_parameters_callback(node, [
        SimpleNamespace(name='max_iterations', value=24),
        SimpleNamespace(name='lidar_correction_gain', value=math.nan),
    ])
    assert result.successful is False
    assert node.params == {'lidar_correction_gain': .35, 'max_iterations': 18}
    result = methods._cached_parameters_callback(node, [
        SimpleNamespace(name='lidar_correction_gain', value=.5),
    ])
    assert result.successful is True
    assert node.params['lidar_correction_gain'] == .5


@pytest.mark.parametrize('invalid', ['nan_position', 'nan_covariance', 'indefinite_covariance',
                                      'zero_quaternion'])
def test_invalid_initial_pose_cannot_poison_localization(invalid):
    methods = callbacks('wall_localizer_node.py', ['_initial_pose_callback', 'yaw_from_quaternion'])
    pose = SimpleNamespace(position=SimpleNamespace(x=1., y=2.),
                           orientation=SimpleNamespace(x=0., y=0., z=0., w=1.))
    covariance = np.eye(6)
    if invalid == 'nan_position':
        pose.position.x = math.nan
    elif invalid == 'nan_covariance':
        covariance[0, 0] = math.nan
    elif invalid == 'indefinite_covariance':
        covariance[0, 1] = covariance[1, 0] = 2.
    else:
        pose.orientation.w = 0.
    message = SimpleNamespace(header=SimpleNamespace(frame_id='map'),
                              pose=SimpleNamespace(pose=pose, covariance=covariance.ravel()))
    node = SimpleNamespace(map_frame='map', get_logger=logger, _pose_lock=threading.RLock(),
        pose=np.array([3., 4., .5]), covariance=np.eye(3), pose_reset_generation=7)
    methods._initial_pose_callback(node, message)
    np.testing.assert_array_equal(node.pose, [3., 4., .5])
    assert node.pose_reset_generation == 7


def test_invalid_icp_covariance_is_rejected_even_with_good_fit_statistics():
    methods = callbacks('wall_localizer_node.py', ['_result_accepted', 'optimization_result_finite'])
    node = SimpleNamespace(params={'min_correspondences': 3, 'max_accepted_rmse': .18,
                                   'max_pose_jump_translation': .25,
                                   'max_pose_jump_rotation': .45})
    result = SimpleNamespace(pose=np.zeros(3), covariance=np.eye(3), converged=True,
        final_translation_step=0., final_rotation_step=0., correspondences=100, rmse=.01)
    assert methods._result_accepted(node, result, np.zeros(3)) == (True, True)
    result.covariance[0, 0] = math.nan
    assert methods._result_accepted(node, result, np.zeros(3)) == (False, False)


def test_nonfinite_pose_does_not_reach_pose_or_tf_publishers():
    methods = callbacks('wall_localizer_node.py', ['_publish_pose', '_publish_tf'])
    node = SimpleNamespace(_pose_lock=threading.RLock(), get_logger=logger,
        pose=np.array([math.nan, 0., 0.]), covariance=np.eye(3), publish_tf=True,
        latest_wheel_pose=np.zeros(3), latest_wheel_received_ns=None)
    # No message constructor/publisher exists in this harness: attempting to
    # publish would fail immediately. Both actual callbacks must return early.
    methods._publish_pose(node, None)
    methods._publish_tf(node, None)


def test_wrong_sensor_frame_preserves_obstacle_ranges_without_self_filtering():
    method = callbacks('scan_footprint_filter_node.py', ['_scan_callback'])._scan_callback
    output = []
    node = SimpleNamespace(lidar_states={'/scan_rear': dict(
        frame_id='lidar_rear', publisher=SimpleNamespace(publish=output.append),
        nowalls_publisher=None)})
    message = scan(ranges=[.12, .13, .14])
    method(node, message, '/scan_rear')
    assert output[0].ranges == [.12, .13, .14]


def wheel_settings(**changes):
    settings = dict(wheel_positions=[[.2, 0.], [0., -.2], [-.2, 0.], [0., .2]],
                    wheel_drive_angles=[math.pi / 2., 0., -math.pi / 2., math.pi],
                    count_signs=[1., 1., 1., 1.], counter_bits=32,
                    counts_per_revolution=[2048.] * 4, wheel_radius=.0254)
    settings.update(changes)
    return settings


def test_standalone_encoder_defaults_use_the_validated_robot_wheel_geometry():
    declare = callbacks('measurement_wheel_node.py', ['_declare_parameters'])._declare_parameters
    defaults = {}
    declare(SimpleNamespace(declare_parameter=lambda name, value: defaults.setdefault(name, value)))
    robot = yaml.safe_load((SOURCE.parent / 'config' / 'robot.yaml').read_text())
    wheels = robot['robot']['measurement_wheels']
    assert defaults['wheel_radius'] == wheels['wheel_radius']
    np.testing.assert_allclose(np.asarray(defaults['wheel_positions']).reshape(-1, 2),
                               wheels['wheel_positions'])
    np.testing.assert_allclose(defaults['wheel_drive_angles_deg'], wheels['wheel_drive_angles_deg'])


@pytest.mark.parametrize('changes', [
    dict(wheel_radius=math.nan), dict(wheel_radius=math.inf),
    dict(count_signs=[math.nan, 1., 1., 1.]), dict(count_signs=[0., 1., 1., 1.]),
    dict(counts_per_revolution=[math.inf] * 4),
    dict(meters_per_count=[-1.] * 4), dict(meters_per_count=[math.nan] * 4),
    dict(counter_bits=.5), dict(counter_bits=65),
    dict(wheel_drive_angles=[math.nan] * 4),
])
def test_invalid_odometry_configuration_fails_before_hardware_is_opened(changes):
    with pytest.raises(ValueError):
        MeasurementWheelKinematics(**wheel_settings(**changes))


def test_invalid_encoder_samples_and_displacement_overflow_are_rejected():
    for value in (math.nan, math.inf, 1.5):
        with pytest.raises(ValueError):
            counter_delta([value], [0])
    with pytest.raises(ValueError):
        counts_to_wheel_displacements([1.e300], meters_per_count=[1.e300])


def test_calibration_rejects_missing_excitation_and_recovers_observable_matrix():
    with pytest.raises(ValueError, match='independent'):
        solve_calibration_matrix(np.zeros((20, 4)), np.zeros((20, 3)))
    rng = np.random.default_rng(15)
    counts = rng.normal(size=(200, 4)) * 1000.
    reference = np.array([[.0001, -.0001, 0., 0.],
                          [0., 0., .0001, -.0001],
                          [.0001, .0001, .0001, .0001]])
    fitted, diagnostics = solve_calibration_matrix(counts, counts @ reference.T)
    np.testing.assert_allclose(fitted, reference, atol=1.e-11)
    assert diagnostics['samples'] == 200


@pytest.mark.parametrize('channels', [[0, 0], [-1, 0], [0, 32768], [0, 1.5]])
def test_invalid_counter_channels_are_rejected_without_loading_driver(channels):
    with pytest.raises(ValueError):
        ContecCounter(device_name='CNT000', channels=channels)


def test_power_cleanup_runs_even_when_switching_off_fails():
    closed = []
    switch = MeasurementWheelPowerSwitch(PowerControlConfig(enabled=False))
    switch.backend = SimpleNamespace(close=lambda: closed.append(True))
    switch.turn_off = lambda: (_ for _ in ()).throw(PowerControlError('GPIO write failed'))
    with pytest.raises(PowerControlError):
        switch.close()
    assert closed == [True]


def test_counter_read_error_re_primes_before_integrating_motion():
    tick = callbacks('measurement_wheel_node.py', ['_timer_callback'])._timer_callback
    readings = iter([ContecCounterError('USB read failed'), [100] * 4])
    def read():
        value = next(readings)
        if isinstance(value, Exception):
            raise value
        return value
    node = SimpleNamespace(
        get_clock=lambda: SimpleNamespace(now=lambda: ClockTime(10.)), get_logger=logger,
        counter=SimpleNamespace(read=read), previous_counts=[0] * 4,
        last_time=ClockTime(9.99), last_status={}, latest_counts=None,
        count_publisher=None, _publish_array=lambda *args: None,
        _publish_status=lambda: None,
    )
    tick(node)
    assert node.previous_counts is None and node.last_time is None
    tick(node)
    assert node.previous_counts == [100] * 4
    assert node.last_status['state'] == 'PRIMED'


def test_counter_clock_rollback_rebases_samples_without_false_motion():
    tick = callbacks('measurement_wheel_node.py', ['_timer_callback'])._timer_callback
    node = SimpleNamespace(
        get_clock=lambda: SimpleNamespace(now=lambda: ClockTime(5.)), get_logger=logger,
        counter=SimpleNamespace(read=lambda: [100] * 4), previous_counts=[0] * 4,
        last_time=ClockTime(10.), last_status={}, latest_counts=None,
        count_publisher=None, _publish_array=lambda *args: None, _publish_status=lambda: None,
    )
    tick(node)
    assert node.previous_counts == [100] * 4
    assert node.last_time.nanoseconds == 5_000_000_000
    assert node.last_status['state'] == 'SKIPPED_CLOCK_JUMP'


@pytest.mark.parametrize('subscribers,twist,age,allowed', [
    (1, [0., 0., 0.], .01, False),
    (0, [.1, 0., 0.], .01, False),
    (0, [0., 0., .1], .01, False),
    (0, [0., 0., 0.], 1., False),
    (0, [0., 0., 0.], .01, True),
])
def test_origin_reset_requires_disconnected_consumers_and_fresh_stationary_feedback(
    subscribers, twist, age, allowed,
):
    reset = callbacks('measurement_wheel_node.py', ['_reset_odometry_callback'])._reset_odometry_callback
    publications = []
    node = SimpleNamespace(
        odom_publisher=SimpleNamespace(get_subscription_count=lambda: subscribers),
        standard_odom_publisher=None,
        get_clock=lambda: SimpleNamespace(now=lambda: ClockTime(10.)),
        last_time=ClockTime(10. - age), max_update_gap_sec=.25,
        latest_twist=np.array(twist), pose=np.array([3., 4., .5]),
        previous_counts=[10] * 4, latest_counts=[20] * 4,
        kinematics=SimpleNamespace(odometry_scale=np.ones(3)), publish_tf=False,
        _publish_status=lambda: None,
        _publish_odometry=lambda stamp, velocity: publications.append(velocity),
    )
    response = reset(node, None, SimpleNamespace())
    assert response.success is allowed
    if allowed:
        np.testing.assert_array_equal(node.pose, np.zeros(3))
        assert node.previous_counts == [20] * 4
        assert len(publications) == 1
    else:
        np.testing.assert_array_equal(node.pose, [3., 4., .5])
        assert node.previous_counts == [10] * 4
        assert publications == []


@pytest.mark.parametrize('descriptor,valid', [
    (b'\xa5\x5a\x14\x00\x00\x00\x04', True),
    (b'\xa5\x5a\x14\x00\x00\x00\x81', False),
])
def test_lidar_identity_probe_opens_with_motor_line_deasserted_and_checks_descriptor(
    monkeypatch, descriptor, valid,
):
    opened = []
    serial_number = bytes(range(16))
    class SerialLink:
        def __init__(self, **options):
            assert options['port'] is None
            assert options['write_timeout'] == .2
            self.dtr = True
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def open(self):
            opened.append(self.dtr)
            assert self.port == '/dev/fake'
        def reset_input_buffer(self):
            pass
        def write(self, request):
            assert request == b'\xa5\x50'
            return len(request)
        def read(self, count):
            return descriptor + b'\x00' * 4 + serial_number
    monkeypatch.setitem(sys.modules, 'serial', SimpleNamespace(Serial=SerialLink))
    monkeypatch.setattr(lidar_ports.time, 'sleep', lambda seconds: None)
    serial_number_result = lidar_ports.probe_device_serial('/dev/fake', timeout=.2)
    assert opened == [False]
    assert serial_number_result == (serial_number.hex().upper() if valid else None)


def test_known_unexpected_lidar_is_not_assigned_as_a_silent_configured_unit():
    lidars = [dict(name='front', device_serial='AAA', serial_port='/configured/front'),
              dict(name='rear', device_serial='BBB', serial_port='/configured/rear')]
    observed = {'/dev/one': 'AAA012345', '/dev/two': 'CCC012345'}
    resolved, responsive, _ = lidar_ports.resolve_lidar_ports(lidars, probe=observed.get,
                                                           ports=list(observed))
    assert resolved == {'front': '/dev/one', 'rear': '/configured/rear'}
    assert responsive == {'front'}
