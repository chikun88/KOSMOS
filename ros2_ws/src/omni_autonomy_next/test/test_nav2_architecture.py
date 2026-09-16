import json
import importlib.util
import re
from pathlib import Path
from types import SimpleNamespace

import yaml

from omni_autonomy_next.configured_goals import load_configured_poses
from omni_autonomy_next.rl_residual import CadClearanceModel
from omni_autonomy_next.route_approaches import (
    load_fixed_departures,
    load_fixed_goal_approaches,
    select_fixed_goal_approach,
    select_fixed_goal_departure,
    select_fixed_pose_departure,
)


CONFIG = Path(__file__).resolve().parents[1] / 'config'
PACKAGE = Path(__file__).resolve().parents[1] / 'omni_autonomy_next'


def _nav2():
    return yaml.safe_load((CONFIG / 'nav2_next.yaml').read_text(encoding='utf-8'))


def _runtime():
    runtime = yaml.safe_load((CONFIG / 'runtime.yaml').read_text(encoding='utf-8'))
    return runtime['runtime_guard']['ros__parameters']


def _default_profile_limits():
    runtime = _runtime()
    profiles = json.loads(runtime['profiles_json'])
    return runtime, profiles[runtime['default_profile']]


def test_mppi_is_holonomic_and_uses_full_footprint():
    params = yaml.safe_load((CONFIG / 'nav2_next.yaml').read_text(encoding='utf-8'))
    controller = params['controller_server']['ros__parameters']
    follow = controller['FollowPath']
    assert follow['plugin'] == 'nav2_mppi_controller::MPPIController'
    assert follow['motion_model'] == 'Omni'
    assert follow['vy_max'] > 0.0
    assert follow['CostCritic']['consider_footprint'] is True
    assert 'PreferForwardCritic' not in follow['critics']
    assert 'PathAngleCritic' not in follow['critics']
    assert 'TwirlingCritic' in follow['critics']
    assert abs(controller['controller_frequency'] * follow['model_dt'] - 1.0) < 0.01
    # 32 x 800 was observed dropping the control loop to 9.7 Hz against
    # controller_frequency 20.0 and leaving routes unfinished.
    assert follow['time_steps'] * follow['batch_size'] <= 15000
    # Exploration in the three-dimensional holonomic control space comes from
    # fresh noise each cycle rather than from more rollouts.
    assert follow['regenerate_noises'] is True
    # The horizon must cover the window Collision Monitor brakes over, so the
    # optimizer sees the obstacle before the independent gate reacts to it.
    # Beyond that, a longer horizon buys nothing and costs loop rate: at
    # time_steps 32 the controller measurably published /cmd_vel_nav at
    # 16.8 Hz against controller_frequency 20.0, so model_dt described a grid
    # the loop did not hold.
    monitor = params['collision_monitor']['ros__parameters']
    horizon = follow['time_steps'] * follow['model_dt']
    assert horizon >= monitor['FootprintApproach']['time_before_collision']
    assert horizon <= 1.5
    assert controller['goal_checker']['xy_goal_tolerance'] <= 0.04
    # A four-wheel 45-degree omni base is symmetric, so reverse travel must not
    # be limited below forward travel.
    assert follow['vx_min'] == -follow['vx_max']


def test_controller_limits_equal_the_profile_the_guard_executes():
    """Nav2 may not plan outside the *speed* envelope RuntimeGuard will pass.

    RuntimeGuard clips its input silently, and every MPPI rollout scored by
    the critics is integrated with these constraints, so a limit the drivebase
    cannot deliver makes the whole cost field optimistic.  The braking bound
    is the safety-relevant one: see
    test_planned_braking_is_never_more_optimistic_than_the_guard.
    """
    params = _nav2()
    follow = params['controller_server']['ros__parameters']['FollowPath']
    smoother = params['velocity_smoother']['ros__parameters']
    runtime, _ = _default_profile_limits()
    # MPPI's reference is the fastest profile; SpeedLimit scales others.
    profile = json.loads(runtime['profiles_json'])['sprint']

    assert follow['vx_max'] == profile['linear']
    assert follow['vy_max'] == profile['lateral']
    assert follow['wz_max'] == profile['angular']

    # velocity_smoother is the one stage that shapes acceleration, so MPPI is
    # pinned to *its* bounds rather than to the guard's.
    assert smoother['max_velocity'] == [
        profile['linear'], profile['lateral'], profile['angular']
    ]
    assert smoother['min_velocity'] == [
        -profile['linear'], -profile['lateral'], -profile['angular']
    ]
    assert follow['ax_max'] == smoother['max_accel'][0]
    assert follow['ay_max'] == smoother['max_accel'][1]
    assert follow['az_max'] == smoother['max_accel'][2]
    assert follow['ax_min'] == smoother['max_decel'][0]
    assert follow['ay_min'] == smoother['max_decel'][1]
    assert smoother['max_decel'] == [-value for value in smoother['max_accel']]

    # The guard's SpeedLimit reference must describe those same MPPI maxima.
    assert runtime['planner_reference_speeds'] == [
        follow['vx_max'], follow['vy_max'], follow['wz_max']
    ]


def test_guard_dynamics_are_a_gate_around_the_shaping_stage_not_a_second_one():
    """RuntimeGuard must not re-limit an already-limited acceleration.

    When the guard carried the same acceleration numbers as velocity_smoother
    it re-shaped every command, and its jerk term became an unmodelled
    second-order lag inside the 20 Hz control loop.  Measured on egg8 with the
    equal values: the guard alone contributed 90 ms of the 140 ms command
    latency and the chain delivered 0.373 m/s^2 against 0.85 m/s^2 configured.
    Holding the guard strictly above the shaping stage restored 0.686 m/s^2
    and cut total latency to 100 ms, with the guard still clamping
    unconditionally whenever anything upstream exceeds it.

    Required ordering:  MPPI = velocity_smoother < profile <= hard_max_*.
    """
    params = _nav2()
    smoother = params['velocity_smoother']['ros__parameters']
    runtime = _runtime()
    profiles = json.loads(runtime['profiles_json'])
    controller_period = (
        1.0 / params['controller_server']['ros__parameters'][
            'controller_frequency'
        ]
    )

    for name, profile in profiles.items():
        assert profile['linear_accel'] > smoother['max_accel'][0], name
        assert profile['angular_accel'] > smoother['max_accel'][2], name
        assert profile['linear_accel'] <= runtime[
            'hard_max_linear_acceleration'], name
        assert profile['angular_accel'] <= runtime[
            'hard_max_angular_acceleration'], name
        assert profile['linear_jerk'] <= runtime['hard_max_linear_jerk'], name
        assert profile['angular_jerk'] <= runtime['hard_max_angular_jerk'], name
        # velocity_smoother's output acceleration steps instantly between 0 and
        # max_accel, so the guard's jerk term always engages on those corners.
        # Catching up must cost well under one controller period, otherwise the
        # guard is a lag element again rather than a gate.
        assert smoother['max_accel'][0] / profile['linear_jerk'] <= (
            0.5 * controller_period
        ), name
        assert smoother['max_accel'][2] / profile['angular_jerk'] <= (
            0.5 * controller_period
        ), name


def test_no_hidden_speed_governor_sits_below_the_validated_profile():
    """The deployed default must be the profile the offline gate validates.

    A 0.65 default speed scale multiplied every profile without any gate
    knowing, so the robot ran a third slower than simulation/run_campaign.py
    ever measured.
    """
    runtime = _runtime()
    assert runtime['default_speed_scale'] == 1.0
    assert 0.0 < runtime['red_zone_speed_scale'] <= 1.0
    gui = (PACKAGE / 'speed_gui_node.py').read_text(encoding='utf-8')
    assert 'self.scale.setValue(100)' in gui


def test_guard_republishes_its_envelope_as_a_nav2_speed_limit():
    runtime = _runtime()
    source = (PACKAGE / 'runtime_guard_node.py').read_text(encoding='utf-8')
    assert runtime['speed_limit_topic'] == '/speed_limit'
    assert runtime['speed_limit_publish_rate_hz'] > 0.0
    assert 'from nav2_msgs.msg import SpeedLimit' in source
    assert 'message.percentage = True' in source


def test_goal_yaw_is_retired_while_translating():
    """An omni base must not serialize translation and rotation.

    The offline model drives yaw toward the goal from the first tick, so the
    online critics have to allow the same simultaneous motion.
    """
    follow = _nav2()['controller_server']['ros__parameters']['FollowPath']
    goal_angle = follow['GoalAngleCritic']
    twirling = follow['TwirlingCritic']
    # Larger than any route on this field: goal yaw is always regulated.
    assert goal_angle['threshold_to_consider'] >= 8.0
    assert twirling['cost_weight'] < goal_angle['cost_weight']


def test_prediction_horizon_fits_inside_local_costmap():
    params = yaml.safe_load((CONFIG / 'nav2_next.yaml').read_text(encoding='utf-8'))
    follow = params['controller_server']['ros__parameters']['FollowPath']
    local = params['local_costmap']['local_costmap']['ros__parameters']
    horizon_distance = follow['time_steps'] * follow['model_dt'] * max(
        follow['vx_max'], follow['vy_max']
    )
    # Nav2 Jazzy declares these two parameters as integer metres.
    assert isinstance(local['width'], int)
    assert isinstance(local['height'], int)
    assert local['resolution'] >= 0.04
    assert horizon_distance + 0.6 < 0.5 * min(local['width'], local['height'])
    assert 'static_layer' not in local['plugins']


def test_command_pipeline_topics_do_not_bypass_guard():
    params = yaml.safe_load((CONFIG / 'nav2_next.yaml').read_text(encoding='utf-8'))
    monitor = params['collision_monitor']['ros__parameters']
    runtime = yaml.safe_load((CONFIG / 'runtime.yaml').read_text(encoding='utf-8'))
    runtime = runtime['runtime_guard']['ros__parameters']
    assert monitor['cmd_vel_in_topic'] == '/cmd_vel_rl'
    assert monitor['cmd_vel_out_topic'] == '/cmd_vel_collision_safe'
    assert runtime['input_topic'] == '/cmd_vel_collision_safe'
    assert runtime['output_topic'] == '/cmd_vel_safe'
    assert runtime['require_rl_policy'] is True
    assert runtime['rl_scale_topic'] == '/rl/speed_scale'
    assert set(monitor['observation_sources']) == {'scan_front', 'scan_rear'}
    assert monitor['enable_stamped_cmd_vel'] is False
    launch_source = (
        Path(__file__).resolve().parents[1] / 'launch' / 'system.launch.py'
    ).read_text(encoding='utf-8')
    setup_source = (PACKAGE.parent / 'setup.py').read_text(encoding='utf-8')
    assert "executable='rl_policy'" in launch_source
    assert "'input_topic': '/cmd_vel_nav_smoothed'" in launch_source
    assert "'output_topic': '/cmd_vel_rl'" in launch_source
    assert 'rl_policy = omni_autonomy_next.rl_policy_node:main' in setup_source


def test_smoother_ramps_open_loop_and_can_leave_rest_on_every_axis():
    """The shaping stage must ramp from its own output, not from measurement.

    CLOSED_LOOP recomputes each acceleration step from the speed measured on
    /wheel/odometry, which is only correct when this node's output is what the
    base executes.  rl_policy, collision_monitor and runtime_guard all sit
    downstream, so the ramp instead chased RuntimeGuard's output: the target
    stayed about one acceleration step above measured, the guard reached it
    within a single smoothing period, and the guard's overshoot clamp reset its
    acceleration state on nearly every tick.  Measured on egg8: 0.373 m/s^2
    delivered against 0.85 m/s^2 configured.  This is independent of odometry
    calibration, which was verified by hand on 2026-08-05 and is sound.
    """
    params = yaml.safe_load((CONFIG / 'nav2_next.yaml').read_text(encoding='utf-8'))
    smoother = params['velocity_smoother']['ros__parameters']
    assert smoother['feedback'] == 'OPEN_LOOP'
    frequency = float(smoother['smoothing_frequency'])
    for accel, deadband in zip(
        smoother['max_accel'], smoother['deadband_velocity']
    ):
        # Otherwise deadbanding erases the first closed-loop acceleration
        # step and measured velocity remains zero forever.
        assert accel / frequency > deadband


def test_directional_collision_gate_can_leave_critical_cad_lane():
    params = yaml.safe_load((CONFIG / 'nav2_next.yaml').read_text(encoding='utf-8'))
    monitor = params['collision_monitor']['ros__parameters']
    # An unconditional stop polygon also blocks a command which moves away
    # from the triggering point. The swept-footprint approach gate is
    # directional and is therefore the active stop in a narrow lane.
    assert 'StopZone' not in monitor['polygons']
    assert monitor['FootprintApproach']['action_type'] == 'approach'
    assert monitor['FootprintApproach']['footprint_topic'] == (
        '/local_costmap/published_footprint'
    )
    assert monitor['FootprintApproach']['time_before_collision'] >= 1.0
    # The red-lane fixture corner can disappear into a scanner's 0.15 m blind
    # zone while covering fewer than four beams. The swept-footprint geometry
    # already rejects points outside the commanded motion, so one hit is the
    # required fail-safe threshold here.
    assert monitor['FootprintApproach']['min_points'] == 1
    assert monitor['scan_front']['source_timeout'] == monitor['source_timeout']
    assert monitor['scan_rear']['source_timeout'] == monitor['source_timeout']
    assert 0.5 <= monitor['SlowZone']['slowdown_ratio'] < 1.0


def test_slow_zone_is_a_proximity_belt_not_a_global_speed_cap():
    """A slowdown polygon reaching far past the body caps speed everywhere.

    This field is enclosed by walls, so a 0.50 m belt was satisfied on nearly
    every route and multiplied the whole run by its ratio.  FootprintApproach
    is the authoritative proximity brake because it scales velocity by the
    real time to collision.
    """
    params = _nav2()
    monitor = params['collision_monitor']['ros__parameters']
    robot = yaml.safe_load((CONFIG / 'robot.yaml').read_text(encoding='utf-8'))
    footprint = robot['robot']['footprint']
    body_x = max(abs(point[0]) for point in footprint)
    body_y = max(abs(point[1]) for point in footprint)

    slow_points = yaml.safe_load(monitor['SlowZone']['points'])
    margin_x = max(abs(point[0]) for point in slow_points) - body_x
    margin_y = max(abs(point[1]) for point in slow_points) - body_y
    assert 0.05 < margin_x <= 0.25, margin_x
    assert 0.05 < margin_y <= 0.25, margin_y
    assert 'FootprintApproach' in monitor['polygons']
    assert monitor['FootprintApproach']['action_type'] == 'approach'
    assert monitor['FootprintApproach']['time_before_collision'] >= 1.0


def test_smac_cost_aware_global_planner_is_selected():
    params = yaml.safe_load((CONFIG / 'nav2_next.yaml').read_text(encoding='utf-8'))
    planner = params['planner_server']['ros__parameters']['GridBased']
    global_map = params['global_costmap']['global_costmap']['ros__parameters']
    assert planner['plugin'] == 'nav2_smac_planner::SmacPlanner2D'
    assert planner['cost_travel_multiplier'] >= 2.0
    assert planner['use_final_approach_orientation'] is False
    assert global_map['plugins'] == ['static_layer', 'inflation_layer']


def test_navigation_start_is_sequenced_after_map_and_localization_setup():
    launch_source = (
        Path(__file__).resolve().parents[1] / 'launch' / 'system.launch.py'
    ).read_text(encoding='utf-8')
    assert "TimerAction(" in launch_source
    assert 'RegisterEventHandler(OnProcessIO(' in launch_source
    assert 'on_stdout=localizer_output' in launch_source
    assert 'on_stderr=localizer_output' in launch_source
    assert "b'LiDARs; initial pose=' in output" in launch_source
    assert "start_navigation('localizer ready')" in launch_source
    assert "start_navigation('readiness timeout fallback')" in launch_source
    assert "navigation_start['done'] = True" in launch_source
    # 準備完了出力が取得できない場合だけ、旧来の安全側期限へフォールバック
    # する。18秒の無条件起動で lifecycle bringup が落ちた履歴があるため、
    # フォールバック自体は短くしない。
    match = re.search(
        r"DeclareLaunchArgument\('navigation_delay', default_value='([\d.]+)'\)",
        launch_source)
    assert match is not None
    assert float(match.group(1)) >= 24.0


def test_launcher_omits_an_empty_optional_goal_argument():
    runner_path = Path(__file__).resolve().parents[4] / 'run.py'
    spec = importlib.util.spec_from_file_location('omni_unified_launcher', runner_path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    args = SimpleNamespace(
        no_gui=False,
        no_rviz=False,
        no_tracker=False,
        navigation_delay=26.0,
        initial_pose_id='1',
        goal_id='',
    )

    command = runner.system_launch_arguments(args, 'demo')
    assert 'goal_id:=' not in command
    args.goal_id = '4'
    assert 'goal_id:=4' in runner.system_launch_arguments(args, 'demo')
    # 追従器が既定。MPPI は --no-tracker のときだけ。
    assert 'tracker:=true' in command
    args.no_tracker = True
    assert 'tracker:=false' in runner.system_launch_arguments(args, 'demo')


def test_image_match_pose_is_the_single_startup_pose_source():
    launch_source = (
        Path(__file__).resolve().parents[1] / 'launch' / 'system.launch.py'
    ).read_text(encoding='utf-8')
    runner_source = (
        Path(__file__).resolve().parents[4] / 'run.py'
    ).read_text(encoding='utf-8')
    poses = yaml.safe_load((CONFIG / 'field_poses.yaml').read_text(encoding='utf-8'))
    image_pose = poses['poses'][1]

    assert image_pose['x'] == -1.8
    assert image_pose['y'] == 4.75
    assert abs(image_pose['yaw'] + 1.5707963267948966) < 1.0e-12
    assert "DeclareLaunchArgument('initial_pose_id', default_value='1')" in launch_source
    assert "configured_poses = load_configured_poses(field_poses_file)" in launch_source
    assert "'true_pose': initial_pose" in launch_source
    assert "'initial_pose': initial_pose" in launch_source
    assert 'f"initial_pose_id:={args.initial_pose_id}"' in runner_source


def test_runtime_guard_pose_subscription_matches_localizer_message_type():
    source = (PACKAGE / 'runtime_guard_node.py').read_text(encoding='utf-8')
    assert "PoseWithCovarianceStamped," in source
    assert "message.pose.pose.position.x" in source
    assert "message.pose.pose.position.y" in source
    assert "log = self.get_logger()" not in source


def test_demo_skips_only_the_operator_arm_interlock():
    launch_source = (
        Path(__file__).resolve().parents[1] / 'launch' / 'system.launch.py'
    ).read_text(encoding='utf-8')
    runtime = yaml.safe_load((CONFIG / 'runtime.yaml').read_text(encoding='utf-8'))
    runtime = runtime['runtime_guard']['ros__parameters']
    source = (PACKAGE / 'runtime_guard_node.py').read_text(encoding='utf-8')

    assert runtime['require_armed'] is True
    assert "'require_armed': not demo_enabled" in launch_source
    assert "armed=self.armed or not require_armed" in source
    assert "'require_motor_link': motors_enabled" in launch_source
    assert "'require_auto_engaged': motors_enabled" in launch_source


def test_rviz_goal_is_forwarded_to_nav2_action():
    launch_source = (
        Path(__file__).resolve().parents[1] / 'launch' / 'system.launch.py'
    ).read_text(encoding='utf-8')
    rviz_source = (
        Path(__file__).resolve().parents[1] / 'rviz' / 'system.rviz'
    ).read_text(encoding='utf-8')
    setup_source = (
        Path(__file__).resolve().parents[1] / 'setup.py'
    ).read_text(encoding='utf-8')

    assert "executable='goal_bridge'" in launch_source
    assert "'input_topic': '/goal_request'" in launch_source
    assert "'action_name': '/navigate_to_pose'" in launch_source
    assert "'goal_id_topic': '/navigation/goal_id_request'" in launch_source
    assert "LaunchConfiguration('goal_id'), value_type=str" in launch_source
    assert "DeclareLaunchArgument('goal_id', default_value='')" in launch_source
    assert 'Value: /goal_request' in rviz_source
    assert 'goal_bridge = omni_autonomy_next.goal_bridge_node:main' in setup_source


def test_fixed_bucket_goals_use_predefined_bidirectional_clear_lanes():
    """地点4/5の進入・退出を再び自由な単一目標へ戻さないこと。"""
    approaches = load_fixed_goal_approaches(CONFIG / 'routes.yaml')
    goals = load_configured_poses(CONFIG / 'field_poses.yaml')
    assert set(approaches) == {'4', '5'}

    expected = {
        ('4', 'upper'): 'fixed_bucket_2_from_upper',
        ('4', 'lower'): 'fixed_bucket_2_from_lower',
        ('5', 'upper'): 'fixed_bucket_3_from_upper',
        ('5', 'lower'): 'fixed_bucket_3_from_lower',
    }
    expected_departure = {
        ('4', 'upper'): 'fixed_bucket_2_to_upper',
        ('4', 'lower'): 'fixed_bucket_2_to_lower',
        ('5', 'upper'): 'fixed_bucket_3_to_upper',
        ('5', 'lower'): 'fixed_bucket_3_to_lower',
    }
    clearance = CadClearanceModel.from_yaml(
        CONFIG / 'field_planning.yaml',
        CONFIG / 'competition_footprints.yaml',
    )
    for goal_id, entry in approaches.items():
        goal = goals[goal_id]
        for side, current_y in (
            ('upper', entry['split_y'] + 1.0),
            ('lower', entry['split_y'] - 1.0),
        ):
            route_id, waypoints = select_fixed_goal_approach(
                approaches, goal_id, (goal['x'], current_y)
            )
            assert route_id == expected[(goal_id, side)]
            assert len(waypoints) == (3 if side == 'lower' else 2)
            waypoint = waypoints[0]
            departure_id, departure_waypoints = select_fixed_goal_departure(
                approaches,
                goals,
                (goal['x'], goal['y']),
                (goal['x'], current_y),
                0.20,
            )
            assert departure_id == expected_departure[(goal_id, side)]
            assert departure_waypoints == [p for p in reversed(waypoints)
                if abs(p['y']-goal['y']) > .01]
            # The final segment is the configured x=-0.80 shooting lane, not
            # a planner-selected diagonal beside the bucket.
            assert waypoint['x'] == -0.80
            lane = waypoints + [goal]
            minimum = min(
                clearance.body_clearance((
                    a['x'] + t * (b['x'] - a['x']),
                    a['y'] + t * (b['y'] - a['y']),
                ), a['yaw'])
                for a, b in zip(lane[:-1], lane[1:])
                for t in (index / 100.0 for index in range(101))
            )
            assert minimum >= 0.048

    # The common 4 <-> 5 shuttle must first leave the origin bucket through
    # its near-side lane, then enter the destination bucket through its
    # corresponding near-side lane.
    departure_4, points_4 = select_fixed_goal_departure(
        approaches,
        goals,
        (goals['4']['x'], goals['4']['y']),
        (goals['5']['x'], goals['5']['y']),
        0.20,
        destination_goal_id='5',
    )
    approach_5, points_5 = select_fixed_goal_approach(
        approaches, '5', (goals['4']['x'], goals['4']['y'])
    )
    assert departure_4 == 'fixed_bucket_2_to_lower'
    assert approach_5 == 'fixed_bucket_3_from_upper'
    assert [point['y'] for point in points_4 + points_5] == [0.75, 0.50, -1.55, -1.95]

    departure_5, points_5 = select_fixed_goal_departure(
        approaches,
        goals,
        (goals['5']['x'], goals['5']['y']),
        (goals['4']['x'], goals['4']['y']),
        0.20,
        destination_goal_id='4',
    )
    approach_4, points_4 = select_fixed_goal_approach(
        approaches, '4', (goals['5']['x'], goals['5']['y'])
    )
    assert departure_5 == 'fixed_bucket_3_to_upper'
    assert approach_4 == 'fixed_bucket_2_from_lower'
    assert [point['y'] for point in points_5 + points_4] == [-1.95, -1.55, 0.50, 0.75, 1.34]

    same_goal_id, same_goal_points = select_fixed_goal_departure(
        approaches,
        goals,
        (goals['4']['x'], goals['4']['y']),
        (goals['4']['x'], goals['4']['y']),
        0.20,
        destination_goal_id='4',
    )
    assert same_goal_id is None
    assert same_goal_points == []

    launch_source = (
        Path(__file__).resolve().parents[1] / 'launch' / 'system.launch.py'
    ).read_text(encoding='utf-8')
    bridge_source = (PACKAGE / 'goal_bridge_node.py').read_text(
        encoding='utf-8'
    )
    setup_source = (
        Path(__file__).resolve().parents[1] / 'setup.py'
    ).read_text(encoding='utf-8')
    tree = (
        Path(__file__).resolve().parents[1] / 'behavior_trees'
        / 'follow_fixed_approach.xml'
    ).read_text(encoding='utf-8')
    assert "'route_action_name': '/navigate_through_poses'" in launch_source
    assert "'routes_file': str(share / 'config' / 'routes.yaml')" in launch_source
    assert 'NavigateThroughPoses.Goal()' in bridge_source
    assert 'select_fixed_goal_departure(' in bridge_source
    assert 'for point in departure_points + transit_points + approach_points' in bridge_source
    assert 'self._refresh_fixed_routes(self.pending_request)' in bridge_source
    assert 'goal.behavior_tree = self.route_behavior_tree' in bridge_source
    assert 'goal.poses = [' in bridge_source
    assert "glob('behavior_trees/*.xml')" in setup_source
    assert '<RateController hz="1.0">' in tree
    assert '<RemovePassedBucketGoals input_goals="{goals}" output_goals="{goals}" radius="0.16" passage_window="0.45"/>' in tree
    assert _nav2()['bt_navigator']['ros__parameters'][
        'default_server_timeout'
    ] >= 200


def test_loading_pose_uses_a_straight_clear_departure_before_turning():
    """地点1では狭い場所で急旋回せず、開口部側のゲートを通る。"""
    departures = load_fixed_departures(CONFIG / 'routes.yaml')
    goals = load_configured_poses(CONFIG / 'field_poses.yaml')
    assert set(departures) == {'1'}

    route_id, waypoints = select_fixed_pose_departure(
        departures, goals,
        (goals['1']['x'], goals['1']['y']), 0.20,
        destination_goal_id='2',
    )
    assert route_id == 'loading_bay_exit'
    assert len(waypoints) == 1
    gate = waypoints[0]
    assert gate['x'] == -1.80
    assert gate['y'] == 4.10
    assert gate['yaw'] == goals['1']['yaw']

    # A retry halfway out must keep the same gate; otherwise the replacement
    # plan can begin rotating inside the slot. Once the gate is reached it is
    # intentionally removed.
    retry_id, retry_points = select_fixed_pose_departure(
        departures, goals, (-1.80, 4.40), 0.20,
        destination_goal_id='2',
    )
    assert retry_id == route_id
    assert retry_points == waypoints
    clear_id, clear_points = select_fixed_pose_departure(
        departures, goals, (-1.80, 4.10), 0.20,
        destination_goal_id='2',
    )
    assert clear_id is None
    assert clear_points == []

    clearance = CadClearanceModel.from_yaml(
        CONFIG / 'field_planning.yaml',
        CONFIG / 'competition_footprints.yaml',
    )
    # The first segment is pure translation at the only safe start yaw.
    departure_margin = min(
        clearance.body_clearance((
            goals['1']['x'] + t * (gate['x'] - goals['1']['x']),
            goals['1']['y'] + t * (gate['y'] - goals['1']['y']),
        ), gate['yaw'])
        for t in (index / 100.0 for index in range(101))
    )
    assert departure_margin >= 0.04
    # At the gate every intermediate rotation from -90 to 0 deg is clear.
    sweep_margin = min(
        clearance.body_clearance(
            (gate['x'], gate['y']),
            gate['yaw'] + t * (goals['2']['yaw'] - gate['yaw']),
        )
        for t in (index / 100.0 for index in range(101))
    )
    assert sweep_margin >= 0.06

    # The next leg can rotate gradually while leaving the obstacle behind.
    onward_margin = min(
        clearance.body_clearance((
            gate['x'] + t * (goals['2']['x'] - gate['x']),
            gate['y'] + t * (goals['2']['y'] - gate['y']),
        ), gate['yaw'] + t * (goals['2']['yaw'] - gate['yaw']))
        for t in (index / 100.0 for index in range(101))
    )
    assert onward_margin >= 0.06

    # This is the regression: the old direct position path started rotating
    # inside the slot, where an intermediate footprint touches the CAD wall.
    old_direct_margin = min(
        clearance.body_clearance((
            goals['1']['x'] + t * (goals['2']['x'] - goals['1']['x']),
            goals['1']['y'] + t * (goals['2']['y'] - goals['1']['y']),
        ), goals['1']['yaw'] + t * (
            goals['2']['yaw'] - goals['1']['yaw']))
        for t in (index / 100.0 for index in range(101))
    )
    assert old_direct_margin == 0.0

    # The tracker sees the combined through-poses path and distributes final
    # yaw over its full arc length. Validate that actual profile too, rather
    # than assuming it will stop and rotate exactly at the intermediate gate.
    combined = []
    for start, end, count in (
        (goals['1'], gate, 65),
        (gate, goals['2'], 245),
    ):
        combined.extend((
            start['x'] + index / count * (end['x'] - start['x']),
            start['y'] + index / count * (end['y'] - start['y']),
        ) for index in range(count))
    combined.append((goals['2']['x'], goals['2']['y']))
    arclength = [0.0]
    for first, second in zip(combined, combined[1:]):
        arclength.append(arclength[-1] + (
            (second[0] - first[0]) ** 2
            + (second[1] - first[1]) ** 2
        ) ** 0.5)
    cutoff = max(arclength[-1] - 0.10, 0.5 * arclength[-1])
    combined_margin = min(
        clearance.body_clearance(
            point,
            goals['1']['yaw'] + min(arc / cutoff, 1.0) * (
                goals['2']['yaw'] - goals['1']['yaw']),
        )
        for point, arc in zip(combined, arclength)
    )
    assert combined_margin >= 0.04

    # Selecting the pose already occupied must remain a no-op.
    same_id, same_points = select_fixed_pose_departure(
        departures, goals,
        (goals['1']['x'], goals['1']['y']), 0.20,
        destination_goal_id='1',
    )
    assert same_id is None
    assert same_points == []

    bridge_source = (PACKAGE / 'goal_bridge_node.py').read_text(encoding='utf-8')
    assert 'select_fixed_pose_departure(' in bridge_source


def test_tf_tolerances_cover_localizer_compute_jitter():
    params = yaml.safe_load((CONFIG / 'nav2_next.yaml').read_text(encoding='utf-8'))
    follow = params['controller_server']['ros__parameters']['FollowPath']
    local = params['local_costmap']['local_costmap']['ros__parameters']
    global_map = params['global_costmap']['global_costmap']['ros__parameters']
    assert follow['transform_tolerance'] >= 0.50
    assert local['obstacle_layer']['tf_filter_tolerance'] >= 0.30
    assert 'obstacle_layer' not in global_map['plugins']


def test_tracking_guard_debounces_rejections_but_times_out_heartbeat():
    runtime = yaml.safe_load((CONFIG / 'runtime.yaml').read_text(encoding='utf-8'))
    runtime = runtime['runtime_guard']['ros__parameters']
    assert (
        0.0 < runtime['tracking_rejection_grace_sec']
        < runtime['tracking_heartbeat_timeout_sec']
    )
    assert runtime['tracking_heartbeat_timeout_sec'] <= 0.5
    source = (PACKAGE / 'runtime_guard_node.py').read_text(encoding='utf-8')
    assert 'tracking_fresh and (' in source
    assert "'tracking_effective': tracking_effective" in source


def test_lidar_ingestion_does_not_share_icp_callback_group():
    source = (PACKAGE / 'wall_localizer_node.py').read_text(encoding='utf-8')
    localization = yaml.safe_load(
        (CONFIG / 'localization.yaml').read_text(encoding='utf-8')
    )['wall_localizer']['ros__parameters']
    assert 'self.scan_callback_group = MutuallyExclusiveCallbackGroup()' in source
    assert 'callback_group=self.scan_callback_group' in source
    assert localization['update_rate_hz'] <= 4.0
    assert localization['max_points_per_lidar'] <= 200
    assert localization['max_iterations'] <= 6


def test_localizer_only_claims_wheel_odometry_when_a_publisher_runs():
    # wheels:=false with a hardcoded use_wheel_odometry left the localizer
    # broadcasting map->odom while nothing published odom->base_link.
    launch_source = (
        Path(__file__).resolve().parents[1] / 'launch' / 'system.launch.py'
    ).read_text(encoding='utf-8')
    assert (
        'wheel_odometry_available = wheels_enabled or demo_enabled'
        in launch_source
    )
    assert "'use_wheel_odometry': wheel_odometry_available" in launch_source
    localization = yaml.safe_load(
        (CONFIG / 'localization.yaml').read_text(encoding='utf-8')
    )['wall_localizer']['ros__parameters']
    # Long enough to ride out a Jetson CPU spike in the wheel callback, short
    # enough that RViz recovers quickly if the counters really do die.
    assert 1.0 <= localization['wheel_odom_tf_timeout_sec'] <= 2.0


def test_tracking_heartbeat_is_independent_of_slow_icp_completion():
    source = (PACKAGE / 'wall_localizer_node.py').read_text(encoding='utf-8')
    localization = yaml.safe_load(
        (CONFIG / 'localization.yaml').read_text(encoding='utf-8')
    )['wall_localizer']['ros__parameters']
    fast_timer_body = source.split('def _fast_publish_callback', 1)[1]
    assert 'self.tracking_publisher.publish(tracking)' in fast_timer_body
    assert 'tracking_solution_timeout_sec' in fast_timer_body
    assert 0.5 < localization['tracking_solution_timeout_sec'] <= 1.0


def test_empty_but_fresh_lidar_revolution_does_not_expire_other_lidar():
    source = (PACKAGE / 'wall_localizer_node.py').read_text(encoding='utf-8')
    scan_body = source.split('def _scan_callback', 1)[1].split(
        'def _scan_point_stamps', 1
    )[0]
    assert 'if not np.any(valid)' not in scan_body
    assert "'generation': self.scan_generations[name]" in scan_body


def test_narrow_lane_uses_strict_path_and_single_hit_approach_gate():
    params = yaml.safe_load((CONFIG / 'nav2_next.yaml').read_text(encoding='utf-8'))
    follow = params['controller_server']['ros__parameters']['FollowPath']
    monitor = params['collision_monitor']['ros__parameters']
    assert follow['PathAlignCritic']['cost_weight'] >= 20.0
    assert follow['PathAlignCritic']['offset_from_furthest'] <= 10
    assert monitor['FootprintApproach']['min_points'] == 1


def test_gui_can_select_cancel_and_observe_saved_goals():
    gui_source = (PACKAGE / 'speed_gui_node.py').read_text(encoding='utf-8')
    goal_source = (PACKAGE / 'goal_bridge_node.py').read_text(encoding='utf-8')
    launch_source = (
        Path(__file__).resolve().parents[1] / 'launch' / 'system.launch.py'
    ).read_text(encoding='utf-8')

    assert "'/navigation/goal_id_request'" in gui_source
    assert "'/navigation/cancel_request'" in gui_source
    assert "'/navigation/goal_status'" in gui_source
    assert 'self.goal_buttons' in gui_source
    assert 'self._start_named_goal' in gui_source
    assert 'DurabilityPolicy.VOLATILE' in gui_source
    assert "self.declare_parameter('cancel_topic'" in goal_source
    assert 'GetState' in goal_source
    assert 'State.PRIMARY_STATE_ACTIVE' in goal_source
    assert "'PREEMPTING'" in goal_source
    assert "'cancel_topic': '/navigation/cancel_request'" in launch_source
