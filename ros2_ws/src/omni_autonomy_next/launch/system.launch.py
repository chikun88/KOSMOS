from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
    RegisterEventHandler,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessIO
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
import yaml

from omni_autonomy_next.config import calibrated_tracking_parameters, load_field, load_robot
from omni_autonomy_next.configured_goals import load_configured_poses
from omni_autonomy_next.lidar_ports import resolve_lidar_ports
from omni_autonomy_next.runtime_dependencies import require_fixed_tf2


def _as_bool(value):
    return str(value).strip().lower() in ('1', 'true', 'yes', 'on')


def _build(context):
    require_fixed_tf2()
    share = Path(get_package_share_directory('omni_autonomy_next'))
    robot_file = LaunchConfiguration('robot_config').perform(context)
    field_file = LaunchConfiguration('field_config').perform(context)
    localization_file = LaunchConfiguration('localization_config').perform(context)
    runtime_file = LaunchConfiguration('runtime_config').perform(context)
    nav2_file = LaunchConfiguration('nav2_params').perform(context)
    field_poses_file = LaunchConfiguration('field_poses').perform(context)
    remembered_poses_file = LaunchConfiguration(
        'remembered_poses_file'
    ).perform(context)
    initial_pose_id = LaunchConfiguration('initial_pose_id').perform(context).strip()
    lidars = LaunchConfiguration('lidars')
    wheels = LaunchConfiguration('wheels')
    demo = LaunchConfiguration('demo')
    motors = LaunchConfiguration('motors')
    gui = LaunchConfiguration('gui')
    rviz = LaunchConfiguration('rviz')
    motors_enabled = _as_bool(LaunchConfiguration('motors').perform(context))
    if (LaunchConfiguration('motion_mode').perform(context) == 'staged_heading'
            and not _as_bool(LaunchConfiguration('tracker').perform(context))):
        raise ValueError('staged_heading requires tracker:=true')
    demo_enabled = _as_bool(LaunchConfiguration('demo').perform(context))
    wheels_enabled = _as_bool(LaunchConfiguration('wheels').perform(context))
    # /wheel/odometry comes from measurement_wheel on hardware and from the
    # in-process simulator in demo mode. Claiming wheel odometry when neither
    # runs leaves the localizer waiting for a topic nobody publishes.
    wheel_odometry_available = wheels_enabled or demo_enabled

    # Validate the original structure used by launch, including the CAD digests.
    load_robot(robot_file)
    field = load_field(field_file)
    with open(robot_file, encoding='utf-8') as stream:
        robot = yaml.safe_load(stream)['robot']
    with open(runtime_file, encoding='utf-8') as stream:
        runtime = yaml.safe_load(stream)['runtime_guard']['ros__parameters']
    configured_poses = load_configured_poses(field_poses_file)
    if initial_pose_id not in configured_poses:
        raise ValueError(
            f'initial_pose_id {initial_pose_id!r} is not configured in '
            f'{field_poses_file}'
        )
    configured_initial_pose = configured_poses[initial_pose_id]
    initial_pose = [
        configured_initial_pose['x'],
        configured_initial_pose['y'],
        configured_initial_pose['yaw'],
    ]
    nodes = [Node(
        package='omni_autonomy_next', executable='run_recorder',
        name='run_recorder', output='screen',
        condition=IfCondition(LaunchConfiguration('record_runs')),
        parameters=[{
            'log_directory': LaunchConfiguration('run_log_directory'),
            'demo': ParameterValue(demo, value_type=bool),
            'operation_mode': 'demo' if demo_enabled else ('hardware' if motors_enabled else 'hardware_motor_off'),
            'motors': motors_enabled,
            'wheels': wheels_enabled,
            'lidars': ParameterValue(lidars, value_type=bool),
            'tracker': ParameterValue(LaunchConfiguration('tracker'), value_type=bool),
            'motion_mode': LaunchConfiguration('motion_mode'),
            'scan_topics': [str(lidar['topic']) for lidar in robot['lidars']],
            'config_files': [robot_file, field_file, localization_file, runtime_file,
                             nav2_file, field_poses_file, remembered_poses_file,
                             str(share / 'config' / 'competition_footprints.yaml'),
                             str(share / 'config' / 'rl_policy.yaml'),
                             str(share / 'config' / 'routes.yaml'),
                             str(share / 'config' / 'mu3_navigation.yaml')],
        }],
    )]
    # robot.yaml names each LiDAR by a by-path device, which encodes the hub port
    # the cable happens to sit in.  Identify the units by their own serial number
    # instead so moving a cable between hub ports does not silently break a scan.
    lidar_names = [str(lidar['name']) for lidar in robot['lidars']]
    lidar_ports = {}
    live_lidars = set(lidar_names)
    if _as_bool(LaunchConfiguration('lidars').perform(context)):
        lidar_ports, live_lidars, lidar_port_notes = resolve_lidar_ports(
            robot['lidars']
        )
        for note in lidar_port_notes:
            print(f'[lidar_ports] {note}')
    dead_lidars = [name for name in lidar_names if name not in live_lidars]
    if dead_lidars:
        # Only a diagnosis for the log.  Which collision-monitor sources are
        # actually used is decided at runtime by scan_source_supervisor, because
        # a launch-time snapshot can be taken mid USB enumeration and because a
        # LiDAR may die or come back long after startup.
        print(
            f'[lidar_ports] no reply from {", ".join(dead_lidars)}; the stack '
            'will run degraded on '
            f'{", ".join(sorted(live_lidars)) or "no"} LiDAR(s)'
        )

    wall_localizer = Node(
        package='omni_autonomy_next', executable='wall_localizer',
        name='wall_localizer', output='screen',
        parameters=[localization_file, {
            'field_config_file': field_file,
            'robot_config_file': robot_file,
            'initial_pose': initial_pose,
            'use_wheel_odometry': wheel_odometry_available,
        }],
    )
    navigation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            str(share / 'launch' / 'navigation.launch.py')
        ),
        launch_arguments={
            'params_file': nav2_file,
            'tracker': LaunchConfiguration('tracker'),
        }.items(),
    )
    navigation_start = {'done': False, 'output': bytearray()}

    def start_navigation(source):
        if navigation_start['done']:
            return []
        navigation_start['done'] = True
        print(f'[startup] starting Nav2 ({source})')
        return [navigation]

    def localizer_output(event):
        output = navigation_start['output']
        output.extend(event.text)
        if len(output) > 4096:
            del output[:-4096]
        if b'LiDARs; initial pose=' in output:
            return start_navigation('localizer ready')
        return []

    def navigation_fallback(_context):
        return start_navigation('readiness timeout fallback')

    for lidar in robot['lidars']:
        serial_port = lidar_ports.get(lidar['name'], str(lidar['serial_port']))
        pose = lidar['pose']
        nodes.append(Node(
            package='tf2_ros', executable='static_transform_publisher',
            name=f'{lidar["name"]}_static_tf', output='screen',
            arguments=[
                '--x', str(pose.get('x', 0.0)), '--y', str(pose.get('y', 0.0)),
                '--z', str(pose.get('z', 0.0)),
                '--roll', str(pose.get('roll', 0.0)),
                '--pitch', str(pose.get('pitch', 0.0)),
                '--yaw', str(pose.get('yaw', 0.0)),
                '--frame-id', str(robot.get('base_frame_id', 'base_link')),
                '--child-frame-id', str(lidar['frame_id']),
            ],
        ))
        nodes.append(Node(
            package='sllidar_ros2', executable='sllidar_node',
            namespace=f'lidar_{lidar["name"]}', name='sllidar_node',
            condition=IfCondition(lidars), output='screen', respawn=True,
            respawn_delay=2.0,
            parameters=[{
                'channel_type': 'serial', 'serial_port': serial_port,
                'serial_baudrate': int(lidar.get('serial_baudrate', 115200)),
                'frame_id': str(lidar['frame_id']),
                'inverted': bool(lidar.get('inverted', False)),
                'angle_compensate': bool(lidar.get('angle_compensate', True)),
                'scan_mode': str(lidar.get('scan_mode', 'Boost')),
            }],
            remappings=[('scan', str(lidar['topic']))],
        ))

    nodes.extend([
        Node(
            package='omni_autonomy_next', executable='synthetic_scans',
            name='synthetic_scans', condition=IfCondition(demo), output='screen',
            parameters=[{
                'field_config_file': field_file, 'robot_config_file': robot_file,
                'true_pose': initial_pose,
                'cmd_vel_topic': '/cmd_vel_safe',
                # 0.0/0.0 is the ideal integrator. Raise them to measure how
                # much drivebase dead time and actuator lag the tuned control
                # loop tolerates before it starts to weave.
                'command_delay_sec': ParameterValue(
                    LaunchConfiguration('sim_command_delay'), value_type=float
                ),
                'velocity_time_constant_sec': ParameterValue(
                    LaunchConfiguration('sim_velocity_tau'), value_type=float
                ),
            }],
        ),
        Node(
            package='omni_autonomy_next', executable='scan_footprint_filter',
            name='scan_footprint_filter', output='screen',
            parameters=[{
                'robot_config_file': robot_file,
                'field_config_file': field_file,
            }],
        ),
        # Takes the silent LiDAR out of the collision monitor's sources so one
        # dead unit degrades autonomy instead of holding the robot at zero.
        Node(
            package='omni_autonomy_next', executable='scan_source_supervisor',
            name='scan_source_supervisor', output='screen',
            parameters=[{'robot_config_file': robot_file}],
        ),
        RegisterEventHandler(OnProcessIO(
            target_action=wall_localizer,
            on_stdout=localizer_output,
            on_stderr=localizer_output,
        )),
        wall_localizer,
        Node(
            package='omni_autonomy_next', executable='measurement_wheel',
            name='measurement_wheel', condition=IfCondition(wheels), output='screen',
            parameters=[{'robot_config_file': robot_file}],
        ),
        Node(
            package='omni_autonomy_next', executable='cad_visualizer',
            name='cad_visualizer', output='screen',
            parameters=[{
                'cad_model_file': field['source_stl'],
                'field_layout_file': field['source_layout'],
                'slice_z_mm': 130.0,
            }],
        ),
        # Start Nav2 as soon as wall_localizer reports that its CPU-heavy CAD
        # lookup is complete. A fixed 26 s delay left a ready robot idle for
        # about 19 s on normal starts. Keep the old delay only as a fallback in
        # case process output is unavailable; the shared guard prevents the
        # IncludeLaunchDescription from ever running twice.
        TimerAction(
            period=LaunchConfiguration('navigation_delay'),
            actions=[OpaqueFunction(function=navigation_fallback)],
        ),
        Node(
            package='omni_autonomy_next', executable='trajectory_tracker',
            name='trajectory_tracker', condition=IfCondition(
                LaunchConfiguration('tracker')),
            output='screen',
            parameters=[{
                'robot_config_file': robot_file,
                'runtime_config_file': runtime_file,
                'nav2_config_file': nav2_file,
                'routes_config_file': str(share / 'config' / 'routes.yaml'),
                'field_config_file': field_file,
                'optimize_yaw': ParameterValue(
                    LaunchConfiguration('optimize_yaw'), value_type=bool),
                'footprint_config_file': str(share / 'config' / 'competition_footprints.yaml'),
                'motion_mode': LaunchConfiguration('motion_mode'),
                'output_topic': '/cmd_vel_nav_smoothed',
            }, calibrated_tracking_parameters(
                robot, hardware=motors_enabled and not demo_enabled,
                motion_mode=LaunchConfiguration('motion_mode').perform(context))],
        ),
        Node(
            package='omni_autonomy_next', executable='rl_policy',
            name='rl_policy', output='screen',
            parameters=[{
                'policy_file': str(share / 'config' / 'rl_policy.yaml'),
                'field_config_file': field_file,
                'input_topic': '/cmd_vel_nav_smoothed',
                'output_topic': '/cmd_vel_rl',
                # MPPI's CostCritic already turns away from a nearby wall.
                # The feed-forward tracker has no obstacle critic, so supply
                # the same baseline response before Collision Monitor. The
                # monitor remains downstream and retains the final stop right.
                'apply_baseline_repulsion': ParameterValue(
                    LaunchConfiguration('tracker'), value_type=bool),
            }],
        ),
        Node(
            package='omni_autonomy_next', executable='runtime_guard',
            name='runtime_guard', output='screen',
            parameters=[runtime_file, {
                'robot_config_file': robot_file,
                'field_poses_file': field_poses_file,
                # Synthetic demo commands only move the in-process simulator;
                # keep operator ARM mandatory in every hardware-capable mode.
                'operation_mode': ('demo' if demo_enabled else 'full' if motors_enabled else 'real'),
                'require_armed': not demo_enabled,
                'require_motor_link': motors_enabled,
                'require_auto_engaged': motors_enabled,
            }],
        ),
        Node(
            package='omni_autonomy_next', executable='goal_bridge',
            name='goal_bridge', output='screen',
            parameters=[{
                'input_topic': '/goal_request',
                'verify_tracker_arrival': ParameterValue(
                    LaunchConfiguration('tracker'), value_type=bool),
                'goal_id_topic': '/navigation/goal_id_request',
                'cancel_topic': '/navigation/cancel_request',
                'status_topic': '/navigation/goal_status',
                'action_name': '/navigate_to_pose',
                'route_action_name': '/navigate_through_poses',
                'field_poses_file': field_poses_file,
                'field_config_file': field_file,
                'footprint_config_file': str(share / 'config' / 'competition_footprints.yaml'),
                'remembered_poses_file': remembered_poses_file,
                'routes_file': str(share / 'config' / 'routes.yaml'),
                'startup_goal_id': ParameterValue(
                    LaunchConfiguration('goal_id'), value_type=str
                ),
            }],
        ),
        Node(
            package='omni_autonomy_next', executable='mu3_navigation',
            name='mu3_navigation', condition=IfCondition(motors), output='screen',
            parameters=[str(share / 'config' / 'mu3_navigation.yaml')],
        ),
        Node(
            package='omni_autonomy_next', executable='motor_udp_bridge',
            name='motor_udp_bridge', condition=IfCondition(motors), output='screen',
            parameters=[{
                'cmd_vel_topic': '/cmd_vel_safe',
                'enable_topic': '/system/armed',
                'estop_topic': '/system/emergency_stop',
                'reset_estop_service': '/system/reset_motor_estop',
                # egg8が開発ボードへそのまま出せるUARTフレームを組んで送り、
                # bacon6はそれを書き換えずUARTへ流す（パススルー）。
                # 足回りの各輪値はここ(Jetson側)で確定する。
                'payload_format': 'v4_uart',
                'local_ip': '192.168.60.1', 'local_port': 8888,
                'remote_ip': '192.168.60.2', 'remote_port': 8888,
                'send_rate_hz': 100.0, 'immediate_send_on_cmd': True,
                'require_healthy_telemetry': True,
                'max_linear_speed': max(float(runtime['hard_max_linear_speed']),
                                        float(runtime['hard_max_lateral_speed'])),
                'max_angular_speed': float(runtime['hard_max_angular_speed']),
                'linear_command_scale': float(robot['drivetrain'].get('linear_command_scale', 1.0)),
                'angular_command_scale': float(robot['drivetrain'].get('angular_command_scale', 1.0)),
            }],
        ),
        Node(
            package='omni_autonomy_next', executable='speed_gui',
            name='speed_gui', condition=IfCondition(gui), output='screen',
            parameters=[{'field_poses_file': field_poses_file}],
        ),
        Node(
            package='rviz2', executable='rviz2', name='rviz2',
            condition=IfCondition(rviz), output='screen',
            arguments=['-d', str(share / 'rviz' / 'system.rviz')],
        ),
    ])
    if demo_enabled and wheels_enabled:
        raise ValueError('demo:=true and wheels:=true would publish duplicate odometry')
    if motors_enabled and demo_enabled:
        raise ValueError('motors:=true is forbidden in synthetic demo mode')
    return nodes


def generate_launch_description():
    share = get_package_share_directory('omni_autonomy_next')
    return LaunchDescription([
        DeclareLaunchArgument('robot_config', default_value=share + '/config/robot.yaml'),
        DeclareLaunchArgument('field_config', default_value=share + '/config/field_planning.yaml'),
        DeclareLaunchArgument('localization_config', default_value=share + '/config/localization.yaml'),
        DeclareLaunchArgument('runtime_config', default_value=share + '/config/runtime.yaml'),
        DeclareLaunchArgument('nav2_params', default_value=share + '/config/nav2_next.yaml'),
        DeclareLaunchArgument('field_poses', default_value=share + '/config/field_poses.yaml'),
        DeclareLaunchArgument(
            'remembered_poses_file',
            default_value=str(
                Path.home() / '.ros' / 'omni_autonomy_next'
                / 'remembered_poses.json'
            ),
        ),
        # The supplied match screenshot reports x=-1.80 m, y=4.75 m and
        # yaw=-90 degrees at the start/loading pose (ID 1).  Read the values
        # from field_poses.yaml instead of duplicating another magic pose here.
        DeclareLaunchArgument('initial_pose_id', default_value='1'),
        DeclareLaunchArgument('lidars', default_value='true'),
        DeclareLaunchArgument('wheels', default_value='true'),
        DeclareLaunchArgument('demo', default_value='false'),
        DeclareLaunchArgument('motors', default_value='false'),
        DeclareLaunchArgument('record_runs', default_value='true'),
        DeclareLaunchArgument('run_log_directory', default_value=str(
            Path.home() / '.ros' / 'omni_autonomy_next' / 'runs')),
        DeclareLaunchArgument('gui', default_value='true'),
        DeclareLaunchArgument('rviz', default_value='true'),
        # Normally Nav2 starts from wall_localizer's readiness output instead
        # of waiting for this value. Keep 26 s as the fallback deadline: cold
        # starts previously took about 15 s, and an unconditional 18 s start
        # made lifecycle bringup fail under load.
        DeclareLaunchArgument('navigation_delay', default_value='26.0'),
        DeclareLaunchArgument('goal_id', default_value=''),
        # Synthetic-demo drivebase model. Defaults reproduce the ideal
        # integrator this demo has always used.
        DeclareLaunchArgument('sim_command_delay', default_value='0.0'),
        DeclareLaunchArgument('sim_velocity_tau', default_value='0.0'),
        # tracker:=false falls back to the 20 Hz MPPI feedback loop. Every
        # safety gate downstream is unchanged; only the stage that produces
        # /cmd_vel_nav_smoothed changes.
        #
        # Default true, restored on 2026-08-07 at the operator's request so the
        # stack matches the 16:59 configuration that day. Record what is and is
        # not established about that choice, because the numbers in the
        # 2026-08-06/07 entries of docs/DEPLOYMENT_STATUS.md are easy to
        # misread:
        #
        # * The tracker's headline figures -- 0.039 m cross-track, 0.706 m/s --
        #   come from launch 2026-08-06-16-30-56, which starts synthetic_scans
        #   and no sllidar_node. That is the demo, not hardware. The 0.463 m
        #   column it gets compared against is hardware. There is no
        #   like-for-like comparison on this robot.
        # * MPPI's apparent weakness there is confounded as well: the 20:46
        #   diagnostic ran with the operator's GUI slider at 31 %, and
        #   0.31 * 0.78 = 0.242 m/s accounts for its 0.203 m/s p95 without any
        #   appeal to loop lag.
        # * Two costs are real and land on the operating path in tracker mode:
        #   while Nav2 is in recovery the tracker publishes zeros for the whole
        #   recovery (PLAN_STALE, measured at 9-14 s a time), and
        #   behavior_server's Spin and BackUp publish to cmd_vel_nav, which
        #   nothing subscribes to here.
        #
        # A hardware diagnose-chain taken at a known speed scale is what would
        # settle it either way; none exists yet.
        DeclareLaunchArgument('tracker', default_value='true'),
        # 姿勢の最適化は既定で無効。駆動系の菱形では体軸合わせが 1.41 倍に
        # なるが、運用プロファイル 0.78 の円の中では 11% しか残らず、その
        # 11% は姿勢を回すのに使う車輪バジェットと釣り合う。実測でも
        # 1->4 区間で 0.707 -> 0.687 m/s とわずかに悪化した。プロファイルを
        # 菱形へ近づけたときに有効化する価値が出る。
        DeclareLaunchArgument('optimize_yaw', default_value='false'),
        DeclareLaunchArgument('motion_mode', default_value='simultaneous'),
        OpaqueFunction(function=_build),
    ])
