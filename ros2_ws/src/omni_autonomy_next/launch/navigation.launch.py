from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from omni_autonomy_next.runtime_dependencies import require_fixed_tf2


def generate_launch_description():
    require_fixed_tf2()
    share = get_package_share_directory('omni_autonomy_next')
    params = LaunchConfiguration('params_file')
    autostart = LaunchConfiguration('autostart')
    log_level = LaunchConfiguration('log_level')
    tracker = LaunchConfiguration('tracker')
    nodes = [
        'controller_server', 'smoother_server', 'planner_server',
        'behavior_server', 'bt_navigator', 'waypoint_follower',
        'velocity_smoother', 'collision_monitor',
    ]
    common = {
        'output': 'screen', 'parameters': [params],
        'arguments': ['--ros-args', '--log-level', log_level],
    }
    return LaunchDescription([
        DeclareLaunchArgument(
            'params_file', default_value=share + '/config/nav2_next.yaml'
        ),
        DeclareLaunchArgument('autostart', default_value='true'),
        DeclareLaunchArgument('log_level', default_value='info'),
        DeclareLaunchArgument('tracker', default_value='true'),
        # In tracker mode controller_server still owns the NavigateToPose
        # lifecycle, the goal checker, the progress checker and recovery, but
        # trajectory_tracker drives.  MPPI is cut to a token search so it does
        # not spend the Jetson's cores optimising a command nobody consumes.
        Node(
            package='nav2_controller', executable='controller_server',
            remappings=[('cmd_vel', 'cmd_vel_nav')],
            condition=UnlessCondition(tracker), **common,
        ),
        Node(
            package='nav2_controller', executable='controller_server',
            remappings=[('cmd_vel', 'cmd_vel_mppi_idle')],
            condition=IfCondition(tracker),
            parameters=[params, {
                'FollowPath.batch_size': 60,
                'FollowPath.time_steps': 10,
                'FollowPath.visualize': False,
            }],
            output='screen',
            arguments=['--ros-args', '--log-level', log_level],
        ),
        Node(
            package='nav2_smoother', executable='smoother_server',
            name='smoother_server', **common,
        ),
        Node(
            package='nav2_planner', executable='planner_server',
            name='planner_server', **common,
        ),
        Node(
            package='nav2_behaviors', executable='behavior_server',
            name='behavior_server', remappings=[('cmd_vel', 'cmd_vel_nav')],
            **common,
        ),
        Node(
            package='nav2_bt_navigator', executable='bt_navigator',
            name='bt_navigator', output='screen',
            parameters=[params, {
                'fixed_bucket_routes_file': share + '/config/routes.yaml',
                # Load the same tree the bridge uses during lifecycle startup.
                # Its action/service clients must be ready before accepting the
                # first fixed route, rather than being created while armed.
                'default_nav_through_poses_bt_xml':
                    share + '/behavior_trees/follow_fixed_approach.xml',
            }],
            arguments=['--ros-args', '--log-level', log_level],
        ),
        Node(
            package='nav2_waypoint_follower', executable='waypoint_follower',
            name='waypoint_follower', **common,
        ),
        # trajectory_tracker produces an already acceleration-limited
        # profile, so in tracker mode this stage would only resample it at
        # 30 Hz and add lag.  It stays in the lifecycle (the manager waits for
        # it) but its output is dead-ended.
        Node(
            package='nav2_velocity_smoother', executable='velocity_smoother',
            name='velocity_smoother',
            remappings=[
                ('cmd_vel', 'cmd_vel_nav'),
                ('cmd_vel_smoothed', 'cmd_vel_nav_smoothed'),
            ],
            condition=UnlessCondition(tracker), **common,
        ),
        Node(
            package='nav2_velocity_smoother', executable='velocity_smoother',
            name='velocity_smoother',
            remappings=[
                ('cmd_vel', 'cmd_vel_mppi_idle'),
                ('cmd_vel_smoothed', 'cmd_vel_smoothed_idle'),
            ],
            condition=IfCondition(tracker), **common,
        ),
        # A source listed in observation_sources that stops publishing makes the
        # monitor hold the robot at zero velocity ("Robot to stop due to invalid
        # source"), so one dead LiDAR would take autonomy away entirely.
        # scan_source_supervisor switches the silent source off at runtime.
        Node(
            package='nav2_collision_monitor', executable='collision_monitor',
            name='collision_monitor', **common,
        ),
        Node(
            package='nav2_lifecycle_manager', executable='lifecycle_manager',
            name='lifecycle_manager_navigation', output='screen',
            parameters=[{'autostart': autostart, 'node_names': nodes}],
        ),
    ])
