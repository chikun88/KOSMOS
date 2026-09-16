from glob import glob
import os

from setuptools import find_packages, setup


package_name = 'omni_autonomy_next'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'behavior_trees'),
         glob('behavior_trees/*.xml')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Robot Team',
    maintainer_email='robot@example.com',
    description='Holonomic MPPI autonomy, safety, and hardware integration for MU3.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'cad_to_field = omni_autonomy_next.cad_import:main',
            'cad_visualizer = omni_autonomy_next.cad_visualizer_node:main',
            'goal_bridge = omni_autonomy_next.goal_bridge_node:main',
            'measurement_wheel = omni_autonomy_next.measurement_wheel_node:main',
            'motor_udp_bridge = omni_autonomy_next.motor_udp_bridge_node:main',
            'mu3_navigation = omni_autonomy_next.mu3_navigation_node:main',
            'rl_policy = omni_autonomy_next.rl_policy_node:main',
            'run_recorder = omni_autonomy_next.run_recorder_node:main',
            'runtime_guard = omni_autonomy_next.runtime_guard_node:main',
            'scan_footprint_filter = omni_autonomy_next.scan_footprint_filter_node:main',
            'scan_source_supervisor = omni_autonomy_next.scan_source_supervisor_node:main',
            'speed_gui = omni_autonomy_next.speed_gui_node:main',
            'synthetic_scans = omni_autonomy_next.synthetic_scan_node:main',
            'trajectory_tracker = omni_autonomy_next.trajectory_tracker_node:main',
            'wall_localizer = omni_autonomy_next.wall_localizer_node:main',
        ],
    },
)
