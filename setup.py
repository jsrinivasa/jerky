from glob import glob
import os

from setuptools import (
    find_packages,
    setup,
)

package_name = 'aloha'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude='test'),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*.launch.py'))),
        (os.path.join('share', package_name, 'config'), glob(os.path.join('config', '*.yaml'))),
        (os.path.join('share', package_name, 'config', 'nav2'), glob(os.path.join('config', 'nav2', '*.yaml'))),
        (os.path.join('share', package_name, 'rviz'), glob(os.path.join('rviz', '*.rviz'))),
        (os.path.join('share', package_name, 'docs'), glob(os.path.join('docs', '*.md'))),
        (os.path.join('share', package_name, 'maps'), glob(os.path.join('maps', '*'))),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'scipy', 'numpy'],
    zip_safe=True,
    author_email='tonyzhao@stanford.edu',
    author='Tony Zhao',
    maintainer='Trossen Robotics',
    maintainer_email='trsupport@trossenrobotics.com',
    description='ALOHA: A Low-cost Open-source Hardware System for Bimanual Teleoperation',
    license='BSD',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'quick_start_planner = scripts.quick_start_planner:main',
            'path_planner_example = scripts.path_planner_example:main',
            'dual_arm_path_planner_example = scripts.dual_arm_path_planner_example:main',
            'sim_planner_interactive = scripts.sim_planner_interactive:main',
            'simple_nav_planner = aloha.simple_nav_planner:main',
            'fake_odometry = aloha.fake_odometry:main',
            'camera_static_tf_publisher = aloha.camera_static_tf_publisher:main',
            'localization_monitor = aloha.localization_monitor:main',
            'laser_scan_merger = aloha.laser_scan_merger:main',
            'auto_localize = aloha.auto_localize:main',
            'navigate_mission = aloha.navigate_mission:main',
            'robot_pose_marker = aloha.robot_pose_marker:main',
            'auto_explore = aloha.auto_explore:main',
        ],
    },
)
