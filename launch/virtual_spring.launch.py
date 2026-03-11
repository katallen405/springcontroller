"""
virtual_spring.launch.py

Launches the virtual spring node with a config file.

Usage:
    ros2 launch virtual_spring_ros2 virtual_spring.launch.py \
        urdf_path:=/path/to/robot.urdf \
        config:=/path/to/springs.yaml
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    urdf_path_arg = DeclareLaunchArgument(
        "urdf_path",
        description="Absolute path to the robot URDF file.",
    )

    config_arg = DeclareLaunchArgument(
        "config",
        default_value=PathJoinSubstitution([
            FindPackageShare("virtual_spring_ros2"),
            "config",
            "springs.yaml",
        ]),
        description="Path to the spring configuration YAML.",
    )

    node = Node(
        package="virtual_spring_ros2",
        executable="virtual_spring_node.py",
        name="virtual_spring_node",
        output="screen",
        parameters=[
            {"urdf_path": LaunchConfiguration("urdf_path")},
            LaunchConfiguration("config"),
        ],
        remappings=[
            ("~/joint_states", "/joint_states"),
            ("~/joint_torques", "/virtual_spring/joint_torques"),
        ],
    )

    return LaunchDescription([urdf_path_arg, config_arg, node])
