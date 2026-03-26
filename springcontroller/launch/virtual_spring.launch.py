"""
virtual_spring.launch.py
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    urdf_path_arg = DeclareLaunchArgument(
        "urdf_path",
        description="Path to the robot URDF or XACRO file.",
    )

    config_arg = DeclareLaunchArgument(
        "config",
        default_value=PathJoinSubstitution([
            FindPackageShare("springcontroller"),
            "config",
            "springs.yaml",
        ]),
        description="Path to the springs configuration YAML.",
    )

    node = Node(
        package="springcontroller",
        executable="virtual_spring_node",
        name="virtual_spring_node",
        output="screen",
        parameters=[
            {
                "urdf_path":   LaunchConfiguration("urdf_path"),
                "config_path": LaunchConfiguration("config"),
            },
            LaunchConfiguration("config"),  # also load as params-file for spring defs
        ],
        remappings=[
            ("~/joint_states",  "/joint_states"),
            ("~/joint_torques", "/virtual_spring/joint_torques"),
        ],
    )

    return LaunchDescription([urdf_path_arg, config_arg, node])
