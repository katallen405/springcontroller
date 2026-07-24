"""
press_to_pin.launch.py

Launches the press_to_pin node, which watches joint-effort residuals for a
human press on the arm and creates a virtual spring pinning the pressed
link to its current position (stiffness proportional to press force).

Assumes virtual_spring_node is already running (it provides the
add_spring service and the commanded-torque topic).
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    config_arg = DeclareLaunchArgument(
        "config",
        default_value=PathJoinSubstitution([
            FindPackageShare("springcontroller"),
            "config",
            "ur3e_press_to_pin.yaml",
        ]),
        description="Path to the press_to_pin configuration YAML.",
    )

    urdf_path_arg = DeclareLaunchArgument(
        "urdf_path",
        default_value="",
        description="Fallback URDF path if /robot_description is absent.",
    )

    node = Node(
        package="springcontroller",
        executable="press_to_pin",
        name="press_to_pin",
        output="screen",
        parameters=[
            LaunchConfiguration("config"),
            {"urdf_path": LaunchConfiguration("urdf_path")},
        ],
    )

    return LaunchDescription([config_arg, urdf_path_arg, node])
