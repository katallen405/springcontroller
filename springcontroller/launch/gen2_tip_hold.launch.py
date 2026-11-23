"""
gen2_tip_hold.launch.py
gen2_spring.launch.py with one tip spring (k = 10 N/m, b = 0) anchored at the
gripper's position (config/gen2_tip_hold_springs.yaml leaves `target` unset,
so virtual_spring_node uses the measured pose).

Every other gen2_spring.launch.py argument passes straight through, e.g.
    ros2 launch springcontroller gen2_tip_hold.launch.py start_bridge:=true

The anchor is set when springcontroller first sees the arm and again on every
torque enable (reanchor_hold_springs_on_enable, default true), so the arm
holds wherever it is when torque comes on. To hold somewhere else: disable
torque, move the arm there (e.g. with move_to_joint_angles), and enable.
"""

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    share = FindPackageShare("springcontroller")
    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([share, "launch", "gen2_spring.launch.py"])),
            launch_arguments={
                "config": PathJoinSubstitution(
                    [share, "config", "gen2_tip_hold_springs.yaml"]),
            }.items(),
        ),
    ])
