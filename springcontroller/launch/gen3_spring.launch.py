"""
gen3_spring.launch.py

Launches the virtual spring controller wired to the Kinova Gen3 via the
gen3_torque_control node (not ros2_control).

Topic wiring
------------
  /kinova/joint_states_lowlevel  →  virtual_spring_node  (joint positions/velocities)
  virtual_spring_node/joint_torques  →  torque_relay
  torque_relay  →  /kinova/joint_torque_command  (Float64MultiArray, 7 values)

Prerequisites
-------------
  ros2 run gen3_torque_control gen3_torque_node   # must be running first
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

GEN3_JOINT_ORDER = [
    "joint_1", "joint_2", "joint_3", "joint_4",
    "joint_5", "joint_6", "joint_7",
]


def generate_launch_description():
    urdf_path_arg = DeclareLaunchArgument(
        "urdf_path",
        default_value=PathJoinSubstitution([
            FindPackageShare("gen3_torque_control"),
            "urdf",
            "gen3_pinocchio.urdf",
        ]),
        description="Path to the Gen3 URDF for pinocchio (joint names must be joint_1..joint_7).",
    )

    config_arg = DeclareLaunchArgument(
        "config",
        default_value=PathJoinSubstitution([
            FindPackageShare("springcontroller"),
            "config",
            "gen3_springs.yaml",
        ]),
        description="Path to the springs configuration YAML.",
    )

    virtual_spring_node = Node(
        package="springcontroller",
        executable="virtual_spring_node",
        name="virtual_spring_node",
        output="screen",
        emulate_tty=True,
        parameters=[
            {
                "urdf_path":   LaunchConfiguration("urdf_path"),
                "config_path": LaunchConfiguration("config"),
            },
            LaunchConfiguration("config"),
        ],
        remappings=[
            # gen3_torque_node publishes here instead of /joint_states
            ("/joint_states", "/kinova/joint_states_lowlevel"),
        ],
    )

    torque_relay_node = Node(
        package="springcontroller",
        executable="torque_relay",
        name="torque_relay",
        output="screen",
        emulate_tty=True,
        parameters=[
            {
                "joint_order":   GEN3_JOINT_ORDER,
                "torque_topic":  "/virtual_spring_node/joint_torques",
                "command_topic": "/kinova/joint_torque_command",
            }
        ],
    )

    return LaunchDescription([
        urdf_path_arg,
        config_arg,
        virtual_spring_node,
        torque_relay_node,
    ])
