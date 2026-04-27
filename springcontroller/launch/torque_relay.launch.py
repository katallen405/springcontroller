from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    # ---------------------------------------------------------------------------
    # Launch arguments – override from the command line or a parent launch file
    # ---------------------------------------------------------------------------
    joint_order_arg = DeclareLaunchArgument(
        "joint_order",
        default_value="[]",
        description=(
            "Ordered list of joint names the relay will publish, e.g. "
            "\"['joint1', 'joint2', 'joint3']\". Must match the names "
            "present in the /virtual_spring/joint_torques JointState message."
        ),
    )

    torque_topic_arg = DeclareLaunchArgument(
        "torque_topic",
        default_value="/virtual_spring_node/joint_torques",
        description="Input topic (sensor_msgs/JointState) carrying spring torques.",
    )

    command_topic_arg = DeclareLaunchArgument(
        "command_topic",
        default_value="/forward_effort_controller/commands",
        description="Output topic (std_msgs/Float64MultiArray) sent to the effort controller.",
    )

    # ---------------------------------------------------------------------------
    # Node
    # ---------------------------------------------------------------------------
    torque_relay_node = Node(
        package="springcontroller", 
        executable="torque_relay",
        name="torque_relay",
        output="screen",
        emulate_tty=True,
        parameters=[
            {
                "joint_order":    LaunchConfiguration("joint_order"),
                "torque_topic":   LaunchConfiguration("torque_topic"),
                "command_topic":  LaunchConfiguration("command_topic"),
            }
        ],
    )

    return LaunchDescription(
        [
            joint_order_arg,
            torque_topic_arg,
            command_topic_arg,
            torque_relay_node,
        ]
    )
