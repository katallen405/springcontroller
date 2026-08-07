"""
ur3e_spring.launch.py

Launches the virtual spring controller wired to a UR3e via ros2_control's
forward_effort_controller. (Formerly named virtual_spring.launch.py --
renamed since it's UR3e-specific, matching gen3_spring.launch.py.)

Topic wiring
------------
  /joint_states                       -> virtual_spring_node  (joint positions/velocities)
  virtual_spring_node/joint_torques   -> torque_relay
  torque_relay -> /forward_effort_controller/commands  (Float64MultiArray, 6 values)
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

# armviz.py needs meshcat/pinocchio, so it's run with this venv's python
# explicitly rather than relying on "python3" from PATH (which won't have
# those unless the venv happens to already be activated in the launching
# shell). See the top-level README.md's "Python venv" section if you're
# using a differently-named venv.
SPRINGCONTROLLER_VENV_PYTHON = "/home/katallen/.springcontroller_venv/bin/python3"

UR3E_JOINT_ORDER = [
    "ur3e_shoulder_pan_joint",
    "ur3e_shoulder_lift_joint",
    "ur3e_elbow_joint",
    "ur3e_wrist_1_joint",
    "ur3e_wrist_2_joint",
    "ur3e_wrist_3_joint",
]


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

    armviz_arg = DeclareLaunchArgument(
        "armviz",
        default_value="false",
        description=(
            "If true, also launch armviz.py (MeshCat-based 3D visualizer for "
            "virtual_spring_node -- shows the robot, spring attachment "
            "points, and targets in a browser). Uses the same urdf_path as "
            "virtual_spring_node. Defaults to false since it opens a browser "
            "tab and isn't needed for normal operation."
        ),
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
            LaunchConfiguration("config"),  # also load as params-file for spring defs
        ],
        remappings=[
            # virtual_spring_node subscribes to plain /joint_states already
            # (the real topic name from ros2_control's joint state
            # broadcaster on the UR3e), so no remap needed there.
            ("~/joint_torques", "/virtual_spring/joint_torques"),
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
                "joint_order":   UR3E_JOINT_ORDER,
                "torque_topic":  "/virtual_spring/joint_torques",
                "command_topic": "/forward_effort_controller/commands",
            }
        ],
    )

    armviz_process = ExecuteProcess(
        cmd=[
            SPRINGCONTROLLER_VENV_PYTHON,
            "/home/katallen/sandbox/src/springcontroller/test/armviz.py",
            "--urdf", LaunchConfiguration("urdf_path"),
        ],
        output="screen",
        condition=IfCondition(LaunchConfiguration("armviz")),
    )

    return LaunchDescription([
        urdf_path_arg,
        config_arg,
        armviz_arg,
        virtual_spring_node,
        torque_relay_node,
        armviz_process,
    ])
