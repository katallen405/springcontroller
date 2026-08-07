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
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# armviz.py needs meshcat/pinocchio, so it's run with this venv's python
# explicitly rather than relying on "python3" from PATH (which won't have
# those unless the venv happens to already be activated in the launching
# shell). See the top-level README.md's "Python venv" section if you're
# using a differently-named venv.
SPRINGCONTROLLER_VENV_PYTHON = "/home/katallen/.springcontroller_venv/bin/python3"

GEN3_JOINT_ORDER = [
    "joint_1", "joint_2", "joint_3", "joint_4",
    "joint_5", "joint_6", "joint_7",
]

# Robotiq 2F-85 actuated (revolute) joints. gen3_torque_control doesn't
# report or drive these, so they're locked rather than left out of the
# URDF entirely -- locking keeps the gripper's mass in the gravity
# compensation while dropping it from the active DOF / joint-state matching.
GRIPPER_LOCKED_JOINTS = [
    "robotiq_85_left_knuckle_joint",
    "robotiq_85_right_knuckle_joint",
    "robotiq_85_left_inner_knuckle_joint",
    "robotiq_85_right_inner_knuckle_joint",
    "robotiq_85_left_finger_tip_joint",
    "robotiq_85_right_finger_tip_joint",
]


def generate_launch_description():
    urdf_path_arg = DeclareLaunchArgument(
        "urdf_path",
        default_value="/home/katallen/sandbox/src/springcontroller/springcontroller/flat_urdf_files/gen3_kinova_flat.urdf",
        description=(
            "Path to the Gen3 URDF for pinocchio. Includes the Robotiq "
            "gripper so its mass counts toward gravity compensation; the "
            "gripper joints are locked (see GRIPPER_LOCKED_JOINTS) since "
            "gen3_torque_control doesn't report/drive them."
        ),
    )

    config_arg = DeclareLaunchArgument(
        "config",
        default_value="/home/katallen/sandbox/src/springcontroller/springcontroller/config/gen3_springs.yaml",
        description="Path to the springs configuration YAML.",
    )

    srdf_path_arg = DeclareLaunchArgument(
        "srdf_path",
        default_value=(
            "/home/katallen/sandbox/install/kinova_gen3_7dof_robotiq_2f_85_moveit_config/"
            "share/kinova_gen3_7dof_robotiq_2f_85_moveit_config/config/gen3.srdf"
        ),
        description=(
            "SRDF used to exclude adjacent-link collision pairs (e.g. "
            "base_link/shoulder_link) from self-collision checking. "
            "Without this, adjacent links that are always touching at the "
            "joint get flagged as permanent false-positive self-collisions "
            "and torques get zeroed every cycle."
        ),
    )

    add_gravity_compensation_arg = DeclareLaunchArgument(
        "add_gravity_compensation",
        default_value="true",
        description=(
            "Whether virtual_spring_node should add software gravity "
            "compensation torques. Kinova Gen3 via gen3_torque_control has "
            "no hardware gravity comp (unlike the UR3e's torque "
            "controller), so this defaults to true here."
        ),
    )

    torque_control_service_arg = DeclareLaunchArgument(
        "torque_control_service",
        default_value="/gen3_torque_control/enable",
        description=(
            "SetBool service used both to auto-enable torque control (if "
            "enable_torque_control:=true) and by virtual_spring_node's "
            "fail-safe to disable it again on repeated computation failure."
        ),
    )

    enable_torque_control_arg = DeclareLaunchArgument(
        "enable_torque_control",
        default_value="false",
        description=(
            "If true, automatically call torque_control_service with "
            "data:true a few seconds after startup, once virtual_spring_node "
            "should be up and publishing valid gravity-compensated torques. "
            "Defaults to false: enabling torque control on real hardware is "
            "left as a deliberate, explicit step -- call the service "
            "yourself when you're ready."
        ),
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
                "add_gravity_compensation": LaunchConfiguration("add_gravity_compensation"),
                "locked_joint_names": GRIPPER_LOCKED_JOINTS,
                "srdf_path": LaunchConfiguration("srdf_path"),
                "torque_disable_service": LaunchConfiguration("torque_control_service"),
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

    # Fires 3s after launch to give virtual_spring_node time to come up and
    # start publishing valid gravity-compensated torques before torque
    # control turns on for real -- enabling any earlier would apply
    # whatever stale/zero command exists on /kinova/joint_torque_command
    # in the meantime.
    enable_torque_control = TimerAction(
        period=3.0,
        actions=[
            ExecuteProcess(
                cmd=[
                    "ros2", "service", "call",
                    LaunchConfiguration("torque_control_service"),
                    "std_srvs/srv/SetBool", "{data: true}",
                ],
                output="screen",
                condition=IfCondition(LaunchConfiguration("enable_torque_control")),
            )
        ],
    )

    armviz_process = ExecuteProcess(
        cmd=[
            SPRINGCONTROLLER_VENV_PYTHON,
            "/home/katallen/sandbox/src/springcontroller/test/armviz.py",
            "--urdf", LaunchConfiguration("urdf_path"),
            "--ros-args",
            "-r", "/joint_states:=/kinova/joint_states_lowlevel",
        ],
        output="screen",
        condition=IfCondition(LaunchConfiguration("armviz")),
    )

    return LaunchDescription([
        urdf_path_arg,
        config_arg,
        srdf_path_arg,
        add_gravity_compensation_arg,
        torque_control_service_arg,
        enable_torque_control_arg,
        armviz_arg,
        virtual_spring_node,
        torque_relay_node,
        enable_torque_control,
        armviz_process,
    ])
