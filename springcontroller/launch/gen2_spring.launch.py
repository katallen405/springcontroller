"""
gen2_spring.launch.py
Launches the virtual spring controller wired to the Kinova Gen2 JACO
(j2s7s300 + custom adaptor + Robotiq 2F-85) via the gen2_torque_control
bridge.

Topic wiring
------------
  /gen2/joint_states_lowlevel  →  virtual_spring_node, armviz
  virtual_spring_node/joint_torques  →  torque_relay
  torque_relay  →  /gen2/joint_torque_command  (Float64MultiArray, 7 values)

Differences from gen3_spring.launch.py
--------------------------------------
* add_gravity_compensation defaults to FALSE: the Gen2's firmware
  compensates gravity itself in torque mode (a zero command holds the
  arm), with the gripper payload set on gen2_torque_node. Adding
  pinocchio's gravity torque on top would double it.
* start_bridge:=true (default false) also starts gen2_torque_node from
  here; otherwise run it first in its own terminal:
      ros2 launch gen2_torque_control gen2_torque_control.launch.py
* No study tooling (audio/video capture, participant routing) -- this is
  for bringing the Gen2 up, not running the study.
* Default springs config has no springs (gen2_springs_none.yaml).
"""

import os

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

# Same venv/paths gen3_spring.launch.py uses -- see its comments.
SPRINGCONTROLLER_VENV_PYTHON = "/home/katallen/.springcontroller_venv/bin/python3"
REPO = "/home/katallen/sandbox/src/springcontroller"
PKG = REPO + "/springcontroller"
PINNED_MESHCAT_SERVER_SCRIPT = REPO + "/test/pinned_meshcat_server.py"

DEFAULT_ROSBAG_DIR = os.path.expanduser("~/gen2_bags")
os.makedirs(DEFAULT_ROSBAG_DIR, exist_ok=True)

GEN2_JOINT_ORDER = [f"j2s7s300_joint_{i}" for i in range(1, 8)]

# Robotiq 2F-85 movable joints in gen2_robotiq_flat.urdf. gen2_torque_control
# doesn't report them, so they're locked out of the active DOF.
GRIPPER_LOCKED_JOINTS = [
    "gripper_finger1_joint",
    "gripper_finger2_joint",
    "gripper_finger1_inner_knuckle_joint",
    "gripper_finger2_inner_knuckle_joint",
    "gripper_finger1_finger_tip_joint",
    "gripper_finger2_finger_tip_joint",
]


def generate_launch_description():
    args = [
        DeclareLaunchArgument(
            "urdf_path", default_value=PKG + "/flat_urdf_files/gen2_robotiq_flat.urdf",
            description="Gen2 + adaptor + Robotiq URDF (see tools/make_gen2_robotiq_urdf.sh)."),
        DeclareLaunchArgument(
            "srdf_path", default_value=PKG + "/config/gen2_robotiq.srdf",
            description="Disabled self-collision pairs (see tools/make_srdf_disabled_pairs.py)."),
        DeclareLaunchArgument(
            "config", default_value=PKG + "/config/gen2_springs_none.yaml",
            description="Springs configuration YAML. Try gen2_robotiq_springs.yaml next."),
        DeclareLaunchArgument(
            "joint_states_topic", default_value="/gen2/joint_states_lowlevel",
            description="Where gen2_torque_node publishes joint states (its launch file's remap)."),
        DeclareLaunchArgument(
            "caution_threshold", default_value="0.05",
            description="Shared by virtual_spring_node and armviz, as in gen3_spring.launch.py."),
        DeclareLaunchArgument(
            "add_gravity_compensation", default_value="false",
            description="Keep false: the Gen2 firmware already compensates gravity."),
        DeclareLaunchArgument(
            "torque_control_service", default_value="/gen2_torque_control/enable",
            description="SetBool service to enable/disable torque control."),
        DeclareLaunchArgument(
            "torque_status_topic", default_value="/gen2_torque_control/status",
            description="ENABLED/DISABLED topic driving the spring ramp-in and liveness check."),
        DeclareLaunchArgument(
            "spring_ramp_duration_sec", default_value="1.5",
            description="Spring-force ramp after each torque enable."),
        DeclareLaunchArgument(
            "start_bridge", default_value="false",
            description="Also start gen2_torque_node from this launch file."),
        DeclareLaunchArgument(
            "gravity_payload", default_value="[0.55, 0.0, 0.0, 0.0]",
            description="Passed to gen2_torque_node when start_bridge:=true."),
        DeclareLaunchArgument(
            "enable_torque_control", default_value="false",
            description="Auto-enable torque once virtual_spring_node is publishing (see "
                        "test/wait_and_enable_torque.py). Default false: enable deliberately."),
        DeclareLaunchArgument(
            "enable_torque_control_allow_danger", default_value="false",
            description="Let enable_torque_control proceed from a DANGER/COLLISION pose."),
        DeclareLaunchArgument(
            "collision_scene", default_value=PKG + "/config/gen2_collision_scene.yaml",
            description="Scene objects YAML (placeholder geometry -- measure first)."),
        DeclareLaunchArgument(
            "load_collision_scene", default_value="false",
            description="Load collision_scene 3 s after startup."),
        DeclareLaunchArgument(
            "armviz", default_value="false",
            description="Also launch the MeshCat visualizer."),
        DeclareLaunchArgument("meshcat_zmq_url", default_value="tcp://127.0.0.1:6001"),
        DeclareLaunchArgument("meshcat_port", default_value="7101"),
        DeclareLaunchArgument(
            "record_rosbag", default_value="true",
            description="Record the bridge/spring topics under rosbag_dir."),
        DeclareLaunchArgument("rosbag_dir", default_value=DEFAULT_ROSBAG_DIR),
    ]

    bridge = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([
            FindPackageShare("gen2_torque_control"), "launch",
            "gen2_torque_control.launch.py"])),
        launch_arguments={"gravity_payload": LaunchConfiguration("gravity_payload")}.items(),
        condition=IfCondition(LaunchConfiguration("start_bridge")),
    )

    virtual_spring_node = Node(
        package="springcontroller",
        executable="virtual_spring_node",
        name="virtual_spring_node",
        output="full",
        emulate_tty=True,
        parameters=[
            {
                "urdf_path": LaunchConfiguration("urdf_path"),
                "config_path": LaunchConfiguration("config"),
                "add_gravity_compensation": LaunchConfiguration("add_gravity_compensation"),
                "locked_joint_names": GRIPPER_LOCKED_JOINTS,
                "srdf_path": LaunchConfiguration("srdf_path"),
                "torque_disable_service": LaunchConfiguration("torque_control_service"),
                "torque_status_topic": LaunchConfiguration("torque_status_topic"),
                "torque_bridge_name": "gen2_torque_control",
                "spring_ramp_duration_sec": LaunchConfiguration("spring_ramp_duration_sec"),
                "caution_threshold": LaunchConfiguration("caution_threshold"),
            },
            LaunchConfiguration("config"),
        ],
        remappings=[
            ("/joint_states", LaunchConfiguration("joint_states_topic")),
        ],
    )

    torque_relay_node = Node(
        package="springcontroller",
        executable="torque_relay",
        name="torque_relay",
        output="screen",
        emulate_tty=True,
        parameters=[{
            "joint_order": GEN2_JOINT_ORDER,
            "torque_topic": "/virtual_spring_node/joint_torques",
            "command_topic": "/gen2/joint_torque_command",
        }],
    )

    enable_torque_control = ExecuteProcess(
        cmd=[
            SPRINGCONTROLLER_VENV_PYTHON,
            REPO + "/test/wait_and_enable_torque.py",
            LaunchConfiguration("torque_control_service"),
            "/virtual_spring_node/joint_torques",
            "/virtual_spring_node/safety_status",
            "--timeout-sec", "10.0",
            "--allow-danger-enable", LaunchConfiguration("enable_torque_control_allow_danger"),
        ],
        output="screen",
        condition=IfCondition(LaunchConfiguration("enable_torque_control")),
    )

    load_collision_scene = TimerAction(
        period=3.0,
        actions=[ExecuteProcess(
            cmd=[
                "ros2", "service", "call",
                "/virtual_spring_node/load_collision_scene",
                "springcontroller_interfaces/srv/LoadCollisionScene",
                ["{yaml_path: '", LaunchConfiguration("collision_scene"), "'}"],
            ],
            output="screen",
            condition=IfCondition(LaunchConfiguration("load_collision_scene")),
        )],
    )

    meshcat_server_process = ExecuteProcess(
        cmd=[
            SPRINGCONTROLLER_VENV_PYTHON, "-u", PINNED_MESHCAT_SERVER_SCRIPT,
            "--zmq-url", LaunchConfiguration("meshcat_zmq_url"),
            "--port", LaunchConfiguration("meshcat_port"),
        ],
        output="screen",
        condition=IfCondition(LaunchConfiguration("armviz")),
    )

    armviz_process = TimerAction(
        period=1.5,
        actions=[ExecuteProcess(
            cmd=[
                SPRINGCONTROLLER_VENV_PYTHON, "-u", REPO + "/test/armviz.py",
                "--urdf", LaunchConfiguration("urdf_path"),
                "--zmq-url", LaunchConfiguration("meshcat_zmq_url"),
                "--caution-threshold", LaunchConfiguration("caution_threshold"),
                "--ros-args",
                "-r", ["/joint_states:=", LaunchConfiguration("joint_states_topic")],
            ],
            output="screen",
            condition=IfCondition(LaunchConfiguration("armviz")),
        )],
    )

    record_rosbag = ExecuteProcess(
        cmd=[
            "ros2", "bag", "record",
            # Unbuffered so a hard-killed recorder still leaves a readable
            # bag -- same reasoning as gen3_spring.launch.py.
            "--max-cache-size", "0",
            LaunchConfiguration("joint_states_topic"),
            "/virtual_spring_node/joint_torques",
            "/virtual_spring_node/repulsion_torques",
            "/virtual_spring_node/joint_limit_repulsion_torques",
            "/virtual_spring_node/safety_status",
            "/virtual_spring_node/springs_updated",
            "/virtual_spring_node/spring_forces",
            "/gen2/joint_torque_command",
            "/gen2_torque_control/status",
            "/gen2_torque_control/move_status",
            "/gen2_torque_control/firmware_gravity_torque",
            "/rosout",
        ],
        cwd=LaunchConfiguration("rosbag_dir"),
        output="screen",
        condition=IfCondition(LaunchConfiguration("record_rosbag")),
    )

    return LaunchDescription([
        # Same DDS-on-loopback backstop as gen3_spring.launch.py.
        SetEnvironmentVariable("ROS_LOCALHOST_ONLY", "1"),
        *args,
        bridge,
        virtual_spring_node,
        torque_relay_node,
        enable_torque_control,
        load_collision_scene,
        meshcat_server_process,
        armviz_process,
        record_rosbag,
    ])
