"""
gen3_spring.launch.py

Launches the virtual spring controller wired to the Kinova Gen3 via the
gen3_torque_control node (not ros2_control).

Topic wiring
------------
  <joint_states_topic> (default /kinova/joint_states_lowlevel)  →  virtual_spring_node, armviz
  virtual_spring_node/joint_torques  →  torque_relay
  torque_relay  →  /kinova/joint_torque_command  (Float64MultiArray, 7 values)

Prerequisites
-------------
  ros2 run gen3_torque_control gen3_torque_node --ros-args \
      -r joint_states:=/kinova/joint_states_lowlevel   # must be running first,
      # with this remap -- see web/index.html's documented launch command.
      # Without it, gen3_torque_node publishes on plain /joint_states, which
      # nothing here subscribes to (both virtual_spring_node and armviz
      # listen on joint_states_topic:= below) -- virtual_spring_node then
      # never receives joint states, so no spring target ever resolves and
      # nothing gets published, silently. Confirmed as a real regression
      # 2026-08-19: virtual_spring_node's own remap was dropped in a WIP
      # snapshot commit on 2026-08-03 and never restored, while armviz's
      # matching remap was restored separately -- the two fell out of sync.
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, SetEnvironmentVariable, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# armviz.py needs meshcat/pinocchio, so it's run with this venv's python
# explicitly rather than relying on "python3" from PATH (which won't have
# those unless the venv happens to already be activated in the launching
# shell). See the top-level README.md's "Python venv" section if you're
# using a differently-named venv.
SPRINGCONTROLLER_VENV_PYTHON = "/home/katallen/.springcontroller_venv/bin/python3"
# meshcat's own `meshcat-server` console script has no --port flag at all
# (it always auto-picks the HTTP port starting from 7000) -- this repo's
# test/pinned_meshcat_server.py wraps ZMQWebSocketBridge directly so the
# HTTP port can actually be pinned, run with this venv's python for the
# same PATH reason as SPRINGCONTROLLER_VENV_PYTHON above.
PINNED_MESHCAT_SERVER_SCRIPT = "/home/katallen/sandbox/src/springcontroller/test/pinned_meshcat_server.py"

# ros2 bag record needs its output directory to exist before the process
# starts (a bad cwd is an immediate hard failure, not a graceful retry).
# Created eagerly here, at launch-description-generation time, rather than
# via a separate ExecuteProcess -- avoids an ordering race against the
# recorder actually starting. Only covers the *default* rosbag_dir; a
# CLI-overridden path needs to exist on its own.
DEFAULT_ROSBAG_DIR = os.path.expanduser("~/gen3_bags")
os.makedirs(DEFAULT_ROSBAG_DIR, exist_ok=True)

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

    joint_states_topic_arg = DeclareLaunchArgument(
        "joint_states_topic",
        default_value="/kinova/joint_states_lowlevel",
        description=(
            "Topic gen3_torque_node actually publishes joint states on when "
            "started with its documented -r joint_states:=... remap (see "
            "web/index.html and the module docstring above) -- both "
            "virtual_spring_node and armviz remap their /joint_states "
            "subscription to this single argument so the two can't fall out "
            "of sync again the way they did 2026-08-03..2026-08-19 (see "
            "git history: virtual_spring_node's remap was dropped in a WIP "
            "snapshot commit and never restored, while armviz's was)."
        ),
    )

    caution_threshold_arg = DeclareLaunchArgument(
        "caution_threshold",
        default_value="0.05",
        description=(
            "Must stay in sync between virtual_spring_node (where it "
            "actually gates get_repulsion_torques(), see "
            "urdf_arm_configuration.py) and armviz (where it only sizes the "
            "translucent gold halo drawn around each collision object) -- "
            "a single launch arg for both, same reasoning as "
            "joint_states_topic above. gen3_springs.yaml does not currently "
            "override this key, so the launch-arg default here is what "
            "actually applies unless overridden on the command line."
        ),
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

    torque_status_topic_arg = DeclareLaunchArgument(
        "torque_status_topic",
        default_value="/gen3_torque_control/status",
        description=(
            "gen3_torque_control's ENABLED/DISABLED status topic. "
            "virtual_spring_node watches this to ramp spring-force torque "
            "in over spring_ramp_duration_sec each time torque control is "
            "actually enabled on the arm (gravity comp is never ramped -- "
            "see torque_ramp_gravity_comp_lesson). Empty disables ramping."
        ),
    )

    spring_ramp_duration_arg = DeclareLaunchArgument(
        "spring_ramp_duration_sec",
        default_value="1.5",
        description=(
            "Duration over which spring-force torque ramps from 0 to full "
            "after each torque-enable event on torque_status_topic."
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

    enable_torque_control_allow_danger_arg = DeclareLaunchArgument(
        "enable_torque_control_allow_danger",
        default_value="false",
        description=(
            "If false (default), enable_torque_control:=true refuses to "
            "enable when virtual_spring_node's ~/safety_status reports the "
            "resting pose is already inside danger_threshold or in "
            "collision -- that's what's been tripping the Kinova's own "
            "actuator fault protection (2026-08-07, 2026-08-13). Set true "
            "only when deliberately testing from a near-collision pose."
        ),
    )

    collision_scene_arg = DeclareLaunchArgument(
        "collision_scene",
        default_value="/home/katallen/sandbox/src/springcontroller/springcontroller/config/gen3_collision_scene.yaml",
        description=(
            "Path to the scene collision-objects YAML (tables, fixtures, "
            "etc.) loaded via the load_collision_scene service -- see "
            "load_collision_scene:= to actually trigger the load."
        ),
    )

    load_collision_scene_arg = DeclareLaunchArgument(
        "load_collision_scene",
        default_value="false",
        description=(
            "If true, call ~/load_collision_scene with the collision_scene "
            "YAML a few seconds after startup so the self/scene-collision "
            "hard clamp has scene objects loaded from the start. Defaults "
            "to false so a stale or placeholder scene file never silently "
            "clamps torques -- opt in once the scene YAML matches the "
            "actual room."
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

    meshcat_zmq_url_arg = DeclareLaunchArgument(
        "meshcat_zmq_url",
        default_value="tcp://127.0.0.1:6001",
        description=(
            "zmq_url for a dedicated meshcat-server this launch file starts "
            "(only when armviz:=true), which armviz.py then attaches to "
            "instead of spawning its own server on a random port. Pinning "
            "this lets other processes (e.g. the study_control_panel_ui's "
            "embedded meshcat iframe) reach a stable, predictable URL "
            "instead of one that changes every run."
        ),
    )

    meshcat_port_arg = DeclareLaunchArgument(
        "meshcat_port",
        default_value="7101",
        description=(
            "HTTP/static-asset port for the pinned meshcat-server (see "
            "meshcat_zmq_url). The viewer is reachable at "
            "http://localhost:<meshcat_port>/static/."
        ),
    )

    record_rosbag_arg = DeclareLaunchArgument(
        "record_rosbag",
        default_value="true",
        description=(
            "If true (default), record the key torque/state/status topics "
            "plus /rosout to a rosbag under rosbag_dir for the duration of "
            "this launch. Replaces ad-hoc 'ros2 topic echo >> file' capture "
            "(imprecise, single-topic, hard to parse) used through "
            "2026-08-07 -- a real bag gives properly-timestamped multi-topic "
            "data, replayable/plottable directly in PlotJuggler. Purely "
            "additive (a subscriber, not in the control path); default true "
            "since unlike enabling torque control there's no downside "
            "beyond disk space."
        ),
    )

    rosbag_dir_arg = DeclareLaunchArgument(
        "rosbag_dir",
        default_value=DEFAULT_ROSBAG_DIR,
        description=(
            "Directory bag files are written under (ros2 bag auto-names "
            "each run within it). A non-default override must already "
            "exist -- only the default is created automatically."
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
                "torque_status_topic": LaunchConfiguration("torque_status_topic"),
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
        parameters=[
            {
                "joint_order":   GEN3_JOINT_ORDER,
                "torque_topic":  "/virtual_spring_node/joint_torques",
                "command_topic": "/kinova/joint_torque_command",
            }
        ],
    )

    # Waits for virtual_spring_node to actually be alive and publishing
    # valid torques (up to 10s) before enabling torque control, instead of
    # blindly firing after a fixed delay. Confirmed live 2026-08-07: a
    # crashed virtual_spring_node (bad springs config) still got torque
    # control enabled under the old fixed-delay approach -- the arm briefly
    # entered torque mode with nobody actually commanding it before
    # gen3_torque_control's own watchdog caught the missing stream and
    # disabled again ~200ms later. Also refuses to enable from an already-
    # unsafe resting pose unless enable_torque_control_allow_danger:=true
    # -- see wait_and_enable_torque.py.
    enable_torque_control = ExecuteProcess(
        cmd=[
            SPRINGCONTROLLER_VENV_PYTHON,
            "/home/katallen/sandbox/src/springcontroller/test/wait_and_enable_torque.py",
            LaunchConfiguration("torque_control_service"),
            "/virtual_spring_node/joint_torques",
            "/virtual_spring_node/safety_status",
            "--timeout-sec", "10.0",
            "--allow-danger-enable", LaunchConfiguration("enable_torque_control_allow_danger"),
        ],
        output="screen",
        condition=IfCondition(LaunchConfiguration("enable_torque_control")),
    )

    # Fires 3s after launch, same reasoning as enable_torque_control: give
    # virtual_spring_node time to come up before calling its service.
    load_collision_scene = TimerAction(
        period=3.0,
        actions=[
            ExecuteProcess(
                cmd=[
                    "ros2", "service", "call",
                    "/virtual_spring_node/load_collision_scene",
                    "springcontroller_interfaces/srv/LoadCollisionScene",
                    ["{yaml_path: '", LaunchConfiguration("collision_scene"), "'}"],
                ],
                output="screen",
                condition=IfCondition(LaunchConfiguration("load_collision_scene")),
            )
        ],
    )

    # Started immediately (not delayed) so it's already listening by the time
    # armviz_process (below) tries to attach to it a moment later.
    meshcat_server_process = ExecuteProcess(
        cmd=[
            # -u: unbuffered stdout. Without it, this script's stdout is
            # block-buffered under ros2 launch (stdout isn't a tty) --
            # pinned_meshcat_server.py only ever prints two short lines
            # total, so they'd sit in the buffer and never actually appear
            # in the launch log until the process exited, making a
            # perfectly-running server look like it had silently failed to
            # start (confirmed 2026-08-19).
            SPRINGCONTROLLER_VENV_PYTHON, "-u",
            PINNED_MESHCAT_SERVER_SCRIPT,
            "--zmq-url", LaunchConfiguration("meshcat_zmq_url"),
            "--port", LaunchConfiguration("meshcat_port"),
        ],
        output="screen",
        condition=IfCondition(LaunchConfiguration("armviz")),
    )

    # Delayed 1.5s so meshcat_server_process has time to bind before armviz.py
    # tries to attach via --zmq-url -- attaching to a not-yet-listening
    # zmq_url would otherwise race the server's own startup.
    armviz_process = TimerAction(
        period=1.5,
        actions=[
            ExecuteProcess(
                cmd=[
                    # -u: see meshcat_server_process above -- same
                    # block-buffering issue applies to armviz.py's own
                    # startup/status prints.
                    SPRINGCONTROLLER_VENV_PYTHON, "-u",
                    "/home/katallen/sandbox/src/springcontroller/test/armviz.py",
                    "--urdf", LaunchConfiguration("urdf_path"),
                    "--zmq-url", LaunchConfiguration("meshcat_zmq_url"),
                    "--caution-threshold", LaunchConfiguration("caution_threshold"),
                    "--ros-args",
                    # Same joint_states_topic argument virtual_spring_node's
                    # own remappings=[...] uses above -- a single source of
                    # truth instead of two hardcoded copies that can (and
                    # did, 2026-08-03..2026-08-19) drift apart.
                    "-r", ["/joint_states:=", LaunchConfiguration("joint_states_topic")],
                ],
                output="screen",
                condition=IfCondition(LaunchConfiguration("armviz")),
            )
        ],
    )

    # Real rosbag of the key torque/state/status topics plus /rosout
    # (folds in every WARN/ERROR log line too, so recorded data and log
    # messages line up in one place) -- see record_rosbag_arg. Topics are
    # captured wherever they're published in the ROS graph, so this picks
    # up gen3_torque_control's joint_states_topic and
    # /gen3_torque_control/status,
    # and press_to_pin's /residuals and /press_events, even though neither
    # node is started by this launch file -- covers a press_to_pin
    # threshold-calibration pass (run press_to_pin.launch.py alongside this)
    # with real recorded per-joint residual data, not just a live PlotJuggler
    # view with nothing kept afterward.
    #
    # /press_to_pin/test_marker has no publisher in this graph -- it's for
    # `ros2 topic pub --once /press_to_pin/test_marker std_msgs/msg/String
    # "{data: 'j2_link2_bottom_toward_base'}"` run by hand right before each
    # calibration press, so the bag has a precise, labeled timestamp for
    # what was pressed/where/which direction instead of relying on a
    # stopwatch or memory to line presses up with residual peaks afterward.
    record_rosbag = ExecuteProcess(
        cmd=[
            "ros2", "bag", "record",
            # Write to disk immediately instead of batching in an in-memory
            # cache -- trades a little write throughput for the bag actually
            # being (mostly) readable if the recorder has to be hard-killed.
            # Confirmed live 2026-08-19: rosbag2's writer needs a clean
            # shutdown to finalize, so a Ctrl-\ during an e-stop incident
            # meant the bag for that whole session was lost outright, even
            # though the per-node ~/.ros/log/ text logs survived fine.
            "--max-cache-size", "0",
            LaunchConfiguration("joint_states_topic"),
            "/virtual_spring_node/joint_torques",
            "/kinova/joint_torque_command",
            "/gen3_torque_control/status",
            "/press_to_pin/residuals",
            "/press_to_pin/press_events",
            "/press_to_pin/test_marker",
            "/rosout",
        ],
        cwd=LaunchConfiguration("rosbag_dir"),
        output="screen",
        condition=IfCondition(LaunchConfiguration("record_rosbag")),
    )

    return LaunchDescription([
        # Keep all ROS2/DDS traffic on loopback, off the Gen3's dedicated
        # Ethernet link entirely -- confirmed 2026-08-19: when the physical
        # E-STOP kills that interface, Cyclone DDS's own multicast
        # discovery/shutdown traffic hung on it too (even for nodes with
        # nothing to do with the robot, e.g. plain `ros2 bag record`),
        # needing SIGKILL across the board. gen3_torque_control's actual
        # robot connection is a separate raw Kortex socket, unrelated to
        # DDS -- this doesn't touch that. Backstops the same setting in
        # ~/.bashrc (this repo's copy survives a machine reimage/different
        # account; doesn't help gen3_torque_node's own bare `ros2 run`,
        # which still needs the shell-level export -- see README).
        SetEnvironmentVariable("ROS_LOCALHOST_ONLY", "1"),
        urdf_path_arg,
        config_arg,
        joint_states_topic_arg,
        caution_threshold_arg,
        srdf_path_arg,
        add_gravity_compensation_arg,
        torque_control_service_arg,
        torque_status_topic_arg,
        spring_ramp_duration_arg,
        enable_torque_control_arg,
        enable_torque_control_allow_danger_arg,
        collision_scene_arg,
        load_collision_scene_arg,
        armviz_arg,
        meshcat_zmq_url_arg,
        meshcat_port_arg,
        record_rosbag_arg,
        rosbag_dir_arg,
        virtual_spring_node,
        torque_relay_node,
        enable_torque_control,
        load_collision_scene,
        meshcat_server_process,
        armviz_process,
        record_rosbag,
    ])
