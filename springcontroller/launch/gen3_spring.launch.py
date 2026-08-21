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

Audio/video recording (record_audio:=true / record_video:=true)
------------------------------------------------------------
  Needs packages this workspace doesn't have installed yet:
      sudo apt install ros-kilted-audio-capture ros-kilted-v4l2-camera \
          ros-kilted-compressed-image-transport ros-kilted-topic-tools
  audio_capture_node encodes straight to mp3 (mono, 16kHz, ~64kbps by
  default) -- already small enough for the bag, no separate downsampling
  step needed. Video is downsampled twice: v4l2_camera_node captures at
  video_image_size (default 640x480, well below most webcams' native
  resolution) and publishes both raw and, once compressed_image_transport
  is installed, a lazy JPEG .../compressed topic; a topic_tools throttle
  node then caps that to video_fps (default 10) before it's recorded --
  only the throttled+compressed topic goes in the bag, never the raw
  feed. Sight-unseen (no camera attached to this dev machine, packages
  not yet installed) -- verify video_image_size/jpeg_quality parameter
  names still match once actually tested live.
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

    record_audio_arg = DeclareLaunchArgument(
        "record_audio",
        default_value="false",
        description=(
            "If true, launch audio_capture_node (package ros-kilted-"
            "audio-capture, not installed on this dev machine as of "
            "2026-08-21 -- see module docstring) and mic-capture audio for "
            "the run. Defaults to false, same reasoning as armviz:= -- "
            "needs real hardware/a package this box doesn't have yet, so "
            "it's an explicit opt-in rather than something that fails "
            "loudly on every plain control-code launch."
        ),
    )

    audio_device_arg = DeclareLaunchArgument(
        "audio_device",
        default_value="",
        description=(
            "ALSA device string for audio_capture_node (empty = system "
            "default capture device, e.g. whichever mic is plugged in). "
            "Only used when record_audio:=true."
        ),
    )

    audio_bitrate_kbps_arg = DeclareLaunchArgument(
        "audio_bitrate_kbps",
        default_value="64",
        description=(
            "mp3 encode bitrate audio_capture_node compresses to before "
            "publishing -- mono/16kHz/64kbps is already small enough that "
            "no further downsampling step is needed to keep a 15+ minute "
            "run's audio out of the bag's way (roughly 7-8MB per 15 "
            "minutes at this default)."
        ),
    )

    record_video_arg = DeclareLaunchArgument(
        "record_video",
        default_value="false",
        description=(
            "If true, launch v4l2_camera_node plus a topic_tools throttle "
            "node and record downsampled webcam video for the run. "
            "Defaults to false, same reasoning as record_audio:= above -- "
            "needs a camera and packages not installed on this dev machine "
            "as of 2026-08-21 (see module docstring)."
        ),
    )

    video_device_arg = DeclareLaunchArgument(
        "video_device",
        default_value="/dev/video0",
        description="V4L2 device path for v4l2_camera_node. Only used when record_video:=true.",
    )

    video_image_size_arg = DeclareLaunchArgument(
        "video_image_size",
        default_value="[640, 480]",
        description=(
            "Capture resolution v4l2_camera_node is told to request from "
            "the camera, well below most webcams' native resolution -- the "
            "first (and cheapest) of the two downsampling steps applied to "
            "video before it's recorded (see video_fps below for the "
            "second). Passed through as YAML so it must stay valid YAML "
            "list syntax, e.g. '[1280, 720]'."
        ),
    )

    video_fps_arg = DeclareLaunchArgument(
        "video_fps",
        default_value="10",
        description=(
            "Frame rate the topic_tools throttle node caps the compressed "
            "video topic to before it's recorded -- the second of the two "
            "downsampling steps (see video_image_size above for the "
            "first). Applied downstream of v4l2_camera_node rather than "
            "via a camera-driver frame-rate parameter so it works "
            "regardless of what frame rates the attached camera actually "
            "supports."
        ),
    )

    video_jpeg_quality_arg = DeclareLaunchArgument(
        "video_jpeg_quality",
        default_value="60",
        description=(
            "JPEG quality (0-100) for compressed_image_transport's lazy "
            ".../compressed publisher on v4l2_camera_node -- the default "
            "95 is needlessly large for a study recording; 60 is "
            "noticeably smaller with no visible quality loss at this "
            "resolution. NOTE: parameter name assumed from "
            "compressed_image_transport's documented convention "
            "('<image_topic>.compressed.jpeg_quality') but not yet "
            "verified live -- no camera attached to this dev machine and "
            "the package isn't installed yet as of 2026-08-21 (see module "
            "docstring). If it's wrong, the param is simply ignored (a "
            "harmless unused-parameter warning), it won't error -- just "
            "confirm the resulting file size actually drops before "
            "relying on it for a long test run."
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

    # mp3-encoded mic capture -- see record_audio_arg / module docstring.
    # Already compressed at the source (no raw PCM ever hits a topic), so
    # unlike video there's no separate downsampling stage needed downstream.
    audio_capture_node = Node(
        package="audio_capture",
        executable="audio_capture_node",
        name="audio_capture",
        namespace="audio",
        output="screen",
        parameters=[{
            "src": "alsasrc",
            "dst": "appsink",
            "device": LaunchConfiguration("audio_device"),
            "format": "mp3",
            "bitrate": LaunchConfiguration("audio_bitrate_kbps"),
            "channels": 1,
            "depth": 16,
            "sample_rate": 16000,
            "sample_format": "S16LE",
        }],
        condition=IfCondition(LaunchConfiguration("record_audio")),
    )

    # Captures at a reduced resolution (video_image_size) and republishes
    # via image_transport, which -- once compressed_image_transport is
    # installed -- lazily advertises a JPEG .../compressed sibling topic the
    # instant something subscribes to it (the throttle node below does).
    # Only ever publishes raw frames to a live subscriber of image_raw
    # itself; nothing here subscribes to that, so the uncompressed stream
    # never actually flows.
    v4l2_camera_node = Node(
        package="v4l2_camera",
        executable="v4l2_camera_node",
        name="camera",
        output="screen",
        parameters=[{
            "video_device": LaunchConfiguration("video_device"),
            "image_size": LaunchConfiguration("video_image_size"),
            # See video_jpeg_quality_arg -- best-effort parameter name,
            # not yet verified live.
            "image_raw.compressed.jpeg_quality": LaunchConfiguration("video_jpeg_quality"),
        }],
        remappings=[
            ("image_raw", "/camera/image_raw"),
        ],
        condition=IfCondition(LaunchConfiguration("record_video")),
    )

    # Second downsampling step (see module docstring): caps the compressed
    # topic to video_fps before it's recorded, independent of whatever
    # frame rate the camera hardware/driver actually produces. Subscribing
    # to /camera/image_raw/compressed here is also what makes
    # compressed_image_transport's lazy publisher on v4l2_camera_node
    # actually turn on in the first place.
    #
    # Output topic ends in /compressed, not /compressed_throttled -- confirmed
    # live 2026-08-21: image_transport-aware viewers (rqt_image_view,
    # image_view) parse a topic's *last* path segment as a transport-plugin
    # name (<base_topic>/<transport>, e.g. .../compressed), so naming this
    # .../compressed_throttled made them try to load a nonexistent
    # "compressed_throttled" transport plugin and fail outright. Only "raw"
    # and "compressed" are real registered transport suffixes -- putting
    # "_throttled" on the base-topic segment instead keeps this a normal,
    # viewer-compatible /<base>/compressed topic.
    video_throttle_node = Node(
        package="topic_tools",
        executable="throttle",
        name="camera_throttle",
        output="screen",
        arguments=[
            "messages",
            "/camera/image_raw/compressed",
            LaunchConfiguration("video_fps"),
            "/camera/image_raw_throttled/compressed",
        ],
        condition=IfCondition(LaunchConfiguration("record_video")),
    )

    # Real rosbag of the key torque/state/status topics plus /rosout
    # (folds in every WARN/ERROR log line too, so recorded data and log
    # messages line up in one place) -- see record_rosbag_arg. Topics are
    # captured wherever they're published in the ROS graph, so this picks
    # up gen3_torque_control's joint_states_topic and
    # /gen3_torque_control/status.
    #
    # /virtual_spring_node/springs_updated is a latched std_msgs/String
    # publishing the full current list of spring names as JSON on every
    # add/remove/update (see virtual_spring_node.py's _publish_springs_
    # updated) -- recording it gives a timestamped history of the spring
    # roster for the run; diffing consecutive messages in the bag recovers
    # exactly which spring was created or destroyed and when.
    #
    # /audio/audio and /camera/image_raw_throttled/compressed are recorded
    # unconditionally here even though audio_capture_node/v4l2_camera_node/
    # video_throttle_node above only actually run when record_audio:=true /
    # record_video:=true -- same pattern as press_to_pin's topics used to
    # follow (a bag recorder subscribed to a topic nobody's publishing on
    # just records nothing for it, harmlessly) so record_audio/record_video
    # can be toggled without also having to edit this topic list.
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
            "/virtual_spring_node/repulsion_torques",
            "/virtual_spring_node/safety_status",
            "/virtual_spring_node/springs_updated",
            "/kinova/joint_torque_command",
            "/gen3_torque_control/status",
            "/audio/audio",
            "/camera/image_raw_throttled/compressed",
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
        record_audio_arg,
        audio_device_arg,
        audio_bitrate_kbps_arg,
        record_video_arg,
        video_device_arg,
        video_image_size_arg,
        video_fps_arg,
        video_jpeg_quality_arg,
        virtual_spring_node,
        torque_relay_node,
        enable_torque_control,
        load_collision_scene,
        meshcat_server_process,
        armviz_process,
        audio_capture_node,
        v4l2_camera_node,
        video_throttle_node,
        record_rosbag,
    ])
