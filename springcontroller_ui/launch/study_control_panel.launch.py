"""
study_control_panel.launch.py

Launches the three pure-software pieces of the study control panel UI
together: rosbridge (so the browser page can call ROS services/topics),
this package's orchestration_node (interlocks, e-stop, study-start preset,
reset-springs, FK helpers), and the static file server for web/index.html.

None of these three touch hardware -- unlike the robot's torque bridge or
driver and its spring launch file (which this UI deliberately never
launches/kills itself), so it's safe to bring all three up together in one
command. robot:=gen3|gen2|ur3e picks the node's config.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, LogInfo,
    OpaqueFunction, SetEnvironmentVariable,
)
from launch.launch_description_sources import AnyLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


ROBOT_CONFIGS = {
    "gen3": "study_control_panel.yaml",
    "gen2": "study_control_panel_gen2.yaml",
    "ur3e": "study_control_panel_ur3e.yaml",
}


def _orchestration_node(context):
    config = LaunchConfiguration("config").perform(context)
    if not config:
        config = os.path.join(
            get_package_share_directory("springcontroller_ui"), "config",
            ROBOT_CONFIGS[LaunchConfiguration("robot").perform(context)],
        )
    return [
        LogInfo(msg="study_control_panel_node config: " + config),
        Node(
            package="springcontroller_ui",
            executable="study_control_panel_node",
            name="study_control_panel_node",
            output="screen",
            emulate_tty=True,
            parameters=[config],
        ),
    ]


def generate_launch_description():
    rosbridge_port_arg = DeclareLaunchArgument(
        "rosbridge_port", default_value="9090",
        description="Port rosbridge_websocket listens on. Must match web/index.html's default.",
    )
    ui_port_arg = DeclareLaunchArgument(
        "ui_port", default_value="8090",
        description=(
            "Port the static control-panel page is served on. Defaults to "
            "8090, distinct from block-painting-helper-study's "
            "bph_userinterface (8080), so both UIs can run at once on one machine."
        ),
    )
    robot_arg = DeclareLaunchArgument(
        "robot", default_value="gen3", choices=list(ROBOT_CONFIGS),
        description="Which arm the panel drives; picks config/" + ", ".join(ROBOT_CONFIGS.values()) + ".",
    )
    config_arg = DeclareLaunchArgument(
        "config", default_value="",
        description=(
            "Path to study_control_panel_node's parameter YAML. Empty (the "
            "default) = the robot:= argument's config in this package."
        ),
    )

    rosbridge = IncludeLaunchDescription(
        AnyLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("rosbridge_server"),
                "launch", "rosbridge_websocket_launch.xml",
            )
        ),
        launch_arguments={"port": LaunchConfiguration("rosbridge_port")}.items(),
    )

    orchestration_node = OpaqueFunction(function=_orchestration_node)

    web_dir = os.path.join(get_package_share_directory("springcontroller_ui"), "web")
    ui_server_process = ExecuteProcess(
        cmd=["python3", os.path.join(web_dir, "ui_server.py"), LaunchConfiguration("ui_port")],
        output="screen",
    )

    # Printed immediately (not delayed) -- unlike ui_server_process's own
    # eventual "serving on port ..." line, this doesn't depend on the
    # static server actually finishing startup first, so it's visible right
    # away instead of scrolling past in the rest of this launch's output.
    url_log = LogInfo(
        msg=["Study control panel: http://localhost:", LaunchConfiguration("ui_port")]
    )

    return LaunchDescription([
        # See gen3_spring.launch.py's copy of this action for why -- keeps
        # this panel's own ROS2/DDS traffic off the Gen3's dedicated
        # Ethernet link, so an E-STOP doesn't hang this UI too, despite it
        # having nothing to do with the robot connection itself. Doesn't
        # affect rosbridge's websocket listener (a separate TCP socket,
        # not DDS) -- the panel stays reachable from other devices on the
        # LAN as before.
        SetEnvironmentVariable("ROS_LOCALHOST_ONLY", "1"),
        rosbridge_port_arg,
        ui_port_arg,
        robot_arg,
        config_arg,
        rosbridge,
        orchestration_node,
        ui_server_process,
        url_log,
    ])
