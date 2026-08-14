"""
study_control_panel.launch.py

Launches the three pure-software pieces of the study control panel UI
together: rosbridge (so the browser page can call ROS services/topics),
this package's orchestration_node (interlocks, e-stop, study-start preset,
reset-springs, FK helpers), and the static file server for web/index.html.

None of these three touch hardware -- unlike gen3_torque_node,
gen3_spring.launch.py, and the ros2_kortex driver (which this UI
deliberately never launches/kills itself; see web/index.html's terminal
reminders), so it's safe to bring all three up together in one command.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription
from launch.launch_description_sources import AnyLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


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
    config_arg = DeclareLaunchArgument(
        "config",
        default_value=os.path.join(
            get_package_share_directory("springcontroller_ui"),
            "config", "study_control_panel.yaml",
        ),
        description="Path to study_control_panel_node's parameter YAML.",
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

    orchestration_node = Node(
        package="springcontroller_ui",
        executable="study_control_panel_node",
        name="study_control_panel_node",
        output="screen",
        emulate_tty=True,
        parameters=[LaunchConfiguration("config")],
    )

    web_dir = os.path.join(get_package_share_directory("springcontroller_ui"), "web")
    ui_server_process = ExecuteProcess(
        cmd=["python3", os.path.join(web_dir, "ui_server.py"), LaunchConfiguration("ui_port")],
        output="screen",
    )

    return LaunchDescription([
        rosbridge_port_arg,
        ui_port_arg,
        config_arg,
        rosbridge,
        orchestration_node,
        ui_server_process,
    ])
