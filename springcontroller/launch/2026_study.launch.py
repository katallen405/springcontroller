"""
2026_study.launch.py

Thin wrapper around gen3_spring.launch.py for the 2026 study sessions --
same node graph and all the same safety-relevant logic (see that file's own
docstring), just with different *defaults* for the args study_procedure.txt
previously had people type by hand every relaunch: record_audio,
record_video, armviz, and load_collision_scene default to true here instead
of false.

Kept as an include rather than a copy so any future fix to
gen3_spring.launch.py (there's a documented history of exactly this kind of
drift breaking data collection -- see that file's own docstring, e.g. the
joint_states_topic remap regression) applies here automatically instead of
needing to be re-applied by hand in two places.

Only the args study day actually touches per participant/condition are
redeclared here -- config, participant_id, condition_name, plus the four
defaults flipped above -- so `ros2 launch springcontroller
2026_study.launch.py -s` shows the ones you'll actually use. Every other
gen3_spring.launch.py arg (urdf_path, caution_threshold, video_device,
enable_torque_control, ...) still works if passed by name -- `ros2 launch`
resolves undeclared name:=value CLI args straight into the shared launch
context, which gen3_spring.launch.py's own DeclareLaunchArgument calls then
pick up -- it's just not listed by -s on this wrapper. See
gen3_spring.launch.py for the full list and their descriptions.

  ros2 launch springcontroller 2026_study.launch.py \
      config:=~/gen3_study_data/<participant_id>/condition1.yaml \
      participant_id:=<participant_id> condition_name:=condition1
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    config_arg = DeclareLaunchArgument(
        "config",
        default_value=(
            "/home/katallen/sandbox/src/springcontroller/springcontroller/"
            "config/gen3_springs.yaml"
        ),
        description=(
            "Path to the springs configuration YAML -- see "
            "gen3_spring.launch.py's config_arg."
        ),
    )

    participant_id_arg = DeclareLaunchArgument(
        "participant_id",
        default_value="",
        description=(
            "Routes the rosbag into ~/gen3_study_data/<participant_id>/ -- "
            "see gen3_spring.launch.py's participant_id_arg."
        ),
    )

    condition_name_arg = DeclareLaunchArgument(
        "condition_name",
        default_value="",
        description=(
            "Labels the rosbag output directory, e.g. 'condition1' -- see "
            "gen3_spring.launch.py's condition_name_arg."
        ),
    )

    record_audio_arg = DeclareLaunchArgument(
        "record_audio",
        default_value="true",
        description=(
            "Defaults to true here (false in gen3_spring.launch.py) -- "
            "this is a study launch file."
        ),
    )

    record_video_arg = DeclareLaunchArgument(
        "record_video",
        default_value="true",
        description=(
            "Defaults to true here (false in gen3_spring.launch.py) -- "
            "this is a study launch file."
        ),
    )

    armviz_arg = DeclareLaunchArgument(
        "armviz",
        default_value="true",
        description=(
            "Defaults to true here (false in gen3_spring.launch.py) -- "
            "matches study_procedure.txt."
        ),
    )

    load_collision_scene_arg = DeclareLaunchArgument(
        "load_collision_scene",
        default_value="true",
        description=(
            "Defaults to true here (false in gen3_spring.launch.py) -- "
            "matches study_procedure.txt."
        ),
    )

    gen3_spring_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("springcontroller"),
                "launch", "gen3_spring.launch.py",
            )
        ),
        launch_arguments={
            "config": LaunchConfiguration("config"),
            "participant_id": LaunchConfiguration("participant_id"),
            "condition_name": LaunchConfiguration("condition_name"),
            "record_audio": LaunchConfiguration("record_audio"),
            "record_video": LaunchConfiguration("record_video"),
            "armviz": LaunchConfiguration("armviz"),
            "load_collision_scene": LaunchConfiguration("load_collision_scene"),
        }.items(),
    )

    return LaunchDescription([
        config_arg,
        participant_id_arg,
        condition_name_arg,
        record_audio_arg,
        record_video_arg,
        armviz_arg,
        load_collision_scene_arg,
        gen3_spring_launch,
    ])
