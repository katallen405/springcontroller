cmake_minimum_required(VERSION 3.8)
project(springcontroller_ui_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/EnableTorqueControl.srv"
  "srv/GetLinkPose.srv"
  "srv/ListLinkNames.srv"
  "srv/MoveToJointAngles.srv"
)

ament_package()
