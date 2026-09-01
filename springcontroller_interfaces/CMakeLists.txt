cmake_minimum_required(VERSION 3.8)
project(springcontroller_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/SpringForce.msg"
  "msg/SpringForces.msg"
  "srv/AddSpring.srv"
  "srv/RemoveSpring.srv"
  "srv/AddJointSpring.srv"
  "srv/AddOrientationSpring.srv"
  "srv/LoadCollisionScene.srv"
  "srv/CheckCollisionAtAngles.srv"
  "srv/UpdateSpring.srv"
  "srv/UpdateCollisionThresholds.srv"
)

ament_package()
