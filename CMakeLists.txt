cmake_minimum_required(VERSION 3.8)
project(virtual_spring_ros2)

# Compiler warnings
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ── Find dependencies ────────────────────────────────────────────────────────
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

# ── Install Python library ───────────────────────────────────────────────────
# Makes `from virtual_spring_ros2.virtual_spring import VirtualSpring` work
ament_python_install_package(${PROJECT_NAME})

# ── Install Python nodes ─────────────────────────────────────────────────────
install(PROGRAMS
  scripts/virtual_spring_node.py
  DESTINATION lib/${PROJECT_NAME}
)

# ── Install launch files ─────────────────────────────────────────────────────
install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}
)

# ── Tests ────────────────────────────────────────────────────────────────────
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()

  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_virtual_spring test/test_virtual_spring.py)
endif()

ament_package()
