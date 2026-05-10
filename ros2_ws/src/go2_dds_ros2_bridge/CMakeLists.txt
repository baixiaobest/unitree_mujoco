cmake_minimum_required(VERSION 3.8)
project(go2_dds_ros2_bridge)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

ament_python_install_package(${PROJECT_NAME})

install(
  PROGRAMS
    go2_dds_ros2_bridge/clock_offset_bridge_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME clock_offset_bridge
)

install(
  PROGRAMS
    go2_dds_ros2_bridge/odom_bridge_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME odom_bridge
)

ament_package()