cmake_minimum_required(VERSION 3.16)
project(go2_cbf_control)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(diagnostic_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(go2_dds_ros2_bridge_msgs REQUIRED)
find_package(Eigen3 REQUIRED)

include(FetchContent)
# Keep the native CBF solver independent of Conda and pin the version used by
# the deployment build. OSQP is fetched into the colcon build tree, not this
# source tree.
set(UNITTESTS OFF CACHE BOOL "" FORCE)
set(PRINTING OFF CACHE BOOL "" FORCE)
set(PROFILING ON CACHE BOOL "" FORCE)
set(CTRLC OFF CACHE BOOL "" FORCE)
set(ENABLE_MKL_PARDISO OFF CACHE BOOL "" FORCE)
FetchContent_Declare(osqp
  GIT_REPOSITORY https://github.com/osqp/osqp.git
  GIT_TAG 0dd00a578cf1c2691c5c379965d504c75bf6cfad
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(osqp)

add_executable(cbf_control src/cbf_control_node.cpp)
target_compile_features(cbf_control PRIVATE cxx_std_17)
target_include_directories(cbf_control PRIVATE include)
target_link_libraries(cbf_control osqpstatic Eigen3::Eigen)
ament_target_dependencies(cbf_control
  rclcpp geometry_msgs sensor_msgs visualization_msgs diagnostic_msgs std_msgs go2_dds_ros2_bridge_msgs)

install(TARGETS cbf_control DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY config DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_cbf_math test/test_cbf_math.cpp)
  if(TARGET test_cbf_math)
    target_compile_features(test_cbf_math PRIVATE cxx_std_17)
    target_include_directories(test_cbf_math PRIVATE include)
  endif()
endif()

ament_package()
