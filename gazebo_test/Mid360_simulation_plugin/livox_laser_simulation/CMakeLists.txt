cmake_minimum_required(VERSION 2.8.3)
project(livox_laser_simulation)

## Compile as C++11, supported in ROS Kinetic and newer
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  livox_ros_driver
  message_runtime
  roscpp
  rospy
  tf
  sensor_msgs
  std_msgs
  std_srvs
  message_generation
)

find_package(gazebo REQUIRED)
find_package(PCL REQUIRED)
include_directories(${GAZEBO_INCLUDE_DIRS})
add_message_files(FILES
    CustomPoint.msg
    CustomMsg.msg
  #   Message2.msg
  )

generate_messages(DEPENDENCIES
  std_msgs
)
#To solve the error which gazebo does not handle the include file well, we must add this line.
include_directories(/usr/include/)

link_directories(${GAZEBO_LIBRARY_DIRS})

include_directories(
         include
        ${catkin_INCLUDE_DIRS}
        ${PCL_INCLUDE_DIRS}
)

catkin_package(
        INCLUDE_DIRS include
        LIBRARIES livox_laser_simulation synchronized_multi_model_motion_world_plugin
        CATKIN_DEPENDS geometry_msgs livox_ros_driver message_runtime roscpp rospy sensor_msgs std_msgs std_srvs tf
)

catkin_install_python(PROGRAMS
  scripts/circle_cmd.py
  scripts/keyboard_cmd.py
  scripts/model_motion_controller.py
  scripts/spawn_or_replace_model.py
  scripts/synchronized_motion_cli.py
  scripts/work_patrol_cmd.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

add_library(livox_laser_simulation SHARED src/livox_points_plugin.cpp src/livox_ode_multiray_shape.cpp)
add_dependencies(livox_laser_simulation ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(livox_laser_simulation ${catkin_LIBRARIES} ${GAZEBO_LIBRARIES} ${PCL_LIBRARIES} RayPlugin)

add_library(synchronized_multi_model_motion_world_plugin SHARED
  src/synchronized_multi_model_motion_world_plugin.cpp)
add_dependencies(synchronized_multi_model_motion_world_plugin
  ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(synchronized_multi_model_motion_world_plugin
  ${catkin_LIBRARIES} ${GAZEBO_LIBRARIES})
