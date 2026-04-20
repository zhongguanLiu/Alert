cmake_minimum_required(VERSION 3.0.2)
project(deform_monitor_v2)

add_compile_options(-std=c++14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")

find_package(catkin REQUIRED COMPONENTS
  fast_lio
  roscpp
  std_msgs
  sensor_msgs
  nav_msgs
  geometry_msgs
  visualization_msgs
  pcl_ros
  pcl_conversions
  message_generation
)

find_package(PCL REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED COMPONENTS system thread)
find_package(Threads REQUIRED)

add_message_files(
  FILES
  AnchorState.msg
  AnchorStates.msg
  MotionCluster.msg
  MotionClusters.msg
  RiskEvidence.msg
  RiskEvidenceArray.msg
  RiskVoxel.msg
  RiskVoxelField.msg
  RiskRegion.msg
  RiskRegions.msg
  PersistentRiskRegion.msg
  PersistentRiskRegions.msg
  StructureMotion.msg
  StructureMotions.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS
    roscpp
    fast_lio
    std_msgs
    sensor_msgs
    nav_msgs
    geometry_msgs
    visualization_msgs
    pcl_ros
    pcl_conversions
    message_runtime
  DEPENDS
    PCL
    EIGEN3
    Boost
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
)

link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

catkin_install_python(PROGRAMS
  scripts/analyze_sim_run.py
  scripts/real_experiment_recorder.py
  scripts/sim_experiment_recorder.py
  scripts/sim_truth_rviz_bridge.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

add_library(${PROJECT_NAME}
  src/data_types.cpp
  src/core/covariance_extractor.cpp
  src/core/anchor_builder.cpp
  src/core/current_observation_extractor.cpp
  src/core/scalar_measurement_builder.cpp
  src/core/imm_information_filter.cpp
  src/core/motion_clusterer.cpp
  src/core/region_hypothesis_builder.cpp
  src/core/region_correspondence_solver.cpp
  src/core/risk_evidence_adapter.cpp
  src/core/risk_field_builder.cpp
  src/core/persistent_risk_region_tracker.cpp
  src/core/reference_manager.cpp
  src/core/structure_unit_builder.cpp
  src/core/structure_unit_tracker.cpp
  src/risk_visualization_publisher.cpp
  src/structure_visualization_publisher.cpp
  src/visualization_publisher.cpp
)

add_dependencies(${PROJECT_NAME}
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
)

target_link_libraries(${PROJECT_NAME}
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${Boost_LIBRARIES}
)

add_executable(deform_monitor_v2_node
  src/deform_monitor_v2_node.cpp
  src/deform_monitor_v2_main.cpp
)

target_compile_definitions(deform_monitor_v2_node PRIVATE
  DEFORM_MONITOR_V2_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)

add_dependencies(deform_monitor_v2_node
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
)

target_link_libraries(deform_monitor_v2_node
  ${PROJECT_NAME}
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${Boost_LIBRARIES}
  Threads::Threads
)

if(CATKIN_ENABLE_TESTING)
  catkin_add_gtest(test_ablation_switches
    test/test_ablation_switches.cpp
    src/deform_monitor_v2_node.cpp
  )

  if(TARGET test_ablation_switches)
    target_link_libraries(test_ablation_switches
      ${PROJECT_NAME}
      ${catkin_LIBRARIES}
      ${PCL_LIBRARIES}
      ${Boost_LIBRARIES}
      Threads::Threads
    )
  endif()
endif()
