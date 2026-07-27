cmake_minimum_required(VERSION 3.0.2)
project(deform_monitor_v2)

add_compile_options(-std=c++14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")

find_package(catkin REQUIRED COMPONENTS
  fast_lio
  livox_ros_driver
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
  AnchorObjectSummary.msg
  AnchorStates.msg
  MotionCluster.msg
  MotionClusters.msg
  ObjectHitStat.msg
  ObjectObservationStats.msg
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
    livox_ros_driver
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
  scripts/algorithm_frame_store.py
  scripts/analyze_anchor_continuity.py
  scripts/analyze_sim_run.py
  scripts/check_sim_object_associations.py
  scripts/compare_algorithm_storage.py
  scripts/real_experiment_recorder.py
  scripts/sim_experiment_recorder.py
  scripts/sim_truth_rviz_bridge.py
  scripts/validate_recorded_run.py
  scripts/validate_sim_experiment_setup.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

add_library(${PROJECT_NAME}
  src/data_types.cpp
  src/core/object_id_association.cpp
  src/core/object_observation_stats.cpp
  src/core/covariance_extractor.cpp
  src/core/observable_subspace.cpp
  src/core/edge_feature.cpp
  src/core/anchor_builder.cpp
  src/core/current_observation_extractor.cpp
  src/core/scalar_measurement_builder.cpp
  src/core/imm_information_filter.cpp
  src/core/weak_plane_motion_detector.cpp
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
  catkin_add_gtest(test_anchor_object_association
    test/test_anchor_object_association.cpp
  )

  if(TARGET test_anchor_object_association)
    target_link_libraries(test_anchor_object_association
      ${PROJECT_NAME}
      ${catkin_LIBRARIES}
      ${PCL_LIBRARIES}
      ${Boost_LIBRARIES}
      Threads::Threads
    )
  endif()

  catkin_add_gtest(test_edge_measurement_model
    test/test_edge_measurement_model.cpp
  )

  if(TARGET test_edge_measurement_model)
    target_link_libraries(test_edge_measurement_model
      ${PROJECT_NAME}
      ${catkin_LIBRARIES}
      ${PCL_LIBRARIES}
      ${Boost_LIBRARIES}
      Threads::Threads
    )
  endif()

  catkin_add_gtest(test_anchor_reference_continuity
    test/test_anchor_reference_continuity.cpp
  )

  if(TARGET test_anchor_reference_continuity)
    target_link_libraries(test_anchor_reference_continuity
      ${PROJECT_NAME}
      ${catkin_LIBRARIES}
      ${PCL_LIBRARIES}
      ${Boost_LIBRARIES}
      Threads::Threads
    )
  endif()

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

  catkin_add_gtest(test_weak_plane_motion_detector
    test/test_weak_plane_motion_detector.cpp
  )

  if(TARGET test_weak_plane_motion_detector)
    target_link_libraries(test_weak_plane_motion_detector
      ${PROJECT_NAME}
      ${catkin_LIBRARIES}
      ${PCL_LIBRARIES}
      ${Boost_LIBRARIES}
      Threads::Threads
    )
  endif()

  catkin_add_gtest(test_persistent_risk_region_tracker
    test/test_persistent_risk_region_tracker.cpp
  )

  if(TARGET test_persistent_risk_region_tracker)
    target_link_libraries(test_persistent_risk_region_tracker
      ${PROJECT_NAME}
      ${catkin_LIBRARIES}
      ${PCL_LIBRARIES}
      ${Boost_LIBRARIES}
      Threads::Threads
    )
  endif()
endif()
