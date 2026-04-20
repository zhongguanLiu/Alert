/*
 * Author: zgliu@cumt.edu.cn
 * Affiliation: China University of Mining and Technology
 * Open-source release date: 2026-04-20
 */

#ifndef DEFORM_MONITOR_V2_VISUALIZATION_PUBLISHER_HPP
#define DEFORM_MONITOR_V2_VISUALIZATION_PUBLISHER_HPP

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/MarkerArray.h>

#include "deform_monitor_v2/AnchorStates.h"
#include "deform_monitor_v2/MotionClusters.h"
#include "deform_monitor_v2/data_types.hpp"

namespace deform_monitor_v2 {

class VisualizationPublisher {
public:
  void SetParams(const VisualizationParams& params, ros::NodeHandle& nh);

  deform_monitor_v2::AnchorStates BuildAnchorStatesMsg(
      const AnchorReferenceVector& anchors,
      const AnchorStateVector& states,
      const CurrentObservationVector& observations,
      const ros::Time& stamp,
      const std::string& frame_id) const;

  deform_monitor_v2::MotionClusters BuildMotionClustersMsg(
      const MotionClusterVector& clusters,
      const ros::Time& stamp,
      const std::string& frame_id) const;

  sensor_msgs::PointCloud2 BuildDebugCloudMsg(
      const AnchorReferenceVector& anchors,
      const AnchorStateVector& states,
      const ros::Time& stamp,
      const std::string& frame_id) const;

  visualization_msgs::MarkerArray BuildAnchorMarkers(
      const AnchorReferenceVector& anchors,
      const AnchorStateVector& states,
      const ros::Time& stamp,
      const std::string& frame_id) const;

  visualization_msgs::MarkerArray BuildMotionMarkers(
      const AnchorReferenceVector& anchors,
      const AnchorStateVector& states,
      const MotionClusterVector& clusters,
      const ros::Time& stamp,
      const std::string& frame_id) const;

  void PublishStructureMigrations(
      const StructureUnitVector& units,
      const StructureMigrationVector& migrations,
      const ros::Time& stamp,
      const std::string& frame_id);

private:
  VisualizationParams params_;
  ros::Publisher migration_pub_;
};

}  // namespace deform_monitor_v2

#endif  // DEFORM_MONITOR_V2_VISUALIZATION_PUBLISHER_HPP
