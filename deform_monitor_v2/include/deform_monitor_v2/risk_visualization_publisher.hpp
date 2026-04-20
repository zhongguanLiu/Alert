/*
 * Author: zgliu@cumt.edu.cn
 * Affiliation: China University of Mining and Technology
 * Open-source release date: 2026-04-20
 */

#ifndef DEFORM_MONITOR_V2_RISK_VISUALIZATION_PUBLISHER_HPP
#define DEFORM_MONITOR_V2_RISK_VISUALIZATION_PUBLISHER_HPP

#include <visualization_msgs/MarkerArray.h>

#include "deform_monitor_v2/RiskEvidenceArray.h"
#include "deform_monitor_v2/RiskRegion.h"
#include "deform_monitor_v2/RiskRegions.h"
#include "deform_monitor_v2/RiskVoxelField.h"
#include "deform_monitor_v2/PersistentRiskRegion.h"
#include "deform_monitor_v2/PersistentRiskRegions.h"
#include "deform_monitor_v2/data_types.hpp"

namespace deform_monitor_v2 {

class RiskVisualizationPublisher {
public:
  void SetParams(const RiskVisualizationParams& params);

  deform_monitor_v2::RiskEvidenceArray BuildRiskEvidenceMsg(
      const RiskEvidenceVector& evidences,
      const ros::Time& stamp,
      const std::string& frame_id) const;

  deform_monitor_v2::RiskVoxelField BuildRiskVoxelFieldMsg(
      const RiskVoxelVector& voxels,
      const ros::Time& stamp,
      const std::string& frame_id) const;

  deform_monitor_v2::RiskRegions BuildRiskRegionsMsg(
      const RiskRegionVector& regions,
      const ros::Time& stamp,
      const std::string& frame_id) const;

  deform_monitor_v2::PersistentRiskRegions BuildPersistentRiskRegionsMsg(
      const PersistentRiskTrackVector& tracks,
      const ros::Time& stamp,
      const std::string& frame_id) const;

  visualization_msgs::MarkerArray BuildRiskMarkers(
      const RiskVoxelVector& voxels,
      const RiskRegionVector& regions,
      const ros::Time& stamp,
      const std::string& frame_id) const;

  visualization_msgs::MarkerArray BuildPersistentRiskMarkers(
      const PersistentRiskTrackVector& tracks,
      const ros::Time& stamp,
      const std::string& frame_id) const;

private:
  RiskVisualizationParams params_;
};

}  // namespace deform_monitor_v2

#endif  // DEFORM_MONITOR_V2_RISK_VISUALIZATION_PUBLISHER_HPP
