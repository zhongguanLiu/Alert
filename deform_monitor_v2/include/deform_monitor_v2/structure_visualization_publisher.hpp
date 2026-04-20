/*
 * Author: zgliu@cumt.edu.cn
 * Affiliation: China University of Mining and Technology
 * Open-source release date: 2026-04-20
 */

#ifndef DEFORM_MONITOR_V2_STRUCTURE_VISUALIZATION_PUBLISHER_HPP
#define DEFORM_MONITOR_V2_STRUCTURE_VISUALIZATION_PUBLISHER_HPP

#include <visualization_msgs/MarkerArray.h>

#include "deform_monitor_v2/StructureMotions.h"
#include "deform_monitor_v2/data_types.hpp"

namespace deform_monitor_v2 {

class StructureVisualizationPublisher {
public:
  void SetParams(const StructureCorrespondenceParams& params);

  deform_monitor_v2::StructureMotions BuildStructureMotionsMsg(
      const StructureMotionVector& motions,
      const ros::Time& stamp,
      const std::string& frame_id) const;

  visualization_msgs::MarkerArray BuildMarkers(
      const StructureMotionVector& motions,
      const ros::Time& stamp,
      const std::string& frame_id) const;

private:
  StructureCorrespondenceParams params_;
};

}  // namespace deform_monitor_v2

#endif  // DEFORM_MONITOR_V2_STRUCTURE_VISUALIZATION_PUBLISHER_HPP
