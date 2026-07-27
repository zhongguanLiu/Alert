#ifndef DEFORM_MONITOR_V2_CORE_MOTION_CLUSTERER_HPP
#define DEFORM_MONITOR_V2_CORE_MOTION_CLUSTERER_HPP

#include "deform_monitor_v2/data_types.hpp"

namespace deform_monitor_v2 {

class MotionClusterer {
public:
  void SetParams(const ClusterParams& params);
  MotionClusterVector Cluster(
      const AnchorReferenceVector& anchors,
      const AnchorStateVector& states) const;

private:
  ClusterParams params_;
};

}  // namespace deform_monitor_v2

#endif  // DEFORM_MONITOR_V2_CORE_MOTION_CLUSTERER_HPP
