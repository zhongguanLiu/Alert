#ifndef DEFORM_MONITOR_V2_WEAK_PLANE_MOTION_DETECTOR_HPP
#define DEFORM_MONITOR_V2_WEAK_PLANE_MOTION_DETECTOR_HPP

#include "deform_monitor_v2/data_types.hpp"

#include <limits>

namespace deform_monitor_v2 {

class WeakPlaneMotionDetector {
public:
  void SetParams(const WeakPlaneMotionParams& params);
  void Update(const AnchorReferenceVector& anchors, AnchorStateVector* states);

private:
  struct ComponentTrack {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    int id = -1;
    Eigen::Vector3d center_R = Eigen::Vector3d::Zero();
    Eigen::Vector3d displacement_R = Eigen::Vector3d::Zero();
    int streak = 0;
    int missed_frames = 0;
  };

  void ResetTracks();

  WeakPlaneMotionParams params_;
  std::vector<ComponentTrack, Eigen::aligned_allocator<ComponentTrack>> component_tracks_;
  int next_component_id_ = 1;
  uint32_t reference_epoch_ = std::numeric_limits<uint32_t>::max();
  size_t anchor_count_ = 0;
};

}  // namespace deform_monitor_v2

#endif  // DEFORM_MONITOR_V2_WEAK_PLANE_MOTION_DETECTOR_HPP
