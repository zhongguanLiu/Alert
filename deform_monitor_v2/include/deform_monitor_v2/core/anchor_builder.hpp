/*
 * Author: zgliu@cumt.edu.cn
 * Affiliation: China University of Mining and Technology
 * Open-source release date: 2026-04-20
 */

#ifndef DEFORM_MONITOR_V2_CORE_ANCHOR_BUILDER_HPP
#define DEFORM_MONITOR_V2_CORE_ANCHOR_BUILDER_HPP

#include "deform_monitor_v2/data_types.hpp"

namespace deform_monitor_v2 {

class AnchorBuilder {
public:
  void SetParams(const AnchorBuildParams& params);
  AnchorReferenceVector BuildFrozenAnchors(
      const ReferenceInitFrameVector& init_frames);


  AnchorReferenceVector BuildIncrementalAnchors(
      const ReferenceInitFrameVector& recent_frames,
      const AnchorReferenceVector& existing_anchors,
      double coverage_radius,
      int start_id,
      int min_visible_frames_override,
      int max_new_anchors);

private:
  AnchorBuildParams params_;
};

}  // namespace deform_monitor_v2

#endif  // DEFORM_MONITOR_V2_CORE_ANCHOR_BUILDER_HPP
