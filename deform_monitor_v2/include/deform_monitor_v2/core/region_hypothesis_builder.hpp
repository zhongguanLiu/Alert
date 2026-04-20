/*
 * Author: zgliu@cumt.edu.cn
 * Affiliation: China University of Mining and Technology
 * Open-source release date: 2026-04-20
 */

#ifndef DEFORM_MONITOR_V2_REGION_HYPOTHESIS_BUILDER_HPP
#define DEFORM_MONITOR_V2_REGION_HYPOTHESIS_BUILDER_HPP

#include "deform_monitor_v2/data_types.hpp"

namespace deform_monitor_v2 {

class RegionHypothesisBuilder {
public:
  void SetParams(const StructureCorrespondenceParams& params);

  void Build(const AnchorReferenceVector& anchors,
             const AnchorStateVector& states,
             const CurrentObservationVector& observations,
             const MotionClusterVector& clusters,
             RegionHypothesisVector* old_regions,
             RegionHypothesisVector* new_regions) const;

private:
  bool IsOldAnchor(const AnchorTrackState& state,
                   const CurrentObservation& observation) const;
  bool IsNewAnchor(const AnchorTrackState& state,
                   const CurrentObservation& observation) const;
  RegionHypothesisVector BuildRegions(const AnchorReferenceVector& anchors,
                                      const AnchorStateVector& states,
                                      const CurrentObservationVector& observations,
                                      const std::vector<size_t>& candidate_indices,
                                      RegionHypothesisKind kind) const;

  StructureCorrespondenceParams params_;
};

}  // namespace deform_monitor_v2

#endif  // DEFORM_MONITOR_V2_REGION_HYPOTHESIS_BUILDER_HPP
