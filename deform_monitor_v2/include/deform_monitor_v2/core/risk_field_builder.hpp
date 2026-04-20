/*
 * Author: zgliu@cumt.edu.cn
 * Affiliation: China University of Mining and Technology
 * Open-source release date: 2026-04-20
 */

#ifndef DEFORM_MONITOR_V2_RISK_FIELD_BUILDER_HPP
#define DEFORM_MONITOR_V2_RISK_FIELD_BUILDER_HPP

#include "deform_monitor_v2/data_types.hpp"

namespace deform_monitor_v2 {

class RiskFieldBuilder {
public:
  void SetParams(const RiskVisualizationParams& params);

  RiskVoxelVector Build(const AnchorReferenceVector& anchors,
                        const RiskEvidenceVector& evidences) const;

  RiskRegionVector ExtractRegions(const RiskVoxelVector& voxels) const;

private:
  RiskVisualizationParams params_;
};

}  // namespace deform_monitor_v2

#endif  // DEFORM_MONITOR_V2_RISK_FIELD_BUILDER_HPP
