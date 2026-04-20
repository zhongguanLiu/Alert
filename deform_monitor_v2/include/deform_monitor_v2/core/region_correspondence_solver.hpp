/*
 * Author: zgliu@cumt.edu.cn
 * Affiliation: China University of Mining and Technology
 * Open-source release date: 2026-04-20
 */

#ifndef DEFORM_MONITOR_V2_REGION_CORRESPONDENCE_SOLVER_HPP
#define DEFORM_MONITOR_V2_REGION_CORRESPONDENCE_SOLVER_HPP

#include "deform_monitor_v2/data_types.hpp"

namespace deform_monitor_v2 {

class RegionCorrespondenceSolver {
public:
  void SetParams(const StructureCorrespondenceParams& params);

  StructureMotionVector Solve(const RegionHypothesisVector& old_regions,
                              const RegionHypothesisVector& new_regions) const;

private:
  double PairCost(const RegionHypothesisState& old_region,
                  const RegionHypothesisState& new_region,
                  double* distance) const;

  StructureCorrespondenceParams params_;
};

}  // namespace deform_monitor_v2

#endif  // DEFORM_MONITOR_V2_REGION_CORRESPONDENCE_SOLVER_HPP
