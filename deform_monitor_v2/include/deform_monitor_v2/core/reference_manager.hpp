/*
 * Author: zgliu@cumt.edu.cn
 * Affiliation: China University of Mining and Technology
 * Open-source release date: 2026-04-20
 */

#ifndef DEFORM_MONITOR_V2_CORE_REFERENCE_MANAGER_HPP
#define DEFORM_MONITOR_V2_CORE_REFERENCE_MANAGER_HPP

#include "deform_monitor_v2/data_types.hpp"

namespace deform_monitor_v2 {

class ReferenceManager {
public:
  void SetParams(const ReferenceParams& params);
  void UpdateReferenceStatistics(
      AnchorReferenceVector* anchors,
      const CurrentObservationVector& observations,
      AnchorStateVector* states);

private:
  ReferenceParams params_;
};

}  // namespace deform_monitor_v2

#endif  // DEFORM_MONITOR_V2_CORE_REFERENCE_MANAGER_HPP
