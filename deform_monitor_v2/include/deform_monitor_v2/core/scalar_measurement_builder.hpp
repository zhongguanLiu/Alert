/*
 * Author: zgliu@cumt.edu.cn
 * Affiliation: China University of Mining and Technology
 * Open-source release date: 2026-04-20
 */

#ifndef DEFORM_MONITOR_V2_CORE_SCALAR_MEASUREMENT_BUILDER_HPP
#define DEFORM_MONITOR_V2_CORE_SCALAR_MEASUREMENT_BUILDER_HPP

#include "deform_monitor_v2/data_types.hpp"

namespace deform_monitor_v2 {

class ScalarMeasurementBuilder {
public:
  void SetParams(const ObservationParams& observation_params,
                 const NoiseParams& noise_params);

  AlignedVector<ScalarMeasurement> BuildMeasurements(
      const AnchorReference& anchor,
      const LocalSupportData& support,
      const PoseCov6D& pose_cov,
      const Eigen::Vector3d& lidar_origin_R) const;

private:
  ObservationParams observation_params_;
  NoiseParams noise_params_;
};

}  // namespace deform_monitor_v2

#endif  // DEFORM_MONITOR_V2_CORE_SCALAR_MEASUREMENT_BUILDER_HPP
