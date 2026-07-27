#ifndef DEFORM_MONITOR_V2_CORE_EDGE_FEATURE_HPP
#define DEFORM_MONITOR_V2_CORE_EDGE_FEATURE_HPP

#include "deform_monitor_v2/data_types.hpp"

#include <cstddef>
#include <vector>

namespace deform_monitor_v2 {

struct EdgeFeatureGeometry {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  bool valid = false;
  Eigen::Vector3d center_R = Eigen::Vector3d::Zero();
  Eigen::Matrix3d sample_cov = Eigen::Matrix3d::Zero();
  std::vector<size_t> point_indices;
};

EdgeFeatureGeometry ComputeEdgeFeatureGeometry(
    const AlignedVector<Eigen::Vector3d>& points_R,
    const Eigen::Vector3d& support_center_R,
    const Eigen::Vector3d& edge_normal_R);

}  // namespace deform_monitor_v2

#endif  // DEFORM_MONITOR_V2_CORE_EDGE_FEATURE_HPP
