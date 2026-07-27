#include "deform_monitor_v2/core/edge_feature.hpp"

namespace deform_monitor_v2 {

EdgeFeatureGeometry ComputeEdgeFeatureGeometry(
    const AlignedVector<Eigen::Vector3d>& points_R,
    const Eigen::Vector3d& support_center_R,
    const Eigen::Vector3d& edge_normal_R) {
  EdgeFeatureGeometry feature;
  feature.center_R = support_center_R;

  const double axis_norm = edge_normal_R.norm();
  if (points_R.empty() || axis_norm < 1.0e-9) {
    return feature;
  }
  const Eigen::Vector3d axis = edge_normal_R / axis_norm;

  Eigen::Vector3d selected_sum_R = Eigen::Vector3d::Zero();
  feature.point_indices.reserve(points_R.size());
  for (size_t i = 0; i < points_R.size(); ++i) {
    if (axis.dot(points_R[i] - support_center_R) > 0.0) {
      feature.point_indices.push_back(i);
      selected_sum_R += points_R[i];
    }
  }
  if (feature.point_indices.empty()) {
    return feature;
  }

  feature.center_R =
      selected_sum_R / static_cast<double>(feature.point_indices.size());

  if (feature.point_indices.size() > 1) {
    for (const size_t index : feature.point_indices) {
      const Eigen::Vector3d delta = points_R[index] - feature.center_R;
      feature.sample_cov += delta * delta.transpose();
    }
    feature.sample_cov /= static_cast<double>(feature.point_indices.size() - 1);
    feature.sample_cov = 0.5 * (feature.sample_cov + feature.sample_cov.transpose());
  }
  feature.valid = true;
  return feature;
}

}  // namespace deform_monitor_v2
