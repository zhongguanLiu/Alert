/*
 * Author: zgliu@cumt.edu.cn
 * Affiliation: China University of Mining and Technology
 * Open-source release date: 2026-04-20
 */

#include "deform_monitor_v2/core/covariance_extractor.hpp"

#include "deform_monitor_v2/data_types.hpp"

namespace deform_monitor_v2 {

Eigen::Matrix<double, 6, 6> CovarianceExtractor::InflateSigmaXi(
    const Eigen::Matrix<double, 6, 6>& Sigma_xi,
    double alpha_xi) {
  const double alpha = std::max(1.0, alpha_xi);
  Eigen::Matrix<double, 6, 6> S = 0.5 * (Sigma_xi + Sigma_xi.transpose());
  S *= alpha;
  S += Eigen::Matrix<double, 6, 6>::Identity() * 1.0e-9;
  return S;
}

Eigen::Matrix3d CovarianceExtractor::PointCovarianceFromReferencePoint(
    const Eigen::Vector3d& x_R,
    const Eigen::Vector3d& lidar_origin_R,
    const Eigen::Matrix<double, 6, 6>& Sigma_xi,
    double sigma_p) {
  const Eigen::Vector3d Rp = x_R - lidar_origin_R;
  Eigen::Matrix<double, 3, 6> J = Eigen::Matrix<double, 3, 6>::Zero();
  J.block<3, 3>(0, 0) = -SkewSymmetric(Rp);
  J.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity();
  Eigen::Matrix3d S = J * Sigma_xi * J.transpose();
  S += Eigen::Matrix3d::Identity() * sigma_p * sigma_p;
  return 0.5 * (S + S.transpose());
}

}  // namespace deform_monitor_v2
