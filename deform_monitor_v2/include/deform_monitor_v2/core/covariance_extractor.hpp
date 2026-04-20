/*
 * Author: zgliu@cumt.edu.cn
 * Affiliation: China University of Mining and Technology
 * Open-source release date: 2026-04-20
 */

#ifndef DEFORM_MONITOR_V2_CORE_COVARIANCE_EXTRACTOR_HPP
#define DEFORM_MONITOR_V2_CORE_COVARIANCE_EXTRACTOR_HPP

#include <Eigen/Dense>

namespace deform_monitor_v2 {

class CovarianceExtractor {
public:
  static Eigen::Matrix<double, 6, 6> InflateSigmaXi(
      const Eigen::Matrix<double, 6, 6>& Sigma_xi,
      double alpha_xi);

  static Eigen::Matrix3d PointCovarianceFromReferencePoint(
      const Eigen::Vector3d& x_R,
      const Eigen::Vector3d& lidar_origin_R,
      const Eigen::Matrix<double, 6, 6>& Sigma_xi,
      double sigma_p);
};

}  // namespace deform_monitor_v2

#endif  // DEFORM_MONITOR_V2_CORE_COVARIANCE_EXTRACTOR_HPP
