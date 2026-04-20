/*
 * Author: zgliu@cumt.edu.cn
 * Affiliation: China University of Mining and Technology
 * Open-source release date: 2026-04-20
 */

#include "deform_monitor_v2/data_types.hpp"

#include <Eigen/SVD>

#include <cmath>

namespace deform_monitor_v2 {

namespace {

Eigen::Matrix<double, 6, 6> DefaultSigmaXi(double rot_sigma, double pos_sigma) {
  Eigen::Matrix<double, 6, 6> S = Eigen::Matrix<double, 6, 6>::Zero();
  S.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * rot_sigma * rot_sigma;
  S.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() * pos_sigma * pos_sigma;
  return S;
}

Eigen::Matrix<double, 6, 6> ExtractSigmaXiRotPosFromFull(
    const std::vector<double>& flat_cov,
    double default_rot_sigma,
    double default_pos_sigma) {
  if (flat_cov.size() != 23 * 23) {
    return DefaultSigmaXi(default_rot_sigma, default_pos_sigma);
  }

  Eigen::Matrix<double, 23, 23> full = Eigen::Matrix<double, 23, 23>::Zero();
  for (int r = 0; r < 23; ++r) {
    for (int c = 0; c < 23; ++c) {
      full(r, c) = flat_cov[r * 23 + c];
    }
  }

  Eigen::Matrix<double, 6, 6> S = Eigen::Matrix<double, 6, 6>::Zero();
  S.block<3, 3>(0, 0) = full.block<3, 3>(3, 3);
  S.block<3, 3>(0, 3) = full.block<3, 3>(3, 0);
  S.block<3, 3>(3, 0) = full.block<3, 3>(0, 3);
  S.block<3, 3>(3, 3) = full.block<3, 3>(0, 0);
  return S;
}

}  // namespace

PoseCov6D PoseCovFromOdometry(const nav_msgs::Odometry& odom,
                              const std::string& layout,
                              double default_rot_sigma,
                              double default_pos_sigma) {
  PoseCov6D pose_cov;
  pose_cov.stamp = odom.header.stamp;

  Eigen::Matrix<double, 6, 6> raw = Eigen::Matrix<double, 6, 6>::Zero();
  for (int r = 0; r < 6; ++r) {
    for (int c = 0; c < 6; ++c) {
      raw(r, c) = odom.pose.covariance[r * 6 + c];
    }
  }

  const double max_abs = raw.cwiseAbs().maxCoeff();
  if (!std::isfinite(max_abs) || max_abs < 1.0e-12) {
    pose_cov.Sigma_xi = DefaultSigmaXi(default_rot_sigma, default_pos_sigma);
    return pose_cov;
  }

  if (layout == "fastlio_rot_pos") {
    pose_cov.Sigma_xi = raw;
  } else {
    pose_cov.Sigma_xi.block<3, 3>(0, 0) = raw.block<3, 3>(3, 3);
    pose_cov.Sigma_xi.block<3, 3>(0, 3) = raw.block<3, 3>(3, 0);
    pose_cov.Sigma_xi.block<3, 3>(3, 0) = raw.block<3, 3>(0, 3);
    pose_cov.Sigma_xi.block<3, 3>(3, 3) = raw.block<3, 3>(0, 0);
  }

  pose_cov.Sigma_xi = 0.5 * (pose_cov.Sigma_xi + pose_cov.Sigma_xi.transpose());
  pose_cov.Sigma_xi += Eigen::Matrix<double, 6, 6>::Identity() * 1.0e-9;
  return pose_cov;
}

PoseCov6D PoseCovFromFastLioMsg(const fast_lio::LioOdomCov& msg,
                                double default_rot_sigma,
                                double default_pos_sigma) {
  PoseCov6D pose_cov;
  pose_cov.stamp = msg.odom.header.stamp;

  const bool has_full =
      msg.state_dim == 23 &&
      msg.state_covariance.size() == static_cast<size_t>(msg.state_dim * msg.state_dim) &&
      (msg.state_order.empty() ||
       msg.state_order == "pos,rot,offset_R_L_I,offset_T_L_I,vel,bg,ba,grav");
  if (has_full) {
    pose_cov.Sigma_xi = ExtractSigmaXiRotPosFromFull(msg.state_covariance,
                                                     default_rot_sigma,
                                                     default_pos_sigma);
  } else {
    pose_cov = PoseCovFromOdometry(msg.odom, "fastlio_rot_pos", default_rot_sigma, default_pos_sigma);
  }

  pose_cov.Sigma_xi = 0.5 * (pose_cov.Sigma_xi + pose_cov.Sigma_xi.transpose());
  pose_cov.Sigma_xi += Eigen::Matrix<double, 6, 6>::Identity() * 1.0e-9;
  return pose_cov;
}

Eigen::Isometry3d PoseFromOdometry(const nav_msgs::Odometry& odom) {
  Eigen::Quaterniond q(odom.pose.pose.orientation.w,
                       odom.pose.pose.orientation.x,
                       odom.pose.pose.orientation.y,
                       odom.pose.pose.orientation.z);
  if (q.norm() < 1.0e-9) {
    q = Eigen::Quaterniond::Identity();
  } else {
    q.normalize();
  }

  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
  T.linear() = q.toRotationMatrix();
  T.translation() = Eigen::Vector3d(odom.pose.pose.position.x,
                                    odom.pose.pose.position.y,
                                    odom.pose.pose.position.z);
  return T;
}

Eigen::Isometry3d PoseFromFastLioMsg(const fast_lio::LioOdomCov& msg) {
  return PoseFromOdometry(msg.odom);
}

Eigen::Matrix3d SkewSymmetric(const Eigen::Vector3d& v) {
  Eigen::Matrix3d M;
  M << 0.0, -v.z(), v.y(),
       v.z(), 0.0, -v.x(),
      -v.y(), v.x(), 0.0;
  return M;
}

Eigen::MatrixXd PseudoInverse(const Eigen::MatrixXd& matrix, double eps) {
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(matrix, Eigen::ComputeThinU | Eigen::ComputeThinV);
  Eigen::VectorXd singular = svd.singularValues();
  Eigen::VectorXd inv = singular;
  const double max_sv = singular.size() > 0 ? singular.maxCoeff() : 0.0;
  for (int i = 0; i < singular.size(); ++i) {
    inv(i) = singular(i) > eps * std::max(1.0, max_sv) ? 1.0 / singular(i) : 0.0;
  }
  return svd.matrixV() * inv.asDiagonal() * svd.matrixU().transpose();
}

double Chi2PseudoInverse(const Eigen::Vector3d& u, const Eigen::Matrix3d& Sigma_u) {
  Eigen::Matrix3d S = 0.5 * (Sigma_u + Sigma_u.transpose());
  S += Eigen::Matrix3d::Identity() * 1.0e-9;
  const Eigen::Matrix3d S_pinv = PseudoInverse(S, 1.0e-9);
  return u.transpose() * S_pinv * u;
}

double Rad2Deg(double rad) {
  return rad * 180.0 / M_PI;
}

double Deg2Rad(double deg) {
  return deg * M_PI / 180.0;
}

double AngleBetweenDeg(const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
  const double na = a.norm();
  const double nb = b.norm();
  if (na < 1.0e-9 || nb < 1.0e-9) {
    return 180.0;
  }
  const double c = std::max(-1.0, std::min(1.0, a.dot(b) / (na * nb)));
  return Rad2Deg(std::acos(c));
}

Eigen::MatrixXd Symmetrize(const Eigen::MatrixXd& M) {
  return 0.5 * (M + M.transpose());
}

}  // namespace deform_monitor_v2
