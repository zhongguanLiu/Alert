#ifndef DEFORM_MONITOR_V2_CORE_OBSERVABLE_SUBSPACE_HPP
#define DEFORM_MONITOR_V2_CORE_OBSERVABLE_SUBSPACE_HPP

#include "deform_monitor_v2/data_types.hpp"

namespace deform_monitor_v2 {

struct ObservableSubspace {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Eigen::Matrix3d basis_R = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d projector_R = Eigen::Matrix3d::Zero();
  int rank = 0;
};

ObservableSubspace BuildObservableSubspace(const AnchorReference& anchor);

Eigen::Vector3d ProjectObservableVector(const Eigen::Vector3d& vector_R,
                                       const ObservableSubspace& subspace,
                                       int observed_rank = -1);

double ProjectedChiSquare(const Eigen::Vector3d& displacement_R,
                          const Eigen::Matrix3d& covariance_R,
                          const ObservableSubspace& subspace,
                          int observed_rank = -1);

}  // namespace deform_monitor_v2

#endif  // DEFORM_MONITOR_V2_CORE_OBSERVABLE_SUBSPACE_HPP
