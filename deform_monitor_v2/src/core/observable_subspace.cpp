#include "deform_monitor_v2/core/observable_subspace.hpp"

#include <Eigen/Eigenvalues>

#include <algorithm>
#include <cmath>

namespace deform_monitor_v2 {
namespace {

Eigen::Vector3d SafeNormalized(const Eigen::Vector3d& vector_R,
                               const Eigen::Vector3d& fallback_R) {
  const double norm = vector_R.norm();
  if (!std::isfinite(norm) || norm < 1.0e-9) {
    return fallback_R;
  }
  return vector_R / norm;
}

int EffectiveRank(const ObservableSubspace& subspace, int observed_rank) {
  if (observed_rank < 0) {
    return subspace.rank;
  }
  return std::max(0, std::min(subspace.rank, observed_rank));
}

}  // namespace

ObservableSubspace BuildObservableSubspace(const AnchorReference& anchor) {
  ObservableSubspace subspace;
  const Eigen::Vector3d n =
      SafeNormalized(anchor.normal_R, Eigen::Vector3d::UnitZ());
  subspace.basis_R.col(0) = n;
  subspace.rank = 1;

  if (anchor.type == AnchorType::EDGE || anchor.type == AnchorType::BAND) {
    const Eigen::Vector3d raw_secondary =
        anchor.type == AnchorType::EDGE ? anchor.edge_normal_R : anchor.basis_R.col(1);
    Eigen::Vector3d secondary = raw_secondary - raw_secondary.dot(n) * n;
    const Eigen::Vector3d fallback =
        std::abs(n.dot(Eigen::Vector3d::UnitX())) < 0.9
            ? n.cross(Eigen::Vector3d::UnitX())
            : n.cross(Eigen::Vector3d::UnitY());
    secondary = SafeNormalized(secondary, SafeNormalized(fallback, Eigen::Vector3d::UnitY()));
    subspace.basis_R.col(1) = secondary;
    subspace.rank = 2;
  }

  subspace.projector_R =
      subspace.basis_R.leftCols(subspace.rank) *
      subspace.basis_R.leftCols(subspace.rank).transpose();
  return subspace;
}

Eigen::Vector3d ProjectObservableVector(const Eigen::Vector3d& vector_R,
                                       const ObservableSubspace& subspace,
                                       int observed_rank) {
  const int rank = EffectiveRank(subspace, observed_rank);
  if (rank <= 0) {
    return Eigen::Vector3d::Zero();
  }
  const auto basis = subspace.basis_R.leftCols(rank);
  return basis * (basis.transpose() * vector_R);
}

double ProjectedChiSquare(const Eigen::Vector3d& displacement_R,
                          const Eigen::Matrix3d& covariance_R,
                          const ObservableSubspace& subspace,
                          int observed_rank) {
  const int rank = EffectiveRank(subspace, observed_rank);
  if (rank <= 0) {
    return 0.0;
  }

  const auto basis = subspace.basis_R.leftCols(rank);
  const Eigen::VectorXd displacement = basis.transpose() * displacement_R;
  Eigen::MatrixXd covariance = basis.transpose() *
                               (0.5 * (covariance_R + covariance_R.transpose())) *
                               basis;
  covariance = 0.5 * (covariance + covariance.transpose());

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(covariance);
  if (eig.info() != Eigen::Success) {
    return 0.0;
  }

  double chi2 = 0.0;
  for (int i = 0; i < rank; ++i) {
    const double eigenvalue = eig.eigenvalues()(i);
    if (!std::isfinite(eigenvalue) || eigenvalue <= 1.0e-12) {
      continue;
    }
    const double component = eig.eigenvectors().col(i).dot(displacement);
    chi2 += component * component / eigenvalue;
  }
  return std::isfinite(chi2) ? chi2 : 0.0;
}

}  // namespace deform_monitor_v2
