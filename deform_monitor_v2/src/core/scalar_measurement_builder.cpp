#include "deform_monitor_v2/core/scalar_measurement_builder.hpp"

#include "deform_monitor_v2/core/observable_subspace.hpp"

#include <cmath>

namespace deform_monitor_v2 {

namespace {

bool IsFiniteScalar(double v) {
  return std::isfinite(v);
}

}  // namespace

void ScalarMeasurementBuilder::SetParams(const ObservationParams& observation_params,
                                         const NoiseParams& noise_params) {
  observation_params_ = observation_params;
  noise_params_ = noise_params;
}

AlignedVector<ScalarMeasurement> ScalarMeasurementBuilder::BuildMeasurements(
    const AnchorReference& anchor,
    const LocalSupportData& support,
    const PoseCov6D& /*pose_cov*/,
    const Eigen::Vector3d& /*lidar_origin_R*/) const {
  AlignedVector<ScalarMeasurement> scalars;
  if (!support.valid || support.support_count < observation_params_.min_support_scalar) {
    return scalars;
  }

  auto maybe_add = [&](const Eigen::Vector3d& h,
                       double z,
                       double r,
                       uint8_t type) -> bool {
    const double h_norm = h.norm();
    if (h_norm < 0.9 || h_norm > 1.1 || !IsFiniteScalar(z) || !IsFiniteScalar(r) || r <= 0.0) {
      return false;
    }
    if (support.support_count < observation_params_.min_support_scalar) {
      return false;
    }
    const double nis_like = std::abs(z) / std::sqrt(r);
    if (nis_like > observation_params_.tau_nis_scalar) {
      return false;
    }
    ScalarMeasurement m;
    m.h_R = h / h_norm;
    m.z = z;
    m.r = r;
    m.type = type;
    scalars.push_back(m);
    return true;
  };

  const ObservableSubspace subspace = BuildObservableSubspace(anchor);
  const Eigen::Vector3d n = subspace.basis_R.col(0);

  // View-dependent noise inflation: applies to all anchor types when the
  // sensor has moved, increasing measurement variance proportionally to
  // range change and view angle change.
  const double view_angle_deg = AngleBetweenDeg(support.view_dir_R, anchor.mean_view_dir_R);
  const double range_delta = std::max(0.0, support.range - anchor.mean_range);
  const double view_noise_scale =
      1.0 + noise_params_.kappa_r * range_delta + noise_params_.kappa_v * view_angle_deg;

  const double sigma_plane = noise_params_.sigma_pi0 * view_noise_scale;
  const double z_plane = n.dot(support.centroid_R - anchor.center_R);
  const double r_plane =
      n.dot(support.centroid_cov * n) +
      n.dot(anchor.Sigma_ref_geom * n) +
      sigma_plane * sigma_plane;
  // The normal measurement is the common observable dimension for all anchor
  // types. If it is invalid, a secondary scalar is not allowed to update the
  // state on its own.
  if (!maybe_add(n, z_plane, r_plane, 0)) {
    return scalars;
  }

  if (anchor.type == AnchorType::EDGE) {
    const Eigen::Vector3d e1 = subspace.basis_R.col(1);
    const int min_edge_support =
        std::max(2, (observation_params_.min_support_scalar + 1) / 2);
    if (support.edge_geometry_valid &&
        support.edge_support_count >= min_edge_support) {
      const double sigma_edge = noise_params_.sigma_edge0 * view_noise_scale;
      const double z_edge = e1.dot(support.edge_centroid_R - anchor.edge_center_R);
      const double r_edge =
          e1.dot((support.edge_centroid_cov + anchor.Sigma_ref_edge) * e1) +
          sigma_edge * sigma_edge;
      maybe_add(e1, z_edge, r_edge, 1);
    }
  } else if (anchor.type == AnchorType::BAND) {
    const Eigen::Vector3d e2 = subspace.basis_R.col(1);
    const Eigen::Vector3d delta = support.band_centroid_R - anchor.band_center_R;
    const double z_band = e2.dot(delta);
    const double r_band =
        e2.dot((support.band_centroid_cov + anchor.Sigma_ref_geom) * e2) +
        noise_params_.sigma_bc0 * noise_params_.sigma_bc0;
    maybe_add(e2, z_band, r_band, 3);
  }

  return scalars;
}

}  // namespace deform_monitor_v2
