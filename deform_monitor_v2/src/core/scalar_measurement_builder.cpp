/*
 * Author: zgliu@cumt.edu.cn
 * Affiliation: China University of Mining and Technology
 * Open-source release date: 2026-04-20
 */

#include "deform_monitor_v2/core/scalar_measurement_builder.hpp"

#include <cmath>

namespace deform_monitor_v2 {

namespace {

bool IsFiniteScalar(double v) {
  return std::isfinite(v);
}

Eigen::Vector3d SafeNormalized(const Eigen::Vector3d& v, const Eigen::Vector3d& fallback) {
  const double n = v.norm();
  if (n < 1.0e-9) {
    return fallback;
  }
  return v / n;
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
    const Eigen::Vector3d& lidar_origin_R) const {
  AlignedVector<ScalarMeasurement> scalars;
  if (!support.valid || support.support_count < observation_params_.min_support_scalar) {
    return scalars;
  }

  auto maybe_add = [&](const Eigen::Vector3d& h,
                       double z,
                       double r,
                       uint8_t type) {
    const double h_norm = h.norm();
    if (h_norm < 0.9 || h_norm > 1.1 || !IsFiniteScalar(z) || !IsFiniteScalar(r) || r <= 0.0) {
      return;
    }
    if (support.support_count < observation_params_.min_support_scalar) {
      return;
    }
    const double nis_like = std::abs(z) / std::sqrt(r);
    if (nis_like > observation_params_.tau_nis_scalar) {
      return;
    }
    ScalarMeasurement m;
    m.h_R = h / h_norm;
    m.z = z;
    m.r = r;
    m.type = type;
    scalars.push_back(m);
  };

  const Eigen::Vector3d n = SafeNormalized(anchor.normal_R, Eigen::Vector3d::UnitZ());
  const Eigen::Vector3d e1 = SafeNormalized(anchor.edge_normal_R, Eigen::Vector3d::UnitX());
  const Eigen::Vector3d e2 = SafeNormalized(anchor.basis_R.col(1), Eigen::Vector3d::UnitY());
  const Eigen::Vector3d radial =
      SafeNormalized(support.centroid_R - lidar_origin_R, Eigen::Vector3d::UnitX());

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
  maybe_add(n, z_plane, r_plane, 0);

  if (anchor.type != AnchorType::PLANE) {
    const double sigma_edge = noise_params_.sigma_edge0 * view_noise_scale;
    const double z_edge = e1.dot(support.edge_centroid_R - anchor.edge_center_R);
    const double r_edge =
        e1.dot((support.edge_centroid_cov + anchor.Sigma_ref_geom) * e1) +
        sigma_edge * sigma_edge;
    maybe_add(e1, z_edge, r_edge, 1);
  }

  const double sigma_rad = noise_params_.sigma_rad0 * view_noise_scale;
  const double z_rad = radial.dot(support.centroid_R - anchor.center_R);
  const double r_rad =
      radial.dot((support.centroid_cov + anchor.Sigma_ref_geom) * radial) +
      sigma_rad * sigma_rad;
  maybe_add(radial, z_rad, r_rad, 2);

  if (support.reacquired) {
    const Eigen::Vector3d delta = support.centroid_R - anchor.center_R;
    const double delta_norm = delta.norm();
    if (delta_norm > 1.0e-3) {
      const Eigen::Vector3d motion_axis = SafeNormalized(delta, n);
      const double r_motion =
          motion_axis.dot((support.centroid_cov + anchor.Sigma_ref_geom) * motion_axis) +
          noise_params_.sigma_bc0 * noise_params_.sigma_bc0;
      maybe_add(motion_axis, motion_axis.dot(delta), r_motion, 4);
    }

    const double z_e1 = e1.dot(delta);
    if (std::abs(z_e1) > 1.0e-3) {
      const double r_e1 =
          e1.dot((support.centroid_cov + anchor.Sigma_ref_geom) * e1) +
          noise_params_.sigma_edge0 * noise_params_.sigma_edge0;
      maybe_add(e1, z_e1, r_e1, 5);
    }

    const double z_e2 = e2.dot(delta);
    if (std::abs(z_e2) > 1.0e-3) {
      const double r_e2 =
          e2.dot((support.centroid_cov + anchor.Sigma_ref_geom) * e2) +
          noise_params_.sigma_bc0 * noise_params_.sigma_bc0;
      maybe_add(e2, z_e2, r_e2, 6);
    }
  }

  if (anchor.type == AnchorType::BAND) {
    const Eigen::Vector3d delta = support.band_centroid_R - anchor.band_center_R;
    const Eigen::Vector3d candidates[4] = {n, e1, e2, radial};
    Eigen::Vector3d axis = n;
    double best_score = -1.0;
    for (const auto& cand : candidates) {
      const double score = std::abs(cand.dot(delta));
      if (score > best_score) {
        best_score = score;
        axis = cand;
      }
    }
    axis = SafeNormalized(axis, n);
    const double z_band = axis.dot(delta);
    const double r_band =
        axis.dot((support.band_centroid_cov + anchor.Sigma_ref_geom) * axis) +
        noise_params_.sigma_bc0 * noise_params_.sigma_bc0;
    maybe_add(axis, z_band, r_band, 3);
  }

  return scalars;
}

}  // namespace deform_monitor_v2
