/*
 * Author: zgliu@cumt.edu.cn
 * Affiliation: China University of Mining and Technology
 * Open-source release date: 2026-04-20
 */

#include "deform_monitor_v2/core/anchor_builder.hpp"

#include <Eigen/Eigenvalues>
#include <pcl/kdtree/kdtree_flann.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <unordered_map>

namespace deform_monitor_v2 {

namespace {

struct SeedKey {
  int x = 0;
  int y = 0;
  int z = 0;

  bool operator==(const SeedKey& other) const {
    return x == other.x && y == other.y && z == other.z;
  }
};

struct SeedKeyHash {
  size_t operator()(const SeedKey& key) const {
    size_t h = std::hash<int>()(key.x);
    h ^= std::hash<int>()(key.y) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<int>()(key.z) + 0x9e3779b9 + (h << 6) + (h >> 2);
    return h;
  }
};

double Clamp01(double value) {
  return std::max(0.0, std::min(1.0, value));
}

pcl::PointXYZI ToPclPoint(const Eigen::Vector3d& p) {
  pcl::PointXYZI pt;
  pt.x = static_cast<float>(p.x());
  pt.y = static_cast<float>(p.y());
  pt.z = static_cast<float>(p.z());
  pt.intensity = 0.0f;
  return pt;
}

Eigen::Vector3d SafeNormalized(const Eigen::Vector3d& v, const Eigen::Vector3d& fallback) {
  const double n = v.norm();
  if (n < 1.0e-9) {
    return fallback;
  }
  return v / n;
}

struct SupportStats {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  bool valid = false;
  Eigen::Vector3d center = Eigen::Vector3d::Zero();
  Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
  Eigen::Vector3d evals = Eigen::Vector3d::Zero();
  Eigen::Matrix3d evecs = Eigen::Matrix3d::Identity();
  Eigen::Vector3d normal = Eigen::Vector3d::UnitZ();
  Eigen::Vector3d dominant = Eigen::Vector3d::UnitX();
  Eigen::Vector3d mean_view_dir = Eigen::Vector3d::UnitX();
  double mean_range = 0.0;
  double mean_incidence_cos = 1.0;
  double linearity = 0.0;
  double planarity = 0.0;
  double scattering = 1.0;
  double info_budget = 0.0;
};

struct AnchorCandidate {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  AnchorReference anchor;
  double score = 0.0;
};

struct VoxelCell {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  int total_points = 0;
  int visible_frames = 0;
  int last_seen_frame = -1;
  Eigen::Vector3d sum = Eigen::Vector3d::Zero();
  AlignedVector<Eigen::Vector3d> points;
  std::vector<uint16_t> frames_seen;
};

SupportStats ComputeSupportStats(const AlignedVector<Eigen::Vector3d>& points,
                                 const Eigen::Vector3d& center_seed,
                                 const std::vector<Eigen::Vector3d>& frame_origins,
                                 const AnchorBuildParams& params) {
  SupportStats stats;
  if (points.size() < 3) {
    return stats;
  }

  stats.center.setZero();
  for (const auto& p : points) {
    stats.center += p;
  }
  stats.center /= static_cast<double>(points.size());

  for (const auto& p : points) {
    const Eigen::Vector3d d = p - stats.center;
    stats.cov += d * d.transpose();
  }
  stats.cov /= std::max(1.0, static_cast<double>(points.size() - 1));
  stats.cov = 0.5 * (stats.cov + stats.cov.transpose());

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(stats.cov);
  if (eig.info() != Eigen::Success) {
    return stats;
  }

  stats.evals = eig.eigenvalues();
  stats.evecs = eig.eigenvectors();
  const double l0 = std::max(1.0e-9, stats.evals(0));
  const double l1 = std::max(1.0e-9, stats.evals(1));
  const double l2 = std::max(1.0e-9, stats.evals(2));
  stats.linearity = Clamp01((l2 - l1) / l2);
  stats.planarity = Clamp01((l1 - l0) / l2);
  stats.scattering = Clamp01(l0 / (l0 + l1 + l2));
  stats.normal = SafeNormalized(stats.evecs.col(0), Eigen::Vector3d::UnitZ());
  stats.dominant = SafeNormalized(stats.evecs.col(2), Eigen::Vector3d::UnitX());

  stats.mean_view_dir.setZero();
  stats.mean_range = 0.0;
  stats.mean_incidence_cos = 0.0;
  double info_sum = 0.0;
  for (const auto& p : points) {
    double range = 0.0;
    Eigen::Vector3d view = Eigen::Vector3d::UnitX();
    if (!frame_origins.empty()) {
      double best_dist = std::numeric_limits<double>::infinity();
      for (const auto& origin : frame_origins) {
        const double dist = (p - origin).norm();
        if (dist < best_dist) {
          best_dist = dist;
          range = dist;
          view = SafeNormalized(p - origin, Eigen::Vector3d::UnitX());
        }
      }
    } else {
      range = p.norm();
      view = SafeNormalized(p, Eigen::Vector3d::UnitX());
    }

    stats.mean_range += range;
    stats.mean_view_dir += view;
    stats.mean_incidence_cos += std::abs(stats.normal.dot(-view));

    const double s_edge = stats.linearity;
    const double s_depth = 0.0;
    const double s_normal = Clamp01(1.0 - stats.scattering);
    const double s_view = Clamp01(std::abs(stats.normal.dot(-view)));
    info_sum += params.beta_edge * s_edge +
                params.beta_depth * s_depth +
                params.beta_normal * s_normal +
                params.beta_view * s_view;
  }

  stats.mean_range /= static_cast<double>(points.size());
  stats.mean_view_dir = SafeNormalized(stats.mean_view_dir, Eigen::Vector3d::UnitX());
  stats.mean_incidence_cos /= static_cast<double>(points.size());
  stats.info_budget = info_sum;
  stats.valid = (stats.center - center_seed).allFinite() && stats.info_budget >= 0.0;
  return stats;
}

AnchorType ChooseAnchorType(const SupportStats& stats) {
  if (stats.linearity > 0.62 && stats.linearity > stats.planarity + 0.10) {
    return AnchorType::EDGE;
  }
  if (stats.planarity > 0.42 && stats.scattering < 0.20) {
    return AnchorType::PLANE;
  }
  return AnchorType::BAND;
}

double DirectionAngleDeg(const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
  return AngleBetweenDeg(a, b);
}

double CovarianceQuality(const SupportStats& stats, int point_count) {
  const double anisotropy =
      Clamp01((stats.evals(2) - stats.evals(0)) / std::max(1.0e-9, stats.evals(2)));
  const double shape_quality =
      0.45 * Clamp01(1.0 - stats.scattering) +
      0.35 * Clamp01(std::max(stats.planarity, stats.linearity)) +
      0.20 * anisotropy;
  const double count_quality = Clamp01(static_cast<double>(point_count) / 12.0);
  return Clamp01(0.70 * shape_quality + 0.30 * count_quality);
}

double TypeStability(const SupportStats& stats, AnchorType type) {
  if (type == AnchorType::EDGE) {
    return Clamp01((stats.linearity - stats.planarity - 0.05) / 0.40);
  }
  if (type == AnchorType::PLANE) {
    const double plane_margin = stats.planarity - std::max(stats.linearity, stats.scattering);
    return Clamp01((plane_margin - 0.02) / 0.35);
  }
  const double balance = 1.0 - std::min(1.0, std::abs(stats.linearity - stats.planarity) / 0.25);
  return Clamp01(0.60 * balance + 0.40 * (1.0 - stats.scattering));
}

}  // namespace

void AnchorBuilder::SetParams(const AnchorBuildParams& params) {
  params_ = params;
}

AnchorReferenceVector AnchorBuilder::BuildFrozenAnchors(const ReferenceInitFrameVector& init_frames) {
  AnchorReferenceVector anchors;
  if (init_frames.empty()) {
    return anchors;
  }

  const double voxel_size = std::max(0.01, params_.voxel_size > 0.0 ? params_.voxel_size
                                                                     : params_.seed_voxel);
  const int min_visible_frames = std::max(1, params_.min_visible_frames);
  const int min_points_per_voxel = std::max(1, params_.min_points_per_voxel);
  const int neighborhood_layers = std::max(0, params_.neighborhood_layers);
  const int min_local_points = std::max(3, params_.min_support_points);
  const double support_radius =
      std::max(params_.radius_min, voxel_size * (0.60 + 0.30 * neighborhood_layers));

  std::vector<Eigen::Vector3d> frame_origins;
  frame_origins.reserve(init_frames.size());

  std::unordered_map<SeedKey, VoxelCell, SeedKeyHash> voxels;
  voxels.reserve(20000);
  size_t total_points = 0;
  for (size_t frame_idx = 0; frame_idx < init_frames.size(); ++frame_idx) {
    const auto& frame = init_frames[frame_idx];
    if (!frame.cloud) {
      continue;
    }
    frame_origins.push_back(frame.lidar_origin_R);
    for (const auto& pt : frame.cloud->points) {
      SeedKey key;
      key.x = static_cast<int>(std::floor(pt.x / voxel_size));
      key.y = static_cast<int>(std::floor(pt.y / voxel_size));
      key.z = static_cast<int>(std::floor(pt.z / voxel_size));
      auto& cell = voxels[key];
      const Eigen::Vector3d p(pt.x, pt.y, pt.z);
      cell.sum += p;
      cell.points.push_back(p);
      ++cell.total_points;
      if (cell.last_seen_frame != static_cast<int>(frame_idx)) {
        cell.last_seen_frame = static_cast<int>(frame_idx);
        ++cell.visible_frames;
        cell.frames_seen.push_back(static_cast<uint16_t>(frame_idx));
      }
      ++total_points;
    }
  }

  if (voxels.empty() || frame_origins.empty()) {
    std::cout << "[deform_monitor_v2] Reference init failed: empty cloud." << std::endl;
    return anchors;
  }

  std::vector<SeedKey> stable_keys;
  stable_keys.reserve(voxels.size());
  for (const auto& kv : voxels) {
    const auto& cell = kv.second;
    if (cell.visible_frames < min_visible_frames || cell.total_points < min_points_per_voxel) {
      continue;
    }
    stable_keys.push_back(kv.first);
  }
  if (stable_keys.empty()) {
    std::cout << "[deform_monitor_v2] Reference init failed: no stable voxels." << std::endl;
    return anchors;
  }

  std::cout << "[deform_monitor_v2] Build frozen anchors: init_frames="
            << init_frames.size()
            << " voxels=" << voxels.size()
            << " stable=" << stable_keys.size()
            << " points=" << total_points << std::endl;

  anchors.reserve(stable_keys.size());
  size_t processed_voxels = 0;
  int next_id = 0;
  for (const auto& key : stable_keys) {
    ++processed_voxels;
    if (processed_voxels % 500 == 0) {
      std::cout << "[deform_monitor_v2] Ref init progress: voxel="
                << processed_voxels << "/" << stable_keys.size()
                << " anchors=" << anchors.size() << std::endl;
    }

    const auto cell_it = voxels.find(key);
    if (cell_it == voxels.end()) {
      continue;
    }
    const auto& cell = cell_it->second;
    const Eigen::Vector3d cell_center = cell.sum / std::max(1, cell.total_points);

    AlignedVector<Eigen::Vector3d> local_points;
    std::vector<bool> frame_seen(init_frames.size(), false);
    for (int dx = -neighborhood_layers; dx <= neighborhood_layers; ++dx) {
      for (int dy = -neighborhood_layers; dy <= neighborhood_layers; ++dy) {
        for (int dz = -neighborhood_layers; dz <= neighborhood_layers; ++dz) {
          SeedKey nk{key.x + dx, key.y + dy, key.z + dz};
          const auto neigh_it = voxels.find(nk);
          if (neigh_it == voxels.end()) {
            continue;
          }
          const auto& neigh = neigh_it->second;
          local_points.insert(local_points.end(), neigh.points.begin(), neigh.points.end());
          for (const auto fid : neigh.frames_seen) {
            if (fid < frame_seen.size()) {
              frame_seen[fid] = true;
            }
          }
        }
      }
    }
    if (static_cast<int>(local_points.size()) < min_local_points) {
      continue;
    }

    int local_visible_frames = 0;
    for (bool seen : frame_seen) {
      if (seen) {
        ++local_visible_frames;
      }
    }
    if (local_visible_frames < min_visible_frames) {
      continue;
    }

    const SupportStats support_stats =
        ComputeSupportStats(local_points, cell_center, frame_origins, params_);
    if (!support_stats.valid || support_stats.info_budget < params_.I_min) {
      continue;
    }

    const AnchorType type = ChooseAnchorType(support_stats);
    const double covariance_quality = CovarianceQuality(support_stats,
                                                       static_cast<int>(local_points.size()));
    const double type_stability = TypeStability(support_stats, type);
    Eigen::Vector3d normal = support_stats.normal;
    Eigen::Vector3d e1 = Eigen::Vector3d::UnitX();
    Eigen::Vector3d e2 = Eigen::Vector3d::UnitY();
    if (type == AnchorType::PLANE) {
      e1 = support_stats.dominant - support_stats.dominant.dot(normal) * normal;
      e1 = SafeNormalized(e1, Eigen::Vector3d::UnitX());
      e2 = SafeNormalized(normal.cross(e1), Eigen::Vector3d::UnitY());
    } else {
      e2 = SafeNormalized(support_stats.dominant, Eigen::Vector3d::UnitY());
      e1 = SafeNormalized(e2.cross(normal), Eigen::Vector3d::UnitX());
      e2 = SafeNormalized(normal.cross(e1), Eigen::Vector3d::UnitY());
    }

    const double avg_local_support =
        static_cast<double>(local_points.size()) / std::max(1, local_visible_frames);
    const double q_support = Clamp01(avg_local_support / 12.0);
    const double q_repeat =
        Clamp01(static_cast<double>(local_visible_frames) /
                static_cast<double>(std::max<size_t>(1, init_frames.size())));
    const double q_center =
        Clamp01(1.0 - (support_stats.center - cell_center).norm() /
                          std::max(0.5 * voxel_size, 1.0e-6));
    const double q_normal = Clamp01(1.0 - support_stats.scattering);
    const double q_contrast =
        Clamp01(std::max(support_stats.linearity, support_stats.planarity));
    double ref_quality = 0.20 * q_support + 0.25 * q_repeat +
                         0.15 * q_center + 0.10 * q_normal + 0.10 * q_contrast +
                         0.10 * covariance_quality + 0.10 * type_stability;
    if (type == AnchorType::EDGE) {
      ref_quality += params_.edge_ref_bonus * Clamp01(support_stats.linearity);
    } else if (type == AnchorType::BAND) {
      ref_quality += params_.band_ref_bonus *
                     Clamp01(std::max(support_stats.linearity, support_stats.planarity));
    }
    ref_quality = Clamp01(ref_quality);
    if (ref_quality < params_.tau_ref_quality) {
      continue;
    }

    AnchorReference anchor;
    anchor.id = next_id++;
    anchor.type = type;
    anchor.center_R = cell_center;
    anchor.normal_R = normal;
    anchor.edge_normal_R = e1;
    anchor.basis_R.col(0) = e1;
    anchor.basis_R.col(1) = e2;
    anchor.basis_R.col(2) = normal;
    anchor.Sigma_ref_geom =
        support_stats.cov / std::max(1.0, static_cast<double>(local_points.size()));
    anchor.Sigma_ref_geom += Eigen::Matrix3d::Identity() * 1.0e-6;
    anchor.support_points_R = cell.points;
    anchor.ref_quality = ref_quality;
    anchor.mean_range = support_stats.mean_range;
    anchor.mean_view_dir_R = support_stats.mean_view_dir;
    anchor.mean_incidence_cos = support_stats.mean_incidence_cos;
    anchor.support_radius = support_radius;
    anchor.support_target_count =
        std::max(min_points_per_voxel, static_cast<int>(std::round(avg_local_support)));
    anchor.point_count = static_cast<int>(local_points.size());

    anchor.edge_center_R = Eigen::Vector3d::Zero();
    int edge_count = 0;
    for (const auto& p : local_points) {
      if (e1.dot(p - anchor.center_R) > 0.0) {
        anchor.edge_center_R += p;
        ++edge_count;
      }
    }
    if (edge_count > 0) {
      anchor.edge_center_R /= static_cast<double>(edge_count);
    } else {
      anchor.edge_center_R = anchor.center_R;
    }
    anchor.band_center_R = anchor.center_R;
    anchor.visible_count = local_visible_frames;
    anchor.matched_count = 0;
    anchor.covariance_quality = covariance_quality;
    anchor.type_stability = type_stability;
    anchor.frozen = true;
    anchors.push_back(anchor);
  }

  std::cout << "[deform_monitor_v2] Frozen anchors done: anchors=" << anchors.size()
            << " / stable_voxels=" << stable_keys.size()
            << " / total_voxels=" << voxels.size() << std::endl;

  if (!anchors.empty()) {
    double mean_visible = 0.0;
    double mean_points = 0.0;
    double mean_cov_q = 0.0;
    double mean_type_stability = 0.0;
    size_t low_visible = 0;
    size_t low_points = 0;
    size_t low_cov_q = 0;
    size_t low_type_stability = 0;
    for (const auto& anchor : anchors) {
      mean_visible += static_cast<double>(anchor.visible_count);
      mean_points += static_cast<double>(anchor.point_count);
      mean_cov_q += anchor.covariance_quality;
      mean_type_stability += anchor.type_stability;
      if (anchor.visible_count < std::max(min_visible_frames + 1, min_visible_frames * 2)) {
        ++low_visible;
      }
      if (anchor.point_count < std::max(min_points_per_voxel * 2, 8)) {
        ++low_points;
      }
      if (anchor.covariance_quality < 0.55) {
        ++low_cov_q;
      }
      if (anchor.type_stability < 0.50) {
        ++low_type_stability;
      }
    }
    const double inv_n = 1.0 / static_cast<double>(anchors.size());
    std::cout << "[deform_monitor_v2] Anchor quality summary:"
              << " mean_visible=" << mean_visible * inv_n
              << " mean_points=" << mean_points * inv_n
              << " mean_cov_q=" << mean_cov_q * inv_n
              << " mean_type_stability=" << mean_type_stability * inv_n
              << " low_visible=" << low_visible
              << " low_points=" << low_points
              << " low_cov_q=" << low_cov_q
              << " low_type_stability=" << low_type_stability
              << std::endl;
  }

  return anchors;
}

AnchorReferenceVector AnchorBuilder::BuildIncrementalAnchors(
    const ReferenceInitFrameVector& recent_frames,
    const AnchorReferenceVector& existing_anchors,
    double coverage_radius,
    int start_id,
    int min_visible_frames_override,
    int max_new_anchors) {
  AnchorReferenceVector anchors;
  if (recent_frames.empty()) {
    return anchors;
  }

  const double voxel_size = std::max(0.01, params_.voxel_size > 0.0 ? params_.voxel_size
                                                                     : params_.seed_voxel);
  const int min_visible_frames = std::max(1, min_visible_frames_override);
  const int min_points_per_voxel = std::max(1, params_.min_points_per_voxel);
  const int neighborhood_layers = std::max(0, params_.neighborhood_layers);
  const int min_local_points = std::max(3, params_.min_support_points);
  const double support_radius =
      std::max(params_.radius_min, voxel_size * (0.60 + 0.30 * neighborhood_layers));


  const double cov_voxel = std::max(0.01, coverage_radius);
  std::unordered_map<SeedKey, bool, SeedKeyHash> coverage_hash;
  coverage_hash.reserve(existing_anchors.size() * 4);
  for (const auto& anchor : existing_anchors) {
    SeedKey key;
    key.x = static_cast<int>(std::floor(anchor.center_R.x() / cov_voxel));
    key.y = static_cast<int>(std::floor(anchor.center_R.y() / cov_voxel));
    key.z = static_cast<int>(std::floor(anchor.center_R.z() / cov_voxel));

    for (int dx = -1; dx <= 1; ++dx) {
      for (int dy = -1; dy <= 1; ++dy) {
        for (int dz = -1; dz <= 1; ++dz) {
          coverage_hash[{key.x + dx, key.y + dy, key.z + dz}] = true;
        }
      }
    }
  }

  std::vector<Eigen::Vector3d> frame_origins;
  frame_origins.reserve(recent_frames.size());

  std::unordered_map<SeedKey, VoxelCell, SeedKeyHash> voxels;
  voxels.reserve(5000);
  for (size_t frame_idx = 0; frame_idx < recent_frames.size(); ++frame_idx) {
    const auto& frame = recent_frames[frame_idx];
    if (!frame.cloud) {
      continue;
    }
    frame_origins.push_back(frame.lidar_origin_R);
    for (const auto& pt : frame.cloud->points) {

      SeedKey cov_key;
      cov_key.x = static_cast<int>(std::floor(pt.x / cov_voxel));
      cov_key.y = static_cast<int>(std::floor(pt.y / cov_voxel));
      cov_key.z = static_cast<int>(std::floor(pt.z / cov_voxel));
      if (coverage_hash.count(cov_key)) {
        continue;
      }

      SeedKey key;
      key.x = static_cast<int>(std::floor(pt.x / voxel_size));
      key.y = static_cast<int>(std::floor(pt.y / voxel_size));
      key.z = static_cast<int>(std::floor(pt.z / voxel_size));
      auto& cell = voxels[key];
      const Eigen::Vector3d p(pt.x, pt.y, pt.z);
      cell.sum += p;
      cell.points.push_back(p);
      ++cell.total_points;
      if (cell.last_seen_frame != static_cast<int>(frame_idx)) {
        cell.last_seen_frame = static_cast<int>(frame_idx);
        ++cell.visible_frames;
        cell.frames_seen.push_back(static_cast<uint16_t>(frame_idx));
      }
    }
  }

  if (voxels.empty() || frame_origins.empty()) {
    return anchors;
  }

  std::vector<SeedKey> stable_keys;
  stable_keys.reserve(voxels.size());
  for (const auto& kv : voxels) {
    const auto& cell = kv.second;
    if (cell.visible_frames < min_visible_frames || cell.total_points < min_points_per_voxel) {
      continue;
    }
    stable_keys.push_back(kv.first);
  }
  if (stable_keys.empty()) {
    return anchors;
  }

  anchors.reserve(std::min(static_cast<size_t>(max_new_anchors), stable_keys.size()));
  int next_id = start_id;
  for (const auto& key : stable_keys) {
    if (static_cast<int>(anchors.size()) >= max_new_anchors) {
      break;
    }

    const auto cell_it = voxels.find(key);
    if (cell_it == voxels.end()) {
      continue;
    }
    const auto& cell = cell_it->second;
    const Eigen::Vector3d cell_center = cell.sum / std::max(1, cell.total_points);


    bool covered = false;
    for (const auto& anchor : existing_anchors) {
      if ((anchor.center_R - cell_center).norm() < coverage_radius) {
        covered = true;
        break;
      }
    }
    if (covered) {
      continue;
    }

    AlignedVector<Eigen::Vector3d> local_points;
    std::vector<bool> frame_seen(recent_frames.size(), false);
    for (int dx = -neighborhood_layers; dx <= neighborhood_layers; ++dx) {
      for (int dy = -neighborhood_layers; dy <= neighborhood_layers; ++dy) {
        for (int dz = -neighborhood_layers; dz <= neighborhood_layers; ++dz) {
          SeedKey nk{key.x + dx, key.y + dy, key.z + dz};
          const auto neigh_it = voxels.find(nk);
          if (neigh_it == voxels.end()) {
            continue;
          }
          const auto& neigh = neigh_it->second;
          local_points.insert(local_points.end(), neigh.points.begin(), neigh.points.end());
          for (const auto fid : neigh.frames_seen) {
            if (fid < frame_seen.size()) {
              frame_seen[fid] = true;
            }
          }
        }
      }
    }
    if (static_cast<int>(local_points.size()) < min_local_points) {
      continue;
    }

    int local_visible_frames = 0;
    for (bool seen : frame_seen) {
      if (seen) {
        ++local_visible_frames;
      }
    }
    if (local_visible_frames < min_visible_frames) {
      continue;
    }

    const SupportStats support_stats =
        ComputeSupportStats(local_points, cell_center, frame_origins, params_);
    if (!support_stats.valid || support_stats.info_budget < params_.I_min) {
      continue;
    }

    const AnchorType type = ChooseAnchorType(support_stats);
    const double covariance_quality = CovarianceQuality(support_stats,
                                                        static_cast<int>(local_points.size()));
    const double type_stability = TypeStability(support_stats, type);
    Eigen::Vector3d normal = support_stats.normal;
    Eigen::Vector3d e1 = Eigen::Vector3d::UnitX();
    Eigen::Vector3d e2 = Eigen::Vector3d::UnitY();
    if (type == AnchorType::PLANE) {
      e1 = support_stats.dominant - support_stats.dominant.dot(normal) * normal;
      e1 = SafeNormalized(e1, Eigen::Vector3d::UnitX());
      e2 = SafeNormalized(normal.cross(e1), Eigen::Vector3d::UnitY());
    } else {
      e2 = SafeNormalized(support_stats.dominant, Eigen::Vector3d::UnitY());
      e1 = SafeNormalized(e2.cross(normal), Eigen::Vector3d::UnitX());
      e2 = SafeNormalized(normal.cross(e1), Eigen::Vector3d::UnitY());
    }

    const double avg_local_support =
        static_cast<double>(local_points.size()) / std::max(1, local_visible_frames);
    const double q_support = Clamp01(avg_local_support / 12.0);
    const double q_repeat =
        Clamp01(static_cast<double>(local_visible_frames) /
                static_cast<double>(std::max<size_t>(1, recent_frames.size())));
    const double q_center =
        Clamp01(1.0 - (support_stats.center - cell_center).norm() /
                          std::max(0.5 * voxel_size, 1.0e-6));
    const double q_normal = Clamp01(1.0 - support_stats.scattering);
    const double q_contrast =
        Clamp01(std::max(support_stats.linearity, support_stats.planarity));
    double ref_quality = 0.20 * q_support + 0.25 * q_repeat +
                         0.15 * q_center + 0.10 * q_normal + 0.10 * q_contrast +
                         0.10 * covariance_quality + 0.10 * type_stability;
    if (type == AnchorType::EDGE) {
      ref_quality += params_.edge_ref_bonus * Clamp01(support_stats.linearity);
    } else if (type == AnchorType::BAND) {
      ref_quality += params_.band_ref_bonus *
                     Clamp01(std::max(support_stats.linearity, support_stats.planarity));
    }
    ref_quality = Clamp01(ref_quality);
    if (ref_quality < params_.tau_ref_quality) {
      continue;
    }

    AnchorReference anchor;
    anchor.id = next_id++;
    anchor.type = type;
    anchor.center_R = cell_center;
    anchor.normal_R = normal;
    anchor.edge_normal_R = e1;
    anchor.basis_R.col(0) = e1;
    anchor.basis_R.col(1) = e2;
    anchor.basis_R.col(2) = normal;
    anchor.Sigma_ref_geom =
        support_stats.cov / std::max(1.0, static_cast<double>(local_points.size()));
    anchor.Sigma_ref_geom += Eigen::Matrix3d::Identity() * 1.0e-6;
    anchor.support_points_R = cell.points;
    anchor.ref_quality = ref_quality;
    anchor.mean_range = support_stats.mean_range;
    anchor.mean_view_dir_R = support_stats.mean_view_dir;
    anchor.mean_incidence_cos = support_stats.mean_incidence_cos;
    anchor.support_radius = support_radius;
    anchor.support_target_count =
        std::max(min_points_per_voxel, static_cast<int>(std::round(avg_local_support)));
    anchor.point_count = static_cast<int>(local_points.size());

    anchor.edge_center_R = Eigen::Vector3d::Zero();
    int edge_count = 0;
    for (const auto& p : local_points) {
      if (e1.dot(p - anchor.center_R) > 0.0) {
        anchor.edge_center_R += p;
        ++edge_count;
      }
    }
    if (edge_count > 0) {
      anchor.edge_center_R /= static_cast<double>(edge_count);
    } else {
      anchor.edge_center_R = anchor.center_R;
    }
    anchor.band_center_R = anchor.center_R;
    anchor.visible_count = local_visible_frames;
    anchor.matched_count = 0;
    anchor.covariance_quality = covariance_quality;
    anchor.type_stability = type_stability;
    anchor.frozen = true;
    anchors.push_back(anchor);
  }

  if (!anchors.empty()) {
    std::cout << "[deform_monitor_v2] Incremental anchors: added=" << anchors.size()
              << " uncovered=" << stable_keys.size()
              << " frames=" << recent_frames.size() << std::endl;
  }

  return anchors;
}

}  // namespace deform_monitor_v2
