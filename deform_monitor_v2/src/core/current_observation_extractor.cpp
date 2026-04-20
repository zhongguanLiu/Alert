/*
 * Author: zgliu@cumt.edu.cn
 * Affiliation: China University of Mining and Technology
 * Open-source release date: 2026-04-20
 */

#include "deform_monitor_v2/core/current_observation_extractor.hpp"

#include "deform_monitor_v2/core/covariance_extractor.hpp"

#include <Eigen/Eigenvalues>

#include <algorithm>
#include <cmath>
#include <limits>

namespace deform_monitor_v2 {

namespace {

double Clamp01(double value) {
  return std::max(0.0, std::min(1.0, value));
}

Eigen::Vector3d SafeNormalized(const Eigen::Vector3d& v, const Eigen::Vector3d& fallback) {
  const double n = v.norm();
  if (n < 1.0e-9) {
    return fallback;
  }
  return v / n;
}

Eigen::Matrix3d PoseOnlyPointCovariance(const Eigen::Vector3d& x_R,
                                        const Eigen::Vector3d& lidar_origin_R,
                                        const Eigen::Matrix<double, 6, 6>& Sigma_xi) {
  const Eigen::Vector3d Rp = x_R - lidar_origin_R;
  Eigen::Matrix<double, 3, 6> J = Eigen::Matrix<double, 3, 6>::Zero();
  J.block<3, 3>(0, 0) = -SkewSymmetric(Rp);
  J.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity();
  Eigen::Matrix3d S = J * Sigma_xi * J.transpose();
  return 0.5 * (S + S.transpose());
}

void FillExpectedObservability(const AnchorReference& anchor,
                               const Eigen::Vector3d& lidar_origin_R,
                               const ObservationParams& observation_params,
                               LocalSupportData* support) {
  if (!support) {
    return;
  }
  const Eigen::Vector3d expected_view =
      SafeNormalized(anchor.center_R - lidar_origin_R, anchor.mean_view_dir_R);
  const double expected_range = (anchor.center_R - lidar_origin_R).norm();
  support->expected_view_angle_deg =
      AngleBetweenDeg(expected_view, anchor.mean_view_dir_R);
  support->expected_range_ratio =
      std::abs(expected_range - anchor.mean_range) / std::max(1.0e-3, anchor.mean_range);
  support->expected_incidence_cos = std::abs(anchor.normal_R.dot(-expected_view));
  support->expected_observable =
      support->expected_view_angle_deg <= 1.25 * observation_params.tau_v_deg &&
      support->expected_range_ratio <= 1.25 * observation_params.tau_r_ratio &&
      support->expected_incidence_cos >= 0.20;
}

ObsGateState DetermineGateState(const LocalSupportData& support,
                                const ObservationParams& observation_params) {
  if (!support.expected_observable) {
    return ObsGateState::NOT_OBSERVABLE;
  }
  if (support.valid && support.comparable) {
    return ObsGateState::OBSERVABLE_MATCHED;
  }
  if (support.valid && support.support_count >= observation_params.min_support_scalar) {
    return ObsGateState::OBSERVABLE_REPLACED;
  }
  if (support.support_count > 0) {
    return ObsGateState::OBSERVABLE_WEAK;
  }
  return ObsGateState::OBSERVABLE_MISSING;
}

CurrentObservation BuildObservationFromSupport(const AnchorReference& anchor,
                                               const LocalSupportData& support,
                                               const PoseCov6D& pose_cov,
                                               const Eigen::Vector3d& lidar_origin_R,
                                               const ObservationParams& observation_params,
                                               const ObservabilityParams& observability_params,
                                               const ScalarMeasurementBuilder& measurement_builder) {
  CurrentObservation obs;
  obs.anchor_id = anchor.id;
  obs.stamp = pose_cov.stamp;
  obs.cmp_score = support.cmp_score;
  obs.support_count = support.support_count;
  obs.fit_rmse = support.fit_rmse;
  obs.overlap_score = support.overlap_score;
  obs.status = support.status;
  obs.gate_state = support.gate_state;
  obs.observable = support.gate_state != ObsGateState::NOT_OBSERVABLE;
  obs.comparable = support.comparable;
  obs.reacquired = support.reacquired;
  obs.matched_center_R = support.centroid_R;
  if (support.valid) {
    obs.matched_delta_R = support.centroid_R - anchor.center_R;
  } else {
    obs.matched_delta_R.setZero();
  }

  if (!support.valid || !support.comparable) {
    const double expected_support = std::max(1.0, static_cast<double>(anchor.support_target_count));
    const double support_loss =
        1.0 - Clamp01(static_cast<double>(support.support_count) / expected_support);
    const double overlap_loss = 1.0 - Clamp01(support.overlap_score);
    const double geom_change =
        std::max(Clamp01(support.fit_rmse / 0.03),
                 Clamp01(support.normal_angle_deg / std::max(1.0, observation_params.tau_n_deg)));
    const double shift_change =
        Clamp01(support.center_shift_norm / std::max(0.03, observation_params.reacquire_radius));
    double disappearance_score =
        0.35 * support_loss + 0.25 * overlap_loss + 0.20 * geom_change + 0.20 * shift_change;
    if (obs.gate_state == ObsGateState::NOT_OBSERVABLE) {
      disappearance_score = 0.0;
    } else if (obs.gate_state == ObsGateState::OBSERVABLE_WEAK) {
      disappearance_score *= 0.40;
    } else if (obs.gate_state == ObsGateState::OBSERVABLE_MISSING) {
      disappearance_score = std::max(disappearance_score, 0.78);
    } else if (obs.gate_state == ObsGateState::OBSERVABLE_REPLACED) {
      disappearance_score = std::max(disappearance_score, 0.68);
    }
    if (anchor.type == AnchorType::PLANE && obs.gate_state != ObsGateState::OBSERVABLE_REPLACED) {
      disappearance_score *= 0.35;
    }
    obs.disappearance_score = Clamp01(disappearance_score);
    obs.status = obs.observable ? ObsStatus::VALID_PARTIAL_OBS : ObsStatus::INVALID_NO_COMPARISON;
    obs.dof_obs = 0;
    return obs;
  }

  obs.scalars = measurement_builder.BuildMeasurements(anchor, support, pose_cov, lidar_origin_R);
  if (obs.scalars.empty()) {
    obs.status = obs.observable ? ObsStatus::VALID_PARTIAL_OBS : ObsStatus::INVALID_NO_COMPARISON;
    obs.comparable = false;
    obs.gate_state = obs.observable ? ObsGateState::OBSERVABLE_WEAK : ObsGateState::NOT_OBSERVABLE;
    obs.disappearance_score =
        obs.observable
            ? Clamp01(0.45 + 0.20 * Clamp01(obs.matched_delta_R.norm() /
                                            std::max(0.03, observation_params.reacquire_radius)))
            : 0.0;
    obs.dof_obs = 0;
    return obs;
  }

  Eigen::Matrix3d info = Eigen::Matrix3d::Zero();
  for (const auto& scalar : obs.scalars) {
    info += (scalar.h_R * scalar.h_R.transpose()) / std::max(1.0e-9, scalar.r);
  }

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(info);
  if (eig.info() != Eigen::Success) {
    obs.status = ObsStatus::VALID_PARTIAL_OBS;
    obs.comparable = true;
    obs.dof_obs = 0;
    return obs;
  }

  int rank = 0;
  for (int i = 0; i < 3; ++i) {
    if (eig.eigenvalues()(i) > observability_params.tau_lambda) {
      ++rank;
    }
  }
  obs.dof_obs = rank;
  obs.status = rank >= 3 ? ObsStatus::VALID_FULL_OBS : ObsStatus::VALID_PARTIAL_OBS;
  obs.comparable = true;
  return obs;
}

}  // namespace

void CurrentObservationExtractor::SetParams(const ObservationParams& params) {
  params_ = params;
  cached_frame_voxel_maps_.clear();
}

void CurrentObservationExtractor::SetTemporalParams(const TemporalFusionParams& params) {
  temporal_params_ = params;
  cached_frame_voxel_maps_.clear();
}

void CurrentObservationExtractor::SetNoiseParams(const NoiseParams& params) {
  noise_params_ = params;
}

void CurrentObservationExtractor::SetCovarianceParams(const CovarianceParams& params) {
  covariance_params_ = params;
  cached_frame_voxel_maps_.clear();
}

void CurrentObservationExtractor::SetObservabilityParams(const ObservabilityParams& params) {
  observability_params_ = params;
}

void CurrentObservationExtractor::SetMeasurementBuilder(const ScalarMeasurementBuilder& builder) {
  measurement_builder_ = builder;
}

void CurrentObservationExtractor::PrepareSingleFrame(
    const pcl::PointCloud<pcl::PointXYZI>::ConstPtr& curr_cloud,
    const PoseCov6D& curr_pose_cov,
    const Eigen::Vector3d& lidar_origin_R) {
  EnsureSingleFrameVoxelMap(curr_cloud, curr_pose_cov, lidar_origin_R);
}

void CurrentObservationExtractor::PrepareTemporalWindow(const ObservationFrameDeque& frames) {
  EnsureWindowVoxelMaps(frames);
}

CurrentObservationExtractor::FrameVoxelMap
CurrentObservationExtractor::BuildFrameVoxelMap(
    const ObservationFrame& frame) const {
  FrameVoxelMap map;
  if (!frame.cloud || frame.cloud->empty()) {
    return map;
  }

  const double voxel_size = std::max(0.01, params_.current_voxel_size);
  map.voxel_size = voxel_size;
  map.stamp = frame.stamp;
  map.pose_cov = frame.pose_cov;
  map.lidar_origin_R = frame.lidar_origin_R;
  map.voxels.reserve(2048);

  for (const auto& pt : frame.cloud->points) {
    const Eigen::Vector3d p(pt.x, pt.y, pt.z);
    VoxelKey key;
    key.x = static_cast<int>(std::floor(p.x() / voxel_size));
    key.y = static_cast<int>(std::floor(p.y() / voxel_size));
    key.z = static_cast<int>(std::floor(p.z() / voxel_size));
    auto& cell = map.voxels[key];
    cell.points_R.push_back(p);
    cell.point_covariances.push_back(
        CovarianceExtractor::PointCovarianceFromReferencePoint(
            p, frame.lidar_origin_R, frame.pose_cov.Sigma_xi, covariance_params_.sigma_p));
    cell.sum_R += p;
    ++cell.total_points;
  }
  map.valid = !map.voxels.empty();
  return map;
}

void CurrentObservationExtractor::EnsureWindowVoxelMaps(
    const ObservationFrameDeque& frames) {
  if (frames.empty()) {
    cached_frame_voxel_maps_.clear();
    return;
  }

  const double voxel_size = std::max(0.01, params_.current_voxel_size);
  bool rebuild = false;
  if (!cached_frame_voxel_maps_.empty()) {
    if (std::abs(cached_frame_voxel_maps_.front().voxel_size - voxel_size) > 1.0e-9) {
      rebuild = true;
    }
  }

  if (rebuild) {
    cached_frame_voxel_maps_.clear();
  }

  while (!cached_frame_voxel_maps_.empty() &&
         cached_frame_voxel_maps_.front().stamp < frames.front().stamp) {
    cached_frame_voxel_maps_.pop_front();
  }

  if (!cached_frame_voxel_maps_.empty()) {
    if (cached_frame_voxel_maps_.size() > frames.size()) {
      rebuild = true;
    } else {
      for (size_t i = 0; i < cached_frame_voxel_maps_.size(); ++i) {
        if (cached_frame_voxel_maps_[i].stamp != frames[i].stamp) {
          rebuild = true;
          break;
        }
      }
    }
  }

  if (rebuild) {
    cached_frame_voxel_maps_.clear();
  }

  const size_t cached_count = cached_frame_voxel_maps_.size();
  for (size_t i = cached_count; i < frames.size(); ++i) {
    cached_frame_voxel_maps_.push_back(BuildFrameVoxelMap(frames[i]));
  }
}

void CurrentObservationExtractor::EnsureSingleFrameVoxelMap(
    const pcl::PointCloud<pcl::PointXYZI>::ConstPtr& curr_cloud,
    const PoseCov6D& curr_pose_cov,
    const Eigen::Vector3d& lidar_origin_R) {
  ObservationFrame frame;
  frame.cloud = curr_cloud;
  frame.pose_cov = curr_pose_cov;
  frame.lidar_origin_R = lidar_origin_R;
  frame.stamp = curr_pose_cov.stamp;
  ObservationFrameDeque frames;
  frames.push_back(frame);
  EnsureWindowVoxelMaps(frames);
}

CurrentObservationExtractor::NearbySupportPool
CurrentObservationExtractor::BuildPoolFromCachedVoxelMaps(
    const AnchorReference& anchor) const {
  NearbySupportPool pool;
  if (cached_frame_voxel_maps_.empty()) {
    return pool;
  }

  pool.frame_pose_covs.reserve(cached_frame_voxel_maps_.size());
  pool.frame_origins_R.reserve(cached_frame_voxel_maps_.size());
  for (const auto& frame_map : cached_frame_voxel_maps_) {
    pool.frame_pose_covs.push_back(frame_map.pose_cov);
    pool.frame_origins_R.push_back(frame_map.lidar_origin_R);
  }
  if (cached_frame_voxel_maps_.size() >= 2) {
    pool.window_span_sec =
        std::max(0.0, (cached_frame_voxel_maps_.back().stamp -
                       cached_frame_voxel_maps_.front().stamp).toSec());
  }

  const double local_radius = std::min(0.16, std::max(0.03, anchor.support_radius + 0.02));
  const double wide_radius = std::max(local_radius, params_.reacquire_radius);
  const double voxel_size =
      std::max(0.01, cached_frame_voxel_maps_.front().voxel_size);
  const int layers = std::max(1, static_cast<int>(std::ceil(wide_radius / voxel_size)));
  const double cell_margin = 0.9 * voxel_size;
  const double max_cell_center_dist = wide_radius + cell_margin;
  const double max_cell_center_dist2 = max_cell_center_dist * max_cell_center_dist;

  VoxelKey center_key;
  center_key.x = static_cast<int>(std::floor(anchor.center_R.x() / voxel_size));
  center_key.y = static_cast<int>(std::floor(anchor.center_R.y() / voxel_size));
  center_key.z = static_cast<int>(std::floor(anchor.center_R.z() / voxel_size));

  if (temporal_params_.use_stable_voxel_fusion && cached_frame_voxel_maps_.size() > 1) {
    struct AggregatedVoxelCell {
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW
      AlignedVector<Eigen::Vector3d> points_R;
      AlignedVector<Eigen::Matrix3d> point_covariances;
      std::vector<int> frame_indices;
      Eigen::Vector3d sum_R = Eigen::Vector3d::Zero();
      int total_points = 0;
      int visible_frames = 0;
    };

    AlignedUnorderedMap<VoxelKey, AggregatedVoxelCell, VoxelKeyHash> fused_voxels;
    fused_voxels.reserve(512);

    for (size_t frame_idx = 0; frame_idx < cached_frame_voxel_maps_.size(); ++frame_idx) {
      const auto& frame_map = cached_frame_voxel_maps_[frame_idx];
      if (!frame_map.valid) {
        continue;
      }
      for (int dx = -layers; dx <= layers; ++dx) {
        for (int dy = -layers; dy <= layers; ++dy) {
          for (int dz = -layers; dz <= layers; ++dz) {
            VoxelKey key{center_key.x + dx, center_key.y + dy, center_key.z + dz};
            const auto cell_it = frame_map.voxels.find(key);
            if (cell_it == frame_map.voxels.end()) {
              continue;
            }
            const auto& cell = cell_it->second;
            if (cell.total_points <= 0) {
              continue;
            }
            const Eigen::Vector3d cell_center = cell.sum_R / static_cast<double>(cell.total_points);
            if ((cell_center - anchor.center_R).squaredNorm() > max_cell_center_dist2) {
              continue;
            }

            auto& fused = fused_voxels[key];
            ++fused.visible_frames;
            fused.total_points += cell.total_points;
            fused.sum_R += cell.sum_R;
            fused.points_R.insert(fused.points_R.end(), cell.points_R.begin(), cell.points_R.end());
            fused.point_covariances.insert(fused.point_covariances.end(),
                                           cell.point_covariances.begin(),
                                           cell.point_covariances.end());
            fused.frame_indices.insert(fused.frame_indices.end(),
                                       cell.points_R.size(),
                                       static_cast<int>(frame_idx));
          }
        }
      }
    }

    const int min_visible_frames =
        std::max(1,
                 std::min(static_cast<int>(cached_frame_voxel_maps_.size()),
                          temporal_params_.min_fused_visible_frames));
    const int min_points_per_voxel =
        std::max(1, temporal_params_.min_fused_points_per_voxel);

    size_t kept_points = 0;
    for (const auto& kv : fused_voxels) {
      const auto& fused = kv.second;
      if (fused.visible_frames < min_visible_frames ||
          fused.total_points < min_points_per_voxel) {
        continue;
      }
      pool.candidate_centers_R.push_back(
          fused.sum_R / static_cast<double>(std::max(1, fused.total_points)));
      kept_points += fused.points_R.size();
    }

    if (pool.candidate_centers_R.empty() || kept_points == 0) {
      return pool;
    }

    pool.points_R.reserve(kept_points);
    pool.point_covariances.reserve(kept_points);
    pool.frame_indices.reserve(kept_points);
    for (const auto& kv : fused_voxels) {
      const auto& fused = kv.second;
      if (fused.visible_frames < min_visible_frames ||
          fused.total_points < min_points_per_voxel) {
        continue;
      }
      pool.points_R.insert(pool.points_R.end(), fused.points_R.begin(), fused.points_R.end());
      pool.point_covariances.insert(pool.point_covariances.end(),
                                    fused.point_covariances.begin(),
                                    fused.point_covariances.end());
      pool.frame_indices.insert(pool.frame_indices.end(),
                                fused.frame_indices.begin(),
                                fused.frame_indices.end());
    }
    return pool;
  }

  size_t total_points = 0;
  size_t total_cells = 0;
  for (const auto& frame_map : cached_frame_voxel_maps_) {
    if (!frame_map.valid) {
      continue;
    }
    for (int dx = -layers; dx <= layers; ++dx) {
      for (int dy = -layers; dy <= layers; ++dy) {
        for (int dz = -layers; dz <= layers; ++dz) {
          VoxelKey key{center_key.x + dx, center_key.y + dy, center_key.z + dz};
          const auto cell_it = frame_map.voxels.find(key);
          if (cell_it == frame_map.voxels.end()) {
            continue;
          }
          const auto& cell = cell_it->second;
          if (cell.total_points <= 0) {
            continue;
          }
          const Eigen::Vector3d cell_center = cell.sum_R / static_cast<double>(cell.total_points);
          if ((cell_center - anchor.center_R).squaredNorm() > max_cell_center_dist2) {
            continue;
          }
          pool.candidate_centers_R.push_back(cell_center);
          total_points += cell.points_R.size();
          ++total_cells;
        }
      }
    }
  }

  if (total_cells == 0 || total_points == 0) {
    return pool;
  }

  pool.points_R.reserve(total_points);
  pool.point_covariances.reserve(total_points);
  pool.frame_indices.reserve(total_points);
  for (size_t frame_idx = 0; frame_idx < cached_frame_voxel_maps_.size(); ++frame_idx) {
    const auto& frame_map = cached_frame_voxel_maps_[frame_idx];
    if (!frame_map.valid) {
      continue;
    }
    for (int dx = -layers; dx <= layers; ++dx) {
      for (int dy = -layers; dy <= layers; ++dy) {
        for (int dz = -layers; dz <= layers; ++dz) {
          VoxelKey key{center_key.x + dx, center_key.y + dy, center_key.z + dz};
          const auto cell_it = frame_map.voxels.find(key);
          if (cell_it == frame_map.voxels.end()) {
            continue;
          }
          const auto& cell = cell_it->second;
          if (cell.total_points <= 0) {
            continue;
          }
          const Eigen::Vector3d cell_center = cell.sum_R / static_cast<double>(cell.total_points);
          if ((cell_center - anchor.center_R).squaredNorm() > max_cell_center_dist2) {
            continue;
          }
          pool.points_R.insert(pool.points_R.end(), cell.points_R.begin(), cell.points_R.end());
          pool.point_covariances.insert(pool.point_covariances.end(),
                                        cell.point_covariances.begin(),
                                        cell.point_covariances.end());
          pool.frame_indices.insert(pool.frame_indices.end(),
                                    cell.points_R.size(),
                                    static_cast<int>(frame_idx));
        }
      }
    }
  }
  return pool;
}

LocalSupportData CurrentObservationExtractor::BuildSupportAtCenter(
    const AnchorReference& anchor,
    const NearbySupportPool& pool,
    const Eigen::Vector3d& candidate_center_R) const {
  LocalSupportData support;
  support.anchor_id = anchor.id;
  support.valid = false;
  support.status = ObsStatus::INVALID_NO_COMPARISON;
  if (!pool.frame_origins_R.empty()) {
    FillExpectedObservability(anchor,
                              pool.frame_origins_R.back(),
                              params_,
                              &support);
  }

  if (pool.points_R.empty() ||
      pool.points_R.size() != pool.point_covariances.size() ||
      pool.points_R.size() != pool.frame_indices.size()) {
    support.gate_state = DetermineGateState(support, params_);
    return support;
  }

  const double search_radius = std::min(0.16, std::max(0.03, anchor.support_radius + 0.02));
  const double radius2 = search_radius * search_radius;

  std::vector<int> selected_frame_indices;
  support.support_points_R.reserve(pool.points_R.size());
  support.point_covariances.reserve(pool.point_covariances.size());
  selected_frame_indices.reserve(pool.frame_indices.size());
  for (size_t i = 0; i < pool.points_R.size(); ++i) {
    const Eigen::Vector3d& p = pool.points_R[i];
    if ((p - candidate_center_R).squaredNorm() > radius2) {
      continue;
    }
    support.support_points_R.push_back(p);
    support.point_covariances.push_back(pool.point_covariances[i]);
    selected_frame_indices.push_back(pool.frame_indices[i]);
  }

  const int min_support_required =
      anchor.type == AnchorType::PLANE ? std::max(params_.min_support_scalar, 3)
                                       : std::max(params_.min_support_scalar, 3);
  support.support_count = static_cast<int>(support.support_points_R.size());
  if (support.support_count < min_support_required) {
    return support;
  }

  support.centroid_R.setZero();
  for (const auto& p : support.support_points_R) {
    support.centroid_R += p;
  }
  support.centroid_R /= static_cast<double>(support.support_count);

  support.centroid_cov.setZero();
  for (const auto& cov : support.point_covariances) {
    support.centroid_cov += cov;
  }
  support.centroid_cov /= std::max(
      1.0, static_cast<double>(support.support_count * support.support_count));
  support.centroid_cov += Eigen::Matrix3d::Identity() * 1.0e-9;

  support.local_cov.setZero();
  for (const auto& p : support.support_points_R) {
    const Eigen::Vector3d d = p - support.centroid_R;
    support.local_cov += d * d.transpose();
  }
  support.local_cov /= std::max(1.0, static_cast<double>(support.support_count - 1));
  support.local_cov = 0.5 * (support.local_cov + support.local_cov.transpose());

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(support.local_cov);
  if (eig.info() != Eigen::Success) {
    return support;
  }

  const Eigen::Vector3d evals = eig.eigenvalues();
  const Eigen::Matrix3d evecs = eig.eigenvectors();
  support.normal_R = SafeNormalized(evecs.col(0), anchor.normal_R);
  if (support.normal_R.dot(anchor.normal_R) < 0.0) {
    support.normal_R = -support.normal_R;
  }
  const Eigen::Vector3d dominant = SafeNormalized(evecs.col(2), anchor.basis_R.col(1));
  support.edge_normal_R = dominant - dominant.dot(support.normal_R) * support.normal_R;
  support.edge_normal_R = SafeNormalized(support.edge_normal_R, anchor.edge_normal_R);
  if (support.edge_normal_R.dot(anchor.edge_normal_R) < 0.0) {
    support.edge_normal_R = -support.edge_normal_R;
  }
  support.band_axis_R = SafeNormalized(support.normal_R.cross(support.edge_normal_R),
                                       anchor.basis_R.col(1));

  support.fit_rmse = std::sqrt(std::max(0.0, evals(0)));
  support.view_dir_R.setZero();
  support.range = 0.0;
  support.incidence_cos = 0.0;
  std::vector<int> frame_support_counts(pool.frame_pose_covs.size(), 0);
  for (size_t i = 0; i < support.support_points_R.size(); ++i) {
    const int frame_idx = selected_frame_indices[i];
    if (frame_idx < 0 || frame_idx >= static_cast<int>(pool.frame_origins_R.size())) {
      continue;
    }
    ++frame_support_counts[static_cast<size_t>(frame_idx)];
    const Eigen::Vector3d view = SafeNormalized(
        support.support_points_R[i] - pool.frame_origins_R[static_cast<size_t>(frame_idx)],
        anchor.mean_view_dir_R);
    support.view_dir_R += view;
    support.range +=
        (support.support_points_R[i] -
         pool.frame_origins_R[static_cast<size_t>(frame_idx)]).norm();
    support.incidence_cos += std::abs(support.normal_R.dot(-view));
  }
  support.view_dir_R = SafeNormalized(support.view_dir_R, anchor.mean_view_dir_R);
  support.range /= static_cast<double>(support.support_count);
  support.incidence_cos /= static_cast<double>(support.support_count);
  support.center_shift_norm = (support.centroid_R - anchor.center_R).norm();

  if (pool.frame_pose_covs.size() > 1) {
    const double total_support = static_cast<double>(support.support_count);
    for (size_t frame_idx = 0; frame_idx < frame_support_counts.size(); ++frame_idx) {
      const int count = frame_support_counts[frame_idx];
      if (count <= 0) {
        continue;
      }
      const double count_d = static_cast<double>(count);
      const double w_corr = (count_d / total_support) * (count_d / total_support);
      const double w_ind = count_d / (total_support * total_support);
      const double extra_w = std::max(0.0, w_corr - w_ind);
      if (extra_w <= 0.0) {
        continue;
      }
      support.centroid_cov +=
          extra_w * temporal_params_.pose_corr_inflation *
          PoseOnlyPointCovariance(
              support.centroid_R,
              pool.frame_origins_R[frame_idx],
              pool.frame_pose_covs[frame_idx].Sigma_xi);
    }

    if (pool.window_span_sec > 1.0e-6) {
      const double sigma_blur =
          temporal_params_.sigma_motion_per_sec * pool.window_span_sec;
      support.centroid_cov +=
          Eigen::Matrix3d::Identity() * sigma_blur * sigma_blur;
    }
  }

  support.edge_centroid_R.setZero();
  int edge_count = 0;
  for (const auto& p : support.support_points_R) {
    if (anchor.edge_normal_R.dot(p - support.centroid_R) <= 0.0) {
      continue;
    }
    support.edge_centroid_R += p;
    ++edge_count;
  }
  if (edge_count > 0) {
    support.edge_centroid_R /= static_cast<double>(edge_count);
    support.edge_centroid_cov = support.centroid_cov *
                                static_cast<double>(support.support_count) /
                                static_cast<double>(edge_count * edge_count);
  } else {
    support.edge_centroid_R = support.centroid_R;
    support.edge_centroid_cov = support.centroid_cov;
  }
  support.band_centroid_R = support.centroid_R;
  support.band_centroid_cov = support.centroid_cov;

  support.valid = true;
  EvaluateComparability(anchor, &support);
  support.gate_state = DetermineGateState(support, params_);
  return support;
}

void CurrentObservationExtractor::EvaluateComparability(const AnchorReference& anchor,
                                                        LocalSupportData* support) const {
  if (!support || !support->valid) {
    return;
  }

  const bool is_plane = anchor.type == AnchorType::PLANE;
  const double tau_n_deg = is_plane ? params_.tau_n_deg : params_.tau_n_deg * 1.10;
  const double tau_v_deg = is_plane ? params_.tau_v_deg : params_.tau_v_deg * 1.30;
  const double tau_r_ratio = is_plane ? params_.tau_r_ratio : params_.tau_r_ratio * 1.40;
  const double tau_cmp = is_plane ? params_.tau_cmp : std::max(0.56, params_.tau_cmp - 0.10);

  support->normal_angle_deg = AngleBetweenDeg(support->normal_R, anchor.normal_R);
  support->view_angle_deg = AngleBetweenDeg(support->view_dir_R, anchor.mean_view_dir_R);
  support->range_ratio =
      std::abs(support->range - anchor.mean_range) / std::max(1.0e-3, anchor.mean_range);

  // Hard gates that are always enforced: geometric consistency and grazing angle
  if (support->normal_angle_deg > tau_n_deg ||
      support->incidence_cos < 0.20) {
    support->comparable = false;
    support->status = ObsStatus::INVALID_NO_COMPARISON;
    return;
  }

  if (params_.soft_view_range_gate) {
    // Soft gate mode: view_angle and range_ratio are absorbed into cmp_score
    // but we still enforce absolute safety caps to reject extreme cases
    if (support->view_angle_deg > params_.hard_view_deg_cap ||
        support->range_ratio > params_.hard_range_ratio_cap) {
      support->comparable = false;
      support->status = ObsStatus::INVALID_NO_COMPARISON;
      return;
    }
  } else {
    // Original hard gate mode: strict thresholds for stationary monitoring
    if (support->view_angle_deg > tau_v_deg ||
        support->range_ratio > tau_r_ratio) {
      support->comparable = false;
      support->status = ObsStatus::INVALID_NO_COMPARISON;
      return;
    }
  }

  const double ref_inc_angle = std::acos(Clamp01(anchor.mean_incidence_cos));
  const double curr_inc_angle = std::acos(Clamp01(support->incidence_cos));
  // When soft gate is on, use the expanded caps as normalization denominators
  // so that larger view/range deviations produce proportionally lower scores
  const double v_denom = params_.soft_view_range_gate ? params_.hard_view_deg_cap : tau_v_deg;
  const double r_denom = params_.soft_view_range_gate ? params_.hard_range_ratio_cap : tau_r_ratio;
  const double S_view = std::max(0.0, 1.0 - support->view_angle_deg / v_denom);
  const double S_range = std::max(0.0, 1.0 - support->range_ratio / r_denom);
  const double S_inc = std::max(0.0, std::cos(curr_inc_angle - ref_inc_angle));
  const double S_quality = Clamp01(1.0 - support->fit_rmse / 0.03);
  const double S_support =
      std::min(1.0, static_cast<double>(support->support_count) /
                        std::max(1.0, static_cast<double>(anchor.support_target_count)));
  support->overlap_score =
      std::min(1.0, static_cast<double>(support->support_count) /
                        std::max(1.0, static_cast<double>(anchor.support_target_count)));
  support->cmp_score = 0.15 * S_view + 0.10 * S_range + 0.10 * S_inc +
                       0.20 * S_quality + 0.20 * S_support +
                       0.25 * support->overlap_score;
  support->comparable = support->cmp_score >= tau_cmp;
  support->status = support->comparable ? ObsStatus::VALID_PARTIAL_OBS
                                        : ObsStatus::INVALID_NO_COMPARISON;
}

LocalSupportData CurrentObservationExtractor::BuildSupportForAnchorFromCachedMaps(
    const AnchorReference& anchor,
    const Eigen::Vector3d& lidar_origin_R) const {
  LocalSupportData empty_support;
  empty_support.anchor_id = anchor.id;
  empty_support.status = ObsStatus::INVALID_NO_COMPARISON;
  FillExpectedObservability(anchor, lidar_origin_R, params_, &empty_support);
  empty_support.gate_state = DetermineGateState(empty_support, params_);

  NearbySupportPool pool = BuildPoolFromCachedVoxelMaps(anchor);
  if (pool.points_R.empty()) {
    return empty_support;
  }

  LocalSupportData local_support = BuildSupportAtCenter(anchor, pool, anchor.center_R);
  if (local_support.comparable) {
    return local_support;
  }

  // Reacquire only for PLANE anchors: EDGE/BAND are geometrically unstable and
  // generate persistent false positives when reacquire finds shifted voxel clusters.
  if (anchor.type != AnchorType::PLANE) {
    return local_support;
  }

  LocalSupportData best_support = local_support;
  double best_score = -std::numeric_limits<double>::infinity();
  if (static_cast<int>(pool.candidate_centers_R.size()) >=
      std::max(params_.min_support_scalar, 2)) {
    const int stride = std::max(
        1, static_cast<int>(pool.candidate_centers_R.size()) /
               std::max(1, params_.max_reacquire_candidates));
    for (size_t i = 0; i < pool.candidate_centers_R.size(); i += static_cast<size_t>(stride)) {
      const Eigen::Vector3d& candidate_center_R = pool.candidate_centers_R[i];
      if ((candidate_center_R - anchor.center_R).norm() < 0.01) {
        continue;
      }
      LocalSupportData candidate_support =
          BuildSupportAtCenter(anchor, pool, candidate_center_R);
      if (!candidate_support.valid || !candidate_support.comparable) {
        continue;
      }
      const double min_shift =
          anchor.type == AnchorType::PLANE
              ? std::max(0.010, 0.08 * params_.reacquire_radius)
              : std::max(0.015, 0.10 * params_.reacquire_radius);
      if (candidate_support.center_shift_norm < min_shift) {
        continue;
      }
      if (anchor.type == AnchorType::PLANE &&
          candidate_support.normal_angle_deg > params_.tau_n_deg) {
        continue;
      }
      const double shift_bonus = Clamp01(candidate_support.center_shift_norm /
                                         std::max(0.02, params_.reacquire_radius));
      const double quality_bonus = Clamp01(1.0 - candidate_support.fit_rmse / 0.02);
      const double score =
          anchor.type == AnchorType::PLANE
              ? 0.65 * candidate_support.cmp_score +
                    0.20 * shift_bonus +
                    0.15 * quality_bonus
              : candidate_support.cmp_score + 0.10 * shift_bonus;
      const double tau_reacquire =
          anchor.type == AnchorType::PLANE
              ? std::max(0.74, params_.tau_reacquire)
              : std::max(0.58, params_.tau_reacquire - 0.12);
      if (score >= std::max(tau_reacquire, best_score)) {
        best_score = score;
        candidate_support.reacquired = true;
        best_support = candidate_support;
      }
    }
  }

  if (best_support.reacquired) {
    best_support.status = ObsStatus::VALID_PARTIAL_OBS;
    best_support.comparable = true;
    return best_support;
  }
  if (local_support.comparable) {
    return local_support;
  }
  return best_support;
}

LocalSupportData CurrentObservationExtractor::BuildSupportForAnchor(
    const AnchorReference& anchor,
    const pcl::PointCloud<pcl::PointXYZI>::ConstPtr& curr_cloud,
    const PoseCov6D& curr_pose_cov,
    const Eigen::Vector3d& lidar_origin_R) {
  if (!curr_cloud || curr_cloud->empty()) {
    LocalSupportData empty_support;
    empty_support.anchor_id = anchor.id;
    empty_support.status = ObsStatus::INVALID_NO_COMPARISON;
    FillExpectedObservability(anchor, lidar_origin_R, params_, &empty_support);
    empty_support.gate_state = DetermineGateState(empty_support, params_);
    return empty_support;
  }

  EnsureSingleFrameVoxelMap(curr_cloud, curr_pose_cov, lidar_origin_R);
  return BuildSupportForAnchorFromCachedMaps(anchor, lidar_origin_R);
}

LocalSupportData CurrentObservationExtractor::BuildSupportForAnchorTemporal(
    const AnchorReference& anchor,
    const ObservationFrameDeque& frames) {
  if (frames.empty()) {
    LocalSupportData empty_support;
    empty_support.anchor_id = anchor.id;
    empty_support.status = ObsStatus::INVALID_NO_COMPARISON;
    return empty_support;
  }

  EnsureWindowVoxelMaps(frames);
  return BuildSupportForAnchorFromCachedMaps(anchor, frames.back().lidar_origin_R);
}

CurrentObservation CurrentObservationExtractor::ExtractForAnchorFromPreparedCache(
    const AnchorReference& anchor,
    const PoseCov6D& pose_cov,
    const Eigen::Vector3d& lidar_origin_R) const {
  const LocalSupportData support = BuildSupportForAnchorFromCachedMaps(anchor, lidar_origin_R);
  return BuildObservationFromSupport(anchor,
                                     support,
                                     pose_cov,
                                     lidar_origin_R,
                                     params_,
                                     observability_params_,
                                     measurement_builder_);
}

}  // namespace deform_monitor_v2
