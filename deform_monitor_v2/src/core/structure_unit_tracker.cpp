/*
 * Author: zgliu@cumt.edu.cn
 * Affiliation: China University of Mining and Technology
 * Open-source release date: 2026-04-20
 */

#include "deform_monitor_v2/structure_unit_tracker.hpp"
#include <unordered_map>
#include <tuple>
#include <cmath>

namespace deform_monitor_v2 {

StructureUnitTracker::StructureUnitTracker(
    const StructureUnitTrackerParams& params,
    const StructureUnitVector& units)
    : params_(params), units_(units) {}

bool StructureUnitTracker::IsStructuralFailure(
    const AnchorTrackState& state) const {

  if (state.gate_state == ObsGateState::NOT_OBSERVABLE) return false;

  if (state.comparable) return false;

  return true;
}

double StructureUnitTracker::ComputeExitScore(
    const StructureUnit& unit,
    const AnchorStateVector& states) const {

  std::unordered_map<int, const AnchorTrackState*> id_map;
  id_map.reserve(states.size());
  for (const auto& s : states) id_map[s.id] = &s;

  double w_sum = 0.0;
  double exit_sum = 0.0;
  for (int mid : unit.member_ids) {
    auto it = id_map.find(mid);
    if (it == id_map.end()) continue;
    const AnchorTrackState& s = *it->second;

    if (s.gate_state == ObsGateState::NOT_OBSERVABLE) continue;

    if (s.observable && s.comparable) continue;
    const double w = 1.0;
    const double failure = IsStructuralFailure(s) ? 1.0 : 0.0;
    exit_sum += failure * w;
    w_sum += w;
  }
  return (w_sum > 1e-9) ? exit_sum / w_sum : 0.0;
}

std::vector<StructureUnitTracker::EntryCandidateCloud>
StructureUnitTracker::FindEntryCandidates(
    const pcl::PointCloud<pcl::PointXYZ>& cloud,
    const StructureUnit& unit,
    const AnchorStateVector& states) const {


  const Eigen::Vector3d margin = Eigen::Vector3d::Constant(params_.search_margin);
  const Eigen::Vector3d search_min = unit.ref_bbox_min_R - margin;
  const Eigen::Vector3d search_max = unit.ref_bbox_max_R + margin;


  std::vector<Eigen::Vector3d> anchor_positions;
  anchor_positions.reserve(states.size());
  for (const auto& s : states) {
    if (s.gate_state == ObsGateState::NOT_OBSERVABLE) continue;
    if (!s.observable) continue;
    anchor_positions.push_back(s.matched_center_R);
  }


  const double inv_voxel = 1.0 / params_.orphan_voxel_size;
  using VoxelKey3 = std::tuple<int, int, int>;
  struct VK3Hash {
    size_t operator()(const VoxelKey3& k) const {
      size_t h = std::hash<int>{}(std::get<0>(k)) * 2654435761ULL;
      h ^= std::hash<int>{}(std::get<1>(k)) * 2246822519ULL;
      h ^= std::hash<int>{}(std::get<2>(k)) * 3266489917ULL;
      return h;
    }
  };
  std::unordered_map<VoxelKey3, std::vector<Eigen::Vector3d>, VK3Hash> voxels;

  const double orphan_r2 = params_.orphan_radius * params_.orphan_radius;
  for (const auto& pt : cloud.points) {
    const Eigen::Vector3d p(pt.x, pt.y, pt.z);

    if ((p - search_min).minCoeff() < 0.0 || (search_max - p).minCoeff() < 0.0)
      continue;

    bool is_orphan = true;
    for (const auto& ap : anchor_positions) {
      if ((p - ap).squaredNorm() < orphan_r2) { is_orphan = false; break; }
    }
    if (!is_orphan) continue;
    const int ix = static_cast<int>(std::floor(p.x() * inv_voxel));
    const int iy = static_cast<int>(std::floor(p.y() * inv_voxel));
    const int iz = static_cast<int>(std::floor(p.z() * inv_voxel));
    voxels[{ix, iy, iz}].push_back(p);
  }


  std::vector<EntryCandidateCloud> candidates;
  candidates.reserve(voxels.size());
  for (auto& kv : voxels) {
    const auto& pts = kv.second;
    if (static_cast<int>(pts.size()) < params_.orphan_min_points) continue;
    EntryCandidateCloud cand;
    cand.point_count = static_cast<int>(pts.size());
    cand.bbox_min = Eigen::Vector3d::Constant(1e9);
    cand.bbox_max = Eigen::Vector3d::Constant(-1e9);
    for (const auto& p : pts) {
      cand.centroid += p;
      cand.bbox_min = cand.bbox_min.cwiseMin(p);
      cand.bbox_max = cand.bbox_max.cwiseMax(p);
    }
    cand.centroid /= static_cast<double>(pts.size());
    cand.normal = unit.ref_normal_R;
    candidates.push_back(cand);
  }
  return candidates;
}

bool StructureUnitTracker::Validate(
    const StructureUnit& unit,
    const EntryCandidateCloud& cand,
    StructureMigration& out) const {


  const Eigen::Vector3d ref_size = unit.ref_bbox_max_R - unit.ref_bbox_min_R;
  const Eigen::Vector3d cur_size = cand.bbox_max - cand.bbox_min;
  double size_score = 0.0;
  int valid_dims = 0;
  for (int k = 0; k < 3; ++k) {
    if (ref_size[k] < 1e-3) continue;
    ++valid_dims;
    const double ratio = std::abs(cur_size[k] - ref_size[k]) / ref_size[k];
    size_score += std::max(0.0, 1.0 - ratio / params_.max_size_gap);
  }
  if (valid_dims > 0) size_score /= valid_dims;
  if (size_score < 0.3) return false;


  const double cos_n = std::abs(cand.normal.dot(unit.ref_normal_R));
  const double thresh_n = std::cos(params_.max_normal_deg * M_PI / 180.0);
  if (cos_n < thresh_n) return false;
  const double normal_score = (cos_n - thresh_n) / (1.0 - thresh_n + 1e-9);


  const Eigen::Vector3d T = cand.centroid - unit.ref_centroid_R;
  const double dist = T.norm();
  if (dist < params_.min_migration_dist || dist > params_.max_migration_dist)
    return false;
  const double dist_score = std::max(0.0, 1.0 - dist / params_.max_migration_dist);

  const double confidence = 0.4 * size_score + 0.4 * normal_score + 0.2 * dist_score;
  if (confidence < params_.min_migration_confidence) return false;

  out.T = T;
  out.entry_centroid_R = cand.centroid;
  out.entry_bbox_min_R = cand.bbox_min;
  out.entry_bbox_max_R = cand.bbox_max;
  out.confidence = confidence;
  out.confirmed = true;
  return true;
}

void StructureUnitTracker::Update(
    const pcl::PointCloud<pcl::PointXYZ>& current_cloud,
    const AnchorStateVector& states,
    StructureMigrationVector& migrations) {


  std::unordered_map<int, int> uid_to_idx;
  uid_to_idx.reserve(migrations.size());
  for (int i = 0; i < static_cast<int>(migrations.size()); ++i)
    uid_to_idx[migrations[i].unit_id] = i;

  for (const auto& unit : units_) {

    auto it = uid_to_idx.find(unit.id);
    if (it != uid_to_idx.end() && migrations[it->second].persistent) continue;

    const double exit_score = ComputeExitScore(unit, states);
    if (exit_score < params_.tau_exit) continue;


    const auto candidates = FindEntryCandidates(current_cloud, unit, states);
    for (const auto& cand : candidates) {
      StructureMigration mig;
      mig.unit_id = unit.id;
      mig.exit_score = exit_score;
      if (Validate(unit, cand, mig)) {
        if (it != uid_to_idx.end()) {
          migrations[it->second] = mig;
        } else {
          uid_to_idx[unit.id] = static_cast<int>(migrations.size());
          migrations.push_back(mig);
          it = uid_to_idx.find(unit.id);
        }

        if (mig.confidence > params_.persistence_confidence) {
          migrations[it->second].persistent = true;
        }
        break;
      }
    }
  }
}

}  // namespace deform_monitor_v2
