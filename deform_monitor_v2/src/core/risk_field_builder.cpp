/*
 * Author: zgliu@cumt.edu.cn
 * Affiliation: China University of Mining and Technology
 * Open-source release date: 2026-04-20
 */

#include "deform_monitor_v2/core/risk_field_builder.hpp"

#include <cmath>
#include <limits>
#include <queue>
#include <unordered_map>

namespace deform_monitor_v2 {

namespace {

struct SpatialVoxelKey {
  int x = 0;
  int y = 0;
  int z = 0;

  bool operator==(const SpatialVoxelKey& other) const {
    return x == other.x && y == other.y && z == other.z;
  }
};

struct SpatialVoxelKeyHash {
  size_t operator()(const SpatialVoxelKey& key) const {
    size_t h = std::hash<int>()(key.x);
    h ^= std::hash<int>()(key.y) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<int>()(key.z) + 0x9e3779b9 + (h << 6) + (h >> 2);
    return h;
  }
};

struct VoxelAccumulator {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Eigen::Vector3d center_sum = Eigen::Vector3d::Zero();
  int center_count = 0;
  double weight_sum = 0.0;
  double weighted_risk = 0.0;
  double weighted_conf = 0.0;
  double weighted_disp = 0.0;
  double weighted_disappear = 0.0;
  int source_count = 0;
};

double Clamp01(double value) {
  return std::max(0.0, std::min(1.0, value));
}

RiskRegionType ClassifyRegionType(double disp_sum, double disappear_sum) {
  if (disp_sum <= 1.0e-9 && disappear_sum <= 1.0e-9) {
    return RiskRegionType::NONE;
  }
  if (disp_sum > 1.25 * disappear_sum) {
    return RiskRegionType::DISPLACEMENT_LIKE;
  }
  if (disappear_sum > 1.25 * disp_sum) {
    return RiskRegionType::DISAPPEARANCE_LIKE;
  }
  return RiskRegionType::MIXED;
}

}  // namespace

void RiskFieldBuilder::SetParams(const RiskVisualizationParams& params) {
  params_ = params;
}

RiskVoxelVector RiskFieldBuilder::Build(const AnchorReferenceVector& anchors,
                                        const RiskEvidenceVector& evidences) const {
  RiskVoxelVector voxels;
  if (anchors.empty() || evidences.empty()) {
    return voxels;
  }

  const double voxel_size = std::max(0.01, params_.voxel_size);
  const double kernel_sigma = std::max(0.02, params_.kernel_sigma);
  const double kernel_radius = std::max(voxel_size, params_.kernel_radius);
  const int kernel_layers =
      std::max(1, static_cast<int>(std::ceil(kernel_radius / voxel_size)));

  std::unordered_map<SpatialVoxelKey, VoxelAccumulator, SpatialVoxelKeyHash> accum;
  accum.reserve(anchors.size() * 2);
  for (const auto& anchor : anchors) {
    SpatialVoxelKey key;
    key.x = static_cast<int>(std::floor(anchor.center_R.x() / voxel_size));
    key.y = static_cast<int>(std::floor(anchor.center_R.y() / voxel_size));
    key.z = static_cast<int>(std::floor(anchor.center_R.z() / voxel_size));
    auto& cell = accum[key];
    cell.center_sum += anchor.center_R;
    ++cell.center_count;
  }

  for (const auto& evidence : evidences) {
    if (!evidence.active) {
      continue;
    }
    if (evidence.confidence < params_.min_confidence ||
        evidence.risk_score < params_.min_risk_score) {
      continue;
    }
    const bool disappear_like =
        evidence.disappearance_score > evidence.displacement_score;
    if (!disappear_like &&
        evidence.graph_neighbor_count < std::max(0, params_.min_graph_neighbors)) {
      continue;
    }

    SpatialVoxelKey center_key;
    center_key.x = static_cast<int>(std::floor(evidence.position_R.x() / voxel_size));
    center_key.y = static_cast<int>(std::floor(evidence.position_R.y() / voxel_size));
    center_key.z = static_cast<int>(std::floor(evidence.position_R.z() / voxel_size));
    for (int dx = -kernel_layers; dx <= kernel_layers; ++dx) {
      for (int dy = -kernel_layers; dy <= kernel_layers; ++dy) {
        for (int dz = -kernel_layers; dz <= kernel_layers; ++dz) {
          SpatialVoxelKey key{center_key.x + dx, center_key.y + dy, center_key.z + dz};
          auto it = accum.find(key);
          if (it == accum.end() || it->second.center_count <= 0) {
            continue;
          }
          const Eigen::Vector3d voxel_center =
              it->second.center_sum / static_cast<double>(it->second.center_count);
          const double dist = (voxel_center - evidence.position_R).norm();
          if (dist > kernel_radius) {
            continue;
          }
          const double w = std::exp(-(dist * dist) / (2.0 * kernel_sigma * kernel_sigma));
          auto& cell = it->second;
          cell.weight_sum += w;
          cell.weighted_risk += w * evidence.risk_score;
          cell.weighted_conf += w * evidence.confidence;
          cell.weighted_disp += w * evidence.displacement_score;
          cell.weighted_disappear += w * evidence.disappearance_score;
          ++cell.source_count;
        }
      }
    }
  }

  voxels.reserve(accum.size());
  for (const auto& kv : accum) {
    const auto& cell = kv.second;
    if (cell.center_count <= 0 || cell.weight_sum <= 1.0e-9) {
      continue;
    }
    RiskVoxelState voxel;
    voxel.center_R = cell.center_sum / static_cast<double>(cell.center_count);
    voxel.risk_score = Clamp01(cell.weighted_risk / cell.weight_sum);
    voxel.confidence = Clamp01(cell.weighted_conf / cell.weight_sum);
    voxel.displacement_component = Clamp01(cell.weighted_disp / cell.weight_sum);
    voxel.disappearance_component = Clamp01(cell.weighted_disappear / cell.weight_sum);
    voxel.source_count = cell.source_count;
    const bool enough_sources =
        voxel.source_count >= std::max(2, params_.min_graph_neighbors);
    const bool disappear_support =
        voxel.disappearance_component > voxel.displacement_component &&
        voxel.source_count >= 1;
    voxel.significant =
        voxel.risk_score >= params_.min_voxel_risk &&
        voxel.confidence >= params_.min_confidence &&
        (enough_sources || disappear_support);
    if (voxel.risk_score > 1.0e-3) {
      voxels.push_back(voxel);
    }
  }

  return voxels;
}

RiskRegionVector RiskFieldBuilder::ExtractRegions(const RiskVoxelVector& voxels) const {
  RiskRegionVector regions;
  if (voxels.empty()) {
    return regions;
  }

  const double voxel_size = std::max(0.01, params_.voxel_size);
  std::unordered_map<SpatialVoxelKey, size_t, SpatialVoxelKeyHash> voxel_to_index;
  voxel_to_index.reserve(voxels.size());
  for (size_t i = 0; i < voxels.size(); ++i) {
    if (!voxels[i].significant) {
      continue;
    }
    SpatialVoxelKey key;
    key.x = static_cast<int>(std::floor(voxels[i].center_R.x() / voxel_size));
    key.y = static_cast<int>(std::floor(voxels[i].center_R.y() / voxel_size));
    key.z = static_cast<int>(std::floor(voxels[i].center_R.z() / voxel_size));
    voxel_to_index[key] = i;
  }
  if (voxel_to_index.empty()) {
    return regions;
  }

  std::vector<uint8_t> visited(voxels.size(), 0);
  int next_id = 0;
  for (const auto& kv : voxel_to_index) {
    const size_t seed_idx = kv.second;
    if (visited[seed_idx]) {
      continue;
    }
    std::queue<size_t> q;
    std::vector<size_t> region_indices;
    q.push(seed_idx);
    visited[seed_idx] = 1;
    while (!q.empty()) {
      const size_t idx = q.front();
      q.pop();
      region_indices.push_back(idx);
      SpatialVoxelKey center_key;
      center_key.x = static_cast<int>(std::floor(voxels[idx].center_R.x() / voxel_size));
      center_key.y = static_cast<int>(std::floor(voxels[idx].center_R.y() / voxel_size));
      center_key.z = static_cast<int>(std::floor(voxels[idx].center_R.z() / voxel_size));
      for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
          for (int dz = -1; dz <= 1; ++dz) {
            if (dx == 0 && dy == 0 && dz == 0) {
              continue;
            }
            SpatialVoxelKey key{center_key.x + dx, center_key.y + dy, center_key.z + dz};
            const auto it = voxel_to_index.find(key);
            if (it == voxel_to_index.end() || visited[it->second]) {
              continue;
            }
            visited[it->second] = 1;
            q.push(it->second);
          }
        }
      }
    }

    if (static_cast<int>(region_indices.size()) < std::max(1, params_.min_region_voxels)) {
      continue;
    }

    RiskRegionState region;
    region.id = next_id++;
    region.bbox_min_R = Eigen::Vector3d::Constant(std::numeric_limits<double>::max());
    region.bbox_max_R = Eigen::Vector3d::Constant(-std::numeric_limits<double>::max());
    Eigen::Vector3d center_sum = Eigen::Vector3d::Zero();
    double conf_sum = 0.0;
    double risk_sum = 0.0;
    double peak_risk = 0.0;
    double disp_sum = 0.0;
    double disappear_sum = 0.0;
    for (const size_t idx : region_indices) {
      const auto& voxel = voxels[idx];
      region.bbox_min_R = region.bbox_min_R.cwiseMin(voxel.center_R);
      region.bbox_max_R = region.bbox_max_R.cwiseMax(voxel.center_R);
      center_sum += voxel.center_R;
      conf_sum += voxel.confidence;
      risk_sum += voxel.risk_score;
      peak_risk = std::max(peak_risk, voxel.risk_score);
      disp_sum += voxel.displacement_component;
      disappear_sum += voxel.disappearance_component;
    }
    region.voxel_count = static_cast<int>(region_indices.size());
    region.center_R = center_sum / std::max(1.0, static_cast<double>(region.voxel_count));
    region.confidence = Clamp01(conf_sum / std::max(1.0, static_cast<double>(region.voxel_count)));
    region.mean_risk = Clamp01(risk_sum / std::max(1.0, static_cast<double>(region.voxel_count)));
    region.peak_risk = Clamp01(peak_risk);
    region.type = ClassifyRegionType(disp_sum, disappear_sum);
    region.significant =
        region.voxel_count >= std::max(1, params_.min_region_voxels) &&
        region.mean_risk >= params_.min_region_mean_risk;
    regions.push_back(region);
  }

  return regions;
}

}  // namespace deform_monitor_v2
