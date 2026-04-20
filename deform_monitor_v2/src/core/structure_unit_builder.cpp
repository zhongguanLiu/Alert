/*
 * Author: zgliu@cumt.edu.cn
 * Affiliation: China University of Mining and Technology
 * Open-source release date: 2026-04-20
 */

#include "deform_monitor_v2/structure_unit_builder.hpp"
#include <unordered_map>
#include <numeric>
#include <cmath>

namespace deform_monitor_v2 {

StructureUnitBuilder::StructureUnitBuilder(const StructureUnitParams& params)
    : params_(params) {}

size_t StructureUnitBuilder::VoxelKeyHash::operator()(const VoxelKey& k) const {
  size_t h = 0;
  h ^= std::hash<int>{}(std::get<0>(k)) + 0x9e3779b9 + (h << 6) + (h >> 2);
  h ^= std::hash<int>{}(std::get<1>(k)) + 0x9e3779b9 + (h << 6) + (h >> 2);
  h ^= std::hash<int>{}(std::get<2>(k)) + 0x9e3779b9 + (h << 6) + (h >> 2);
  return h;
}

bool StructureUnitBuilder::ShouldConnect(const AnchorReference& a,
                                          const AnchorReference& b,
                                          RegionEdgeType& out_type) const {

  double dist = (a.center_R - b.center_R).norm();
  if (dist > params_.region_radius) return false;


  double cos_n = std::abs(a.normal_R.dot(b.normal_R));
  double thresh_n = std::cos(params_.region_normal_deg * M_PI / 180.0);
  if (cos_n < thresh_n) return false;


  if (a.type == AnchorType::PLANE && b.type == AnchorType::PLANE) {
    out_type = RegionEdgeType::COPLANAR;
    return true;
  }
  if (a.type == AnchorType::EDGE && b.type == AnchorType::EDGE) {
    double cos_e = std::abs(a.edge_normal_R.dot(b.edge_normal_R));
    double thresh_e = std::cos(params_.region_edge_dir_deg * M_PI / 180.0);
    if (cos_e < thresh_e) return false;
    out_type = RegionEdgeType::COPLANAR;
    return true;
  }
  if ((a.type == AnchorType::PLANE && b.type == AnchorType::EDGE) ||
      (a.type == AnchorType::EDGE && b.type == AnchorType::PLANE)) {
    out_type = RegionEdgeType::ADJACENT;
    return true;
  }

  return false;
}

StructureUnitVector StructureUnitBuilder::Build(
    const AnchorReferenceVector& anchors) {
  if (anchors.empty()) return {};

  const int N = static_cast<int>(anchors.size());
  const double inv_hash = 1.0 / params_.region_spatial_hash;


  std::unordered_map<VoxelKey, std::vector<int>, VoxelKeyHash> voxel_map;
  voxel_map.reserve(N * 2);
  for (int i = 0; i < N; ++i) {
    const auto& c = anchors[i].center_R;
    int ix = static_cast<int>(std::floor(c.x() * inv_hash));
    int iy = static_cast<int>(std::floor(c.y() * inv_hash));
    int iz = static_cast<int>(std::floor(c.z() * inv_hash));
    voxel_map[{ix, iy, iz}].push_back(i);
  }


  std::vector<int> parent(N);
  std::iota(parent.begin(), parent.end(), 0);
  std::function<int(int)> find = [&](int x) -> int {
    return parent[x] == x ? x : (parent[x] = find(parent[x]));
  };
  auto unite = [&](int a, int b) {
    a = find(a); b = find(b);
    if (a != b) parent[a] = b;
  };


  using EdgeKey = std::pair<int,int>;
  struct EdgeKeyHash {
    size_t operator()(const EdgeKey& k) const {
      return std::hash<int64_t>{}((int64_t)k.first << 32 | (uint32_t)k.second);
    }
  };
  std::unordered_map<EdgeKey, RegionEdge, EdgeKeyHash> all_edges;

  const int radius_voxels = static_cast<int>(
      std::ceil(params_.region_radius / params_.region_spatial_hash)) + 1;

  for (int i = 0; i < N; ++i) {
    const auto& c = anchors[i].center_R;
    int ix = static_cast<int>(std::floor(c.x() * inv_hash));
    int iy = static_cast<int>(std::floor(c.y() * inv_hash));
    int iz = static_cast<int>(std::floor(c.z() * inv_hash));

    for (int dx = -radius_voxels; dx <= radius_voxels; ++dx)
    for (int dy = -radius_voxels; dy <= radius_voxels; ++dy)
    for (int dz = -radius_voxels; dz <= radius_voxels; ++dz) {
      auto it = voxel_map.find({ix+dx, iy+dy, iz+dz});
      if (it == voxel_map.end()) continue;
      for (int j : it->second) {
        if (j <= i) continue;
        RegionEdge edge;
        RegionEdgeType etype = RegionEdgeType::COPLANAR;
        if (!ShouldConnect(anchors[i], anchors[j], etype)) continue;
        edge.anchor_id_a = anchors[i].id;
        edge.anchor_id_b = anchors[j].id;
        edge.dist_ref = (anchors[i].center_R - anchors[j].center_R).norm();
        edge.normal_cos_ref = anchors[i].normal_R.dot(anchors[j].normal_R);
        edge.edge_type = etype;
        all_edges[{i, j}] = edge;
        unite(i, j);
      }
    }
  }


  std::unordered_map<int, std::vector<int>> components;
  for (int i = 0; i < N; ++i) {
    components[find(i)].push_back(i);
  }


  StructureUnitVector units;
  int uid = 0;
  for (auto& kv : components) {
    auto& members = kv.second;
    if (static_cast<int>(members.size()) < params_.region_min_members) continue;

    StructureUnit unit;
    unit.id = uid++;

    for (int mi : members) {
      unit.member_ids.push_back(anchors[mi].id);

      for (int mj : members) {
        if (mj <= mi) continue;
        auto it = all_edges.find({mi, mj});
        if (it != all_edges.end()) unit.edge_set.push_back(it->second);
      }

      for (const auto& pt : anchors[mi].support_points_R)
        unit.ref_pointcloud.push_back(pt);
    }


    Eigen::Vector3d wc = Eigen::Vector3d::Zero();
    Eigen::Vector3d wn = Eigen::Vector3d::Zero();
    Eigen::Vector3d bbox_min = Eigen::Vector3d::Constant(1e9);
    Eigen::Vector3d bbox_max = Eigen::Vector3d::Constant(-1e9);
    double w_sum = 0.0;
    for (int idx : members) {
      double w = anchors[idx].ref_quality;
      wc += w * anchors[idx].center_R;
      wn += w * anchors[idx].normal_R;
      w_sum += w;
      bbox_min = bbox_min.cwiseMin(anchors[idx].center_R);
      bbox_max = bbox_max.cwiseMax(anchors[idx].center_R);
    }
    unit.ref_centroid_R = (w_sum > 1e-9) ? wc / w_sum : wc / members.size();
    unit.ref_normal_R = (wn.norm() > 1e-9) ? wn.normalized() : Eigen::Vector3d::UnitZ();
    unit.ref_bbox_min_R = bbox_min;
    unit.ref_bbox_max_R = bbox_max;
    unit.mean_ref_quality = (w_sum > 0) ? w_sum / members.size() : 0.0;

    units.push_back(std::move(unit));
  }

  return units;
}

}  // namespace deform_monitor_v2
