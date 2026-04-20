/*
 * Author: zgliu@cumt.edu.cn
 * Affiliation: China University of Mining and Technology
 * Open-source release date: 2026-04-20
 */

#include "deform_monitor_v2/core/region_correspondence_solver.hpp"

#include <algorithm>

namespace deform_monitor_v2 {

namespace {

double Clamp01(double value) {
  return std::max(0.0, std::min(1.0, value));
}

double BoxDiag(const Eigen::Vector3d& bmin, const Eigen::Vector3d& bmax) {
  return (bmax - bmin).norm();
}

}  // namespace

void RegionCorrespondenceSolver::SetParams(const StructureCorrespondenceParams& params) {
  params_ = params;
}

double RegionCorrespondenceSolver::PairCost(const RegionHypothesisState& old_region,
                                            const RegionHypothesisState& new_region,
                                            double* distance) const {
  const Eigen::Vector3d delta = new_region.center_curr_R - old_region.center_ref_R;
  const double dist = delta.norm();
  if (distance) {
    *distance = dist;
  }
  if (dist > params_.max_match_distance) {
    return std::numeric_limits<double>::infinity();
  }

  const double old_diag = BoxDiag(old_region.bbox_ref_min_R, old_region.bbox_ref_max_R);
  const double new_diag = BoxDiag(new_region.bbox_curr_min_R, new_region.bbox_curr_max_R);
  const double size_gap = std::abs(old_diag - new_diag);
  if (size_gap > params_.max_size_gap) {
    return std::numeric_limits<double>::infinity();
  }

  const double normal_deg =
      AngleBetweenDeg(old_region.mean_normal_R, new_region.mean_normal_R);
  if (normal_deg > params_.max_normal_deg) {
    return std::numeric_limits<double>::infinity();
  }

  const double type_l1 =
      (old_region.type_histogram - new_region.type_histogram).lpNorm<1>();
  if (type_l1 > params_.max_type_l1) {
    return std::numeric_limits<double>::infinity();
  }

  double motion_term = 1.0;
  if (new_region.mean_motion_R.norm() > 1.0e-6 && delta.norm() > 1.0e-6) {
    const double cos_v =
        new_region.mean_motion_R.normalized().dot(delta.normalized());
    motion_term = 0.5 * (1.0 - Clamp01(0.5 * (cos_v + 1.0)));
  }

  const double persistence_term =
      std::min(1.0,
               std::abs(old_region.time_persistence - new_region.time_persistence) /
                   std::max(1.0, std::max(old_region.time_persistence,
                                          new_region.time_persistence)));

  const double cost =
      params_.weight_dist * (dist / std::max(1.0e-6, params_.max_match_distance)) +
      params_.weight_size * (size_gap / std::max(1.0e-6, params_.max_size_gap)) +
      params_.weight_normal * (normal_deg / std::max(1.0e-6, params_.max_normal_deg)) +
      params_.weight_type * (type_l1 / std::max(1.0e-6, params_.max_type_l1)) +
      params_.weight_motion * motion_term +
      params_.weight_persistence * persistence_term;
  return cost;
}

StructureMotionVector RegionCorrespondenceSolver::Solve(
    const RegionHypothesisVector& old_regions,
    const RegionHypothesisVector& new_regions) const {
  StructureMotionVector motions;
  struct CandidateMatch {
    int old_idx = -1;
    int new_idx = -1;
    double cost = std::numeric_limits<double>::infinity();
    double distance = 0.0;
  };

  std::vector<CandidateMatch> candidates;
  for (size_t oi = 0; oi < old_regions.size(); ++oi) {
    for (size_t ni = 0; ni < new_regions.size(); ++ni) {
      CandidateMatch candidate;
      candidate.old_idx = static_cast<int>(oi);
      candidate.new_idx = static_cast<int>(ni);
      candidate.cost = PairCost(old_regions[oi], new_regions[ni], &candidate.distance);
      if (std::isfinite(candidate.cost) && candidate.cost <= params_.max_match_cost) {
        candidates.push_back(candidate);
      }
    }
  }

  std::sort(candidates.begin(), candidates.end(),
            [](const CandidateMatch& a, const CandidateMatch& b) { return a.cost < b.cost; });
  std::vector<uint8_t> old_used(old_regions.size(), 0);
  std::vector<uint8_t> new_used(new_regions.size(), 0);

  int next_id = 0;
  for (const auto& candidate : candidates) {
    const size_t oi = static_cast<size_t>(candidate.old_idx);
    const size_t ni = static_cast<size_t>(candidate.new_idx);
    if (old_used[oi] || new_used[ni]) {
      continue;
    }
    old_used[oi] = 1;
    new_used[ni] = 1;

    StructureMotionState motion;
    motion.id = next_id++;
    motion.old_region_id = old_regions[oi].id;
    motion.new_region_id = new_regions[ni].id;
    motion.old_center_R = old_regions[oi].center_ref_R;
    motion.new_center_R = new_regions[ni].center_curr_R;
    motion.bbox_old_min_R = old_regions[oi].bbox_ref_min_R;
    motion.bbox_old_max_R = old_regions[oi].bbox_ref_max_R;
    motion.bbox_new_min_R = new_regions[ni].bbox_curr_min_R;
    motion.bbox_new_max_R = new_regions[ni].bbox_curr_max_R;
    motion.motion_R = motion.new_center_R - motion.old_center_R;
    motion.distance = candidate.distance;
    motion.match_cost = candidate.cost;
    motion.support_old = static_cast<int>(old_regions[oi].anchor_ids.size());
    motion.support_new = static_cast<int>(new_regions[ni].anchor_ids.size());
    motion.confidence =
        Clamp01((1.0 - candidate.cost / std::max(1.0e-6, params_.max_match_cost)) *
                std::sqrt(std::max(0.0, old_regions[oi].confidence * new_regions[ni].confidence)));
    motion.type = old_regions[oi].mean_disappearance_score > new_regions[ni].mean_disp_norm
                      ? StructureMotionType::DISAPPEARANCE_LINK
                      : StructureMotionType::DISPLACEMENT_LINK;
    motion.significant =
        motion.confidence >= params_.min_confidence &&
        motion.distance >= params_.min_motion_distance &&
        motion.support_old >= params_.old_min_anchor_count &&
        motion.support_new >= params_.new_min_anchor_count;
    motions.push_back(motion);
  }

  return motions;
}

}  // namespace deform_monitor_v2
