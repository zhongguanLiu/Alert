/*
 * Author: zgliu@cumt.edu.cn
 * Affiliation: China University of Mining and Technology
 * Open-source release date: 2026-04-20
 */

#include "deform_monitor_v2/core/region_hypothesis_builder.hpp"

#include <limits>
#include <queue>

namespace deform_monitor_v2 {

namespace {

double Clamp01(double value) {
  return std::max(0.0, std::min(1.0, value));
}

Eigen::Vector3d CurrentAnchorPosition(const AnchorReference& anchor,
                                      const AnchorTrackState& state,
                                      const CurrentObservation& observation) {
  if (state.reacquired || observation.reacquired) {
    return observation.matched_center_R;
  }
  if (state.comparable && observation.observable &&
      observation.matched_center_R.squaredNorm() > 1.0e-10) {
    return observation.matched_center_R;
  }
  return anchor.center_R + state.x_mix.block<3, 1>(0, 0);
}

double AnchorWeight(const AnchorTrackState& state) {
  const double base =
      0.35 * Clamp01(state.local_contrast_score / 4.0) +
      0.35 * Clamp01(state.graph_temporal_score) +
      0.30 * Clamp01(state.graph_coherent_score);
  if (state.mode == DetectionMode::DISAPPEARANCE) {
    return std::max(0.10, 0.5 * base + 0.5 * Clamp01(state.disappearance_score));
  }
  return std::max(0.10, 0.5 * base + 0.5 * Clamp01(state.disp_norm / 0.08));
}

Eigen::Vector3d TypeHistogram(const std::vector<size_t>& members,
                              const AnchorReferenceVector& anchors) {
  Eigen::Vector3d hist = Eigen::Vector3d::Zero();
  for (const size_t idx : members) {
    switch (anchors[idx].type) {
      case AnchorType::PLANE:
        hist.x() += 1.0;
        break;
      case AnchorType::EDGE:
        hist.y() += 1.0;
        break;
      case AnchorType::BAND:
        hist.z() += 1.0;
        break;
    }
  }
  if (hist.sum() > 1.0e-9) {
    hist /= hist.sum();
  }
  return hist;
}

}  // namespace

void RegionHypothesisBuilder::SetParams(const StructureCorrespondenceParams& params) {
  params_ = params;
}

void RegionHypothesisBuilder::Build(const AnchorReferenceVector& anchors,
                                    const AnchorStateVector& states,
                                    const CurrentObservationVector& observations,
                                    const MotionClusterVector& /*clusters*/,
                                    RegionHypothesisVector* old_regions,
                                    RegionHypothesisVector* new_regions) const {
  if (!old_regions || !new_regions) {
    return;
  }
  old_regions->clear();
  new_regions->clear();
  if (anchors.size() != states.size() || anchors.size() != observations.size()) {
    return;
  }

  std::vector<size_t> old_candidates;
  std::vector<size_t> new_candidates;
  old_candidates.reserve(anchors.size());
  new_candidates.reserve(anchors.size());

  for (size_t i = 0; i < anchors.size(); ++i) {
    if (IsOldAnchor(states[i], observations[i])) {
      old_candidates.push_back(i);
    }
    if (IsNewAnchor(states[i], observations[i])) {
      new_candidates.push_back(i);
    }
  }

  *old_regions = BuildRegions(anchors, states, observations, old_candidates,
                              RegionHypothesisKind::OLD_REGION);
  *new_regions = BuildRegions(anchors, states, observations, new_candidates,
                              RegionHypothesisKind::NEW_REGION);
}

bool RegionHypothesisBuilder::IsOldAnchor(const AnchorTrackState& state,
                                          const CurrentObservation& observation) const {
  if (state.mode == DetectionMode::DISAPPEARANCE || state.disappearance_candidate) {
    return state.disappearance_score >= params_.old_score_threshold;
  }
  if (observation.gate_state == ObsGateState::OBSERVABLE_MISSING ||
      observation.gate_state == ObsGateState::OBSERVABLE_REPLACED) {
    return observation.disappearance_score >= params_.old_score_threshold;
  }
  // Reacquired anchors: reference position is the "old" location, matched position is
  // the "new" location. Include these so structure_correspondence can form displacement
  // motion vectors for micro-deformation events where the surface stays visible.
  if ((state.reacquired || observation.reacquired) &&
      state.mode == DetectionMode::DISPLACEMENT) {
    return state.disp_norm >= params_.new_disp_threshold;
  }
  return false;
}

bool RegionHypothesisBuilder::IsNewAnchor(const AnchorTrackState& state,
                                          const CurrentObservation& observation) const {
  if (state.mode != DetectionMode::DISPLACEMENT) {
    return false;
  }
  if (!(state.reacquired || observation.reacquired || state.cluster_member ||
        state.graph_candidate || state.persistent_candidate || state.significant)) {
    return false;
  }
  return state.disp_norm >= params_.new_disp_threshold;
}

RegionHypothesisVector RegionHypothesisBuilder::BuildRegions(
    const AnchorReferenceVector& anchors,
    const AnchorStateVector& states,
    const CurrentObservationVector& observations,
    const std::vector<size_t>& candidate_indices,
    RegionHypothesisKind kind) const {
  RegionHypothesisVector regions;
  if (candidate_indices.empty()) {
    return regions;
  }

  std::vector<int> candidate_pos(anchors.size(), -1);
  for (size_t i = 0; i < candidate_indices.size(); ++i) {
    candidate_pos[candidate_indices[i]] = static_cast<int>(i);
  }

  std::vector<uint8_t> visited(candidate_indices.size(), 0);
  int next_id = 0;
  for (size_t seed_pos = 0; seed_pos < candidate_indices.size(); ++seed_pos) {
    if (visited[seed_pos]) {
      continue;
    }
    std::queue<size_t> q;
    std::vector<size_t> members;
    visited[seed_pos] = 1;
    q.push(candidate_indices[seed_pos]);
    while (!q.empty()) {
      const size_t idx = q.front();
      q.pop();
      members.push_back(idx);
      for (const int nb : anchors[idx].neighbor_indices) {
        if (nb < 0) {
          continue;
        }
        const size_t nb_idx = static_cast<size_t>(nb);
        if (nb_idx >= candidate_pos.size()) {
          continue;
        }
        const int pos = candidate_pos[nb_idx];
        if (pos < 0 || visited[static_cast<size_t>(pos)]) {
          continue;
        }
        visited[static_cast<size_t>(pos)] = 1;
        q.push(nb_idx);
      }
    }

    const int min_count =
        kind == RegionHypothesisKind::OLD_REGION ? params_.old_min_anchor_count
                                                 : params_.new_min_anchor_count;
    if (static_cast<int>(members.size()) < min_count) {
      continue;
    }

    RegionHypothesisState region;
    region.id = next_id++;
    region.kind = kind;
    region.anchor_ids.reserve(members.size());
    region.bbox_ref_min_R =
        Eigen::Vector3d::Constant(std::numeric_limits<double>::max());
    region.bbox_ref_max_R =
        Eigen::Vector3d::Constant(-std::numeric_limits<double>::max());
    region.bbox_curr_min_R = region.bbox_ref_min_R;
    region.bbox_curr_max_R = region.bbox_ref_max_R;

    Eigen::Vector3d sum_ref = Eigen::Vector3d::Zero();
    Eigen::Vector3d sum_curr = Eigen::Vector3d::Zero();
    Eigen::Vector3d sum_normal = Eigen::Vector3d::Zero();
    Eigen::Vector3d sum_motion = Eigen::Vector3d::Zero();
    double sum_w = 0.0;
    double sum_disp = 0.0;
    double sum_disappear = 0.0;
    double sum_graph = 0.0;
    double sum_persist = 0.0;
    double sum_conf = 0.0;

    for (const size_t idx : members) {
      region.anchor_ids.push_back(anchors[idx].id);
      const double w = AnchorWeight(states[idx]);
      const Eigen::Vector3d curr_pos =
          CurrentAnchorPosition(anchors[idx], states[idx], observations[idx]);
      region.bbox_ref_min_R = region.bbox_ref_min_R.cwiseMin(anchors[idx].center_R);
      region.bbox_ref_max_R = region.bbox_ref_max_R.cwiseMax(anchors[idx].center_R);
      region.bbox_curr_min_R = region.bbox_curr_min_R.cwiseMin(curr_pos);
      region.bbox_curr_max_R = region.bbox_curr_max_R.cwiseMax(curr_pos);
      sum_ref += w * anchors[idx].center_R;
      sum_curr += w * curr_pos;
      sum_normal += w * anchors[idx].normal_R;
      sum_motion += w * states[idx].x_mix.block<3, 1>(0, 0);
      sum_disp += w * states[idx].disp_norm;
      sum_disappear += w * states[idx].disappearance_score;
      sum_graph += w * std::max(states[idx].graph_temporal_score, states[idx].graph_coherent_score);
      sum_persist += w * static_cast<double>(std::max(states[idx].stable_streak,
                                                     states[idx].disappearance_streak));
      sum_conf += w * Clamp01(0.5 * states[idx].local_contrast_score / 4.0 +
                              0.5 * states[idx].graph_temporal_score);
      sum_w += w;
    }

    if (sum_w < 1.0e-9) {
      continue;
    }
    region.center_ref_R = sum_ref / sum_w;
    region.center_curr_R = sum_curr / sum_w;
    region.mean_normal_R = sum_normal.norm() > 1.0e-9 ? sum_normal.normalized()
                                                       : Eigen::Vector3d::UnitZ();
    region.mean_motion_R = sum_motion / sum_w;
    region.mean_disp_norm = sum_disp / sum_w;
    region.mean_disappearance_score = sum_disappear / sum_w;
    region.mean_graph_score = sum_graph / sum_w;
    region.confidence = Clamp01(sum_conf / sum_w);
    region.time_persistence = sum_persist / sum_w;
    region.type_histogram = TypeHistogram(members, anchors);
    region.significant = region.confidence >= params_.min_confidence;
    regions.push_back(region);
  }

  return regions;
}

}  // namespace deform_monitor_v2
