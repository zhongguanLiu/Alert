/*
 * Author: zgliu@cumt.edu.cn
 * Affiliation: China University of Mining and Technology
 * Open-source release date: 2026-04-20
 */

#include "deform_monitor_v2/core/persistent_risk_region_tracker.hpp"

#include <algorithm>
#include <cmath>

namespace deform_monitor_v2 {

namespace {

struct MatchCandidate {
  size_t region_idx = 0;
  size_t track_idx = 0;
  double cost = 0.0;
};

double Clamp01(double value) {
  return std::max(0.0, std::min(1.0, value));
}

double BBoxVolume(const Eigen::Vector3d& bbox_min, const Eigen::Vector3d& bbox_max) {
  const Eigen::Vector3d extent = (bbox_max - bbox_min).cwiseMax(Eigen::Vector3d::Zero());
  return extent.x() * extent.y() * extent.z();
}

void ExpandBBox(Eigen::Vector3d* bbox_min, Eigen::Vector3d* bbox_max, double pad) {
  const Eigen::Vector3d offset = Eigen::Vector3d::Constant(std::max(0.0, pad));
  *bbox_min -= offset;
  *bbox_max += offset;
}

double BBoxIoU(const Eigen::Vector3d& a_min,
               const Eigen::Vector3d& a_max,
               const Eigen::Vector3d& b_min,
               const Eigen::Vector3d& b_max,
               double pad) {
  Eigen::Vector3d padded_a_min = a_min;
  Eigen::Vector3d padded_a_max = a_max;
  Eigen::Vector3d padded_b_min = b_min;
  Eigen::Vector3d padded_b_max = b_max;
  ExpandBBox(&padded_a_min, &padded_a_max, pad);
  ExpandBBox(&padded_b_min, &padded_b_max, pad);

  const Eigen::Vector3d inter_min = padded_a_min.cwiseMax(padded_b_min);
  const Eigen::Vector3d inter_max = padded_a_max.cwiseMin(padded_b_max);
  const Eigen::Vector3d inter_extent = (inter_max - inter_min).cwiseMax(Eigen::Vector3d::Zero());
  const double inter_volume = inter_extent.x() * inter_extent.y() * inter_extent.z();
  if (inter_volume <= 0.0) {
    return 0.0;
  }
  const double vol_a = BBoxVolume(padded_a_min, padded_a_max);
  const double vol_b = BBoxVolume(padded_b_min, padded_b_max);
  const double denom = vol_a + vol_b - inter_volume;
  if (denom <= 1.0e-12) {
    return 0.0;
  }
  return inter_volume / denom;
}

double BBoxDiagonal(const Eigen::Vector3d& bbox_min, const Eigen::Vector3d& bbox_max) {
  return (bbox_max - bbox_min).norm();
}

Eigen::Vector3d BBoxCenter(const Eigen::Vector3d& bbox_min, const Eigen::Vector3d& bbox_max) {
  return 0.5 * (bbox_min + bbox_max);
}

bool IsPlanarLike(RiskRegionType type) {
  return type == RiskRegionType::DISPLACEMENT_LIKE || type == RiskRegionType::MIXED;
}

bool IsStablePlanarLikeType(RiskRegionType stable_type, RiskRegionType current_type) {
  return IsPlanarLike(stable_type) && IsPlanarLike(current_type);
}

RiskRegionType MergeRegionTypeSummary(RiskRegionType lhs, RiskRegionType rhs) {
  if (lhs == RiskRegionType::NONE) {
    return rhs;
  }
  if (rhs == RiskRegionType::NONE || lhs == rhs) {
    return lhs;
  }
  return RiskRegionType::MIXED;
}

bool TypeCompatible(RiskRegionType track_type, RiskRegionType region_type) {
  if (track_type == RiskRegionType::NONE || region_type == RiskRegionType::NONE) {
    return true;
  }
  if (track_type == region_type) {
    return true;
  }
  return track_type == RiskRegionType::MIXED || region_type == RiskRegionType::MIXED;
}

void UpdateStableRegionType(PersistentRiskTrackState* track, RiskRegionType region_type) {
  if (region_type == RiskRegionType::NONE) {
    return;
  }
  if (track->stable_type_streak <= 0 || track->stable_region_type == RiskRegionType::NONE) {
    track->stable_region_type = region_type;
    track->stable_type_streak = 1;
    return;
  }
  if (track->stable_region_type == region_type ||
      IsStablePlanarLikeType(track->stable_region_type, region_type)) {
    if (IsPlanarLike(region_type)) {
      track->stable_region_type = region_type;
    }
    track->stable_type_streak += 1;
    return;
  }
  track->stable_region_type = region_type;
  track->stable_type_streak = 1;
}

double MatchCost(const PersistentRiskParams& params,
                 const PersistentRiskTrackState& track,
                 const RiskRegionState& region) {
  const double center_distance = (region.center_R - track.last_center_R).norm();
  const double bbox_iou = BBoxIoU(track.union_bbox_min_R,
                                  track.union_bbox_max_R,
                                  region.bbox_min_R,
                                  region.bbox_max_R,
                                  std::max(0.02, 0.10 * params.max_center_distance));
  const double risk_gap = std::abs(region.mean_risk - track.ema_mean_risk);
  const double distance_term = center_distance / std::max(1.0e-6, params.max_center_distance);
  const double iou_term = 1.0 - bbox_iou;
  const double type_term = TypeCompatible(track.region_type, region.type) ? 0.0 : 1.0;
  const double risk_term = risk_gap / std::max(1.0e-6, params.max_risk_gap);
  return 0.55 * distance_term + 0.25 * iou_term + 0.10 * type_term + 0.10 * risk_term;
}

void PushMatchHistory(PersistentRiskTrackState* track, bool matched, int window_size) {
  const int capped_window = std::max(1, window_size);
  if (static_cast<int>(track->match_history.size()) >= capped_window) {
    track->matched_region_count_window -= static_cast<int>(track->match_history.front());
    track->match_history.pop_front();
  }
  track->match_history.push_back(static_cast<uint8_t>(matched ? 1 : 0));
  track->matched_region_count_window += matched ? 1 : 0;
  if (track->matched_region_count_window < 0) {
    track->matched_region_count_window = 0;
  }
}

void AbsorbFragmentIntoTrack(PersistentRiskTrackState* track,
                             const RiskRegionState& region,
                             const ros::Time& stamp) {
  const double fragment_support = static_cast<double>(std::max(1, region.voxel_count));
  const double merged_support = std::max(1.0, track->support_mass) + fragment_support;
  track->union_bbox_min_R = track->union_bbox_min_R.cwiseMin(region.bbox_min_R);
  track->union_bbox_max_R = track->union_bbox_max_R.cwiseMax(region.bbox_max_R);
  track->last_bbox_min_R = track->union_bbox_min_R;
  track->last_bbox_max_R = track->union_bbox_max_R;
  track->last_center_R = BBoxCenter(track->union_bbox_min_R, track->union_bbox_max_R);
  track->ema_mean_risk =
      (track->ema_mean_risk * std::max(1.0, track->support_mass) + region.mean_risk * fragment_support) /
      merged_support;
  track->ema_peak_risk = std::max(track->ema_peak_risk, region.peak_risk);
  track->ema_confidence =
      (track->ema_confidence * std::max(1.0, track->support_mass) + region.confidence * fragment_support) /
      merged_support;
  track->support_mass = merged_support;
  track->ema_voxel_count = merged_support;
  track->accumulated_risk += region.mean_risk * fragment_support;
  track->spatial_span = BBoxDiagonal(track->union_bbox_min_R, track->union_bbox_max_R);
  track->region_type = MergeRegionTypeSummary(track->region_type, region.type);
  track->stable_region_type = MergeRegionTypeSummary(track->stable_region_type, region.type);
  track->last_update = stamp;
}

}  // namespace

void PersistentRiskRegionTracker::SetParams(const PersistentRiskParams& params) {
  params_ = params;
}

void PersistentRiskRegionTracker::Reset() {
  tracks_.clear();
  next_track_id_ = 0;
}

PersistentRiskTrackVector PersistentRiskRegionTracker::Update(const RiskRegionVector& regions,
                                                              const ros::Time& stamp) {
  if (!params_.enable) {
    return PersistentRiskTrackVector();
  }

  const auto matches = MatchRegionsToTracks(regions);
  std::vector<uint8_t> matched_tracks(tracks_.size(), 0);
  std::vector<uint8_t> matched_regions(regions.size(), 0);

  for (const auto& match : matches) {
    const size_t track_idx = match.second;
    const size_t region_idx = match.first;
    if (track_idx >= tracks_.size() || region_idx >= regions.size()) {
      continue;
    }
    UpdateMatchedTrack(tracks_[track_idx], regions[region_idx], stamp);
    matched_tracks[track_idx] = 1;
    matched_regions[region_idx] = 1;
  }

  std::vector<MatchCandidate> fragment_candidates;
  fragment_candidates.reserve(tracks_.size() * regions.size());
  for (size_t region_idx = 0; region_idx < regions.size(); ++region_idx) {
    if (matched_regions[region_idx]) {
      continue;
    }
    const auto& region = regions[region_idx];
    for (size_t track_idx = 0; track_idx < tracks_.size(); ++track_idx) {
      const auto& track = tracks_[track_idx];
      if (!TypeCompatible(track.region_type, region.type)) {
        continue;
      }
      const double center_distance = (region.center_R - track.last_center_R).norm();
      if (center_distance >= params_.max_center_distance) {
        continue;
      }
      const double bbox_iou = BBoxIoU(track.union_bbox_min_R,
                                      track.union_bbox_max_R,
                                      region.bbox_min_R,
                                      region.bbox_max_R,
                                      std::max(0.02, 0.10 * params_.max_center_distance));
      if (bbox_iou <= params_.min_bbox_iou) {
        continue;
      }
      const double risk_gap = std::abs(region.mean_risk - track.ema_mean_risk);
      if (risk_gap > params_.max_risk_gap) {
        continue;
      }
      fragment_candidates.push_back(MatchCandidate{region_idx, track_idx, MatchCost(params_, track, region)});
    }
  }

  std::sort(fragment_candidates.begin(),
            fragment_candidates.end(),
            [](const MatchCandidate& lhs, const MatchCandidate& rhs) {
              if (lhs.cost != rhs.cost) {
                return lhs.cost < rhs.cost;
              }
              if (lhs.track_idx != rhs.track_idx) {
                return lhs.track_idx < rhs.track_idx;
              }
              return lhs.region_idx < rhs.region_idx;
            });

  for (const auto& candidate : fragment_candidates) {
    const size_t region_idx = candidate.region_idx;
    const size_t track_idx = candidate.track_idx;
    if (region_idx >= regions.size() || track_idx >= tracks_.size() || matched_regions[region_idx]) {
      continue;
    }
    auto& track = tracks_[track_idx];
    if (!matched_tracks[track_idx]) {
      UpdateMatchedTrack(track, regions[region_idx], stamp);
      matched_tracks[track_idx] = 1;
    } else {
      AbsorbFragmentIntoTrack(&track, regions[region_idx], stamp);
    }
    matched_regions[region_idx] = 1;
  }

  for (size_t track_idx = 0; track_idx < tracks_.size(); ++track_idx) {
    if (!matched_tracks[track_idx]) {
      continue;
    }
    auto& track = tracks_[track_idx];
    RefreshConfirmationState(&track);
  }

  UpdateUnmatchedTracks(stamp, matched_tracks);

  tracks_.erase(std::remove_if(tracks_.begin(),
                               tracks_.end(),
                               [this](const PersistentRiskTrackState& track) {
                                 return ShouldDeleteTrack(track);
                               }),
                tracks_.end());

  for (size_t i = 0; i < regions.size(); ++i) {
    if (!matched_regions[i]) {
      tracks_.push_back(SpawnNewTrack(regions[i], stamp));
    }
  }

  return tracks_;
}

std::vector<std::pair<size_t, size_t>> PersistentRiskRegionTracker::MatchRegionsToTracks(
    const RiskRegionVector& regions) const {
  std::vector<std::pair<size_t, size_t>> matches;
  if (tracks_.empty() || regions.empty()) {
    return matches;
  }

  std::vector<MatchCandidate> candidates;
  candidates.reserve(tracks_.size() * regions.size());
  for (size_t region_idx = 0; region_idx < regions.size(); ++region_idx) {
    const auto& region = regions[region_idx];
    for (size_t track_idx = 0; track_idx < tracks_.size(); ++track_idx) {
      const auto& track = tracks_[track_idx];
      if (!TypeCompatible(track.region_type, region.type)) {
        continue;
      }
      const double center_distance = (region.center_R - track.last_center_R).norm();
      if (center_distance >= params_.max_center_distance) {
        continue;
      }
      const double bbox_iou = BBoxIoU(track.union_bbox_min_R,
                                      track.union_bbox_max_R,
                                      region.bbox_min_R,
                                      region.bbox_max_R,
                                      std::max(0.02, 0.10 * params_.max_center_distance));
      if (bbox_iou <= params_.min_bbox_iou) {
        continue;
      }
      const double risk_gap = std::abs(region.mean_risk - track.ema_mean_risk);
      if (risk_gap > params_.max_risk_gap) {
        continue;
      }
      candidates.push_back(MatchCandidate{region_idx, track_idx, MatchCost(params_, track, region)});
    }
  }

  std::sort(candidates.begin(), candidates.end(), [](const MatchCandidate& lhs, const MatchCandidate& rhs) {
    if (lhs.cost != rhs.cost) {
      return lhs.cost < rhs.cost;
    }
    if (lhs.track_idx != rhs.track_idx) {
      return lhs.track_idx < rhs.track_idx;
    }
    return lhs.region_idx < rhs.region_idx;
  });

  std::vector<uint8_t> region_used(regions.size(), 0);
  std::vector<uint8_t> track_used(tracks_.size(), 0);
  for (const auto& candidate : candidates) {
    if (region_used[candidate.region_idx] || track_used[candidate.track_idx]) {
      continue;
    }
    region_used[candidate.region_idx] = 1;
    track_used[candidate.track_idx] = 1;
    matches.emplace_back(candidate.region_idx, candidate.track_idx);
  }

  return matches;
}

void PersistentRiskRegionTracker::UpdateMatchedTrack(PersistentRiskTrackState& track,
                                                     const RiskRegionState& region,
                                                     const ros::Time& stamp) {
  const double alpha = Clamp01(params_.ema_alpha);
  const double one_minus_alpha = 1.0 - alpha;
  const double current_voxel_count = static_cast<double>(std::max(1, region.voxel_count));
  const double current_risk_mass = region.mean_risk * current_voxel_count;

  track.region_type = region.type != RiskRegionType::NONE ? region.type : track.region_type;
  track.last_center_R = region.center_R;
  track.last_bbox_min_R = region.bbox_min_R;
  track.last_bbox_max_R = region.bbox_max_R;
  if (track.age_frames <= 0) {
    track.union_bbox_min_R = region.bbox_min_R;
    track.union_bbox_max_R = region.bbox_max_R;
  } else {
    track.union_bbox_min_R = track.union_bbox_min_R.cwiseMin(region.bbox_min_R);
    track.union_bbox_max_R = track.union_bbox_max_R.cwiseMax(region.bbox_max_R);
  }

  track.hit_streak += 1;
  track.miss_streak = 0;
  track.age_frames += 1;
  PushMatchHistory(&track, true, params_.window_size);
  track.prev_support_mass = track.support_mass;
  track.prev_accumulated_risk = track.accumulated_risk;
  UpdateStableRegionType(&track, region.type);
  track.planar_like_streak = IsPlanarLike(region.type) ? track.planar_like_streak + 1 : 0;

  track.ema_mean_risk = alpha * region.mean_risk + one_minus_alpha * track.ema_mean_risk;
  track.ema_peak_risk = alpha * region.peak_risk + one_minus_alpha * track.ema_peak_risk;
  track.ema_confidence = alpha * region.confidence + one_minus_alpha * track.ema_confidence;
  track.ema_voxel_count = alpha * current_voxel_count + one_minus_alpha * track.ema_voxel_count;
  track.support_mass = track.ema_voxel_count;
  track.accumulated_risk += current_risk_mass;
  track.spatial_span = BBoxDiagonal(track.union_bbox_min_R, track.union_bbox_max_R);
  track.last_update = stamp;
  RefreshConfirmationState(&track);
}

void PersistentRiskRegionTracker::UpdateUnmatchedTracks(
    const ros::Time& stamp,
    const std::vector<uint8_t>& matched_tracks) {
  const double miss_decay = std::max(0.70, 1.0 - 0.25 * Clamp01(params_.ema_alpha));
  for (size_t track_idx = 0; track_idx < tracks_.size(); ++track_idx) {
    if (track_idx < matched_tracks.size() && matched_tracks[track_idx]) {
      continue;
    }
    auto& track = tracks_[track_idx];
    track.age_frames += 1;
    track.miss_streak += 1;
    track.hit_streak = 0;
    PushMatchHistory(&track, false, params_.window_size);
    track.ema_mean_risk *= miss_decay;
    track.ema_peak_risk *= miss_decay;
    track.ema_confidence *= miss_decay;
    track.ema_voxel_count *= miss_decay;
    track.support_mass = track.ema_voxel_count;
    track.spatial_span = BBoxDiagonal(track.union_bbox_min_R, track.union_bbox_max_R);
    if (!track.ever_confirmed) {
      track.stable_region_type = RiskRegionType::NONE;
      track.stable_type_streak = 0;
      track.region_type = RiskRegionType::NONE;
    }
    track.planar_like_streak = 0;
    if (track.ever_confirmed && track.miss_streak >= std::max(1, params_.miss_frames_to_fading)) {
      track.state = PersistentRiskState::FADING;
    } else if (!track.ever_confirmed) {
      track.state = PersistentRiskState::CANDIDATE;
    }
    track.last_update = stamp;
  }
}

PersistentRiskTrackState PersistentRiskRegionTracker::SpawnNewTrack(const RiskRegionState& region,
                                                                    const ros::Time& stamp) {
  PersistentRiskTrackState track;
  track.track_id = next_track_id_++;
  track.state = PersistentRiskState::CANDIDATE;
  track.region_type = region.type;
  track.last_center_R = region.center_R;
  track.last_bbox_min_R = region.bbox_min_R;
  track.last_bbox_max_R = region.bbox_max_R;
  track.union_bbox_min_R = region.bbox_min_R;
  track.union_bbox_max_R = region.bbox_max_R;
  track.hit_streak = 1;
  track.miss_streak = 0;
  track.age_frames = 1;
  track.matched_region_count_window = 1;
  track.ever_confirmed = false;
  track.ema_mean_risk = region.mean_risk;
  track.ema_peak_risk = region.peak_risk;
  track.accumulated_risk = region.mean_risk * static_cast<double>(std::max(1, region.voxel_count));
  track.ema_confidence = region.confidence;
  track.ema_voxel_count = static_cast<double>(std::max(1, region.voxel_count));
  track.support_mass = track.ema_voxel_count;
  track.spatial_span = BBoxDiagonal(track.union_bbox_min_R, track.union_bbox_max_R);
  track.stable_region_type = region.type;
  track.stable_type_streak = region.type == RiskRegionType::NONE ? 0 : 1;
  track.planar_like_streak = IsPlanarLike(region.type) ? 1 : 0;
  track.prev_support_mass = 0.0;
  track.prev_accumulated_risk = 0.0;
  track.last_update = stamp;
  track.match_history.clear();
  track.match_history.push_back(1);
  return track;
}

bool PersistentRiskRegionTracker::ShouldConfirmTrack(const PersistentRiskTrackState& track) const {
  const int min_hits_to_confirm = std::max(1, params_.min_hits_to_confirm);
  const int min_hit_streak_to_confirm = std::max(1, params_.min_hit_streak_to_confirm);
  const bool recent_hits = track.matched_region_count_window >= min_hits_to_confirm;
  const bool hit_streak_ready = track.hit_streak >= min_hit_streak_to_confirm;
  const bool signal_ready = track.ema_mean_risk >= params_.min_confirmed_mean_risk &&
                            track.ema_confidence >= params_.min_confirmed_confidence;
  const double support_delta = track.support_mass - track.prev_support_mass;
  const double risk_delta = track.accumulated_risk - track.prev_accumulated_risk;
  const double support_growth_threshold =
      std::max(0.25, 0.15 * std::max(1.0, track.prev_support_mass));
  const double risk_growth_threshold =
      std::max(0.25, 0.10 * std::max(1.0, track.prev_accumulated_risk));
  const bool meaningful_growth = support_delta >= support_growth_threshold ||
                                 risk_delta >= risk_growth_threshold;
  const bool conventional = recent_hits &&
                            hit_streak_ready &&
                            signal_ready &&
                            track.support_mass >= params_.min_confirmed_support_mass;
  const bool stable_planar_type = IsPlanarLike(track.stable_region_type) &&
                                  track.stable_type_streak >= std::max(2, params_.min_hit_streak_to_confirm);
  const bool sparse_planar = params_.allow_sparse_planar_regions &&
                             recent_hits &&
                             hit_streak_ready &&
                             signal_ready &&
                             track.spatial_span >= params_.min_confirmed_span &&
                             stable_planar_type &&
                             meaningful_growth;
  return conventional || sparse_planar;
}

void PersistentRiskRegionTracker::RefreshConfirmationState(PersistentRiskTrackState* track) {
  if (!track) {
    return;
  }
  if (track->ever_confirmed) {
    track->state = PersistentRiskState::CONFIRMED;
    return;
  }
  if (ShouldConfirmTrack(*track)) {
    track->ever_confirmed = true;
    track->state = PersistentRiskState::CONFIRMED;
    return;
  }
  track->state = PersistentRiskState::CANDIDATE;
}

bool PersistentRiskRegionTracker::ShouldDeleteTrack(const PersistentRiskTrackState& track) const {
  if (track.miss_streak >= std::max(1, params_.miss_frames_to_delete)) {
    return true;
  }
  return track.state == PersistentRiskState::FADING &&
         track.ema_mean_risk < params_.fading_risk_floor;
}

}  // namespace deform_monitor_v2
