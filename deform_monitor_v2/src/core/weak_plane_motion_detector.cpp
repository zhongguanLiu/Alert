#include "deform_monitor_v2/core/weak_plane_motion_detector.hpp"

#include "deform_monitor_v2/core/observable_subspace.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <unordered_map>

namespace deform_monitor_v2 {

namespace {

struct DisjointSet {
  explicit DisjointSet(size_t count) : parent(count), rank(count, 0) {
    std::iota(parent.begin(), parent.end(), 0);
  }

  size_t Find(size_t value) {
    if (parent[value] != value) {
      parent[value] = Find(parent[value]);
    }
    return parent[value];
  }

  void Unite(size_t left, size_t right) {
    left = Find(left);
    right = Find(right);
    if (left == right) {
      return;
    }
    if (rank[left] < rank[right]) {
      std::swap(left, right);
    }
    parent[right] = left;
    if (rank[left] == rank[right]) {
      ++rank[left];
    }
  }

  std::vector<size_t> parent;
  std::vector<int> rank;
};

double DirectionCosine(const Eigen::Vector3d& left, const Eigen::Vector3d& right) {
  const double denominator = left.norm() * right.norm();
  if (denominator <= 1.0e-12) {
    return -1.0;
  }
  return left.dot(right) / denominator;
}

bool CurrentEligible(const AnchorReference& anchor,
                     const AnchorTrackState& state,
                     const WeakPlaneMotionParams& params) {
  const bool supported_type =
      anchor.type == AnchorType::PLANE || params.enable_mixed_types;
  const double amplitude = anchor.type == AnchorType::PLANE
                               ? state.disp_normal
                               : state.disp_norm;
  return supported_type && state.gate_state == ObsGateState::OBSERVABLE_MATCHED &&
         state.observable && state.comparable && !state.reacquired && state.dof_obs > 0 &&
         anchor.ref_quality >= params.min_ref_quality &&
         anchor.covariance_quality >= params.min_covariance_quality &&
         anchor.type_stability >= params.min_type_stability &&
         amplitude >= params.min_anchor_disp &&
         state.chi2_stat >= params.min_mean_chi2;
}

bool StableExteriorBackground(const AnchorReference& anchor,
                              const AnchorTrackState& state,
                              const WeakPlaneMotionParams& params) {
  const double amplitude = anchor.type == AnchorType::PLANE
                               ? state.disp_normal
                               : state.disp_norm;
  return state.gate_state == ObsGateState::OBSERVABLE_MATCHED &&
         state.observable && state.comparable && !state.reacquired && state.dof_obs > 0 &&
         anchor.ref_quality >= params.min_ref_quality &&
         anchor.covariance_quality >= params.min_covariance_quality &&
         anchor.type_stability >= params.min_type_stability &&
         amplitude < params.min_anchor_disp;
}

Eigen::Vector3d ObservableDisplacement(const AnchorReference& anchor,
                                       const AnchorTrackState& state) {
  return ProjectObservableVector(state.x_mix.block<3, 1>(0, 0),
                                 BuildObservableSubspace(anchor),
                                 state.dof_obs);
}

}  // namespace

void WeakPlaneMotionDetector::SetParams(const WeakPlaneMotionParams& params) {
  params_ = params;
  ResetTracks();
}

void WeakPlaneMotionDetector::ResetTracks() {
  component_tracks_.clear();
  next_component_id_ = 1;
  reference_epoch_ = std::numeric_limits<uint32_t>::max();
  anchor_count_ = 0;
}

void WeakPlaneMotionDetector::Update(const AnchorReferenceVector& anchors,
                                     AnchorStateVector* states) {
  if (!states || states->size() != anchors.size()) {
    return;
  }

  std::vector<uint8_t> qualified(anchors.size(), 0);
  std::vector<uint8_t> qualified_current(anchors.size(), 0);
  for (auto& state : *states) {
    state.weak_plane_candidate = false;
    state.weak_plane_group_size = 0;
    state.weak_plane_current_support = 0;
    state.weak_plane_temporal_frame_support = 0;
    state.weak_plane_group_disp = 0.0;
    state.weak_plane_mean_chi2 = 0.0;
    state.weak_plane_direction_consistency = 0.0;
    state.weak_plane_group_residual = 0.0;
    state.weak_plane_component_id = -1;
    state.weak_plane_exterior_background_support = 0;
    state.weak_plane_mixed_type_support = 0;
  }

  if (!params_.enable || anchors.empty()) {
    ResetTracks();
    for (auto& state : *states) {
      state.weak_plane_streak = 0;
      state.weak_plane_cached_valid = false;
      state.weak_plane_cached_age_frames = 0;
      state.weak_plane_cached_displacement_R.setZero();
      state.weak_plane_cached_chi2 = 0.0;
    }
    return;
  }

  const uint32_t current_reference_epoch = anchors.front().reference_epoch;
  if (anchor_count_ != anchors.size() || reference_epoch_ != current_reference_epoch) {
    component_tracks_.clear();
    next_component_id_ = 1;
    anchor_count_ = anchors.size();
    reference_epoch_ = current_reference_epoch;
  }

  std::vector<size_t> eligible;
  std::vector<int> eligible_position(anchors.size(), -1);
  std::vector<uint8_t> fresh_eligible(anchors.size(), 0);
  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> displacement(
      anchors.size(), Eigen::Vector3d::Zero());
  for (size_t index = 0; index < anchors.size(); ++index) {
    auto& state = (*states)[index];
    const bool supported_type =
        anchors[index].type == AnchorType::PLANE || params_.enable_mixed_types;
    if (!supported_type || state.reacquired) {
      state.weak_plane_cached_valid = false;
      state.weak_plane_cached_age_frames = 0;
      state.weak_plane_cached_displacement_R.setZero();
      state.weak_plane_cached_chi2 = 0.0;
      continue;
    }
    if (CurrentEligible(anchors[index], state, params_)) {
      fresh_eligible[index] = 1;
      state.weak_plane_cached_valid = true;
      state.weak_plane_cached_age_frames = 0;
      state.weak_plane_cached_displacement_R =
          ObservableDisplacement(anchors[index], state);
      state.weak_plane_cached_chi2 = state.chi2_stat;
    } else if (state.weak_plane_cached_valid) {
      state.weak_plane_cached_age_frames += 1;
      if (state.weak_plane_cached_age_frames >=
          std::max(1, params_.temporal_window_frames)) {
        state.weak_plane_cached_valid = false;
        state.weak_plane_cached_displacement_R.setZero();
        state.weak_plane_cached_chi2 = 0.0;
      }
    }
    if (!state.weak_plane_cached_valid) {
      continue;
    }
    eligible_position[index] = static_cast<int>(eligible.size());
    eligible.push_back(index);
    displacement[index] = state.weak_plane_cached_displacement_R;
  }

  DisjointSet groups(eligible.size());
  for (size_t position = 0; position < eligible.size(); ++position) {
    const size_t index = eligible[position];
    for (const int raw_neighbor : anchors[index].neighbor_indices) {
      if (raw_neighbor < 0 || static_cast<size_t>(raw_neighbor) >= anchors.size()) {
        continue;
      }
      const size_t neighbor = static_cast<size_t>(raw_neighbor);
      const int neighbor_position = eligible_position[neighbor];
      if (neighbor_position < 0 || static_cast<size_t>(neighbor_position) <= position) {
        continue;
      }
      if ((anchors[index].center_R - anchors[neighbor].center_R).norm() > params_.radius) {
        continue;
      }
      if (anchors[index].type == AnchorType::PLANE &&
          anchors[neighbor].type == AnchorType::PLANE &&
          AngleBetweenDeg(anchors[index].normal_R, anchors[neighbor].normal_R) >
              params_.max_normal_deg) {
        continue;
      }
      if (DirectionCosine(displacement[index], displacement[neighbor]) <
          params_.min_direction_consistency) {
        continue;
      }
      groups.Unite(position, static_cast<size_t>(neighbor_position));
    }
  }

  std::unordered_map<size_t, std::vector<size_t>> components;
  for (size_t position = 0; position < eligible.size(); ++position) {
    components[groups.Find(position)].push_back(eligible[position]);
  }

  struct QualifiedComponent {
    std::vector<size_t> indices;
    Eigen::Vector3d center_R = Eigen::Vector3d::Zero();
    Eigen::Vector3d mean_displacement_R = Eigen::Vector3d::Zero();
    int current_support = 0;
    int temporal_frame_support = 0;
    int exterior_background_support = 0;
    int mixed_type_support = 0;
  };
  std::vector<std::vector<size_t>> ordered_components;
  ordered_components.reserve(components.size());
  for (auto& component_entry : components) {
    ordered_components.push_back(std::move(component_entry.second));
  }
  std::sort(ordered_components.begin(), ordered_components.end(),
            [](const std::vector<size_t>& left, const std::vector<size_t>& right) {
              return left.front() < right.front();
            });

  std::vector<QualifiedComponent> qualified_components;
  for (const auto& component : ordered_components) {
    if (component.size() < static_cast<size_t>(std::max(1, params_.min_support))) {
      continue;
    }

    Eigen::Vector3d mean = Eigen::Vector3d::Zero();
    Eigen::Vector3d center = Eigen::Vector3d::Zero();
    double mean_chi2 = 0.0;
    double magnitude_sum = 0.0;
    int current_support = 0;
    std::vector<uint8_t> temporal_ages(
        static_cast<size_t>(std::max(1, params_.temporal_window_frames)), 0);
    int mixed_type_support = 0;
    for (const size_t index : component) {
      mean += displacement[index];
      center += anchors[index].center_R;
      mean_chi2 += (*states)[index].weak_plane_cached_chi2;
      magnitude_sum += displacement[index].norm();
      const int age = std::max(0, (*states)[index].weak_plane_cached_age_frames);
      if (age == 0) {
        ++current_support;
      }
      if (age < static_cast<int>(temporal_ages.size())) {
        temporal_ages[static_cast<size_t>(age)] = 1;
      }
      if (anchors[index].type != AnchorType::PLANE) {
        ++mixed_type_support;
      }
    }
    mean /= static_cast<double>(component.size());
    center /= static_cast<double>(component.size());
    mean_chi2 /= static_cast<double>(component.size());

    double squared_residual_sum = 0.0;
    for (const size_t index : component) {
      squared_residual_sum += (displacement[index] - mean).squaredNorm();
    }
    const double group_residual =
        std::sqrt(squared_residual_sum / static_cast<double>(component.size()));
    const double direction_consistency =
        mean.norm() * static_cast<double>(component.size()) /
        std::max(1.0e-12, magnitude_sum);
    const int temporal_frame_support = static_cast<int>(
        std::count(temporal_ages.begin(), temporal_ages.end(), uint8_t{1}));
    const int required_temporal_frames =
        current_support >= std::max(1, params_.min_support)
            ? 1
            : std::max(1, params_.min_temporal_frames);
    const bool group_qualified =
        current_support >= std::max(1, params_.min_current_support) &&
        temporal_frame_support >= required_temporal_frames &&
        mean.norm() >= params_.min_group_disp &&
        mean_chi2 >= params_.min_mean_chi2 &&
        direction_consistency >= params_.min_direction_consistency &&
        group_residual <= params_.max_group_residual;

    std::vector<uint8_t> in_component(anchors.size(), 0);
    std::vector<uint8_t> exterior_seen(anchors.size(), 0);
    for (const size_t index : component) {
      in_component[index] = 1;
    }
    int exterior_background_support = 0;
    for (const size_t index : component) {
      for (const int raw_neighbor : anchors[index].neighbor_indices) {
        if (raw_neighbor < 0 || static_cast<size_t>(raw_neighbor) >= anchors.size()) {
          continue;
        }
        const size_t neighbor = static_cast<size_t>(raw_neighbor);
        if (in_component[neighbor] || exterior_seen[neighbor]) {
          continue;
        }
        if ((anchors[index].center_R - anchors[neighbor].center_R).norm() >
            params_.radius) {
          continue;
        }
        if (!StableExteriorBackground(anchors[neighbor], (*states)[neighbor], params_)) {
          continue;
        }
        exterior_seen[neighbor] = 1;
        ++exterior_background_support;
      }
    }

    for (const size_t index : component) {
      auto& state = (*states)[index];
      state.weak_plane_group_size = static_cast<int>(component.size());
      state.weak_plane_current_support = current_support;
      state.weak_plane_temporal_frame_support = temporal_frame_support;
      state.weak_plane_group_disp = mean.norm();
      state.weak_plane_mean_chi2 = mean_chi2;
      state.weak_plane_direction_consistency = direction_consistency;
      state.weak_plane_group_residual = group_residual;
      state.weak_plane_exterior_background_support = exterior_background_support;
      state.weak_plane_mixed_type_support = mixed_type_support;
    }
    if (group_qualified) {
      QualifiedComponent qualified_component;
      qualified_component.indices = component;
      qualified_component.center_R = center;
      qualified_component.mean_displacement_R = mean;
      qualified_component.current_support = current_support;
      qualified_component.temporal_frame_support = temporal_frame_support;
      qualified_component.exterior_background_support = exterior_background_support;
      qualified_component.mixed_type_support = mixed_type_support;
      qualified_components.push_back(std::move(qualified_component));
    }
  }

  std::vector<uint8_t> track_used(component_tracks_.size(), 0);
  std::vector<ComponentTrack, Eigen::aligned_allocator<ComponentTrack>> next_tracks;
  next_tracks.reserve(component_tracks_.size() + qualified_components.size());
  for (const auto& component : qualified_components) {
    int best_track = -1;
    double best_distance = std::numeric_limits<double>::infinity();
    for (size_t track_index = 0; track_index < component_tracks_.size(); ++track_index) {
      if (track_used[track_index]) {
        continue;
      }
      const auto& track = component_tracks_[track_index];
      const double distance = (component.center_R - track.center_R).norm();
      if (distance > std::max(params_.radius, params_.component_match_radius) ||
          DirectionCosine(component.mean_displacement_R, track.displacement_R) <
              params_.component_match_direction_cos) {
        continue;
      }
      if (distance < best_distance) {
        best_distance = distance;
        best_track = static_cast<int>(track_index);
      }
    }

    ComponentTrack track;
    if (best_track >= 0) {
      track = component_tracks_[static_cast<size_t>(best_track)];
      track_used[static_cast<size_t>(best_track)] = 1;
      track.streak += 1;
    } else {
      track.id = next_component_id_++;
      track.streak = 1;
    }
    track.center_R = component.center_R;
    track.displacement_R = component.mean_displacement_R;
    track.missed_frames = 0;
    next_tracks.push_back(track);

    for (const size_t index : component.indices) {
      auto& state = (*states)[index];
      qualified[index] = 1;
      state.weak_plane_component_id = track.id;
      state.weak_plane_streak = track.streak;
      if (!fresh_eligible[index]) {
        continue;
      }
      const bool exterior_gate_passed =
          anchors[index].type == AnchorType::PLANE ||
          !params_.require_exterior_background_for_non_plane ||
          component.exterior_background_support >=
              std::max(1, params_.min_exterior_background_support);
      if (exterior_gate_passed) {
        qualified_current[index] = 1;
      }
    }
  }

  for (size_t track_index = 0; track_index < component_tracks_.size(); ++track_index) {
    if (track_used[track_index]) {
      continue;
    }
    ComponentTrack track = component_tracks_[track_index];
    ++track.missed_frames;
    track.streak = std::max(0, track.streak - std::max(1, params_.streak_decay));
    if (track.missed_frames <= std::max(0, params_.component_max_missed_frames) &&
        track.streak > 0) {
      next_tracks.push_back(track);
    }
  }
  component_tracks_ = std::move(next_tracks);

  for (size_t index = 0; index < states->size(); ++index) {
    auto& state = (*states)[index];
    if (!qualified[index]) {
      state.weak_plane_streak = std::max(
          0, state.weak_plane_streak - std::max(1, params_.streak_decay));
    }
    state.weak_plane_candidate =
        qualified_current[index] &&
        state.weak_plane_streak >= std::max(1, params_.min_streak);
  }
}

}  // namespace deform_monitor_v2
