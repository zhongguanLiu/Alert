/*
 * Author: zgliu@cumt.edu.cn
 * Affiliation: China University of Mining and Technology
 * Open-source release date: 2026-04-20
 */

#ifndef DEFORM_MONITOR_V2_PERSISTENT_RISK_REGION_TRACKER_HPP
#define DEFORM_MONITOR_V2_PERSISTENT_RISK_REGION_TRACKER_HPP

#include "deform_monitor_v2/data_types.hpp"

namespace deform_monitor_v2 {

class PersistentRiskRegionTracker {
public:
  void SetParams(const PersistentRiskParams& params);
  void Reset();
  PersistentRiskTrackVector Update(const RiskRegionVector& regions, const ros::Time& stamp);

private:
  std::vector<std::pair<size_t, size_t>> MatchRegionsToTracks(const RiskRegionVector& regions) const;
  void UpdateMatchedTrack(PersistentRiskTrackState& track,
                          const RiskRegionState& region,
                          const ros::Time& stamp);
  void UpdateUnmatchedTracks(const ros::Time& stamp,
                             const std::vector<uint8_t>& matched_tracks);
  PersistentRiskTrackState SpawnNewTrack(const RiskRegionState& region,
                                         const ros::Time& stamp);
  bool ShouldConfirmTrack(const PersistentRiskTrackState& track) const;
  bool ShouldDeleteTrack(const PersistentRiskTrackState& track) const;
  void RefreshConfirmationState(PersistentRiskTrackState* track);

  PersistentRiskParams params_;
  PersistentRiskTrackVector tracks_;
  int next_track_id_ = 0;
};

}  // namespace deform_monitor_v2

#endif  // DEFORM_MONITOR_V2_PERSISTENT_RISK_REGION_TRACKER_HPP
