#include <gtest/gtest.h>

#include "deform_monitor_v2/core/persistent_risk_region_tracker.hpp"

namespace deform_monitor_v2 {
namespace {

RiskRegionState MakeRegion(double center_x, uint16_t object_id) {
  RiskRegionState region;
  region.type = RiskRegionType::DISPLACEMENT_LIKE;
  region.object_id = object_id;
  region.object_id_valid = true;
  region.object_id_confidence = 1.0;
  region.observed_object_id = object_id;
  region.observed_object_id_valid = true;
  region.observed_object_id_confidence = 1.0;
  region.object_association_state = ObjectAssociationState::CONSISTENT;
  region.association_consistent_count = 1;
  region.center_R = Eigen::Vector3d(center_x, 0.0, 0.0);
  region.bbox_min_R = region.center_R - Eigen::Vector3d::Constant(0.04);
  region.bbox_max_R = region.center_R + Eigen::Vector3d::Constant(0.04);
  region.mean_risk = 0.8;
  region.peak_risk = 0.9;
  region.confidence = 0.9;
  region.voxel_count = 6;
  return region;
}

PersistentRiskParams TestParams() {
  PersistentRiskParams params;
  params.enable = true;
  params.max_center_distance = 0.12;
  params.min_bbox_iou = 0.01;
  params.max_risk_gap = 0.4;
  params.miss_frames_to_delete = 8;
  params.candidate_identity_memory_frames = 4;
  params.max_prediction_sec = 3.0;
  params.velocity_ema_alpha = 1.0;
  params.min_hits_to_confirm = 10;
  params.min_hit_streak_to_confirm = 10;
  params.enable_identity_backed_intermittent_confirmation = true;
  params.intermittent_min_hits_to_confirm = 4;
  params.min_identity_confidence = 0.9;
  params.identity_reconnect_distance_scale = 2.0;
  return params;
}

PersistentRiskParams IntermittentConfirmationParams() {
  PersistentRiskParams params = TestParams();
  params.window_size = 7;
  params.min_hits_to_confirm = 5;
  params.min_hit_streak_to_confirm = 5;
  params.min_confirmed_mean_risk = 0.55;
  params.min_confirmed_confidence = 0.5;
  params.min_confirmed_support_mass = 4.0;
  params.min_confirmed_span = 0.1;
  return params;
}

void ExpectAllCandidates(const PersistentRiskTrackVector& tracks) {
  for (const auto& track : tracks) {
    EXPECT_FALSE(track.ever_confirmed);
    EXPECT_EQ(track.state, PersistentRiskState::CANDIDATE);
  }
}

TEST(PersistentRiskRegionTrackerTest, ReconnectsAfterShortGapUsingMotionPrediction) {
  PersistentRiskRegionTracker tracker;
  tracker.SetParams(TestParams());

  auto tracks = tracker.Update(RiskRegionVector{MakeRegion(0.0, 1)}, ros::Time(10.0));
  ASSERT_EQ(tracks.size(), 1u);
  const int track_id = tracks.front().track_id;
  tracks = tracker.Update(RiskRegionVector{MakeRegion(0.1, 1)}, ros::Time(11.0));
  ASSERT_EQ(tracks.size(), 1u);
  tracks = tracker.Update(RiskRegionVector{}, ros::Time(12.0));
  ASSERT_EQ(tracks.size(), 1u);
  tracks = tracker.Update(RiskRegionVector{MakeRegion(0.3, 1)}, ros::Time(13.0));

  ASSERT_EQ(tracks.size(), 1u);
  EXPECT_EQ(tracks.front().track_id, track_id);
  EXPECT_EQ(tracks.front().miss_streak, 0);
}

TEST(PersistentRiskRegionTrackerTest, RejectsExplicitCrossObjectMerge) {
  PersistentRiskRegionTracker tracker;
  tracker.SetParams(TestParams());

  auto tracks = tracker.Update(RiskRegionVector{MakeRegion(0.0, 1)}, ros::Time(10.0));
  ASSERT_EQ(tracks.size(), 1u);
  tracks = tracker.Update(RiskRegionVector{MakeRegion(0.0, 2)}, ros::Time(11.0));

  ASSERT_EQ(tracks.size(), 2u);
  EXPECT_NE(tracks[0].track_id, tracks[1].track_id);
}

TEST(PersistentRiskRegionTrackerTest, RetainsCandidateUntilConfiguredDeleteLimit) {
  PersistentRiskRegionTracker tracker;
  tracker.SetParams(TestParams());

  auto tracks = tracker.Update(RiskRegionVector{MakeRegion(0.0, 1)}, ros::Time(10.0));
  ASSERT_EQ(tracks.size(), 1u);
  for (int miss = 1; miss < 8; ++miss) {
    tracks = tracker.Update(RiskRegionVector{}, ros::Time(10.0 + miss));
    ASSERT_EQ(tracks.size(), 1u) << "miss=" << miss;
  }
  tracks = tracker.Update(RiskRegionVector{}, ros::Time(18.0));
  EXPECT_TRUE(tracks.empty());
}

TEST(PersistentRiskRegionTrackerTest, ConfirmsIntermittentPlanarEvidenceForStableIdentity) {
  PersistentRiskRegionTracker tracker;
  tracker.SetParams(IntermittentConfirmationParams());

  auto tracks = tracker.Update(RiskRegionVector{MakeRegion(0.00, 1)}, ros::Time(10.0));
  tracks = tracker.Update(RiskRegionVector{}, ros::Time(11.0));
  tracks = tracker.Update(RiskRegionVector{MakeRegion(0.01, 1)}, ros::Time(12.0));
  tracks = tracker.Update(RiskRegionVector{}, ros::Time(13.0));
  tracks = tracker.Update(RiskRegionVector{MakeRegion(0.02, 1)}, ros::Time(14.0));
  ExpectAllCandidates(tracks);
  tracks = tracker.Update(RiskRegionVector{MakeRegion(0.03, 1)}, ros::Time(15.0));

  ASSERT_EQ(tracks.size(), 1u);
  EXPECT_TRUE(tracks.front().ever_confirmed);
  EXPECT_EQ(tracks.front().state, PersistentRiskState::CONFIRMED);
  EXPECT_EQ(tracks.front().matched_region_count_window, 4);
}

TEST(PersistentRiskRegionTrackerTest, MismatchedIdentitiesNeverMergeOrIntermittentlyConfirm) {
  PersistentRiskRegionTracker tracker;
  tracker.SetParams(IntermittentConfirmationParams());

  PersistentRiskTrackVector tracks;
  for (int frame = 0; frame < 6; ++frame) {
    const uint16_t object_id = frame % 2 == 0 ? 1 : 2;
    tracks = tracker.Update(
        RiskRegionVector{MakeRegion(0.01 * frame, object_id)},
        ros::Time(10.0 + frame));
  }

  ASSERT_EQ(tracks.size(), 2u);
  ExpectAllCandidates(tracks);
}

TEST(PersistentRiskRegionTrackerTest, MissingIdentityCannotUseIntermittentConfirmation) {
  PersistentRiskRegionTracker tracker;
  tracker.SetParams(IntermittentConfirmationParams());

  PersistentRiskTrackVector tracks;
  for (int frame = 0; frame < 7; ++frame) {
    if (frame == 1 || frame == 3 || frame == 5) {
      tracks = tracker.Update(RiskRegionVector{}, ros::Time(10.0 + frame));
      continue;
    }
    auto region = MakeRegion(0.01 * frame, 1);
    region.object_id_valid = false;
    region.observed_object_id_valid = false;
    region.object_association_state = ObjectAssociationState::UNAVAILABLE;
    region.association_consistent_count = 0;
    tracks = tracker.Update(RiskRegionVector{region}, ros::Time(10.0 + frame));
  }

  ASSERT_EQ(tracks.size(), 1u);
  ExpectAllCandidates(tracks);
}

TEST(PersistentRiskRegionTrackerTest, IsolatedShortBurstRemainsCandidate) {
  PersistentRiskRegionTracker tracker;
  tracker.SetParams(IntermittentConfirmationParams());

  PersistentRiskTrackVector tracks;
  for (int frame = 0; frame < 4; ++frame) {
    tracks = tracker.Update(
        RiskRegionVector{MakeRegion(0.01 * frame, 1)},
        ros::Time(10.0 + frame));
  }

  ASSERT_EQ(tracks.size(), 1u);
  ExpectAllCandidates(tracks);
}

TEST(PersistentRiskRegionTrackerTest, StrongSameIdentityUsesBoundedReconnectGate) {
  PersistentRiskRegionTracker tracker;
  auto params = TestParams();
  params.max_prediction_sec = 0.0;
  tracker.SetParams(params);

  auto tracks = tracker.Update(RiskRegionVector{MakeRegion(0.0, 1)}, ros::Time(10.0));
  ASSERT_EQ(tracks.size(), 1u);
  const int track_id = tracks.front().track_id;
  tracks = tracker.Update(RiskRegionVector{}, ros::Time(11.0));
  tracks = tracker.Update(RiskRegionVector{MakeRegion(0.18, 1)}, ros::Time(12.0));

  ASSERT_EQ(tracks.size(), 1u);
  EXPECT_EQ(tracks.front().track_id, track_id);
}

}  // namespace
}  // namespace deform_monitor_v2

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
