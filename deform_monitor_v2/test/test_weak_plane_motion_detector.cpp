#include <gtest/gtest.h>

#include "deform_monitor_v2/core/weak_plane_motion_detector.hpp"

namespace deform_monitor_v2 {
namespace {

void ConnectAll(AnchorReferenceVector* anchors) {
  for (size_t i = 0; i < anchors->size(); ++i) {
    for (size_t j = 0; j < anchors->size(); ++j) {
      if (i != j) {
        (*anchors)[i].neighbor_indices.push_back(static_cast<int>(j));
      }
    }
  }
}

AnchorReferenceVector MakePlaneAnchors(size_t count) {
  AnchorReferenceVector anchors(count);
  for (size_t i = 0; i < count; ++i) {
    anchors[i].id = static_cast<int>(i);
    anchors[i].type = AnchorType::PLANE;
    anchors[i].center_R = Eigen::Vector3d(0.02 * static_cast<double>(i), 0.0, 0.0);
    anchors[i].normal_R = Eigen::Vector3d::UnitZ();
    anchors[i].ref_quality = 0.9;
    anchors[i].covariance_quality = 0.9;
    anchors[i].type_stability = 0.9;
  }
  ConnectAll(&anchors);
  return anchors;
}

AnchorStateVector MakeStates(size_t count, double displacement) {
  AnchorStateVector states(count);
  for (size_t i = 0; i < count; ++i) {
    states[i].id = static_cast<int>(i);
    states[i].type = AnchorType::PLANE;
    states[i].gate_state = ObsGateState::OBSERVABLE_MATCHED;
    states[i].observable = true;
    states[i].comparable = true;
    states[i].dof_obs = 1;
    states[i].x_mix(2) = displacement;
    states[i].disp_norm = std::abs(displacement);
    states[i].disp_normal = std::abs(displacement);
    states[i].chi2_stat = 9.0;
  }
  return states;
}

WeakPlaneMotionParams TestParams() {
  WeakPlaneMotionParams params;
  params.enable = true;
  params.radius = 0.20;
  params.min_support = 6;
  params.min_streak = 3;
  params.min_anchor_disp = 0.003;
  params.min_group_disp = 0.004;
  params.min_mean_chi2 = 4.0;
  params.min_direction_consistency = 0.80;
  params.max_group_residual = 0.004;
  return params;
}

TEST(WeakPlaneMotionDetectorTest, RequiresSpatialAndTemporalGroupSupport) {
  const AnchorReferenceVector anchors = MakePlaneAnchors(6);
  AnchorStateVector states = MakeStates(6, 0.006);
  WeakPlaneMotionDetector detector;
  detector.SetParams(TestParams());

  detector.Update(anchors, &states);
  EXPECT_FALSE(states.front().weak_plane_candidate);
  EXPECT_EQ(states.front().weak_plane_streak, 1);
  detector.Update(anchors, &states);
  EXPECT_FALSE(states.front().weak_plane_candidate);
  detector.Update(anchors, &states);

  for (const auto& state : states) {
    EXPECT_TRUE(state.weak_plane_candidate);
    EXPECT_EQ(state.weak_plane_group_size, 6);
    EXPECT_NEAR(state.weak_plane_group_disp, 0.006, 1.0e-9);
    EXPECT_NEAR(state.weak_plane_direction_consistency, 1.0, 1.0e-9);
  }
}

TEST(WeakPlaneMotionDetectorTest, RejectsDirectionallyIncoherentPlaneMotion) {
  const AnchorReferenceVector anchors = MakePlaneAnchors(6);
  AnchorStateVector states = MakeStates(6, 0.006);
  for (size_t i = 0; i < states.size(); i += 2) {
    states[i].x_mix(2) = -0.006;
  }
  WeakPlaneMotionDetector detector;
  detector.SetParams(TestParams());

  for (int frame = 0; frame < 4; ++frame) {
    detector.Update(anchors, &states);
  }
  for (const auto& state : states) {
    EXPECT_FALSE(state.weak_plane_candidate);
  }
}

TEST(WeakPlaneMotionDetectorTest, RejectsReacquiredAndNonPlaneAnchors) {
  AnchorReferenceVector anchors = MakePlaneAnchors(6);
  AnchorStateVector states = MakeStates(6, 0.006);
  states[0].reacquired = true;
  anchors[1].type = AnchorType::EDGE;
  WeakPlaneMotionDetector detector;
  detector.SetParams(TestParams());

  for (int frame = 0; frame < 4; ++frame) {
    detector.Update(anchors, &states);
  }
  for (const auto& state : states) {
    EXPECT_FALSE(state.weak_plane_candidate);
  }
}

TEST(WeakPlaneMotionDetectorTest, AccumulatesDistinctAnchorsAcrossShortWindow) {
  const AnchorReferenceVector anchors = MakePlaneAnchors(6);
  AnchorStateVector states = MakeStates(6, 0.006);
  WeakPlaneMotionDetector detector;
  WeakPlaneMotionParams params = TestParams();
  params.temporal_window_frames = 5;
  params.min_current_support = 2;
  params.min_temporal_frames = 2;
  detector.SetParams(params);

  for (int frame = 0; frame < 4; ++frame) {
    for (size_t index = 0; index < states.size(); ++index) {
      const bool current_half =
          (frame % 2 == 0) ? index < 3 : index >= 3;
      states[index].gate_state = current_half
                                     ? ObsGateState::OBSERVABLE_MATCHED
                                     : ObsGateState::NOT_OBSERVABLE;
      states[index].observable = current_half;
      states[index].comparable = current_half;
    }
    detector.Update(anchors, &states);
  }

  int candidate_count = 0;
  for (const auto& state : states) {
    candidate_count += state.weak_plane_candidate ? 1 : 0;
  }
  EXPECT_EQ(candidate_count, 3);
  EXPECT_EQ(states[3].weak_plane_group_size, 6);
  EXPECT_EQ(states[3].weak_plane_current_support, 3);
  EXPECT_GE(states[3].weak_plane_temporal_frame_support, 2);
  EXPECT_GE(states[3].weak_plane_streak, 3);
}

TEST(WeakPlaneMotionDetectorTest, RequiresFreshCurrentSupportForTemporalGroup) {
  const AnchorReferenceVector anchors = MakePlaneAnchors(6);
  AnchorStateVector states = MakeStates(6, 0.006);
  WeakPlaneMotionDetector detector;
  WeakPlaneMotionParams params = TestParams();
  params.temporal_window_frames = 5;
  params.min_current_support = 2;
  params.min_temporal_frames = 2;
  detector.SetParams(params);

  for (int frame = 0; frame < 2; ++frame) {
    for (size_t index = 0; index < states.size(); ++index) {
      const bool current_half = (frame == 0) ? index < 3 : index >= 3;
      states[index].gate_state = current_half
                                     ? ObsGateState::OBSERVABLE_MATCHED
                                     : ObsGateState::NOT_OBSERVABLE;
      states[index].observable = current_half;
      states[index].comparable = current_half;
    }
    detector.Update(anchors, &states);
  }

  for (size_t index = 0; index < states.size(); ++index) {
    const bool only_one_current = index == 0;
    states[index].gate_state = only_one_current
                                   ? ObsGateState::OBSERVABLE_MATCHED
                                   : ObsGateState::NOT_OBSERVABLE;
    states[index].observable = only_one_current;
    states[index].comparable = only_one_current;
  }
  detector.Update(anchors, &states);

  for (const auto& state : states) {
    EXPECT_FALSE(state.weak_plane_candidate);
    EXPECT_EQ(state.weak_plane_current_support, 1);
  }
}

TEST(WeakPlaneMotionDetectorTest, TracksComponentAcrossCompleteAnchorMembershipChange) {
  const AnchorReferenceVector anchors = MakePlaneAnchors(12);
  AnchorStateVector states = MakeStates(12, 0.006);
  WeakPlaneMotionDetector detector;
  WeakPlaneMotionParams params = TestParams();
  params.temporal_window_frames = 1;
  params.component_match_radius = 0.20;
  detector.SetParams(params);

  for (int frame = 0; frame < 3; ++frame) {
    for (size_t index = 0; index < states.size(); ++index) {
      const bool current = (frame % 2 == 0) ? index < 6 : index >= 6;
      states[index].gate_state = current
                                     ? ObsGateState::OBSERVABLE_MATCHED
                                     : ObsGateState::NOT_OBSERVABLE;
      states[index].observable = current;
      states[index].comparable = current;
    }
    detector.Update(anchors, &states);
  }

  for (size_t index = 0; index < 6; ++index) {
    EXPECT_TRUE(states[index].weak_plane_candidate);
    EXPECT_EQ(states[index].weak_plane_streak, 3);
    EXPECT_GT(states[index].weak_plane_component_id, 0);
  }
  for (size_t index = 6; index < states.size(); ++index) {
    EXPECT_FALSE(states[index].weak_plane_candidate);
  }
}

TEST(WeakPlaneMotionDetectorTest, MixedObservableTypesFormOneMotionComponent) {
  AnchorReferenceVector anchors = MakePlaneAnchors(6);
  AnchorStateVector states = MakeStates(6, 0.006);
  for (size_t index = 3; index < anchors.size(); ++index) {
    anchors[index].type = AnchorType::EDGE;
    states[index].type = AnchorType::EDGE;
  }
  WeakPlaneMotionDetector detector;
  WeakPlaneMotionParams params = TestParams();
  params.enable_mixed_types = true;
  params.require_exterior_background_for_non_plane = false;
  detector.SetParams(params);

  for (int frame = 0; frame < 3; ++frame) {
    detector.Update(anchors, &states);
  }

  for (const auto& state : states) {
    EXPECT_TRUE(state.weak_plane_candidate);
    EXPECT_EQ(state.weak_plane_group_size, 6);
    EXPECT_EQ(state.weak_plane_mixed_type_support, 3);
  }
}

TEST(WeakPlaneMotionDetectorTest, NonPlaneWeakMotionRequiresStationaryExteriorSupport) {
  AnchorReferenceVector anchors = MakePlaneAnchors(8);
  AnchorStateVector states = MakeStates(8, 0.006);
  for (size_t index = 0; index < anchors.size(); ++index) {
    anchors[index].type = AnchorType::EDGE;
    states[index].type = AnchorType::EDGE;
  }
  for (size_t index = 6; index < states.size(); ++index) {
    states[index].x_mix.setZero();
    states[index].disp_norm = 0.0;
    states[index].disp_normal = 0.0;
    states[index].chi2_stat = 0.0;
    states[index].gate_state = ObsGateState::NOT_OBSERVABLE;
    states[index].observable = false;
    states[index].comparable = false;
  }
  WeakPlaneMotionDetector detector;
  WeakPlaneMotionParams params = TestParams();
  params.enable_mixed_types = true;
  params.require_exterior_background_for_non_plane = true;
  params.min_exterior_background_support = 2;
  detector.SetParams(params);

  for (int frame = 0; frame < 3; ++frame) {
    detector.Update(anchors, &states);
  }
  for (size_t index = 0; index < 6; ++index) {
    EXPECT_FALSE(states[index].weak_plane_candidate);
    EXPECT_EQ(states[index].weak_plane_exterior_background_support, 0);
  }

  for (size_t index = 6; index < states.size(); ++index) {
    states[index].gate_state = ObsGateState::OBSERVABLE_MATCHED;
    states[index].observable = true;
    states[index].comparable = true;
  }
  detector.Update(anchors, &states);
  for (size_t index = 0; index < 6; ++index) {
    EXPECT_TRUE(states[index].weak_plane_candidate);
    EXPECT_EQ(states[index].weak_plane_exterior_background_support, 2);
  }
}

}  // namespace
}  // namespace deform_monitor_v2

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
