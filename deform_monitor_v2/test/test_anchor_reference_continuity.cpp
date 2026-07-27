#include <gtest/gtest.h>

#include <ros/ros.h>

#include "deform_monitor_v2/core/reference_manager.hpp"
#include "deform_monitor_v2/data_types.hpp"
#include "deform_monitor_v2/visualization_publisher.hpp"

namespace deform_monitor_v2 {
namespace {

TEST(AnchorReferenceContinuityTest, PublishesReferenceLifecycleAndCurrentMatch) {
  AnchorReference anchor;
  anchor.id = 17;
  anchor.type = AnchorType::BAND;
  anchor.center_R = Eigen::Vector3d(1.0, 2.0, 3.0);
  anchor.reference_epoch = 4;
  anchor.reference_stamp = ros::Time(42, 500000000);
  anchor.reference_origin = AnchorReferenceOrigin::INITIAL;

  AnchorTrackState state;
  state.id = anchor.id;
  state.type = anchor.type;
  state.observable = true;
  state.comparable = true;
  state.reacquired = true;
  state.directional_S = Eigen::Vector3d(0.06, 0.0, 0.0);
  state.directional_magnitude_sum = 0.08;
  state.directional_persistent = true;

  CurrentObservation observation;
  observation.anchor_id = anchor.id;
  observation.observable = true;
  observation.comparable = true;
  observation.reacquired = true;
  observation.gate_state = ObsGateState::OBSERVABLE_MATCHED;
  observation.matched_center_R = Eigen::Vector3d(1.4, 2.1, 3.0);
  observation.matched_delta_R = Eigen::Vector3d(0.4, 0.1, 0.0);

  VisualizationPublisher publisher;
  const auto message = publisher.BuildAnchorStatesMsg(
      AnchorReferenceVector{anchor},
      AnchorStateVector{state},
      CurrentObservationVector{observation},
      ros::Time(50, 0),
      "camera_init",
      4,
      ros::Time(42, 500000000));

  ASSERT_EQ(message.anchors.size(), 1u);
  EXPECT_EQ(message.reference_epoch, 4u);
  EXPECT_EQ(message.reference_initialized_at, ros::Time(42, 500000000));

  const auto& published = message.anchors.front();
  EXPECT_EQ(published.obs_state,
            static_cast<uint8_t>(ObsGateState::OBSERVABLE_MATCHED));
  EXPECT_TRUE(published.observable);
  EXPECT_DOUBLE_EQ(published.matched_center.x, 1.4);
  EXPECT_DOUBLE_EQ(published.matched_center.y, 2.1);
  EXPECT_DOUBLE_EQ(published.matched_center.z, 3.0);
  EXPECT_DOUBLE_EQ(published.matched_delta.x, 0.4);
  EXPECT_DOUBLE_EQ(published.matched_delta.y, 0.1);
  EXPECT_DOUBLE_EQ(published.matched_delta.z, 0.0);
  EXPECT_EQ(published.reference_epoch, 4u);
  EXPECT_EQ(published.reference_stamp, ros::Time(42, 500000000));
  EXPECT_EQ(published.reference_origin,
            static_cast<uint8_t>(AnchorReferenceOrigin::INITIAL));
  EXPECT_DOUBLE_EQ(published.directional_strength, 0.06);
  EXPECT_DOUBLE_EQ(published.directional_coherence, 0.75);
  EXPECT_TRUE(published.directional_persistent);
}

TEST(AnchorReferenceContinuityTest, ObservationLossDoesNotMutateFrozenReference) {
  AnchorReference anchor;
  anchor.id = 9;
  anchor.type = AnchorType::EDGE;
  anchor.center_R = Eigen::Vector3d(0.4, -0.2, 1.3);
  anchor.reference_epoch = 6;
  anchor.reference_stamp = ros::Time(80, 0);
  anchor.reference_origin = AnchorReferenceOrigin::INITIAL;
  anchor.frozen = true;

  CurrentObservation observation;
  observation.anchor_id = anchor.id;
  observation.observable = false;
  observation.comparable = false;
  observation.gate_state = ObsGateState::NOT_OBSERVABLE;

  AnchorTrackState state;
  state.id = anchor.id;
  state.type = anchor.type;

  AnchorReferenceVector anchors{anchor};
  CurrentObservationVector observations{observation};
  AnchorStateVector states{state};
  ReferenceManager manager;
  manager.SetParams(ReferenceParams{});
  manager.UpdateReferenceStatistics(&anchors, observations, &states);

  ASSERT_EQ(anchors.size(), 1u);
  EXPECT_TRUE(anchors.front().center_R.isApprox(anchor.center_R, 1.0e-12));
  EXPECT_EQ(anchors.front().reference_epoch, anchor.reference_epoch);
  EXPECT_EQ(anchors.front().reference_stamp, anchor.reference_stamp);
  EXPECT_EQ(anchors.front().reference_origin, anchor.reference_origin);
  EXPECT_TRUE(anchors.front().frozen);
}

TEST(AnchorReferenceContinuityTest, DetectedObjectBoxUsesAllFrozenAnchorsOfObject) {
  AnchorReference first;
  first.id = 1;
  first.object_id = 31;
  first.object_id_valid = true;
  first.center_R = Eigen::Vector3d(0.0, 0.0, 0.0);
  AnchorReference second = first;
  second.id = 2;
  second.center_R = Eigen::Vector3d(1.0, 2.0, 3.0);

  AnchorTrackState quiet;
  AnchorTrackState detected;
  detected.significant = true;
  detected.mode = DetectionMode::DISPLACEMENT;

  VisualizationParams params;
  params.show_detected_object_boxes = true;
  params.detected_object_box_margin = 0.10;
  VisualizationPublisher publisher;
  publisher.SetParams(params);

  const auto markers = publisher.BuildMotionMarkers(
      AnchorReferenceVector{first, second},
      AnchorStateVector{quiet, detected},
      MotionClusterVector{},
      ros::Time(60, 0),
      "camera_init");

  const visualization_msgs::Marker* object_box = nullptr;
  for (const auto& marker : markers.markers) {
    if (marker.ns == "detected_objects") {
      object_box = &marker;
      break;
    }
  }
  ASSERT_NE(object_box, nullptr);
  EXPECT_EQ(object_box->type, visualization_msgs::Marker::CUBE);
  EXPECT_DOUBLE_EQ(object_box->pose.position.x, 0.5);
  EXPECT_DOUBLE_EQ(object_box->pose.position.y, 1.0);
  EXPECT_DOUBLE_EQ(object_box->pose.position.z, 1.5);
  EXPECT_DOUBLE_EQ(object_box->scale.x, 1.2);
  EXPECT_DOUBLE_EQ(object_box->scale.y, 2.2);
  EXPECT_DOUBLE_EQ(object_box->scale.z, 3.2);
}

}  // namespace
}  // namespace deform_monitor_v2

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  ros::init(argc, argv, "anchor_reference_continuity_test",
            ros::init_options::AnonymousName);
  return RUN_ALL_TESTS();
}
