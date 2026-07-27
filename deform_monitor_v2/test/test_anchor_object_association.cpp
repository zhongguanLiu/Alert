#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>

#include "deform_monitor_v2/core/anchor_builder.hpp"
#include "deform_monitor_v2/core/current_observation_extractor.hpp"
#include "deform_monitor_v2/core/motion_clusterer.hpp"
#include "deform_monitor_v2/core/object_id_association.hpp"
#include "deform_monitor_v2/core/object_observation_stats.hpp"
#include "deform_monitor_v2/core/persistent_risk_region_tracker.hpp"
#include "deform_monitor_v2/core/risk_evidence_adapter.hpp"
#include "deform_monitor_v2/core/risk_field_builder.hpp"
#include "deform_monitor_v2/data_types.hpp"
#include "deform_monitor_v2/risk_visualization_publisher.hpp"
#include "deform_monitor_v2/visualization_publisher.hpp"

namespace deform_monitor_v2 {
namespace {

AnchorBuildParams PermissiveAnchorParams() {
  AnchorBuildParams params;
  params.I_min = 0.0;
  params.tau_ref_quality = 0.0;
  params.voxel_size = 0.20;
  params.min_visible_frames = 2;
  params.min_points_per_voxel = 3;
  params.neighborhood_layers = 0;
  params.radius_min = 0.01;
  params.min_support_points = 5;
  params.edge_ref_bonus = 0.0;
  params.band_ref_bonus = 0.0;
  return params;
}

ObjectAssociationParams EnabledAssociationParams(double min_purity = 0.80) {
  ObjectAssociationParams params;
  params.enable = true;
  params.invalid_id = 0;
  params.max_id = 254;
  params.min_support_points = 3;
  params.min_purity = min_purity;
  params.quantization_tolerance = 0.25;
  return params;
}

ReferenceInitFrameVector MakePlanarFrames(const std::vector<uint16_t>& labels) {
  ReferenceInitFrameVector frames;
  for (int frame_index = 0; frame_index < 2; ++frame_index) {
    ReferenceInitFrame frame;
    frame.cloud.reset(new pcl::PointCloud<pcl::PointXYZI>());
    frame.lidar_origin_R = Eigen::Vector3d::Zero();
    frame.stamp = ros::Time(10 + frame_index, 0);
    for (int ix = 0; ix < 5; ++ix) {
      for (int iy = 0; iy < 5; ++iy) {
        const size_t point_index = static_cast<size_t>(ix * 5 + iy);
        pcl::PointXYZI point;
        point.x = 0.06f + 0.01f * static_cast<float>(ix);
        point.y = 0.06f + 0.01f * static_cast<float>(iy);
        point.z = 1.00f;
        point.intensity = static_cast<float>(labels.at(point_index));
        frame.cloud->points.push_back(point);
      }
    }
    frame.cloud->width = frame.cloud->points.size();
    frame.cloud->height = 1;
    frames.push_back(std::move(frame));
  }
  return frames;
}

AnchorReference MakeObservedPlaneAnchor(uint16_t object_id) {
  AnchorReference anchor;
  anchor.id = 9;
  anchor.type = AnchorType::PLANE;
  anchor.center_R = Eigen::Vector3d(1.0, 0.0, 0.0);
  anchor.normal_R = Eigen::Vector3d::UnitX();
  anchor.edge_normal_R = Eigen::Vector3d::UnitY();
  anchor.basis_R.col(0) = anchor.normal_R;
  anchor.basis_R.col(1) = Eigen::Vector3d::UnitY();
  anchor.basis_R.col(2) = Eigen::Vector3d::UnitZ();
  anchor.Sigma_ref_geom = Eigen::Matrix3d::Identity() * 1.0e-6;
  anchor.mean_range = 1.0;
  anchor.mean_view_dir_R = Eigen::Vector3d::UnitX();
  anchor.mean_incidence_cos = 1.0;
  anchor.edge_center_R = anchor.center_R;
  anchor.band_center_R = anchor.center_R;
  anchor.support_radius = 0.10;
  anchor.support_target_count = 25;
  anchor.object_id = object_id;
  anchor.object_id_valid = true;
  return anchor;
}

pcl::PointCloud<pcl::PointXYZI>::Ptr MakeCurrentPlaneCloud(uint16_t object_id) {
  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>());
  for (int iy = -2; iy <= 2; ++iy) {
    for (int iz = -2; iz <= 2; ++iz) {
      pcl::PointXYZI point;
      point.x = 1.0f;
      point.y = 0.02f * static_cast<float>(iy);
      point.z = 0.02f * static_cast<float>(iz);
      point.intensity = static_cast<float>(object_id);
      cloud->points.push_back(point);
    }
  }
  cloud->width = cloud->points.size();
  cloud->height = 1;
  return cloud;
}

CurrentObservation ObserveCurrentPlane(uint16_t reference_object_id,
                                       uint16_t current_object_id) {
  ObservationParams observation_params;
  observation_params.current_voxel_size = 0.05;
  observation_params.min_support_scalar = 3;
  observation_params.tau_cmp = 0.0;
  observation_params.tau_nis_scalar = 100.0;

  NoiseParams noise_params;
  ScalarMeasurementBuilder measurement_builder;
  measurement_builder.SetParams(observation_params, noise_params);

  CurrentObservationExtractor extractor;
  extractor.SetParams(observation_params);
  extractor.SetNoiseParams(noise_params);
  extractor.SetCovarianceParams(CovarianceParams());
  extractor.SetObservabilityParams(ObservabilityParams());
  extractor.SetMeasurementBuilder(measurement_builder);
  extractor.SetObjectAssociationParams(EnabledAssociationParams());

  PoseCov6D pose_cov;
  pose_cov.Sigma_xi = Eigen::Matrix<double, 6, 6>::Identity() * 1.0e-8;
  pose_cov.stamp = ros::Time(20, 0);
  extractor.PrepareSingleFrame(
      MakeCurrentPlaneCloud(current_object_id), pose_cov, Eigen::Vector3d::Zero());
  return extractor.ExtractForAnchorFromPreparedCache(
      MakeObservedPlaneAnchor(reference_object_id),
      pose_cov,
      Eigen::Vector3d::Zero(),
      Eigen::Vector3d::Zero());
}

TEST(ObjectIdAssociationTest, ReportsValidDominantIdentity) {
  const ObjectIdAssociationResult result = AssociateObjectIdSamples(
      {17.0f, 17.0f, 17.0f, 17.0f, 23.0f},
      EnabledAssociationParams(0.80));

  EXPECT_EQ(result.status, ObjectIdAssociationStatus::VALID);
  EXPECT_TRUE(result.valid);
  EXPECT_EQ(result.object_id, 17u);
  EXPECT_DOUBLE_EQ(result.confidence, 0.80);
  EXPECT_EQ(result.support_count, 5);
  EXPECT_EQ(result.distinct_id_count, 2);
}

TEST(ObjectIdAssociationTest, DistinguishesMixedIdentityFromMissingData) {
  const ObjectIdAssociationResult result = AssociateObjectIdSamples(
      {17.0f, 17.0f, 17.0f, 23.0f, 23.0f},
      EnabledAssociationParams(0.80));

  EXPECT_EQ(result.status, ObjectIdAssociationStatus::MIXED);
  EXPECT_FALSE(result.valid);
  EXPECT_EQ(result.object_id, 0u);
  EXPECT_DOUBLE_EQ(result.confidence, 0.60);
  EXPECT_EQ(result.support_count, 5);
  EXPECT_EQ(result.distinct_id_count, 2);
}

TEST(ObjectIdAssociationTest, ReportsInsufficientValidLabelSupport) {
  const ObjectIdAssociationResult result = AssociateObjectIdSamples(
      {17.0f, 17.0f, 0.0f, 999.0f, 4.5f},
      EnabledAssociationParams());

  EXPECT_EQ(result.status, ObjectIdAssociationStatus::INSUFFICIENT);
  EXPECT_FALSE(result.valid);
  EXPECT_EQ(result.object_id, 0u);
  EXPECT_DOUBLE_EQ(result.confidence, 1.0);
  EXPECT_EQ(result.support_count, 2);
  EXPECT_EQ(result.distinct_id_count, 1);
}

TEST(ObjectIdAssociationTest, ReportsDisabledWithoutInterpretingIntensity) {
  ObjectAssociationParams params = EnabledAssociationParams();
  params.enable = false;

  const ObjectIdAssociationResult result =
      AssociateObjectIdSamples({17.0f, 17.0f, 17.0f}, params);

  EXPECT_EQ(result.status, ObjectIdAssociationStatus::DISABLED);
  EXPECT_FALSE(result.valid);
  EXPECT_EQ(result.object_id, 0u);
  EXPECT_DOUBLE_EQ(result.confidence, 0.0);
  EXPECT_EQ(result.support_count, 0);
  EXPECT_EQ(result.distinct_id_count, 0);
}

TEST(AnchorObjectAssociationTest, FreezesDominantPointObjectIdAtAnchorCreation) {
  AnchorBuilder builder;
  builder.SetParams(PermissiveAnchorParams());
  builder.SetObjectAssociationParams(EnabledAssociationParams());

  const auto anchors = builder.BuildFrozenAnchors(
      MakePlanarFrames(std::vector<uint16_t>(25, 17)));

  ASSERT_EQ(anchors.size(), 1u);
  EXPECT_TRUE(anchors.front().object_id_valid);
  EXPECT_EQ(anchors.front().object_id, 17u);
  EXPECT_DOUBLE_EQ(anchors.front().object_id_confidence, 1.0);
  EXPECT_EQ(anchors.front().object_id_support_count, 50);
  EXPECT_LT(anchors.front().shape_linearity, 0.10);
  EXPECT_GT(anchors.front().shape_planarity, 0.90);
  EXPECT_LT(anchors.front().shape_scattering, 0.10);
}

TEST(AnchorObjectAssociationTest, RejectsMixedBoundaryVoxelBelowPurityThreshold) {
  std::vector<uint16_t> labels(25, 17);
  std::fill(labels.begin() + 15, labels.end(), 23);

  AnchorBuilder builder;
  builder.SetParams(PermissiveAnchorParams());
  builder.SetObjectAssociationParams(EnabledAssociationParams(0.80));

  const auto anchors = builder.BuildFrozenAnchors(MakePlanarFrames(labels));

  ASSERT_EQ(anchors.size(), 1u);
  EXPECT_FALSE(anchors.front().object_id_valid);
  EXPECT_EQ(anchors.front().object_id, 0u);
  EXPECT_DOUBLE_EQ(anchors.front().object_id_confidence, 0.60);
  EXPECT_EQ(anchors.front().object_id_support_count, 50);
}

TEST(AnchorObjectAssociationTest, AssociationDoesNotChangeAnchorGeometry) {
  const auto frames = MakePlanarFrames(std::vector<uint16_t>(25, 17));

  AnchorBuilder plain_builder;
  plain_builder.SetParams(PermissiveAnchorParams());
  const auto plain_anchors = plain_builder.BuildFrozenAnchors(frames);

  AnchorBuilder associated_builder;
  associated_builder.SetParams(PermissiveAnchorParams());
  associated_builder.SetObjectAssociationParams(EnabledAssociationParams());
  const auto associated_anchors = associated_builder.BuildFrozenAnchors(frames);

  ASSERT_EQ(plain_anchors.size(), associated_anchors.size());
  ASSERT_EQ(plain_anchors.size(), 1u);
  EXPECT_EQ(plain_anchors.front().type, associated_anchors.front().type);
  EXPECT_TRUE(plain_anchors.front().center_R.isApprox(
      associated_anchors.front().center_R, 1.0e-12));
  EXPECT_TRUE(plain_anchors.front().basis_R.isApprox(
      associated_anchors.front().basis_R, 1.0e-12));
  EXPECT_TRUE(plain_anchors.front().Sigma_ref_geom.isApprox(
      associated_anchors.front().Sigma_ref_geom, 1.0e-12));
  EXPECT_DOUBLE_EQ(plain_anchors.front().ref_quality,
                   associated_anchors.front().ref_quality);
  EXPECT_DOUBLE_EQ(plain_anchors.front().shape_linearity,
                   associated_anchors.front().shape_linearity);
  EXPECT_DOUBLE_EQ(plain_anchors.front().shape_planarity,
                   associated_anchors.front().shape_planarity);
  EXPECT_DOUBLE_EQ(plain_anchors.front().shape_scattering,
                   associated_anchors.front().shape_scattering);
}

TEST(CurrentSupportObjectAssociationTest, ReportsConsistentDominantIdentity) {
  const CurrentObservation observation = ObserveCurrentPlane(17, 17);

  EXPECT_TRUE(observation.observed_object_id_valid);
  EXPECT_EQ(observation.observed_object_id, 17u);
  EXPECT_DOUBLE_EQ(observation.observed_object_id_confidence, 1.0);
  EXPECT_EQ(observation.observed_object_id_support_count, 25);
  EXPECT_EQ(observation.object_association_state,
            ObjectAssociationState::CONSISTENT);
}

TEST(CurrentSupportObjectAssociationTest, ReportsReferenceCurrentMismatch) {
  const CurrentObservation observation = ObserveCurrentPlane(17, 23);

  EXPECT_TRUE(observation.observed_object_id_valid);
  EXPECT_EQ(observation.observed_object_id, 23u);
  EXPECT_EQ(observation.object_association_state,
            ObjectAssociationState::MISMATCH);
}

TEST(CurrentSupportObjectAssociationTest, LabelsDoNotChangeGeometryOrMeasurements) {
  const CurrentObservation consistent = ObserveCurrentPlane(17, 17);
  const CurrentObservation mismatch = ObserveCurrentPlane(17, 23);

  EXPECT_EQ(consistent.support_count, mismatch.support_count);
  EXPECT_EQ(consistent.status, mismatch.status);
  EXPECT_EQ(consistent.gate_state, mismatch.gate_state);
  EXPECT_EQ(consistent.comparable, mismatch.comparable);
  EXPECT_EQ(consistent.observable, mismatch.observable);
  EXPECT_EQ(consistent.reacquired, mismatch.reacquired);
  EXPECT_DOUBLE_EQ(consistent.cmp_score, mismatch.cmp_score);
  EXPECT_DOUBLE_EQ(consistent.fit_rmse, mismatch.fit_rmse);
  EXPECT_DOUBLE_EQ(consistent.overlap_score, mismatch.overlap_score);
  EXPECT_TRUE(consistent.matched_center_R.isApprox(mismatch.matched_center_R, 0.0));
  EXPECT_TRUE(consistent.matched_delta_R.isApprox(mismatch.matched_delta_R, 0.0));
  ASSERT_EQ(consistent.scalars.size(), mismatch.scalars.size());
  for (size_t i = 0; i < consistent.scalars.size(); ++i) {
    EXPECT_EQ(consistent.scalars[i].type, mismatch.scalars[i].type);
    EXPECT_TRUE(consistent.scalars[i].h_R.isApprox(mismatch.scalars[i].h_R, 0.0));
    EXPECT_DOUBLE_EQ(consistent.scalars[i].z, mismatch.scalars[i].z);
    EXPECT_DOUBLE_EQ(consistent.scalars[i].r, mismatch.scalars[i].r);
  }
}

TEST(AssociationAuditPropagationTest, PublishesAnchorAndEvidenceAuditFields) {
  AnchorReference anchor = MakeObservedPlaneAnchor(17);
  anchor.object_id_confidence = 0.95;
  anchor.object_id_support_count = 40;
  AnchorReferenceVector anchors{anchor};

  AnchorTrackState state;
  state.id = anchor.id;
  state.observable = true;
  state.comparable = true;
  state.significant = true;
  state.gate_state = ObsGateState::OBSERVABLE_MATCHED;
  state.x_mix(0) = 0.03;
  AnchorStateVector states{state};

  CurrentObservation observation;
  observation.anchor_id = anchor.id;
  observation.observable = true;
  observation.comparable = true;
  observation.gate_state = ObsGateState::OBSERVABLE_MATCHED;
  observation.support_count = 12;
  observation.observed_object_id = 23;
  observation.observed_object_id_valid = true;
  observation.observed_object_id_confidence = 0.90;
  observation.observed_object_id_support_count = 12;
  observation.object_association_state = ObjectAssociationState::MISMATCH;
  CurrentObservationVector observations{observation};

  RiskVisualizationParams risk_params;
  risk_params.min_confidence = 0.0;
  risk_params.min_risk_score = 0.0;
  RiskEvidenceAdapter adapter;
  adapter.SetParams(risk_params, SignificanceParams(), GraphTemporalParams());
  const RiskEvidenceVector evidences =
      adapter.Build(anchors, states, observations, MotionClusterVector());

  ASSERT_EQ(evidences.size(), 1u);
  EXPECT_EQ(evidences[0].observed_object_id, 23u);
  EXPECT_TRUE(evidences[0].observed_object_id_valid);
  EXPECT_DOUBLE_EQ(evidences[0].observed_object_id_confidence, 0.90);
  EXPECT_EQ(evidences[0].observed_object_id_support_count, 12);
  EXPECT_EQ(evidences[0].object_association_state,
            ObjectAssociationState::MISMATCH);

  VisualizationPublisher visualization_publisher;
  const auto anchor_msg = visualization_publisher.BuildAnchorStatesMsg(
      anchors, states, observations, ros::Time(20, 0), "reference", 1, ros::Time(10, 0));
  ASSERT_EQ(anchor_msg.anchors.size(), 1u);
  EXPECT_EQ(anchor_msg.anchors[0].observed_object_id, 23u);
  EXPECT_TRUE(anchor_msg.anchors[0].observed_object_id_valid);
  EXPECT_DOUBLE_EQ(anchor_msg.anchors[0].observed_object_id_confidence, 0.90);
  EXPECT_EQ(anchor_msg.anchors[0].observed_object_id_support_count, 12);
  EXPECT_EQ(anchor_msg.anchors[0].object_association_state,
            static_cast<uint8_t>(ObjectAssociationState::MISMATCH));

  RiskVisualizationPublisher risk_publisher;
  const auto evidence_msg = risk_publisher.BuildRiskEvidenceMsg(
      evidences, ros::Time(20, 0), "reference");
  ASSERT_EQ(evidence_msg.evidences.size(), 1u);
  EXPECT_EQ(evidence_msg.evidences[0].observed_object_id, 23u);
  EXPECT_EQ(evidence_msg.evidences[0].object_association_state,
            static_cast<uint8_t>(ObjectAssociationState::MISMATCH));
}

TEST(AssociationAuditPropagationTest, ClusterPreservesCrossObjectAmbiguity) {
  AnchorReference first = MakeObservedPlaneAnchor(17);
  first.id = 1;
  first.center_R = Eigen::Vector3d(1.0, 0.0, 0.0);
  first.neighbor_indices = {1};
  AnchorReference second = first;
  second.id = 2;
  second.center_R = Eigen::Vector3d(1.0, 0.04, 0.0);
  second.neighbor_indices = {0};
  AnchorReferenceVector anchors{first, second};

  AnchorTrackState first_state;
  first_state.id = first.id;
  first_state.significant = true;
  first_state.comparable = true;
  first_state.x_mix(0) = 0.03;
  first_state.observed_object_id = 17;
  first_state.observed_object_id_valid = true;
  first_state.observed_object_id_confidence = 1.0;
  first_state.object_association_state = ObjectAssociationState::CONSISTENT;
  AnchorTrackState second_state = first_state;
  second_state.id = second.id;
  second_state.observed_object_id = 23;
  second_state.object_association_state = ObjectAssociationState::MISMATCH;
  AnchorStateVector states{first_state, second_state};

  ClusterParams params;
  params.tau_corr = 0.0;
  params.tau_edge_score = 0.0;
  MotionClusterer clusterer;
  clusterer.SetParams(params);
  const MotionClusterVector clusters = clusterer.Cluster(anchors, states);

  ASSERT_EQ(clusters.size(), 1u);
  EXPECT_TRUE(clusters[0].object_id_valid);
  EXPECT_EQ(clusters[0].object_id, 17u);
  EXPECT_TRUE(clusters[0].observed_object_id_ambiguous);
  EXPECT_FALSE(clusters[0].observed_object_id_valid);
  EXPECT_EQ(clusters[0].association_consistent_count, 1);
  EXPECT_EQ(clusters[0].association_mismatch_count, 1);
  EXPECT_EQ(clusters[0].object_association_state,
            ObjectAssociationState::MISMATCH);
}

TEST(AssociationAuditPropagationTest, RegionAndPersistentTrackKeepAssociationEvidence) {
  AnchorReference anchor = MakeObservedPlaneAnchor(17);
  AnchorReferenceVector anchors{anchor};

  RiskEvidenceState evidence;
  evidence.id = anchor.id;
  evidence.anchor_type = anchor.type;
  evidence.object_id = 17;
  evidence.object_id_valid = true;
  evidence.object_id_confidence = 0.95;
  evidence.observed_object_id = 23;
  evidence.observed_object_id_valid = true;
  evidence.observed_object_id_confidence = 0.90;
  evidence.observed_object_id_support_count = 8;
  evidence.object_association_state = ObjectAssociationState::MISMATCH;
  evidence.position_R = anchor.center_R;
  evidence.displacement_score = 0.0;
  evidence.disappearance_score = 1.0;
  evidence.confidence = 1.0;
  evidence.risk_score = 1.0;
  evidence.active = true;

  RiskVisualizationParams params;
  params.voxel_size = 0.05;
  params.kernel_radius = 0.05;
  params.min_confidence = 0.0;
  params.min_risk_score = 0.0;
  params.min_voxel_risk = 0.0;
  params.min_region_voxels = 1;
  params.min_region_mean_risk = 0.0;
  RiskFieldBuilder builder;
  builder.SetParams(params);
  const RiskVoxelVector voxels = builder.Build(anchors, RiskEvidenceVector{evidence});
  const RiskRegionVector regions = builder.ExtractRegions(voxels);

  ASSERT_EQ(regions.size(), 1u);
  EXPECT_EQ(regions[0].observed_object_id, 23u);
  EXPECT_TRUE(regions[0].observed_object_id_valid);
  EXPECT_EQ(regions[0].association_mismatch_count, 1);
  EXPECT_EQ(regions[0].object_association_state,
            ObjectAssociationState::MISMATCH);

  PersistentRiskRegionTracker tracker;
  tracker.SetParams(PersistentRiskParams());
  const PersistentRiskTrackVector tracks = tracker.Update(regions, ros::Time(20, 0));
  ASSERT_EQ(tracks.size(), 1u);
  EXPECT_EQ(tracks[0].observed_object_id, 23u);
  EXPECT_TRUE(tracks[0].observed_object_id_valid);
  EXPECT_EQ(tracks[0].association_mismatch_count, 1);
  EXPECT_EQ(tracks[0].object_association_state,
            ObjectAssociationState::MISMATCH);
}

TEST(ObjectObservationStatsTest, AggregatesFiveFramesWithoutPointLevelLogging) {
  ObjectObservationStatsAccumulator accumulator;
  accumulator.SetParams(EnabledAssociationParams());

  for (int frame_index = 0; frame_index < 5; ++frame_index) {
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>());
    const std::vector<float> labels = frame_index < 3
                                          ? std::vector<float>{17.0f, 17.0f, 23.0f, 0.0f}
                                          : std::vector<float>{17.0f, 23.0f, 23.0f, 4.5f};
    for (size_t i = 0; i < labels.size(); ++i) {
      pcl::PointXYZI point;
      point.x = static_cast<float>(i);
      point.intensity = labels[i];
      cloud->points.push_back(point);
    }
    cloud->width = cloud->points.size();
    cloud->height = 1;
    accumulator.AddFrame(cloud, ros::Time(30 + frame_index, 0));
  }

  const ObjectObservationStatsState summary = accumulator.BuildSummary(
      4, ObjectObservationPhase::MONITORING);
  ASSERT_EQ(summary.frame_count, 5u);
  EXPECT_EQ(summary.total_point_count, 20u);
  EXPECT_EQ(summary.valid_label_point_count, 15u);
  EXPECT_EQ(summary.invalid_label_point_count, 5u);
  EXPECT_EQ(summary.window_start, ros::Time(30, 0));
  EXPECT_EQ(summary.window_end, ros::Time(34, 0));
  ASSERT_EQ(summary.objects.size(), 2u);
  EXPECT_EQ(summary.objects[0].object_id, 17u);
  EXPECT_EQ(summary.objects[0].point_count, 8u);
  EXPECT_EQ(summary.objects[0].visible_frame_count, 5u);
  EXPECT_EQ(summary.objects[1].object_id, 23u);
  EXPECT_EQ(summary.objects[1].point_count, 7u);
  EXPECT_EQ(summary.objects[1].visible_frame_count, 5u);
}

}  // namespace
}  // namespace deform_monitor_v2

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
