#include <gtest/gtest.h>

#include <sys/stat.h>
#include <unistd.h>

#include <fstream>
#include <string>

#include <ros/master.h>
#include <ros/ros.h>
#include <std_msgs/Empty.h>

#include "deform_monitor_v2/core/imm_information_filter.hpp"
#include "deform_monitor_v2/core/risk_evidence_adapter.hpp"
#include "deform_monitor_v2/core/risk_field_builder.hpp"
#include "deform_monitor_v2/core/persistent_risk_region_tracker.hpp"
#include "deform_monitor_v2/risk_visualization_publisher.hpp"
#define private public
#include "deform_monitor_v2/deform_monitor_v2_node.hpp"
#undef private

namespace deform_monitor_v2 {
namespace {

void EnsureRosInitialized() {
  if (ros::isInitialized()) {
    return;
  }
  int argc = 0;
  char** argv = nullptr;
  ros::init(argc, argv, "deform_monitor_v2_ablation_test", ros::init_options::AnonymousName);
}

AnchorReference MakePlaneAnchor() {
  AnchorReference anchor;
  anchor.id = 7;
  anchor.type = AnchorType::PLANE;
  anchor.basis_R = Eigen::Matrix3d::Identity();
  anchor.normal_R = Eigen::Vector3d::UnitZ();
  anchor.edge_normal_R = Eigen::Vector3d::UnitX();
  anchor.Sigma_ref_geom = Eigen::Matrix3d::Identity() * 1.0e-5;
  return anchor;
}

CurrentObservation MakeNormalObservation(double z_value, double variance) {
  CurrentObservation observation;
  observation.anchor_id = 7;
  observation.comparable = true;
  observation.observable = true;
  observation.gate_state = ObsGateState::OBSERVABLE_MATCHED;
  ScalarMeasurement scalar;
  scalar.h_R = Eigen::Vector3d::UnitZ();
  scalar.z = z_value;
  scalar.r = variance;
  observation.scalars.push_back(scalar);
  return observation;
}

ImmInformationFilter MakeFilter(bool enable_type_constraint,
                                bool enable_model_competition,
                                bool enable_cusum,
                                bool enable_directional) {
  ImmInformationFilter filter;
  ImmParams imm_params;
  imm_params.enable_type_constraint = enable_type_constraint;
  imm_params.enable_model_competition = enable_model_competition;

  ObservabilityParams observability_params;
  observability_params.tau_lambda = 1000.0;

  SignificanceParams significance_params;
  significance_params.enable_cusum = enable_cusum;
  significance_params.cusum_k = 1.0;
  significance_params.cusum_h = 2.0;

  DirectionalMotionParams directional_params;
  directional_params.enable = enable_directional;
  directional_params.tau_s = 0.01;
  directional_params.tau_c = 0.1;

  filter.SetParams(
      imm_params, observability_params, significance_params, directional_params, 0.8);
  return filter;
}

ImmInformationFilter MakeDirectionalFilter(double tau_s, double tau_c) {
  ImmInformationFilter filter;
  ImmParams imm_params;
  ObservabilityParams observability_params;
  SignificanceParams significance_params;
  DirectionalMotionParams directional_params;
  directional_params.enable = true;
  directional_params.lambda0 = 0.95;
  directional_params.tau_s = tau_s;
  directional_params.tau_c = tau_c;
  filter.SetParams(imm_params,
                   observability_params,
                   significance_params,
                   directional_params,
                   0.8);
  return filter;
}

std::string MakeTempDir() {
  char path_template[] = "/tmp/deform_runtime_testXXXXXX";
  char* created = ::mkdtemp(path_template);
  EXPECT_NE(created, nullptr);
  return created ? std::string(created) : std::string();
}

}  // namespace

TEST(StageRuntimeLoggerTest, WritesJsonlRecordWithAllFields) {
  const std::string temp_dir = MakeTempDir();
  ASSERT_FALSE(temp_dir.empty());

  StageRuntimeLogger logger;
  ASSERT_TRUE(logger.Initialize(temp_dir + "/runtime"));

  StageRuntimeRecord record;
  record.stamp = ros::Time(12, 345000000);
  record.frame_index = 7;
  record.reference_epoch = 3;
  record.total_ms = 4.5;
  record.stage_a_ms = 1.1;
  record.stage_b_ms = 1.2;
  record.stage_c_ms = 1.3;
  record.stage_d_ms = 0.9;
  record.input_frame_count = 2;
  record.input_point_count = 42000;
  record.input_point_counts_by_frame = {21000, 21000};
  record.anchor_count = 310;
  record.comparable_anchor_count = 240;
  record.significant_anchor_count = 17;
  record.cluster_count = 4;
  record.risk_evidence_count = 15;
  record.risk_voxel_count = 26;
  record.risk_region_count = 3;
  record.persistent_track_count = 2;

  ASSERT_TRUE(logger.Write(record));

  std::ifstream input(logger.log_path().c_str());
  ASSERT_TRUE(input.good());
  std::string line;
  ASSERT_TRUE(static_cast<bool>(std::getline(input, line)));
  EXPECT_NE(line.find("\"frame_index\":7"), std::string::npos);
  EXPECT_NE(line.find("\"reference_epoch\":3"), std::string::npos);
  EXPECT_NE(line.find("\"total_ms\":4.500"), std::string::npos);
  EXPECT_NE(line.find("\"stage_a_ms\":1.100"), std::string::npos);
  EXPECT_NE(line.find("\"stage_b_ms\":1.200"), std::string::npos);
  EXPECT_NE(line.find("\"stage_c_ms\":1.300"), std::string::npos);
  EXPECT_NE(line.find("\"stage_d_ms\":0.900"), std::string::npos);
  EXPECT_NE(line.find("\"input_frame_count\":2"), std::string::npos);
  EXPECT_NE(line.find("\"input_point_count\":42000"), std::string::npos);
  EXPECT_NE(line.find("\"input_point_counts_by_frame\":[21000,21000]"),
            std::string::npos);
  EXPECT_NE(line.find("\"anchor_count\":310"), std::string::npos);
  EXPECT_NE(line.find("\"comparable_anchor_count\":240"), std::string::npos);
  EXPECT_NE(line.find("\"significant_anchor_count\":17"), std::string::npos);
  EXPECT_NE(line.find("\"cluster_count\":4"), std::string::npos);
  EXPECT_NE(line.find("\"risk_evidence_count\":15"), std::string::npos);
  EXPECT_NE(line.find("\"risk_voxel_count\":26"), std::string::npos);
  EXPECT_NE(line.find("\"risk_region_count\":3"), std::string::npos);
  EXPECT_NE(line.find("\"persistent_track_count\":2"), std::string::npos);
}

TEST(ScopedWallTimerTest, AccumulatesElapsedMillisecondsOnScopeExit) {
  double measured_ms = 0.0;
  {
    ScopedWallTimer timer(&measured_ms);
    ::usleep(2000);
  }
  EXPECT_GT(measured_ms, 0.1);
}

TEST(DeformMonitorV2NodeRuntimeTest, WritesStageRuntimeRecordToConfiguredOutputDir) {
  EnsureRosInitialized();
  if (!ros::master::check()) {
    GTEST_SKIP() << "roscore is required for the node runtime writer test";
  }
  const std::string temp_dir = MakeTempDir();
  ASSERT_FALSE(temp_dir.empty());

  ros::NodeHandle private_nh("~");
  private_nh.setParam("deform_monitor/runtime/output_dir", temp_dir + "/runtime");

  DeformMonitorV2Node node;
  StageRuntimeRecord record;
  record.stamp = ros::Time(21, 500000000);
  record.frame_index = 3;
  record.total_ms = 8.0;
  record.stage_a_ms = 1.0;
  record.stage_b_ms = 2.0;
  record.stage_c_ms = 3.0;
  record.stage_d_ms = 2.0;

  node.WriteStageRuntimeRecord(record);

  std::ifstream input((temp_dir + "/runtime/stage_runtime.jsonl").c_str());
  ASSERT_TRUE(input.good());
  std::string line;
  ASSERT_TRUE(static_cast<bool>(std::getline(input, line)));
  EXPECT_NE(line.find("\"frame_index\":3"), std::string::npos);
  EXPECT_NE(line.find("\"total_ms\":8.000"), std::string::npos);
}

TEST(ImmInformationFilterAblationTest, DisableTypeConstraintStopsSuppressedDirectionPenalty) {
  AnchorReference anchor = MakePlaneAnchor();
  CurrentObservation observation = MakeNormalObservation(0.0, 1.0e-4);

  AnchorTrackState constrained_state;
  constrained_state.id = anchor.id;
  constrained_state.type = anchor.type;
  ImmInformationFilter constrained_filter = MakeFilter(true, false, true, true);
  constrained_filter.InitializeAnchorState(&constrained_state);
  constrained_state.model1.x(0) = 0.05;
  constrained_state.model1.mu = 1.0;
  constrained_state.model0.mu = 0.0;
  constrained_filter.Update(&constrained_state, anchor, observation);

  AnchorTrackState unconstrained_state;
  unconstrained_state.id = anchor.id;
  unconstrained_state.type = anchor.type;
  ImmInformationFilter unconstrained_filter = MakeFilter(false, false, true, true);
  unconstrained_filter.InitializeAnchorState(&unconstrained_state);
  unconstrained_state.model1.x(0) = 0.05;
  unconstrained_state.model1.mu = 1.0;
  unconstrained_state.model0.mu = 0.0;
  unconstrained_filter.Update(&unconstrained_state, anchor, observation);

  EXPECT_LT(std::abs(constrained_state.x_mix(0)), std::abs(unconstrained_state.x_mix(0)));
  EXPECT_LT(std::abs(constrained_state.x_mix(0)), 0.03);
  EXPECT_GT(std::abs(unconstrained_state.x_mix(0)), 0.03);
}

TEST(ImmInformationFilterAblationTest, SingleModelModePinsOutputToActiveModel) {
  AnchorReference anchor = MakePlaneAnchor();
  CurrentObservation observation = MakeNormalObservation(0.02, 1.0e-4);

  AnchorTrackState state;
  state.id = anchor.id;
  state.type = anchor.type;

  ImmInformationFilter filter = MakeFilter(true, false, true, true);
  filter.InitializeAnchorState(&state);
  filter.Predict(&state, 0.1);
  filter.Update(&state, anchor, observation);

  EXPECT_DOUBLE_EQ(state.model0.mu, 0.0);
  EXPECT_DOUBLE_EQ(state.model1.mu, 1.0);
  EXPECT_TRUE(state.x_mix.isApprox(state.model1.x, 1.0e-9));
  EXPECT_TRUE(state.P_mix.isApprox(state.model1.P, 1.0e-9));
}

TEST(ImmInformationFilterAblationTest, DisableCusumAndDirectionalKeepsPersistenceOff) {
  AnchorReference anchor = MakePlaneAnchor();

  AnchorTrackState state;
  state.id = anchor.id;
  state.type = anchor.type;
  state.comparable = true;
  state.observable = true;
  state.dof_obs = 1;
  state.chi2_stat = 100.0;
  state.x_mix(2) = 0.05;

  ImmInformationFilter filter = MakeFilter(true, true, false, false);
  filter.InitializeAnchorState(&state);
  state.comparable = true;
  state.observable = true;
  state.dof_obs = 1;
  state.chi2_stat = 100.0;
  state.x_mix(2) = 0.05;

  filter.UpdateCusum(&state);
  filter.UpdateDirectionalMotion(&state, anchor, 1.0, 1.0);

  EXPECT_DOUBLE_EQ(state.cusum_score, 0.0);
  EXPECT_FALSE(state.persistent_candidate);
  EXPECT_TRUE(state.directional_S.isZero(1.0e-12));
  EXPECT_DOUBLE_EQ(state.directional_quality_sum, 0.0);
  EXPECT_FALSE(state.directional_persistent);
}

TEST(ImmInformationFilterDirectionalTest, ConsistentCentimeterMotionPassesDimensionlessCoherence) {
  const AnchorReference anchor = MakePlaneAnchor();
  ImmInformationFilter filter = MakeDirectionalFilter(0.05, 0.65);
  AnchorTrackState state;
  filter.InitializeAnchorState(&state);
  state.comparable = true;

  for (int i = 0; i < 3; ++i) {
    state.x_mix.block<3, 1>(0, 0) = 0.03 * Eigen::Vector3d::UnitZ();
    filter.UpdateDirectionalMotion(&state, anchor, 1.0, 1.0);
  }

  EXPECT_TRUE(state.directional_persistent);
}

TEST(ImmInformationFilterDirectionalTest, AccumulationStrengthIsStableAcrossSamplingRates) {
  const AnchorReference anchor = MakePlaneAnchor();
  ImmInformationFilter filter = MakeDirectionalFilter(0.01, 0.0);
  AnchorTrackState slow_state;
  AnchorTrackState fast_state;
  filter.InitializeAnchorState(&slow_state);
  filter.InitializeAnchorState(&fast_state);
  slow_state.comparable = true;
  fast_state.comparable = true;

  for (int i = 0; i < 5; ++i) {
    slow_state.x_mix.block<3, 1>(0, 0) = 0.03 * Eigen::Vector3d::UnitZ();
    filter.UpdateDirectionalMotion(&slow_state, anchor, 1.0, 1.0);
  }
  for (int i = 0; i < 50; ++i) {
    fast_state.x_mix.block<3, 1>(0, 0) = 0.03 * Eigen::Vector3d::UnitZ();
    filter.UpdateDirectionalMotion(&fast_state, anchor, 1.0, 0.1);
  }

  EXPECT_NEAR(slow_state.directional_S.norm(), fast_state.directional_S.norm(), 0.01);
  EXPECT_NEAR(slow_state.directional_magnitude_sum,
              fast_state.directional_magnitude_sum, 0.01);
}

TEST(ImmInformationFilterDirectionalTest, AlternatingMotionCancelsInsteadOfReinforcing) {
  const AnchorReference anchor = MakePlaneAnchor();
  ImmInformationFilter filter = MakeDirectionalFilter(0.04, 0.0);
  AnchorTrackState state;
  filter.InitializeAnchorState(&state);
  state.comparable = true;

  state.x_mix.block<3, 1>(0, 0) = 0.03 * Eigen::Vector3d::UnitZ();
  filter.UpdateDirectionalMotion(&state, anchor, 1.0, 1.0);
  state.x_mix.block<3, 1>(0, 0) = -0.03 * Eigen::Vector3d::UnitZ();
  filter.UpdateDirectionalMotion(&state, anchor, 1.0, 1.0);

  EXPECT_FALSE(state.directional_persistent);
  EXPECT_LT(state.directional_S.norm(), 0.01);
}

TEST(RiskEvidenceAdapterTest, UnconfirmedHighDisplacementDoesNotBypassFinalDecision) {
  RiskEvidenceAdapter adapter;
  RiskVisualizationParams risk_params;
  risk_params.min_confidence = 0.0;
  risk_params.min_risk_score = 0.0;
  SignificanceParams significance_params;
  significance_params.tau_A_norm = 0.01;
  significance_params.tau_A_normal = 0.01;
  significance_params.tau_A_edge = 0.01;
  GraphTemporalParams graph_params;
  adapter.SetParams(risk_params, significance_params, graph_params);

  AnchorReferenceVector anchors(1, MakePlaneAnchor());
  anchors[0].ref_quality = 1.0;
  anchors[0].covariance_quality = 1.0;
  anchors[0].type_stability = 1.0;
  anchors[0].object_id = 31;
  anchors[0].object_id_valid = true;
  anchors[0].object_id_confidence = 0.96;
  AnchorStateVector states(1);
  states[0].observable = true;
  states[0].comparable = true;
  states[0].gate_state = ObsGateState::OBSERVABLE_MATCHED;
  states[0].disp_norm = 0.10;
  states[0].disp_normal = 0.10;
  states[0].chi2_stat = 100.0;
  states[0].significant = false;
  states[0].mode = DetectionMode::NONE;
  CurrentObservationVector observations(1);
  observations[0].support_count = 10;

  const RiskEvidenceVector evidence =
      adapter.Build(anchors, states, observations, MotionClusterVector());

  ASSERT_EQ(evidence.size(), 1u);
  EXPECT_GT(evidence[0].displacement_score, 0.25);
  EXPECT_FALSE(evidence[0].active);
}

TEST(RiskEvidenceAdapterTest, FinalSignificantDecisionActivatesEligibleEvidence) {
  RiskEvidenceAdapter adapter;
  RiskVisualizationParams risk_params;
  risk_params.min_confidence = 0.0;
  risk_params.min_risk_score = 0.0;
  SignificanceParams significance_params;
  significance_params.tau_A_norm = 0.01;
  significance_params.tau_A_normal = 0.01;
  significance_params.tau_A_edge = 0.01;
  GraphTemporalParams graph_params;
  adapter.SetParams(risk_params, significance_params, graph_params);

  AnchorReferenceVector anchors(1, MakePlaneAnchor());
  anchors[0].ref_quality = 1.0;
  anchors[0].covariance_quality = 1.0;
  anchors[0].type_stability = 1.0;
  anchors[0].object_id = 31;
  anchors[0].object_id_valid = true;
  anchors[0].object_id_confidence = 0.96;
  AnchorStateVector states(1);
  states[0].observable = true;
  states[0].comparable = true;
  states[0].gate_state = ObsGateState::OBSERVABLE_MATCHED;
  states[0].disp_norm = 0.10;
  states[0].disp_normal = 0.10;
  states[0].chi2_stat = 100.0;
  states[0].significant = true;
  states[0].mode = DetectionMode::DISPLACEMENT;
  CurrentObservationVector observations(1);
  observations[0].support_count = 10;

  const RiskEvidenceVector evidence =
      adapter.Build(anchors, states, observations, MotionClusterVector());

  ASSERT_EQ(evidence.size(), 1u);
  EXPECT_TRUE(evidence[0].active);
  EXPECT_EQ(evidence[0].object_id, 31u);
  EXPECT_TRUE(evidence[0].object_id_valid);
  EXPECT_DOUBLE_EQ(evidence[0].object_id_confidence, 0.96);
}

TEST(RiskObjectAssociationTest, PropagatesOneObjectAndFlagsCrossObjectMerge) {
  RiskVisualizationParams risk_params;
  risk_params.voxel_size = 0.05;
  risk_params.kernel_sigma = 0.05;
  risk_params.kernel_radius = 0.05;
  risk_params.min_confidence = 0.0;
  risk_params.min_graph_neighbors = 0;
  risk_params.min_risk_score = 0.0;
  risk_params.min_voxel_risk = 0.0;
  risk_params.min_region_voxels = 1;
  risk_params.min_region_mean_risk = 0.0;

  AnchorReference first_anchor = MakePlaneAnchor();
  first_anchor.center_R = Eigen::Vector3d(0.001, 0.001, 0.001);
  AnchorReference second_anchor = first_anchor;
  second_anchor.id = 8;
  second_anchor.center_R = Eigen::Vector3d(0.002, 0.001, 0.001);
  const AnchorReferenceVector anchors{first_anchor, second_anchor};

  RiskEvidenceState first;
  first.id = 7;
  first.position_R = first_anchor.center_R;
  first.object_id = 31;
  first.object_id_valid = true;
  first.object_id_confidence = 0.95;
  first.confidence = 1.0;
  first.risk_score = 1.0;
  first.displacement_score = 1.0;
  first.active = true;
  RiskEvidenceState second = first;
  second.id = 8;
  second.position_R = second_anchor.center_R;

  RiskFieldBuilder builder;
  builder.SetParams(risk_params);
  const auto one_object_voxels = builder.Build(
      anchors, RiskEvidenceVector{first, second});
  const auto one_object_regions = builder.ExtractRegions(one_object_voxels);

  ASSERT_EQ(one_object_regions.size(), 1u);
  EXPECT_EQ(one_object_regions[0].object_id, 31u);
  EXPECT_TRUE(one_object_regions[0].object_id_valid);
  EXPECT_FALSE(one_object_regions[0].object_id_ambiguous);

  PersistentRiskRegionTracker tracker;
  PersistentRiskParams persistent_params;
  persistent_params.enable = true;
  tracker.SetParams(persistent_params);
  const auto tracks = tracker.Update(one_object_regions, ros::Time(1, 0));
  ASSERT_EQ(tracks.size(), 1u);
  EXPECT_EQ(tracks[0].object_id, 31u);
  EXPECT_TRUE(tracks[0].object_id_valid);
  EXPECT_FALSE(tracks[0].object_id_ambiguous);

  RiskVisualizationPublisher publisher;
  const auto message = publisher.BuildPersistentRiskRegionsMsg(
      tracks, ros::Time(1, 0), "camera_init", 2);
  ASSERT_EQ(message.regions.size(), 1u);
  EXPECT_EQ(message.regions[0].object_id, 31u);
  EXPECT_TRUE(message.regions[0].object_id_valid);

  second.object_id = 32;
  const auto mixed_voxels = builder.Build(
      anchors, RiskEvidenceVector{first, second});
  const auto mixed_regions = builder.ExtractRegions(mixed_voxels);
  ASSERT_EQ(mixed_regions.size(), 1u);
  EXPECT_FALSE(mixed_regions[0].object_id_valid);
  EXPECT_TRUE(mixed_regions[0].object_id_ambiguous);
}

TEST(DeformMonitorV2NodeAblationTest, ResetReferencePublishesEmptyOutputsWhenAblationsAreEnabled) {
  EnsureRosInitialized();
  if (!ros::master::check()) {
    GTEST_SKIP() << "roscore is required for the node publication smoke test";
  }

  ros::NodeHandle private_nh("~");
  private_nh.deleteParam("deform_monitor");
  private_nh.setParam("deform_monitor/risk_visualization/enable", true);
  private_nh.setParam("deform_monitor/risk_visualization/publish_evidence", true);
  private_nh.setParam("deform_monitor/risk_visualization/publish_regions", true);
  private_nh.setParam("deform_monitor/risk_visualization/publish_markers", true);
  private_nh.setParam("deform_monitor/risk_visualization/publish_voxels", true);
  private_nh.setParam("deform_monitor/persistent_risk/enable", true);
  private_nh.setParam("deform_monitor/structure_correspondence/enable", true);
  private_nh.setParam("deform_monitor/structure_correspondence/publish_motions", true);
  private_nh.setParam("deform_monitor/structure_correspondence/publish_markers", true);

  private_nh.setParam("deform_monitor/ablation/variant", "smoke_all_disabled");
  private_nh.setParam("deform_monitor/ablation/disable_covariance_inflation", true);
  private_nh.setParam("deform_monitor/ablation/disable_type_constraint", true);
  private_nh.setParam("deform_monitor/ablation/single_model_ekf", true);
  private_nh.setParam("deform_monitor/ablation/disable_cusum", true);
  private_nh.setParam("deform_monitor/ablation/disable_directional_accumulation", true);
  private_nh.setParam("deform_monitor/ablation/disable_drift_compensation", true);

  DeformMonitorV2Node node;
  const uint32_t previous_epoch = node.reference_epoch_id_;
  node.reference_initialized_stamp_ = ros::Time(10, 0);
  EXPECT_NO_FATAL_FAILURE(node.ResetReferenceCallback(std_msgs::EmptyConstPtr()));
  EXPECT_EQ(node.reference_epoch_id_, previous_epoch + 1u);
  EXPECT_TRUE(node.reference_initialized_stamp_.isZero());
}

}  // namespace deform_monitor_v2

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  if (!ros::isInitialized()) {
    ros::init(argc, argv, "deform_monitor_v2_ablation_test", ros::init_options::AnonymousName);
  }
  return RUN_ALL_TESTS();
}
