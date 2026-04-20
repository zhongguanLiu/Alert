/*
 * Author: zgliu@cumt.edu.cn
 * Affiliation: China University of Mining and Technology
 * Open-source release date: 2026-04-20
 */

#include <gtest/gtest.h>

#include <sys/stat.h>
#include <unistd.h>

#include <fstream>
#include <string>

#include <ros/master.h>
#include <ros/ros.h>
#include <std_msgs/Empty.h>

#include "deform_monitor_v2/core/imm_information_filter.hpp"
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
  record.total_ms = 4.5;
  record.stage_a_ms = 1.1;
  record.stage_b_ms = 1.2;
  record.stage_c_ms = 1.3;
  record.stage_d_ms = 0.9;

  ASSERT_TRUE(logger.Write(record));

  std::ifstream input(logger.log_path().c_str());
  ASSERT_TRUE(input.good());
  std::string line;
  ASSERT_TRUE(static_cast<bool>(std::getline(input, line)));
  EXPECT_NE(line.find("\"frame_index\":7"), std::string::npos);
  EXPECT_NE(line.find("\"total_ms\":4.500"), std::string::npos);
  EXPECT_NE(line.find("\"stage_a_ms\":1.100"), std::string::npos);
  EXPECT_NE(line.find("\"stage_b_ms\":1.200"), std::string::npos);
  EXPECT_NE(line.find("\"stage_c_ms\":1.300"), std::string::npos);
  EXPECT_NE(line.find("\"stage_d_ms\":0.900"), std::string::npos);
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
  EXPECT_NO_FATAL_FAILURE(node.ResetReferenceCallback(std_msgs::EmptyConstPtr()));
}

}  // namespace deform_monitor_v2

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  if (!ros::isInitialized()) {
    ros::init(argc, argv, "deform_monitor_v2_ablation_test", ros::init_options::AnonymousName);
  }
  return RUN_ALL_TESTS();
}
