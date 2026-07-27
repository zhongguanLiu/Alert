#include <gtest/gtest.h>

#include <Eigen/Eigenvalues>

#include "deform_monitor_v2/core/edge_feature.hpp"
#include "deform_monitor_v2/core/imm_information_filter.hpp"
#include "deform_monitor_v2/core/motion_clusterer.hpp"
#include "deform_monitor_v2/core/observable_subspace.hpp"
#include "deform_monitor_v2/core/scalar_measurement_builder.hpp"

namespace deform_monitor_v2 {
namespace {

AlignedVector<Eigen::Vector3d> MakeEdgePoints() {
  return {
      Eigen::Vector3d(-0.04, -0.02, 0.0),
      Eigen::Vector3d(0.00, -0.02, 0.0),
      Eigen::Vector3d(0.04, -0.02, 0.0),
      Eigen::Vector3d(-0.04, 0.02, 0.0),
      Eigen::Vector3d(0.00, 0.02, 0.0),
      Eigen::Vector3d(0.04, 0.02, 0.0),
  };
}

TEST(EdgeFeatureGeometryTest, UsesTranslationEquivalentPartitionCenter) {
  const AlignedVector<Eigen::Vector3d> reference_points = MakeEdgePoints();
  const Eigen::Vector3d translation(0.03, -0.01, 0.02);
  AlignedVector<Eigen::Vector3d> current_points = reference_points;
  for (auto& point : current_points) {
    point += translation;
  }

  const EdgeFeatureGeometry reference = ComputeEdgeFeatureGeometry(
      reference_points, Eigen::Vector3d::Zero(), Eigen::Vector3d::UnitY());
  const EdgeFeatureGeometry current = ComputeEdgeFeatureGeometry(
      current_points, translation, Eigen::Vector3d::UnitY());

  ASSERT_TRUE(reference.valid);
  ASSERT_TRUE(current.valid);
  ASSERT_EQ(reference.point_indices.size(), current.point_indices.size());
  EXPECT_TRUE((current.center_R - reference.center_R).isApprox(translation, 1.0e-12));
  EXPECT_TRUE(current.sample_cov.isApprox(reference.sample_cov, 1.0e-12));
}

TEST(ScalarMeasurementBuilderTest, EdgeDoesNotCreateTangentOrRadialDof) {
  ObservationParams observation_params;
  observation_params.min_support_scalar = 5;
  observation_params.tau_nis_scalar = 1000.0;

  NoiseParams noise_params;
  ScalarMeasurementBuilder builder;
  builder.SetParams(observation_params, noise_params);

  AnchorReference anchor;
  anchor.type = AnchorType::EDGE;
  anchor.center_R = Eigen::Vector3d::Zero();
  anchor.normal_R = Eigen::Vector3d::UnitZ();
  anchor.edge_normal_R = Eigen::Vector3d::UnitY();
  anchor.basis_R.col(0) = Eigen::Vector3d::UnitY();
  anchor.basis_R.col(1) = Eigen::Vector3d::UnitX();
  anchor.basis_R.col(2) = Eigen::Vector3d::UnitZ();
  anchor.edge_center_R = Eigen::Vector3d(0.0, 0.02, 0.0);
  anchor.Sigma_ref_geom = Eigen::Matrix3d::Identity() * 1.0e-6;
  anchor.Sigma_ref_edge = Eigen::Matrix3d::Identity() * 1.0e-6;

  LocalSupportData support;
  support.valid = true;
  support.support_count = 6;
  support.edge_support_count = 3;
  support.edge_geometry_valid = true;
  support.centroid_R = Eigen::Vector3d(0.04, 0.0, 0.0);
  support.edge_centroid_R = anchor.edge_center_R + Eigen::Vector3d(0.04, 0.0, 0.0);
  support.centroid_cov = Eigen::Matrix3d::Identity() * 1.0e-6;
  support.edge_centroid_cov = Eigen::Matrix3d::Identity() * 1.0e-6;
  support.view_dir_R = Eigen::Vector3d::UnitX();
  support.range = 1.0;
  support.reacquired = true;

  const AlignedVector<ScalarMeasurement> measurements = builder.BuildMeasurements(
      anchor, support, PoseCov6D(), Eigen::Vector3d(-1.0, 0.0, 0.0));

  ASSERT_EQ(measurements.size(), 2u);
  EXPECT_EQ(measurements[0].type, 0u);
  EXPECT_EQ(measurements[1].type, 1u);
  EXPECT_NEAR(measurements[0].z, 0.0, 1.0e-12);
  EXPECT_NEAR(measurements[1].z, 0.0, 1.0e-12);

  Eigen::Matrix3d information = Eigen::Matrix3d::Zero();
  for (const auto& measurement : measurements) {
    information += measurement.h_R * measurement.h_R.transpose();
  }
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(information);
  ASSERT_EQ(eig.info(), Eigen::Success);
  int rank = 0;
  for (int i = 0; i < 3; ++i) {
    rank += eig.eigenvalues()(i) > 1.0e-9 ? 1 : 0;
  }
  EXPECT_EQ(rank, 2);
}

TEST(ScalarMeasurementBuilderTest, ReacquiredPlaneRemainsNormalOnly) {
  ObservationParams observation_params;
  observation_params.min_support_scalar = 3;
  observation_params.tau_nis_scalar = 1000.0;
  ScalarMeasurementBuilder builder;
  builder.SetParams(observation_params, NoiseParams());

  AnchorReference anchor;
  anchor.type = AnchorType::PLANE;
  anchor.normal_R = Eigen::Vector3d::UnitZ();
  anchor.basis_R = Eigen::Matrix3d::Identity();
  anchor.Sigma_ref_geom = Eigen::Matrix3d::Identity() * 1.0e-6;

  LocalSupportData support;
  support.valid = true;
  support.reacquired = true;
  support.support_count = 5;
  support.centroid_R = Eigen::Vector3d(0.08, -0.06, 0.01);
  support.centroid_cov = Eigen::Matrix3d::Identity() * 1.0e-6;
  support.view_dir_R = Eigen::Vector3d::UnitX();

  const auto measurements = builder.BuildMeasurements(
      anchor, support, PoseCov6D(), Eigen::Vector3d::Zero());

  ASSERT_EQ(measurements.size(), 1u);
  EXPECT_EQ(measurements[0].type, 0u);
  EXPECT_TRUE(measurements[0].h_R.isApprox(Eigen::Vector3d::UnitZ(), 1.0e-12));
}

TEST(ScalarMeasurementBuilderTest, BandUsesNormalAndDeclaredBandAxis) {
  ObservationParams observation_params;
  observation_params.min_support_scalar = 3;
  observation_params.tau_nis_scalar = 1000.0;
  ScalarMeasurementBuilder builder;
  builder.SetParams(observation_params, NoiseParams());

  AnchorReference anchor;
  anchor.type = AnchorType::BAND;
  anchor.normal_R = Eigen::Vector3d::UnitZ();
  anchor.basis_R.col(0) = Eigen::Vector3d::UnitY();
  anchor.basis_R.col(1) = Eigen::Vector3d::UnitX();
  anchor.basis_R.col(2) = Eigen::Vector3d::UnitZ();
  anchor.Sigma_ref_geom = Eigen::Matrix3d::Identity() * 1.0e-6;

  LocalSupportData support;
  support.valid = true;
  support.support_count = 5;
  support.centroid_R = Eigen::Vector3d(0.02, 0.04, 0.01);
  support.band_centroid_R = support.centroid_R;
  support.centroid_cov = Eigen::Matrix3d::Identity() * 1.0e-6;
  support.band_centroid_cov = Eigen::Matrix3d::Identity() * 1.0e-6;
  support.view_dir_R = Eigen::Vector3d::UnitX();

  const auto measurements = builder.BuildMeasurements(
      anchor, support, PoseCov6D(), Eigen::Vector3d::Zero());

  ASSERT_EQ(measurements.size(), 2u);
  EXPECT_EQ(measurements[0].type, 0u);
  EXPECT_EQ(measurements[1].type, 3u);
  EXPECT_TRUE(measurements[1].h_R.isApprox(Eigen::Vector3d::UnitX(), 1.0e-12));
}

TEST(ObservableSubspaceTest, ProjectedChiSquareIgnoresSuppressedPlaneTangents) {
  AnchorReference anchor;
  anchor.type = AnchorType::PLANE;
  anchor.normal_R = Eigen::Vector3d::UnitZ();
  const ObservableSubspace subspace = BuildObservableSubspace(anchor);

  EXPECT_EQ(subspace.rank, 1);
  const Eigen::Matrix3d covariance = Eigen::Matrix3d::Identity() * 0.01;
  EXPECT_NEAR(ProjectedChiSquare(Eigen::Vector3d(1.0, 2.0, 0.0),
                                covariance,
                                subspace,
                                1),
              0.0,
              1.0e-12);
  EXPECT_NEAR(ProjectedChiSquare(Eigen::Vector3d(1.0, 2.0, 0.1),
                                covariance,
                                subspace,
                                1),
              1.0,
              1.0e-12);
}

TEST(DirectionalMotionTest, AccumulatesOnlyInstantaneousProjectedEvidence) {
  SignificanceParams significance;
  significance.alpha_s = 0.05;
  significance.tau_A_normal = 0.01;
  DirectionalMotionParams directional;
  directional.enable = true;

  ImmInformationFilter filter;
  filter.SetParams(ImmParams(), ObservabilityParams(), significance, directional, 0.8);

  AnchorReference anchor;
  anchor.type = AnchorType::PLANE;
  anchor.normal_R = Eigen::Vector3d::UnitZ();

  AnchorTrackState state;
  state.comparable = true;
  state.observable = true;
  state.gate_state = ObsGateState::OBSERVABLE_MATCHED;
  state.dof_obs = 1;
  state.x_mix(2) = 0.02;
  state.chi2_stat = 0.5;
  filter.UpdateDirectionalMotion(&state, anchor, 1.0, 1.0);
  EXPECT_FALSE(state.instantaneous_displacement_evidence);
  EXPECT_NEAR(state.directional_S.norm(), 0.0, 1.0e-12);

  state.chi2_stat = 100.0;
  filter.UpdateDirectionalMotion(&state, anchor, 1.0, 1.0);
  EXPECT_TRUE(state.instantaneous_displacement_evidence);
  EXPECT_GT(state.directional_S.norm(), 0.0);
}

TEST(MotionClustererTest, RawPersistenceCannotEnterClustering) {
  AnchorReference anchor;
  anchor.id = 7;
  AnchorTrackState state;
  state.id = 7;
  state.persistent_candidate = true;
  state.directional_persistent = true;
  state.significant = false;

  MotionClusterer clusterer;
  clusterer.SetParams(ClusterParams());
  EXPECT_TRUE(clusterer.Cluster(AnchorReferenceVector{anchor}, AnchorStateVector{state}).empty());
}

}  // namespace
}  // namespace deform_monitor_v2
