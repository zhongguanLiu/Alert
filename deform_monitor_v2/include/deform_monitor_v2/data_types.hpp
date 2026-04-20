/*
 * Author: zgliu@cumt.edu.cn
 * Affiliation: China University of Mining and Technology
 * Open-source release date: 2026-04-20
 */

#ifndef DEFORM_MONITOR_V2_DATA_TYPES_HPP
#define DEFORM_MONITOR_V2_DATA_TYPES_HPP

#include <Eigen/Dense>
#include <fast_lio/LioOdomCov.h>
#include <nav_msgs/Odometry.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <ros/time.h>
#include <std_msgs/Empty.h>

#include <cstdint>
#include <deque>
#include <string>
#include <unordered_map>
#include <vector>

namespace deform_monitor_v2 {

template <typename T>
using AlignedVector = std::vector<T, Eigen::aligned_allocator<T>>;

template <typename T>
using AlignedDeque = std::deque<T, Eigen::aligned_allocator<T>>;

template <typename K, typename V, typename Hash = std::hash<K>, typename Eq = std::equal_to<K>>
using AlignedUnorderedMap =
    std::unordered_map<K, V, Hash, Eq, Eigen::aligned_allocator<std::pair<const K, V>>>;

enum class AnchorType : uint8_t {
  PLANE = 0,
  EDGE = 1,
  BAND = 2
};

enum class DetectionMode : uint8_t {
  NONE = 0,
  DISPLACEMENT = 1,
  DISAPPEARANCE = 2
};

enum class RiskRegionType : uint8_t {
  NONE = 0,
  DISPLACEMENT_LIKE = 1,
  DISAPPEARANCE_LIKE = 2,
  MIXED = 3
};

enum class PersistentRiskState : uint8_t {
  CANDIDATE = 0,
  CONFIRMED = 1,
  FADING = 2
};

struct PersistentRiskParams {
  bool enable = true;
  double max_center_distance = 0.35;
  double min_bbox_iou = 0.02;
  double max_risk_gap = 0.35;
  double ema_alpha = 0.45;
  int window_size = 5;
  int min_hits_to_confirm = 3;
  int min_hit_streak_to_confirm = 2;
  double min_confirmed_mean_risk = 0.55;
  double min_confirmed_confidence = 0.50;
  double min_confirmed_support_mass = 2.5;
  double min_confirmed_span = 0.10;
  int miss_frames_to_fading = 2;
  int miss_frames_to_delete = 4;
  double fading_risk_floor = 0.40;
  bool allow_sparse_planar_regions = true;
};

enum class RegionHypothesisKind : uint8_t {
  OLD_REGION = 0,
  NEW_REGION = 1
};

enum class StructureMotionType : uint8_t {
  NONE = 0,
  DISPLACEMENT_LINK = 1,
  DISAPPEARANCE_LINK = 2,
  MIXED = 3
};

enum class ObsStatus : uint8_t {
  INVALID_NO_COMPARISON = 0,
  VALID_PARTIAL_OBS = 1,
  VALID_FULL_OBS = 2
};

enum class ObsGateState : uint8_t {
  NOT_OBSERVABLE = 0,
  OBSERVABLE_MATCHED = 1,
  OBSERVABLE_WEAK = 2,
  OBSERVABLE_MISSING = 3,
  OBSERVABLE_REPLACED = 4
};

struct IoParams {
  std::string cloud_topic = "/cloud_registered";
  std::string odom_topic = "/Odometry";
  std::string covariance_topic = "/lio_odom_cov";
  std::string reset_reference_topic = "/deform/reset_reference";
  std::string reference_frame = "camera_init";
  bool cloud_already_in_reference_frame = true;
  std::string odom_cov_layout = "fastlio_rot_pos";
  double default_rot_sigma = 0.02;
  double default_pos_sigma = 0.01;
};

struct ReferenceParams {
  int init_frames = 60;
  double tau_ref_quality = 0.65;
  double tau_cmp_ref = 0.75;
  double tau_ref_stable = 3.0;
  int N_ref_stable = 20;
  double tau_mu0 = 0.8;
};

struct CovarianceParams {
  double alpha_xi = 2.0;
  double sigma_p = 0.003;
};

struct ObservationParams {
  double tau_n_deg = 25.0;
  double tau_v_deg = 20.0;
  double tau_r_ratio = 0.25;
  double tau_cmp = 0.60;
  double current_voxel_size = 0.05;
  double tau_reacquire = 0.52;
  double reacquire_radius = 0.18;
  int max_reacquire_candidates = 48;
  int min_support_scalar = 2;
  double tau_nis_scalar = 6.0;
  bool soft_view_range_gate = false;    // When true, view_angle and range_ratio
                                        // become soft factors in cmp_score instead
                                        // of hard reject gates. Designed for mobile
                                        // platform monitoring where the sensor moves
                                        // after reference initialization.
  double hard_view_deg_cap = 60.0;      // Absolute hard cap when soft gate is on
  double hard_range_ratio_cap = 0.80;   // Absolute hard cap when soft gate is on
};

struct TemporalFusionParams {
  bool enable = true;
  int window_frames = 5;
  int min_frames = 5;
  int step_frames = 5;
  double max_window_sec = 0.60;
  double pose_corr_inflation = 1.0;
  double sigma_motion_per_sec = 0.010;
  bool use_stable_voxel_fusion = true;
  int min_fused_visible_frames = 2;
  int min_fused_points_per_voxel = 3;
};

struct BackgroundBiasParams {
  bool enable = true;
  int min_anchor_count = 12;
  int min_scalar_count = 24;
  int min_support_points = 5;
  double min_cmp_score = 0.80;
  double max_anchor_disp = 0.03;
  double ridge_lambda = 1.0e-6;
  double huber_delta = 0.008;
  double max_bias_norm = 0.04;
  double adaptive_disp_step1 = 0.05;
  double adaptive_disp_step2 = 0.08;
  double ema_alpha = 0.4;
  double min_stable_ratio = 0.40;
};

struct LocalContrastParams {
  bool enable = true;
  double radius = 0.28;
  int min_neighbors = 6;
  bool enable_plane_background_for_edges = true;
  int min_plane_neighbors = 2;
  int min_support_points = 4;
  double min_cmp_score = 0.78;
  double tau_bg_normal_deg = 35.0;
  double max_background_disp = 0.03;
  double min_background_sigma = 0.004;
  double tau_rel_norm = 0.012;
  double tau_rel_normal = 0.006;
  double tau_rel_edge = 0.008;
  double tau_contrast_score = 2.5;
  double tau_plane_contrast_score = 1.8;
};

struct GraphTemporalParams {
  bool enable = true;
  double spatial_hash_size = 0.05;
  double radius = 0.14;
  int min_neighbors = 2;
  double tau_normal_deg = 40.0;
  double tau_disp_cos = 0.55;
  double tau_coherent_diff = 0.012;
  double tau_temporal_change = 0.006;
  double min_anchor_disp = 0.008;
  double ema_alpha = 0.80;
  double cusum_k = 0.004;
  double cusum_h = 0.040;

  double cusum_lambda = 0.95;
  double tau_graph_support = 0.55;
  double tau_graph_temporal = 0.60;
};

struct AnchorBuildParams {
  double I_min = 20.0;
  double beta_edge = 1.0;
  double beta_depth = 1.0;
  double beta_normal = 0.5;
  double beta_view = 0.5;
  double tau_ref_quality = 0.65;
  double voxel_size = 0.05;
  int min_visible_frames = 3;
  int min_points_per_voxel = 4;
  int neighborhood_layers = 1;
  double seed_voxel = 0.02;
  double radius_min = 0.015;
  int min_support_points = 5;
  double edge_ref_bonus = 0.18;
  double band_ref_bonus = 0.10;
};

struct NoiseParams {
  double sigma_pi0 = 0.003;
  double sigma_edge0 = 0.006;
  double sigma_rad0 = 0.004;
  double sigma_bc0 = 0.006;
  double kappa_r = 0.25;
  double kappa_v = 0.02;
};

struct ObservabilityParams {
  double tau_lambda = 2500.0;
  double tau_sigma_max = 0.05;
};

struct ImmParams {
  bool enable_model_competition = true;
  bool enable_type_constraint = true;
  double rho = 0.90;
  double p00 = 0.98;
  double p01 = 0.02;
  double p10 = 0.05;
  double p11 = 0.95;
  double q_u0 = 9.0e-8;
  double q_v0 = 1.0e-6;
  double q_u1 = 6.4e-7;
  double q_v1 = 9.0e-6;
};

struct SignificanceParams {
  bool enable_cusum = true;
  double alpha_s = 0.01;
  double tau_A_norm = 0.010;
  double tau_A_normal = 0.006;
  double tau_A_edge = 0.008;
  double tau_disappear = 0.70;
  int disappear_frames = 3;
  double cusum_k = 0.5;
  double cusum_h = 5.0;

  double cusum_lambda = 0.95;

  double cusum_cap_factor = 3.0;
};

struct DirectionalMotionParams {
  bool enable = true;
  double lambda0 = 0.85;
  double tau_s = 0.015;
  double tau_c = 0.40;
  double tau_d = 3.0;
  double epsilon = 1.0e-4;
};

struct AblationParams {
  std::string variant = "full_pipeline";
  bool disable_covariance_inflation = false;
  bool disable_type_constraint = false;
  bool single_model_ekf = false;
  bool disable_cusum = false;
  bool disable_directional_accumulation = false;
  bool disable_drift_compensation = false;
};

struct ClusterParams {
  double spatial_hash_size = 0.05;
  double tau_d = 0.20;
  double tau_ng_deg = 30.0;
  double tau_u_cos = 0.7;
  double tau_corr = 0.5;
  bool enable_compact_motion = true;
  double compact_tau_d = 0.16;
  double compact_tau_ng_deg = 65.0;
  double compact_tau_u_cos = 0.25;
  double compact_tau_corr = 0.15;
  double compact_seed_disp = 0.012;
  double compact_seed_chi2 = 15.0;
  int compact_min_cluster_size = 3;
  double compact_tau_cluster_disp = 0.015;
  double compact_tau_cluster_chi2 = 40.0;
  double compact_max_bbox_diag = 0.45;
  double beta_dir = 0.35;
  double beta_mag = 0.20;
  double beta_dist = 0.20;
  double beta_time = 0.25;
  double sigma_m = 0.01;
  double sigma_d = 0.15;
  double tau_edge_score = 0.60;
  int min_cluster_size = 3;
  int min_cluster_size_disappear = 2;
  double tau_cluster_disp = 0.012;
  double tau_cluster_chi2 = 9.21;
  double tau_cluster_disappear = 0.72;
};

struct StructureCorrespondenceParams {
  bool enable = false;
  bool publish_motions = true;
  bool publish_markers = true;
  int old_min_anchor_count = 2;
  int new_min_anchor_count = 2;
  double max_match_distance = 0.60;
  double max_size_gap = 0.35;
  double max_normal_deg = 45.0;
  double max_type_l1 = 1.20;
  double max_match_cost = 1.20;
  double min_confidence = 0.25;
  double min_motion_distance = 0.02;
  double weight_dist = 0.35;
  double weight_size = 0.15;
  double weight_normal = 0.15;
  double weight_type = 0.10;
  double weight_motion = 0.15;
  double weight_persistence = 0.10;
  double old_score_threshold = 0.55;
  double new_disp_threshold = 0.01;
  double marker_arrow_scale = 0.018;
  double marker_outline_width = 0.012;
  double marker_old_alpha = 0.85;
  double marker_new_alpha = 0.85;
  double marker_arrow_alpha = 0.95;
};

struct VisualizationParams {
  bool show_all_anchors = false;
  bool show_comparable_anchors = false;
  bool show_cluster_boxes = false;
  bool show_cluster_text = false;
  bool text_only_significant = true;
  bool arrows_only_clustered_or_reacquired = true;
  double min_arrow_disp = 0.015;
  double min_arrow_contrast_score = 3.0;
  double max_arrow_disp = 0.08;
  double arrow_disp_scale = 10.0;
  double arrow_shaft_diameter = 0.012;
  double arrow_head_diameter = 0.022;
  double arrow_head_length = 0.030;
  double cluster_box_alpha = 0.30;
  double cluster_outline_alpha = 0.95;
  double cluster_outline_width = 0.012;
  double cluster_min_box_size = 0.06;
};

struct RiskVisualizationParams {
  bool enable = false;
  bool publish_evidence = true;
  bool publish_voxels = true;
  bool publish_regions = true;
  bool publish_markers = true;
  double voxel_size = 0.05;
  double kernel_sigma = 0.08;
  double kernel_radius = 0.12;
  double min_confidence = 0.35;
  int min_graph_neighbors = 2;
  double min_risk_score = 0.35;
  double min_voxel_risk = 0.25;
  int min_region_voxels = 4;
  double min_region_mean_risk = 0.40;
  double low_risk_alpha = 0.12;
  double high_risk_alpha = 0.85;
  double region_outline_alpha = 0.95;
  double region_outline_width = 0.018;
};

struct StructureUnitParams {
  double region_spatial_hash = 0.10;
  double region_radius = 0.20;
  double region_normal_deg = 45.0;
  double region_edge_dir_deg = 30.0;
  int region_min_members = 3;
};

struct StructureUnitTrackerParams {
  double tau_exit = 0.5;
  double search_margin = 0.5;
  double orphan_radius = 0.08;
  double orphan_voxel_size = 0.05;
  double max_size_gap = 0.40;
  double max_normal_deg = 45.0;
  double min_migration_dist = 0.02;
  double max_migration_dist = 1.0;
  double min_migration_confidence = 0.5;
  int orphan_min_points = 5;
  double persistence_confidence = 0.75;
};

struct IncrementalParams {
  bool enable = true;
  double coverage_radius = 0.08;
  int warmup_frames = 6;
  int min_visible_frames = 3;
  int max_new_anchors = 500;
};

struct DeformMonitorParams {
  IoParams io;
  ReferenceParams reference;
  CovarianceParams covariance;
  ObservationParams observation;
  TemporalFusionParams temporal;
  BackgroundBiasParams background_bias;
  LocalContrastParams local_contrast;
  GraphTemporalParams graph_temporal;
  AnchorBuildParams anchor;
  NoiseParams noise;
  ObservabilityParams observability;
  ImmParams imm;
  SignificanceParams significance;
  DirectionalMotionParams directional_motion;
  AblationParams ablation;
  ClusterParams cluster;
  StructureUnitParams structure_unit;
  StructureUnitTrackerParams structure_tracker;
  StructureCorrespondenceParams structure_correspondence;
  VisualizationParams visualization;
  RiskVisualizationParams risk_visualization;
  PersistentRiskParams persistent_risk;
  IncrementalParams incremental;
};

struct PoseCov6D {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Eigen::Matrix<double, 6, 6> Sigma_xi = Eigen::Matrix<double, 6, 6>::Identity();
  ros::Time stamp;
};

struct ScalarMeasurement {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Eigen::Vector3d h_R = Eigen::Vector3d::UnitZ();
  double z = 0.0;
  double r = 1.0;
  uint8_t type = 0;
};

struct AnchorReference {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  int id = -1;
  AnchorType type = AnchorType::PLANE;
  Eigen::Vector3d center_R = Eigen::Vector3d::Zero();
  Eigen::Matrix3d basis_R = Eigen::Matrix3d::Identity();
  Eigen::Vector3d normal_R = Eigen::Vector3d::UnitZ();
  Eigen::Vector3d edge_normal_R = Eigen::Vector3d::UnitX();
  Eigen::Matrix3d Sigma_ref_geom = Eigen::Matrix3d::Identity() * 1.0e-4;
  AlignedVector<Eigen::Vector3d> support_points_R;
  double ref_quality = 0.0;
  double mean_range = 0.0;
  Eigen::Vector3d mean_view_dir_R = Eigen::Vector3d::UnitX();
  double mean_incidence_cos = 1.0;
  Eigen::Vector3d edge_center_R = Eigen::Vector3d::Zero();
  Eigen::Vector3d band_center_R = Eigen::Vector3d::Zero();
  double support_radius = 0.0;
  int support_target_count = 0;
  int point_count = 0;
  int visible_count = 0;
  int matched_count = 0;
  double covariance_quality = 0.0;
  double type_stability = 0.0;
  bool frozen = false;
  std::vector<int> neighbor_indices;
};

struct LocalSupportData {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  int anchor_id = -1;
  ros::Time stamp;
  AlignedVector<Eigen::Vector3d> support_points_R;
  AlignedVector<Eigen::Matrix3d> point_covariances;
  Eigen::Vector3d centroid_R = Eigen::Vector3d::Zero();
  Eigen::Matrix3d centroid_cov = Eigen::Matrix3d::Identity() * 1.0e-4;
  Eigen::Vector3d edge_centroid_R = Eigen::Vector3d::Zero();
  Eigen::Matrix3d edge_centroid_cov = Eigen::Matrix3d::Identity() * 1.0e-4;
  Eigen::Vector3d band_centroid_R = Eigen::Vector3d::Zero();
  Eigen::Matrix3d band_centroid_cov = Eigen::Matrix3d::Identity() * 1.0e-4;
  Eigen::Matrix3d local_cov = Eigen::Matrix3d::Zero();
  Eigen::Vector3d normal_R = Eigen::Vector3d::UnitZ();
  Eigen::Vector3d edge_normal_R = Eigen::Vector3d::UnitX();
  Eigen::Vector3d band_axis_R = Eigen::Vector3d::UnitY();
  Eigen::Vector3d view_dir_R = Eigen::Vector3d::UnitX();
  double range = 0.0;
  double incidence_cos = 1.0;
  double cmp_score = 0.0;
  double fit_rmse = 0.0;
  double overlap_score = 0.0;
  double normal_angle_deg = 180.0;
  double view_angle_deg = 180.0;
  double range_ratio = 1.0;
  double expected_view_angle_deg = 180.0;
  double expected_range_ratio = 1.0;
  double expected_incidence_cos = 0.0;
  double center_shift_norm = 0.0;
  int support_count = 0;
  ObsStatus status = ObsStatus::INVALID_NO_COMPARISON;
  ObsGateState gate_state = ObsGateState::NOT_OBSERVABLE;
  bool valid = false;
  bool comparable = false;
  bool expected_observable = false;
  bool reacquired = false;
};

struct CurrentObservation {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  int anchor_id = -1;
  ros::Time stamp;
  AlignedVector<ScalarMeasurement> scalars;
  double cmp_score = 0.0;
  ObsStatus status = ObsStatus::INVALID_NO_COMPARISON;
  ObsGateState gate_state = ObsGateState::NOT_OBSERVABLE;
  int support_count = 0;
  double fit_rmse = 0.0;
  double overlap_score = 0.0;
  int dof_obs = 0;
  bool comparable = false;
  bool observable = false;
  bool reacquired = false;
  double disappearance_score = 0.0;
  Eigen::Vector3d matched_center_R = Eigen::Vector3d::Zero();
  Eigen::Vector3d matched_delta_R = Eigen::Vector3d::Zero();
};

struct ModelState {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Eigen::Matrix<double, 6, 1> x = Eigen::Matrix<double, 6, 1>::Zero();
  Eigen::Matrix<double, 6, 6> P = Eigen::Matrix<double, 6, 6>::Identity();
  double mu = 0.5;
};

struct AnchorTrackState {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  int id = -1;
  AnchorType type = AnchorType::PLANE;
  ModelState model0;
  ModelState model1;
  Eigen::Matrix<double, 6, 1> x_mix = Eigen::Matrix<double, 6, 1>::Zero();
  Eigen::Matrix<double, 6, 6> P_mix = Eigen::Matrix<double, 6, 6>::Identity();
  int dof_obs = 0;
  double chi2_stat = 0.0;
  double disp_norm = 0.0;
  double disp_normal = 0.0;
  double disp_edge = 0.0;
  double cusum_score = 0.0;
  double disappearance_score = 0.0;
  bool comparable = false;
  bool observable = false;
  ObsGateState gate_state = ObsGateState::NOT_OBSERVABLE;
  bool significant = false;
  int local_bg_count = 0;
  double local_bg_disp_norm = 0.0;
  double local_bg_sigma = 0.0;
  double local_contrast_score = 0.0;
  double local_rel_norm = 0.0;
  double local_rel_normal = 0.0;
  double local_rel_edge = 0.0;
  int plane_bg_count = 0;
  double plane_bg_disp_norm = 0.0;
  double plane_bg_sigma = 0.0;
  double plane_contrast_score = 0.0;
  double plane_rel_norm = 0.0;
  double plane_rel_normal = 0.0;
  double plane_rel_edge = 0.0;
  int graph_neighbor_count = 0;
  double graph_coherent_score = 0.0;
  double graph_temporal_score = 0.0;
  double graph_persistence_score = 0.0;
  double graph_diff_norm = 0.0;
  bool graph_candidate = false;
  bool persistent_candidate = false;
  bool disappearance_candidate = false;
  bool reacquired = false;
  DetectionMode mode = DetectionMode::NONE;
  int stable_streak = 0;
  int disappearance_streak = 0;
  int dead_count = 0;
  bool cluster_member = false;
  Eigen::Vector3d matched_center_R = Eigen::Vector3d::Zero();
  std::deque<double> cusum_history;
  std::deque<double> evidence_history;
  ros::Time last_update;

  Eigen::Vector3d directional_S = Eigen::Vector3d::Zero();
  double directional_quality_sum = 0.0;
  bool directional_persistent = false;

  Eigen::Vector3d D_max = Eigen::Vector3d::Zero();
  bool permanent_deformed = false;
};

struct MotionClusterState {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  int id = -1;
  std::vector<int> anchor_ids;
  DetectionMode mode = DetectionMode::NONE;
  Eigen::Vector3d center_R = Eigen::Vector3d::Zero();
  Eigen::Vector3d bbox_min_R = Eigen::Vector3d::Zero();
  Eigen::Vector3d bbox_max_R = Eigen::Vector3d::Zero();
  Eigen::Vector3d disp_mean_R = Eigen::Vector3d::Zero();
  Eigen::Matrix3d disp_cov = Eigen::Matrix3d::Identity();
  double chi2_stat = 0.0;
  double disp_norm = 0.0;
  double evidence_score = 0.0;
  double confidence = 0.0;
  int support_count = 0;
  bool significant = false;
};

using AnchorReferenceVector = AlignedVector<AnchorReference>;
using CurrentObservationVector = AlignedVector<CurrentObservation>;
using AnchorStateVector = AlignedVector<AnchorTrackState>;
using MotionClusterVector = AlignedVector<MotionClusterState>;

struct RiskEvidenceState {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  int id = -1;
  AnchorType anchor_type = AnchorType::PLANE;
  ObsGateState obs_state = ObsGateState::NOT_OBSERVABLE;
  DetectionMode mode = DetectionMode::NONE;
  Eigen::Vector3d position_R = Eigen::Vector3d::Zero();
  Eigen::Vector3d displacement_R = Eigen::Vector3d::Zero();
  double displacement_score = 0.0;
  double disappearance_score = 0.0;
  double graph_score = 0.0;
  double confidence = 0.0;
  double risk_score = 0.0;
  int graph_neighbor_count = 0;
  bool observable = false;
  bool comparable = false;
  bool active = false;
};

struct RiskVoxelState {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Eigen::Vector3d center_R = Eigen::Vector3d::Zero();
  double risk_score = 0.0;
  double confidence = 0.0;
  double displacement_component = 0.0;
  double disappearance_component = 0.0;
  int source_count = 0;
  bool significant = false;
};

struct RiskRegionState {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  int id = -1;
  RiskRegionType type = RiskRegionType::NONE;
  Eigen::Vector3d center_R = Eigen::Vector3d::Zero();
  Eigen::Vector3d bbox_min_R = Eigen::Vector3d::Zero();
  Eigen::Vector3d bbox_max_R = Eigen::Vector3d::Zero();
  double mean_risk = 0.0;
  double peak_risk = 0.0;
  double confidence = 0.0;
  int voxel_count = 0;
  bool significant = false;
};

struct PersistentRiskTrackState {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  int track_id = -1;
  PersistentRiskState state = PersistentRiskState::CANDIDATE;
  RiskRegionType region_type = RiskRegionType::NONE;
  Eigen::Vector3d last_center_R = Eigen::Vector3d::Zero();
  Eigen::Vector3d last_bbox_min_R = Eigen::Vector3d::Zero();
  Eigen::Vector3d last_bbox_max_R = Eigen::Vector3d::Zero();
  Eigen::Vector3d union_bbox_min_R = Eigen::Vector3d::Zero();
  Eigen::Vector3d union_bbox_max_R = Eigen::Vector3d::Zero();
  int hit_streak = 0;
  int miss_streak = 0;
  int age_frames = 0;
  int matched_region_count_window = 0;
  bool ever_confirmed = false;
  double ema_mean_risk = 0.0;
  double ema_peak_risk = 0.0;
  double accumulated_risk = 0.0;
  double ema_confidence = 0.0;
  double ema_voxel_count = 0.0;
  double support_mass = 0.0;
  double spatial_span = 0.0;
  RiskRegionType stable_region_type = RiskRegionType::NONE;
  int stable_type_streak = 0;
  int planar_like_streak = 0;
  double prev_support_mass = 0.0;
  double prev_accumulated_risk = 0.0;
  ros::Time last_update;
  std::deque<uint8_t> match_history;
};

using PersistentRiskTrackVector = AlignedVector<PersistentRiskTrackState>;

struct RegionHypothesisState {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  int id = -1;
  RegionHypothesisKind kind = RegionHypothesisKind::OLD_REGION;
  std::vector<int> anchor_ids;
  Eigen::Vector3d center_ref_R = Eigen::Vector3d::Zero();
  Eigen::Vector3d center_curr_R = Eigen::Vector3d::Zero();
  Eigen::Vector3d bbox_ref_min_R = Eigen::Vector3d::Zero();
  Eigen::Vector3d bbox_ref_max_R = Eigen::Vector3d::Zero();
  Eigen::Vector3d bbox_curr_min_R = Eigen::Vector3d::Zero();
  Eigen::Vector3d bbox_curr_max_R = Eigen::Vector3d::Zero();
  Eigen::Vector3d mean_normal_R = Eigen::Vector3d::UnitZ();
  Eigen::Vector3d mean_motion_R = Eigen::Vector3d::Zero();
  Eigen::Vector3d type_histogram = Eigen::Vector3d::Zero();
  double mean_disp_norm = 0.0;
  double mean_disappearance_score = 0.0;
  double mean_graph_score = 0.0;
  double confidence = 0.0;
  double time_persistence = 0.0;
  bool significant = false;
};

struct StructureMotionState {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  int id = -1;
  int old_region_id = -1;
  int new_region_id = -1;
  StructureMotionType type = StructureMotionType::NONE;
  Eigen::Vector3d old_center_R = Eigen::Vector3d::Zero();
  Eigen::Vector3d new_center_R = Eigen::Vector3d::Zero();
  Eigen::Vector3d bbox_old_min_R = Eigen::Vector3d::Zero();
  Eigen::Vector3d bbox_old_max_R = Eigen::Vector3d::Zero();
  Eigen::Vector3d bbox_new_min_R = Eigen::Vector3d::Zero();
  Eigen::Vector3d bbox_new_max_R = Eigen::Vector3d::Zero();
  Eigen::Vector3d motion_R = Eigen::Vector3d::Zero();
  double distance = 0.0;
  double match_cost = 0.0;
  double confidence = 0.0;
  int support_old = 0;
  int support_new = 0;
  bool significant = false;
};

using RiskEvidenceVector = AlignedVector<RiskEvidenceState>;
using RiskVoxelVector = AlignedVector<RiskVoxelState>;
using RiskRegionVector = AlignedVector<RiskRegionState>;
using RegionHypothesisVector = AlignedVector<RegionHypothesisState>;
using StructureMotionVector = AlignedVector<StructureMotionState>;

struct ObservationFrame {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  pcl::PointCloud<pcl::PointXYZI>::ConstPtr cloud;
  PoseCov6D pose_cov;
  Eigen::Vector3d lidar_origin_R = Eigen::Vector3d::Zero();
  ros::Time stamp;
};

struct ReferenceInitFrame {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud;
  Eigen::Vector3d lidar_origin_R = Eigen::Vector3d::Zero();
  PoseCov6D pose_cov;
  ros::Time stamp;
};

using ObservationFrameDeque = AlignedDeque<ObservationFrame>;
using ReferenceInitFrameVector = AlignedVector<ReferenceInitFrame>;

PoseCov6D PoseCovFromOdometry(const nav_msgs::Odometry& odom,
                              const std::string& layout,
                              double default_rot_sigma,
                              double default_pos_sigma);

PoseCov6D PoseCovFromFastLioMsg(const fast_lio::LioOdomCov& msg,
                                double default_rot_sigma,
                                double default_pos_sigma);

Eigen::Isometry3d PoseFromOdometry(const nav_msgs::Odometry& odom);
Eigen::Isometry3d PoseFromFastLioMsg(const fast_lio::LioOdomCov& msg);

Eigen::Matrix3d SkewSymmetric(const Eigen::Vector3d& v);
Eigen::MatrixXd PseudoInverse(const Eigen::MatrixXd& matrix, double eps = 1.0e-9);
double Chi2PseudoInverse(const Eigen::Vector3d& u, const Eigen::Matrix3d& Sigma_u);
double Rad2Deg(double rad);
double Deg2Rad(double deg);
double AngleBetweenDeg(const Eigen::Vector3d& a, const Eigen::Vector3d& b);
Eigen::MatrixXd Symmetrize(const Eigen::MatrixXd& M);

// ─────────────────────────────────────────────

// ─────────────────────────────────────────────

enum class RegionEdgeType : uint8_t { COPLANAR = 0, ADJACENT = 1 };

struct RegionEdge {
  int anchor_id_a   = -1;
  int anchor_id_b   = -1;
  double dist_ref   = 0.0;
  double normal_cos_ref = 0.0;
  RegionEdgeType edge_type = RegionEdgeType::COPLANAR;
};

struct StructureUnit {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  int id = -1;
  std::vector<int> member_ids;
  std::vector<RegionEdge> edge_set;
  AlignedVector<Eigen::Vector3d> ref_pointcloud;
  Eigen::Vector3d ref_centroid_R = Eigen::Vector3d::Zero();
  Eigen::Vector3d ref_bbox_min_R = Eigen::Vector3d::Zero();
  Eigen::Vector3d ref_bbox_max_R = Eigen::Vector3d::Zero();
  Eigen::Vector3d ref_normal_R   = Eigen::Vector3d::UnitZ();
  double mean_ref_quality = 0.0;
};

struct StructureMigration {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  int unit_id = -1;
  Eigen::Vector3d T                = Eigen::Vector3d::Zero();
  Eigen::Vector3d entry_centroid_R = Eigen::Vector3d::Zero();
  Eigen::Vector3d entry_bbox_min_R = Eigen::Vector3d::Zero();
  Eigen::Vector3d entry_bbox_max_R = Eigen::Vector3d::Zero();
  double exit_score  = 0.0;
  double confidence  = 0.0;
  bool confirmed     = false;
  bool persistent    = false;
};

using StructureUnitVector      = AlignedVector<StructureUnit>;
using StructureMigrationVector = AlignedVector<StructureMigration>;

}  // namespace deform_monitor_v2

#endif  // DEFORM_MONITOR_V2_DATA_TYPES_HPP
