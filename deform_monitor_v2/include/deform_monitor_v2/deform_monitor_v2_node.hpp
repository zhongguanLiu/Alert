/*
 * Author: zgliu@cumt.edu.cn
 * Affiliation: China University of Mining and Technology
 * Open-source release date: 2026-04-20
 */

#ifndef DEFORM_MONITOR_V2_NODE_HPP
#define DEFORM_MONITOR_V2_NODE_HPP

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/Empty.h>

#include "deform_monitor_v2/core/anchor_builder.hpp"
#include "deform_monitor_v2/core/covariance_extractor.hpp"
#include "deform_monitor_v2/core/current_observation_extractor.hpp"
#include "deform_monitor_v2/core/imm_information_filter.hpp"
#include "deform_monitor_v2/core/motion_clusterer.hpp"
#include "deform_monitor_v2/core/region_correspondence_solver.hpp"
#include "deform_monitor_v2/core/persistent_risk_region_tracker.hpp"
#include "deform_monitor_v2/core/region_hypothesis_builder.hpp"
#include "deform_monitor_v2/core/risk_evidence_adapter.hpp"
#include "deform_monitor_v2/core/risk_field_builder.hpp"
#include "deform_monitor_v2/core/reference_manager.hpp"
#include "deform_monitor_v2/core/scalar_measurement_builder.hpp"
#include "deform_monitor_v2/data_types.hpp"
#include "deform_monitor_v2/risk_visualization_publisher.hpp"
#include "deform_monitor_v2/structure_visualization_publisher.hpp"
#include "deform_monitor_v2/visualization_publisher.hpp"
#include "deform_monitor_v2/structure_unit_builder.hpp"
#include "deform_monitor_v2/structure_unit_tracker.hpp"

namespace deform_monitor_v2 {

struct StageRuntimeRecord {
  ros::Time stamp;
  uint64_t frame_index = 0;
  double total_ms = 0.0;
  double stage_a_ms = 0.0;
  double stage_b_ms = 0.0;
  double stage_c_ms = 0.0;
  double stage_d_ms = 0.0;
};

class ScopedWallTimer {
public:
  explicit ScopedWallTimer(double* accumulator_ms);
  ~ScopedWallTimer();

  ScopedWallTimer(const ScopedWallTimer&) = delete;
  ScopedWallTimer& operator=(const ScopedWallTimer&) = delete;

private:
  double* accumulator_ms_ = nullptr;
  std::chrono::steady_clock::time_point start_time_;
};

class StageRuntimeLogger {
public:
  StageRuntimeLogger() = default;
  ~StageRuntimeLogger();

  bool Initialize(const std::string& output_dir);
  bool Write(const StageRuntimeRecord& record);
  void Close();

  bool is_initialized() const { return initialized_; }
  const std::string& log_path() const { return log_path_; }

private:
  std::ofstream stream_;
  std::string log_path_;
  bool initialized_ = false;
};

class DeformMonitorV2Node {
public:
  DeformMonitorV2Node();
  ~DeformMonitorV2Node() = default;

  void Run();
  void CloudCallback(const sensor_msgs::PointCloud2ConstPtr& msg);
  void CovarianceCallback(const fast_lio::LioOdomCovConstPtr& msg);
  void ResetReferenceCallback(const std_msgs::EmptyConstPtr& msg);

private:
  struct EdgeKey {
    int a = -1;
    int b = -1;

    bool operator==(const EdgeKey& other) const {
      return a == other.a && b == other.b;
    }
  };

  struct EdgeKeyHash {
    size_t operator()(const EdgeKey& key) const {
      size_t h = std::hash<int>()(key.a);
      h ^= std::hash<int>()(key.b) + 0x9e3779b9 + (h << 6) + (h >> 2);
      return h;
    }
  };

  struct EdgeTemporalState {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Eigen::Vector3d delta_ema = Eigen::Vector3d::Zero();
    double coherent_ema = 0.0;
    double temporal_consistency = 0.0;
    double persistence_score = 0.0;
    int valid_streak = 0;
    ros::Time last_update;
  };

  struct TimedCloud {
    sensor_msgs::PointCloud2 msg;
  };

  struct TimedCov {
    fast_lio::LioOdomCov msg;
  };

  struct LoggedAnchorAlertState {
    DetectionMode mode = DetectionMode::NONE;
    ObsGateState gate_state = ObsGateState::NOT_OBSERVABLE;
    bool significant = false;
    bool cluster_member = false;
    bool reacquired = false;
    ros::Time last_logged_stamp;
  };

  using FrameInput = ObservationFrame;

  void LoadParameters();
  void ApplyAblationOverrides();
  void InitializeReferenceIfNeeded(const FrameInput& frame);
  void ProcessFrame(const FrameInput& frame);
  void ProcessFrameWindow(const ObservationFrameDeque& frames);
  void PublishResults(const ros::Time& stamp);
  void PublishEmptyResults(const ros::Time& stamp);
  void TryProcessQueuedFrames();
  void WorkerLoop();
  void InitializeDiagnosticsLog();
  void ShutdownDiagnosticsLog();
  void MaybeInitializeRuntimeLog();
  void FlushPendingRuntimeRecords();
  void WriteStageRuntimeRecord(const StageRuntimeRecord& record);
  void WriteFrameDiagnostics(const ros::Time& stamp,
                             size_t comparable_count,
                             size_t significant_count,
                             size_t cluster_count);
  void ResetReferenceStateLocked();
  void BuildReferenceAdjacency();
  bool PopSynchronizedFrame(sensor_msgs::PointCloud2* cloud_msg,
                            fast_lio::LioOdomCov* cov_msg);
  bool EstimateBackgroundBias(const CurrentObservationVector& observations,
                              Eigen::Vector3d* bias_R,
                              size_t* used_anchor_count,
                              size_t* used_scalar_count,
                              size_t* total_comparable_plane_count) const;
  void ApplyBackgroundBias(const Eigen::Vector3d& bias_R,
                           CurrentObservationVector* observations) const;
  void UpdateLocalContrastStates();
  void UpdateGraphTemporalStates(const ros::Time& stamp);
  static double TimeDistanceSec(const ros::Time& a, const ros::Time& b);
  void PrintStartupSummary() const;
  void PrintFrameSummary(const ros::Time& stamp,
                         size_t comparable_count,
                         size_t significant_count,
                         size_t cluster_count,
                         double total_ms);
  void MaybePrintPipelineStatus();
  void TryIncrementalAnchorPromotion(const ObservationFrameDeque& frames);
  bool JudgeAnchorSignificance(const AnchorReference& anchor, AnchorTrackState* state) const;
  FrameInput BuildFrameInput(const sensor_msgs::PointCloud2& cloud_msg,
                             const fast_lio::LioOdomCov& cov_msg) const;

  ros::NodeHandle nh_;
  ros::NodeHandle private_nh_;

  ros::Subscriber cloud_sub_;
  ros::Subscriber covariance_sub_;
  ros::Subscriber reset_reference_sub_;

  ros::Publisher anchors_pub_;
  ros::Publisher clusters_pub_;
  ros::Publisher debug_cloud_pub_;
  ros::Publisher anchor_markers_pub_;
  ros::Publisher motion_markers_pub_;
  ros::Publisher risk_evidence_pub_;
  ros::Publisher risk_voxels_pub_;
  ros::Publisher risk_regions_pub_;
  ros::Publisher risk_markers_pub_;
  ros::Publisher persistent_risk_regions_pub_;
  ros::Publisher persistent_risk_markers_pub_;
  ros::Publisher structure_motions_pub_;
  ros::Publisher structure_markers_pub_;

  DeformMonitorParams params_;
  CovarianceExtractor cov_extractor_;
  AnchorBuilder anchor_builder_;
  ScalarMeasurementBuilder scalar_builder_;
  CurrentObservationExtractor obs_extractor_;
  ImmInformationFilter imm_filter_;
  MotionClusterer clusterer_;
  RegionHypothesisBuilder region_hypothesis_builder_;
  RegionCorrespondenceSolver region_correspondence_solver_;
  RiskEvidenceAdapter risk_adapter_;
  RiskFieldBuilder risk_field_builder_;
  ReferenceManager ref_manager_;
  RiskVisualizationPublisher risk_viz_publisher_;
  StructureVisualizationPublisher structure_viz_publisher_;
  VisualizationPublisher viz_publisher_;

  std::deque<TimedCloud> cloud_queue_;
  std::deque<TimedCov> covariance_queue_;
  mutable std::mutex data_mutex_;
  mutable std::mutex processing_mutex_;
  std::condition_variable data_cv_;
  std::thread worker_thread_;
  bool stop_worker_ = false;

  int init_frame_count_ = 0;
  bool reference_ready_ = false;
  ReferenceInitFrameVector init_frames_;
  ObservationFrameDeque temporal_window_frames_;
  size_t frames_since_last_window_process_ = 0;

  AnchorReferenceVector anchors_;
  AnchorStateVector anchor_states_;
  CurrentObservationVector observations_;
  MotionClusterVector clusters_;
  RegionHypothesisVector old_regions_;
  RegionHypothesisVector new_regions_;
  StructureMotionVector structure_motions_;

  StructureUnitVector structure_units_;
  StructureMigrationVector structure_migrations_;
  std::unique_ptr<StructureUnitTracker> structure_unit_tracker_;
  RiskEvidenceVector risk_evidence_;
  RiskVoxelVector risk_voxels_;
  RiskRegionVector risk_regions_;
  PersistentRiskTrackVector persistent_risk_tracks_;
  PersistentRiskRegionTracker persistent_risk_tracker_;
  AlignedUnorderedMap<EdgeKey, EdgeTemporalState, EdgeKeyHash> edge_temporal_states_;

  ros::Time detection_time_base_stamp_;
  mutable ros::Time last_detection_stamp_;
  ros::Time last_cloud_stamp_;
  ros::Time last_covariance_stamp_;
  ros::Time last_processed_stamp_;
  ros::WallTime last_status_wall_time_;
  size_t cloud_msg_count_ = 0;
  size_t covariance_msg_count_ = 0;
  size_t synchronized_frame_count_ = 0;
  bool first_cloud_logged_ = false;
  bool first_covariance_logged_ = false;
  bool first_sync_logged_ = false;
  Eigen::Vector3d last_background_bias_R_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d smoothed_background_bias_R_ = Eigen::Vector3d::Zero();
  bool has_previous_bias_ = false;
  bool drift_estimation_degraded_ = false;
  size_t last_background_bias_anchor_count_ = 0;
  size_t last_background_bias_scalar_count_ = 0;

  ReferenceInitFrameVector incremental_init_frames_;
  int incremental_frame_count_ = 0;

  std::ofstream diagnostics_log_;
  std::string diagnostics_log_path_;
  int diag_frame_counter_ = 0;
  std::unordered_map<int, LoggedAnchorAlertState> logged_anchor_alerts_;
  StageRuntimeLogger runtime_logger_;
  std::string runtime_output_dir_;
  std::string runtime_output_dir_param_name_ = "/deform_monitor/runtime_output_dir";
  std::deque<StageRuntimeRecord> pending_runtime_records_;
  uint64_t runtime_frame_index_ = 0;
};

}  // namespace deform_monitor_v2

#endif  // DEFORM_MONITOR_V2_NODE_HPP
