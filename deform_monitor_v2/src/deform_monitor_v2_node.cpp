/*
 * Author: zgliu@cumt.edu.cn
 * Affiliation: China University of Mining and Technology
 * Open-source release date: 2026-04-20
 */

#include "deform_monitor_v2/deform_monitor_v2_node.hpp"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <Eigen/Eigenvalues>
#include <deform_monitor_v2/StructureMotions.h>
#include <pcl/common/io.h>
#include <pcl_conversions/pcl_conversions.h>

namespace deform_monitor_v2 {

namespace {

constexpr double kDiagnosticsAlertRefreshSec = 5.0;

struct SpatialVoxelKey {
  int x = 0;
  int y = 0;
  int z = 0;

  bool operator==(const SpatialVoxelKey& other) const {
    return x == other.x && y == other.y && z == other.z;
  }
};

struct SpatialVoxelKeyHash {
  size_t operator()(const SpatialVoxelKey& key) const {
    size_t h = std::hash<int>()(key.x);
    h ^= std::hash<int>()(key.y) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<int>()(key.z) + 0x9e3779b9 + (h << 6) + (h >> 2);
    return h;
  }
};

template <typename T>
T LoadParam(ros::NodeHandle& nh, const std::string& name, const T& default_value) {
  T value = default_value;
  if (nh.getParam(name, value)) {
    return value;
  }
  ros::param::param<T>("/" + name, value, default_value);
  return value;
}

template <typename QueueT>
void TrimQueue(QueueT* queue, size_t max_size) {
  while (queue && queue->size() > max_size) {
    queue->pop_front();
  }
}

double Clamp01(double value) {
  return std::max(0.0, std::min(1.0, value));
}

pcl::PointCloud<pcl::PointXYZI>::Ptr TransformCloud(
    const pcl::PointCloud<pcl::PointXYZI>::ConstPtr& in,
    const Eigen::Isometry3d& T) {
  pcl::PointCloud<pcl::PointXYZI>::Ptr out(new pcl::PointCloud<pcl::PointXYZI>());
  if (!in) {
    return out;
  }
  out->reserve(in->size());
  for (const auto& pt : in->points) {
    Eigen::Vector3d p = T * Eigen::Vector3d(pt.x, pt.y, pt.z);
    pcl::PointXYZI out_pt = pt;
    out_pt.x = static_cast<float>(p.x());
    out_pt.y = static_cast<float>(p.y());
    out_pt.z = static_cast<float>(p.z());
    out->points.push_back(out_pt);
  }
  out->width = out->points.size();
  out->height = 1;
  out->is_dense = false;
  return out;
}

double InverseNormalCdf(double p) {
  // Acklam rational approximation.
  static const double a1 = -3.969683028665376e+01;
  static const double a2 = 2.209460984245205e+02;
  static const double a3 = -2.759285104469687e+02;
  static const double a4 = 1.383577518672690e+02;
  static const double a5 = -3.066479806614716e+01;
  static const double a6 = 2.506628277459239e+00;

  static const double b1 = -5.447609879822406e+01;
  static const double b2 = 1.615858368580409e+02;
  static const double b3 = -1.556989798598866e+02;
  static const double b4 = 6.680131188771972e+01;
  static const double b5 = -1.328068155288572e+01;

  static const double c1 = -7.784894002430293e-03;
  static const double c2 = -3.223964580411365e-01;
  static const double c3 = -2.400758277161838e+00;
  static const double c4 = -2.549732539343734e+00;
  static const double c5 = 4.374664141464968e+00;
  static const double c6 = 2.938163982698783e+00;

  static const double d1 = 7.784695709041462e-03;
  static const double d2 = 3.224671290700398e-01;
  static const double d3 = 2.445134137142996e+00;
  static const double d4 = 3.754408661907416e+00;

  const double plow = 0.02425;
  const double phigh = 1.0 - plow;

  if (p <= 0.0) {
    return -std::numeric_limits<double>::infinity();
  }
  if (p >= 1.0) {
    return std::numeric_limits<double>::infinity();
  }

  if (p < plow) {
    const double q = std::sqrt(-2.0 * std::log(p));
    return (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
           ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0);
  }
  if (p > phigh) {
    const double q = std::sqrt(-2.0 * std::log(1.0 - p));
    return -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
           ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0);
  }

  const double q = p - 0.5;
  const double r = q * q;
  return (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q /
         (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0);
}

double Chi2ThresholdByDof(int dof, double alpha_s) {
  if (dof <= 0) {
    return std::numeric_limits<double>::infinity();
  }
  const double p = std::max(1.0e-6, std::min(1.0 - 1.0e-6, 1.0 - alpha_s));
  const double z = InverseNormalCdf(p);
  const double k = static_cast<double>(dof);
  const double base = 1.0 - 2.0 / (9.0 * k) + z * std::sqrt(2.0 / (9.0 * k));
  return k * base * base * base;
}

std::string DetectionModeToString(DetectionMode mode) {
  switch (mode) {
    case DetectionMode::NONE:
      return "NONE";
    case DetectionMode::DISPLACEMENT:
      return "DISPLACEMENT";
    case DetectionMode::DISAPPEARANCE:
      return "DISAPPEARANCE";
  }
  return "UNKNOWN";
}

std::string AnchorTypeToString(AnchorType type) {
  switch (type) {
    case AnchorType::PLANE:
      return "PLANE";
    case AnchorType::EDGE:
      return "EDGE";
    case AnchorType::BAND:
      return "BAND";
  }
  return "UNKNOWN";
}

std::string ObsGateStateToString(ObsGateState state) {
  switch (state) {
    case ObsGateState::NOT_OBSERVABLE:
      return "NOT_OBSERVABLE";
    case ObsGateState::OBSERVABLE_MATCHED:
      return "OBSERVABLE_MATCHED";
    case ObsGateState::OBSERVABLE_WEAK:
      return "OBSERVABLE_WEAK";
    case ObsGateState::OBSERVABLE_MISSING:
      return "OBSERVABLE_MISSING";
    case ObsGateState::OBSERVABLE_REPLACED:
      return "OBSERVABLE_REPLACED";
  }
  return "UNKNOWN";
}

std::string ObsStatusToString(ObsStatus status) {
  switch (status) {
    case ObsStatus::INVALID_NO_COMPARISON:
      return "INVALID_NO_COMPARISON";
    case ObsStatus::VALID_PARTIAL_OBS:
      return "VALID_PARTIAL_OBS";
    case ObsStatus::VALID_FULL_OBS:
      return "VALID_FULL_OBS";
  }
  return "UNKNOWN";
}

std::string FormatVec3(const Eigen::Vector3d& v) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(4)
      << "(" << v.x() << "," << v.y() << "," << v.z() << ")";
  return oss.str();
}

bool EnsureDirectoryExists(const std::string& path) {
  if (path.empty()) {
    return false;
  }

  struct stat info;
  if (stat(path.c_str(), &info) == 0) {
    return S_ISDIR(info.st_mode);
  }

  std::string current;
  if (!path.empty() && path.front() == '/') {
    current = "/";
  }

  std::stringstream ss(path);
  std::string segment;
  while (std::getline(ss, segment, '/')) {
    if (segment.empty()) {
      continue;
    }
    if (!current.empty() && current.back() != '/') {
      current += "/";
    }
    current += segment;

    if (stat(current.c_str(), &info) == 0) {
      if (!S_ISDIR(info.st_mode)) {
        return false;
      }
      continue;
    }
    if (mkdir(current.c_str(), 0755) != 0 && errno != EEXIST) {
      return false;
    }
  }

  return stat(path.c_str(), &info) == 0 && S_ISDIR(info.st_mode);
}

std::string MakeRunLogFilename() {
  const std::time_t now = std::time(nullptr);
  std::tm local_tm;
  localtime_r(&now, &local_tm);
  char buffer[64];
  if (std::strftime(buffer, sizeof(buffer), "%Y%m%d_%H%M%S", &local_tm) == 0) {
    return "anchor_diagnostics_unknown.log";
  }
  std::ostringstream oss;
  oss << "anchor_diagnostics_" << buffer << "_pid" << static_cast<long>(::getpid()) << ".log";
  return oss.str();
}

std::string FormatTimeJson(const ros::Time& stamp) {
  std::ostringstream oss;
  oss << "{\"secs\":" << stamp.sec
      << ",\"nsecs\":" << stamp.nsec
      << ",\"sec\":" << std::fixed << std::setprecision(9) << stamp.toSec()
      << "}";
  return oss.str();
}

template <typename Fn>
void ParallelFor(size_t count, Fn&& fn) {
  if (count == 0) {
    return;
  }

  const size_t hardware_threads =
      std::max<size_t>(1, static_cast<size_t>(std::thread::hardware_concurrency()));
  const size_t worker_count = std::min(count, hardware_threads);
  if (worker_count < 2 || count < 256) {
    for (size_t i = 0; i < count; ++i) {
      fn(i);
    }
    return;
  }

  const size_t block_size = (count + worker_count - 1) / worker_count;
  std::vector<std::thread> workers;
  workers.reserve(worker_count - 1);
  std::mutex exception_mutex;
  std::exception_ptr first_exception;

  auto run_block = [&](size_t begin, size_t end) {
    try {
      for (size_t i = begin; i < end; ++i) {
        fn(i);
      }
    } catch (...) {
      std::lock_guard<std::mutex> lock(exception_mutex);
      if (!first_exception) {
        first_exception = std::current_exception();
      }
    }
  };

  for (size_t worker_idx = 1; worker_idx < worker_count; ++worker_idx) {
    const size_t begin = worker_idx * block_size;
    if (begin >= count) {
      break;
    }
    const size_t end = std::min(count, begin + block_size);
    workers.emplace_back(run_block, begin, end);
  }

  run_block(0, std::min(count, block_size));

  for (auto& worker : workers) {
    worker.join();
  }
  if (first_exception) {
    std::rethrow_exception(first_exception);
  }
}

}  // namespace

ScopedWallTimer::ScopedWallTimer(double* accumulator_ms)
    : accumulator_ms_(accumulator_ms),
      start_time_(std::chrono::steady_clock::now()) {}

ScopedWallTimer::~ScopedWallTimer() {
  if (!accumulator_ms_) {
    return;
  }
  const auto end_time = std::chrono::steady_clock::now();
  const auto elapsed =
      std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end_time - start_time_);
  *accumulator_ms_ += elapsed.count();
}

StageRuntimeLogger::~StageRuntimeLogger() {
  Close();
}

bool StageRuntimeLogger::Initialize(const std::string& output_dir) {
  if (initialized_) {
    return true;
  }
  if (output_dir.empty() || !EnsureDirectoryExists(output_dir)) {
    return false;
  }

  log_path_ = output_dir + "/stage_runtime.jsonl";
  stream_.open(log_path_.c_str(), std::ios::out | std::ios::app);
  initialized_ = stream_.is_open();
  if (!initialized_) {
    log_path_.clear();
  }
  return initialized_;
}

bool StageRuntimeLogger::Write(const StageRuntimeRecord& record) {
  if (!initialized_ || !stream_.is_open()) {
    return false;
  }

  stream_ << "{\"stamp\":" << FormatTimeJson(record.stamp)
          << ",\"frame_index\":" << record.frame_index
          << ",\"total_ms\":" << std::fixed << std::setprecision(3) << record.total_ms
          << ",\"stage_a_ms\":" << record.stage_a_ms
          << ",\"stage_b_ms\":" << record.stage_b_ms
          << ",\"stage_c_ms\":" << record.stage_c_ms
          << ",\"stage_d_ms\":" << record.stage_d_ms
          << "}" << std::endl;
  stream_.flush();
  return stream_.good();
}

void StageRuntimeLogger::Close() {
  if (stream_.is_open()) {
    stream_.flush();
    stream_.close();
  }
  initialized_ = false;
  log_path_.clear();
}

DeformMonitorV2Node::DeformMonitorV2Node() : private_nh_("~") {
  LoadParameters();

  cloud_sub_ = nh_.subscribe(params_.io.cloud_topic, 20, &DeformMonitorV2Node::CloudCallback, this);
  covariance_sub_ =
      nh_.subscribe(params_.io.covariance_topic, 50, &DeformMonitorV2Node::CovarianceCallback, this);
  reset_reference_sub_ =
      nh_.subscribe(params_.io.reset_reference_topic,
                    2,
                    &DeformMonitorV2Node::ResetReferenceCallback,
                    this);

  anchors_pub_ = nh_.advertise<deform_monitor_v2::AnchorStates>("/deform/anchors", 10, true);
  clusters_pub_ = nh_.advertise<deform_monitor_v2::MotionClusters>("/deform/clusters", 10, true);
  debug_cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/deform/debug_cloud", 10, true);
  anchor_markers_pub_ =
      nh_.advertise<visualization_msgs::MarkerArray>("/deform/anchor_markers", 10, true);
  motion_markers_pub_ =
      nh_.advertise<visualization_msgs::MarkerArray>("/deform/motion_markers", 10, true);
  risk_evidence_pub_ =
      nh_.advertise<deform_monitor_v2::RiskEvidenceArray>("/deform/risk_evidence", 10, true);
  risk_voxels_pub_ =
      nh_.advertise<deform_monitor_v2::RiskVoxelField>("/deform/risk_voxels", 10, true);
  risk_regions_pub_ =
      nh_.advertise<deform_monitor_v2::RiskRegions>("/deform/risk_regions", 10, true);
  risk_markers_pub_ =
      nh_.advertise<visualization_msgs::MarkerArray>("/deform/risk_markers", 10, true);
  persistent_risk_regions_pub_ = nh_.advertise<deform_monitor_v2::PersistentRiskRegions>(
      "/deform/persistent_risk_regions", 10, true);
  persistent_risk_markers_pub_ = nh_.advertise<visualization_msgs::MarkerArray>(
      "/deform/persistent_risk_markers", 10, true);
  structure_motions_pub_ =
      nh_.advertise<deform_monitor_v2::StructureMotions>("/deform/structure_motions", 10, true);
  structure_markers_pub_ =
      nh_.advertise<visualization_msgs::MarkerArray>("/deform/structure_markers", 10, true);

  InitializeDiagnosticsLog();
  PrintStartupSummary();
}

void DeformMonitorV2Node::InitializeDiagnosticsLog() {
  std::string log_dir = "log";
#ifdef DEFORM_MONITOR_V2_SOURCE_DIR
  log_dir = std::string(DEFORM_MONITOR_V2_SOURCE_DIR) + "/log";
#endif
  if (!EnsureDirectoryExists(log_dir)) {
    std::cerr << "[deform_monitor_v2] Failed to create diagnostics dir: " << log_dir << std::endl;
    return;
  }

  diagnostics_log_path_ = log_dir + "/" + MakeRunLogFilename();
  diagnostics_log_.open(diagnostics_log_path_.c_str(), std::ios::out | std::ios::trunc);
  if (!diagnostics_log_.is_open()) {
    std::cerr << "[deform_monitor_v2] Failed to open diagnostics log: " << diagnostics_log_path_ << std::endl;
    diagnostics_log_path_.clear();
    return;
  }

  diagnostics_log_ << "# Deform Monitor V2 Anchor Diagnostics" << std::endl;
  diagnostics_log_ << "node=" << ros::this_node::getName() << std::endl;
  diagnostics_log_ << "pid=" << static_cast<long>(::getpid()) << std::endl;
  diagnostics_log_ << "wall_start=" << ros::WallTime::now().toSec() << std::endl;
  diagnostics_log_ << "cloud_topic=" << params_.io.cloud_topic
                   << " covariance_topic=" << params_.io.covariance_topic << std::endl;
  diagnostics_log_ << "temporal.enable=" << (params_.temporal.enable ? 1 : 0)
                   << " window_frames=" << params_.temporal.window_frames
                   << " min_frames=" << params_.temporal.min_frames
                   << " step_frames=" << params_.temporal.step_frames
                   << " max_window_sec=" << params_.temporal.max_window_sec
                   << " use_stable_voxel_fusion="
                   << (params_.temporal.use_stable_voxel_fusion ? 1 : 0)
                   << " min_fused_visible_frames=" << params_.temporal.min_fused_visible_frames
                   << " min_fused_points_per_voxel=" << params_.temporal.min_fused_points_per_voxel
                   << std::endl;
  diagnostics_log_ << "significance.alpha_s=" << params_.significance.alpha_s
                   << " tau_A_norm=" << params_.significance.tau_A_norm
                   << " tau_A_normal=" << params_.significance.tau_A_normal
                   << " tau_A_edge=" << params_.significance.tau_A_edge
                   << " tau_disappear=" << params_.significance.tau_disappear
                   << " disappear_frames=" << params_.significance.disappear_frames << std::endl;
  diagnostics_log_ << "local_contrast.enable=" << (params_.local_contrast.enable ? 1 : 0)
                   << " min_neighbors=" << params_.local_contrast.min_neighbors
                   << " tau_contrast_score=" << params_.local_contrast.tau_contrast_score
                   << " tau_rel_norm=" << params_.local_contrast.tau_rel_norm
                   << " tau_rel_normal=" << params_.local_contrast.tau_rel_normal
                   << " tau_rel_edge=" << params_.local_contrast.tau_rel_edge << std::endl;
  diagnostics_log_ << "graph_temporal.enable=" << (params_.graph_temporal.enable ? 1 : 0)
                   << " min_neighbors=" << params_.graph_temporal.min_neighbors
                   << " tau_graph_support=" << params_.graph_temporal.tau_graph_support
                   << " tau_graph_temporal=" << params_.graph_temporal.tau_graph_temporal
                   << " cusum_h=" << params_.graph_temporal.cusum_h << std::endl;
  diagnostics_log_ << "ablation.variant=" << params_.ablation.variant
                   << " disable_covariance_inflation="
                   << (params_.ablation.disable_covariance_inflation ? 1 : 0)
                   << " disable_type_constraint="
                   << (params_.ablation.disable_type_constraint ? 1 : 0)
                   << " single_model_ekf=" << (params_.ablation.single_model_ekf ? 1 : 0)
                   << " disable_cusum=" << (params_.ablation.disable_cusum ? 1 : 0)
                   << " disable_directional_accumulation="
                   << (params_.ablation.disable_directional_accumulation ? 1 : 0)
                   << " disable_drift_compensation="
                   << (params_.ablation.disable_drift_compensation ? 1 : 0) << std::endl;
  diagnostics_log_ << "visualization.min_arrow_disp=" << params_.visualization.min_arrow_disp
                   << " min_arrow_contrast_score=" << params_.visualization.min_arrow_contrast_score
                   << " arrows_only_clustered_or_reacquired="
                   << (params_.visualization.arrows_only_clustered_or_reacquired ? 1 : 0)
                   << std::endl;
  diagnostics_log_ << std::endl;
  diagnostics_log_.flush();

  std::cout << "[deform_monitor_v2] Diagnostics log: " << diagnostics_log_path_ << std::endl;
}

void DeformMonitorV2Node::ShutdownDiagnosticsLog() {
  if (diagnostics_log_.is_open()) {
    diagnostics_log_ << "[SHUTDOWN] wall_time=" << ros::WallTime::now().toSec() << std::endl;
    diagnostics_log_.flush();
    diagnostics_log_.close();
  }
  runtime_logger_.Close();
}

void DeformMonitorV2Node::MaybeInitializeRuntimeLog() {
  if (runtime_logger_.is_initialized()) {
    return;
  }

  std::string output_dir = runtime_output_dir_;
  if (output_dir.empty() && !runtime_output_dir_param_name_.empty()) {
    ros::param::get(runtime_output_dir_param_name_, output_dir);
  }
  if (output_dir.empty()) {
    return;
  }
  runtime_output_dir_ = output_dir;
  if (runtime_logger_.Initialize(output_dir)) {
    FlushPendingRuntimeRecords();
  }
}

void DeformMonitorV2Node::FlushPendingRuntimeRecords() {
  if (!runtime_logger_.is_initialized()) {
    return;
  }
  while (!pending_runtime_records_.empty()) {
    if (!runtime_logger_.Write(pending_runtime_records_.front())) {
      return;
    }
    pending_runtime_records_.pop_front();
  }
}

void DeformMonitorV2Node::WriteStageRuntimeRecord(const StageRuntimeRecord& record) {
  MaybeInitializeRuntimeLog();
  if (runtime_logger_.is_initialized()) {
    runtime_logger_.Write(record);
    return;
  }
  pending_runtime_records_.push_back(record);
  while (pending_runtime_records_.size() > 256) {
    pending_runtime_records_.pop_front();
  }
}

void DeformMonitorV2Node::LoadParameters() {
  params_.io.cloud_topic = LoadParam<std::string>(private_nh_, "io/cloud_topic", params_.io.cloud_topic);
  params_.io.covariance_topic =
      LoadParam<std::string>(private_nh_, "io/covariance_topic", params_.io.covariance_topic);
  params_.io.reset_reference_topic =
      LoadParam<std::string>(private_nh_, "io/reset_reference_topic",
                             params_.io.reset_reference_topic);
  params_.io.reference_frame =
      LoadParam<std::string>(private_nh_, "io/reference_frame", params_.io.reference_frame);
  params_.io.cloud_already_in_reference_frame =
      LoadParam<bool>(private_nh_,
                      "io/cloud_already_in_reference_frame",
                      params_.io.cloud_already_in_reference_frame);
  params_.io.odom_cov_layout =
      LoadParam<std::string>(private_nh_, "io/odom_cov_layout", params_.io.odom_cov_layout);
  params_.io.default_rot_sigma =
      LoadParam<double>(private_nh_, "io/default_rot_sigma", params_.io.default_rot_sigma);
  params_.io.default_pos_sigma =
      LoadParam<double>(private_nh_, "io/default_pos_sigma", params_.io.default_pos_sigma);

  params_.reference.init_frames =
      LoadParam<int>(private_nh_, "deform_monitor/reference/init_frames", params_.reference.init_frames);
  params_.reference.tau_ref_quality =
      LoadParam<double>(private_nh_, "deform_monitor/reference/tau_ref_quality", params_.reference.tau_ref_quality);
  params_.reference.tau_cmp_ref =
      LoadParam<double>(private_nh_, "deform_monitor/reference/tau_cmp_ref", params_.reference.tau_cmp_ref);
  params_.reference.tau_ref_stable =
      LoadParam<double>(private_nh_, "deform_monitor/reference/tau_ref_stable", params_.reference.tau_ref_stable);
  params_.reference.N_ref_stable =
      LoadParam<int>(private_nh_, "deform_monitor/reference/N_ref_stable", params_.reference.N_ref_stable);
  params_.reference.tau_mu0 =
      LoadParam<double>(private_nh_, "deform_monitor/reference/tau_mu0", params_.reference.tau_mu0);

  params_.covariance.alpha_xi =
      LoadParam<double>(private_nh_, "deform_monitor/covariance/alpha_xi", params_.covariance.alpha_xi);
  params_.covariance.sigma_p =
      LoadParam<double>(private_nh_, "deform_monitor/covariance/sigma_p", params_.covariance.sigma_p);

  params_.observation.tau_n_deg =
      LoadParam<double>(private_nh_, "deform_monitor/observation/tau_n_deg", params_.observation.tau_n_deg);
  params_.observation.tau_v_deg =
      LoadParam<double>(private_nh_, "deform_monitor/observation/tau_v_deg", params_.observation.tau_v_deg);
  params_.observation.tau_r_ratio =
      LoadParam<double>(private_nh_, "deform_monitor/observation/tau_r_ratio", params_.observation.tau_r_ratio);
  params_.observation.tau_cmp =
      LoadParam<double>(private_nh_, "deform_monitor/observation/tau_cmp", params_.observation.tau_cmp);
  params_.observation.current_voxel_size =
      LoadParam<double>(private_nh_, "deform_monitor/observation/current_voxel_size",
                        params_.observation.current_voxel_size);
  params_.observation.tau_reacquire =
      LoadParam<double>(private_nh_, "deform_monitor/observation/tau_reacquire",
                        params_.observation.tau_reacquire);
  params_.observation.reacquire_radius =
      LoadParam<double>(private_nh_, "deform_monitor/observation/reacquire_radius",
                        params_.observation.reacquire_radius);
  params_.observation.max_reacquire_candidates =
      LoadParam<int>(private_nh_, "deform_monitor/observation/max_reacquire_candidates",
                     params_.observation.max_reacquire_candidates);
  params_.observation.min_support_scalar =
      LoadParam<int>(private_nh_, "deform_monitor/observation/min_support_scalar",
                     params_.observation.min_support_scalar);
  params_.observation.tau_nis_scalar =
      LoadParam<double>(private_nh_, "deform_monitor/observation/tau_nis_scalar",
                        params_.observation.tau_nis_scalar);
  params_.observation.soft_view_range_gate =
      LoadParam<bool>(private_nh_, "deform_monitor/observation/soft_view_range_gate",
                      params_.observation.soft_view_range_gate);
  params_.observation.hard_view_deg_cap =
      LoadParam<double>(private_nh_, "deform_monitor/observation/hard_view_deg_cap",
                        params_.observation.hard_view_deg_cap);
  params_.observation.hard_range_ratio_cap =
      LoadParam<double>(private_nh_, "deform_monitor/observation/hard_range_ratio_cap",
                        params_.observation.hard_range_ratio_cap);

  params_.temporal.enable =
      LoadParam<bool>(private_nh_, "deform_monitor/temporal/enable", params_.temporal.enable);
  params_.temporal.window_frames =
      LoadParam<int>(private_nh_, "deform_monitor/temporal/window_frames",
                     params_.temporal.window_frames);
  params_.temporal.min_frames =
      LoadParam<int>(private_nh_, "deform_monitor/temporal/min_frames",
                     params_.temporal.min_frames);
  params_.temporal.step_frames =
      LoadParam<int>(private_nh_, "deform_monitor/temporal/step_frames",
                     params_.temporal.step_frames);
  params_.temporal.max_window_sec =
      LoadParam<double>(private_nh_, "deform_monitor/temporal/max_window_sec",
                        params_.temporal.max_window_sec);
  params_.temporal.pose_corr_inflation =
      LoadParam<double>(private_nh_, "deform_monitor/temporal/pose_corr_inflation",
                        params_.temporal.pose_corr_inflation);
  params_.temporal.sigma_motion_per_sec =
      LoadParam<double>(private_nh_, "deform_monitor/temporal/sigma_motion_per_sec",
                        params_.temporal.sigma_motion_per_sec);
  params_.temporal.use_stable_voxel_fusion =
      LoadParam<bool>(private_nh_, "deform_monitor/temporal/use_stable_voxel_fusion",
                      params_.temporal.use_stable_voxel_fusion);
  params_.temporal.min_fused_visible_frames =
      LoadParam<int>(private_nh_, "deform_monitor/temporal/min_fused_visible_frames",
                     params_.temporal.min_fused_visible_frames);
  params_.temporal.min_fused_points_per_voxel =
      LoadParam<int>(private_nh_, "deform_monitor/temporal/min_fused_points_per_voxel",
                     params_.temporal.min_fused_points_per_voxel);

  params_.background_bias.enable =
      LoadParam<bool>(private_nh_, "deform_monitor/background_bias/enable",
                      params_.background_bias.enable);
  params_.background_bias.min_anchor_count =
      LoadParam<int>(private_nh_, "deform_monitor/background_bias/min_anchor_count",
                     params_.background_bias.min_anchor_count);
  params_.background_bias.min_scalar_count =
      LoadParam<int>(private_nh_, "deform_monitor/background_bias/min_scalar_count",
                     params_.background_bias.min_scalar_count);
  params_.background_bias.min_support_points =
      LoadParam<int>(private_nh_, "deform_monitor/background_bias/min_support_points",
                     params_.background_bias.min_support_points);
  params_.background_bias.min_cmp_score =
      LoadParam<double>(private_nh_, "deform_monitor/background_bias/min_cmp_score",
                        params_.background_bias.min_cmp_score);
  params_.background_bias.max_anchor_disp =
      LoadParam<double>(private_nh_, "deform_monitor/background_bias/max_anchor_disp",
                        params_.background_bias.max_anchor_disp);
  params_.background_bias.ridge_lambda =
      LoadParam<double>(private_nh_, "deform_monitor/background_bias/ridge_lambda",
                        params_.background_bias.ridge_lambda);
  params_.background_bias.huber_delta =
      LoadParam<double>(private_nh_, "deform_monitor/background_bias/huber_delta",
                        params_.background_bias.huber_delta);
  params_.background_bias.max_bias_norm =
      LoadParam<double>(private_nh_, "deform_monitor/background_bias/max_bias_norm",
                        params_.background_bias.max_bias_norm);
  params_.background_bias.adaptive_disp_step1 =
      LoadParam<double>(private_nh_, "deform_monitor/background_bias/adaptive_disp_step1",
                        params_.background_bias.adaptive_disp_step1);
  params_.background_bias.adaptive_disp_step2 =
      LoadParam<double>(private_nh_, "deform_monitor/background_bias/adaptive_disp_step2",
                        params_.background_bias.adaptive_disp_step2);
  params_.background_bias.ema_alpha =
      LoadParam<double>(private_nh_, "deform_monitor/background_bias/ema_alpha",
                        params_.background_bias.ema_alpha);
  params_.background_bias.min_stable_ratio =
      LoadParam<double>(private_nh_, "deform_monitor/background_bias/min_stable_ratio",
                        params_.background_bias.min_stable_ratio);

  params_.local_contrast.enable =
      LoadParam<bool>(private_nh_, "deform_monitor/local_contrast/enable",
                      params_.local_contrast.enable);
  params_.local_contrast.radius =
      LoadParam<double>(private_nh_, "deform_monitor/local_contrast/radius",
                        params_.local_contrast.radius);
  params_.local_contrast.min_neighbors =
      LoadParam<int>(private_nh_, "deform_monitor/local_contrast/min_neighbors",
                     params_.local_contrast.min_neighbors);
  params_.local_contrast.enable_plane_background_for_edges =
      LoadParam<bool>(private_nh_,
                      "deform_monitor/local_contrast/enable_plane_background_for_edges",
                      params_.local_contrast.enable_plane_background_for_edges);
  params_.local_contrast.min_plane_neighbors =
      LoadParam<int>(private_nh_, "deform_monitor/local_contrast/min_plane_neighbors",
                     params_.local_contrast.min_plane_neighbors);
  params_.local_contrast.min_support_points =
      LoadParam<int>(private_nh_, "deform_monitor/local_contrast/min_support_points",
                     params_.local_contrast.min_support_points);
  params_.local_contrast.min_cmp_score =
      LoadParam<double>(private_nh_, "deform_monitor/local_contrast/min_cmp_score",
                        params_.local_contrast.min_cmp_score);
  params_.local_contrast.tau_bg_normal_deg =
      LoadParam<double>(private_nh_, "deform_monitor/local_contrast/tau_bg_normal_deg",
                        params_.local_contrast.tau_bg_normal_deg);
  params_.local_contrast.max_background_disp =
      LoadParam<double>(private_nh_, "deform_monitor/local_contrast/max_background_disp",
                        params_.local_contrast.max_background_disp);
  params_.local_contrast.min_background_sigma =
      LoadParam<double>(private_nh_, "deform_monitor/local_contrast/min_background_sigma",
                        params_.local_contrast.min_background_sigma);
  params_.local_contrast.tau_rel_norm =
      LoadParam<double>(private_nh_, "deform_monitor/local_contrast/tau_rel_norm",
                        params_.local_contrast.tau_rel_norm);
  params_.local_contrast.tau_rel_normal =
      LoadParam<double>(private_nh_, "deform_monitor/local_contrast/tau_rel_normal",
                        params_.local_contrast.tau_rel_normal);
  params_.local_contrast.tau_rel_edge =
      LoadParam<double>(private_nh_, "deform_monitor/local_contrast/tau_rel_edge",
                        params_.local_contrast.tau_rel_edge);
  params_.local_contrast.tau_contrast_score =
      LoadParam<double>(private_nh_, "deform_monitor/local_contrast/tau_contrast_score",
                        params_.local_contrast.tau_contrast_score);
  params_.local_contrast.tau_plane_contrast_score =
      LoadParam<double>(private_nh_, "deform_monitor/local_contrast/tau_plane_contrast_score",
                        params_.local_contrast.tau_plane_contrast_score);

  params_.graph_temporal.enable =
      LoadParam<bool>(private_nh_, "deform_monitor/graph_temporal/enable",
                      params_.graph_temporal.enable);
  params_.graph_temporal.spatial_hash_size =
      LoadParam<double>(private_nh_, "deform_monitor/graph_temporal/spatial_hash_size",
                        params_.graph_temporal.spatial_hash_size);
  params_.graph_temporal.radius =
      LoadParam<double>(private_nh_, "deform_monitor/graph_temporal/radius",
                        params_.graph_temporal.radius);
  params_.graph_temporal.min_neighbors =
      LoadParam<int>(private_nh_, "deform_monitor/graph_temporal/min_neighbors",
                     params_.graph_temporal.min_neighbors);
  params_.graph_temporal.tau_normal_deg =
      LoadParam<double>(private_nh_, "deform_monitor/graph_temporal/tau_normal_deg",
                        params_.graph_temporal.tau_normal_deg);
  params_.graph_temporal.tau_disp_cos =
      LoadParam<double>(private_nh_, "deform_monitor/graph_temporal/tau_disp_cos",
                        params_.graph_temporal.tau_disp_cos);
  params_.graph_temporal.tau_coherent_diff =
      LoadParam<double>(private_nh_, "deform_monitor/graph_temporal/tau_coherent_diff",
                        params_.graph_temporal.tau_coherent_diff);
  params_.graph_temporal.tau_temporal_change =
      LoadParam<double>(private_nh_, "deform_monitor/graph_temporal/tau_temporal_change",
                        params_.graph_temporal.tau_temporal_change);
  params_.graph_temporal.min_anchor_disp =
      LoadParam<double>(private_nh_, "deform_monitor/graph_temporal/min_anchor_disp",
                        params_.graph_temporal.min_anchor_disp);
  params_.graph_temporal.ema_alpha =
      LoadParam<double>(private_nh_, "deform_monitor/graph_temporal/ema_alpha",
                        params_.graph_temporal.ema_alpha);
  params_.graph_temporal.cusum_k =
      LoadParam<double>(private_nh_, "deform_monitor/graph_temporal/cusum_k",
                        params_.graph_temporal.cusum_k);
  params_.graph_temporal.cusum_h =
      LoadParam<double>(private_nh_, "deform_monitor/graph_temporal/cusum_h",
                        params_.graph_temporal.cusum_h);

  params_.graph_temporal.cusum_lambda =
      LoadParam<double>(private_nh_, "deform_monitor/graph_temporal/cusum_lambda",
                        params_.graph_temporal.cusum_lambda);
  params_.graph_temporal.tau_graph_support =
      LoadParam<double>(private_nh_, "deform_monitor/graph_temporal/tau_graph_support",
                        params_.graph_temporal.tau_graph_support);
  params_.graph_temporal.tau_graph_temporal =
      LoadParam<double>(private_nh_, "deform_monitor/graph_temporal/tau_graph_temporal",
                        params_.graph_temporal.tau_graph_temporal);
  if (params_.graph_temporal.spatial_hash_size <= 0.0) {
    params_.graph_temporal.spatial_hash_size = params_.anchor.voxel_size;
  }

  params_.anchor.I_min =
      LoadParam<double>(private_nh_, "deform_monitor/anchor/I_min", params_.anchor.I_min);
  params_.anchor.beta_edge =
      LoadParam<double>(private_nh_, "deform_monitor/anchor/beta_edge", params_.anchor.beta_edge);
  params_.anchor.beta_depth =
      LoadParam<double>(private_nh_, "deform_monitor/anchor/beta_depth", params_.anchor.beta_depth);
  params_.anchor.beta_normal =
      LoadParam<double>(private_nh_, "deform_monitor/anchor/beta_normal", params_.anchor.beta_normal);
  params_.anchor.beta_view =
      LoadParam<double>(private_nh_, "deform_monitor/anchor/beta_view", params_.anchor.beta_view);
  params_.anchor.voxel_size =
      LoadParam<double>(private_nh_, "deform_monitor/anchor/voxel_size", params_.anchor.voxel_size);
  params_.anchor.min_visible_frames =
      LoadParam<int>(private_nh_, "deform_monitor/anchor/min_visible_frames",
                     params_.anchor.min_visible_frames);
  params_.anchor.min_points_per_voxel =
      LoadParam<int>(private_nh_, "deform_monitor/anchor/min_points_per_voxel",
                     params_.anchor.min_points_per_voxel);
  params_.anchor.neighborhood_layers =
      LoadParam<int>(private_nh_, "deform_monitor/anchor/neighborhood_layers",
                     params_.anchor.neighborhood_layers);
  params_.anchor.seed_voxel =
      LoadParam<double>(private_nh_, "deform_monitor/anchor/seed_voxel", params_.anchor.seed_voxel);
  params_.anchor.radius_min =
      LoadParam<double>(private_nh_, "deform_monitor/anchor/radius_min", params_.anchor.radius_min);
  params_.anchor.min_support_points =
      LoadParam<int>(private_nh_, "deform_monitor/anchor/min_support_points",
                     params_.anchor.min_support_points);
  params_.anchor.edge_ref_bonus =
      LoadParam<double>(private_nh_, "deform_monitor/anchor/edge_ref_bonus",
                        params_.anchor.edge_ref_bonus);
  params_.anchor.band_ref_bonus =
      LoadParam<double>(private_nh_, "deform_monitor/anchor/band_ref_bonus",
                        params_.anchor.band_ref_bonus);
  params_.anchor.tau_ref_quality = params_.reference.tau_ref_quality;

  params_.noise.sigma_pi0 =
      LoadParam<double>(private_nh_, "deform_monitor/noise/sigma_pi0", params_.noise.sigma_pi0);
  params_.noise.sigma_edge0 =
      LoadParam<double>(private_nh_, "deform_monitor/noise/sigma_edge0", params_.noise.sigma_edge0);
  params_.noise.sigma_rad0 =
      LoadParam<double>(private_nh_, "deform_monitor/noise/sigma_rad0", params_.noise.sigma_rad0);
  params_.noise.sigma_bc0 =
      LoadParam<double>(private_nh_, "deform_monitor/noise/sigma_bc0", params_.noise.sigma_bc0);
  params_.noise.kappa_r =
      LoadParam<double>(private_nh_, "deform_monitor/noise/kappa_r", params_.noise.kappa_r);
  params_.noise.kappa_v =
      LoadParam<double>(private_nh_, "deform_monitor/noise/kappa_v", params_.noise.kappa_v);

  params_.observability.tau_lambda =
      LoadParam<double>(private_nh_, "deform_monitor/observability/tau_lambda",
                        params_.observability.tau_lambda);
  params_.observability.tau_sigma_max =
      LoadParam<double>(private_nh_, "deform_monitor/observability/tau_sigma_max",
                        params_.observability.tau_sigma_max);

  params_.imm.rho = LoadParam<double>(private_nh_, "deform_monitor/imm/rho", params_.imm.rho);
  params_.imm.enable_model_competition =
      LoadParam<bool>(private_nh_,
                      "deform_monitor/imm/enable_model_competition",
                      params_.imm.enable_model_competition);
  params_.imm.enable_type_constraint =
      LoadParam<bool>(private_nh_,
                      "deform_monitor/imm/enable_type_constraint",
                      params_.imm.enable_type_constraint);
  params_.imm.p00 = LoadParam<double>(private_nh_, "deform_monitor/imm/p00", params_.imm.p00);
  params_.imm.p01 = LoadParam<double>(private_nh_, "deform_monitor/imm/p01", params_.imm.p01);
  params_.imm.p10 = LoadParam<double>(private_nh_, "deform_monitor/imm/p10", params_.imm.p10);
  params_.imm.p11 = LoadParam<double>(private_nh_, "deform_monitor/imm/p11", params_.imm.p11);
  params_.imm.q_u0 = LoadParam<double>(private_nh_, "deform_monitor/imm/q_u0", params_.imm.q_u0);
  params_.imm.q_v0 = LoadParam<double>(private_nh_, "deform_monitor/imm/q_v0", params_.imm.q_v0);
  params_.imm.q_u1 = LoadParam<double>(private_nh_, "deform_monitor/imm/q_u1", params_.imm.q_u1);
  params_.imm.q_v1 = LoadParam<double>(private_nh_, "deform_monitor/imm/q_v1", params_.imm.q_v1);

  params_.significance.alpha_s =
      LoadParam<double>(private_nh_, "deform_monitor/significance/alpha_s", params_.significance.alpha_s);
  params_.significance.enable_cusum =
      LoadParam<bool>(private_nh_,
                      "deform_monitor/significance/enable_cusum",
                      params_.significance.enable_cusum);
  params_.significance.tau_A_norm =
      LoadParam<double>(private_nh_, "deform_monitor/significance/tau_A_norm", params_.significance.tau_A_norm);
  params_.significance.tau_A_normal =
      LoadParam<double>(private_nh_, "deform_monitor/significance/tau_A_normal",
                        params_.significance.tau_A_normal);
  params_.significance.tau_A_edge =
      LoadParam<double>(private_nh_, "deform_monitor/significance/tau_A_edge", params_.significance.tau_A_edge);
  params_.significance.tau_disappear =
      LoadParam<double>(private_nh_, "deform_monitor/significance/tau_disappear",
                        params_.significance.tau_disappear);
  params_.significance.disappear_frames =
      LoadParam<int>(private_nh_, "deform_monitor/significance/disappear_frames",
                     params_.significance.disappear_frames);
  params_.significance.cusum_k =
      LoadParam<double>(private_nh_, "deform_monitor/significance/cusum_k", params_.significance.cusum_k);
  params_.significance.cusum_h =
      LoadParam<double>(private_nh_, "deform_monitor/significance/cusum_h", params_.significance.cusum_h);

  params_.significance.cusum_lambda =
      LoadParam<double>(private_nh_, "deform_monitor/significance/cusum_lambda", params_.significance.cusum_lambda);
  params_.significance.cusum_cap_factor =
      LoadParam<double>(private_nh_, "deform_monitor/significance/cusum_cap_factor", params_.significance.cusum_cap_factor);

  params_.directional_motion.lambda0 =
      LoadParam<double>(private_nh_, "deform_monitor/directional_motion/lambda0",
                        params_.directional_motion.lambda0);
  params_.directional_motion.enable =
      LoadParam<bool>(private_nh_,
                      "deform_monitor/directional_motion/enable",
                      params_.directional_motion.enable);
  params_.directional_motion.tau_s =
      LoadParam<double>(private_nh_, "deform_monitor/directional_motion/tau_s",
                        params_.directional_motion.tau_s);
  params_.directional_motion.tau_c =
      LoadParam<double>(private_nh_, "deform_monitor/directional_motion/tau_c",
                        params_.directional_motion.tau_c);
  params_.directional_motion.tau_d =
      LoadParam<double>(private_nh_, "deform_monitor/directional_motion/tau_d",
                        params_.directional_motion.tau_d);
  params_.directional_motion.epsilon =
      LoadParam<double>(private_nh_, "deform_monitor/directional_motion/epsilon",
                        params_.directional_motion.epsilon);

  params_.cluster.tau_d =
      LoadParam<double>(private_nh_, "deform_monitor/cluster/tau_d", params_.cluster.tau_d);
  params_.cluster.tau_ng_deg =
      LoadParam<double>(private_nh_, "deform_monitor/cluster/tau_ng_deg", params_.cluster.tau_ng_deg);
  params_.cluster.tau_u_cos =
      LoadParam<double>(private_nh_, "deform_monitor/cluster/tau_u_cos", params_.cluster.tau_u_cos);
  params_.cluster.tau_corr =
      LoadParam<double>(private_nh_, "deform_monitor/cluster/tau_corr", params_.cluster.tau_corr);
  params_.cluster.enable_compact_motion =
      LoadParam<bool>(private_nh_, "deform_monitor/cluster/enable_compact_motion",
                      params_.cluster.enable_compact_motion);
  params_.cluster.compact_tau_d =
      LoadParam<double>(private_nh_, "deform_monitor/cluster/compact_tau_d",
                        params_.cluster.compact_tau_d);
  params_.cluster.compact_tau_ng_deg =
      LoadParam<double>(private_nh_, "deform_monitor/cluster/compact_tau_ng_deg",
                        params_.cluster.compact_tau_ng_deg);
  params_.cluster.compact_tau_u_cos =
      LoadParam<double>(private_nh_, "deform_monitor/cluster/compact_tau_u_cos",
                        params_.cluster.compact_tau_u_cos);
  params_.cluster.compact_tau_corr =
      LoadParam<double>(private_nh_, "deform_monitor/cluster/compact_tau_corr",
                        params_.cluster.compact_tau_corr);
  params_.cluster.compact_seed_disp =
      LoadParam<double>(private_nh_, "deform_monitor/cluster/compact_seed_disp",
                        params_.cluster.compact_seed_disp);
  params_.cluster.compact_seed_chi2 =
      LoadParam<double>(private_nh_, "deform_monitor/cluster/compact_seed_chi2",
                        params_.cluster.compact_seed_chi2);
  params_.cluster.compact_min_cluster_size =
      LoadParam<int>(private_nh_, "deform_monitor/cluster/compact_min_cluster_size",
                     params_.cluster.compact_min_cluster_size);
  params_.cluster.compact_tau_cluster_disp =
      LoadParam<double>(private_nh_, "deform_monitor/cluster/compact_tau_cluster_disp",
                        params_.cluster.compact_tau_cluster_disp);
  params_.cluster.compact_tau_cluster_chi2 =
      LoadParam<double>(private_nh_, "deform_monitor/cluster/compact_tau_cluster_chi2",
                        params_.cluster.compact_tau_cluster_chi2);
  params_.cluster.compact_max_bbox_diag =
      LoadParam<double>(private_nh_, "deform_monitor/cluster/compact_max_bbox_diag",
                        params_.cluster.compact_max_bbox_diag);
  params_.cluster.spatial_hash_size =
      LoadParam<double>(private_nh_, "deform_monitor/cluster/spatial_hash_size",
                        params_.cluster.spatial_hash_size);
  params_.cluster.beta_dir =
      LoadParam<double>(private_nh_, "deform_monitor/cluster/beta_dir", params_.cluster.beta_dir);
  params_.cluster.beta_mag =
      LoadParam<double>(private_nh_, "deform_monitor/cluster/beta_mag", params_.cluster.beta_mag);
  params_.cluster.beta_dist =
      LoadParam<double>(private_nh_, "deform_monitor/cluster/beta_dist", params_.cluster.beta_dist);
  params_.cluster.beta_time =
      LoadParam<double>(private_nh_, "deform_monitor/cluster/beta_time", params_.cluster.beta_time);
  params_.cluster.sigma_m =
      LoadParam<double>(private_nh_, "deform_monitor/cluster/sigma_m", params_.cluster.sigma_m);
  params_.cluster.sigma_d =
      LoadParam<double>(private_nh_, "deform_monitor/cluster/sigma_d", params_.cluster.sigma_d);
  params_.cluster.tau_edge_score =
      LoadParam<double>(private_nh_, "deform_monitor/cluster/tau_edge_score", params_.cluster.tau_edge_score);
  params_.cluster.min_cluster_size =
      LoadParam<int>(private_nh_, "deform_monitor/cluster/min_cluster_size",
                     params_.cluster.min_cluster_size);
  params_.cluster.min_cluster_size_disappear =
      LoadParam<int>(private_nh_, "deform_monitor/cluster/min_cluster_size_disappear",
                     params_.cluster.min_cluster_size_disappear);
  params_.cluster.tau_cluster_disp =
      LoadParam<double>(private_nh_, "deform_monitor/cluster/tau_cluster_disp",
                        params_.cluster.tau_cluster_disp);
  params_.cluster.tau_cluster_chi2 =
      LoadParam<double>(private_nh_, "deform_monitor/cluster/tau_cluster_chi2",
                        params_.cluster.tau_cluster_chi2);
  params_.cluster.tau_cluster_disappear =
      LoadParam<double>(private_nh_, "deform_monitor/cluster/tau_cluster_disappear",
                        params_.cluster.tau_cluster_disappear);

  params_.structure_correspondence.enable =
      LoadParam<bool>(private_nh_, "deform_monitor/structure_correspondence/enable",
                      params_.structure_correspondence.enable);
  params_.structure_correspondence.publish_motions =
      LoadParam<bool>(private_nh_, "deform_monitor/structure_correspondence/publish_motions",
                      params_.structure_correspondence.publish_motions);
  params_.structure_correspondence.publish_markers =
      LoadParam<bool>(private_nh_, "deform_monitor/structure_correspondence/publish_markers",
                      params_.structure_correspondence.publish_markers);
  params_.structure_correspondence.old_min_anchor_count =
      LoadParam<int>(private_nh_, "deform_monitor/structure_correspondence/old_min_anchor_count",
                     params_.structure_correspondence.old_min_anchor_count);
  params_.structure_correspondence.new_min_anchor_count =
      LoadParam<int>(private_nh_, "deform_monitor/structure_correspondence/new_min_anchor_count",
                     params_.structure_correspondence.new_min_anchor_count);
  params_.structure_correspondence.max_match_distance =
      LoadParam<double>(private_nh_, "deform_monitor/structure_correspondence/max_match_distance",
                        params_.structure_correspondence.max_match_distance);
  params_.structure_correspondence.max_size_gap =
      LoadParam<double>(private_nh_, "deform_monitor/structure_correspondence/max_size_gap",
                        params_.structure_correspondence.max_size_gap);
  params_.structure_correspondence.max_normal_deg =
      LoadParam<double>(private_nh_, "deform_monitor/structure_correspondence/max_normal_deg",
                        params_.structure_correspondence.max_normal_deg);
  params_.structure_correspondence.max_type_l1 =
      LoadParam<double>(private_nh_, "deform_monitor/structure_correspondence/max_type_l1",
                        params_.structure_correspondence.max_type_l1);
  params_.structure_correspondence.max_match_cost =
      LoadParam<double>(private_nh_, "deform_monitor/structure_correspondence/max_match_cost",
                        params_.structure_correspondence.max_match_cost);
  params_.structure_correspondence.min_confidence =
      LoadParam<double>(private_nh_, "deform_monitor/structure_correspondence/min_confidence",
                        params_.structure_correspondence.min_confidence);
  params_.structure_correspondence.min_motion_distance =
      LoadParam<double>(private_nh_, "deform_monitor/structure_correspondence/min_motion_distance",
                        params_.structure_correspondence.min_motion_distance);
  params_.structure_correspondence.weight_dist =
      LoadParam<double>(private_nh_, "deform_monitor/structure_correspondence/weight_dist",
                        params_.structure_correspondence.weight_dist);
  params_.structure_correspondence.weight_size =
      LoadParam<double>(private_nh_, "deform_monitor/structure_correspondence/weight_size",
                        params_.structure_correspondence.weight_size);
  params_.structure_correspondence.weight_normal =
      LoadParam<double>(private_nh_, "deform_monitor/structure_correspondence/weight_normal",
                        params_.structure_correspondence.weight_normal);
  params_.structure_correspondence.weight_type =
      LoadParam<double>(private_nh_, "deform_monitor/structure_correspondence/weight_type",
                        params_.structure_correspondence.weight_type);
  params_.structure_correspondence.weight_motion =
      LoadParam<double>(private_nh_, "deform_monitor/structure_correspondence/weight_motion",
                        params_.structure_correspondence.weight_motion);
  params_.structure_correspondence.weight_persistence =
      LoadParam<double>(private_nh_, "deform_monitor/structure_correspondence/weight_persistence",
                        params_.structure_correspondence.weight_persistence);
  params_.structure_correspondence.old_score_threshold =
      LoadParam<double>(private_nh_, "deform_monitor/structure_correspondence/old_score_threshold",
                        params_.structure_correspondence.old_score_threshold);
  params_.structure_correspondence.new_disp_threshold =
      LoadParam<double>(private_nh_, "deform_monitor/structure_correspondence/new_disp_threshold",
                        params_.structure_correspondence.new_disp_threshold);
  params_.structure_correspondence.marker_arrow_scale =
      LoadParam<double>(private_nh_, "deform_monitor/structure_correspondence/marker_arrow_scale",
                        params_.structure_correspondence.marker_arrow_scale);
  params_.structure_correspondence.marker_outline_width =
      LoadParam<double>(private_nh_, "deform_monitor/structure_correspondence/marker_outline_width",
                        params_.structure_correspondence.marker_outline_width);
  params_.structure_correspondence.marker_old_alpha =
      LoadParam<double>(private_nh_, "deform_monitor/structure_correspondence/marker_old_alpha",
                        params_.structure_correspondence.marker_old_alpha);
  params_.structure_correspondence.marker_new_alpha =
      LoadParam<double>(private_nh_, "deform_monitor/structure_correspondence/marker_new_alpha",
                        params_.structure_correspondence.marker_new_alpha);
  params_.structure_correspondence.marker_arrow_alpha =
      LoadParam<double>(private_nh_, "deform_monitor/structure_correspondence/marker_arrow_alpha",
                        params_.structure_correspondence.marker_arrow_alpha);
  if (params_.cluster.spatial_hash_size <= 0.0) {
    params_.cluster.spatial_hash_size = params_.anchor.voxel_size;
  }
  params_.visualization.show_cluster_text =
      LoadParam<bool>(private_nh_, "deform_monitor/visualization/show_cluster_text",
                      params_.visualization.show_cluster_text);
  params_.visualization.show_all_anchors =
      LoadParam<bool>(private_nh_, "deform_monitor/visualization/show_all_anchors",
                      params_.visualization.show_all_anchors);
  params_.visualization.show_comparable_anchors =
      LoadParam<bool>(private_nh_, "deform_monitor/visualization/show_comparable_anchors",
                      params_.visualization.show_comparable_anchors);
  params_.visualization.show_cluster_boxes =
      LoadParam<bool>(private_nh_, "deform_monitor/visualization/show_cluster_boxes",
                      params_.visualization.show_cluster_boxes);
  params_.visualization.text_only_significant =
      LoadParam<bool>(private_nh_, "deform_monitor/visualization/text_only_significant",
                      params_.visualization.text_only_significant);
  params_.visualization.arrows_only_clustered_or_reacquired =
      LoadParam<bool>(private_nh_, "deform_monitor/visualization/arrows_only_clustered_or_reacquired",
                      params_.visualization.arrows_only_clustered_or_reacquired);
  params_.visualization.min_arrow_disp =
      LoadParam<double>(private_nh_, "deform_monitor/visualization/min_arrow_disp",
                        params_.visualization.min_arrow_disp);
  params_.visualization.min_arrow_contrast_score =
      LoadParam<double>(private_nh_, "deform_monitor/visualization/min_arrow_contrast_score",
                        params_.visualization.min_arrow_contrast_score);
  params_.visualization.max_arrow_disp =
      LoadParam<double>(private_nh_, "deform_monitor/visualization/max_arrow_disp",
                        params_.visualization.max_arrow_disp);
  params_.visualization.arrow_disp_scale =
      LoadParam<double>(private_nh_, "deform_monitor/visualization/arrow_disp_scale",
                        params_.visualization.arrow_disp_scale);
  params_.visualization.arrow_shaft_diameter =
      LoadParam<double>(private_nh_, "deform_monitor/visualization/arrow_shaft_diameter",
                        params_.visualization.arrow_shaft_diameter);
  params_.visualization.arrow_head_diameter =
      LoadParam<double>(private_nh_, "deform_monitor/visualization/arrow_head_diameter",
                        params_.visualization.arrow_head_diameter);
  params_.visualization.arrow_head_length =
      LoadParam<double>(private_nh_, "deform_monitor/visualization/arrow_head_length",
                        params_.visualization.arrow_head_length);
  params_.visualization.cluster_box_alpha =
      LoadParam<double>(private_nh_, "deform_monitor/visualization/cluster_box_alpha",
                        params_.visualization.cluster_box_alpha);
  params_.visualization.cluster_outline_alpha =
      LoadParam<double>(private_nh_, "deform_monitor/visualization/cluster_outline_alpha",
                        params_.visualization.cluster_outline_alpha);
  params_.visualization.cluster_outline_width =
      LoadParam<double>(private_nh_, "deform_monitor/visualization/cluster_outline_width",
                        params_.visualization.cluster_outline_width);
  params_.visualization.cluster_min_box_size =
      LoadParam<double>(private_nh_, "deform_monitor/visualization/cluster_min_box_size",
                        params_.visualization.cluster_min_box_size);
  params_.risk_visualization.enable =
      LoadParam<bool>(private_nh_, "deform_monitor/risk_visualization/enable",
                      params_.risk_visualization.enable);
  params_.risk_visualization.publish_evidence =
      LoadParam<bool>(private_nh_, "deform_monitor/risk_visualization/publish_evidence",
                      params_.risk_visualization.publish_evidence);
  params_.risk_visualization.publish_voxels =
      LoadParam<bool>(private_nh_, "deform_monitor/risk_visualization/publish_voxels",
                      params_.risk_visualization.publish_voxels);
  params_.risk_visualization.publish_regions =
      LoadParam<bool>(private_nh_, "deform_monitor/risk_visualization/publish_regions",
                      params_.risk_visualization.publish_regions);
  params_.risk_visualization.publish_markers =
      LoadParam<bool>(private_nh_, "deform_monitor/risk_visualization/publish_markers",
                      params_.risk_visualization.publish_markers);
  params_.risk_visualization.voxel_size =
      LoadParam<double>(private_nh_, "deform_monitor/risk_visualization/voxel_size",
                        params_.risk_visualization.voxel_size);
  params_.risk_visualization.kernel_sigma =
      LoadParam<double>(private_nh_, "deform_monitor/risk_visualization/kernel_sigma",
                        params_.risk_visualization.kernel_sigma);
  params_.risk_visualization.kernel_radius =
      LoadParam<double>(private_nh_, "deform_monitor/risk_visualization/kernel_radius",
                        params_.risk_visualization.kernel_radius);
  params_.risk_visualization.min_confidence =
      LoadParam<double>(private_nh_, "deform_monitor/risk_visualization/min_confidence",
                        params_.risk_visualization.min_confidence);
  params_.risk_visualization.min_graph_neighbors =
      LoadParam<int>(private_nh_, "deform_monitor/risk_visualization/min_graph_neighbors",
                     params_.risk_visualization.min_graph_neighbors);
  params_.risk_visualization.min_risk_score =
      LoadParam<double>(private_nh_, "deform_monitor/risk_visualization/min_risk_score",
                        params_.risk_visualization.min_risk_score);
  params_.risk_visualization.min_voxel_risk =
      LoadParam<double>(private_nh_, "deform_monitor/risk_visualization/min_voxel_risk",
                        params_.risk_visualization.min_voxel_risk);
  params_.risk_visualization.min_region_voxels =
      LoadParam<int>(private_nh_, "deform_monitor/risk_visualization/min_region_voxels",
                     params_.risk_visualization.min_region_voxels);
  params_.risk_visualization.min_region_mean_risk =
      LoadParam<double>(private_nh_, "deform_monitor/risk_visualization/min_region_mean_risk",
                        params_.risk_visualization.min_region_mean_risk);
  params_.risk_visualization.low_risk_alpha =
      LoadParam<double>(private_nh_, "deform_monitor/risk_visualization/low_risk_alpha",
                        params_.risk_visualization.low_risk_alpha);
  params_.risk_visualization.high_risk_alpha =
      LoadParam<double>(private_nh_, "deform_monitor/risk_visualization/high_risk_alpha",
                        params_.risk_visualization.high_risk_alpha);
  params_.risk_visualization.region_outline_alpha =
      LoadParam<double>(private_nh_, "deform_monitor/risk_visualization/region_outline_alpha",
                        params_.risk_visualization.region_outline_alpha);
  params_.risk_visualization.region_outline_width =
      LoadParam<double>(private_nh_, "deform_monitor/risk_visualization/region_outline_width",
                        params_.risk_visualization.region_outline_width);

  params_.persistent_risk.enable =
      LoadParam<bool>(private_nh_, "deform_monitor/persistent_risk/enable",
                      params_.persistent_risk.enable);
  params_.persistent_risk.max_center_distance =
      LoadParam<double>(private_nh_, "deform_monitor/persistent_risk/max_center_distance",
                        params_.persistent_risk.max_center_distance);
  params_.persistent_risk.min_bbox_iou =
      LoadParam<double>(private_nh_, "deform_monitor/persistent_risk/min_bbox_iou",
                        params_.persistent_risk.min_bbox_iou);
  params_.persistent_risk.max_risk_gap =
      LoadParam<double>(private_nh_, "deform_monitor/persistent_risk/max_risk_gap",
                        params_.persistent_risk.max_risk_gap);
  params_.persistent_risk.ema_alpha =
      LoadParam<double>(private_nh_, "deform_monitor/persistent_risk/ema_alpha",
                        params_.persistent_risk.ema_alpha);
  params_.persistent_risk.window_size =
      LoadParam<int>(private_nh_, "deform_monitor/persistent_risk/window_size",
                     params_.persistent_risk.window_size);
  params_.persistent_risk.min_hits_to_confirm =
      LoadParam<int>(private_nh_, "deform_monitor/persistent_risk/min_hits_to_confirm",
                     params_.persistent_risk.min_hits_to_confirm);
  params_.persistent_risk.min_hit_streak_to_confirm =
      LoadParam<int>(private_nh_, "deform_monitor/persistent_risk/min_hit_streak_to_confirm",
                     params_.persistent_risk.min_hit_streak_to_confirm);
  params_.persistent_risk.min_confirmed_mean_risk =
      LoadParam<double>(private_nh_, "deform_monitor/persistent_risk/min_confirmed_mean_risk",
                        params_.persistent_risk.min_confirmed_mean_risk);
  params_.persistent_risk.min_confirmed_confidence =
      LoadParam<double>(private_nh_, "deform_monitor/persistent_risk/min_confirmed_confidence",
                        params_.persistent_risk.min_confirmed_confidence);
  params_.persistent_risk.min_confirmed_support_mass =
      LoadParam<double>(private_nh_, "deform_monitor/persistent_risk/min_confirmed_support_mass",
                        params_.persistent_risk.min_confirmed_support_mass);
  params_.persistent_risk.min_confirmed_span =
      LoadParam<double>(private_nh_, "deform_monitor/persistent_risk/min_confirmed_span",
                        params_.persistent_risk.min_confirmed_span);
  params_.persistent_risk.miss_frames_to_fading =
      LoadParam<int>(private_nh_, "deform_monitor/persistent_risk/miss_frames_to_fading",
                     params_.persistent_risk.miss_frames_to_fading);
  params_.persistent_risk.miss_frames_to_delete =
      LoadParam<int>(private_nh_, "deform_monitor/persistent_risk/miss_frames_to_delete",
                     params_.persistent_risk.miss_frames_to_delete);
  params_.persistent_risk.fading_risk_floor =
      LoadParam<double>(private_nh_, "deform_monitor/persistent_risk/fading_risk_floor",
                        params_.persistent_risk.fading_risk_floor);
  params_.persistent_risk.allow_sparse_planar_regions =
      LoadParam<bool>(private_nh_, "deform_monitor/persistent_risk/allow_sparse_planar_regions",
                      params_.persistent_risk.allow_sparse_planar_regions);

  params_.structure_unit.region_spatial_hash =
      LoadParam<double>(private_nh_, "structure_unit/region_spatial_hash",
                        params_.structure_unit.region_spatial_hash);
  params_.structure_unit.region_radius =
      LoadParam<double>(private_nh_, "structure_unit/region_radius",
                        params_.structure_unit.region_radius);
  params_.structure_unit.region_normal_deg =
      LoadParam<double>(private_nh_, "structure_unit/region_normal_deg",
                        params_.structure_unit.region_normal_deg);
  params_.structure_unit.region_edge_dir_deg =
      LoadParam<double>(private_nh_, "structure_unit/region_edge_dir_deg",
                        params_.structure_unit.region_edge_dir_deg);
  params_.structure_unit.region_min_members =
      LoadParam<int>(private_nh_, "structure_unit/region_min_members",
                     params_.structure_unit.region_min_members);

  params_.structure_tracker.tau_exit =
      LoadParam<double>(private_nh_, "structure_tracker/tau_exit",
                        params_.structure_tracker.tau_exit);
  params_.structure_tracker.search_margin =
      LoadParam<double>(private_nh_, "structure_tracker/search_margin",
                        params_.structure_tracker.search_margin);
  params_.structure_tracker.orphan_radius =
      LoadParam<double>(private_nh_, "structure_tracker/orphan_radius",
                        params_.structure_tracker.orphan_radius);
  params_.structure_tracker.orphan_voxel_size =
      LoadParam<double>(private_nh_, "structure_tracker/orphan_voxel_size",
                        params_.structure_tracker.orphan_voxel_size);
  params_.structure_tracker.max_size_gap =
      LoadParam<double>(private_nh_, "structure_tracker/max_size_gap",
                        params_.structure_tracker.max_size_gap);
  params_.structure_tracker.max_normal_deg =
      LoadParam<double>(private_nh_, "structure_tracker/max_normal_deg",
                        params_.structure_tracker.max_normal_deg);
  params_.structure_tracker.min_migration_dist =
      LoadParam<double>(private_nh_, "structure_tracker/min_migration_dist",
                        params_.structure_tracker.min_migration_dist);
  params_.structure_tracker.max_migration_dist =
      LoadParam<double>(private_nh_, "structure_tracker/max_migration_dist",
                        params_.structure_tracker.max_migration_dist);
  params_.structure_tracker.min_migration_confidence =
      LoadParam<double>(private_nh_, "structure_tracker/min_migration_confidence",
                        params_.structure_tracker.min_migration_confidence);
  params_.structure_tracker.orphan_min_points =
      LoadParam<int>(private_nh_, "structure_tracker/orphan_min_points",
                     params_.structure_tracker.orphan_min_points);
  params_.structure_tracker.persistence_confidence =
      LoadParam<double>(private_nh_, "structure_tracker/persistence_confidence",
                        params_.structure_tracker.persistence_confidence);

  params_.incremental.enable =
      LoadParam<bool>(private_nh_, "deform_monitor/incremental/enable",
                      params_.incremental.enable);
  params_.incremental.coverage_radius =
      LoadParam<double>(private_nh_, "deform_monitor/incremental/coverage_radius",
                        params_.incremental.coverage_radius);
  params_.incremental.warmup_frames =
      LoadParam<int>(private_nh_, "deform_monitor/incremental/warmup_frames",
                     params_.incremental.warmup_frames);
  params_.incremental.min_visible_frames =
      LoadParam<int>(private_nh_, "deform_monitor/incremental/min_visible_frames",
                     params_.incremental.min_visible_frames);
  params_.incremental.max_new_anchors =
      LoadParam<int>(private_nh_, "deform_monitor/incremental/max_new_anchors",
                     params_.incremental.max_new_anchors);

  runtime_output_dir_ =
      LoadParam<std::string>(private_nh_, "deform_monitor/runtime/output_dir", runtime_output_dir_);
  runtime_output_dir_param_name_ = LoadParam<std::string>(
      private_nh_,
      "deform_monitor/runtime/output_dir_param",
      runtime_output_dir_param_name_);

  params_.ablation.variant =
      LoadParam<std::string>(private_nh_, "deform_monitor/ablation/variant", params_.ablation.variant);
  params_.ablation.disable_covariance_inflation =
      LoadParam<bool>(private_nh_,
                      "deform_monitor/ablation/disable_covariance_inflation",
                      params_.ablation.disable_covariance_inflation);
  params_.ablation.disable_type_constraint =
      LoadParam<bool>(private_nh_,
                      "deform_monitor/ablation/disable_type_constraint",
                      params_.ablation.disable_type_constraint);
  params_.ablation.single_model_ekf =
      LoadParam<bool>(private_nh_,
                      "deform_monitor/ablation/single_model_ekf",
                      params_.ablation.single_model_ekf);
  params_.ablation.disable_cusum =
      LoadParam<bool>(private_nh_,
                      "deform_monitor/ablation/disable_cusum",
                      params_.ablation.disable_cusum);
  params_.ablation.disable_directional_accumulation =
      LoadParam<bool>(private_nh_,
                      "deform_monitor/ablation/disable_directional_accumulation",
                      params_.ablation.disable_directional_accumulation);
  params_.ablation.disable_drift_compensation =
      LoadParam<bool>(private_nh_,
                      "deform_monitor/ablation/disable_drift_compensation",
                      params_.ablation.disable_drift_compensation);

  ApplyAblationOverrides();

  anchor_builder_.SetParams(params_.anchor);
  scalar_builder_.SetParams(params_.observation, params_.noise);
  obs_extractor_.SetParams(params_.observation);
  obs_extractor_.SetTemporalParams(params_.temporal);
  obs_extractor_.SetNoiseParams(params_.noise);
  obs_extractor_.SetCovarianceParams(params_.covariance);
  obs_extractor_.SetObservabilityParams(params_.observability);
  obs_extractor_.SetMeasurementBuilder(scalar_builder_);
  imm_filter_.SetParams(params_.imm, params_.observability, params_.significance,
                        params_.directional_motion, params_.reference.tau_mu0);
  clusterer_.SetParams(params_.cluster);
  region_hypothesis_builder_.SetParams(params_.structure_correspondence);
  region_correspondence_solver_.SetParams(params_.structure_correspondence);
  risk_adapter_.SetParams(params_.risk_visualization, params_.significance,
                          params_.graph_temporal);
  risk_field_builder_.SetParams(params_.risk_visualization);
  ref_manager_.SetParams(params_.reference);
  risk_viz_publisher_.SetParams(params_.risk_visualization);
  persistent_risk_tracker_.SetParams(params_.persistent_risk);
  structure_viz_publisher_.SetParams(params_.structure_correspondence);
  viz_publisher_.SetParams(params_.visualization, nh_);
}

void DeformMonitorV2Node::ApplyAblationOverrides() {
  if (params_.ablation.disable_covariance_inflation) {
    params_.covariance.alpha_xi = 1.0;
  }
  if (params_.ablation.disable_type_constraint) {
    params_.imm.enable_type_constraint = false;
  }
  if (params_.ablation.single_model_ekf) {
    params_.imm.enable_model_competition = false;
  }
  if (params_.ablation.disable_cusum) {
    params_.significance.enable_cusum = false;
  }
  if (params_.ablation.disable_directional_accumulation) {
    params_.directional_motion.enable = false;
  }
  if (params_.ablation.disable_drift_compensation) {
    params_.background_bias.enable = false;
  }
}

void DeformMonitorV2Node::Run() {
  {
    std::lock_guard<std::mutex> lock(data_mutex_);
    stop_worker_ = false;
  }
  worker_thread_ = std::thread(&DeformMonitorV2Node::WorkerLoop, this);
  ros::spin();
  {
    std::lock_guard<std::mutex> lock(data_mutex_);
    stop_worker_ = true;
  }
  data_cv_.notify_all();
  if (worker_thread_.joinable()) {
    worker_thread_.join();
  }
  ShutdownDiagnosticsLog();
}

void DeformMonitorV2Node::WorkerLoop() {
  while (true) {
    {
      std::unique_lock<std::mutex> lock(data_mutex_);
      data_cv_.wait(lock, [this]() {
        return stop_worker_ || (!cloud_queue_.empty() && !covariance_queue_.empty());
      });
      if (stop_worker_) {
        return;
      }
    }
    TryProcessQueuedFrames();
  }
}

void DeformMonitorV2Node::CloudCallback(const sensor_msgs::PointCloud2ConstPtr& msg) {
  {
    std::lock_guard<std::mutex> lock(data_mutex_);
    cloud_queue_.push_back(TimedCloud{*msg});
    TrimQueue(&cloud_queue_, 200);
    ++cloud_msg_count_;
    last_cloud_stamp_ = msg->header.stamp;
    if (!first_cloud_logged_) {
      std::cout << "[deform_monitor_v2] First cloud: stamp=" << msg->header.stamp.toSec()
                << " frame_id=" << msg->header.frame_id
                << " width=" << msg->width
                << " height=" << msg->height << std::endl;
      first_cloud_logged_ = true;
    }
  }
  data_cv_.notify_one();
  MaybePrintPipelineStatus();
}

void DeformMonitorV2Node::CovarianceCallback(const fast_lio::LioOdomCovConstPtr& msg) {
  {
    std::lock_guard<std::mutex> lock(data_mutex_);
    covariance_queue_.push_back(TimedCov{*msg});
    TrimQueue(&covariance_queue_, 400);
    ++covariance_msg_count_;
    last_covariance_stamp_ = msg->odom.header.stamp;
    if (!first_covariance_logged_) {
      std::cout << "[deform_monitor_v2] First covariance: stamp=" << msg->odom.header.stamp.toSec()
                << " state_dim=" << msg->state_dim
                << " cov_size=" << msg->state_covariance.size() << std::endl;
      first_covariance_logged_ = true;
    }
  }
  data_cv_.notify_one();
  MaybePrintPipelineStatus();
}

void DeformMonitorV2Node::ResetReferenceCallback(const std_msgs::EmptyConstPtr& /*msg*/) {
  {
    std::lock_guard<std::mutex> processing_lock(processing_mutex_);
    std::lock_guard<std::mutex> data_lock(data_mutex_);
    ResetReferenceStateLocked();
  }
  PublishEmptyResults(ros::Time::now());
  if (diagnostics_log_.is_open()) {
    diagnostics_log_ << "[RESET] wall_time=" << ros::WallTime::now().toSec()
                     << " reason=manual_reset_reference"
                     << std::endl;
    diagnostics_log_.flush();
  }
  std::cout << "[deform_monitor_v2] Manual reference reset received. Reinit on next synced frame."
            << std::endl;
}

double DeformMonitorV2Node::TimeDistanceSec(const ros::Time& a, const ros::Time& b) {
  return std::abs((a - b).toSec());
}

bool DeformMonitorV2Node::EstimateBackgroundBias(const CurrentObservationVector& observations,
                                                 Eigen::Vector3d* bias_R,
                                                 size_t* used_anchor_count,
                                                 size_t* used_scalar_count,
                                                 size_t* total_comparable_plane_count) const {
  if (!bias_R) {
    return false;
  }
  *bias_R = Eigen::Vector3d::Zero();
  if (used_anchor_count) {
    *used_anchor_count = 0;
  }
  if (used_scalar_count) {
    *used_scalar_count = 0;
  }
  if (total_comparable_plane_count) {
    *total_comparable_plane_count = 0;
  }
  if (!params_.background_bias.enable || observations.size() != anchors_.size()) {
    return false;
  }

  struct ScalarEq {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Eigen::Vector3d h = Eigen::Vector3d::Zero();
    double z = 0.0;
    double r = 1.0;
  };



  struct CandidateAnchor {
    size_t obs_index;
    double disp_norm;
  };
  std::vector<CandidateAnchor> candidates;
  candidates.reserve(128);

  for (size_t i = 0; i < observations.size(); ++i) {
    const auto& obs = observations[i];
    const auto& anchor = anchors_[i];
    if (anchor.type != AnchorType::PLANE) {
      continue;
    }
    if (!obs.comparable || obs.reacquired) {
      continue;
    }
    if (obs.support_count < params_.background_bias.min_support_points) {
      continue;
    }
    if (obs.cmp_score < params_.background_bias.min_cmp_score) {
      continue;
    }
    candidates.push_back({i, obs.matched_delta_R.norm()});
  }

  if (total_comparable_plane_count) {
    *total_comparable_plane_count = candidates.size();
  }


  const double thresholds[] = {
      params_.background_bias.max_anchor_disp,
      params_.background_bias.adaptive_disp_step1,
      params_.background_bias.adaptive_disp_step2};
  const size_t min_anchors =
      static_cast<size_t>(std::max(1, params_.background_bias.min_anchor_count));
  const size_t min_scalars =
      static_cast<size_t>(std::max(1, params_.background_bias.min_scalar_count));

  for (const double threshold : thresholds) {
    AlignedVector<ScalarEq> eqs;
    eqs.reserve(256);
    size_t anchor_count = 0;

    for (const auto& cand : candidates) {
      if (cand.disp_norm > threshold) {
        continue;
      }
      const auto& obs = observations[cand.obs_index];
      size_t added_for_anchor = 0;
      for (const auto& scalar : obs.scalars) {
        if (scalar.type != 0 && scalar.type != 2) {
          continue;
        }
        ScalarEq eq;
        eq.h = scalar.h_R;
        eq.z = scalar.z;
        eq.r = std::max(1.0e-9, scalar.r);
        eqs.push_back(eq);
        ++added_for_anchor;
      }
      if (added_for_anchor > 0) {
        ++anchor_count;
      }
    }

    if (anchor_count < min_anchors || eqs.size() < min_scalars) {
      continue;
    }


    Eigen::Vector3d b = Eigen::Vector3d::Zero();
    const double ridge = std::max(1.0e-12, params_.background_bias.ridge_lambda);
    const double huber_delta = std::max(1.0e-4, params_.background_bias.huber_delta);
    for (int iter = 0; iter < 3; ++iter) {
      Eigen::Matrix3d A = ridge * Eigen::Matrix3d::Identity();
      Eigen::Vector3d rhs = Eigen::Vector3d::Zero();
      for (const auto& eq : eqs) {
        const double residual = eq.z - eq.h.dot(b);
        const double robust_w =
            std::min(1.0, huber_delta / std::max(huber_delta, std::abs(residual)));
        const double w = robust_w / eq.r;
        A += w * (eq.h * eq.h.transpose());
        rhs += w * eq.h * eq.z;
      }
      Eigen::LDLT<Eigen::Matrix3d> ldlt(A);
      if (ldlt.info() != Eigen::Success) {
        return false;
      }
      b = ldlt.solve(rhs);
    }

    if (!b.allFinite() || b.norm() > params_.background_bias.max_bias_norm) {
      return false;
    }

    *bias_R = b;
    if (used_anchor_count) {
      *used_anchor_count = anchor_count;
    }
    if (used_scalar_count) {
      *used_scalar_count = eqs.size();
    }
    return true;
  }

  return false;
}

void DeformMonitorV2Node::ApplyBackgroundBias(const Eigen::Vector3d& bias_R,
                                              CurrentObservationVector* observations) const {
  if (!observations) {
    return;
  }
  for (auto& obs : *observations) {
    for (auto& scalar : obs.scalars) {
      scalar.z -= scalar.h_R.dot(bias_R);
    }
    if (obs.matched_delta_R.allFinite()) {
      obs.matched_delta_R -= bias_R;
      obs.matched_center_R -= bias_R;
    }
  }
}

void DeformMonitorV2Node::UpdateLocalContrastStates() {
  for (auto& state : anchor_states_) {
    state.local_bg_count = 0;
    state.local_bg_disp_norm = 0.0;
    state.local_bg_sigma = 0.0;
    state.local_contrast_score = 0.0;
    state.local_rel_norm = state.disp_norm;
    state.local_rel_normal = state.disp_normal;
    state.local_rel_edge = state.disp_edge;
    state.plane_bg_count = 0;
    state.plane_bg_disp_norm = 0.0;
    state.plane_bg_sigma = 0.0;
    state.plane_contrast_score = 0.0;
    state.plane_rel_norm = state.disp_norm;
    state.plane_rel_normal = state.disp_normal;
    state.plane_rel_edge = state.disp_edge;
  }

  if (!params_.local_contrast.enable || anchors_.size() != anchor_states_.size() ||
      observations_.size() != anchors_.size()) {
    return;
  }

  const double radius = std::max(0.05, params_.local_contrast.radius);
  const double sigma_dist = std::max(0.05, 0.6 * radius);
  const double sigma_dist2 = sigma_dist * sigma_dist;
  const double min_bg_sigma =
      std::max(1.0e-4, params_.local_contrast.min_background_sigma);
  const double max_background_disp =
      std::max(1.0e-4, params_.local_contrast.max_background_disp);
  const auto compute_bg_stats =
      [&](const AnchorReference& anchor,
          const AnchorTrackState& state,
          const std::vector<std::pair<double, Eigen::Vector3d>>& samples,
          int* bg_count,
          double* bg_disp_norm,
          double* bg_sigma_out,
          double* contrast_score,
          double* rel_norm,
          double* rel_normal,
          double* rel_edge) {
        if (!bg_count || !bg_disp_norm || !bg_sigma_out || !contrast_score || !rel_norm ||
            !rel_normal || !rel_edge) {
          return;
        }
        if (samples.empty()) {
          return;
        }
        Eigen::Vector3d weighted_bg = Eigen::Vector3d::Zero();
        double weight_sum = 0.0;
        for (const auto& sample : samples) {
          weighted_bg += sample.first * sample.second;
          weight_sum += sample.first;
        }
        if (weight_sum < 1.0e-9) {
          return;
        }
        const Eigen::Vector3d bg_mean = weighted_bg / weight_sum;
        double weighted_var = 0.0;
        for (const auto& sample : samples) {
          weighted_var += sample.first * (sample.second - bg_mean).squaredNorm();
        }
        weighted_var /= weight_sum;
        const double bg_sigma =
            std::sqrt(std::max(min_bg_sigma * min_bg_sigma, weighted_var / 3.0));

        const Eigen::Vector3d ui = state.x_mix.block<3, 1>(0, 0);
        const Eigen::Vector3d rel = ui - bg_mean;
        const double rel_norm_v = rel.norm();
        const double rel_normal_v = std::abs(anchor.normal_R.dot(rel));
        const double rel_edge_v = std::abs(anchor.edge_normal_R.dot(rel));
        const double signal = anchor.type == AnchorType::PLANE
                                  ? std::max(rel_norm_v, 1.5 * rel_normal_v)
                                  : std::max(rel_norm_v,
                                             std::max(1.2 * rel_edge_v, rel_normal_v));

        *bg_count = static_cast<int>(samples.size());
        *bg_disp_norm = bg_mean.norm();
        *bg_sigma_out = bg_sigma;
        *contrast_score = signal / bg_sigma;
        *rel_norm = rel_norm_v;
        *rel_normal = rel_normal_v;
        *rel_edge = rel_edge_v;
      };

  for (size_t i = 0; i < anchors_.size(); ++i) {
    std::vector<std::pair<double, Eigen::Vector3d>> bg_samples;
    bg_samples.reserve(32);
    std::vector<std::pair<double, Eigen::Vector3d>> plane_bg_samples;
    plane_bg_samples.reserve(16);

    const bool require_plane_neighbors = anchors_[i].type == AnchorType::PLANE;
    for (const int neighbor_idx : anchors_[i].neighbor_indices) {
      const size_t j = static_cast<size_t>(neighbor_idx);
      const double dist = (anchors_[i].center_R - anchors_[j].center_R).norm();
      if (dist > radius) {
        continue;
      }
      if (!anchor_states_[j].comparable || anchor_states_[j].reacquired) {
        continue;
      }
      if (anchor_states_[j].mode == DetectionMode::DISAPPEARANCE) {
        continue;
      }
      if (observations_[j].support_count < params_.local_contrast.min_support_points ||
          observations_[j].cmp_score < params_.local_contrast.min_cmp_score) {
        continue;
      }
      if (anchor_states_[j].disp_norm > max_background_disp) {
        continue;
      }
      if (require_plane_neighbors && anchors_[j].type != AnchorType::PLANE) {
        continue;
      }
      if (AngleBetweenDeg(anchors_[i].normal_R, anchors_[j].normal_R) >
          params_.local_contrast.tau_bg_normal_deg) {
        continue;
      }

      const double w = std::exp(-(dist * dist) / (2.0 * sigma_dist2));
      const Eigen::Vector3d uj = anchor_states_[j].x_mix.block<3, 1>(0, 0);
      bg_samples.emplace_back(w, uj);
      if (params_.local_contrast.enable_plane_background_for_edges &&
          anchors_[i].type != AnchorType::PLANE &&
          anchors_[j].type == AnchorType::PLANE) {
        plane_bg_samples.emplace_back(w, uj);
      }
    }

    if (bg_samples.size() >=
        static_cast<size_t>(std::max(1, params_.local_contrast.min_neighbors))) {
      compute_bg_stats(anchors_[i], anchor_states_[i], bg_samples,
                       &anchor_states_[i].local_bg_count,
                       &anchor_states_[i].local_bg_disp_norm,
                       &anchor_states_[i].local_bg_sigma,
                       &anchor_states_[i].local_contrast_score,
                       &anchor_states_[i].local_rel_norm,
                       &anchor_states_[i].local_rel_normal,
                       &anchor_states_[i].local_rel_edge);
    }
    if (params_.local_contrast.enable_plane_background_for_edges &&
        anchors_[i].type != AnchorType::PLANE &&
        plane_bg_samples.size() >=
            static_cast<size_t>(std::max(1, params_.local_contrast.min_plane_neighbors))) {
      compute_bg_stats(anchors_[i], anchor_states_[i], plane_bg_samples,
                       &anchor_states_[i].plane_bg_count,
                       &anchor_states_[i].plane_bg_disp_norm,
                       &anchor_states_[i].plane_bg_sigma,
                       &anchor_states_[i].plane_contrast_score,
                       &anchor_states_[i].plane_rel_norm,
                       &anchor_states_[i].plane_rel_normal,
                       &anchor_states_[i].plane_rel_edge);
    }
  }
}

void DeformMonitorV2Node::UpdateGraphTemporalStates(const ros::Time& stamp) {
  for (auto& state : anchor_states_) {
    state.graph_neighbor_count = 0;
    state.graph_coherent_score = 0.0;
    state.graph_temporal_score = 0.0;
    state.graph_persistence_score = 0.0;
    state.graph_diff_norm = 0.0;
    state.graph_candidate = false;
  }

  if (!params_.graph_temporal.enable || anchors_.size() != anchor_states_.size() ||
      anchors_.empty()) {
    return;
  }

  const double radius =
      std::max(std::max(0.01, params_.graph_temporal.spatial_hash_size),
               params_.graph_temporal.radius);
  const double tau_disp_cos = std::max(-0.99, std::min(0.99, params_.graph_temporal.tau_disp_cos));
  const double tau_coherent_diff = std::max(1.0e-6, params_.graph_temporal.tau_coherent_diff);
  const double tau_temporal_change = std::max(1.0e-6, params_.graph_temporal.tau_temporal_change);
  const double alpha = std::max(0.0, std::min(0.98, params_.graph_temporal.ema_alpha));
  const double min_anchor_disp = std::max(1.0e-6, params_.graph_temporal.min_anchor_disp);

  std::vector<int> neighbor_count(anchors_.size(), 0);
  std::vector<double> coherent_sum(anchors_.size(), 0.0);
  std::vector<double> temporal_sum(anchors_.size(), 0.0);
  std::vector<double> persistence_sum(anchors_.size(), 0.0);
  std::vector<double> diff_sum(anchors_.size(), 0.0);

  auto make_edge_key = [](int a, int b) {
    if (a > b) {
      std::swap(a, b);
    }
    return EdgeKey{a, b};
  };

  for (size_t i = 0; i < anchors_.size(); ++i) {
    const AnchorTrackState& state_i = anchor_states_[i];
    if (state_i.gate_state != ObsGateState::OBSERVABLE_MATCHED ||
        !state_i.observable || !state_i.comparable) {
      continue;
    }

    const Eigen::Vector3d ui = state_i.x_mix.block<3, 1>(0, 0);
    const double ni = ui.norm();

    for (const int neighbor_idx : anchors_[i].neighbor_indices) {
      const size_t j = static_cast<size_t>(neighbor_idx);
      if (j <= i) {
        continue;
      }
      const AnchorTrackState& state_j = anchor_states_[j];
      if (state_j.gate_state != ObsGateState::OBSERVABLE_MATCHED ||
          !state_j.observable || !state_j.comparable) {
        continue;
      }
      const double dist = (anchors_[i].center_R - anchors_[j].center_R).norm();
      if (dist > radius) {
        continue;
      }
      if (AngleBetweenDeg(anchors_[i].normal_R, anchors_[j].normal_R) >
          params_.graph_temporal.tau_normal_deg) {
        continue;
      }

      const Eigen::Vector3d uj = state_j.x_mix.block<3, 1>(0, 0);
      const double nj = uj.norm();
      const double min_mag = std::min(ni, nj);
      if (min_mag < 0.5 * min_anchor_disp) {
        continue;
      }

      double dir_cos = 1.0;
      if (ni > 1.0e-6 && nj > 1.0e-6) {
        dir_cos = ui.dot(uj) / (ni * nj);
      }
      if (dir_cos < tau_disp_cos) {
        continue;
      }

      const Eigen::Vector3d delta_ij = ui - uj;
      const double diff_norm = delta_ij.norm();
      const double mag_support = Clamp01(min_mag / std::max(min_anchor_disp, 1.0e-6));
      const double dir_support =
          Clamp01((dir_cos - tau_disp_cos) / std::max(1.0e-6, 1.0 - tau_disp_cos));
      const double diff_support = std::exp(-diff_norm / tau_coherent_diff);
      const double edge_coherence = mag_support * dir_support * diff_support;
      if (edge_coherence < 1.0e-3) {
        continue;
      }

      const EdgeKey edge_key = make_edge_key(anchor_states_[i].id, anchor_states_[j].id);
      EdgeTemporalState& edge_state = edge_temporal_states_[edge_key];
      const Eigen::Vector3d prev_delta = edge_state.delta_ema;
      const double temporal_obs =
          std::exp(-(delta_ij - prev_delta).norm() / tau_temporal_change);
      edge_state.delta_ema = alpha * edge_state.delta_ema + (1.0 - alpha) * delta_ij;
      edge_state.coherent_ema =
          alpha * edge_state.coherent_ema + (1.0 - alpha) * edge_coherence;
      edge_state.temporal_consistency =
          alpha * edge_state.temporal_consistency + (1.0 - alpha) * temporal_obs;
      if (params_.significance.enable_cusum) {

        const double gt_lambda = params_.graph_temporal.cusum_lambda;
        const double gt_cap = 3.0 * params_.graph_temporal.cusum_h;
        edge_state.persistence_score =
            std::min(gt_cap,
                     std::max(0.0,
                              gt_lambda * edge_state.persistence_score +
                                  edge_coherence -
                                  params_.graph_temporal.cusum_k));
      } else {
        edge_state.persistence_score = 0.0;
      }
      if (edge_coherence > 0.2) {
        ++edge_state.valid_streak;
      } else {
        edge_state.valid_streak = std::max(0, edge_state.valid_streak - 1);
      }
      edge_state.last_update = stamp;

      ++neighbor_count[i];
      ++neighbor_count[j];
      coherent_sum[i] += edge_state.coherent_ema;
      coherent_sum[j] += edge_state.coherent_ema;
      temporal_sum[i] += edge_state.temporal_consistency;
      temporal_sum[j] += edge_state.temporal_consistency;
      persistence_sum[i] += edge_state.persistence_score;
      persistence_sum[j] += edge_state.persistence_score;
      diff_sum[i] += diff_norm;
      diff_sum[j] += diff_norm;
    }
  }

  for (size_t i = 0; i < anchor_states_.size(); ++i) {
    AnchorTrackState& state = anchor_states_[i];
    if (neighbor_count[i] <= 0) {
      continue;
    }
    const double count = static_cast<double>(neighbor_count[i]);
    state.graph_neighbor_count = neighbor_count[i];
    state.graph_coherent_score = coherent_sum[i] / count;
    state.graph_temporal_score = temporal_sum[i] / count;
    state.graph_persistence_score = persistence_sum[i] / count;
    state.graph_diff_norm = diff_sum[i] / count;

    const bool enough_neighbors =
        state.graph_neighbor_count >= std::max(1, params_.graph_temporal.min_neighbors);
    const bool matched_now =
        state.gate_state == ObsGateState::OBSERVABLE_MATCHED && state.observable && state.comparable;
    const bool coherent_graph =
        state.graph_coherent_score > params_.graph_temporal.tau_graph_support &&
        state.graph_temporal_score > params_.graph_temporal.tau_graph_temporal;
    const bool persistent_graph =
        params_.significance.enable_cusum &&
        state.graph_persistence_score > params_.graph_temporal.cusum_h;
    const bool motion_seed =
        state.disp_norm > min_anchor_disp ||
        (params_.significance.enable_cusum &&
         state.cusum_score > 0.5 * params_.significance.cusum_h);
    state.graph_candidate = matched_now && enough_neighbors && motion_seed &&
                            (coherent_graph || persistent_graph);
  }
}

bool DeformMonitorV2Node::PopSynchronizedFrame(sensor_msgs::PointCloud2* cloud_msg,
                                               fast_lio::LioOdomCov* cov_msg) {
  constexpr double kSyncToleranceSec = 0.05;
  if (!cloud_msg || !cov_msg) {
    return false;
  }

  while (!cloud_queue_.empty() && !covariance_queue_.empty()) {
    const ros::Time cloud_stamp = cloud_queue_.front().msg.header.stamp;
    size_t cov_idx = covariance_queue_.size();
    double best_dist = kSyncToleranceSec;
    for (size_t i = 0; i < covariance_queue_.size(); ++i) {
      const ros::Time candidate_stamp = covariance_queue_[i].msg.odom.header.stamp;
      const double dist = std::abs((candidate_stamp - cloud_stamp).toSec());
      if (dist <= best_dist) {
        best_dist = dist;
        cov_idx = i;
      }
    }

    if (cov_idx < covariance_queue_.size()) {
      *cloud_msg = cloud_queue_.front().msg;
      *cov_msg = covariance_queue_[cov_idx].msg;
      cloud_queue_.pop_front();
      covariance_queue_.erase(covariance_queue_.begin(),
                              covariance_queue_.begin() + static_cast<long>(cov_idx + 1));
      return true;
    }

    const ros::Time cov_stamp = covariance_queue_.front().msg.odom.header.stamp;
    if (cov_stamp + ros::Duration(kSyncToleranceSec) < cloud_stamp) {
      covariance_queue_.pop_front();
      continue;
    }
    if (cloud_stamp + ros::Duration(kSyncToleranceSec) < cov_stamp) {
      cloud_queue_.pop_front();
      continue;
    }
    return false;
  }

  return false;
}

void DeformMonitorV2Node::ResetReferenceStateLocked() {
  init_frame_count_ = 0;
  reference_ready_ = false;
  init_frames_.clear();
  temporal_window_frames_.clear();
  frames_since_last_window_process_ = 0;
  edge_temporal_states_.clear();

  anchors_.clear();
  anchor_states_.clear();
  observations_.clear();
  clusters_.clear();
  old_regions_.clear();
  new_regions_.clear();
  structure_motions_.clear();
  risk_evidence_.clear();
  risk_voxels_.clear();
  risk_regions_.clear();
  persistent_risk_tracks_.clear();
  persistent_risk_tracker_.Reset();

  structure_unit_tracker_.reset();
  structure_units_.clear();
  structure_migrations_.clear();

  cloud_queue_.clear();
  covariance_queue_.clear();

  incremental_init_frames_.clear();
  incremental_frame_count_ = 0;

  last_background_bias_R_.setZero();
  smoothed_background_bias_R_.setZero();
  has_previous_bias_ = false;
  drift_estimation_degraded_ = false;
  last_background_bias_anchor_count_ = 0;
  last_background_bias_scalar_count_ = 0;
  logged_anchor_alerts_.clear();
  detection_time_base_stamp_ = ros::Time();
  last_detection_stamp_ = ros::Time();
  last_processed_stamp_ = ros::Time();
}

void DeformMonitorV2Node::BuildReferenceAdjacency() {
  if (anchors_.empty()) {
    return;
  }

  const double radius = std::max(
      params_.local_contrast.radius,
      std::max(params_.graph_temporal.radius,
               std::max(params_.cluster.tau_d, params_.cluster.compact_tau_d)));
  const double voxel_size = std::max(0.01, params_.anchor.voxel_size);
  const int neighbor_layers = std::max(1, static_cast<int>(std::ceil(radius / voxel_size)));

  for (auto& anchor : anchors_) {
    anchor.neighbor_indices.clear();
  }

  std::unordered_map<SpatialVoxelKey, std::vector<size_t>, SpatialVoxelKeyHash> voxel_to_anchor_indices;
  voxel_to_anchor_indices.reserve(anchors_.size());
  for (size_t idx = 0; idx < anchors_.size(); ++idx) {
    SpatialVoxelKey key;
    key.x = static_cast<int>(std::floor(anchors_[idx].center_R.x() / voxel_size));
    key.y = static_cast<int>(std::floor(anchors_[idx].center_R.y() / voxel_size));
    key.z = static_cast<int>(std::floor(anchors_[idx].center_R.z() / voxel_size));
    voxel_to_anchor_indices[key].push_back(idx);
  }

  for (size_t i = 0; i < anchors_.size(); ++i) {
    SpatialVoxelKey center_key;
    center_key.x = static_cast<int>(std::floor(anchors_[i].center_R.x() / voxel_size));
    center_key.y = static_cast<int>(std::floor(anchors_[i].center_R.y() / voxel_size));
    center_key.z = static_cast<int>(std::floor(anchors_[i].center_R.z() / voxel_size));
    for (int dx = -neighbor_layers; dx <= neighbor_layers; ++dx) {
      for (int dy = -neighbor_layers; dy <= neighbor_layers; ++dy) {
        for (int dz = -neighbor_layers; dz <= neighbor_layers; ++dz) {
          SpatialVoxelKey key{center_key.x + dx, center_key.y + dy, center_key.z + dz};
          const auto voxel_it = voxel_to_anchor_indices.find(key);
          if (voxel_it == voxel_to_anchor_indices.end()) {
            continue;
          }
          for (const size_t j : voxel_it->second) {
            if (j == i) {
              continue;
            }
            const double dist = (anchors_[i].center_R - anchors_[j].center_R).norm();
            if (dist > radius) {
              continue;
            }
            anchors_[i].neighbor_indices.push_back(static_cast<int>(j));
          }
        }
      }
    }
    std::sort(anchors_[i].neighbor_indices.begin(), anchors_[i].neighbor_indices.end());
    anchors_[i].neighbor_indices.erase(
        std::unique(anchors_[i].neighbor_indices.begin(), anchors_[i].neighbor_indices.end()),
        anchors_[i].neighbor_indices.end());
  }
}

void DeformMonitorV2Node::TryProcessQueuedFrames() {
  std::lock_guard<std::mutex> processing_lock(processing_mutex_);
  while (true) {
    sensor_msgs::PointCloud2 cloud_msg;
    fast_lio::LioOdomCov cov_msg;
    {
      std::lock_guard<std::mutex> lock(data_mutex_);
      if (!PopSynchronizedFrame(&cloud_msg, &cov_msg)) {
        break;
      }
    }

    FrameInput frame = BuildFrameInput(cloud_msg, cov_msg);
    ++synchronized_frame_count_;
    last_processed_stamp_ = frame.stamp;
    if (!first_sync_logged_) {
      std::cout << "[deform_monitor_v2] First synced frame: stamp=" << frame.stamp.toSec()
                << " cloud_points=" << frame.cloud->size() << std::endl;
      first_sync_logged_ = true;
    }
    InitializeReferenceIfNeeded(frame);
    if (reference_ready_) {
      if (!params_.temporal.enable) {
        ProcessFrame(frame);
        continue;
      }

      temporal_window_frames_.push_back(frame);
      ++frames_since_last_window_process_;

      const size_t max_window_frames =
          static_cast<size_t>(std::max(1, params_.temporal.window_frames));
      while (temporal_window_frames_.size() > max_window_frames) {
        temporal_window_frames_.pop_front();
      }
      while (temporal_window_frames_.size() > 1 &&
             (temporal_window_frames_.back().stamp -
              temporal_window_frames_.front().stamp).toSec() >
                 params_.temporal.max_window_sec) {
        temporal_window_frames_.pop_front();
      }

      const size_t min_frames =
          static_cast<size_t>(std::max(1, params_.temporal.min_frames));
      if (temporal_window_frames_.size() < min_frames) {
        continue;
      }

      const size_t step_frames =
          static_cast<size_t>(std::max(1, params_.temporal.step_frames));
      if (frames_since_last_window_process_ < step_frames) {
        continue;
      }

      ProcessFrameWindow(temporal_window_frames_);
      frames_since_last_window_process_ = 0;
    }
  }
}

DeformMonitorV2Node::FrameInput DeformMonitorV2Node::BuildFrameInput(
    const sensor_msgs::PointCloud2& cloud_msg,
    const fast_lio::LioOdomCov& cov_msg) const {
  FrameInput frame;
  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>());
  pcl::fromROSMsg(cloud_msg, *cloud);
  const Eigen::Isometry3d T_RL = PoseFromFastLioMsg(cov_msg);
  frame.pose_cov = PoseCovFromFastLioMsg(cov_msg,
                                         params_.io.default_rot_sigma,
                                         params_.io.default_pos_sigma);
  frame.pose_cov.Sigma_xi =
      CovarianceExtractor::InflateSigmaXi(frame.pose_cov.Sigma_xi, params_.covariance.alpha_xi);
  frame.lidar_origin_R = T_RL.translation();
  frame.stamp = cloud_msg.header.stamp.isZero() ? cov_msg.odom.header.stamp : cloud_msg.header.stamp;

  const bool cloud_in_reference =
      params_.io.cloud_already_in_reference_frame ||
      cloud_msg.header.frame_id == params_.io.reference_frame;
  if (!cloud_in_reference) {
    cloud = TransformCloud(cloud, T_RL);
  }
  frame.cloud = cloud;
  return frame;
}

void DeformMonitorV2Node::InitializeReferenceIfNeeded(const FrameInput& frame) {
  if (reference_ready_) {
    return;
  }

  ReferenceInitFrame init_frame;
  init_frame.cloud.reset(new pcl::PointCloud<pcl::PointXYZI>(*frame.cloud));
  init_frame.lidar_origin_R = frame.lidar_origin_R;
  init_frame.pose_cov = frame.pose_cov;
  init_frame.stamp = frame.stamp;
  init_frames_.push_back(std::move(init_frame));
  ++init_frame_count_;

  if (init_frame_count_ < params_.reference.init_frames) {
    return;
  }

  anchors_ = anchor_builder_.BuildFrozenAnchors(init_frames_);
  BuildReferenceAdjacency();
  anchor_states_.clear();
  anchor_states_.reserve(anchors_.size());
  for (const auto& anchor : anchors_) {
    AnchorTrackState state;
    state.id = anchor.id;
    state.type = anchor.type;
    state.last_update = frame.stamp;
    imm_filter_.InitializeAnchorState(&state);
    anchor_states_.push_back(state);
  }
  observations_.resize(anchors_.size());
  reference_ready_ = !anchors_.empty();
  if (reference_ready_) {
    detection_time_base_stamp_ = frame.stamp;
    last_detection_stamp_ = frame.stamp;
  } else {
    detection_time_base_stamp_ = ros::Time();
    last_detection_stamp_ = ros::Time();
  }
  init_frames_.clear();
  temporal_window_frames_.clear();
  frames_since_last_window_process_ = 0;

  if (reference_ready_) {

    StructureUnitBuilder su_builder(params_.structure_unit);
    structure_units_ = su_builder.Build(anchors_);
    structure_unit_tracker_ = std::make_unique<StructureUnitTracker>(
        params_.structure_tracker, structure_units_);
    ROS_INFO("[deform_monitor_v2] Structure units ready: %zu",
             structure_units_.size());
    std::cout << "[deform_monitor_v2] Reference init done, anchors=" << anchors_.size()
              << " frames=" << init_frame_count_ << std::endl;
    if (diagnostics_log_.is_open()) {
      diagnostics_log_ << "[REFERENCE] stamp=" << frame.stamp.toSec()
                       << " anchors=" << anchors_.size()
                       << " init_frames=" << init_frame_count_ << std::endl;
      diagnostics_log_.flush();
    }
  } else {
    std::cout << "[deform_monitor_v2] Reference init failed: anchors=0"
              << " frames=" << init_frame_count_
              << ". No RViz output." << std::endl;
    if (diagnostics_log_.is_open()) {
      diagnostics_log_ << "[REFERENCE] stamp=" << frame.stamp.toSec()
                       << " anchors=0"
                       << " init_frames=" << init_frame_count_
                       << " status=failed" << std::endl;
      diagnostics_log_.flush();
    }
  }
}

bool DeformMonitorV2Node::JudgeAnchorSignificance(const AnchorReference& anchor,
                                                  AnchorTrackState* state) const {
  if (!state) {
    return false;
  }
  bool displacement_sig = false;
  if (state->gate_state == ObsGateState::OBSERVABLE_MATCHED && state->comparable &&
      state->dof_obs > 0) {
    const Eigen::Matrix3d Sigma_u = 0.5 * (state->P_mix.block<3, 3>(0, 0) +
                                           state->P_mix.block<3, 3>(0, 0).transpose());
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(Sigma_u);
    int dof_eff = 0;
    if (eig.info() == Eigen::Success) {
      for (int i = 0; i < 3; ++i) {
        if (eig.eigenvalues()(i) <
            params_.observability.tau_sigma_max * params_.observability.tau_sigma_max) {
          ++dof_eff;
        }
      }
    }
    if (dof_eff > 0) {
      const double chi2_threshold = Chi2ThresholdByDof(dof_eff, params_.significance.alpha_s);
      const bool stat_sig = state->chi2_stat > chi2_threshold;
      const bool amp_sig =
          state->type == AnchorType::PLANE
              ? state->disp_normal > params_.significance.tau_A_normal
              : (state->disp_norm > params_.significance.tau_A_norm ||
                 state->disp_normal > params_.significance.tau_A_normal ||
                 state->disp_edge > params_.significance.tau_A_edge);
      displacement_sig = stat_sig && amp_sig;
    }
  }

  if (params_.local_contrast.enable) {
    const bool enough_bg =
        state->local_bg_count >= std::max(1, params_.local_contrast.min_neighbors);
    const bool rel_amp_sig =
        state->type == AnchorType::PLANE
            ? (state->local_rel_normal > params_.local_contrast.tau_rel_normal ||
               state->local_rel_norm > params_.local_contrast.tau_rel_norm)
            : (state->local_rel_norm > params_.local_contrast.tau_rel_norm ||
               state->local_rel_normal > params_.local_contrast.tau_rel_normal ||
               state->local_rel_edge > params_.local_contrast.tau_rel_edge);
    const bool contrast_sig =
        enough_bg &&
        state->local_contrast_score > params_.local_contrast.tau_contrast_score;
    bool plane_bg_sig = false;
    if (params_.local_contrast.enable_plane_background_for_edges &&
        state->type != AnchorType::PLANE) {
      const bool enough_plane_bg =
          state->plane_bg_count >= std::max(1, params_.local_contrast.min_plane_neighbors);
      const bool plane_rel_amp_sig =
          state->plane_rel_norm > params_.local_contrast.tau_rel_norm ||
          state->plane_rel_normal > params_.local_contrast.tau_rel_normal ||
          state->plane_rel_edge > params_.local_contrast.tau_rel_edge;
      const bool plane_contrast_sig =
          enough_plane_bg &&
          state->plane_contrast_score > params_.local_contrast.tau_plane_contrast_score;
      plane_bg_sig = enough_plane_bg && plane_rel_amp_sig && plane_contrast_sig;
    }


    if (state->type != AnchorType::PLANE) {
      displacement_sig = displacement_sig && ((rel_amp_sig && contrast_sig) || plane_bg_sig);
    }
  }

  const bool graph_motion_sig =
      state->graph_candidate &&
      state->graph_neighbor_count >= std::max(1, params_.graph_temporal.min_neighbors) &&
      state->graph_coherent_score > params_.graph_temporal.tau_graph_support &&
      state->graph_temporal_score > params_.graph_temporal.tau_graph_temporal;
  state->persistent_candidate =
      state->cusum_score > params_.significance.cusum_h ||
      state->graph_persistence_score > params_.graph_temporal.cusum_h ||
      state->directional_persistent;
  if (state->type == AnchorType::PLANE) {
    state->disappearance_candidate = false;
  }



  if (state->type == AnchorType::PLANE && state->reacquired &&
      (state->matched_center_R - anchor.center_R).norm() > 0.30) {
    displacement_sig = false;
  }


  const bool is_isolated_reacquired =
      state->reacquired &&
      state->local_bg_count == 0 &&
      state->graph_neighbor_count == 0;
  if (params_.directional_motion.enable &&
      is_isolated_reacquired &&
      !state->directional_persistent) {
    displacement_sig = false;
  }
  if (displacement_sig && (state->persistent_candidate || graph_motion_sig)) {
    state->mode = DetectionMode::DISPLACEMENT;
  } else {
    state->mode = DetectionMode::NONE;
  }
  return displacement_sig && (state->persistent_candidate || graph_motion_sig);
}

void DeformMonitorV2Node::ProcessFrame(const FrameInput& frame) {
  ObservationFrameDeque single_frame_window;
  single_frame_window.push_back(frame);
  ProcessFrameWindow(single_frame_window);
}

void DeformMonitorV2Node::ProcessFrameWindow(const ObservationFrameDeque& frames) {
  if (frames.empty()) {
    return;
  }

  const FrameInput& frame = frames.back();
  const ros::Time stamp = frames.back().stamp;
  StageRuntimeRecord runtime_record;
  runtime_record.stamp = stamp;
  runtime_record.frame_index = runtime_frame_index_++;
  size_t comparable_count = 0;
  size_t significant_count = 0;

  {
    ScopedWallTimer total_timer(&runtime_record.total_ms);

    const bool use_temporal_window = params_.temporal.enable && frames.size() > 1;
    observations_.assign(anchors_.size(), CurrentObservation());
    {
      ScopedWallTimer stage_a_timer(&runtime_record.stage_a_ms);
      if (use_temporal_window) {
        obs_extractor_.PrepareTemporalWindow(frames);
      } else {
        obs_extractor_.PrepareSingleFrame(frame.cloud, frame.pose_cov, frame.lidar_origin_R);
      }
      ParallelFor(anchors_.size(), [this, &frame](size_t i) {
        observations_[i] =
            obs_extractor_.ExtractForAnchorFromPreparedCache(anchors_[i],
                                                             frame.pose_cov,
                                                             frame.lidar_origin_R);
      });
    }

    {
      ScopedWallTimer stage_d_timer(&runtime_record.stage_d_ms);
      last_background_bias_R_.setZero();
      last_background_bias_anchor_count_ = 0;
      last_background_bias_scalar_count_ = 0;
      drift_estimation_degraded_ = false;
      if (params_.background_bias.enable) {
        Eigen::Vector3d bias_R = Eigen::Vector3d::Zero();
        size_t used_anchor_count = 0;
        size_t used_scalar_count = 0;
        size_t total_comparable_plane = 0;
        if (EstimateBackgroundBias(observations_, &bias_R, &used_anchor_count,
                                   &used_scalar_count, &total_comparable_plane)) {
          const double stable_ratio = (total_comparable_plane > 0)
              ? static_cast<double>(used_anchor_count) / total_comparable_plane
              : 0.0;
          if (stable_ratio < params_.background_bias.min_stable_ratio &&
              has_previous_bias_) {
            drift_estimation_degraded_ = true;
            bias_R = smoothed_background_bias_R_;
          } else {
            if (has_previous_bias_) {
              const double alpha = params_.background_bias.ema_alpha;
              bias_R = alpha * bias_R + (1.0 - alpha) * smoothed_background_bias_R_;
            }
            smoothed_background_bias_R_ = bias_R;
            has_previous_bias_ = true;
          }

          ApplyBackgroundBias(bias_R, &observations_);
          last_background_bias_R_ = bias_R;
          last_background_bias_anchor_count_ = used_anchor_count;
          last_background_bias_scalar_count_ = used_scalar_count;
        } else if (has_previous_bias_) {
          drift_estimation_degraded_ = true;
          ApplyBackgroundBias(smoothed_background_bias_R_, &observations_);
          last_background_bias_R_ = smoothed_background_bias_R_;
        }
      }
    }

    {
      ScopedWallTimer stage_b_timer(&runtime_record.stage_b_ms);
      ParallelFor(anchor_states_.size(), [this, &stamp](size_t i) {
        double dt = (stamp - anchor_states_[i].last_update).toSec();
        if (dt <= 0.0 || !std::isfinite(dt)) {
          dt = 0.1;
        }

        anchor_states_[i].cluster_member = false;
        imm_filter_.Predict(&anchor_states_[i], dt);
        imm_filter_.Update(&anchor_states_[i], anchors_[i], observations_[i]);

        const Eigen::Vector3d u = anchor_states_[i].x_mix.block<3, 1>(0, 0);
        const Eigen::Matrix3d Su = anchor_states_[i].P_mix.block<3, 3>(0, 0);
        anchor_states_[i].disp_norm = u.norm();
        anchor_states_[i].disp_normal = std::abs(anchors_[i].normal_R.dot(u));
        anchor_states_[i].disp_edge = std::abs(anchors_[i].edge_normal_R.dot(u));
        anchor_states_[i].chi2_stat = Chi2PseudoInverse(u, Su);
        anchor_states_[i].comparable = observations_[i].comparable;
        anchor_states_[i].observable = observations_[i].observable;
        anchor_states_[i].gate_state = observations_[i].gate_state;
        anchor_states_[i].reacquired = observations_[i].reacquired;
        anchor_states_[i].matched_center_R = observations_[i].matched_center_R;
      });
    }

    {
      ScopedWallTimer stage_c_timer(&runtime_record.stage_c_ms);
      ParallelFor(anchor_states_.size(), [this, &stamp](size_t i) {
        double dt = (stamp - anchor_states_[i].last_update).toSec();
        if (dt <= 0.0 || !std::isfinite(dt)) {
          dt = 0.1;
        }
        imm_filter_.UpdateCusum(&anchor_states_[i]);
        imm_filter_.UpdateDirectionalMotion(&anchor_states_[i], anchors_[i],
                                            observations_[i].cmp_score, dt);

        if ((observations_[i].gate_state == ObsGateState::OBSERVABLE_MISSING ||
             observations_[i].gate_state == ObsGateState::OBSERVABLE_REPLACED) &&
            observations_[i].disappearance_score >= params_.significance.tau_disappear) {
          ++anchor_states_[i].disappearance_streak;
          anchor_states_[i].disappearance_score =
              std::max(observations_[i].disappearance_score,
                       0.70 * anchor_states_[i].disappearance_score +
                           0.30 * observations_[i].disappearance_score);
        } else {
          anchor_states_[i].disappearance_streak = 0;
          const double decay =
              observations_[i].gate_state == ObsGateState::NOT_OBSERVABLE ? 0.60 : 0.80;
          anchor_states_[i].disappearance_score *= decay;
          if (anchor_states_[i].disappearance_score < 0.05) {
            anchor_states_[i].disappearance_score = 0.0;
          }
        }
        anchor_states_[i].disappearance_candidate =
            anchor_states_[i].disappearance_streak >= params_.significance.disappear_frames &&
            anchor_states_[i].disappearance_score >= params_.significance.tau_disappear;
        anchor_states_[i].last_update = stamp;
      });

      UpdateLocalContrastStates();
      UpdateGraphTemporalStates(stamp);

      for (size_t i = 0; i < anchor_states_.size(); ++i) {
        anchor_states_[i].significant = JudgeAnchorSignificance(anchors_[i], &anchor_states_[i]);
        const double evidence = anchor_states_[i].mode == DetectionMode::DISAPPEARANCE
                                    ? anchor_states_[i].disappearance_score
                                    : anchor_states_[i].cusum_score;
        anchor_states_[i].evidence_history.push_back(evidence);
        while (anchor_states_[i].evidence_history.size() > 20) {
          anchor_states_[i].evidence_history.pop_front();
        }

        if (anchor_states_[i].comparable) {
          ++comparable_count;
        }
        if (anchor_states_[i].significant || anchor_states_[i].disappearance_candidate) {
          ++significant_count;
        }
      }
    }

    {
      ScopedWallTimer stage_d_timer(&runtime_record.stage_d_ms);
      clusters_ = clusterer_.Cluster(anchors_, anchor_states_);
      for (const auto& cluster : clusters_) {
        if (!cluster.significant) {
          continue;
        }
        for (const int anchor_id : cluster.anchor_ids) {
          for (auto& state : anchor_states_) {
            if (state.id == anchor_id) {
              state.cluster_member = true;
              break;
            }
          }
        }
      }

      old_regions_.clear();
      new_regions_.clear();
      structure_motions_.clear();
      if (params_.structure_correspondence.enable) {
        region_hypothesis_builder_.Build(anchors_,
                                         anchor_states_,
                                         observations_,
                                         clusters_,
                                         &old_regions_,
                                         &new_regions_);
        structure_motions_ =
            region_correspondence_solver_.Solve(old_regions_, new_regions_);
      }

      risk_evidence_.clear();
      risk_voxels_.clear();
      risk_regions_.clear();
      if (params_.risk_visualization.enable || params_.persistent_risk.enable) {
        risk_evidence_ = risk_adapter_.Build(anchors_, anchor_states_, observations_, clusters_);
        risk_voxels_ = risk_field_builder_.Build(anchors_, risk_evidence_);
        risk_regions_ = risk_field_builder_.ExtractRegions(risk_voxels_);
      }
      if (params_.persistent_risk.enable) {
        persistent_risk_tracks_ = persistent_risk_tracker_.Update(risk_regions_, stamp);
      } else {
        persistent_risk_tracks_.clear();
      }

      ref_manager_.UpdateReferenceStatistics(&anchors_, observations_, &anchor_states_);

      if (structure_unit_tracker_) {
        pcl::PointCloud<pcl::PointXYZ> cloud_xyz;
        for (const auto& f : frames) {
          if (!f.cloud) {
            continue;
          }
          pcl::PointCloud<pcl::PointXYZ> tmp;
          pcl::copyPointCloud(*f.cloud, tmp);
          cloud_xyz += tmp;
        }
        structure_unit_tracker_->Update(cloud_xyz, anchor_states_, structure_migrations_);
      }

      TryIncrementalAnchorPromotion(frames);
      PublishResults(stamp);
    }

    WriteFrameDiagnostics(stamp, comparable_count, significant_count, clusters_.size());
  }

  WriteStageRuntimeRecord(runtime_record);
  PrintFrameSummary(stamp, comparable_count, significant_count, clusters_.size(),
                    runtime_record.total_ms);
}

void DeformMonitorV2Node::TryIncrementalAnchorPromotion(
    const ObservationFrameDeque& frames) {
  if (!params_.incremental.enable || !reference_ready_ || frames.empty()) {
    return;
  }


  const auto& last_frame = frames.back();
  ReferenceInitFrame init_frame;
  init_frame.cloud.reset(new pcl::PointCloud<pcl::PointXYZI>(*last_frame.cloud));
  init_frame.lidar_origin_R = last_frame.lidar_origin_R;
  init_frame.pose_cov = last_frame.pose_cov;
  init_frame.stamp = last_frame.stamp;
  incremental_init_frames_.push_back(std::move(init_frame));
  ++incremental_frame_count_;


  if (incremental_frame_count_ < params_.incremental.warmup_frames) {
    return;
  }


  int start_id = 0;
  for (const auto& anchor : anchors_) {
    start_id = std::max(start_id, anchor.id + 1);
  }

  AnchorReferenceVector new_anchors = anchor_builder_.BuildIncrementalAnchors(
      incremental_init_frames_,
      anchors_,
      params_.incremental.coverage_radius,
      start_id,
      params_.incremental.min_visible_frames,
      params_.incremental.max_new_anchors);


  incremental_init_frames_.clear();
  incremental_frame_count_ = 0;

  if (new_anchors.empty()) {
    return;
  }


  const size_t old_size = anchors_.size();
  for (auto& anchor : new_anchors) {
    anchors_.push_back(std::move(anchor));
  }


  const ros::Time stamp = frames.back().stamp;
  for (size_t i = old_size; i < anchors_.size(); ++i) {
    AnchorTrackState state;
    state.id = anchors_[i].id;
    state.type = anchors_[i].type;
    state.last_update = stamp;
    imm_filter_.InitializeAnchorState(&state);
    anchor_states_.push_back(state);
  }


  observations_.resize(anchors_.size());


  BuildReferenceAdjacency();

  std::cout << "[deform_monitor_v2] Anchor promotion: added=" << new_anchors.size()
            << " total=" << anchors_.size() << std::endl;
  if (diagnostics_log_.is_open()) {
    diagnostics_log_ << "[INCREMENTAL] stamp=" << stamp.toSec()
                     << " new_anchors=" << new_anchors.size()
                     << " total_anchors=" << anchors_.size() << std::endl;
    diagnostics_log_.flush();
  }
}

void DeformMonitorV2Node::PublishResults(const ros::Time& stamp) {
  anchors_pub_.publish(
      viz_publisher_.BuildAnchorStatesMsg(anchors_, anchor_states_, observations_, stamp,
                                          params_.io.reference_frame));
  clusters_pub_.publish(
      viz_publisher_.BuildMotionClustersMsg(clusters_, stamp, params_.io.reference_frame));
  debug_cloud_pub_.publish(
      viz_publisher_.BuildDebugCloudMsg(anchors_, anchor_states_, stamp,
                                        params_.io.reference_frame));
  anchor_markers_pub_.publish(
      viz_publisher_.BuildAnchorMarkers(anchors_, anchor_states_, stamp, params_.io.reference_frame));
  motion_markers_pub_.publish(viz_publisher_.BuildMotionMarkers(
      anchors_, anchor_states_, clusters_, stamp, params_.io.reference_frame));
  if (params_.structure_correspondence.enable) {
    if (params_.structure_correspondence.publish_motions) {
      structure_motions_pub_.publish(structure_viz_publisher_.BuildStructureMotionsMsg(
          structure_motions_, stamp, params_.io.reference_frame));
    }
    if (params_.structure_correspondence.publish_markers) {
      structure_markers_pub_.publish(
          structure_viz_publisher_.BuildMarkers(structure_motions_, stamp,
                                                params_.io.reference_frame));
    }
  }

  viz_publisher_.PublishStructureMigrations(
      structure_units_, structure_migrations_, stamp, params_.io.reference_frame);
  if (params_.risk_visualization.enable) {
    if (params_.risk_visualization.publish_evidence) {
      risk_evidence_pub_.publish(
          risk_viz_publisher_.BuildRiskEvidenceMsg(risk_evidence_, stamp,
                                                   params_.io.reference_frame));
    }
    if (params_.risk_visualization.publish_voxels) {
      risk_voxels_pub_.publish(
          risk_viz_publisher_.BuildRiskVoxelFieldMsg(risk_voxels_, stamp,
                                                     params_.io.reference_frame));
    }
    if (params_.risk_visualization.publish_regions) {
      risk_regions_pub_.publish(
          risk_viz_publisher_.BuildRiskRegionsMsg(risk_regions_, stamp,
                                                  params_.io.reference_frame));
    }
    if (params_.risk_visualization.publish_markers) {
      risk_markers_pub_.publish(
          risk_viz_publisher_.BuildRiskMarkers(risk_voxels_, risk_regions_, stamp,
                                               params_.io.reference_frame));
    }
  }
  if (params_.persistent_risk.enable) {
    persistent_risk_regions_pub_.publish(
        risk_viz_publisher_.BuildPersistentRiskRegionsMsg(persistent_risk_tracks_, stamp,
                                                          params_.io.reference_frame));
    persistent_risk_markers_pub_.publish(
        risk_viz_publisher_.BuildPersistentRiskMarkers(persistent_risk_tracks_, stamp,
                                                       params_.io.reference_frame));
  }
}

void DeformMonitorV2Node::PublishEmptyResults(const ros::Time& stamp) {
  AnchorReferenceVector empty_anchors;
  AnchorStateVector empty_states;
  CurrentObservationVector empty_observations;
  MotionClusterVector empty_clusters;
  StructureMotionVector empty_structure_motions;
  RiskEvidenceVector empty_evidence;
  RiskVoxelVector empty_voxels;
  RiskRegionVector empty_regions;
  PersistentRiskTrackVector empty_persistent_tracks;
  anchors_pub_.publish(viz_publisher_.BuildAnchorStatesMsg(empty_anchors,
                                                           empty_states,
                                                           empty_observations,
                                                           stamp,
                                                           params_.io.reference_frame));
  clusters_pub_.publish(
      viz_publisher_.BuildMotionClustersMsg(empty_clusters, stamp, params_.io.reference_frame));
  debug_cloud_pub_.publish(
      viz_publisher_.BuildDebugCloudMsg(empty_anchors, empty_states, stamp,
                                        params_.io.reference_frame));
  anchor_markers_pub_.publish(viz_publisher_.BuildAnchorMarkers(
      empty_anchors, empty_states, stamp, params_.io.reference_frame));
  motion_markers_pub_.publish(viz_publisher_.BuildMotionMarkers(
      empty_anchors, empty_states, empty_clusters, stamp, params_.io.reference_frame));
  if (params_.structure_correspondence.enable) {
    if (params_.structure_correspondence.publish_motions) {
      structure_motions_pub_.publish(structure_viz_publisher_.BuildStructureMotionsMsg(
          empty_structure_motions, stamp, params_.io.reference_frame));
    }
    if (params_.structure_correspondence.publish_markers) {
      structure_markers_pub_.publish(
          structure_viz_publisher_.BuildMarkers(empty_structure_motions, stamp,
                                                params_.io.reference_frame));
    }
  }
  if (params_.risk_visualization.enable) {
    if (params_.risk_visualization.publish_evidence) {
      risk_evidence_pub_.publish(
          risk_viz_publisher_.BuildRiskEvidenceMsg(empty_evidence, stamp,
                                                   params_.io.reference_frame));
    }
    if (params_.risk_visualization.publish_voxels) {
      risk_voxels_pub_.publish(
          risk_viz_publisher_.BuildRiskVoxelFieldMsg(empty_voxels, stamp,
                                                     params_.io.reference_frame));
    }
    if (params_.risk_visualization.publish_regions) {
      risk_regions_pub_.publish(
          risk_viz_publisher_.BuildRiskRegionsMsg(empty_regions, stamp,
                                                  params_.io.reference_frame));
    }
    if (params_.risk_visualization.publish_markers) {
      risk_markers_pub_.publish(
          risk_viz_publisher_.BuildRiskMarkers(empty_voxels, empty_regions, stamp,
                                               params_.io.reference_frame));
    }
  }
  if (params_.persistent_risk.enable) {
    persistent_risk_regions_pub_.publish(risk_viz_publisher_.BuildPersistentRiskRegionsMsg(
        empty_persistent_tracks, stamp, params_.io.reference_frame));
    persistent_risk_markers_pub_.publish(risk_viz_publisher_.BuildPersistentRiskMarkers(
        empty_persistent_tracks, stamp, params_.io.reference_frame));
  }
}

void DeformMonitorV2Node::PrintStartupSummary() const {
  std::cout << "[deform_monitor_v2] Node start" << std::endl;
  std::cout << "  Cloud: " << params_.io.cloud_topic << std::endl;
  std::cout << "  Covariance: " << params_.io.covariance_topic << std::endl;
  std::cout << "  Reset topic: " << params_.io.reset_reference_topic << std::endl;
  std::cout << "  Ref frame: " << params_.io.reference_frame << std::endl;
  std::cout << "  Init frames: " << params_.reference.init_frames << std::endl;
  std::cout << "  Cloud in ref: " << (params_.io.cloud_already_in_reference_frame ? "yes" : "no")
            << std::endl;
  std::cout << "    Temporal fusion: "
            << (params_.temporal.enable ? "on" : "off")
            << "  window=" << params_.temporal.window_frames
            << " step=" << params_.temporal.step_frames
            << " frames" << std::endl;
  std::cout << "  Ablation variant: " << params_.ablation.variant << std::endl;
  std::cout << "    Stage-B IMM: "
            << (params_.imm.enable_model_competition ? "on" : "off")
            << " type constraint: " << (params_.imm.enable_type_constraint ? "on" : "off")
            << " alpha_xi=" << params_.covariance.alpha_xi << std::endl;
  std::cout << "  Stage-C CUSUM: "
            << (params_.significance.enable_cusum ? "on" : "off")
            << " directional accum: "
            << (params_.directional_motion.enable ? "on" : "off") << std::endl;
  std::cout << "  Bias removal: "
            << (params_.background_bias.enable ? "on" : "off") << std::endl;
  std::cout << "  Local contrast: "
            << (params_.local_contrast.enable ? "on" : "off") << std::endl;
  std::cout << "  Graph-temporal: "
            << (params_.graph_temporal.enable ? "on" : "off") << std::endl;
  std::cout << "  Region mapping: "
            << (params_.structure_correspondence.enable ? "on" : "off") << std::endl;
  std::cout << "  Risk viz: "
            << (params_.risk_visualization.enable ? "on" : "off") << std::endl;
  std::cout << "  RViz topics: /deform/debug_cloud, /deform/anchor_markers, /deform/motion_markers"
            << std::endl;
  if (params_.risk_visualization.enable) {
    std::cout << "  Risk viz topics: /deform/risk_voxels, /deform/risk_regions, /deform/risk_markers"
              << std::endl;
  }
}

void DeformMonitorV2Node::PrintFrameSummary(const ros::Time& stamp,
                                            size_t comparable_count,
                                            size_t significant_count,
                                            size_t cluster_count,
                                            double total_ms) {
  double detection_interval_sec = 0.0;
  if (!last_detection_stamp_.isZero()) {
    detection_interval_sec = std::max(0.0, (stamp - last_detection_stamp_).toSec());
  } else if (!detection_time_base_stamp_.isZero()) {
    detection_interval_sec =
        std::max(0.0, (stamp - detection_time_base_stamp_).toSec());
  }
  last_detection_stamp_ = stamp;

  const size_t graph_count =
      static_cast<size_t>(std::count_if(anchor_states_.begin(),
                                        anchor_states_.end(),
                                        [](const AnchorTrackState& state) {
                                          return state.graph_candidate;
                                        }));
  std::cout << std::fixed << std::setprecision(3)
            << "[deform_monitor_v2] dt=" << detection_interval_sec << "s"
            << " anchors=" << anchors_.size()
            << " comparable=" << comparable_count
            << " significant=" << significant_count
            << " graph=" << graph_count
            << " clusters=" << cluster_count
            << " bg_bias=(" << last_background_bias_R_.x()
            << "," << last_background_bias_R_.y()
            << "," << last_background_bias_R_.z()
            << ")"
            << " bg_used=" << last_background_bias_anchor_count_
            << "/" << last_background_bias_scalar_count_
            << std::defaultfloat << std::endl;

  // Print total processing time once per second (~10 frames at 10 Hz)
  static int print_frame_counter = 0;
  if (++print_frame_counter >= 10) {
    print_frame_counter = 0;
    ROS_INFO("[deform_monitor_v2] total_ms=%.2f", total_ms);
  }
}

void DeformMonitorV2Node::WriteFrameDiagnostics(const ros::Time& stamp,
                                                size_t comparable_count,
                                                size_t significant_count,
                                                size_t cluster_count) {
  if (!diagnostics_log_.is_open()) {
    return;
  }

  double detection_interval_sec = 0.0;
  if (!last_detection_stamp_.isZero()) {
    detection_interval_sec = std::max(0.0, (stamp - last_detection_stamp_).toSec());
  } else if (!detection_time_base_stamp_.isZero()) {
    detection_interval_sec = std::max(0.0, (stamp - detection_time_base_stamp_).toSec());
  }

  const size_t graph_count =
      static_cast<size_t>(std::count_if(anchor_states_.begin(),
                                        anchor_states_.end(),
                                        [](const AnchorTrackState& state) {
                                          return state.graph_candidate;
                                        }));

  std::vector<size_t> isolated_indices;
  isolated_indices.reserve(anchor_states_.size());
  for (size_t i = 0; i < anchor_states_.size(); ++i) {
    const AnchorTrackState& state = anchor_states_[i];
    const bool alert_anchor =
        state.significant || state.mode == DetectionMode::DISAPPEARANCE;
    if (alert_anchor && !state.cluster_member) {
      isolated_indices.push_back(i);
    }
  }

  diagnostics_log_ << std::fixed << std::setprecision(3)
                   << "[FRAME] stamp=" << stamp.toSec()
                   << " dt=" << detection_interval_sec
                   << " anchors=" << anchors_.size()
                   << " comparable=" << comparable_count
                   << " significant=" << significant_count
                   << " graph=" << graph_count
                   << " clusters=" << cluster_count
                   << " isolated_alerts=" << isolated_indices.size()
                   << " bg_bias=(" << last_background_bias_R_.x()
                   << "," << last_background_bias_R_.y()
                   << "," << last_background_bias_R_.z()
                   << ")"
                   << " bg_used=" << last_background_bias_anchor_count_
                   << "/" << last_background_bias_scalar_count_
                   << std::endl;


  ++diag_frame_counter_;
  if (diag_frame_counter_ % 10 == 0) {
    struct DirEntry { size_t idx; double s_norm; };
    std::vector<DirEntry> entries;
    entries.reserve(anchor_states_.size());
    for (size_t i = 0; i < anchor_states_.size(); ++i) {
      if (anchor_states_[i].comparable) {
        entries.push_back({i, anchor_states_[i].directional_S.norm()});
      }
    }
    std::sort(entries.begin(), entries.end(),
              [](const DirEntry& a, const DirEntry& b) { return a.s_norm > b.s_norm; });
    const size_t top_n = std::min<size_t>(3, entries.size());
    for (size_t k = 0; k < top_n; ++k) {
      const size_t i = entries[k].idx;
      const AnchorTrackState& s = anchor_states_[i];
      diagnostics_log_ << "  [dir_diag]"
                       << " id=" << s.id
                       << " dir_S_norm=" << s.directional_S.norm()
                       << " dir_quality=" << s.directional_quality_sum
                       << " dir_ratio=" << (s.directional_quality_sum > 1e-9
                            ? s.directional_S.norm() / s.directional_quality_sum : 0.0)
                       << " dir_persistent=" << (s.directional_persistent ? 1 : 0)
                       << " disp_norm=" << s.disp_norm
                       << " cusum=" << s.cusum_score
                       << std::endl;
    }
  }

  std::vector<int> inactive_ids;
  inactive_ids.reserve(logged_anchor_alerts_.size());
  for (const auto& entry : logged_anchor_alerts_) {
    inactive_ids.push_back(entry.first);
  }

  for (const size_t idx : isolated_indices) {
    const AnchorReference& anchor = anchors_[idx];
    const AnchorTrackState& state = anchor_states_[idx];
    const CurrentObservation& obs = observations_[idx];
    auto inactive_it = std::find(inactive_ids.begin(), inactive_ids.end(), state.id);
    if (inactive_it != inactive_ids.end()) {
      inactive_ids.erase(inactive_it);
    }

    LoggedAnchorAlertState& logged_state = logged_anchor_alerts_[state.id];
    const bool state_changed =
        logged_state.mode != state.mode ||
        logged_state.gate_state != state.gate_state ||
        logged_state.significant != state.significant ||
        logged_state.cluster_member != state.cluster_member ||
        logged_state.reacquired != state.reacquired;
    const bool refresh_due =
        logged_state.last_logged_stamp.isZero() ||
        (stamp - logged_state.last_logged_stamp).toSec() >= kDiagnosticsAlertRefreshSec;
    if (!state_changed && !refresh_due) {
      continue;
    }

    const Eigen::Vector3d disp = state.x_mix.block<3, 1>(0, 0);
    diagnostics_log_ << "  [isolated]"
                     << " id=" << state.id
                     << " type=" << AnchorTypeToString(anchor.type)
                     << " mode=" << DetectionModeToString(state.mode)
                     << " center=" << FormatVec3(anchor.center_R)
                     << " disp=" << FormatVec3(disp)
                     << " matched_delta=" << FormatVec3(obs.matched_delta_R)
                     << " disp_norm=" << state.disp_norm
                     << " disp_normal=" << state.disp_normal
                     << " disp_edge=" << state.disp_edge
                     << " chi2=" << state.chi2_stat
                     << " cusum=" << state.cusum_score
                     << " cmp=" << obs.cmp_score
                     << " overlap=" << obs.overlap_score
                     << " fit_rmse=" << obs.fit_rmse
                     << " support=" << obs.support_count
                     << " dof=" << state.dof_obs
                     << " status=" << ObsStatusToString(obs.status)
                     << " gate=" << ObsGateStateToString(state.gate_state)
                     << " comparable=" << (state.comparable ? 1 : 0)
                     << " observable=" << (state.observable ? 1 : 0)
                     << " significant=" << (state.significant ? 1 : 0)
                     << " persistent=" << (state.persistent_candidate ? 1 : 0)
                     << " disappearance_candidate=" << (state.disappearance_candidate ? 1 : 0)
                     << " disappearance_score=" << state.disappearance_score
                     << " reacquired=" << (state.reacquired ? 1 : 0)
                     << " local_bg=" << state.local_bg_count
                     << " local_contrast=" << state.local_contrast_score
                     << " local_rel_norm=" << state.local_rel_norm
                     << " local_rel_normal=" << state.local_rel_normal
                     << " local_rel_edge=" << state.local_rel_edge
                     << " plane_bg=" << state.plane_bg_count
                     << " plane_contrast=" << state.plane_contrast_score
                     << " graph_neighbors=" << state.graph_neighbor_count
                     << " graph_candidate=" << (state.graph_candidate ? 1 : 0)
                     << " graph_support=" << state.graph_coherent_score
                     << " graph_temporal=" << state.graph_temporal_score
                     << " graph_persistence=" << state.graph_persistence_score
                     << " dir_S_norm=" << state.directional_S.norm()
                     << " dir_quality=" << state.directional_quality_sum
                     << " dir_persistent=" << (state.directional_persistent ? 1 : 0)
                     << std::endl;

    logged_state.mode = state.mode;
    logged_state.gate_state = state.gate_state;
    logged_state.significant = state.significant;
    logged_state.cluster_member = state.cluster_member;
    logged_state.reacquired = state.reacquired;
    logged_state.last_logged_stamp = stamp;
  }

  for (const int anchor_id : inactive_ids) {
    logged_anchor_alerts_.erase(anchor_id);
  }
  diagnostics_log_ << std::endl;
  diagnostics_log_.flush();
}

void DeformMonitorV2Node::MaybePrintPipelineStatus() {
  const ros::WallTime now = ros::WallTime::now();
  if (!last_status_wall_time_.isZero() && (now - last_status_wall_time_).toSec() < 2.0) {
    return;
  }
  last_status_wall_time_ = now;

  std::unique_lock<std::mutex> processing_lock(processing_mutex_, std::try_to_lock);
  if (!processing_lock.owns_lock()) {
    return;
  }
  std::lock_guard<std::mutex> data_lock(data_mutex_);
  std::cout << "[deform_monitor_v2] Status"
            << " cloud=" << cloud_msg_count_
            << " cov=" << covariance_msg_count_
            << " synced=" << synchronized_frame_count_
            << " init=" << init_frame_count_ << "/" << params_.reference.init_frames
            << " anchors=" << anchors_.size()
            << " cloud_queue=" << cloud_queue_.size()
            << " cov_queue=" << covariance_queue_.size()
            << " window=" << temporal_window_frames_.size()
            << " reference=" << (reference_ready_ ? "ready" : "not_ready");

  if (cloud_msg_count_ == 0) {
    std::cout << " | no cloud topic " << params_.io.cloud_topic;
  } else if (covariance_msg_count_ == 0) {
    std::cout << " | no covariance topic " << params_.io.covariance_topic;
  } else if (synchronized_frame_count_ == 0) {
    std::cout << " | waiting for cloud/cov sync";
  } else if (!reference_ready_ && init_frame_count_ < params_.reference.init_frames) {
    std::cout << " | waiting for reference init";
  } else if (!reference_ready_ && init_frame_count_ >= params_.reference.init_frames) {
    std::cout << " | reference init failed, anchors=0";
  }

  std::cout << std::endl;
}

}  // namespace deform_monitor_v2
