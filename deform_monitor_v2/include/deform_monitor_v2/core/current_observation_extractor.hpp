/*
 * Author: zgliu@cumt.edu.cn
 * Affiliation: China University of Mining and Technology
 * Open-source release date: 2026-04-20
 */

#ifndef DEFORM_MONITOR_V2_CORE_CURRENT_OBSERVATION_EXTRACTOR_HPP
#define DEFORM_MONITOR_V2_CORE_CURRENT_OBSERVATION_EXTRACTOR_HPP

#include "deform_monitor_v2/core/scalar_measurement_builder.hpp"
#include "deform_monitor_v2/data_types.hpp"

namespace deform_monitor_v2 {

class CurrentObservationExtractor {
public:
  void SetParams(const ObservationParams& params);
  void SetTemporalParams(const TemporalFusionParams& params);
  void SetNoiseParams(const NoiseParams& params);
  void SetCovarianceParams(const CovarianceParams& params);
  void SetObservabilityParams(const ObservabilityParams& params);
  void SetMeasurementBuilder(const ScalarMeasurementBuilder& builder);

  void PrepareSingleFrame(const pcl::PointCloud<pcl::PointXYZI>::ConstPtr& curr_cloud,
                          const PoseCov6D& curr_pose_cov,
                          const Eigen::Vector3d& lidar_origin_R);
  void PrepareTemporalWindow(const ObservationFrameDeque& frames);
  CurrentObservation ExtractForAnchorFromPreparedCache(
      const AnchorReference& anchor,
      const PoseCov6D& pose_cov,
      const Eigen::Vector3d& lidar_origin_R) const;

private:
  struct VoxelKey {
    int x = 0;
    int y = 0;
    int z = 0;

    bool operator==(const VoxelKey& other) const {
      return x == other.x && y == other.y && z == other.z;
    }
  };

  struct VoxelKeyHash {
    size_t operator()(const VoxelKey& key) const {
      size_t h = std::hash<int>()(key.x);
      h ^= std::hash<int>()(key.y) + 0x9e3779b9 + (h << 6) + (h >> 2);
      h ^= std::hash<int>()(key.z) + 0x9e3779b9 + (h << 6) + (h >> 2);
      return h;
    }
  };

  struct NearbySupportPool {
    AlignedVector<Eigen::Vector3d> candidate_centers_R;
    AlignedVector<Eigen::Vector3d> points_R;
    AlignedVector<Eigen::Matrix3d> point_covariances;
    std::vector<int> frame_indices;
    std::vector<PoseCov6D> frame_pose_covs;
    std::vector<Eigen::Vector3d> frame_origins_R;
    double window_span_sec = 0.0;
  };

  struct FrameVoxelCell {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    AlignedVector<Eigen::Vector3d> points_R;
    AlignedVector<Eigen::Matrix3d> point_covariances;
    Eigen::Vector3d sum_R = Eigen::Vector3d::Zero();
    int total_points = 0;
  };

  struct FrameVoxelMap {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    AlignedUnorderedMap<VoxelKey, FrameVoxelCell, VoxelKeyHash> voxels;
    PoseCov6D pose_cov;
    Eigen::Vector3d lidar_origin_R = Eigen::Vector3d::Zero();
    double voxel_size = 0.05;
    ros::Time stamp;
    bool valid = false;
  };

  LocalSupportData BuildSupportAtCenter(
      const AnchorReference& anchor,
      const NearbySupportPool& pool,
      const Eigen::Vector3d& candidate_center_R) const;

  void EvaluateComparability(const AnchorReference& anchor, LocalSupportData* support) const;

  LocalSupportData BuildSupportForAnchor(
      const AnchorReference& anchor,
      const pcl::PointCloud<pcl::PointXYZI>::ConstPtr& curr_cloud,
      const PoseCov6D& curr_pose_cov,
      const Eigen::Vector3d& lidar_origin_R);

  LocalSupportData BuildSupportForAnchorTemporal(
      const AnchorReference& anchor,
      const ObservationFrameDeque& frames);
  LocalSupportData BuildSupportForAnchorFromCachedMaps(
      const AnchorReference& anchor,
      const Eigen::Vector3d& lidar_origin_R) const;

  FrameVoxelMap BuildFrameVoxelMap(const ObservationFrame& frame) const;
  void EnsureWindowVoxelMaps(const ObservationFrameDeque& frames);
  void EnsureSingleFrameVoxelMap(const pcl::PointCloud<pcl::PointXYZI>::ConstPtr& curr_cloud,
                                 const PoseCov6D& curr_pose_cov,
                                 const Eigen::Vector3d& lidar_origin_R);
  NearbySupportPool BuildPoolFromCachedVoxelMaps(const AnchorReference& anchor) const;

  ObservationParams params_;
  TemporalFusionParams temporal_params_;
  NoiseParams noise_params_;
  CovarianceParams covariance_params_;
  ObservabilityParams observability_params_;
  ScalarMeasurementBuilder measurement_builder_;
  AlignedDeque<FrameVoxelMap> cached_frame_voxel_maps_;
};

}  // namespace deform_monitor_v2

#endif  // DEFORM_MONITOR_V2_CORE_CURRENT_OBSERVATION_EXTRACTOR_HPP
