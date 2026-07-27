#ifndef DEFORM_MONITOR_V2_CORE_OBJECT_OBSERVATION_STATS_HPP
#define DEFORM_MONITOR_V2_CORE_OBJECT_OBSERVATION_STATS_HPP

#include <cstdint>
#include <map>

#include "deform_monitor_v2/data_types.hpp"

namespace deform_monitor_v2 {

enum class ObjectObservationPhase : uint8_t {
  REFERENCE = 0,
  MONITORING = 1
};

struct ObjectHitStatState {
  uint16_t object_id = 0;
  uint64_t point_count = 0;
  uint32_t visible_frame_count = 0;
};

struct ObjectObservationStatsState {
  uint32_t reference_epoch = 0;
  ObjectObservationPhase phase = ObjectObservationPhase::MONITORING;
  ros::Time window_start;
  ros::Time window_end;
  uint32_t frame_count = 0;
  uint64_t total_point_count = 0;
  uint64_t valid_label_point_count = 0;
  uint64_t invalid_label_point_count = 0;
  std::vector<ObjectHitStatState> objects;
};

class ObjectObservationStatsAccumulator {
public:
  void SetParams(const ObjectAssociationParams& params);
  void Reset();
  void AddFrame(const pcl::PointCloud<pcl::PointXYZI>::ConstPtr& cloud,
                const ros::Time& stamp);
  ObjectObservationStatsState BuildSummary(
      uint32_t reference_epoch,
      ObjectObservationPhase phase) const;

private:
  struct MutableObjectStat {
    uint64_t point_count = 0;
    uint32_t visible_frame_count = 0;
  };

  ObjectAssociationParams params_;
  ros::Time window_start_;
  ros::Time window_end_;
  uint32_t frame_count_ = 0;
  uint64_t total_point_count_ = 0;
  uint64_t valid_label_point_count_ = 0;
  uint64_t invalid_label_point_count_ = 0;
  std::map<uint16_t, MutableObjectStat> object_stats_;
};

}  // namespace deform_monitor_v2

#endif  // DEFORM_MONITOR_V2_CORE_OBJECT_OBSERVATION_STATS_HPP
