#include "deform_monitor_v2/core/object_observation_stats.hpp"

#include "deform_monitor_v2/core/object_id_association.hpp"

#include <set>

namespace deform_monitor_v2 {

void ObjectObservationStatsAccumulator::SetParams(
    const ObjectAssociationParams& params) {
  params_ = params;
  Reset();
}

void ObjectObservationStatsAccumulator::Reset() {
  window_start_ = ros::Time();
  window_end_ = ros::Time();
  frame_count_ = 0;
  total_point_count_ = 0;
  valid_label_point_count_ = 0;
  invalid_label_point_count_ = 0;
  object_stats_.clear();
}

void ObjectObservationStatsAccumulator::AddFrame(
    const pcl::PointCloud<pcl::PointXYZI>::ConstPtr& cloud,
    const ros::Time& stamp) {
  if (frame_count_ == 0) {
    window_start_ = stamp;
  }
  window_end_ = stamp;
  ++frame_count_;

  std::set<uint16_t> visible_objects;
  if (!cloud) {
    return;
  }
  total_point_count_ += cloud->size();
  for (const auto& point : cloud->points) {
    uint16_t object_id = params_.invalid_id;
    if (!DecodeObjectIdSample(point.intensity, params_, &object_id)) {
      ++invalid_label_point_count_;
      continue;
    }
    ++valid_label_point_count_;
    ++object_stats_[object_id].point_count;
    visible_objects.insert(object_id);
  }
  for (const uint16_t object_id : visible_objects) {
    ++object_stats_[object_id].visible_frame_count;
  }
}

ObjectObservationStatsState ObjectObservationStatsAccumulator::BuildSummary(
    uint32_t reference_epoch,
    ObjectObservationPhase phase) const {
  ObjectObservationStatsState summary;
  summary.reference_epoch = reference_epoch;
  summary.phase = phase;
  summary.window_start = window_start_;
  summary.window_end = window_end_;
  summary.frame_count = frame_count_;
  summary.total_point_count = total_point_count_;
  summary.valid_label_point_count = valid_label_point_count_;
  summary.invalid_label_point_count = invalid_label_point_count_;
  summary.objects.reserve(object_stats_.size());
  for (const auto& item : object_stats_) {
    ObjectHitStatState object;
    object.object_id = item.first;
    object.point_count = item.second.point_count;
    object.visible_frame_count = item.second.visible_frame_count;
    summary.objects.push_back(object);
  }
  return summary;
}

}  // namespace deform_monitor_v2
