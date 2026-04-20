/*
 * Author: zgliu@cumt.edu.cn
 * Affiliation: China University of Mining and Technology
 * Open-source release date: 2026-04-20
 */

#include "deform_monitor_v2/structure_visualization_publisher.hpp"

#include <std_msgs/ColorRGBA.h>

namespace deform_monitor_v2 {

namespace {

geometry_msgs::Point ToPoint(const Eigen::Vector3d& p) {
  geometry_msgs::Point out;
  out.x = p.x();
  out.y = p.y();
  out.z = p.z();
  return out;
}

geometry_msgs::Vector3 ToVector3(const Eigen::Vector3d& p) {
  geometry_msgs::Vector3 out;
  out.x = p.x();
  out.y = p.y();
  out.z = p.z();
  return out;
}

void AppendBoxOutlinePoints(const Eigen::Vector3d& bmin,
                            const Eigen::Vector3d& bmax,
                            std::vector<geometry_msgs::Point>* points) {
  if (!points) {
    return;
  }
  const Eigen::Vector3d c000(bmin.x(), bmin.y(), bmin.z());
  const Eigen::Vector3d c001(bmin.x(), bmin.y(), bmax.z());
  const Eigen::Vector3d c010(bmin.x(), bmax.y(), bmin.z());
  const Eigen::Vector3d c011(bmin.x(), bmax.y(), bmax.z());
  const Eigen::Vector3d c100(bmax.x(), bmin.y(), bmin.z());
  const Eigen::Vector3d c101(bmax.x(), bmin.y(), bmax.z());
  const Eigen::Vector3d c110(bmax.x(), bmax.y(), bmin.z());
  const Eigen::Vector3d c111(bmax.x(), bmax.y(), bmax.z());
  const Eigen::Vector3d edges[][2] = {
      {c000, c001}, {c000, c010}, {c000, c100}, {c001, c011}, {c001, c101}, {c010, c011},
      {c010, c110}, {c100, c101}, {c100, c110}, {c011, c111}, {c101, c111}, {c110, c111}};
  for (const auto& edge : edges) {
    points->push_back(ToPoint(edge[0]));
    points->push_back(ToPoint(edge[1]));
  }
}

}  // namespace

void StructureVisualizationPublisher::SetParams(const StructureCorrespondenceParams& params) {
  params_ = params;
}

deform_monitor_v2::StructureMotions StructureVisualizationPublisher::BuildStructureMotionsMsg(
    const StructureMotionVector& motions,
    const ros::Time& stamp,
    const std::string& frame_id) const {
  deform_monitor_v2::StructureMotions msg;
  msg.header.stamp = stamp;
  msg.header.frame_id = frame_id;
  msg.motions.reserve(motions.size());
  for (const auto& motion : motions) {
    deform_monitor_v2::StructureMotion out;
    out.id = motion.id;
    out.old_region_id = motion.old_region_id;
    out.new_region_id = motion.new_region_id;
    out.motion_type = static_cast<uint8_t>(motion.type);
    out.old_center = ToPoint(motion.old_center_R);
    out.new_center = ToPoint(motion.new_center_R);
    out.bbox_old_min = ToPoint(motion.bbox_old_min_R);
    out.bbox_old_max = ToPoint(motion.bbox_old_max_R);
    out.bbox_new_min = ToPoint(motion.bbox_new_min_R);
    out.bbox_new_max = ToPoint(motion.bbox_new_max_R);
    out.motion = ToVector3(motion.motion_R);
    out.distance = motion.distance;
    out.match_cost = motion.match_cost;
    out.confidence = motion.confidence;
    out.support_old = motion.support_old;
    out.support_new = motion.support_new;
    out.significant = motion.significant;
    msg.motions.push_back(out);
  }
  return msg;
}

visualization_msgs::MarkerArray StructureVisualizationPublisher::BuildMarkers(
    const StructureMotionVector& motions,
    const ros::Time& stamp,
    const std::string& frame_id) const {
  visualization_msgs::MarkerArray array;
  visualization_msgs::Marker clear;
  clear.header.stamp = stamp;
  clear.header.frame_id = frame_id;
  clear.action = visualization_msgs::Marker::DELETEALL;
  array.markers.push_back(clear);

  int id = 0;
  visualization_msgs::Marker old_boxes;
  old_boxes.header.stamp = stamp;
  old_boxes.header.frame_id = frame_id;
  old_boxes.ns = "structure_old_regions";
  old_boxes.id = id++;
  old_boxes.type = visualization_msgs::Marker::LINE_LIST;
  old_boxes.action = visualization_msgs::Marker::ADD;
  old_boxes.pose.orientation.w = 1.0;
  old_boxes.scale.x = std::max(1.0e-3, params_.marker_outline_width);
  old_boxes.color.r = 0.95f;
  old_boxes.color.g = 0.10f;
  old_boxes.color.b = 0.75f;
  old_boxes.color.a = static_cast<float>(params_.marker_old_alpha);

  visualization_msgs::Marker new_boxes = old_boxes;
  new_boxes.ns = "structure_new_regions";
  new_boxes.id = id++;
  new_boxes.color.r = 0.10f;
  new_boxes.color.g = 0.95f;
  new_boxes.color.b = 0.95f;
  new_boxes.color.a = static_cast<float>(params_.marker_new_alpha);

  for (const auto& motion : motions) {
    if (!motion.significant) {
      continue;
    }
    AppendBoxOutlinePoints(motion.bbox_old_min_R, motion.bbox_old_max_R, &old_boxes.points);
    AppendBoxOutlinePoints(motion.bbox_new_min_R, motion.bbox_new_max_R, &new_boxes.points);
  }
  array.markers.push_back(old_boxes);
  array.markers.push_back(new_boxes);

  int arrow_idx = 0;
  for (const auto& motion : motions) {
    if (!motion.significant) {
      continue;
    }
    visualization_msgs::Marker arrow;
    arrow.header.stamp = stamp;
    arrow.header.frame_id = frame_id;
    arrow.ns = "structure_motion_arrows";
    arrow.id = id + arrow_idx++;
    arrow.type = visualization_msgs::Marker::ARROW;
    arrow.action = visualization_msgs::Marker::ADD;
    arrow.pose.orientation.w = 1.0;
    arrow.scale.x = std::max(1.0e-3, params_.marker_arrow_scale);
    arrow.scale.y = std::max(1.0e-3, 1.5 * params_.marker_arrow_scale);
    arrow.scale.z = std::max(1.0e-3, 2.0 * params_.marker_arrow_scale);
    arrow.color.r = 1.0f;
    arrow.color.g = 0.75f;
    arrow.color.b = 0.05f;
    arrow.color.a = static_cast<float>(params_.marker_arrow_alpha);
    arrow.points.push_back(ToPoint(motion.old_center_R));
    arrow.points.push_back(ToPoint(motion.new_center_R));
    array.markers.push_back(arrow);
  }
  return array;
}

}  // namespace deform_monitor_v2
