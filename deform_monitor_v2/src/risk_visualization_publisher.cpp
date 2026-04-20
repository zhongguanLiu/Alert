/*
 * Author: zgliu@cumt.edu.cn
 * Affiliation: China University of Mining and Technology
 * Open-source release date: 2026-04-20
 */

#include "deform_monitor_v2/risk_visualization_publisher.hpp"

#include <algorithm>
#include <visualization_msgs/Marker.h>

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

std_msgs::ColorRGBA RiskColor(double risk, double alpha_low, double alpha_high) {
  std_msgs::ColorRGBA c;
  const double t = std::max(0.0, std::min(1.0, risk));
  c.a = alpha_low + (alpha_high - alpha_low) * t;
  if (t < 0.5) {
    const double s = t / 0.5;
    c.r = static_cast<float>(s);
    c.g = static_cast<float>(0.7 + 0.3 * s);
    c.b = static_cast<float>(1.0 - 0.8 * s);
  } else {
    const double s = (t - 0.5) / 0.5;
    c.r = 1.0f;
    c.g = static_cast<float>(1.0 - 0.8 * s);
    c.b = static_cast<float>(0.2 * (1.0 - s));
  }
  return c;
}

std_msgs::ColorRGBA PersistentTrackColor(const PersistentRiskTrackState& track,
                                        double alpha) {
  std_msgs::ColorRGBA c;
  const double t = std::max(0.0, std::min(1.0, track.ema_mean_risk));
  if (t < 0.5) {
    const double s = t / 0.5;
    c.r = static_cast<float>(0.15 + 0.55 * s);
    c.g = static_cast<float>(0.35 + 0.50 * s);
    c.b = static_cast<float>(0.95 - 0.55 * s);
  } else {
    const double s = (t - 0.5) / 0.5;
    c.r = 0.70f;
    c.g = static_cast<float>(0.85 - 0.45 * s);
    c.b = static_cast<float>(0.35 - 0.20 * s);
  }
  c.a = static_cast<float>(std::max(0.0, std::min(1.0, alpha)));
  return c;
}

double PersistentTrackPersistence(const PersistentRiskTrackState& track) {
  const double age = std::max(1, track.age_frames);
  const double recent = static_cast<double>(track.matched_region_count_window) / age;
  const double streak = std::min(1.0, static_cast<double>(track.hit_streak) /
                                         std::max(1.0, static_cast<double>(track.hit_streak + track.miss_streak)));
  return std::max(recent, streak);
}

double PersistentTrackAlpha(const PersistentRiskTrackState& track) {
  const double persistence = PersistentTrackPersistence(track);
  switch (track.state) {
    case PersistentRiskState::CONFIRMED:
      return 0.55 + 0.30 * persistence;
    case PersistentRiskState::FADING:
      return 0.22 + 0.22 * persistence;
    case PersistentRiskState::CANDIDATE:
    default:
      return 0.12 + 0.14 * persistence;
  }
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
      {c000, c001}, {c000, c010}, {c000, c100}, {c001, c011},
      {c001, c101}, {c010, c011}, {c010, c110}, {c100, c101},
      {c100, c110}, {c011, c111}, {c101, c111}, {c110, c111}};
  for (const auto& edge : edges) {
    points->push_back(ToPoint(edge[0]));
    points->push_back(ToPoint(edge[1]));
  }
}

void AppendBoxShellTriangles(const Eigen::Vector3d& bmin,
                             const Eigen::Vector3d& bmax,
                             std::vector<geometry_msgs::Point>* points) {
  if (!points) {
    return;
  }
  const geometry_msgs::Point c000 = ToPoint(Eigen::Vector3d(bmin.x(), bmin.y(), bmin.z()));
  const geometry_msgs::Point c001 = ToPoint(Eigen::Vector3d(bmin.x(), bmin.y(), bmax.z()));
  const geometry_msgs::Point c010 = ToPoint(Eigen::Vector3d(bmin.x(), bmax.y(), bmin.z()));
  const geometry_msgs::Point c011 = ToPoint(Eigen::Vector3d(bmin.x(), bmax.y(), bmax.z()));
  const geometry_msgs::Point c100 = ToPoint(Eigen::Vector3d(bmax.x(), bmin.y(), bmin.z()));
  const geometry_msgs::Point c101 = ToPoint(Eigen::Vector3d(bmax.x(), bmin.y(), bmax.z()));
  const geometry_msgs::Point c110 = ToPoint(Eigen::Vector3d(bmax.x(), bmax.y(), bmin.z()));
  const geometry_msgs::Point c111 = ToPoint(Eigen::Vector3d(bmax.x(), bmax.y(), bmax.z()));

  const auto push_triangle = [&](const geometry_msgs::Point& a,
                                 const geometry_msgs::Point& b,
                                 const geometry_msgs::Point& c) {
    points->push_back(a);
    points->push_back(b);
    points->push_back(c);
  };

  push_triangle(c000, c001, c011);
  push_triangle(c000, c011, c010);
  push_triangle(c100, c110, c111);
  push_triangle(c100, c111, c101);
  push_triangle(c000, c100, c101);
  push_triangle(c000, c101, c001);
  push_triangle(c010, c011, c111);
  push_triangle(c010, c111, c110);
  push_triangle(c000, c010, c110);
  push_triangle(c000, c110, c100);
  push_triangle(c001, c101, c111);
  push_triangle(c001, c111, c011);
}

uint8_t ToPersistentRiskStateCode(PersistentRiskState state) {
  switch (state) {
    case PersistentRiskState::CANDIDATE:
      return deform_monitor_v2::PersistentRiskRegion::STATE_CANDIDATE;
    case PersistentRiskState::CONFIRMED:
      return deform_monitor_v2::PersistentRiskRegion::STATE_CONFIRMED;
    case PersistentRiskState::FADING:
      return deform_monitor_v2::PersistentRiskRegion::STATE_FADING;
  }
  return deform_monitor_v2::PersistentRiskRegion::STATE_CANDIDATE;
}

uint8_t ToPersistentRiskRegionTypeCode(RiskRegionType type) {
  switch (type) {
    case RiskRegionType::NONE:
      return deform_monitor_v2::PersistentRiskRegion::REGION_NONE;
    case RiskRegionType::DISPLACEMENT_LIKE:
      return deform_monitor_v2::PersistentRiskRegion::REGION_DISPLACEMENT_LIKE;
    case RiskRegionType::DISAPPEARANCE_LIKE:
      return deform_monitor_v2::PersistentRiskRegion::REGION_DISAPPEARANCE_LIKE;
    case RiskRegionType::MIXED:
      return deform_monitor_v2::PersistentRiskRegion::REGION_MIXED;
  }
  return deform_monitor_v2::PersistentRiskRegion::REGION_NONE;
}

}  // namespace

void RiskVisualizationPublisher::SetParams(const RiskVisualizationParams& params) {
  params_ = params;
}

deform_monitor_v2::RiskEvidenceArray RiskVisualizationPublisher::BuildRiskEvidenceMsg(
    const RiskEvidenceVector& evidences,
    const ros::Time& stamp,
    const std::string& frame_id) const {
  deform_monitor_v2::RiskEvidenceArray msg;
  msg.header.stamp = stamp;
  msg.header.frame_id = frame_id;
  msg.evidences.reserve(evidences.size());
  for (const auto& evidence : evidences) {
    deform_monitor_v2::RiskEvidence out;
    out.id = evidence.id;
    out.anchor_type = static_cast<uint8_t>(evidence.anchor_type);
    out.obs_state = static_cast<uint8_t>(evidence.obs_state);
    out.mode = static_cast<uint8_t>(evidence.mode);
    out.position = ToPoint(evidence.position_R);
    out.displacement = ToVector3(evidence.displacement_R);
    out.displacement_score = evidence.displacement_score;
    out.disappearance_score = evidence.disappearance_score;
    out.graph_score = evidence.graph_score;
    out.confidence = evidence.confidence;
    out.risk_score = evidence.risk_score;
    out.graph_neighbor_count = evidence.graph_neighbor_count;
    out.observable = evidence.observable;
    out.comparable = evidence.comparable;
    out.active = evidence.active;
    msg.evidences.push_back(out);
  }
  return msg;
}

deform_monitor_v2::RiskVoxelField RiskVisualizationPublisher::BuildRiskVoxelFieldMsg(
    const RiskVoxelVector& voxels,
    const ros::Time& stamp,
    const std::string& frame_id) const {
  deform_monitor_v2::RiskVoxelField msg;
  msg.header.stamp = stamp;
  msg.header.frame_id = frame_id;
  msg.voxel_size = params_.voxel_size;
  msg.voxels.reserve(voxels.size());
  for (const auto& voxel : voxels) {
    deform_monitor_v2::RiskVoxel out;
    out.center = ToPoint(voxel.center_R);
    out.risk_score = voxel.risk_score;
    out.confidence = voxel.confidence;
    out.displacement_component = voxel.displacement_component;
    out.disappearance_component = voxel.disappearance_component;
    out.source_count = voxel.source_count;
    out.significant = voxel.significant;
    msg.voxels.push_back(out);
  }
  return msg;
}

deform_monitor_v2::RiskRegions RiskVisualizationPublisher::BuildRiskRegionsMsg(
    const RiskRegionVector& regions,
    const ros::Time& stamp,
    const std::string& frame_id) const {
  deform_monitor_v2::RiskRegions msg;
  msg.header.stamp = stamp;
  msg.header.frame_id = frame_id;
  msg.regions.reserve(regions.size());
  for (const auto& region : regions) {
    deform_monitor_v2::RiskRegion out;
    out.id = region.id;
    out.region_type = static_cast<uint8_t>(region.type);
    out.center = ToPoint(region.center_R);
    out.bbox_min = ToPoint(region.bbox_min_R);
    out.bbox_max = ToPoint(region.bbox_max_R);
    out.mean_risk = region.mean_risk;
    out.peak_risk = region.peak_risk;
    out.confidence = region.confidence;
    out.voxel_count = region.voxel_count;
    out.significant = region.significant;
    msg.regions.push_back(out);
  }
  return msg;
}

deform_monitor_v2::PersistentRiskRegions RiskVisualizationPublisher::BuildPersistentRiskRegionsMsg(
    const PersistentRiskTrackVector& tracks,
    const ros::Time& stamp,
    const std::string& frame_id) const {
  deform_monitor_v2::PersistentRiskRegions msg;
  msg.header.stamp = stamp;
  msg.header.frame_id = frame_id;
  msg.regions.reserve(tracks.size());
  for (const auto& track : tracks) {
    deform_monitor_v2::PersistentRiskRegion out;
    const Eigen::Vector3d bbox_min =
        track.ever_confirmed ? track.union_bbox_min_R : track.last_bbox_min_R;
    const Eigen::Vector3d bbox_max =
        track.ever_confirmed ? track.union_bbox_max_R : track.last_bbox_max_R;
    out.track_id = track.track_id;
    out.state = ToPersistentRiskStateCode(track.state);
    out.region_type = ToPersistentRiskRegionTypeCode(track.region_type);
    out.center = ToPoint(track.last_center_R);
    out.bbox_min = ToPoint(bbox_min);
    out.bbox_max = ToPoint(bbox_max);
    out.mean_risk = track.ema_mean_risk;
    out.peak_risk = track.ema_peak_risk;
    out.confidence = track.ema_confidence;
    out.accumulated_risk = track.accumulated_risk;
    out.support_mass = track.support_mass;
    out.spatial_span = track.spatial_span;
    out.hit_streak = track.hit_streak;
    out.miss_streak = track.miss_streak;
    out.age_frames = track.age_frames;
    out.confirmed = track.state == PersistentRiskState::CONFIRMED;
    msg.regions.push_back(out);
  }
  return msg;
}

visualization_msgs::MarkerArray RiskVisualizationPublisher::BuildRiskMarkers(
    const RiskVoxelVector& voxels,
    const RiskRegionVector& regions,
    const ros::Time& stamp,
    const std::string& frame_id) const {
  visualization_msgs::MarkerArray array;
  visualization_msgs::Marker clear;
  clear.header.stamp = stamp;
  clear.header.frame_id = frame_id;
  clear.action = visualization_msgs::Marker::DELETEALL;
  array.markers.push_back(clear);

  int id = 0;
  visualization_msgs::Marker cubes;
  cubes.header.stamp = stamp;
  cubes.header.frame_id = frame_id;
  cubes.ns = "risk_voxels";
  cubes.id = id++;
  cubes.type = visualization_msgs::Marker::CUBE_LIST;
  cubes.action = visualization_msgs::Marker::ADD;
  cubes.pose.orientation.w = 1.0;
  cubes.scale.x = std::max(0.01, params_.voxel_size);
  cubes.scale.y = std::max(0.01, params_.voxel_size);
  cubes.scale.z = std::max(0.01, params_.voxel_size);
  cubes.points.reserve(voxels.size());
  cubes.colors.reserve(voxels.size());
  for (const auto& voxel : voxels) {
    if (voxel.risk_score < 0.5 * params_.min_voxel_risk ||
        voxel.confidence < 0.5 * params_.min_confidence) {
      continue;
    }
    cubes.points.push_back(ToPoint(voxel.center_R));
    cubes.colors.push_back(
        RiskColor(voxel.risk_score, params_.low_risk_alpha, params_.high_risk_alpha));
  }
  array.markers.push_back(cubes);

  visualization_msgs::Marker outlines;
  outlines.header.stamp = stamp;
  outlines.header.frame_id = frame_id;
  outlines.ns = "risk_regions";
  outlines.id = id++;
  outlines.type = visualization_msgs::Marker::LINE_LIST;
  outlines.action = visualization_msgs::Marker::ADD;
  outlines.pose.orientation.w = 1.0;
  outlines.scale.x = std::max(1.0e-3, params_.region_outline_width);
  outlines.color.r = 1.0f;
  outlines.color.g = 0.35f;
  outlines.color.b = 0.05f;
  outlines.color.a = params_.region_outline_alpha;
  for (const auto& region : regions) {
    if (!region.significant) {
      continue;
    }
    AppendBoxOutlinePoints(region.bbox_min_R, region.bbox_max_R, &outlines.points);
  }
  array.markers.push_back(outlines);
  return array;
}

visualization_msgs::MarkerArray RiskVisualizationPublisher::BuildPersistentRiskMarkers(
    const PersistentRiskTrackVector& tracks,
    const ros::Time& stamp,
    const std::string& frame_id) const {
  visualization_msgs::MarkerArray array;
  visualization_msgs::Marker clear;
  clear.header.stamp = stamp;
  clear.header.frame_id = frame_id;
  clear.action = visualization_msgs::Marker::DELETEALL;
  array.markers.push_back(clear);

  int marker_id = 0;
  for (const auto& track : tracks) {
    visualization_msgs::Marker shell;
    shell.header.stamp = stamp;
    shell.header.frame_id = frame_id;
    shell.ns = "persistent_risk_regions";
    shell.id = track.track_id >= 0 ? track.track_id : marker_id++;
    shell.type = track.state == PersistentRiskState::CONFIRMED
                    ? visualization_msgs::Marker::TRIANGLE_LIST
                    : visualization_msgs::Marker::LINE_LIST;
    shell.action = visualization_msgs::Marker::ADD;
    shell.pose.orientation.w = 1.0;
    shell.scale.x = track.state == PersistentRiskState::CONFIRMED
                        ? 1.0
                        : std::max(1.0e-3, params_.region_outline_width * 0.8);
    shell.scale.y = shell.scale.x;
    shell.scale.z = shell.scale.x;
    shell.color = PersistentTrackColor(track, PersistentTrackAlpha(track));
    if (track.state == PersistentRiskState::CONFIRMED) {
      shell.points.reserve(36);
      AppendBoxShellTriangles(track.union_bbox_min_R, track.union_bbox_max_R, &shell.points);
    } else {
      shell.points.reserve(24);
      AppendBoxOutlinePoints(track.last_bbox_min_R, track.last_bbox_max_R, &shell.points);
    }
    array.markers.push_back(shell);
  }

  return array;
}

}  // namespace deform_monitor_v2
