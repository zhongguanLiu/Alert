/*
 * Author: zgliu@cumt.edu.cn
 * Affiliation: China University of Mining and Technology
 * Open-source release date: 2026-04-20
 */

#include "deform_monitor_v2/visualization_publisher.hpp"

#include <pcl_conversions/pcl_conversions.h>

#include <sstream>
#include <unordered_map>
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

void FillMatrix3(const Eigen::Matrix3d& M, boost::array<double, 9>* out) {
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      (*out)[r * 3 + c] = M(r, c);
    }
  }
}

void FillMatrix6(const Eigen::Matrix<double, 6, 6>& M, boost::array<double, 36>* out) {
  for (int r = 0; r < 6; ++r) {
    for (int c = 0; c < 6; ++c) {
      (*out)[r * 6 + c] = M(r, c);
    }
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

bool IsActiveAnchor(const AnchorTrackState& state, const VisualizationParams& params) {
  return state.significant || state.persistent_candidate || state.disappearance_candidate ||
         state.reacquired || params.show_all_anchors ||
         (params.show_comparable_anchors && state.comparable);
}

std_msgs::ColorRGBA BuildAnchorColor(const AnchorTrackState& state, double alpha) {
  std_msgs::ColorRGBA color;
  color.a = static_cast<float>(alpha);
  if (state.mode == DetectionMode::DISAPPEARANCE) {
    color.r = 0.95f;
    color.g = 0.15f;
    color.b = 0.70f;
    return color;
  }

  color.r = state.significant ? 0.95f : 0.25f;
  color.g = state.significant ? 0.25f : 0.80f;
  color.b = state.reacquired ? 0.95f : 0.20f;
  return color;
}

std_msgs::ColorRGBA BuildClusterColor(const MotionClusterState& cluster, double alpha) {
  std_msgs::ColorRGBA color;
  color.a = static_cast<float>(alpha);
  if (cluster.mode == DetectionMode::DISAPPEARANCE) {
    color.r = 0.95f;
    color.g = 0.20f;
    color.b = 0.75f;
    return color;
  }

  color.r = cluster.significant ? 0.95f : 0.85f;
  color.g = cluster.significant ? 0.35f : 0.70f;
  color.b = 0.10f;
  return color;
}

}  // namespace

void VisualizationPublisher::SetParams(const VisualizationParams& params, ros::NodeHandle& nh) {
  params_ = params;
  migration_pub_ =
      nh.advertise<visualization_msgs::MarkerArray>("/deform/structure_migrations", 10, true);
}

deform_monitor_v2::AnchorStates VisualizationPublisher::BuildAnchorStatesMsg(
    const AnchorReferenceVector& anchors,
    const AnchorStateVector& states,
    const CurrentObservationVector& observations,
    const ros::Time& stamp,
    const std::string& frame_id) const {
  deform_monitor_v2::AnchorStates msg;
  msg.header.stamp = stamp;
  msg.header.frame_id = frame_id;
  const size_t N = std::min(anchors.size(), std::min(states.size(), observations.size()));
  msg.anchors.reserve(N);
  for (size_t i = 0; i < N; ++i) {
    deform_monitor_v2::AnchorState state_msg;
    state_msg.id = anchors[i].id;
    state_msg.anchor_type = static_cast<uint8_t>(anchors[i].type);
    state_msg.center = ToPoint(anchors[i].center_R);
    state_msg.normal = ToVector3(anchors[i].normal_R);
    state_msg.edge_normal = ToVector3(anchors[i].edge_normal_R);
    state_msg.visible_count = anchors[i].visible_count;
    state_msg.point_count = anchors[i].point_count;
    state_msg.ref_quality = anchors[i].ref_quality;
    state_msg.covariance_quality = anchors[i].covariance_quality;
    state_msg.type_stability = anchors[i].type_stability;
    const Eigen::Vector3d disp = states[i].x_mix.block<3, 1>(0, 0);
    const Eigen::Vector3d vel = states[i].x_mix.block<3, 1>(3, 0);
    for (int k = 0; k < 3; ++k) {
      state_msg.disp_mean[k] = disp(k);
      state_msg.vel_mean[k] = vel(k);
    }
    FillMatrix3(states[i].P_mix.block<3, 3>(0, 0), &state_msg.disp_cov);
    FillMatrix6(states[i].P_mix, &state_msg.state_cov);
    state_msg.dof_obs = states[i].dof_obs;
    state_msg.chi2_stat = states[i].chi2_stat;
    state_msg.disp_norm = states[i].disp_norm;
    state_msg.disp_normal = states[i].disp_normal;
    state_msg.disp_edge = states[i].disp_edge;
    state_msg.cmp_score = observations[i].cmp_score;
    state_msg.cusum_score = states[i].cusum_score;
    state_msg.comparable = states[i].comparable;
    state_msg.significant = states[i].significant;
    state_msg.reacquired = states[i].reacquired;
    state_msg.detection_mode = static_cast<uint8_t>(states[i].mode);
    state_msg.disappearance_score = states[i].disappearance_score;
    state_msg.ref_center = ToPoint(anchors[i].center_R);
    msg.anchors.push_back(state_msg);
  }
  return msg;
}

deform_monitor_v2::MotionClusters VisualizationPublisher::BuildMotionClustersMsg(
    const MotionClusterVector& clusters,
    const ros::Time& stamp,
    const std::string& frame_id) const {
  deform_monitor_v2::MotionClusters msg;
  msg.header.stamp = stamp;
  msg.header.frame_id = frame_id;
  msg.clusters.reserve(clusters.size());
  for (const auto& cluster : clusters) {
    deform_monitor_v2::MotionCluster cmsg;
    cmsg.id = cluster.id;
    cmsg.anchor_ids = cluster.anchor_ids;
    cmsg.center = ToPoint(cluster.center_R);
    cmsg.bbox_min = ToPoint(cluster.bbox_min_R);
    cmsg.bbox_max = ToPoint(cluster.bbox_max_R);
    for (int k = 0; k < 3; ++k) {
      cmsg.disp_mean[k] = cluster.disp_mean_R(k);
    }
    FillMatrix3(cluster.disp_cov, &cmsg.disp_cov);
    cmsg.chi2_stat = cluster.chi2_stat;
    cmsg.disp_norm = cluster.disp_norm;
    cmsg.confidence = cluster.confidence;
    cmsg.support_count = cluster.support_count;
    cmsg.significant = cluster.significant;
    msg.clusters.push_back(cmsg);
  }
  return msg;
}

sensor_msgs::PointCloud2 VisualizationPublisher::BuildDebugCloudMsg(
    const AnchorReferenceVector& anchors,
    const AnchorStateVector& states,
    const ros::Time& stamp,
    const std::string& frame_id) const {
  pcl::PointCloud<pcl::PointXYZI> cloud;
  const size_t N = std::min(anchors.size(), states.size());
  cloud.points.reserve(N);
  for (size_t i = 0; i < N; ++i) {
    const bool active_anchor = states[i].significant ||
                               states[i].persistent_candidate ||
                               states[i].disappearance_candidate ||
                               states[i].reacquired ||
                               params_.show_all_anchors ||
                               (params_.show_comparable_anchors && states[i].comparable);
    if (!active_anchor) {
      continue;
    }
    pcl::PointXYZI pt;
    pt.x = anchors[i].center_R.x();
    pt.y = anchors[i].center_R.y();
    pt.z = anchors[i].center_R.z();
    pt.intensity = static_cast<float>(
        states[i].mode == DetectionMode::DISAPPEARANCE
            ? states[i].disappearance_score * 1000.0
            : (states[i].significant ? states[i].disp_norm * 1000.0
                                     : states[i].chi2_stat));
    cloud.points.push_back(pt);
  }
  cloud.width = cloud.points.size();
  cloud.height = 1;
  cloud.is_dense = false;

  sensor_msgs::PointCloud2 msg;
  pcl::toROSMsg(cloud, msg);
  msg.header.stamp = stamp;
  msg.header.frame_id = frame_id;
  return msg;
}

visualization_msgs::MarkerArray VisualizationPublisher::BuildAnchorMarkers(
    const AnchorReferenceVector& anchors,
    const AnchorStateVector& states,
    const ros::Time& stamp,
    const std::string& frame_id) const {
  visualization_msgs::MarkerArray array;
  visualization_msgs::Marker clear;
  clear.header.stamp = stamp;
  clear.header.frame_id = frame_id;
  clear.action = visualization_msgs::Marker::DELETEALL;
  array.markers.push_back(clear);

  int id = 0;
  const size_t N = std::min(anchors.size(), states.size());
  for (size_t i = 0; i < N; ++i) {
    if (!IsActiveAnchor(states[i], params_)) {
      continue;
    }

    visualization_msgs::Marker marker;
    marker.header.stamp = stamp;
    marker.header.frame_id = frame_id;
    marker.ns = "anchors";
    marker.id = id++;
    marker.type = visualization_msgs::Marker::SPHERE;
    marker.action = visualization_msgs::Marker::ADD;
    marker.pose.orientation.w = 1.0;
    marker.pose.position = ToPoint(anchors[i].center_R);
    marker.scale.x = 0.04;
    marker.scale.y = 0.04;
    marker.scale.z = 0.04;
    marker.color = BuildAnchorColor(states[i], states[i].comparable ? 0.85 : 0.45);
    array.markers.push_back(marker);
  }

  return array;
}

visualization_msgs::MarkerArray VisualizationPublisher::BuildMotionMarkers(
    const AnchorReferenceVector& anchors,
    const AnchorStateVector& states,
    const MotionClusterVector& clusters,
    const ros::Time& stamp,
    const std::string& frame_id) const {
  visualization_msgs::MarkerArray array;
  visualization_msgs::Marker clear;
  clear.header.stamp = stamp;
  clear.header.frame_id = frame_id;
  clear.action = visualization_msgs::Marker::DELETEALL;
  array.markers.push_back(clear);
  int id = 0;

  const size_t N = std::min(anchors.size(), states.size());
  for (size_t i = 0; i < N; ++i) {
    if (!IsActiveAnchor(states[i], params_)) {
      continue;
    }

    const bool reacquired_supported =
        states[i].reacquired &&
        (states[i].cluster_member || states[i].graph_candidate || states[i].graph_neighbor_count >= 2);
    const bool arrow_gate =
        !params_.arrows_only_clustered_or_reacquired ||
        states[i].cluster_member ||
        states[i].graph_candidate ||
        reacquired_supported ||
        states[i].directional_persistent ||
        (states[i].local_contrast_score >= params_.min_arrow_contrast_score &&
         states[i].graph_neighbor_count >= 2);
    const bool show_arrow =
        states[i].mode == DetectionMode::DISPLACEMENT &&
        states[i].disp_norm >= params_.min_arrow_disp &&
        arrow_gate &&
        (states[i].significant || states[i].graph_candidate || reacquired_supported);


    const bool show_permanent_arrow =
        states[i].permanent_deformed &&
        !states[i].significant &&
        states[i].D_max.norm() >= params_.min_arrow_disp;
    if (!show_arrow && !show_permanent_arrow) {
      continue;
    }

    if (show_arrow) {
      visualization_msgs::Marker arrow;
      arrow.header.stamp = stamp;
      arrow.header.frame_id = frame_id;
      arrow.ns = "anchor_disp";
      arrow.id = id++;
      arrow.type = visualization_msgs::Marker::ARROW;
      arrow.action = visualization_msgs::Marker::ADD;
      arrow.scale.x = params_.arrow_shaft_diameter;
      arrow.scale.y = params_.arrow_head_diameter;
      arrow.scale.z = params_.arrow_head_length;
      arrow.pose.orientation.w = 1.0;
      arrow.color = BuildAnchorColor(states[i], 0.95);
      geometry_msgs::Point p0 = ToPoint(anchors[i].center_R);
      Eigen::Vector3d disp = states[i].x_mix.block<3, 1>(0, 0);
      const double max_arrow_disp = std::max(params_.min_arrow_disp, params_.max_arrow_disp);
      const double disp_norm = disp.norm();
      if (disp_norm > max_arrow_disp && disp_norm > 1.0e-9) {
        disp *= max_arrow_disp / disp_norm;
      }
      const double scale = std::max(1.0, params_.arrow_disp_scale);
      geometry_msgs::Point p1 = ToPoint(anchors[i].center_R + disp * scale);
      arrow.points.push_back(p0);
      arrow.points.push_back(p1);
      array.markers.push_back(arrow);

      if (reacquired_supported) {
        visualization_msgs::Marker reacq;
        reacq.header = arrow.header;
        reacq.ns = "reacquired_centers";
        reacq.id = id++;
        reacq.type = visualization_msgs::Marker::SPHERE;
        reacq.action = visualization_msgs::Marker::ADD;
        reacq.pose.orientation.w = 1.0;
        reacq.pose.position = ToPoint(states[i].matched_center_R);
        reacq.scale.x = 0.025;
        reacq.scale.y = 0.025;
        reacq.scale.z = 0.025;
        reacq.color = BuildAnchorColor(states[i], 0.95);
        array.markers.push_back(reacq);
      }
    }

    if (show_permanent_arrow) {
      visualization_msgs::Marker parrow;
      parrow.header.stamp = stamp;
      parrow.header.frame_id = frame_id;
      parrow.ns = "permanent_deform";
      parrow.id = id++;
      parrow.type = visualization_msgs::Marker::ARROW;
      parrow.action = visualization_msgs::Marker::ADD;
      parrow.scale.x = params_.arrow_shaft_diameter * 0.6;
      parrow.scale.y = params_.arrow_head_diameter * 0.6;
      parrow.scale.z = params_.arrow_head_length * 0.6;
      parrow.pose.orientation.w = 1.0;
      parrow.color.r = 1.0f;
      parrow.color.g = 0.50f;
      parrow.color.b = 0.0f;
      parrow.color.a = 0.75f;
      Eigen::Vector3d d_max = states[i].D_max;
      const double max_arrow_disp = std::max(params_.min_arrow_disp, params_.max_arrow_disp);
      if (d_max.norm() > max_arrow_disp && d_max.norm() > 1.0e-9) {
        d_max *= max_arrow_disp / d_max.norm();
      }
      parrow.points.push_back(ToPoint(anchors[i].center_R));
      parrow.points.push_back(ToPoint(anchors[i].center_R + d_max));
      array.markers.push_back(parrow);
    }
  }

  for (const auto& cluster : clusters) {
    if (!cluster.significant) {
      continue;
    }
    if (!params_.show_cluster_boxes) {
      continue;
    }
    visualization_msgs::Marker box;
    box.header.stamp = stamp;
    box.header.frame_id = frame_id;
    box.ns = "clusters";
    box.id = id++;
    box.type = visualization_msgs::Marker::CUBE;
    box.action = visualization_msgs::Marker::ADD;
    box.pose.orientation.w = 1.0;
    box.pose.position = ToPoint(cluster.center_R);
    box.scale.x = std::max(params_.cluster_min_box_size,
                           cluster.bbox_max_R.x() - cluster.bbox_min_R.x());
    box.scale.y = std::max(params_.cluster_min_box_size,
                           cluster.bbox_max_R.y() - cluster.bbox_min_R.y());
    box.scale.z = std::max(params_.cluster_min_box_size,
                           cluster.bbox_max_R.z() - cluster.bbox_min_R.z());
    box.color = BuildClusterColor(cluster, params_.cluster_box_alpha);
    array.markers.push_back(box);

    visualization_msgs::Marker outline;
    outline.header = box.header;
    outline.ns = "cluster_outline";
    outline.id = id++;
    outline.type = visualization_msgs::Marker::LINE_LIST;
    outline.action = visualization_msgs::Marker::ADD;
    outline.pose.orientation.w = 1.0;
    outline.scale.x = params_.cluster_outline_width;
    outline.color = BuildClusterColor(cluster, params_.cluster_outline_alpha);
    AppendBoxOutlinePoints(cluster.bbox_min_R, cluster.bbox_max_R, &outline.points);
    array.markers.push_back(outline);

    if (cluster.mode == DetectionMode::DISPLACEMENT && cluster.disp_norm > 1.0e-4) {
      visualization_msgs::Marker carr;
      carr.header = box.header;
      carr.ns = "cluster_disp";
      carr.id = id++;
      carr.type = visualization_msgs::Marker::ARROW;
      carr.action = visualization_msgs::Marker::ADD;
      carr.scale.x = params_.arrow_shaft_diameter * 1.2;
      carr.scale.y = params_.arrow_head_diameter * 1.2;
      carr.scale.z = params_.arrow_head_length * 1.2;
      carr.color = BuildClusterColor(cluster, 0.95);
      carr.points.push_back(ToPoint(cluster.center_R));
      carr.points.push_back(ToPoint(cluster.center_R + cluster.disp_mean_R));
      array.markers.push_back(carr);
    }

    const bool show_text =
        params_.show_cluster_text &&
        (!params_.text_only_significant || cluster.significant);
    if (show_text) {
      visualization_msgs::Marker text;
      text.header = box.header;
      text.ns = "cluster_text";
      text.id = id++;
      text.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
      text.action = visualization_msgs::Marker::ADD;
      text.pose.orientation.w = 1.0;
      text.pose.position = ToPoint(cluster.bbox_max_R + Eigen::Vector3d(0.0, 0.0, 0.05));
      text.scale.z = 0.07;
      text.color.a = 0.95;
      text.color.r = 1.0;
      text.color.g = 1.0;
      text.color.b = 1.0;
      std::ostringstream oss;
      if (cluster.mode == DetectionMode::DISAPPEARANCE) {
        oss << "C" << cluster.id << " gone score=" << cluster.evidence_score;
      } else {
        oss << "C" << cluster.id << " disp |u|=" << cluster.disp_norm
            << " chi2=" << cluster.chi2_stat;
      }
      text.text = oss.str();
      array.markers.push_back(text);
    }
  }

  return array;
}


void VisualizationPublisher::PublishStructureMigrations(
    const StructureUnitVector& units,
    const StructureMigrationVector& migrations,
    const ros::Time& stamp,
    const std::string& frame_id) {

  visualization_msgs::MarkerArray markers;
  int mid = 0;


  {
    visualization_msgs::Marker clear_marker;
    clear_marker.action = visualization_msgs::Marker::DELETEALL;
    markers.markers.push_back(clear_marker);
  }


  std::unordered_map<int, const StructureUnit*> uid_map;
  for (const auto& u : units) uid_map[u.id] = &u;

  for (const auto& mig : migrations) {
    if (!mig.confirmed) continue;
    auto it = uid_map.find(mig.unit_id);
    if (it == uid_map.end()) continue;
    const StructureUnit& unit = *it->second;


    {
      visualization_msgs::Marker m;
      m.header.frame_id = frame_id;
      m.header.stamp = stamp;
      m.ns = "migration_old_bbox";
      m.id = mid++;
      m.type = visualization_msgs::Marker::LINE_LIST;
      m.action = visualization_msgs::Marker::ADD;
      m.scale.x = 0.01;
      m.color.r = 1.0f; m.color.g = 0.0f; m.color.b = 0.0f; m.color.a = 0.6f;
      m.pose.orientation.w = 1.0;
      AppendBoxOutlinePoints(unit.ref_bbox_min_R, unit.ref_bbox_max_R, &m.points);
      markers.markers.push_back(m);
    }


    {
      visualization_msgs::Marker m;
      m.header.frame_id = frame_id;
      m.header.stamp = stamp;
      m.ns = "migration_new_bbox";
      m.id = mid++;
      m.type = visualization_msgs::Marker::LINE_LIST;
      m.action = visualization_msgs::Marker::ADD;
      m.scale.x = 0.01;
      m.color.r = 0.0f; m.color.g = 1.0f; m.color.b = 0.0f; m.color.a = 0.6f;
      m.pose.orientation.w = 1.0;
      AppendBoxOutlinePoints(mig.entry_bbox_min_R, mig.entry_bbox_max_R, &m.points);
      markers.markers.push_back(m);
    }


    {
      visualization_msgs::Marker m;
      m.header.frame_id = frame_id;
      m.header.stamp = stamp;
      m.ns = "migration_arrow";
      m.id = mid++;
      m.type = visualization_msgs::Marker::ARROW;
      m.action = visualization_msgs::Marker::ADD;
      m.scale.x = 0.02; m.scale.y = 0.04; m.scale.z = 0.06;
      m.color.r = 1.0f; m.color.g = 1.0f; m.color.b = 1.0f; m.color.a = 0.9f;
      m.pose.orientation.w = 1.0;
      geometry_msgs::Point p_start, p_end;
      p_start.x = unit.ref_centroid_R.x();
      p_start.y = unit.ref_centroid_R.y();
      p_start.z = unit.ref_centroid_R.z();
      p_end.x = mig.entry_centroid_R.x();
      p_end.y = mig.entry_centroid_R.y();
      p_end.z = mig.entry_centroid_R.z();
      m.points.push_back(p_start);
      m.points.push_back(p_end);
      markers.markers.push_back(m);
    }


    {
      visualization_msgs::Marker m;
      m.header.frame_id = frame_id;
      m.header.stamp = stamp;
      m.ns = "migration_text";
      m.id = mid++;
      m.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
      m.action = visualization_msgs::Marker::ADD;
      m.pose.position.x = mig.entry_centroid_R.x();
      m.pose.position.y = mig.entry_centroid_R.y();
      m.pose.position.z = mig.entry_centroid_R.z() + 0.1;
      m.pose.orientation.w = 1.0;
      m.scale.z = 0.05;
      m.color.r = 1.0f; m.color.g = 1.0f; m.color.b = 1.0f; m.color.a = 1.0f;
      char buf[64];
      snprintf(buf, sizeof(buf), "U%d d=%.3fm c=%.2f",
               mig.unit_id, mig.T.norm(), mig.confidence);
      m.text = buf;
      markers.markers.push_back(m);
    }
  }

  migration_pub_.publish(markers);
}

}  // namespace deform_monitor_v2
