#include "deform_monitor_v2/core/motion_clusterer.hpp"

#include "deform_monitor_v2/core/observable_subspace.hpp"

#include <cmath>
#include <limits>
#include <numeric>
#include <unordered_map>

namespace deform_monitor_v2 {

namespace {

struct DisjointSet {
  explicit DisjointSet(size_t n) : parent(n), rank(n, 0) {
    std::iota(parent.begin(), parent.end(), 0);
  }

  size_t Find(size_t x) {
    if (parent[x] != x) {
      parent[x] = Find(parent[x]);
    }
    return parent[x];
  }

  void Unite(size_t a, size_t b) {
    a = Find(a);
    b = Find(b);
    if (a == b) {
      return;
    }
    if (rank[a] < rank[b]) {
      std::swap(a, b);
    }
    parent[b] = a;
    if (rank[a] == rank[b]) {
      ++rank[a];
    }
  }

  std::vector<size_t> parent;
  std::vector<int> rank;
};

double Clamp01(double value) {
  return std::max(0.0, std::min(1.0, value));
}

double TimeCorrelation(const std::deque<double>& a, const std::deque<double>& b) {
  const size_t n = std::min(a.size(), b.size());
  if (n < 3) {
    return 0.0;
  }
  double mean_a = 0.0;
  double mean_b = 0.0;
  for (size_t i = 0; i < n; ++i) {
    mean_a += a[a.size() - n + i];
    mean_b += b[b.size() - n + i];
  }
  mean_a /= static_cast<double>(n);
  mean_b /= static_cast<double>(n);

  double cov = 0.0;
  double var_a = 0.0;
  double var_b = 0.0;
  for (size_t i = 0; i < n; ++i) {
    const double da = a[a.size() - n + i] - mean_a;
    const double db = b[b.size() - n + i] - mean_b;
    cov += da * db;
    var_a += da * da;
    var_b += db * db;
  }
  if (var_a < 1.0e-9 || var_b < 1.0e-9) {
    return 0.0;
  }
  const double corr = cov / std::sqrt(var_a * var_b);
  return Clamp01(0.5 * (corr + 1.0));
}

ObjectAssociationState SummarizeAssociationState(int consistent_count,
                                                 int mismatch_count,
                                                 int mixed_count) {
  if (mixed_count > 0) {
    return ObjectAssociationState::MIXED;
  }
  if (mismatch_count > 0) {
    return ObjectAssociationState::MISMATCH;
  }
  if (consistent_count > 0) {
    return ObjectAssociationState::CONSISTENT;
  }
  return ObjectAssociationState::UNAVAILABLE;
}

}  // namespace

void MotionClusterer::SetParams(const ClusterParams& params) {
  params_ = params;
}

MotionClusterVector MotionClusterer::Cluster(
    const AnchorReferenceVector& anchors,
    const AnchorStateVector& states) const {
  MotionClusterVector clusters;
  if (anchors.size() != states.size() || anchors.empty()) {
    return clusters;
  }

  std::vector<size_t> active;
  active.reserve(anchors.size());
  for (size_t i = 0; i < anchors.size(); ++i) {
    if (states[i].significant || states[i].disappearance_candidate) {
      active.push_back(i);
    }
  }
  if (active.empty()) {
    return clusters;
  }

  std::vector<int> active_pos(anchors.size(), -1);
  for (size_t ai = 0; ai < active.size(); ++ai) {
    active_pos[active[ai]] = static_cast<int>(ai);
  }

  DisjointSet dsu(active.size());
  const double neighbor_radius = std::max(params_.tau_d, params_.compact_tau_d);
  for (size_t ai = 0; ai < active.size(); ++ai) {
    const size_t i = active[ai];
    for (const int neighbor_idx : anchors[i].neighbor_indices) {
      const size_t j = static_cast<size_t>(neighbor_idx);
      if (j <= i) {
        continue;
      }
      const int aj_pos = active_pos[j];
      if (aj_pos < 0) {
        continue;
      }
      const size_t aj = static_cast<size_t>(aj_pos);
      if (aj <= ai) {
        continue;
      }
      const double dist = (anchors[i].center_R - anchors[j].center_R).norm();
      if (dist > neighbor_radius) {
        continue;
      }
      const double normal_angle = AngleBetweenDeg(anchors[i].normal_R, anchors[j].normal_R);
      if (normal_angle > params_.tau_ng_deg) {
        continue;
      }

      const ObservableSubspace subspace_i = BuildObservableSubspace(anchors[i]);
      const ObservableSubspace subspace_j = BuildObservableSubspace(anchors[j]);
      const Eigen::Vector3d ui = ProjectObservableVector(
          states[i].x_mix.block<3, 1>(0, 0), subspace_i, states[i].dof_obs);
      const Eigen::Vector3d uj = ProjectObservableVector(
          states[j].x_mix.block<3, 1>(0, 0), subspace_j, states[j].dof_obs);
      const double ni = ui.norm();
      const double nj = uj.norm();
      double dir_cos = 0.0;
      if (ni < 1.0e-6 && nj < 1.0e-6) {
        dir_cos = 1.0;
      } else if (ni > 1.0e-6 && nj > 1.0e-6) {
        dir_cos = ui.dot(uj) / (ni * nj);
      }
      if (dir_cos < params_.tau_u_cos) {
        continue;
      }

      const double corr_time =
          TimeCorrelation(states[i].evidence_history, states[j].evidence_history);
      const double graph_time =
          std::min(states[i].graph_temporal_score, states[j].graph_temporal_score);
      const double graph_coherent =
          std::min(states[i].graph_coherent_score, states[j].graph_coherent_score);
      const double temporal_support = std::max(corr_time, graph_time);
      if (temporal_support < params_.tau_corr) {
        continue;
      }

      const bool disappear_pair =
          states[i].mode == DetectionMode::DISAPPEARANCE ||
          states[j].mode == DetectionMode::DISAPPEARANCE ||
          states[i].disappearance_candidate ||
          states[j].disappearance_candidate;
      const double score = disappear_pair
                               ? 0.40 * std::exp(-(dist * dist) /
                                                 std::max(1.0e-6, params_.sigma_d * params_.sigma_d)) +
                                     0.30 * Clamp01(temporal_support) +
                                     0.30 * std::exp(-std::abs(states[i].disappearance_score -
                                                               states[j].disappearance_score) /
                                                     std::max(1.0e-6, params_.tau_cluster_disappear))
                               : params_.beta_dir * Clamp01(dir_cos) +
                                     params_.beta_mag *
                                         std::exp(-std::abs(ni - nj) /
                                                  std::max(1.0e-6, params_.sigma_m)) +
                                     params_.beta_dist *
                                         std::exp(-(dist * dist) /
                                                  std::max(1.0e-6,
                                                           params_.sigma_d * params_.sigma_d)) +
                                     params_.beta_time *
                                         Clamp01(std::max(temporal_support, graph_coherent));
      if (score >= params_.tau_edge_score) {
        dsu.Unite(ai, aj);
        continue;
      }

      if (!params_.enable_compact_motion) {
        continue;
      }
      if (disappear_pair) {
        continue;
      }
      if (!states[i].comparable || !states[j].comparable) {
        continue;
      }
      if (anchors[i].type == AnchorType::PLANE && anchors[j].type == AnchorType::PLANE) {
        continue;
      }
      if (dist > params_.compact_tau_d) {
        continue;
      }
      if (normal_angle > params_.compact_tau_ng_deg) {
        continue;
      }
      if (dir_cos < params_.compact_tau_u_cos) {
        continue;
      }
      if (temporal_support < params_.compact_tau_corr) {
        continue;
      }
      const bool seed_i = ni > params_.compact_seed_disp ||
                          states[i].chi2_stat > params_.compact_seed_chi2 ||
                          states[i].graph_candidate;
      const bool seed_j = nj > params_.compact_seed_disp ||
                          states[j].chi2_stat > params_.compact_seed_chi2 ||
                          states[j].graph_candidate;
      if (!(seed_i || seed_j)) {
        continue;
      }
      dsu.Unite(ai, aj);
    }
  }

  std::unordered_map<size_t, std::vector<size_t>> groups;
  for (size_t k = 0; k < active.size(); ++k) {
    groups[dsu.Find(k)].push_back(active[k]);
  }

  int next_id = 0;
  for (const auto& group : groups) {
    MotionClusterState cluster;
    cluster.id = next_id++;
    cluster.anchor_ids.reserve(group.second.size());
    cluster.bbox_min_R = Eigen::Vector3d::Constant(std::numeric_limits<double>::max());
    cluster.bbox_max_R = Eigen::Vector3d::Constant(-std::numeric_limits<double>::max());

    int disappear_votes = 0;
    double disappear_score_sum = 0.0;
    Eigen::Matrix3d sum_W = Eigen::Matrix3d::Zero();
    Eigen::Vector3d sum_Wu = Eigen::Vector3d::Zero();
    Eigen::Vector3d weighted_center = Eigen::Vector3d::Zero();
    double center_weight_sum = 0.0;
    uint16_t reference_object_id = 0;
    bool reference_object_seen = false;
    bool reference_object_complete = true;
    bool reference_object_ambiguous = false;
    double reference_object_confidence_sum = 0.0;
    int reference_object_confidence_count = 0;
    uint16_t observed_object_id = 0;
    bool observed_object_seen = false;
    bool observed_object_complete = true;
    bool observed_object_ambiguous = false;
    double observed_object_confidence_sum = 0.0;
    int observed_object_confidence_count = 0;

    for (const size_t idx : group.second) {
      cluster.anchor_ids.push_back(anchors[idx].id);
      cluster.bbox_min_R = cluster.bbox_min_R.cwiseMin(anchors[idx].center_R);
      cluster.bbox_max_R = cluster.bbox_max_R.cwiseMax(anchors[idx].center_R);
      if (states[idx].mode == DetectionMode::DISAPPEARANCE ||
          states[idx].disappearance_candidate) {
        ++disappear_votes;
        disappear_score_sum += states[idx].disappearance_score;
      }

      const Eigen::Matrix3d Sigma_u = states[idx].P_mix.block<3, 3>(0, 0);
      const ObservableSubspace subspace = BuildObservableSubspace(anchors[idx]);
      const int observable_rank = std::max(0, std::min(subspace.rank, states[idx].dof_obs));
      Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
      Eigen::Vector3d u = Eigen::Vector3d::Zero();
      if (observable_rank > 0) {
        const auto basis = subspace.basis_R.leftCols(observable_rank);
        const Eigen::MatrixXd projected_covariance = basis.transpose() * Sigma_u * basis;
        W = basis * PseudoInverse(projected_covariance, 1.0e-9) * basis.transpose();
        u = ProjectObservableVector(
            states[idx].x_mix.block<3, 1>(0, 0), subspace, observable_rank);
      }
      sum_W += W;
      sum_Wu += W * u;

      const double w_center = 1.0 / (Sigma_u.trace() + 1.0e-6);
      weighted_center += w_center * anchors[idx].center_R;
      center_weight_sum += w_center;

      if (!anchors[idx].object_id_valid) {
        reference_object_complete = false;
      } else {
        if (!reference_object_seen) {
          reference_object_id = anchors[idx].object_id;
          reference_object_seen = true;
        } else if (reference_object_id != anchors[idx].object_id) {
          reference_object_ambiguous = true;
        }
        reference_object_confidence_sum += Clamp01(anchors[idx].object_id_confidence);
        ++reference_object_confidence_count;
      }

      if (!states[idx].observed_object_id_valid) {
        observed_object_complete = false;
      } else {
        if (!observed_object_seen) {
          observed_object_id = states[idx].observed_object_id;
          observed_object_seen = true;
        } else if (observed_object_id != states[idx].observed_object_id) {
          observed_object_ambiguous = true;
        }
        observed_object_confidence_sum +=
            Clamp01(states[idx].observed_object_id_confidence);
        ++observed_object_confidence_count;
      }

      switch (states[idx].object_association_state) {
        case ObjectAssociationState::CONSISTENT:
          ++cluster.association_consistent_count;
          break;
        case ObjectAssociationState::MISMATCH:
          ++cluster.association_mismatch_count;
          break;
        case ObjectAssociationState::MIXED:
          ++cluster.association_mixed_count;
          break;
        case ObjectAssociationState::UNAVAILABLE:
        default:
          ++cluster.association_unavailable_count;
          break;
      }
    }

    cluster.object_id = reference_object_id;
    cluster.object_id_ambiguous = reference_object_ambiguous;
    cluster.object_id_valid = reference_object_seen && reference_object_complete &&
                              !reference_object_ambiguous;
    cluster.object_id_confidence =
        cluster.object_id_valid && reference_object_confidence_count > 0
            ? Clamp01(reference_object_confidence_sum /
                      static_cast<double>(reference_object_confidence_count))
            : 0.0;
    cluster.observed_object_id = observed_object_id;
    cluster.observed_object_id_ambiguous = observed_object_ambiguous;
    cluster.observed_object_id_valid = observed_object_seen && observed_object_complete &&
                                       !observed_object_ambiguous;
    cluster.observed_object_id_confidence =
        cluster.observed_object_id_valid && observed_object_confidence_count > 0
            ? Clamp01(observed_object_confidence_sum /
                      static_cast<double>(observed_object_confidence_count))
            : 0.0;
    cluster.object_association_state = SummarizeAssociationState(
        cluster.association_consistent_count,
        cluster.association_mismatch_count,
        cluster.association_mixed_count);

    cluster.mode = disappear_votes * 2 >= static_cast<int>(group.second.size())
                       ? DetectionMode::DISAPPEARANCE
                       : DetectionMode::DISPLACEMENT;
    if (sum_W.norm() < 1.0e-9 && cluster.mode != DetectionMode::DISAPPEARANCE) {
      continue;
    }

    if (cluster.mode == DetectionMode::DISPLACEMENT) {
      cluster.disp_cov = PseudoInverse(sum_W, 1.0e-9);
      cluster.disp_mean_R = cluster.disp_cov * sum_Wu;
    } else {
      cluster.disp_cov = Eigen::Matrix3d::Identity() * 1.0e-3;
      cluster.disp_mean_R.setZero();
    }
    cluster.center_R = weighted_center / std::max(1.0e-9, center_weight_sum);
    cluster.chi2_stat = cluster.mode == DetectionMode::DISPLACEMENT
                            ? Chi2PseudoInverse(cluster.disp_mean_R, cluster.disp_cov)
                            : disappear_score_sum / std::max(1.0, static_cast<double>(group.second.size()));
    cluster.disp_norm = cluster.mode == DetectionMode::DISPLACEMENT
                            ? cluster.disp_mean_R.norm()
                            : cluster.chi2_stat;
    cluster.evidence_score = cluster.mode == DetectionMode::DISPLACEMENT
                                 ? cluster.disp_norm
                                 : cluster.chi2_stat;
    cluster.support_count = static_cast<int>(group.second.size());
    const Eigen::Vector3d bbox_size = cluster.bbox_max_R - cluster.bbox_min_R;
    const double bbox_diag = bbox_size.norm();
    cluster.confidence = cluster.mode == DetectionMode::DISPLACEMENT
                             ? Clamp01(0.4 * static_cast<double>(cluster.support_count) /
                                           std::max(1, params_.min_cluster_size) +
                                       0.3 * cluster.disp_norm /
                                           std::max(1.0e-6, params_.tau_cluster_disp) +
                                       0.3 * cluster.chi2_stat /
                                           std::max(1.0e-6, params_.tau_cluster_chi2))
                             : Clamp01(0.5 * static_cast<double>(cluster.support_count) /
                                           std::max(1, params_.min_cluster_size_disappear) +
                                       0.5 * cluster.evidence_score /
                                           std::max(1.0e-6, params_.tau_cluster_disappear));
    const bool strict_displacement_sig =
        cluster.support_count >= params_.min_cluster_size &&
        cluster.chi2_stat > params_.tau_cluster_chi2 &&
        cluster.disp_norm > params_.tau_cluster_disp;
    bool compact_displacement_sig = false;
    if (params_.enable_compact_motion &&
        cluster.mode == DetectionMode::DISPLACEMENT) {
      int non_plane_count = 0;
      int comparable_count = 0;
      for (const size_t idx : group.second) {
        if (anchors[idx].type != AnchorType::PLANE) {
          ++non_plane_count;
        }
        if (states[idx].comparable) {
          ++comparable_count;
        }
      }
      compact_displacement_sig =
          cluster.support_count >= params_.compact_min_cluster_size &&
          comparable_count >= params_.compact_min_cluster_size &&
          non_plane_count >= 1 &&
          bbox_diag <= params_.compact_max_bbox_diag &&
          cluster.disp_norm > params_.compact_tau_cluster_disp &&
          cluster.chi2_stat > params_.compact_tau_cluster_chi2;
    }
    cluster.significant = cluster.mode == DetectionMode::DISPLACEMENT
                              ? (strict_displacement_sig || compact_displacement_sig)
                              : cluster.support_count >= params_.min_cluster_size_disappear &&
                                    cluster.evidence_score > params_.tau_cluster_disappear;
    clusters.push_back(cluster);
  }

  return clusters;
}

}  // namespace deform_monitor_v2
