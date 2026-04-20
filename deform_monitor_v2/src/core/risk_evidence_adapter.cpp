/*
 * Author: zgliu@cumt.edu.cn
 * Affiliation: China University of Mining and Technology
 * Open-source release date: 2026-04-20
 */

#include "deform_monitor_v2/core/risk_evidence_adapter.hpp"

#include <algorithm>

namespace deform_monitor_v2 {

namespace {

double Clamp01(double value) {
  return std::max(0.0, std::min(1.0, value));
}

double Max3(double a, double b, double c) {
  return std::max(a, std::max(b, c));
}

}  // namespace

void RiskEvidenceAdapter::SetParams(const RiskVisualizationParams& risk_params,
                                    const SignificanceParams& significance_params,
                                    const GraphTemporalParams& graph_params) {
  risk_params_ = risk_params;
  significance_params_ = significance_params;
  graph_params_ = graph_params;
}

RiskEvidenceVector RiskEvidenceAdapter::Build(
    const AnchorReferenceVector& anchors,
    const AnchorStateVector& states,
    const CurrentObservationVector& observations,
    const MotionClusterVector& /*clusters*/) const {
  RiskEvidenceVector evidences;
  const size_t N = std::min(anchors.size(), std::min(states.size(), observations.size()));
  evidences.reserve(N);
  for (size_t i = 0; i < N; ++i) {
    const auto& anchor = anchors[i];
    const auto& state = states[i];
    const auto& obs = observations[i];

    RiskEvidenceState evidence;
    evidence.id = anchor.id;
    evidence.anchor_type = anchor.type;
    evidence.obs_state = state.gate_state;
    evidence.mode = state.mode;
    evidence.position_R = anchor.center_R;
    evidence.displacement_R = state.x_mix.block<3, 1>(0, 0);
    evidence.graph_neighbor_count = state.graph_neighbor_count;
    evidence.observable = state.observable;
    evidence.comparable = state.comparable;

    const double amp_norm =
        Clamp01(state.disp_norm / std::max(1.0e-6, significance_params_.tau_A_norm));
    const double amp_normal =
        Clamp01(state.disp_normal / std::max(1.0e-6, significance_params_.tau_A_normal));
    const double amp_edge =
        Clamp01(state.disp_edge / std::max(1.0e-6, significance_params_.tau_A_edge));
    const double amp_score = anchor.type == AnchorType::PLANE
                                 ? std::max(amp_norm, amp_normal)
                                 : Max3(amp_norm, amp_normal, amp_edge);
    const double chi2_score = Clamp01(state.chi2_stat / 25.0);
    evidence.displacement_score = Clamp01(0.55 * amp_score + 0.45 * chi2_score);

    const double disappear_base = Clamp01(state.disappearance_score);
    const double disappear_boost =
        state.disappearance_candidate || state.mode == DetectionMode::DISAPPEARANCE ? 1.0 : 0.0;
    evidence.disappearance_score = Clamp01(std::max(disappear_base, 0.75 * disappear_boost));

    const double graph_support = Clamp01(state.graph_coherent_score);
    const double graph_temporal = Clamp01(state.graph_temporal_score);
    const double graph_persistent =
        Clamp01(state.graph_persistence_score / std::max(1.0e-6, graph_params_.cusum_h));
    evidence.graph_score =
        Clamp01(0.35 * graph_support + 0.35 * graph_temporal + 0.30 * graph_persistent);

    const double ref_conf =
        Clamp01(0.40 * anchor.ref_quality +
                0.35 * anchor.covariance_quality +
                0.25 * anchor.type_stability);
    const double obs_conf =
        state.gate_state == ObsGateState::OBSERVABLE_MATCHED
            ? 1.0
            : (state.gate_state == ObsGateState::OBSERVABLE_REPLACED ||
               state.gate_state == ObsGateState::OBSERVABLE_MISSING
                   ? 0.65
                   : (state.gate_state == ObsGateState::OBSERVABLE_WEAK ? 0.40 : 0.10));
    const double support_conf =
        Clamp01(static_cast<double>(obs.support_count) / 6.0);
    evidence.confidence = Clamp01(0.40 * ref_conf + 0.35 * obs_conf + 0.25 * support_conf);

    const double displacement_risk =
        Clamp01(0.60 * evidence.displacement_score + 0.40 * evidence.graph_score);
    const double disappear_risk = evidence.disappearance_score;
    const double base_risk = std::max(displacement_risk, disappear_risk);
    evidence.risk_score = Clamp01(evidence.confidence * base_risk);

    evidence.active =
        evidence.observable &&
        evidence.confidence >= risk_params_.min_confidence &&
        evidence.risk_score >= risk_params_.min_risk_score &&
        (state.significant || state.persistent_candidate || state.disappearance_candidate ||
         state.graph_candidate || evidence.displacement_score > 0.25 ||
         evidence.disappearance_score > 0.25);

    evidences.push_back(evidence);
  }
  return evidences;
}

}  // namespace deform_monitor_v2
