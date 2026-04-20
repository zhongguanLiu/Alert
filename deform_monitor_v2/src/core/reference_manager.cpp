/*
 * Author: zgliu@cumt.edu.cn
 * Affiliation: China University of Mining and Technology
 * Open-source release date: 2026-04-20
 */

#include "deform_monitor_v2/core/reference_manager.hpp"

namespace deform_monitor_v2 {

void ReferenceManager::SetParams(const ReferenceParams& params) {
  params_ = params;
}

void ReferenceManager::UpdateReferenceStatistics(
    AnchorReferenceVector* anchors,
    const CurrentObservationVector& observations,
    AnchorStateVector* states) {
  if (!anchors || !states || anchors->size() != observations.size() || anchors->size() != states->size()) {
    return;
  }

  for (size_t i = 0; i < anchors->size(); ++i) {
    auto& anchor = anchors->at(i);
    const auto& obs = observations[i];
    auto& state = states->at(i);

    if (obs.comparable) {
      ++anchor.visible_count;
    }

    const bool blocked = state.significant || state.cluster_member || state.persistent_candidate ||
                         state.disappearance_candidate;
    const bool stable_segment = obs.comparable &&
                                obs.cmp_score >= params_.tau_cmp_ref &&
                                state.chi2_stat < params_.tau_ref_stable &&
                                state.model0.mu > params_.tau_mu0 &&
                                !blocked;

    if (stable_segment) {
      ++state.stable_streak;
    } else {
      state.stable_streak = 0;
    }

    if (stable_segment && state.stable_streak >= params_.N_ref_stable) {
      ++anchor.matched_count;
    }
  }
}

}  // namespace deform_monitor_v2
