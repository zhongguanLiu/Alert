/*
 * Author: zgliu@cumt.edu.cn
 * Affiliation: China University of Mining and Technology
 * Open-source release date: 2026-04-20
 */

#ifndef DEFORM_MONITOR_V2_CORE_IMM_INFORMATION_FILTER_HPP
#define DEFORM_MONITOR_V2_CORE_IMM_INFORMATION_FILTER_HPP

#include "deform_monitor_v2/data_types.hpp"

namespace deform_monitor_v2 {

class ImmInformationFilter {
public:
  void SetParams(const ImmParams& imm_params,
                 const ObservabilityParams& observability_params,
                 const SignificanceParams& significance_params,
                 const DirectionalMotionParams& directional_params,
                 double tau_mu0);

  void InitializeAnchorState(AnchorTrackState* state) const;
  void Predict(AnchorTrackState* state, double dt) const;
  void Update(AnchorTrackState* state,
              const AnchorReference& anchor,
              const CurrentObservation& obs) const;
  void UpdateCusum(AnchorTrackState* state) const;
  void UpdateDirectionalMotion(AnchorTrackState* state,
                               const AnchorReference& anchor,
                               double cmp_score,
                               double dt) const;

private:
  void MixModelStates(const AnchorTrackState& state,
                      Eigen::Matrix<double, 6, 1>* x0,
                      Eigen::Matrix<double, 6, 6>* P0,
                      Eigen::Matrix<double, 6, 1>* x1,
                      Eigen::Matrix<double, 6, 6>* P1,
                      double* mu0_pred,
                      double* mu1_pred) const;

  ImmParams imm_params_;
  ObservabilityParams observability_params_;
  SignificanceParams significance_params_;
  DirectionalMotionParams directional_params_;
  double tau_mu0_ = 0.8;
};

}  // namespace deform_monitor_v2

#endif  // DEFORM_MONITOR_V2_CORE_IMM_INFORMATION_FILTER_HPP
