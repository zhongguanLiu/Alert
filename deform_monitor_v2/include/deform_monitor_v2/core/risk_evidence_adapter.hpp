#ifndef DEFORM_MONITOR_V2_RISK_EVIDENCE_ADAPTER_HPP
#define DEFORM_MONITOR_V2_RISK_EVIDENCE_ADAPTER_HPP

#include "deform_monitor_v2/data_types.hpp"

namespace deform_monitor_v2 {

class RiskEvidenceAdapter {
public:
  void SetParams(const RiskVisualizationParams& risk_params,
                 const SignificanceParams& significance_params,
                 const GraphTemporalParams& graph_params);
  void SetParams(const RiskVisualizationParams& risk_params,
                 const SignificanceParams& significance_params,
                 const GraphTemporalParams& graph_params,
                 const WeakPlaneMotionParams& weak_plane_params);

  RiskEvidenceVector Build(const AnchorReferenceVector& anchors,
                           const AnchorStateVector& states,
                           const CurrentObservationVector& observations,
                           const MotionClusterVector& clusters) const;

private:
  RiskVisualizationParams risk_params_;
  SignificanceParams significance_params_;
  GraphTemporalParams graph_params_;
  WeakPlaneMotionParams weak_plane_params_;
};

}  // namespace deform_monitor_v2

#endif  // DEFORM_MONITOR_V2_RISK_EVIDENCE_ADAPTER_HPP
