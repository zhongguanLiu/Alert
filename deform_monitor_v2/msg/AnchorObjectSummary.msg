# Lightweight per-object anchor accounting for one processing cycle.
# Populated over ALL anchors considered that cycle (not just the ones
# included in AnchorStates.anchors, which is filtered to comparable==true
# to keep message size bounded). object_id==0 aggregates anchors with no
# valid object association.
uint16 object_id
int32 total_count
int32 comparable_count
int32 significant_count
int32 plane_count
int32 edge_count
int32 band_count
# Exclusion-reason tally for anchors NOT included in AnchorStates.anchors
# (comparable == false), collapsed from ObsGateState:
#   excluded_not_observable   -- gate_state == OBS_NOT_OBSERVABLE
#   excluded_weak_or_missing  -- gate_state in {OBSERVABLE_WEAK, OBSERVABLE_MISSING, OBSERVABLE_REPLACED}
#                                but still failed the comparable gate
int32 excluded_not_observable
int32 excluded_weak_or_missing
