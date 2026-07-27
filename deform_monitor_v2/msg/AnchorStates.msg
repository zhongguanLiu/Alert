std_msgs/Header header
uint32 reference_epoch
time reference_initialized_at
# anchors contains only comparable==true anchors this cycle (kept small and
# bounded); total_anchor_count and object_summaries account for ALL anchors,
# including the ones excluded from this array. See AnchorObjectSummary.msg.
int32 total_anchor_count
deform_monitor_v2/AnchorObjectSummary[] object_summaries
deform_monitor_v2/AnchorState[] anchors
