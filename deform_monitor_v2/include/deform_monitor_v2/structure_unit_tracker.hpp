/*
 * Author: zgliu@cumt.edu.cn
 * Affiliation: China University of Mining and Technology
 * Open-source release date: 2026-04-20
 */

#ifndef DEFORM_MONITOR_V2_STRUCTURE_UNIT_TRACKER_HPP
#define DEFORM_MONITOR_V2_STRUCTURE_UNIT_TRACKER_HPP

#include "deform_monitor_v2/data_types.hpp"
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace deform_monitor_v2 {

class StructureUnitTracker {
public:
  StructureUnitTracker(const StructureUnitTrackerParams& params,
                       const StructureUnitVector& units);


  void Update(const pcl::PointCloud<pcl::PointXYZ>& current_cloud,
              const AnchorStateVector& states,
              StructureMigrationVector& migrations);

private:
  StructureUnitTrackerParams params_;
  StructureUnitVector units_;

  double ComputeExitScore(const StructureUnit& unit,
                          const AnchorStateVector& states) const;

  bool IsStructuralFailure(const AnchorTrackState& state) const;

  struct EntryCandidateCloud {
    Eigen::Vector3d centroid     = Eigen::Vector3d::Zero();
    Eigen::Vector3d bbox_min     = Eigen::Vector3d::Zero();
    Eigen::Vector3d bbox_max     = Eigen::Vector3d::Zero();
    Eigen::Vector3d normal       = Eigen::Vector3d::UnitZ();
    int point_count = 0;
  };

  std::vector<EntryCandidateCloud> FindEntryCandidates(
      const pcl::PointCloud<pcl::PointXYZ>& cloud,
      const StructureUnit& unit,
      const AnchorStateVector& states) const;

  bool Validate(const StructureUnit& unit,
                const EntryCandidateCloud& candidate,
                StructureMigration& out_migration) const;
};

}  // namespace deform_monitor_v2

#endif  // DEFORM_MONITOR_V2_STRUCTURE_UNIT_TRACKER_HPP
