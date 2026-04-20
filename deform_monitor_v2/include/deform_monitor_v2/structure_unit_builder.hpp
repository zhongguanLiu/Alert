/*
 * Author: zgliu@cumt.edu.cn
 * Affiliation: China University of Mining and Technology
 * Open-source release date: 2026-04-20
 */

#ifndef DEFORM_MONITOR_V2_STRUCTURE_UNIT_BUILDER_HPP
#define DEFORM_MONITOR_V2_STRUCTURE_UNIT_BUILDER_HPP
#include "deform_monitor_v2/data_types.hpp"
#include <vector>

namespace deform_monitor_v2 {

class StructureUnitBuilder {
public:
  explicit StructureUnitBuilder(const StructureUnitParams& params);


  StructureUnitVector Build(const AnchorReferenceVector& anchors);

private:
  StructureUnitParams params_;


  using VoxelKey = std::tuple<int, int, int>;
  struct VoxelKeyHash {
    size_t operator()(const VoxelKey& k) const;
  };

  bool ShouldConnect(const AnchorReference& a,
                     const AnchorReference& b,
                     RegionEdgeType& out_type) const;
};

}  // namespace deform_monitor_v2

#endif  // DEFORM_MONITOR_V2_STRUCTURE_UNIT_BUILDER_HPP
