#ifndef DEFORM_MONITOR_V2_CORE_OBJECT_ID_ASSOCIATION_HPP
#define DEFORM_MONITOR_V2_CORE_OBJECT_ID_ASSOCIATION_HPP

#include <cstdint>
#include <vector>

#include "deform_monitor_v2/data_types.hpp"

namespace deform_monitor_v2 {

enum class ObjectIdAssociationStatus : uint8_t {
  DISABLED = 0,
  INSUFFICIENT = 1,
  MIXED = 2,
  VALID = 3
};

struct ObjectIdAssociationResult {
  uint16_t object_id = 0;
  bool valid = false;
  double confidence = 0.0;
  int support_count = 0;
  int distinct_id_count = 0;
  ObjectIdAssociationStatus status = ObjectIdAssociationStatus::DISABLED;
};

bool DecodeObjectIdSample(float sample,
                          const ObjectAssociationParams& params,
                          uint16_t* object_id);

ObjectIdAssociationResult AssociateObjectIdSamples(
    const std::vector<float>& samples,
    const ObjectAssociationParams& params);

}  // namespace deform_monitor_v2

#endif  // DEFORM_MONITOR_V2_CORE_OBJECT_ID_ASSOCIATION_HPP
