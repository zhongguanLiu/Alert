#include "deform_monitor_v2/core/object_id_association.hpp"

#include <algorithm>
#include <cmath>
#include <unordered_map>

namespace deform_monitor_v2 {

bool DecodeObjectIdSample(float sample,
                          const ObjectAssociationParams& params,
                          uint16_t* object_id) {
  if (!object_id || !params.enable || !std::isfinite(sample)) {
    return false;
  }
  const double rounded = std::round(static_cast<double>(sample));
  if (std::abs(static_cast<double>(sample) - rounded) >
      std::max(0.0, params.quantization_tolerance)) {
    return false;
  }
  if (rounded < 0.0 || rounded > static_cast<double>(params.max_id)) {
    return false;
  }
  const uint16_t decoded = static_cast<uint16_t>(rounded);
  if (decoded == params.invalid_id) {
    return false;
  }
  *object_id = decoded;
  return true;
}

ObjectIdAssociationResult AssociateObjectIdSamples(
    const std::vector<float>& samples,
    const ObjectAssociationParams& params) {
  ObjectIdAssociationResult result;
  result.object_id = params.invalid_id;
  if (!params.enable) {
    result.status = ObjectIdAssociationStatus::DISABLED;
    return result;
  }

  std::unordered_map<uint16_t, int> votes;
  for (const float sample : samples) {
    uint16_t object_id = params.invalid_id;
    if (!DecodeObjectIdSample(sample, params, &object_id)) {
      continue;
    }
    ++votes[object_id];
    ++result.support_count;
  }

  result.distinct_id_count = static_cast<int>(votes.size());
  if (result.support_count <= 0 || votes.empty()) {
    result.status = ObjectIdAssociationStatus::INSUFFICIENT;
    return result;
  }

  uint16_t dominant_id = params.invalid_id;
  int dominant_count = 0;
  for (const auto& vote : votes) {
    if (vote.second > dominant_count ||
        (vote.second == dominant_count && vote.first < dominant_id)) {
      dominant_id = vote.first;
      dominant_count = vote.second;
    }
  }
  result.confidence = static_cast<double>(dominant_count) /
                      static_cast<double>(result.support_count);

  if (result.support_count < std::max(1, params.min_support_points)) {
    result.status = ObjectIdAssociationStatus::INSUFFICIENT;
    return result;
  }

  const double min_purity = std::max(0.0, std::min(1.0, params.min_purity));
  if (result.confidence < min_purity) {
    result.status = ObjectIdAssociationStatus::MIXED;
    return result;
  }

  result.object_id = dominant_id;
  result.valid = true;
  result.status = ObjectIdAssociationStatus::VALID;
  return result;
}

}  // namespace deform_monitor_v2
