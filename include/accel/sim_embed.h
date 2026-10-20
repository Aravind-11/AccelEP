#pragma once

#include <cstdint>

namespace accel {

void SimEmbed(const int32_t* ids, int64_t id_offset, const float* table, float* out, int64_t seq,
              int64_t hidden, int64_t vocab);

}  // namespace accel
