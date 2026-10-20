#pragma once

#include <cmath>
#include <cstdint>
#include <limits>

namespace accel {

void SimAttend(const float* q, const float* k, const float* v, float* scores, float* out, int64_t seq,
               int64_t position, int64_t heads, int64_t kv_heads, int64_t head_dim);

}  // namespace accel
