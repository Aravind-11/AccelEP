#pragma once

#include <cstdint>
#include <cstring>

namespace accel {

void SimAppendCache(const float* src, float* cache, int64_t seq, int64_t position, int64_t n_heads,
                    int64_t head_dim);

}  // namespace accel
