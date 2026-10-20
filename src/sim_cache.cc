#include "accel/sim_cache.h"

#include <cstdint>
#include <cstring>

namespace accel {

void SimAppendCache(const float* src, float* cache, int64_t seq, int64_t position, int64_t n_heads,
                    int64_t head_dim) {
  const size_t row = static_cast<size_t>(n_heads) * static_cast<size_t>(head_dim);
  std::memcpy(cache + static_cast<size_t>(position) * row, src,
              static_cast<size_t>(seq) * row * sizeof(float));
}

}  // namespace accel
