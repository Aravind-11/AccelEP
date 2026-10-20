#include "accel/sim_rope.h"

namespace accel {

void SimRope(float* states, const float* cos_table, const float* sin_table, int64_t seq,
             int64_t n_heads, int64_t head_dim, int64_t position) {
  const int64_t half = head_dim / 2;
  for (int64_t s = 0; s < seq; ++s) {
    const float* c = cos_table + (position + s) * head_dim;
    const float* sn = sin_table + (position + s) * head_dim;
    for (int64_t h = 0; h < n_heads; ++h) {
      float* row = states + (s * n_heads + h) * head_dim;
      for (int64_t i = 0; i < half; ++i) {
        const float first = row[i];
        const float second = row[half + i];
        row[i] = first * c[i] - second * sn[i];
        row[half + i] = second * c[half + i] + first * sn[half + i];
      }
    }
  }
}

}  // namespace accel
