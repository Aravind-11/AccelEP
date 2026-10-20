#include "accel/sim_attend.h"

#include <cmath>
#include <cstdint>
#include <limits>

namespace accel {

void SimAttend(const float* q, const float* k, const float* v, float* scores, float* out, int64_t seq,
               int64_t position, int64_t heads, int64_t kv_heads, int64_t head_dim) {
  const float scale = 1.f / std::sqrt(static_cast<float>(head_dim));
  const int64_t repeats = heads / kv_heads;
  for (int64_t s = 0; s < seq; ++s) {
    const int64_t limit = position + s;
    for (int64_t h = 0; h < heads; ++h) {
      const int64_t kv = h / repeats;
      const float* qv = q + (s * heads + h) * head_dim;
      float max_score = -std::numeric_limits<float>::infinity();
      for (int64_t t = 0; t <= limit; ++t) {
        const float* kvv = k + (t * kv_heads + kv) * head_dim;
        float dot = 0.f;
        for (int64_t d = 0; d < head_dim; ++d) {
          dot += qv[d] * kvv[d];
        }
        scores[t] = dot * scale;
        if (scores[t] > max_score) {
          max_score = scores[t];
        }
      }
      float sum = 0.f;
      for (int64_t t = 0; t <= limit; ++t) {
        scores[t] = std::exp(scores[t] - max_score);
        sum += scores[t];
      }
      float* dest = out + (s * heads + h) * head_dim;
      for (int64_t d = 0; d < head_dim; ++d) {
        float acc = 0.f;
        for (int64_t t = 0; t <= limit; ++t) {
          const float* vv = v + (t * kv_heads + kv) * head_dim;
          acc += (scores[t] / sum) * vv[d];
        }
        dest[d] = acc;
      }
    }
  }
}

}  // namespace accel
