#include "accel/sim_embed.h"
#include "accel/sim_rmsnorm.h"
#include "accel/sim_rope.h"
#include "accel/sim_cache.h"
#include "accel/sim_attend.h"
#include "accel/sim_silu.h"
#include "accel/sim_add.h"
#include "accel/sim_sample.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>

namespace {

int Fail(const char* name) {
  std::fprintf(stderr, "%s\n", name);
  return 1;
}

bool Near(float got, double want) {
  return std::fabs(static_cast<double>(got) - want) <= 1e-6;
}

// Reference is the double mean of squares, then the weight multiply.
bool RmsMatches(const float* y, const float* row, const float* weight, int64_t dim, float eps) {
  double square = 0.0;
  for (int64_t i = 0; i < dim; ++i) {
    const double value = static_cast<double>(row[i]);
    square += value * value;
  }
  const double inv = 1.0 / std::sqrt(square / static_cast<double>(dim) + static_cast<double>(eps));
  for (int64_t i = 0; i < dim; ++i) {
    const double want = static_cast<double>(row[i]) * inv * static_cast<double>(weight[i]);
    if (!Near(y[i], want)) {
      return false;
    }
  }
  return true;
}

}  // namespace

int main() {
  try {
    const float table[] = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f};
    const int32_t ids[] = {2, 0};
    float embedded[4] = {};
    accel::SimEmbed(ids, 0, table, embedded, 2, 2, 4);
    const float want_embed[] = {5.f, 6.f, 1.f, 2.f};
    for (int i = 0; i < 4; ++i) {
      if (embedded[i] != want_embed[i]) {
        return Fail("Embed");
      }
    }
    bool threw = false;
    try {
      const int32_t bad[] = {9};
      float scratch[2] = {};
      accel::SimEmbed(bad, 0, table, scratch, 1, 2, 4);
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    if (!threw) {
      return Fail("Embed");
    }
  } catch (...) {
    return Fail("Embed");
  }

  try {
    const float weight[] = {1.f, 2.f, 0.5f, 1.5f};
    const float eps = 1e-5f;
    const float x[] = {
        1.f, -1.f, 2.f, 0.f, 4.f, -2.f, 0.5f, 3.f,
    };
    float y[4] = {};
    accel::SimRmsNorm(x, weight, y, 1, 4, eps, 0);
    if (!RmsMatches(y, x, weight, 4, eps)) {
      return Fail("RMSNorm");
    }
    float y_offset[8];
    for (int i = 0; i < 8; ++i) {
      y_offset[i] = 42.f;
    }
    accel::SimRmsNorm(x, weight, y_offset, 1, 4, eps, 1);
    if (!RmsMatches(y_offset, x + 4, weight, 4, eps)) {
      return Fail("RMSNorm");
    }
    for (int i = 4; i < 8; ++i) {
      if (y_offset[i] != 42.f) {
        return Fail("RMSNorm");
      }
    }
  } catch (...) {
    return Fail("RMSNorm");
  }

  try {
    float kept[] = {8.f, -1.f, 0.5f, 4.f};
    const float kept_before[] = {8.f, -1.f, 0.5f, 4.f};
    const float cos_keep[] = {1.f, 1.f, 1.f, 1.f};
    const float sin_keep[] = {0.f, 0.f, 0.f, 0.f};
    accel::SimRope(kept, cos_keep, sin_keep, 1, 1, 4, 0);
    for (int i = 0; i < 4; ++i) {
      if (kept[i] != kept_before[i]) {
        return Fail("RoPE");
      }
    }

    const float in[] = {1.f, 2.f, 3.f, 4.f};
    float turned[] = {1.f, 2.f, 3.f, 4.f};
    // Row 1 is the absolute position under test. Row 0 must not be read.
    const float cos_table[] = {
        1.f, 1.f, 1.f, 1.f, 0.f, 0.f, 0.f, 0.f,
    };
    const float sin_table[] = {
        0.f, 0.f, 0.f, 0.f, 1.f, 1.f, 1.f, 1.f,
    };
    accel::SimRope(turned, cos_table, sin_table, 1, 1, 4, 1);
    const int64_t half = 2;
    for (int64_t i = 0; i < half; ++i) {
      if (turned[i] != -in[half + i] || turned[half + i] != in[i]) {
        return Fail("RoPE");
      }
    }
  } catch (...) {
    return Fail("RoPE");
  }

  try {
    const float src[] = {9.f, 8.f, 7.f, 6.f};
    float cache[8];
    for (int i = 0; i < 8; ++i) {
      cache[i] = -5.f;
    }
    accel::SimAppendCache(src, cache, 2, 1, 1, 2);
    for (int i = 0; i < 4; ++i) {
      if (cache[2 + i] != src[i]) {
        return Fail("Cache");
      }
    }
    if (cache[0] != -5.f || cache[1] != -5.f || cache[6] != -5.f || cache[7] != -5.f) {
      return Fail("Cache");
    }
  } catch (...) {
    return Fail("Cache");
  }

  try {
    const float q = 1.f;
    const float k = 1.f;
    const float v = 3.f;
    float scores = 0.f;
    float out = 0.f;
    accel::SimAttend(&q, &k, &v, &scores, &out, 1, 0, 1, 1, 1);
    if (out != 3.f) {
      return Fail("Attention");
    }
  } catch (...) {
    return Fail("Attention");
  }

  try {
    float gate[] = {0.f};
    const float up[] = {2.f};
    accel::SimSiluMul(gate, up, 1);
    if (gate[0] != 0.f) {
      return Fail("SiLU");
    }
  } catch (...) {
    return Fail("SiLU");
  }

  try {
    float dest[] = {1.f, 2.f};
    const float extra[] = {3.f, 4.f};
    accel::SimAdd(dest, extra, 2);
    if (dest[0] != 4.f || dest[1] != 6.f) {
      return Fail("Add");
    }
  } catch (...) {
    return Fail("Add");
  }

  try {
    const float logits[] = {1.f, 3.f, 3.f};
    int32_t chosen[] = {-1};
    uint32_t rng = 1u;
    accel::SimSampleToken(logits, chosen, 0, 3, 0.f, &rng);
    if (chosen[0] != 1) {
      return Fail("Sample");
    }
    chosen[0] = -1;
    accel::SimSampleToken(logits, chosen, 0, 3, -1.f, &rng);
    if (chosen[0] != 1) {
      return Fail("Sample");
    }
  } catch (...) {
    return Fail("Sample");
  }

  return 0;
}
