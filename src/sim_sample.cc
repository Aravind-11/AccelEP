#include "accel/sim_sample.h"

#include <cmath>
#include <cstdint>

namespace accel {
namespace {

uint32_t NextRand(uint32_t& state) {
  uint32_t x = state;
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  state = x == 0 ? 0x6d2b79f5u : x;
  return state;
}

}  // namespace

void SimSampleToken(const float* logits, int32_t* chosen_ids, int64_t chosen_index, int64_t vocab,
                    float temperature, uint32_t* rng_state) {
  if (!(temperature > 0.f)) {
    int64_t best = 0;
    float best_v = logits[0];
    for (int64_t i = 1; i < vocab; ++i) {
      if (logits[i] > best_v) {
        best_v = logits[i];
        best = i;
      }
    }
    chosen_ids[chosen_index] = static_cast<int32_t>(best);
    return;
  }

  float max_v = logits[0];
  for (int64_t i = 1; i < vocab; ++i) {
    if (logits[i] > max_v) {
      max_v = logits[i];
    }
  }
  const double inv_t = 1.0 / static_cast<double>(temperature);
  double sum = 0.0;
  for (int64_t i = 0; i < vocab; ++i) {
    sum += std::exp((static_cast<double>(logits[i]) - static_cast<double>(max_v)) * inv_t);
  }
  if (!(sum > 0.0)) {
    chosen_ids[chosen_index] = 0;
    return;
  }
  const double draw = (static_cast<double>(NextRand(*rng_state)) + 0.5) / 4294967296.0 * sum;
  double acc = 0.0;
  for (int64_t i = 0; i < vocab; ++i) {
    acc += std::exp((static_cast<double>(logits[i]) - static_cast<double>(max_v)) * inv_t);
    if (acc >= draw) {
      chosen_ids[chosen_index] = static_cast<int32_t>(i);
      return;
    }
  }
  chosen_ids[chosen_index] = static_cast<int32_t>(vocab - 1);
}

}  // namespace accel
