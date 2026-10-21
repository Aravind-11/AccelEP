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
  state = x;
  return state;
}

}  // namespace

struct TopSlot {
  int64_t id;
  float logit;
};

void SimSampleToken(const float* logits, int32_t* chosen_ids, int64_t chosen_index, int64_t vocab,
                    float temperature, uint32_t* rng_state, int top_k) {
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

  // Compare the caller's top_k with the vocabulary before the cap. top_k >= vocab
  // is the full draw even when the vocabulary is larger than 64.
  const bool full = top_k <= 0 || static_cast<int64_t>(top_k) >= vocab;
  int capped = top_k;
  if (capped > 64) {
    capped = 64;
  }
  TopSlot table[64];
  int filled = 0;
  if (!full) {
    for (int64_t i = 0; i < vocab; ++i) {
      const float value = logits[i];
      if (filled < capped) {
        table[filled++] = TopSlot{i, value};
        continue;
      }
      int worst = 0;
      for (int slot = 1; slot < filled; ++slot) {
        const bool lower = table[slot].logit < table[worst].logit;
        const bool later_tie = table[slot].logit == table[worst].logit && table[slot].id > table[worst].id;
        if (lower || later_tie) {
          worst = slot;
        }
      }
      if (value > table[worst].logit || (value == table[worst].logit && i < table[worst].id)) {
        table[worst] = TopSlot{i, value};
      }
    }
    for (int i = 1; i < filled; ++i) {
      const TopSlot key = table[i];
      int j = i;
      while (j > 0 && table[j - 1].id > key.id) {
        table[j] = table[j - 1];
        --j;
      }
      table[j] = key;
    }
  }

  float max_v = full ? logits[0] : table[0].logit;
  if (full) {
    for (int64_t i = 1; i < vocab; ++i) {
      if (logits[i] > max_v) {
        max_v = logits[i];
      }
    }
  } else {
    for (int slot = 1; slot < filled; ++slot) {
      if (table[slot].logit > max_v) {
        max_v = table[slot].logit;
      }
    }
  }
  const double inv_t = 1.0 / static_cast<double>(temperature);
  const int64_t span = full ? vocab : filled;
  double sum = 0.0;
  for (int64_t s = 0; s < span; ++s) {
    const int64_t id = full ? s : table[s].id;
    sum += std::exp((static_cast<double>(logits[id]) - static_cast<double>(max_v)) * inv_t);
  }
  if (!(sum > 0.0)) {
    chosen_ids[chosen_index] = 0;
    return;
  }
  const double draw = (static_cast<double>(NextRand(*rng_state)) + 0.5) / 4294967296.0 * sum;
  double acc = 0.0;
  for (int64_t s = 0; s < span; ++s) {
    const int64_t id = full ? s : table[s].id;
    acc += std::exp((static_cast<double>(logits[id]) - static_cast<double>(max_v)) * inv_t);
    if (acc >= draw) {
      chosen_ids[chosen_index] = static_cast<int32_t>(id);
      return;
    }
  }
  chosen_ids[chosen_index] = static_cast<int32_t>(full ? vocab - 1 : table[filled - 1].id);
}

}  // namespace accel
