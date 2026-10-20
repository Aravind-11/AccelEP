#pragma once

#include <cstdint>

namespace accel {

// Writes chosen_ids[chosen_index]. Does not reseed a zero rng_state.
void SimSampleToken(const float* logits, int32_t* chosen_ids, int64_t chosen_index, int64_t vocab,
                    float temperature, uint32_t* rng_state);

}  // namespace accel
