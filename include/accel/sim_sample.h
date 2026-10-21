#pragma once

#include <cstdint>

namespace accel {

// Writes chosen_ids[chosen_index]. Does not reseed a zero rng_state.
// temperature <= 0 is argmax. top_k <= 0 or top_k >= vocab draws from the whole vocabulary.
// A smaller positive top_k keeps at most 64 logits and walks those slots in increasing token id.
void SimSampleToken(const float* logits, int32_t* chosen_ids, int64_t chosen_index, int64_t vocab,
                    float temperature, uint32_t* rng_state, int top_k);

}  // namespace accel
