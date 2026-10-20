#pragma once

#include <cstdint>

namespace accel {

// Hugging Face rotate_half: each half pairs with the other half. Not interleaved
// even/odd RoPE. cos and sin rows are head_dim wide, indexed by absolute position.
void SimRope(float* states, const float* cos_table, const float* sin_table, int64_t seq,
             int64_t n_heads, int64_t head_dim, int64_t position);

}  // namespace accel
