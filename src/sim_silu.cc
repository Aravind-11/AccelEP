#include "accel/sim_silu.h"

#include <cmath>
#include <cstdint>

namespace accel {

void SimSiluMul(float* gate, const float* up, int64_t count) {
  for (int64_t i = 0; i < count; ++i) {
    const float x = gate[i];
    gate[i] = (x / (1.f + std::exp(-x))) * up[i];
  }
}

}  // namespace accel
