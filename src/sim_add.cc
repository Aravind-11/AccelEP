#include "accel/sim_add.h"

namespace accel {

void SimAdd(float* dest, const float* extra, int64_t count) {
  for (int64_t i = 0; i < count; ++i) {
    dest[i] += extra[i];
  }
}

}  // namespace accel
