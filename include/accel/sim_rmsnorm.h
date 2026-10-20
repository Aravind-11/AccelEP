#pragma once

#include <cstdint>

namespace accel {

void SimRmsNorm(const float* x, const float* weight, float* y, int64_t rows, int64_t dim, float eps, int64_t row_offset);

}  // namespace accel
