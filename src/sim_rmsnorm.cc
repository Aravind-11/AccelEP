#include "accel/sim_rmsnorm.h"

#include <cmath>

namespace accel {

void SimRmsNorm(const float* x, const float* weight, float* y, int64_t rows, int64_t dim, float eps, int64_t row_offset) {
  for (int64_t row = 0; row < rows; ++row) {
    const float* in = x + (row_offset + row) * dim;
    float* out = y + row * dim;
    double square = 0.0;
    for (int64_t i = 0; i < dim; ++i) {
      square += static_cast<double>(in[i]) * static_cast<double>(in[i]);
    }
    const double inv = 1.0 / std::sqrt(square / static_cast<double>(dim) + static_cast<double>(eps));
    for (int64_t i = 0; i < dim; ++i) {
      out[i] = static_cast<float>(static_cast<double>(in[i]) * inv) * weight[i];
    }
  }
}

}  // namespace accel
