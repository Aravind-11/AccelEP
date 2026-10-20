#include "accel/sim_embed.h"

#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace accel {

void SimEmbed(const int32_t* ids, int64_t id_offset, const float* table, float* out, int64_t seq,
              int64_t hidden, int64_t vocab) {
  for (int64_t s = 0; s < seq; ++s) {
    const int32_t id = ids[id_offset + s];
    if (id < 0 || id >= vocab) {
      throw std::invalid_argument("token id is outside the vocabulary");
    }
    std::memcpy(out + static_cast<size_t>(s) * static_cast<size_t>(hidden),
                table + static_cast<size_t>(id) * static_cast<size_t>(hidden),
                static_cast<size_t>(hidden) * sizeof(float));
  }
}

}  // namespace accel
