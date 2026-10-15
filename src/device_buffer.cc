#include "accel/device.h"

namespace accel {

void OwnedBuffer::Reset() noexcept {
  if (device != nullptr && buffer.data != nullptr) {
    device->Free(buffer);
  }
  device = nullptr;
  buffer = {};
  shape.clear();
}

}  // namespace accel
