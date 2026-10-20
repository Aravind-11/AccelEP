#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace accel {

enum class Activation { None, Relu, Gelu };

inline const char* ActivationName(Activation activation) {
  switch (activation) {
    case Activation::Relu:
      return "relu";
    case Activation::Gelu:
      return "gelu";
    case Activation::None:
      return "none";
  }
  return "none";
}

// Memory that lives on the accelerator. Host code does not read or write
// `data` except by passing it back to the Device that allocated it.
// `id` is 0 for a slice. Allocate mints a process-wide id starting at 1.
struct DeviceBuffer {
  void* data = nullptr;
  size_t bytes = 0;
  uint64_t id = 0;
};

// Owning device allocation. The destructor frees `buffer` on `device`.
struct OwnedBuffer {
  class Device* device = nullptr;
  DeviceBuffer buffer{};
  std::vector<int64_t> shape;

  OwnedBuffer() = default;
  OwnedBuffer(const OwnedBuffer&) = delete;
  OwnedBuffer& operator=(const OwnedBuffer&) = delete;

  OwnedBuffer(OwnedBuffer&& other) noexcept
      : device(other.device), buffer(other.buffer), shape(std::move(other.shape)) {
    other.device = nullptr;
    other.buffer = {};
  }

  OwnedBuffer& operator=(OwnedBuffer&& other) noexcept {
    if (this != &other) {
      Reset();
      device = other.device;
      buffer = other.buffer;
      shape = std::move(other.shape);
      other.device = nullptr;
      other.buffer = {};
    }
    return *this;
  }

  ~OwnedBuffer() { Reset(); }

  void Reset() noexcept;
};

// Accelerator contract. A hardware backend implements this. SimDevice is the
// in-tree stand-in: a separate heap plus the same copy and kernel calls.
class Device {
 public:
  virtual ~Device() = default;

  virtual const char* Name() const = 0;

  virtual DeviceBuffer Allocate(size_t bytes) = 0;
  virtual void Free(DeviceBuffer buffer) = 0;

  virtual void CopyHostToDevice(DeviceBuffer dst, const void* src, size_t bytes) = 0;
  virtual void CopyDeviceToHost(void* dst, DeviceBuffer src, size_t bytes) = 0;

  // Row-major C[m, n] = A[m, k] @ B[k, n]. Overwrites C.
  virtual void MatMul(DeviceBuffer a, DeviceBuffer b, DeviceBuffer c, int64_t m, int64_t k, int64_t n) = 0;

  // src is row-major [rows, cols]. dst is row-major [cols, rows].
  virtual void Transpose(DeviceBuffer src, DeviceBuffer dst, int64_t rows, int64_t cols) = 0;

  // c[i, j] = alpha * c[i, j] + beta * broadcast(bias)[i, j], then an optional
  // activation. `bias` is null when there is no bias. `bias_shape` is
  // right-aligned against [m, n] and must already be unidirectional-broadcastable.
  virtual void GemmEpilogue(DeviceBuffer c, int64_t m, int64_t n, float alpha, const DeviceBuffer* bias,
                            const std::vector<int64_t>& bias_shape, float beta,
                            Activation activation = Activation::None) = 0;
};

std::unique_ptr<Device> CreateSimDevice();

// Process-wide count of device allocations that have not been freed.
// Destroying a SimDevice does not sweep this count.
int64_t LiveDeviceAllocs();

// Naive i-k-j matmul used by tests. Same split multiply-then-add order as
// SimDevice::MatMul. Does not note stats or packed-GEMM probes.
void ReferenceMatMul(const float* a, const float* b, float* c, int64_t m, int64_t k, int64_t n);

// Test seams for the packed kernel. Not part of the plugin ABI.
int64_t PackedMatMulCalls();
int64_t PackedKTrips();
void ResetPackedGemmProbes();

}  // namespace accel
