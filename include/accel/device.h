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

  // ids are int32. out[s, :] = table[ids[id_offset + s], :] with table [vocab, hidden].
  virtual void Embedding(DeviceBuffer ids, int64_t id_offset, DeviceBuffer table, DeviceBuffer out, int64_t seq,
                         int64_t hidden, int64_t vocab) = 0;

  // y[row, :] = rms_norm(x[row_offset + row, :]). weight has `dim` floats.
  virtual void RmsNorm(DeviceBuffer x, DeviceBuffer weight, DeviceBuffer y, int64_t rows, int64_t dim, float eps,
                       int64_t row_offset) = 0;

  // In-place rotary embedding. cos and sin rows are head_dim wide at absolute positions.
  virtual void Rope(DeviceBuffer states, DeviceBuffer cos_table, DeviceBuffer sin_table, int64_t seq, int64_t n_heads,
                    int64_t head_dim, int64_t position) = 0;

  // Copy `seq` rows of n_heads * head_dim floats into cache starting at `position`.
  virtual void AppendCache(DeviceBuffer src, DeviceBuffer cache, int64_t seq, int64_t position, int64_t n_heads,
                           int64_t head_dim) = 0;

  // Causal attention. k and v already hold rows [0, position + seq). scores is scratch.
  virtual void Attention(DeviceBuffer q, DeviceBuffer k, DeviceBuffer v, DeviceBuffer scores, DeviceBuffer out,
                         int64_t seq, int64_t position, int64_t heads, int64_t kv_heads, int64_t head_dim) = 0;

  // gate[i] = silu(gate[i]) * up[i].
  virtual void SiluMul(DeviceBuffer gate, DeviceBuffer up, int64_t count) = 0;

  // dest[i] += extra[i].
  virtual void ResidualAdd(DeviceBuffer dest, DeviceBuffer extra, int64_t count) = 0;

  // Writes one int32 to chosen_ids[chosen_index]. temperature <= 0 is argmax
  // and keeps the earlier id on a tie. top_k <= 0 or top_k >= vocab draws from the whole vocabulary.
  // A smaller positive top_k keeps at most 64 logits.
  virtual void SampleToken(DeviceBuffer logits, DeviceBuffer chosen_ids, int64_t chosen_index, int64_t vocab,
                           float temperature, uint32_t* rng_state, int top_k) = 0;
};

std::unique_ptr<Device> CreateSimDevice();

// Forwards every call to a SimDevice except MatMul, which uses a 32x64x32 panel.
// Results match ReferenceMatMul bit for bit. The page and the plugin keep CreateSimDevice.
std::unique_ptr<Device> CreateTiledDevice();

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
