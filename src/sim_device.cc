#include "accel/device.h"
#include "accel/runtime.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <new>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace accel {
namespace {

size_t CheckedBytes(int64_t rows, int64_t cols) {
  if (rows < 0 || cols < 0) {
    throw std::invalid_argument("negative matrix dimension");
  }
  if (rows != 0 && static_cast<uint64_t>(cols) > (UINT64_MAX / sizeof(float)) / static_cast<uint64_t>(rows)) {
    throw std::overflow_error("matrix is too large");
  }
  return static_cast<size_t>(rows) * static_cast<size_t>(cols) * sizeof(float);
}

void RequireBytes(const DeviceBuffer& buffer, size_t need, const char* what) {
  if (need == 0) {
    return;
  }
  if (buffer.data == nullptr || buffer.bytes < need) {
    throw std::invalid_argument(what);
  }
}

int64_t BiasOffset(const std::vector<int64_t>& shape, int64_t i, int64_t j) {
  const int rank = static_cast<int>(shape.size());
  if (rank == 0) {
    return 0;
  }
  std::vector<int64_t> strides(static_cast<size_t>(rank), 1);
  for (int d = rank - 2; d >= 0; --d) {
    strides[static_cast<size_t>(d)] =
        strides[static_cast<size_t>(d + 1)] * shape[static_cast<size_t>(d + 1)];
  }
  const int64_t coords[2] = {i, j};
  int64_t offset = 0;
  for (int d = 0; d < rank; ++d) {
    const int target_axis = 2 - rank + d;
    int64_t coord = coords[target_axis];
    if (shape[static_cast<size_t>(d)] == 1) {
      coord = 0;
    }
    offset += coord * strides[static_cast<size_t>(d)];
  }
  return offset;
}

class SimDevice final : public Device {
 public:
  const char* Name() const override { return "sim"; }

  DeviceBuffer Allocate(size_t bytes) override {
    const size_t store = bytes == 0 ? 1 : bytes;
    void* ptr = nullptr;
    if (posix_memalign(&ptr, 64, store) != 0 || ptr == nullptr) {
      throw std::bad_alloc();
    }
    std::memset(ptr, 0, store);
    {
      std::lock_guard<std::mutex> lock(mu_);
      live_.emplace(ptr, bytes);
    }
    return DeviceBuffer{ptr, bytes};
  }

  void Free(DeviceBuffer buffer) override {
    if (buffer.data == nullptr) {
      return;
    }
    std::lock_guard<std::mutex> lock(mu_);
    auto it = live_.find(buffer.data);
    if (it == live_.end() || it->second != buffer.bytes) {
      throw std::logic_error("free of unknown device buffer");
    }
    live_.erase(it);
    std::free(buffer.data);
  }

  void CopyHostToDevice(DeviceBuffer dst, const void* src, size_t bytes) override {
    if (bytes == 0) {
      return;
    }
    if (src == nullptr || dst.data == nullptr || bytes > dst.bytes) {
      throw std::invalid_argument("invalid host-to-device copy");
    }
    std::memcpy(dst.data, src, bytes);
    NoteBytesToDevice(bytes);
  }

  void CopyDeviceToHost(void* dst, DeviceBuffer src, size_t bytes) override {
    if (bytes == 0) {
      return;
    }
    if (dst == nullptr || src.data == nullptr || bytes > src.bytes) {
      throw std::invalid_argument("invalid device-to-host copy");
    }
    std::memcpy(dst, src.data, bytes);
    NoteBytesToHost(bytes);
  }

  void MatMul(DeviceBuffer a, DeviceBuffer b, DeviceBuffer c, int64_t m, int64_t k, int64_t n) override {
    RequireBytes(a, CheckedBytes(m, k), "MatMul A is out of range");
    RequireBytes(b, CheckedBytes(k, n), "MatMul B is out of range");
    RequireBytes(c, CheckedBytes(m, n), "MatMul C is out of range");
    auto* A = static_cast<float*>(a.data);
    auto* B = static_cast<float*>(b.data);
    auto* C = static_cast<float*>(c.data);
    for (int64_t i = 0; i < m; ++i) {
      for (int64_t j = 0; j < n; ++j) {
        C[i * n + j] = 0.f;
      }
      for (int64_t kk = 0; kk < k; ++kk) {
        const float aik = A[i * k + kk];
        const float* b_row = B + kk * n;
        float* c_row = C + i * n;
        for (int64_t j = 0; j < n; ++j) {
          c_row[j] += aik * b_row[j];
        }
      }
    }
    NoteKernelMatmul();
  }

  void Transpose(DeviceBuffer src, DeviceBuffer dst, int64_t rows, int64_t cols) override {
    RequireBytes(src, CheckedBytes(rows, cols), "Transpose source is out of range");
    RequireBytes(dst, CheckedBytes(cols, rows), "Transpose destination is out of range");
    const auto* in = static_cast<const float*>(src.data);
    auto* out = static_cast<float*>(dst.data);
    for (int64_t i = 0; i < rows; ++i) {
      for (int64_t j = 0; j < cols; ++j) {
        out[j * rows + i] = in[i * cols + j];
      }
    }
  }

  void GemmEpilogue(DeviceBuffer c, int64_t m, int64_t n, float alpha, const DeviceBuffer* bias,
                    const std::vector<int64_t>& bias_shape, float beta, bool relu) override {
    RequireBytes(c, CheckedBytes(m, n), "Gemm epilogue destination is out of range");
    const bool has_bias = bias != nullptr;
    if (has_bias) {
      int64_t bias_elems = 1;
      for (int64_t dim : bias_shape) {
        bias_elems *= dim;
      }
      if (bias_elems < 0) {
        throw std::invalid_argument("negative bias shape");
      }
      RequireBytes(*bias, static_cast<size_t>(bias_elems) * sizeof(float), "Gemm bias is out of range");
    }
    auto* out = static_cast<float*>(c.data);
    const float* bias_data = has_bias ? static_cast<const float*>(bias->data) : nullptr;
    for (int64_t i = 0; i < m; ++i) {
      for (int64_t j = 0; j < n; ++j) {
        float value = alpha * out[i * n + j];
        if (bias_data != nullptr) {
          value += beta * bias_data[BiasOffset(bias_shape, i, j)];
        }
        if (relu && value < 0.f) {
          value = 0.f;
        }
        out[i * n + j] = value;
      }
    }
  }

 private:
  std::mutex mu_;
  std::unordered_map<void*, size_t> live_;
};

}  // namespace

std::unique_ptr<Device> CreateSimDevice() { return std::make_unique<SimDevice>(); }

}  // namespace accel
