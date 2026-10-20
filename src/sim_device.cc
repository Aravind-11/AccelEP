#include "accel/device.h"
#include "accel/runtime.h"
#include "accel/sim_add.h"
#include "accel/sim_attend.h"
#include "accel/sim_cache.h"
#include "accel/sim_embed.h"
#include "accel/sim_rmsnorm.h"
#include "accel/sim_rope.h"
#include "accel/sim_sample.h"
#include "accel/sim_silu.h"

#include <algorithm>
#include <atomic>
#include <cmath>
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

constexpr int kMr = 8;
constexpr int kNr = 8;
constexpr int kKc = 64;
constexpr int kMc = 64;
constexpr int kNc = 256;

struct LiveInfo {
  size_t bytes = 0;
  uint64_t id = 0;
};

std::atomic<uint64_t> g_next_id{1};
std::atomic<int64_t> g_live_allocs{0};
std::atomic<int64_t> g_packed_calls{0};
std::atomic<int64_t> g_packed_trips{0};

size_t CheckedBytes(int64_t rows, int64_t cols) {
  if (rows < 0 || cols < 0) {
    throw std::invalid_argument("negative matrix dimension");
  }
  if (rows != 0 && static_cast<uint64_t>(cols) > (UINT64_MAX / sizeof(float)) / static_cast<uint64_t>(rows)) {
    throw std::overflow_error("matrix is too large");
  }
  return static_cast<size_t>(rows) * static_cast<size_t>(cols) * sizeof(float);
}

int64_t CheckedSpan(int64_t offset, int64_t count) {
  if (offset < 0 || count < 0) {
    throw std::invalid_argument("negative matrix dimension");
  }
  if (offset > INT64_MAX - count) {
    throw std::overflow_error("matrix is too large");
  }
  return offset + count;
}

size_t CheckedFloats(int64_t depth, int64_t rows, int64_t cols) {
  if (depth < 0 || rows < 0 || cols < 0) {
    throw std::invalid_argument("negative matrix dimension");
  }
  if (depth == 0 || rows == 0 || cols == 0) {
    return 0;
  }
  if (static_cast<uint64_t>(rows) > UINT64_MAX / static_cast<uint64_t>(depth)) {
    throw std::overflow_error("matrix is too large");
  }
  const uint64_t plane = static_cast<uint64_t>(depth) * static_cast<uint64_t>(rows);
  if (static_cast<uint64_t>(cols) > UINT64_MAX / plane) {
    throw std::overflow_error("matrix is too large");
  }
  const uint64_t count = plane * static_cast<uint64_t>(cols);
  if (count > UINT64_MAX / sizeof(float)) {
    throw std::overflow_error("matrix is too large");
  }
  return static_cast<size_t>(count * sizeof(float));
}

size_t CheckedIntBytes(int64_t count) {
  if (count < 0) {
    throw std::invalid_argument("negative matrix dimension");
  }
  if (count != 0 && static_cast<uint64_t>(count) > UINT64_MAX / sizeof(int32_t)) {
    throw std::overflow_error("matrix is too large");
  }
  return static_cast<size_t>(count) * sizeof(int32_t);
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
    strides[static_cast<size_t>(d)] = strides[static_cast<size_t>(d + 1)] * shape[static_cast<size_t>(d + 1)];
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

  // Intentionally empty. A leaked allocation stays in the process-wide count.
  ~SimDevice() override {}

  DeviceBuffer Allocate(size_t bytes) override {
    const size_t store = bytes == 0 ? 1 : bytes;
    void* ptr = nullptr;
    if (posix_memalign(&ptr, 64, store) != 0 || ptr == nullptr) {
      throw std::bad_alloc();
    }
    std::memset(ptr, 0, store);
    const uint64_t id = g_next_id.fetch_add(1, std::memory_order_relaxed);
    {
      std::lock_guard<std::mutex> lock(mu_);
      live_.emplace(ptr, LiveInfo{bytes, id});
    }
    g_live_allocs.fetch_add(1, std::memory_order_relaxed);
    return DeviceBuffer{ptr, bytes, id};
  }

  void Free(DeviceBuffer buffer) override {
    if (buffer.data == nullptr) {
      return;
    }
    std::lock_guard<std::mutex> lock(mu_);
    auto it = live_.find(buffer.data);
    if (it == live_.end() || it->second.bytes != buffer.bytes || it->second.id != buffer.id) {
      throw std::logic_error("free of unknown device buffer");
    }
    live_.erase(it);
    g_live_allocs.fetch_sub(1, std::memory_order_relaxed);
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
    PackedMatMul(A, B, C, m, k, n);
    g_packed_calls.fetch_add(1, std::memory_order_relaxed);
    g_packed_trips.fetch_add(k, std::memory_order_relaxed);
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
    NoteDeviceTranspose();
  }

  void GemmEpilogue(DeviceBuffer c, int64_t m, int64_t n, float alpha, const DeviceBuffer* bias,
                    const std::vector<int64_t>& bias_shape, float beta, Activation activation) override {
    RequireBytes(c, CheckedBytes(m, n), "Gemm epilogue destination is out of range");
    const bool has_bias = bias != nullptr;
    if (has_bias) {
      int64_t bias_elems = 1;
      for (int64_t dim : bias_shape) {
        if (dim < 0) {
          throw std::invalid_argument("negative bias shape");
        }
        bias_elems *= dim;
      }
      RequireBytes(*bias, static_cast<size_t>(bias_elems) * sizeof(float), "Gemm bias is out of range");
    }
    auto* out = static_cast<float*>(c.data);
    const float* bias_data = has_bias ? static_cast<const float*>(bias->data) : nullptr;
    for (int64_t i = 0; i < m; ++i) {
      for (int64_t j = 0; j < n; ++j) {
        const float prod = alpha * out[i * n + j];
        float value = prod;
        if (bias_data != nullptr) {
          const float bprod = beta * bias_data[BiasOffset(bias_shape, i, j)];
          value = value + bprod;
        }
        if (activation == Activation::Relu) {
          if (value < 0.f) {
            value = 0.f;
          }
        } else if (activation == Activation::Gelu) {
          value = 0.5f * value * (1.f + std::erff(value * 0.7071067811865476f));
        }
        out[i * n + j] = value;
      }
    }
  }

  void Embedding(DeviceBuffer ids, int64_t id_offset, DeviceBuffer table, DeviceBuffer out, int64_t seq, int64_t hidden,
                 int64_t vocab) override {
    const int64_t id_count = CheckedSpan(id_offset, seq);
    RequireBytes(ids, CheckedIntBytes(id_count), "Embedding ids are out of range");
    RequireBytes(table, CheckedBytes(vocab, hidden), "Embedding table is out of range");
    RequireBytes(out, CheckedBytes(seq, hidden), "Embedding output is out of range");
    SimEmbed(static_cast<const int32_t*>(ids.data), id_offset, static_cast<const float*>(table.data),
             static_cast<float*>(out.data), seq, hidden, vocab);
  }

  void RmsNorm(DeviceBuffer x, DeviceBuffer weight, DeviceBuffer y, int64_t rows, int64_t dim, float eps,
               int64_t row_offset) override {
    const int64_t x_rows = CheckedSpan(row_offset, rows);
    RequireBytes(x, CheckedBytes(x_rows, dim), "RmsNorm input is out of range");
    RequireBytes(weight, CheckedBytes(dim, 1), "RmsNorm weight is out of range");
    RequireBytes(y, CheckedBytes(rows, dim), "RmsNorm output is out of range");
    SimRmsNorm(static_cast<const float*>(x.data), static_cast<const float*>(weight.data), static_cast<float*>(y.data),
               rows, dim, eps, row_offset);
  }

  void Rope(DeviceBuffer states, DeviceBuffer cos_table, DeviceBuffer sin_table, int64_t seq, int64_t n_heads,
            int64_t head_dim, int64_t position) override {
    const int64_t positions = CheckedSpan(position, seq);
    RequireBytes(states, CheckedFloats(seq, n_heads, head_dim), "Rope states are out of range");
    RequireBytes(cos_table, CheckedBytes(positions, head_dim), "Rope cos table is out of range");
    RequireBytes(sin_table, CheckedBytes(positions, head_dim), "Rope sin table is out of range");
    SimRope(static_cast<float*>(states.data), static_cast<const float*>(cos_table.data),
            static_cast<const float*>(sin_table.data), seq, n_heads, head_dim, position);
  }

  void AppendCache(DeviceBuffer src, DeviceBuffer cache, int64_t seq, int64_t position, int64_t n_heads,
                   int64_t head_dim) override {
    const int64_t rows = CheckedSpan(position, seq);
    RequireBytes(src, CheckedFloats(seq, n_heads, head_dim), "AppendCache source is out of range");
    RequireBytes(cache, CheckedFloats(rows, n_heads, head_dim), "AppendCache cache is out of range");
    SimAppendCache(static_cast<const float*>(src.data), static_cast<float*>(cache.data), seq, position, n_heads,
                   head_dim);
  }

  void Attention(DeviceBuffer q, DeviceBuffer k, DeviceBuffer v, DeviceBuffer scores, DeviceBuffer out, int64_t seq,
                 int64_t position, int64_t heads, int64_t kv_heads, int64_t head_dim) override {
    if (kv_heads <= 0 || (heads > 0 && heads % kv_heads != 0)) {
      throw std::invalid_argument("Attention kv heads are out of range");
    }
    const int64_t seen = CheckedSpan(position, seq);
    RequireBytes(q, CheckedFloats(seq, heads, head_dim), "Attention query is out of range");
    RequireBytes(k, CheckedFloats(seen, kv_heads, head_dim), "Attention key is out of range");
    RequireBytes(v, CheckedFloats(seen, kv_heads, head_dim), "Attention value is out of range");
    RequireBytes(scores, CheckedBytes(seen, 1), "Attention scores are out of range");
    RequireBytes(out, CheckedFloats(seq, heads, head_dim), "Attention output is out of range");
    SimAttend(static_cast<const float*>(q.data), static_cast<const float*>(k.data), static_cast<const float*>(v.data),
              static_cast<float*>(scores.data), static_cast<float*>(out.data), seq, position, heads, kv_heads,
              head_dim);
  }

  void SiluMul(DeviceBuffer gate, DeviceBuffer up, int64_t count) override {
    const size_t bytes = CheckedBytes(count, 1);
    RequireBytes(gate, bytes, "SiluMul gate is out of range");
    RequireBytes(up, bytes, "SiluMul up is out of range");
    SimSiluMul(static_cast<float*>(gate.data), static_cast<const float*>(up.data), count);
  }

  void ResidualAdd(DeviceBuffer dest, DeviceBuffer extra, int64_t count) override {
    const size_t bytes = CheckedBytes(count, 1);
    RequireBytes(dest, bytes, "ResidualAdd destination is out of range");
    RequireBytes(extra, bytes, "ResidualAdd extra is out of range");
    SimAdd(static_cast<float*>(dest.data), static_cast<const float*>(extra.data), count);
  }

  void SampleToken(DeviceBuffer logits, DeviceBuffer chosen_ids, int64_t chosen_index, int64_t vocab,
                   float temperature, uint32_t* rng_state) override {
    if (vocab <= 0 || chosen_index < 0) {
      throw std::invalid_argument("SampleToken is out of range");
    }
    if (rng_state == nullptr) {
      throw std::invalid_argument("SampleToken rng is out of range");
    }
    RequireBytes(logits, CheckedBytes(vocab, 1), "SampleToken logits are out of range");
    RequireBytes(chosen_ids, CheckedIntBytes(CheckedSpan(chosen_index, 1)), "SampleToken ids are out of range");
    SimSampleToken(static_cast<const float*>(logits.data), static_cast<int32_t*>(chosen_ids.data), chosen_index, vocab,
                   temperature, rng_state);
  }

 private:
  static void PackedMatMul(float* A, float* B, float* C, int64_t m, int64_t k, int64_t n) {
    for (int64_t mc = 0; mc < m; mc += kMc) {
      const int64_t m_end = std::min(mc + static_cast<int64_t>(kMc), m);
      for (int64_t nc = 0; nc < n; nc += kNc) {
        const int64_t n_end = std::min(nc + static_cast<int64_t>(kNc), n);
        for (int64_t i0 = mc; i0 < m_end; i0 += kMr) {
          const int64_t mr = std::min(static_cast<int64_t>(kMr), m_end - i0);
          for (int64_t j0 = nc; j0 < n_end; j0 += kNr) {
            const int64_t nr = std::min(static_cast<int64_t>(kNr), n_end - j0);
            float acc[kMr][kNr];
            for (int ii = 0; ii < kMr; ++ii) {
              for (int jj = 0; jj < kNr; ++jj) {
                acc[ii][jj] = 0.f;
              }
            }
            for (int64_t pc = 0; pc < k; pc += kKc) {
              const int64_t kc = std::min(static_cast<int64_t>(kKc), k - pc);
              for (int64_t ii = 0; ii < mr; ++ii) {
                for (int64_t kk = 0; kk < kc; ++kk) {
                  const float aik = A[(i0 + ii) * k + (pc + kk)];
                  const float* b_row = B + (pc + kk) * n + j0;
                  for (int64_t jj = 0; jj < nr; ++jj) {
                    const float prod = aik * b_row[jj];
                    acc[ii][jj] = acc[ii][jj] + prod;
                  }
                }
              }
            }
            for (int64_t ii = 0; ii < mr; ++ii) {
              float* c_row = C + (i0 + ii) * n + j0;
              for (int64_t jj = 0; jj < nr; ++jj) {
                c_row[jj] = acc[ii][jj];
              }
            }
          }
        }
      }
    }
  }

  std::mutex mu_;
  std::unordered_map<void*, LiveInfo> live_;
};

}  // namespace

int64_t LiveDeviceAllocs() { return g_live_allocs.load(std::memory_order_relaxed); }

int64_t PackedMatMulCalls() { return g_packed_calls.load(std::memory_order_relaxed); }
int64_t PackedKTrips() { return g_packed_trips.load(std::memory_order_relaxed); }

void ResetPackedGemmProbes() {
  g_packed_calls.store(0, std::memory_order_relaxed);
  g_packed_trips.store(0, std::memory_order_relaxed);
}

void ReferenceMatMul(const float* a, const float* b, float* c, int64_t m, int64_t k, int64_t n) {
  for (int64_t i = 0; i < m; ++i) {
    for (int64_t j = 0; j < n; ++j) {
      c[i * n + j] = 0.f;
    }
    for (int64_t kk = 0; kk < k; ++kk) {
      const float aik = a[i * k + kk];
      const float* b_row = b + kk * n;
      float* c_row = c + i * n;
      for (int64_t j = 0; j < n; ++j) {
        const float prod = aik * b_row[j];
        c_row[j] = c_row[j] + prod;
      }
    }
  }
}

std::unique_ptr<Device> CreateSimDevice() { return std::make_unique<SimDevice>(); }

}  // namespace accel
