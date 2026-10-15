#include "accel/runtime.h"

#include <atomic>
#include <limits>
#include <stdexcept>
#include <utility>

namespace accel {
namespace {

struct Stats {
  std::atomic<int64_t> matmul_ops{0};
  std::atomic<int64_t> gemm_ops{0};
  std::atomic<int64_t> fused_matmul_ops{0};
  std::atomic<int64_t> kernel_matmuls{0};
  std::atomic<int64_t> bytes_to_device{0};
  std::atomic<int64_t> bytes_to_host{0};
};

Stats& GlobalStats() {
  static Stats stats;
  return stats;
}

bool BroadcastShapes(const std::vector<int64_t>& a, const std::vector<int64_t>& b,
                     std::vector<int64_t>* out) {
  const size_t n = a.size() > b.size() ? a.size() : b.size();
  out->assign(n, 1);
  for (size_t i = 0; i < n; ++i) {
    const bool a_has = i >= n - a.size();
    const bool b_has = i >= n - b.size();
    const int64_t da = a_has ? a[i - (n - a.size())] : 1;
    const int64_t db = b_has ? b[i - (n - b.size())] : 1;
    if (da < 1 || db < 1) {
      // A zero-length batch is representable. Negative dims are not.
      if (da < 0 || db < 0) {
        return false;
      }
    }
    if (da != db && da != 1 && db != 1) {
      return false;
    }
    (*out)[i] = da == 1 ? db : da;
  }
  return true;
}

bool BiasBroadcastable(const std::vector<int64_t>& shape, int64_t m, int64_t n) {
  if (shape.size() > 2) {
    return false;
  }
  const int64_t target[2] = {m, n};
  const int rank = static_cast<int>(shape.size());
  for (int d = 0; d < rank; ++d) {
    const int64_t dim = shape[static_cast<size_t>(d)];
    const int64_t want = target[2 - rank + d];
    if (dim < 0 || (dim != 1 && dim != want)) {
      return false;
    }
  }
  return true;
}

int64_t BatchOffset(const std::vector<int64_t>& tensor_batch, const std::vector<int64_t>& out_index) {
  if (tensor_batch.empty()) {
    return 0;
  }
  std::vector<int64_t> strides(tensor_batch.size(), 1);
  for (size_t i = tensor_batch.size(); i-- > 1;) {
    strides[i - 1] = strides[i] * tensor_batch[i];
  }
  const size_t out_rank = out_index.size();
  const size_t tensor_rank = tensor_batch.size();
  int64_t offset = 0;
  for (size_t i = 0; i < tensor_rank; ++i) {
    const size_t out_axis = out_rank - tensor_rank + i;
    int64_t coord = out_index[out_axis];
    if (tensor_batch[i] == 1) {
      coord = 0;
    }
    offset += coord * strides[i];
  }
  return offset;
}

void NextIndex(std::vector<int64_t>* index, const std::vector<int64_t>& shape) {
  for (size_t axis = index->size(); axis-- > 0;) {
    ++(*index)[axis];
    if ((*index)[axis] < shape[axis]) {
      return;
    }
    (*index)[axis] = 0;
  }
}

DeviceBuffer Resolve(Device& device, const Operand& op, OwnedBuffer* uploaded) {
  if (op.on_device != nullptr) {
    if (op.host != nullptr) {
      throw std::invalid_argument("an operand needs exactly one of host or device memory");
    }
    if (op.on_device->shape != op.shape) {
      throw std::invalid_argument("device operand shape does not match");
    }
    return op.on_device->buffer;
  }
  if (op.host == nullptr && ElementCount(op.shape) != 0) {
    throw std::invalid_argument("missing host data");
  }
  *uploaded = Upload(device, op.host, op.shape);
  return uploaded->buffer;
}

DeviceBuffer Slice(DeviceBuffer base, int64_t elem_offset, int64_t elems) {
  auto* origin = static_cast<float*>(base.data);
  // A zero-element tensor still has a non-null allocation, so the pointer stays valid.
  return DeviceBuffer{origin + elem_offset, static_cast<size_t>(elems) * sizeof(float)};
}

void GemmDims(const std::vector<int64_t>& a, bool trans_a, const std::vector<int64_t>& b, bool trans_b,
              int64_t* m, int64_t* k, int64_t* n) {
  *m = trans_a ? a[1] : a[0];
  *k = trans_a ? a[0] : a[1];
  *n = trans_b ? b[0] : b[1];
}

}  // namespace

void ResetAcceleratorStats() {
  Stats& stats = GlobalStats();
  stats.matmul_ops.store(0);
  stats.gemm_ops.store(0);
  stats.fused_matmul_ops.store(0);
  stats.kernel_matmuls.store(0);
  stats.bytes_to_device.store(0);
  stats.bytes_to_host.store(0);
}

AcceleratorStats GetAcceleratorStats() {
  const Stats& stats = GlobalStats();
  AcceleratorStats out;
  out.matmul_ops = stats.matmul_ops.load();
  out.gemm_ops = stats.gemm_ops.load();
  out.fused_matmul_ops = stats.fused_matmul_ops.load();
  out.kernel_matmuls = stats.kernel_matmuls.load();
  out.bytes_to_device = stats.bytes_to_device.load();
  out.bytes_to_host = stats.bytes_to_host.load();
  return out;
}

void NoteMatmulOp() { GlobalStats().matmul_ops.fetch_add(1); }
void NoteGemmOp() { GlobalStats().gemm_ops.fetch_add(1); }
void NoteFusedMatmulOp() { GlobalStats().fused_matmul_ops.fetch_add(1); }
void NoteKernelMatmul() { GlobalStats().kernel_matmuls.fetch_add(1); }
void NoteBytesToDevice(size_t bytes) {
  GlobalStats().bytes_to_device.fetch_add(static_cast<int64_t>(bytes));
}
void NoteBytesToHost(size_t bytes) {
  GlobalStats().bytes_to_host.fetch_add(static_cast<int64_t>(bytes));
}

int64_t ElementCount(const std::vector<int64_t>& shape) {
  int64_t count = 1;
  for (int64_t dim : shape) {
    if (dim < 0) {
      throw std::invalid_argument("dynamic dimension");
    }
    if (dim != 0 && count > std::numeric_limits<int64_t>::max() / dim) {
      throw std::overflow_error("tensor is too large");
    }
    count *= dim;
  }
  return count;
}

bool InferMatMulShape(const std::vector<int64_t>& a, const std::vector<int64_t>& b,
                      std::vector<int64_t>* out) {
  if (out == nullptr || a.size() < 2 || b.size() < 2) {
    return false;
  }
  for (int64_t dim : a) {
    if (dim < 0) return false;
  }
  for (int64_t dim : b) {
    if (dim < 0) return false;
  }
  if (a.back() != b[b.size() - 2]) {
    return false;
  }
  const std::vector<int64_t> a_batch(a.begin(), a.end() - 2);
  const std::vector<int64_t> b_batch(b.begin(), b.end() - 2);
  std::vector<int64_t> batch;
  if (!BroadcastShapes(a_batch, b_batch, &batch)) {
    return false;
  }
  batch.push_back(a[a.size() - 2]);
  batch.push_back(b.back());
  *out = std::move(batch);
  return true;
}

bool InferGemmShape(const std::vector<int64_t>& a, bool trans_a, const std::vector<int64_t>& b, bool trans_b,
                    const std::vector<int64_t>* bias, std::vector<int64_t>* out) {
  if (out == nullptr || a.size() != 2 || b.size() != 2) {
    return false;
  }
  for (int64_t dim : a) {
    if (dim < 0) return false;
  }
  for (int64_t dim : b) {
    if (dim < 0) return false;
  }
  const int64_t k_a = trans_a ? a[0] : a[1];
  const int64_t k_b = trans_b ? b[1] : b[0];
  if (k_a != k_b) {
    return false;
  }
  const int64_t m = trans_a ? a[1] : a[0];
  const int64_t n = trans_b ? b[0] : b[1];
  if (bias != nullptr && !BiasBroadcastable(*bias, m, n)) {
    return false;
  }
  *out = {m, n};
  return true;
}

bool InferFusedMatMulShape(const std::vector<int64_t>& a, bool trans_a, const std::vector<int64_t>& b, bool trans_b,
                           std::vector<int64_t>* out) {
  if (out == nullptr || a.size() < 2 || b.size() < 2) {
    return false;
  }
  std::vector<int64_t> a_logical = a;
  std::vector<int64_t> b_logical = b;
  if (trans_a) {
    std::swap(a_logical[a_logical.size() - 2], a_logical.back());
  }
  if (trans_b) {
    std::swap(b_logical[b_logical.size() - 2], b_logical.back());
  }
  return InferMatMulShape(a_logical, b_logical, out);
}

OwnedBuffer Upload(Device& device, const float* data, const std::vector<int64_t>& shape) {
  const int64_t count = ElementCount(shape);
  if (count > 0 && data == nullptr) {
    throw std::invalid_argument("missing host data");
  }
  OwnedBuffer owned;
  owned.device = &device;
  owned.shape = shape;
  owned.buffer = device.Allocate(static_cast<size_t>(count) * sizeof(float));
  if (count > 0) {
    device.CopyHostToDevice(owned.buffer, data, static_cast<size_t>(count) * sizeof(float));
  }
  return owned;
}

void ExecuteMatMul(Device& device, const Operand& a, const Operand& b, float* out_host) {
  std::vector<int64_t> out_shape;
  if (!InferMatMulShape(a.shape, b.shape, &out_shape)) {
    throw std::invalid_argument("MatMul shapes are not supported");
  }
  const int64_t m = a.shape[a.shape.size() - 2];
  const int64_t k = a.shape.back();
  const int64_t n = b.shape.back();
  const std::vector<int64_t> a_batch(a.shape.begin(), a.shape.end() - 2);
  const std::vector<int64_t> b_batch(b.shape.begin(), b.shape.end() - 2);
  const std::vector<int64_t> out_batch(out_shape.begin(), out_shape.end() - 2);

  OwnedBuffer a_upload;
  OwnedBuffer b_upload;
  const DeviceBuffer a_buf = Resolve(device, a, &a_upload);
  const DeviceBuffer b_buf = Resolve(device, b, &b_upload);
  const int64_t out_count = ElementCount(out_shape);
  if (out_count > 0 && out_host == nullptr) {
    throw std::invalid_argument("missing MatMul output");
  }
  OwnedBuffer out;
  out.device = &device;
  out.shape = out_shape;
  out.buffer = device.Allocate(static_cast<size_t>(out_count) * sizeof(float));

  const int64_t batches = out_batch.empty() ? 1 : ElementCount(out_batch);
  std::vector<int64_t> index(out_batch.size(), 0);
  for (int64_t batch = 0; batch < batches; ++batch) {
    const int64_t a_off = BatchOffset(a_batch, index) * m * k;
    const int64_t b_off = BatchOffset(b_batch, index) * k * n;
    const int64_t c_off = batch * m * n;
    device.MatMul(Slice(a_buf, a_off, m * k), Slice(b_buf, b_off, k * n), Slice(out.buffer, c_off, m * n), m, k, n);
    if (batch + 1 < batches) {
      NextIndex(&index, out_batch);
    }
  }
  if (out_count > 0) {
    device.CopyDeviceToHost(out_host, out.buffer, static_cast<size_t>(out_count) * sizeof(float));
  }
  NoteMatmulOp();
}

void ExecuteGemm(Device& device, const Operand& a, const Operand& b, const Operand* bias,
                 const GemmConfig& config, float* out_host) {
  std::vector<int64_t> out_shape;
  const std::vector<int64_t>* bias_shape = bias == nullptr ? nullptr : &bias->shape;
  if (!InferGemmShape(a.shape, config.trans_a, b.shape, config.trans_b, bias_shape, &out_shape)) {
    throw std::invalid_argument("Gemm shapes are not supported");
  }
  int64_t m = 0;
  int64_t k = 0;
  int64_t n = 0;
  GemmDims(a.shape, config.trans_a, b.shape, config.trans_b, &m, &k, &n);

  OwnedBuffer a_upload;
  OwnedBuffer b_upload;
  OwnedBuffer bias_upload;
  OwnedBuffer a_transposed;
  OwnedBuffer b_transposed;
  const DeviceBuffer a_src = Resolve(device, a, &a_upload);
  const DeviceBuffer b_src = Resolve(device, b, &b_upload);
  DeviceBuffer a_mat = a_src;
  DeviceBuffer b_mat = b_src;
  if (config.trans_a) {
    a_transposed.device = &device;
    a_transposed.shape = {m, k};
    a_transposed.buffer = device.Allocate(static_cast<size_t>(m * k) * sizeof(float));
    device.Transpose(a_src, a_transposed.buffer, k, m);
    a_mat = a_transposed.buffer;
  }
  if (config.trans_b) {
    b_transposed.device = &device;
    b_transposed.shape = {k, n};
    b_transposed.buffer = device.Allocate(static_cast<size_t>(k * n) * sizeof(float));
    device.Transpose(b_src, b_transposed.buffer, n, k);
    b_mat = b_transposed.buffer;
  }

  const int64_t out_count = m * n;
  if (out_count > 0 && out_host == nullptr) {
    throw std::invalid_argument("missing Gemm output");
  }
  OwnedBuffer out;
  out.device = &device;
  out.shape = out_shape;
  out.buffer = device.Allocate(static_cast<size_t>(out_count) * sizeof(float));
  device.MatMul(a_mat, b_mat, out.buffer, m, k, n);

  const bool scale = config.alpha != 1.f;
  if (bias != nullptr || scale) {
    DeviceBuffer bias_buf{};
    const DeviceBuffer* bias_ptr = nullptr;
    if (bias != nullptr) {
      bias_buf = Resolve(device, *bias, &bias_upload);
      bias_ptr = &bias_buf;
    }
    device.GemmEpilogue(out.buffer, m, n, config.alpha, bias_ptr, bias == nullptr ? std::vector<int64_t>{} : bias->shape,
                        config.beta);
  }
  if (out_count > 0) {
    device.CopyDeviceToHost(out_host, out.buffer, static_cast<size_t>(out_count) * sizeof(float));
  }
  NoteGemmOp();
}

void ExecuteFusedMatMul(Device& device, const Operand& a, const Operand& b, const GemmConfig& config,
                        float* out_host) {
  std::vector<int64_t> out_shape;
  if (!InferFusedMatMulShape(a.shape, config.trans_a, b.shape, config.trans_b, &out_shape)) {
    throw std::invalid_argument("FusedMatMul shapes are not supported");
  }
  const int64_t a_rows = a.shape[a.shape.size() - 2];
  const int64_t a_cols = a.shape.back();
  const int64_t b_rows = b.shape[b.shape.size() - 2];
  const int64_t b_cols = b.shape.back();
  const int64_t m = config.trans_a ? a_cols : a_rows;
  const int64_t k = config.trans_a ? a_rows : a_cols;
  const int64_t n = config.trans_b ? b_rows : b_cols;
  const std::vector<int64_t> a_batch(a.shape.begin(), a.shape.end() - 2);
  const std::vector<int64_t> b_batch(b.shape.begin(), b.shape.end() - 2);
  const std::vector<int64_t> out_batch(out_shape.begin(), out_shape.end() - 2);

  OwnedBuffer a_upload;
  OwnedBuffer b_upload;
  OwnedBuffer a_transposed;
  OwnedBuffer b_transposed;
  const DeviceBuffer a_buf = Resolve(device, a, &a_upload);
  const DeviceBuffer b_buf = Resolve(device, b, &b_upload);
  if (config.trans_a) {
    a_transposed.device = &device;
    a_transposed.shape = {m, k};
    a_transposed.buffer = device.Allocate(static_cast<size_t>(m * k) * sizeof(float));
  }
  if (config.trans_b) {
    b_transposed.device = &device;
    b_transposed.shape = {k, n};
    b_transposed.buffer = device.Allocate(static_cast<size_t>(k * n) * sizeof(float));
  }

  const int64_t out_count = ElementCount(out_shape);
  if (out_count > 0 && out_host == nullptr) {
    throw std::invalid_argument("missing FusedMatMul output");
  }
  OwnedBuffer out;
  out.device = &device;
  out.shape = out_shape;
  out.buffer = device.Allocate(static_cast<size_t>(out_count) * sizeof(float));

  const int64_t batches = out_batch.empty() ? 1 : ElementCount(out_batch);
  std::vector<int64_t> index(out_batch.size(), 0);
  for (int64_t batch = 0; batch < batches; ++batch) {
    const int64_t a_off = BatchOffset(a_batch, index) * a_rows * a_cols;
    const int64_t b_off = BatchOffset(b_batch, index) * b_rows * b_cols;
    const int64_t c_off = batch * m * n;
    DeviceBuffer a_mat = Slice(a_buf, a_off, a_rows * a_cols);
    DeviceBuffer b_mat = Slice(b_buf, b_off, b_rows * b_cols);
    if (config.trans_a) {
      device.Transpose(a_mat, a_transposed.buffer, a_rows, a_cols);
      a_mat = a_transposed.buffer;
    }
    if (config.trans_b) {
      device.Transpose(b_mat, b_transposed.buffer, b_rows, b_cols);
      b_mat = b_transposed.buffer;
    }
    device.MatMul(a_mat, b_mat, Slice(out.buffer, c_off, m * n), m, k, n);
    if (batch + 1 < batches) {
      NextIndex(&index, out_batch);
    }
  }
  if (out_count > 0 && config.alpha != 1.f) {
    device.GemmEpilogue(out.buffer, out_count / n, n, config.alpha, nullptr, {}, 1.f);
  }
  if (out_count > 0) {
    device.CopyDeviceToHost(out_host, out.buffer, static_cast<size_t>(out_count) * sizeof(float));
  }
  NoteFusedMatmulOp();
}

}  // namespace accel
