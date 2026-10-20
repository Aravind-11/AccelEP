#include "accel/runtime.h"

#include <atomic>
#include <cmath>
#include <fstream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace accel {
namespace {

struct Stats {
  std::atomic<int64_t> matmul_ops{0};
  std::atomic<int64_t> gemm_ops{0};
  std::atomic<int64_t> kernel_matmuls{0};
  std::atomic<int64_t> relu_epilogues{0};
  std::atomic<int64_t> gelu_epilogues{0};
  std::atomic<int64_t> add_epilogues{0};
  std::atomic<int64_t> device_transposes{0};
  std::atomic<int64_t> bytes_to_device{0};
  std::atomic<int64_t> bytes_to_host{0};
};

Stats& GlobalStats() {
  static Stats stats;
  return stats;
}

std::mutex& ProfileMu() {
  static std::mutex mu;
  return mu;
}

std::vector<ProfileRecord>& ProfileLog() {
  static std::vector<ProfileRecord> log;
  return log;
}

std::mutex& TouchMu() {
  static std::mutex mu;
  return mu;
}

std::vector<ConstantTouch>& TouchLog() {
  static std::vector<ConstantTouch> log;
  return log;
}

bool BroadcastShapes(const std::vector<int64_t>& a, const std::vector<int64_t>& b, std::vector<int64_t>* out) {
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
  // Slices do not own an allocation id.
  return DeviceBuffer{origin + elem_offset, static_cast<size_t>(elems) * sizeof(float), 0};
}

void GemmDims(const std::vector<int64_t>& a, bool trans_a, const std::vector<int64_t>& b, bool trans_b, int64_t* m,
              int64_t* k, int64_t* n) {
  *m = trans_a ? a[1] : a[0];
  *k = trans_a ? a[0] : a[1];
  *n = trans_b ? b[0] : b[1];
}

bool SwapLastTwo(std::vector<int64_t>* shape) {
  if (shape == nullptr || shape->size() < 2) {
    return false;
  }
  std::swap((*shape)[shape->size() - 2], shape->back());
  return true;
}

void NoteActivation(Activation activation) {
  if (activation == Activation::Relu) {
    NoteReluEpilogue();
  } else if (activation == Activation::Gelu) {
    NoteGeluEpilogue();
  }
}

void Deliver(Device& device, OwnedBuffer* out, int64_t out_count, float* out_host, OwnedBuffer* retain) {
  if (retain != nullptr) {
    retain->Reset();
    *retain = std::move(*out);
    return;
  }
  if (out_count > 0) {
    device.CopyDeviceToHost(out_host, out->buffer, static_cast<size_t>(out_count) * sizeof(float));
  }
}

std::string JsonEscape(const std::string& text) {
  std::string out;
  out.reserve(text.size() + 2);
  for (unsigned char ch : text) {
    switch (ch) {
      case '"':
        out += "\\\"";
        break;
      case '\\':
        out += "\\\\";
        break;
      default:
        out.push_back(static_cast<char>(ch));
        break;
    }
  }
  return out;
}

std::string ShapeJson(const std::vector<int64_t>& shape) {
  std::string out = "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i != 0) {
      out += ',';
    }
    out += std::to_string(shape[i]);
  }
  out += ']';
  return out;
}

std::string ProfileJson() {
  std::lock_guard<std::mutex> lock(ProfileMu());
  const std::vector<ProfileRecord>& log = ProfileLog();
  if (log.empty()) {
    return "{\"records\":[]}";
  }
  std::string out = "{\"records\":[";
  for (size_t r = 0; r < log.size(); ++r) {
    const ProfileRecord& rec = log[r];
    if (r != 0) {
      out += ',';
    }
    out += "{\"op\":\"";
    out += JsonEscape(rec.op);
    out += "\",\"activation\":\"";
    out += JsonEscape(rec.activation);
    out += "\",\"inputs\":[";
    for (size_t i = 0; i < rec.inputs.size(); ++i) {
      if (i != 0) {
        out += ',';
      }
      out += "{\"name\":\"";
      out += JsonEscape(rec.inputs[i].name);
      out += "\",\"shape\":";
      out += ShapeJson(rec.inputs[i].shape);
      out += '}';
    }
    out += "],\"output\":{\"name\":\"";
    out += JsonEscape(rec.output.name);
    out += "\",\"shape\":";
    out += ShapeJson(rec.output.shape);
    out += "},\"bytes_to_device\":";
    out += std::to_string(rec.bytes_to_device);
    out += ",\"bytes_to_host\":";
    out += std::to_string(rec.bytes_to_host);
    out += '}';
  }
  out += "]}";
  return out;
}

void ExecuteFusedImpl(Device& device, const Operand& a, const Operand& b, const FusedMatMulConfig& config,
                      Activation activation, const Operand* residual, float* out_host, OwnedBuffer* retain,
                      bool note_matmul) {
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
  const DeviceBuffer a_buf = Resolve(device, a, &a_upload);
  const DeviceBuffer b_buf = Resolve(device, b, &b_upload);
  const int64_t out_count = ElementCount(out_shape);
  if (retain == nullptr && out_count > 0 && out_host == nullptr) {
    throw std::invalid_argument("missing FusedMatMul output");
  }
  OwnedBuffer out;
  out.device = &device;
  out.shape = out_shape;
  out.buffer = device.Allocate(static_cast<size_t>(out_count) * sizeof(float));

  OwnedBuffer a_transposed;
  OwnedBuffer b_transposed;
  if (config.trans_a) {
    a_transposed.device = &device;
    a_transposed.shape = {m, k};
    a_transposed.buffer = device.Allocate(static_cast<size_t>(ElementCount(a_transposed.shape)) * sizeof(float));
  }
  if (config.trans_b) {
    b_transposed.device = &device;
    b_transposed.shape = {k, n};
    b_transposed.buffer = device.Allocate(static_cast<size_t>(ElementCount(b_transposed.shape)) * sizeof(float));
  }

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
  if (out_count > 0 && (config.alpha != 1.f || activation != Activation::None)) {
    device.GemmEpilogue(out.buffer, out_count, 1, config.alpha, nullptr, {}, 0.f, activation);
  }
  NoteActivation(activation);
  if (residual != nullptr) {
    OwnedBuffer residual_upload;
    if (out_count > 0) {
      const DeviceBuffer residual_buf = Resolve(device, *residual, &residual_upload);
      device.GemmEpilogue(out.buffer, 1, out_count, 1.f, &residual_buf, std::vector<int64_t>{out_count}, 1.f,
                          Activation::None);
    }
    NoteAddEpilogue();
  }
  Deliver(device, &out, out_count, out_host, retain);
  if (note_matmul) {
    NoteMatmulOp();
  }
}

}  // namespace

void ResetAcceleratorStats() {
  Stats& stats = GlobalStats();
  stats.matmul_ops.store(0);
  stats.gemm_ops.store(0);
  stats.kernel_matmuls.store(0);
  stats.relu_epilogues.store(0);
  stats.gelu_epilogues.store(0);
  stats.add_epilogues.store(0);
  stats.device_transposes.store(0);
  stats.bytes_to_device.store(0);
  stats.bytes_to_host.store(0);
  std::lock_guard<std::mutex> lock(ProfileMu());
  ProfileLog().clear();
}

AcceleratorStats GetAcceleratorStats() {
  const Stats& stats = GlobalStats();
  AcceleratorStats out;
  out.matmul_ops = stats.matmul_ops.load();
  out.gemm_ops = stats.gemm_ops.load();
  out.kernel_matmuls = stats.kernel_matmuls.load();
  out.relu_epilogues = stats.relu_epilogues.load();
  out.gelu_epilogues = stats.gelu_epilogues.load();
  out.add_epilogues = stats.add_epilogues.load();
  out.device_transposes = stats.device_transposes.load();
  out.bytes_to_device = stats.bytes_to_device.load();
  out.bytes_to_host = stats.bytes_to_host.load();
  return out;
}

void NoteMatmulOp() { GlobalStats().matmul_ops.fetch_add(1); }
void NoteGemmOp() { GlobalStats().gemm_ops.fetch_add(1); }
void NoteKernelMatmul() { GlobalStats().kernel_matmuls.fetch_add(1); }
void NoteReluEpilogue() { GlobalStats().relu_epilogues.fetch_add(1); }
void NoteGeluEpilogue() { GlobalStats().gelu_epilogues.fetch_add(1); }
void NoteAddEpilogue() { GlobalStats().add_epilogues.fetch_add(1); }
void NoteDeviceTranspose() { GlobalStats().device_transposes.fetch_add(1); }
void NoteBytesToDevice(size_t bytes) { GlobalStats().bytes_to_device.fetch_add(static_cast<int64_t>(bytes)); }
void NoteBytesToHost(size_t bytes) { GlobalStats().bytes_to_host.fetch_add(static_cast<int64_t>(bytes)); }

void AppendProfileRecord(ProfileRecord record) {
  std::lock_guard<std::mutex> lock(ProfileMu());
  ProfileLog().push_back(std::move(record));
}

int WriteProfile(const char* path) {
  if (path == nullptr) {
    return 1;
  }
  const std::string json = ProfileJson();
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) {
    return 1;
  }
  out << json;
  if (!out) {
    return 1;
  }
  return 0;
}

void ClearProfileForTest() {
  std::lock_guard<std::mutex> lock(ProfileMu());
  ProfileLog().clear();
}

void NoteConstantTouch(uint64_t id, const void* address, uint64_t bytes) {
  ConstantTouch touch;
  touch.id = id;
  touch.address = reinterpret_cast<uint64_t>(address);
  touch.bytes = bytes;
  std::lock_guard<std::mutex> lock(TouchMu());
  TouchLog().push_back(touch);
}

void ClearConstantTouches() {
  std::lock_guard<std::mutex> lock(TouchMu());
  TouchLog().clear();
}

int64_t ConstantTouchCount() {
  std::lock_guard<std::mutex> lock(TouchMu());
  return static_cast<int64_t>(TouchLog().size());
}

uint64_t ConstantTouchId(int64_t index) {
  std::lock_guard<std::mutex> lock(TouchMu());
  if (index < 0 || static_cast<size_t>(index) >= TouchLog().size()) {
    return 0;
  }
  return TouchLog()[static_cast<size_t>(index)].id;
}

uint64_t ConstantTouchAddress(int64_t index) {
  std::lock_guard<std::mutex> lock(TouchMu());
  if (index < 0 || static_cast<size_t>(index) >= TouchLog().size()) {
    return 0;
  }
  return TouchLog()[static_cast<size_t>(index)].address;
}

uint64_t ConstantTouchBytes(int64_t index) {
  std::lock_guard<std::mutex> lock(TouchMu());
  if (index < 0 || static_cast<size_t>(index) >= TouchLog().size()) {
    return 0;
  }
  return TouchLog()[static_cast<size_t>(index)].bytes;
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

bool InferMatMulShape(const std::vector<int64_t>& a, const std::vector<int64_t>& b, std::vector<int64_t>* out) {
  if (out == nullptr || a.size() < 2 || b.size() < 2) {
    return false;
  }
  for (int64_t dim : a) {
    if (dim < 0) {
      return false;
    }
  }
  for (int64_t dim : b) {
    if (dim < 0) {
      return false;
    }
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

bool InferFusedMatMulShape(const std::vector<int64_t>& a, bool trans_a, const std::vector<int64_t>& b, bool trans_b,
                           std::vector<int64_t>* out) {
  std::vector<int64_t> a_logical = a;
  std::vector<int64_t> b_logical = b;
  if (trans_a && !SwapLastTwo(&a_logical)) {
    return false;
  }
  if (trans_b && !SwapLastTwo(&b_logical)) {
    return false;
  }
  return InferMatMulShape(a_logical, b_logical, out);
}

bool InferGemmShape(const std::vector<int64_t>& a, bool trans_a, const std::vector<int64_t>& b, bool trans_b,
                    const std::vector<int64_t>* bias, std::vector<int64_t>* out) {
  if (out == nullptr || a.size() != 2 || b.size() != 2) {
    return false;
  }
  for (int64_t dim : a) {
    if (dim < 0) {
      return false;
    }
  }
  for (int64_t dim : b) {
    if (dim < 0) {
      return false;
    }
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

void PretransposeLastTwo(Device& device, const OwnedBuffer& src, OwnedBuffer* dst) {
  if (dst == nullptr || src.shape.size() < 2 || src.buffer.data == nullptr) {
    throw std::invalid_argument("pretranspose needs a matrix");
  }
  std::vector<int64_t> shape = src.shape;
  const int64_t rows = shape[shape.size() - 2];
  const int64_t cols = shape.back();
  std::swap(shape[shape.size() - 2], shape.back());
  const std::vector<int64_t> batch(src.shape.begin(), src.shape.end() - 2);
  const int64_t batches = ElementCount(batch);
  dst->Reset();
  dst->device = &device;
  dst->shape = std::move(shape);
  dst->buffer = device.Allocate(static_cast<size_t>(ElementCount(dst->shape)) * sizeof(float));
  const int64_t tile = rows * cols;
  for (int64_t b = 0; b < batches; ++b) {
    auto* src_ptr = static_cast<float*>(src.buffer.data) + b * tile;
    auto* dst_ptr = static_cast<float*>(dst->buffer.data) + b * tile;
    const DeviceBuffer src_tile{src_ptr, static_cast<size_t>(tile) * sizeof(float), 0};
    const DeviceBuffer dst_tile{dst_ptr, static_cast<size_t>(tile) * sizeof(float), 0};
    device.Transpose(src_tile, dst_tile, rows, cols);
  }
}

void ExecuteFusedMatMul(Device& device, const Operand& a, const Operand& b, const FusedMatMulConfig& config,
                        Activation activation, float* out_host, OwnedBuffer* retain) {
  ExecuteFusedImpl(device, a, b, config, activation, nullptr, out_host, retain, true);
}

void ExecuteMatMul(Device& device, const Operand& a, const Operand& b, float* out_host, Activation activation,
                   const Operand* residual, OwnedBuffer* retain) {
  ExecuteFusedImpl(device, a, b, {}, activation, residual, out_host, retain, true);
}

void ExecuteGemm(Device& device, const Operand& a, const Operand& b, const Operand* bias, const GemmConfig& config,
                 float* out_host, Activation activation, OwnedBuffer* retain) {
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
  if (retain == nullptr && out_count > 0 && out_host == nullptr) {
    throw std::invalid_argument("missing Gemm output");
  }
  OwnedBuffer out;
  out.device = &device;
  out.shape = out_shape;
  out.buffer = device.Allocate(static_cast<size_t>(out_count) * sizeof(float));
  device.MatMul(a_mat, b_mat, out.buffer, m, k, n);

  const bool scale = config.alpha != 1.f;
  if (bias != nullptr || scale || activation != Activation::None) {
    DeviceBuffer bias_buf{};
    const DeviceBuffer* bias_ptr = nullptr;
    if (bias != nullptr) {
      bias_buf = Resolve(device, *bias, &bias_upload);
      bias_ptr = &bias_buf;
    }
    device.GemmEpilogue(out.buffer, m, n, config.alpha, bias_ptr, bias == nullptr ? std::vector<int64_t>{} : bias->shape,
                        config.beta, activation);
  }
  NoteActivation(activation);
  Deliver(device, &out, out_count, out_host, retain);
  NoteGemmOp();
}

KvTableStore::KvTableStore(Device& device) : device_(&device) {}

KvTableStore::Slot* KvTableStore::Find(const std::string& name) {
  for (auto& entry : slots_) {
    if (entry.first == name) {
      return &entry.second;
    }
  }
  return nullptr;
}

const KvTableStore::Slot* KvTableStore::Find(const std::string& name) const {
  for (const auto& entry : slots_) {
    if (entry.first == name) {
      return &entry.second;
    }
  }
  return nullptr;
}

void KvTableStore::Create(const std::string& name, int64_t rows, int64_t cols) {
  if (name.empty() || rows < 0 || cols < 0 || Find(name) != nullptr) {
    throw std::invalid_argument("KvTable Create rejected");
  }
  Slot slot;
  slot.rows = rows;
  slot.cols = cols;
  slot.filled = 0;
  slot.buffer.device = device_;
  slot.buffer.shape = {rows, cols};
  slot.buffer.buffer = device_->Allocate(static_cast<size_t>(rows * cols) * sizeof(float));
  slots_.emplace_back(name, std::move(slot));
}

void KvTableStore::Append(const std::string& name, const float* data, int64_t n_rows) {
  Slot* slot = Find(name);
  if (slot == nullptr || n_rows < 0 || slot->filled > slot->rows - n_rows) {
    throw std::invalid_argument("KvTable Append rejected");
  }
  if (n_rows > 0 && data == nullptr) {
    throw std::invalid_argument("KvTable Append rejected");
  }
  if (n_rows > 0) {
    auto* base = static_cast<float*>(slot->buffer.buffer.data);
    const int64_t elems = n_rows * slot->cols;
    const DeviceBuffer tile{base + slot->filled * slot->cols, static_cast<size_t>(elems) * sizeof(float), 0};
    device_->CopyHostToDevice(tile, data, tile.bytes);
  }
  slot->filled += n_rows;
}

void KvTableStore::Update(const std::string& name, int64_t row, const float* data, int64_t n_rows) {
  Slot* slot = Find(name);
  if (slot == nullptr || row < 0 || n_rows < 0 || row > slot->filled || n_rows > slot->filled - row) {
    throw std::invalid_argument("KvTable Update rejected");
  }
  if (n_rows > 0 && data == nullptr) {
    throw std::invalid_argument("KvTable Update rejected");
  }
  if (n_rows > 0) {
    auto* base = static_cast<float*>(slot->buffer.buffer.data);
    const int64_t elems = n_rows * slot->cols;
    const DeviceBuffer tile{base + row * slot->cols, static_cast<size_t>(elems) * sizeof(float), 0};
    device_->CopyHostToDevice(tile, data, tile.bytes);
  }
}

bool KvTableStore::Contains(const std::string& name) const { return Find(name) != nullptr; }

void KvTableStore::Read(const std::string& name, float* dst, int64_t n_rows) const {
  const Slot* slot = Find(name);
  if (slot == nullptr || dst == nullptr || n_rows < 0 || n_rows > slot->filled) {
    throw std::invalid_argument("KvTable Read rejected");
  }
  if (n_rows == 0) {
    return;
  }
  const DeviceBuffer src{slot->buffer.buffer.data, static_cast<size_t>(n_rows * slot->cols) * sizeof(float),
                         slot->buffer.buffer.id};
  device_->CopyDeviceToHost(dst, src, src.bytes);
}

const void* KvTableStore::Storage(const std::string& name) const {
  const Slot* slot = Find(name);
  if (slot == nullptr) {
    throw std::invalid_argument("KvTable Storage rejected");
  }
  return slot->buffer.buffer.data;
}

int64_t KvTableStore::Filled(const std::string& name) const {
  const Slot* slot = Find(name);
  if (slot == nullptr) {
    throw std::invalid_argument("KvTable Filled rejected");
  }
  return slot->filled;
}

}  // namespace accel
