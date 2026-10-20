#include "accel/runtime.h"
#include "recording_device.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace {

int g_failures = 0;

void Check(bool cond, const char* expr, const char* file, int line) {
  if (!cond) {
    std::fprintf(stderr, "FAIL %s:%d %s\n", file, line, expr);
    ++g_failures;
  }
}

#define CHECK(expr) Check(static_cast<bool>(expr), #expr, __FILE__, __LINE__)

void ExpectClose(const std::vector<float>& got, const std::vector<float>& want) {
  CHECK(got.size() == want.size());
  if (got.size() != want.size()) {
    return;
  }
  for (size_t i = 0; i < got.size(); ++i) {
    const float scale = std::fabs(want[i]) > 1.f ? std::fabs(want[i]) : 1.f;
    if (std::fabs(got[i] - want[i]) > 1e-5f * scale) {
      std::fprintf(stderr, "FAIL mismatch at %zu got %g want %g\n", i, got[i], want[i]);
      ++g_failures;
      return;
    }
  }
}

void TestKnownMatMul() {
  const std::vector<float> a = {1, 2, 3, 4};
  const std::vector<float> b = {5, 6, 7, 8};
  auto device = accel::CreateSimDevice();
  std::vector<float> out(4);
  accel::ExecuteMatMul(*device, {a.data(), nullptr, {2, 2}}, {b.data(), nullptr, {2, 2}}, out.data());
  ExpectClose(out, {19, 22, 43, 50});
}

void TestIdentityBatch() {
  const std::vector<float> a = {1, 2, 3, 4, 5, 6, 7, 8};
  const std::vector<float> b = {1, 0, 0, 1};
  auto device = accel::CreateSimDevice();
  std::vector<float> out(8);
  accel::ExecuteMatMul(*device, {a.data(), nullptr, {2, 2, 2}}, {b.data(), nullptr, {2, 2}}, out.data());
  ExpectClose(out, a);
}

void TestBroadcastHandValues() {
  // A is (2, 1, 2, 2), both copies of [[1, 2], [3, 4]].
  const std::vector<float> a = {1, 2, 3, 4, 1, 2, 3, 4};
  // B is (1, 2, 2, 2): identity, then [[0, 1], [1, 0]].
  const std::vector<float> b = {1, 0, 0, 1, 0, 1, 1, 0};
  auto device = accel::CreateSimDevice();
  std::vector<float> out(16);
  accel::ExecuteMatMul(*device, {a.data(), nullptr, {2, 1, 2, 2}}, {b.data(), nullptr, {1, 2, 2, 2}}, out.data());
  const std::vector<float> want = {
      1, 2, 3, 4, 2, 1, 4, 3, 1, 2, 3, 4, 2, 1, 4, 3,
  };
  ExpectClose(out, want);
}

void TestGemmTransBias() {
  const std::vector<float> a = {1, 2, 3, 4, 5, 6};
  const std::vector<float> b = {1, 0, 1, 0, 1, 1};
  const std::vector<float> bias = {1, 3};
  auto device = accel::CreateSimDevice();
  std::vector<float> out(4);
  accel::GemmConfig config;
  config.trans_b = true;
  config.alpha = 2.f;
  const accel::Operand bias_op{bias.data(), nullptr, {2}};
  accel::ExecuteGemm(*device, {a.data(), nullptr, {2, 3}}, {b.data(), nullptr, {2, 3}}, &bias_op, config,
                     out.data());
  ExpectClose(out, {9, 13, 21, 25});
}

void TestWeightStaysOnDevice() {
  const std::vector<float> a = {1, 2, 3, 4};
  std::vector<float> b = {5, 6, 7, 8};
  auto device = accel::CreateSimDevice();
  accel::OwnedBuffer weight = accel::Upload(*device, b.data(), {2, 2});
  b[0] = 0;
  accel::ResetAcceleratorStats();
  std::vector<float> out(4);
  accel::Operand b_op;
  b_op.on_device = &weight;
  b_op.shape = weight.shape;
  accel::ExecuteMatMul(*device, {a.data(), nullptr, {2, 2}}, b_op, out.data());
  ExpectClose(out, {19, 22, 43, 50});
  const accel::AcceleratorStats stats = accel::GetAcceleratorStats();
  CHECK(stats.bytes_to_device == static_cast<int64_t>(a.size() * sizeof(float)));
  CHECK(stats.matmul_ops == 1);
  CHECK(stats.kernel_matmuls == 1);
  CHECK(stats.bytes_to_host == static_cast<int64_t>(out.size() * sizeof(float)));
}

void TestZeroK() {
  const std::vector<float> a;
  const std::vector<float> b;
  auto device = accel::CreateSimDevice();
  std::vector<float> out(6, 7.f);
  accel::ExecuteMatMul(*device, {a.data(), nullptr, {2, 0}}, {b.data(), nullptr, {0, 3}}, out.data());
  ExpectClose(out, {0, 0, 0, 0, 0, 0});
}

void TestRejectsBadShape() {
  std::vector<int64_t> out;
  CHECK(!accel::InferMatMulShape({2, 3}, {4, 5}, &out));
  CHECK(!accel::InferGemmShape({2, 3}, false, {2, 3}, false, nullptr, &out));
  auto device = accel::CreateSimDevice();
  bool threw = false;
  try {
    std::vector<float> a(6, 1.f);
    std::vector<float> b(6, 1.f);
    std::vector<float> c(4);
    accel::ExecuteMatMul(*device, {a.data(), nullptr, {2, 3}}, {b.data(), nullptr, {2, 3}}, c.data());
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  CHECK(threw);
}

void TestUnknownFreeThrows() {
  auto device = accel::CreateSimDevice();
  float local = 1.f;
  bool threw = false;
  try {
    device->Free({&local, sizeof(local)});
  } catch (const std::logic_error&) {
    threw = true;
  }
  CHECK(threw);
}

void TestFusedMatMulTransAAlpha() {
  // Stored A is [K, M] = [3, 2]. transA yields logical [[1, 3, 5], [2, 4, 6]].
  const std::vector<float> a = {1, 2, 3, 4, 5, 6};
  const std::vector<float> b = {1, 0, 0, 1, 1, 1};
  auto device = accel::CreateSimDevice();
  std::vector<float> out(4);
  accel::FusedMatMulConfig config;
  config.trans_a = true;
  config.alpha = 1.5f;
  accel::ExecuteFusedMatMul(*device, {a.data(), nullptr, {3, 2}}, {b.data(), nullptr, {3, 2}}, config,
                            accel::Activation::None, out.data());
  ExpectClose(out, {9, 12, 12, 15});
}

void TestFusedMatMulBroadcastTransB() {
  const std::vector<float> a = {1, 2, 3, 4, 5, 6, 1, 0, 1, 0, 1, 1};
  // Physical [2, 3], transposed on the last two dimensions to [[1, 0], [0, 1], [1, 1]].
  const std::vector<float> b = {1, 0, 1, 0, 1, 1};
  auto device = accel::CreateSimDevice();
  std::vector<float> out(8);
  accel::FusedMatMulConfig config;
  config.trans_b = true;
  accel::ExecuteFusedMatMul(*device, {a.data(), nullptr, {2, 2, 3}}, {b.data(), nullptr, {2, 3}}, config,
                            accel::Activation::None, out.data());
  ExpectClose(out, {4, 5, 10, 11, 2, 1, 1, 2});
}

void TestMatMulReluStaysOnDevice() {
  const std::vector<float> a = {1, -2, -3, 4};
  const std::vector<float> b = {1, 0, 0, 1};
  auto device = accel::CreateSimDevice();
  accel::ResetAcceleratorStats();
  std::vector<float> out(4, 9.f);
  accel::ExecuteMatMul(*device, {a.data(), nullptr, {2, 2}}, {b.data(), nullptr, {2, 2}}, out.data(),
                       accel::Activation::Relu);
  ExpectClose(out, {1, 0, 0, 4});
  const accel::AcceleratorStats stats = accel::GetAcceleratorStats();
  CHECK(stats.relu_epilogues == 1);
  CHECK(stats.matmul_ops == 1);
  CHECK(stats.bytes_to_host == static_cast<int64_t>(out.size() * sizeof(float)));
  CHECK(stats.bytes_to_device == static_cast<int64_t>((a.size() + b.size()) * sizeof(float)));
}

void TestGemmReluEpilogue() {
  const std::vector<float> a = {1, 0, 0, 1};
  const std::vector<float> b = {1, 0, 0, 1};
  const std::vector<float> bias = {-2, 3};
  auto device = accel::CreateSimDevice();
  accel::ResetAcceleratorStats();
  std::vector<float> out(4);
  accel::GemmConfig config;
  const accel::Operand bias_op{bias.data(), nullptr, {2}};
  accel::ExecuteGemm(*device, {a.data(), nullptr, {2, 2}}, {b.data(), nullptr, {2, 2}}, &bias_op, config, out.data(),
                     accel::Activation::Relu);
  ExpectClose(out, {0, 3, 0, 4});
  CHECK(accel::GetAcceleratorStats().relu_epilogues == 1);
  CHECK(accel::GetAcceleratorStats().gemm_ops == 1);
}

void TestRejectsFusedShape() {
  std::vector<int64_t> out;
  // transA turns [2, 3] into [3, 2], which does not contract with [3, 4].
  CHECK(!accel::InferFusedMatMulShape({2, 3}, true, {3, 4}, false, &out));
}

uint32_t Bits(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

float FromBits(uint32_t bits) {
  float value = 0.f;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

uint64_t HashFloats(const std::vector<float>& values) {
  return accel::Fnv1a64(values.empty() ? nullptr : values.data(), values.size() * sizeof(float));
}

void FillLcg(std::vector<float>* out, int64_t count, uint32_t* seed) {
  out->resize(static_cast<size_t>(count));
  for (int64_t i = 0; i < count; ++i) {
    *seed = *seed * 1664525u + 1013904223u;
    const int shifted = static_cast<int>(*seed % 10007u) - 5003;
    (*out)[static_cast<size_t>(i)] = static_cast<float>(static_cast<double>(shifted) * 0.1);
  }
}

void TestPackedGemmProbes() {
  auto check_shape = [](int64_t m, int64_t k, int64_t n, uint32_t seed) {
    std::vector<float> a;
    std::vector<float> b;
    FillLcg(&a, m * k, &seed);
    FillLcg(&b, k * n, &seed);
    auto device = accel::CreateSimDevice();
    accel::ResetPackedGemmProbes();
    std::vector<float> out(static_cast<size_t>(m * n), 1.f);
    accel::ExecuteMatMul(*device, {a.data(), nullptr, {m, k}}, {b.data(), nullptr, {k, n}}, out.data());
    std::vector<float> ref(static_cast<size_t>(m * n), 1.f);
    accel::ReferenceMatMul(a.data(), b.data(), ref.data(), m, k, n);
    CHECK(std::memcmp(out.data(), ref.data(), out.size() * sizeof(float)) == 0);
    CHECK(accel::PackedMatMulCalls() == 1);
    CHECK(accel::PackedKTrips() == k);
    CHECK(accel::PackedKTrips() != 64);
    CHECK(accel::PackedKTrips() != 192);
    const int64_t calls = accel::PackedMatMulCalls();
    const int64_t trips = accel::PackedKTrips();
    accel::ReferenceMatMul(a.data(), b.data(), ref.data(), m, k, n);
    accel::ResetAcceleratorStats();
    CHECK(accel::PackedMatMulCalls() == calls);
    CHECK(accel::PackedKTrips() == trips);
  };
  check_shape(128, 160, 96, 1);
  check_shape(13, 19, 17, 2);

  auto device = accel::CreateSimDevice();
  accel::ResetPackedGemmProbes();
  accel::ResetAcceleratorStats();
  std::vector<float> out(1, 3.f);
  const float dummy = 0.f;
  accel::ExecuteMatMul(*device, {&dummy, nullptr, {1, 0}}, {&dummy, nullptr, {0, 1}}, out.data());
  CHECK(out[0] == 0.f);
  CHECK(accel::PackedMatMulCalls() == 1);
  CHECK(accel::PackedKTrips() == 0);
  CHECK(accel::GetAcceleratorStats().kernel_matmuls == 1);
}

void TestInferRejectsNegative() {
  std::vector<int64_t> out;
  CHECK(!accel::InferMatMulShape({-1, 8}, {8, 4}, &out));
  CHECK(!accel::InferMatMulShape({2, -1}, {-1, 4}, &out));
}

void TestSplitProductBits() {
  const float a_host[2] = {FromBits(0xBE2694CDu), FromBits(0x3E41AF7Du)};
  const float b_host[2] = {FromBits(0x3F800000u), FromBits(0x3F5C2CB8u)};
  auto device = accel::CreateSimDevice();
  accel::OwnedBuffer a = accel::Upload(*device, a_host, {1, 2});
  accel::OwnedBuffer b = accel::Upload(*device, b_host, {2, 1});
  accel::DeviceBuffer c = device->Allocate(sizeof(float));
  device->MatMul(a.buffer, b.buffer, c, 1, 2, 1);
  float out = 0.f;
  device->CopyDeviceToHost(&out, c, sizeof(out));
  device->Free(c);
  CHECK(Bits(out) == 0xB5200000u);
}

void TestNonFiniteEpilogueBits() {
  auto device = accel::CreateSimDevice();
  const float pos_inf = FromBits(0x7F800000u);
  const float neg_inf = FromBits(0xFF800000u);
  const float zero = 0.f;
  const float one = 1.f;
  std::vector<float> out(1);

  accel::ResetAcceleratorStats();
  accel::ExecuteMatMul(*device, {&pos_inf, nullptr, {1, 1}}, {&zero, nullptr, {1, 1}}, out.data(),
                       accel::Activation::Relu);
  const uint32_t nan_bits = Bits(out[0]);
  CHECK(((nan_bits >> 23) & 0xffu) == 0xffu);
  CHECK((nan_bits & 0x7fffffu) != 0u);
  CHECK(accel::GetAcceleratorStats().relu_epilogues == 1);

  accel::ExecuteMatMul(*device, {&pos_inf, nullptr, {1, 1}}, {&one, nullptr, {1, 1}}, out.data());
  CHECK(Bits(out[0]) == 0x7F800000u);

  accel::ResetAcceleratorStats();
  accel::ExecuteMatMul(*device, {&neg_inf, nullptr, {1, 1}}, {&one, nullptr, {1, 1}}, out.data(),
                       accel::Activation::Relu);
  CHECK(Bits(out[0]) == 0x00000000u);
  CHECK(accel::GetAcceleratorStats().relu_epilogues == 1);

  accel::GemmConfig config;
  config.alpha = -1.f;
  const float signed_zero_a = 0.f;
  const float signed_zero_b = 1.f;
  accel::ExecuteGemm(*device, {&signed_zero_a, nullptr, {1, 1}}, {&signed_zero_b, nullptr, {1, 1}}, nullptr, config,
                     out.data(), accel::Activation::Relu);
  CHECK(Bits(out[0]) == 0x80000000u);
}

void TestMatMulResidualOrientation() {
  const std::vector<float> a = {1, 2, 3, 4};
  const std::vector<float> b = {1, 0, 0, 1};
  const std::vector<float> residual = {-1, 5, 0, -4};
  auto device = accel::CreateSimDevice();
  accel::ResetAcceleratorStats();
  std::vector<float> out(4, 9.f);
  const accel::Operand residual_op{residual.data(), nullptr, {2, 2}};
  accel::ExecuteMatMul(*device, {a.data(), nullptr, {2, 2}}, {b.data(), nullptr, {2, 2}}, out.data(),
                       accel::Activation::None, &residual_op);
  ExpectClose(out, {0, 7, 3, 0});
  CHECK(accel::GetAcceleratorStats().add_epilogues == 1);
  CHECK(accel::GetAcceleratorStats().relu_epilogues == 0);
  CHECK(accel::GetAcceleratorStats().kernel_matmuls == 1);

  accel::ResetAcceleratorStats();
  const float dummy = 0.f;
  const accel::Operand empty_residual{&dummy, nullptr, {0}};
  std::vector<float> empty;
  accel::ExecuteMatMul(*device, {&dummy, nullptr, {0, 1}}, {&dummy, nullptr, {1, 1}}, empty.data(),
                       accel::Activation::None, &empty_residual);
  CHECK(accel::GetAcceleratorStats().add_epilogues == 1);
  CHECK(accel::GetAcceleratorStats().kernel_matmuls == 1);
}

void TestPretransposeLastTwo() {
  const std::vector<float> values = {1, 2, 3, 4, 5, 6};
  auto device = accel::CreateSimDevice();
  accel::OwnedBuffer src = accel::Upload(*device, values.data(), {3, 2});
  accel::ResetAcceleratorStats();
  accel::OwnedBuffer dst;
  accel::PretransposeLastTwo(*device, src, &dst);
  CHECK(dst.shape.size() == 2 && dst.shape[0] == 2 && dst.shape[1] == 3);
  std::vector<float> back(6);
  device->CopyDeviceToHost(back.data(), dst.buffer, back.size() * sizeof(float));
  ExpectClose(back, {1, 3, 5, 2, 4, 6});
  CHECK(accel::GetAcceleratorStats().device_transposes == 1);
}

void TestKvTableStore() {
  auto device = accel::CreateSimDevice();
  accel::KvTableStore store(*device);
  store.Create("k", 4, 2);
  float host[4] = {1.f, 2.f, 3.f, 4.f};
  store.Append("k", host, 2);
  host[0] = 0.f;
  const void* storage = store.Storage("k");
  accel::ResetAcceleratorStats();
  const float extra[2] = {5.f, 6.f};
  store.Append("k", extra, 1);
  CHECK(accel::GetAcceleratorStats().bytes_to_device == static_cast<int64_t>(2 * sizeof(float)));
  CHECK(store.Storage("k") == storage);
  CHECK(store.Filled("k") == 3);
  float back[6] = {};
  store.Read("k", back, 3);
  ExpectClose(std::vector<float>(back, back + 6), {1.f, 2.f, 3.f, 4.f, 5.f, 6.f});
  accel::KvTableStore other(*device);
  CHECK(!other.Contains("k"));
  CHECK(store.Contains("k"));
  accel::ResetAcceleratorStats();
  CHECK(store.Filled("k") == 3);

  bool update_threw = false;
  try {
    store.Update("k", 10, extra, 1);
  } catch (const std::invalid_argument&) {
    update_threw = true;
  }
  CHECK(update_threw);
  bool append_threw = false;
  try {
    store.Append("k", extra, 5);
  } catch (const std::invalid_argument&) {
    append_threw = true;
  }
  CHECK(append_threw);
}

class ScopedAcceleratorStats {
 public:
  ScopedAcceleratorStats() { accel::ResetAcceleratorStats(); }
  ~ScopedAcceleratorStats() {}
  ScopedAcceleratorStats(const ScopedAcceleratorStats&) = delete;
  ScopedAcceleratorStats& operator=(const ScopedAcceleratorStats&) = delete;
};

void TestScopedStatsDoNotResetOnExit() {
  const std::vector<float> a = {1, 0, 0, 1};
  const std::vector<float> b = {1, 0, 0, 1};
  auto device = accel::CreateSimDevice();
  std::vector<float> out(4);
  {
    ScopedAcceleratorStats guard;
    accel::ExecuteMatMul(*device, {a.data(), nullptr, {2, 2}}, {b.data(), nullptr, {2, 2}}, out.data());
    CHECK(accel::GetAcceleratorStats().matmul_ops == 1);
  }
  CHECK(accel::GetAcceleratorStats().matmul_ops == 1);
  accel::ExecuteMatMul(*device, {a.data(), nullptr, {2, 2}}, {b.data(), nullptr, {2, 2}}, out.data());
  CHECK(accel::GetAcceleratorStats().matmul_ops == 2);
  accel::ResetAcceleratorStats();
  accel::ExecuteMatMul(*device, {a.data(), nullptr, {2, 2}}, {b.data(), nullptr, {2, 2}}, out.data());
  CHECK(accel::GetAcceleratorStats().matmul_ops == 1);
}

void TestLiveAllocIds() {
  const int64_t before = accel::LiveDeviceAllocs();
  auto device = accel::CreateSimDevice();
  accel::DeviceBuffer first = device->Allocate(128);
  const uint64_t first_id = first.id;
  device->Free(first);
  accel::DeviceBuffer second = device->Allocate(128);
  CHECK(second.id > first_id);
  device->Free(second);
  CHECK(accel::LiveDeviceAllocs() == before);

  auto leaking = accel::CreateSimDevice();
  accel::DeviceBuffer leaked = leaking->Allocate(64);
  const uint64_t leaked_id = leaked.id;
  const int64_t live = accel::LiveDeviceAllocs();
  leaking.reset();
  CHECK(accel::LiveDeviceAllocs() == live);
  auto later = accel::CreateSimDevice();
  accel::DeviceBuffer after = later->Allocate(64);
  CHECK(after.id > leaked_id);
  later->Free(after);
}

std::vector<accel::TraceOp> ComputeOps(const std::vector<accel::TraceOp>& trace) {
  std::vector<accel::TraceOp> ops;
  for (const accel::TraceOp& op : trace) {
    if (op.op == "Transpose" || op.op == "MatMul" || op.op == "GemmEpilogue") {
      ops.push_back(op);
    }
  }
  return ops;
}

int CountOp(const std::vector<accel::TraceOp>& trace, const char* name) {
  int count = 0;
  for (const accel::TraceOp& op : trace) {
    if (op.op == name) {
      ++count;
    }
  }
  return count;
}

void TestRecordingDevice() {
  CHECK(accel::Fnv1a64(nullptr, 0) == 14695981039346656037ull);
  CHECK(accel::Fnv1a64("a", 1) == 0xaf63dc4c8601ec8cull);

  const std::vector<float> a = {1, 2, 3, 4, 5, 6};
  const std::vector<float> b = {1, 0, 0, 1, 1, 1};
  auto inner = accel::CreateSimDevice();
  accel::RecordingDevice recorder(*inner);
  accel::ResetAcceleratorStats();
  std::vector<float> out(4);
  accel::FusedMatMulConfig config;
  config.trans_a = true;
  config.alpha = 1.5f;
  accel::ExecuteFusedMatMul(recorder, {a.data(), nullptr, {3, 2}}, {b.data(), nullptr, {3, 2}}, config,
                            accel::Activation::None, out.data());
  ExpectClose(out, {9, 12, 12, 15});
  const std::vector<accel::TraceOp> trace = recorder.Trace();
  CHECK(CountOp(trace, "Allocate") == 4);
  CHECK(CountOp(trace, "CopyHostToDevice") == 2);
  CHECK(CountOp(trace, "CopyDeviceToHost") == 1);
  CHECK(CountOp(trace, "Free") == 4);
  std::vector<accel::TraceOp> allocs;
  std::vector<accel::TraceOp> frees;
  uint64_t first_h2d = 0;
  bool saw_h2d = false;
  for (const accel::TraceOp& op : trace) {
    if (op.op == "Allocate") {
      allocs.push_back(op);
    } else if (op.op == "Free") {
      frees.push_back(op);
    } else if (op.op == "CopyHostToDevice" && !saw_h2d) {
      first_h2d = op.hash;
      saw_h2d = true;
    }
  }
  CHECK(allocs.size() == 4);
  CHECK(allocs[0].id == 1 && allocs[0].bytes == 24);
  CHECK(allocs[1].id == 2 && allocs[1].bytes == 24);
  CHECK(allocs[2].id == 3 && allocs[2].bytes == 16);
  CHECK(allocs[3].id == 4 && allocs[3].bytes == 24);
  CHECK(frees.size() == 4);
  CHECK(frees[0].id == 4 && frees[1].id == 3 && frees[2].id == 2 && frees[3].id == 1);
  const std::vector<accel::TraceOp> compute = ComputeOps(trace);
  CHECK(compute.size() == 3);
  CHECK(compute[0].op == "Transpose" && compute[0].rows == 3 && compute[0].cols == 2);
  CHECK(compute[0].src_hash == first_h2d);
  CHECK(compute[0].dst_hash == HashFloats({1, 3, 5, 2, 4, 6}));
  CHECK(compute[1].op == "MatMul" && compute[1].m == 2 && compute[1].k == 3 && compute[1].n == 2);
  CHECK(compute[1].c_hash == HashFloats({6, 8, 8, 10}));
  CHECK(compute[2].op == "GemmEpilogue" && compute[2].m == 4 && compute[2].n == 1);
  CHECK(compute[2].alpha == 1.5f && compute[2].beta == 0.f && !compute[2].has_bias);
  CHECK(compute[2].activation == "none");
  CHECK(compute[2].hash_after == HashFloats({9, 12, 12, 15}));
  const accel::AcceleratorStats stats = accel::GetAcceleratorStats();
  CHECK(stats.bytes_to_device == 48);
  CHECK(stats.bytes_to_host == 160);
  CHECK(stats.matmul_ops == 1);
  CHECK(stats.gemm_ops == 0);
  CHECK(stats.kernel_matmuls == 1);
  CHECK(stats.relu_epilogues == 0);
  const size_t traced = recorder.Trace().size();
  auto replay_device = accel::CreateSimDevice();
  CHECK(recorder.Replay(*replay_device));
  CHECK(accel::GetAcceleratorStats().kernel_matmuls == 2);
  CHECK(recorder.Trace().size() == traced);

  const std::vector<float> batch_a = {1, 2, 3, 4, 5, 6, 1, 0, 1, 0, 1, 1};
  const std::vector<float> batch_b = {1, 0, 1, 0, 1, 1};
  auto inner_b = accel::CreateSimDevice();
  accel::RecordingDevice recorder_b(*inner_b);
  accel::ResetAcceleratorStats();
  std::vector<float> batch_out(8);
  accel::FusedMatMulConfig batch_config;
  batch_config.trans_b = true;
  accel::ExecuteFusedMatMul(recorder_b, {batch_a.data(), nullptr, {2, 2, 3}}, {batch_b.data(), nullptr, {2, 3}},
                            batch_config, accel::Activation::None, batch_out.data());
  ExpectClose(batch_out, {4, 5, 10, 11, 2, 1, 1, 2});
  const std::vector<accel::TraceOp> batch_compute = ComputeOps(recorder_b.Trace());
  CHECK(batch_compute.size() == 4);
  CHECK(batch_compute[0].op == "Transpose" && batch_compute[1].op == "MatMul");
  CHECK(batch_compute[2].op == "Transpose" && batch_compute[3].op == "MatMul");
  CHECK(batch_compute[0].dst.id == batch_compute[2].dst.id);
  CHECK(batch_compute[0].src.id != 1 && batch_compute[2].src.id != 1);
  CHECK(batch_compute[3].a.id == 1 && batch_compute[3].a.byte_offset == 24 && batch_compute[3].a.length == 24);
  CHECK(batch_compute[3].c.id == 3 && batch_compute[3].c.byte_offset == 16 && batch_compute[3].c.length == 16);
  CHECK(accel::GetAcceleratorStats().bytes_to_device == 72);
}

void TestHostCopyIsPrivate() {
  float host[4] = {1.f, 2.f, 3.f, 4.f};
  auto device = accel::CreateSimDevice();
  accel::OwnedBuffer uploaded = accel::Upload(*device, host, {2, 2});
  host[0] = 0.f;
  float back[4] = {};
  device->CopyDeviceToHost(back, uploaded.buffer, sizeof(back));
  CHECK(back[0] == 1.f && back[1] == 2.f && back[2] == 3.f && back[3] == 4.f);
  CHECK(uploaded.buffer.data != host);
}

}  // namespace

int main() {
  TestKnownMatMul();
  TestIdentityBatch();
  TestBroadcastHandValues();
  TestGemmTransBias();
  TestWeightStaysOnDevice();
  TestZeroK();
  TestRejectsBadShape();
  TestUnknownFreeThrows();
  TestFusedMatMulTransAAlpha();
  TestFusedMatMulBroadcastTransB();
  TestMatMulReluStaysOnDevice();
  TestGemmReluEpilogue();
  TestRejectsFusedShape();
  TestPackedGemmProbes();
  TestInferRejectsNegative();
  TestSplitProductBits();
  TestNonFiniteEpilogueBits();
  TestMatMulResidualOrientation();
  TestPretransposeLastTwo();
  TestKvTableStore();
  TestScopedStatsDoNotResetOnExit();
  TestLiveAllocIds();
  TestRecordingDevice();
  TestHostCopyIsPrivate();
  if (g_failures != 0) {
    std::fprintf(stderr, "%d checks failed\n", g_failures);
    return 1;
  }
  std::printf("runtime tests passed\n");
  return 0;
}
