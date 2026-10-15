#include "accel/runtime.h"

#include <cmath>
#include <cstdio>
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

void TestFusedMatMulTransAAlpha() {
  const std::vector<float> a = {1, 2, 3, 4, 5, 6};
  const std::vector<float> b = {1, 0, 0, 1, 1, 1};
  auto device = accel::CreateSimDevice();
  std::vector<float> out(4);
  accel::GemmConfig config;
  config.trans_a = true;
  config.alpha = 1.5f;
  accel::ExecuteFusedMatMul(*device, {a.data(), nullptr, {3, 2}}, {b.data(), nullptr, {3, 2}}, config, out.data());
  ExpectClose(out, {9, 12, 12, 15});
}

void TestFusedMatMulBatchTransB() {
  const std::vector<float> a = {1, 2, 3, 4, 5, 6, 1, 0, 0, 0, 1, 0};
  const std::vector<float> b = {1, 0, 1, 0, 1, 1};
  auto device = accel::CreateSimDevice();
  accel::ResetAcceleratorStats();
  std::vector<float> out(8);
  accel::GemmConfig config;
  config.trans_b = true;
  accel::ExecuteFusedMatMul(*device, {a.data(), nullptr, {2, 2, 3}}, {b.data(), nullptr, {2, 3}}, config, out.data());
  ExpectClose(out, {4, 5, 10, 11, 1, 0, 0, 1});
  const accel::AcceleratorStats stats = accel::GetAcceleratorStats();
  CHECK(stats.fused_matmul_ops == 1);
  CHECK(stats.matmul_ops == 0);
  CHECK(stats.kernel_matmuls == 2);
}

void TestFusedMatMulAlphaZero() {
  const std::vector<float> a = {1, 2, 3, 4};
  const std::vector<float> b = {5, 6, 7, 8};
  auto device = accel::CreateSimDevice();
  std::vector<float> out(4, 9.f);
  accel::GemmConfig config;
  config.alpha = 0.f;
  accel::ExecuteFusedMatMul(*device, {a.data(), nullptr, {2, 2}}, {b.data(), nullptr, {2, 2}}, config, out.data());
  ExpectClose(out, {0, 0, 0, 0});
}

void TestFusedWeightStaysOnDevice() {
  const std::vector<float> a = {1, 2, 3, 4, 5, 6};
  std::vector<float> b = {1, 0, 1, 0, 1, 1};
  auto device = accel::CreateSimDevice();
  accel::OwnedBuffer weight = accel::Upload(*device, b.data(), {2, 3});
  b[0] = 0;
  accel::ResetAcceleratorStats();
  std::vector<float> out(4);
  accel::Operand b_op;
  b_op.on_device = &weight;
  b_op.shape = weight.shape;
  accel::GemmConfig config;
  config.trans_b = true;
  accel::ExecuteFusedMatMul(*device, {a.data(), nullptr, {2, 3}}, b_op, config, out.data());
  ExpectClose(out, {4, 5, 10, 11});
  const accel::AcceleratorStats stats = accel::GetAcceleratorStats();
  CHECK(stats.bytes_to_device == static_cast<int64_t>(a.size() * sizeof(float)));
  CHECK(stats.fused_matmul_ops == 1);
  CHECK(stats.matmul_ops == 0);
  CHECK(stats.kernel_matmuls == 1);
  CHECK(stats.bytes_to_host == static_cast<int64_t>(out.size() * sizeof(float)));
}

void TestRejectsBadShape() {
  std::vector<int64_t> out;
  CHECK(!accel::InferMatMulShape({2, 3}, {4, 5}, &out));
  CHECK(!accel::InferGemmShape({2, 3}, false, {2, 3}, false, nullptr, &out));
  CHECK(!accel::InferFusedMatMulShape({4}, false, {4, 3}, false, &out));
  CHECK(!accel::InferFusedMatMulShape({2, 3}, true, {4, 5}, false, &out));
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
  TestFusedMatMulTransAAlpha();
  TestFusedMatMulBatchTransB();
  TestFusedMatMulAlphaZero();
  TestFusedWeightStaysOnDevice();
  TestWeightStaysOnDevice();
  TestZeroK();
  TestRejectsBadShape();
  TestUnknownFreeThrows();
  TestHostCopyIsPrivate();
  if (g_failures != 0) {
    std::fprintf(stderr, "%d checks failed\n", g_failures);
    return 1;
  }
  std::printf("runtime tests passed\n");
  return 0;
}
