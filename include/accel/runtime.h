#pragma once

#include "accel/device.h"

#include <cstdint>
#include <vector>

namespace accel {

struct AcceleratorStats {
  int64_t matmul_ops = 0;
  int64_t gemm_ops = 0;
  int64_t kernel_matmuls = 0;
  int64_t relu_epilogues = 0;
  int64_t bytes_to_device = 0;
  int64_t bytes_to_host = 0;
};

void ResetAcceleratorStats();
AcceleratorStats GetAcceleratorStats();
void NoteMatmulOp();
void NoteGemmOp();
void NoteKernelMatmul();
void NoteReluEpilogue();
void NoteBytesToDevice(size_t bytes);
void NoteBytesToHost(size_t bytes);

struct Operand {
  const float* host = nullptr;
  const OwnedBuffer* on_device = nullptr;
  std::vector<int64_t> shape;
};

struct GemmConfig {
  bool trans_a = false;
  bool trans_b = false;
  float alpha = 1.f;
  float beta = 1.f;
};

// com.microsoft.FusedMatMul: MatMul, then transpose of the last two dimensions
// and a scale. Batch-dimension transpose is not part of this config.
struct FusedMatMulConfig {
  bool trans_a = false;
  bool trans_b = false;
  float alpha = 1.f;
};

int64_t ElementCount(const std::vector<int64_t>& shape);

bool InferMatMulShape(const std::vector<int64_t>& a, const std::vector<int64_t>& b,
                      std::vector<int64_t>* out);

// Logical MatMul shape after transposing the last two dimensions of either input.
bool InferFusedMatMulShape(const std::vector<int64_t>& a, bool trans_a, const std::vector<int64_t>& b,
                           bool trans_b, std::vector<int64_t>* out);

// `bias` is null when Gemm has no C. On success `out` is [M, N].
bool InferGemmShape(const std::vector<int64_t>& a, bool trans_a,
                    const std::vector<int64_t>& b, bool trans_b,
                    const std::vector<int64_t>* bias, std::vector<int64_t>* out);

OwnedBuffer Upload(Device& device, const float* data, const std::vector<int64_t>& shape);

// Writes the MatMul result into `out_host`, which must hold InferMatMulShape elements.
// `relu` applies the epilogue on device before the result is copied back.
void ExecuteMatMul(Device& device, const Operand& a, const Operand& b, float* out_host, bool relu = false);

// Writes the FusedMatMul result into `out_host`.
void ExecuteFusedMatMul(Device& device, const Operand& a, const Operand& b, const FusedMatMulConfig& config,
                        bool relu, float* out_host);

// Writes the Gemm result into `out_host`. `bias` is null when the op has no C.
void ExecuteGemm(Device& device, const Operand& a, const Operand& b, const Operand* bias,
                 const GemmConfig& config, float* out_host, bool relu = false);

}  // namespace accel
