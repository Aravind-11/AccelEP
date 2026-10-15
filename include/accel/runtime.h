#pragma once

#include "accel/device.h"

#include <cstdint>
#include <vector>

namespace accel {

struct AcceleratorStats {
  int64_t matmul_ops = 0;
  int64_t gemm_ops = 0;
  int64_t fused_matmul_ops = 0;
  int64_t kernel_matmuls = 0;
  int64_t bytes_to_device = 0;
  int64_t bytes_to_host = 0;
};

void ResetAcceleratorStats();
AcceleratorStats GetAcceleratorStats();
void NoteMatmulOp();
void NoteGemmOp();
void NoteFusedMatmulOp();
void NoteKernelMatmul();
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

int64_t ElementCount(const std::vector<int64_t>& shape);

bool InferMatMulShape(const std::vector<int64_t>& a, const std::vector<int64_t>& b,
                      std::vector<int64_t>* out);

// `bias` is null when Gemm has no C. On success `out` is [M, N].
bool InferGemmShape(const std::vector<int64_t>& a, bool trans_a,
                    const std::vector<int64_t>& b, bool trans_b,
                    const std::vector<int64_t>* bias, std::vector<int64_t>* out);

// com.microsoft.FusedMatMul with transBatchA and transBatchB left at 0.
// Rank 2 or higher. transA/transB swap the last two dimensions. `out` is the
// NumPy matmul shape of those logical matrices, then scaled by alpha at runtime.
bool InferFusedMatMulShape(const std::vector<int64_t>& a, bool trans_a,
                           const std::vector<int64_t>& b, bool trans_b,
                           std::vector<int64_t>* out);

OwnedBuffer Upload(Device& device, const float* data, const std::vector<int64_t>& shape);

// Writes the MatMul result into `out_host`, which must hold InferMatMulShape elements.
void ExecuteMatMul(Device& device, const Operand& a, const Operand& b, float* out_host);

// Writes the Gemm result into `out_host`. `bias` is null when the op has no C.
void ExecuteGemm(Device& device, const Operand& a, const Operand& b, const Operand* bias,
                 const GemmConfig& config, float* out_host);

// Writes alpha * matmul into `out_host`. `config.beta` is ignored.
// transA/transB transpose each matrix on the last two dimensions.
void ExecuteFusedMatMul(Device& device, const Operand& a, const Operand& b,
                        const GemmConfig& config, float* out_host);

}  // namespace accel
