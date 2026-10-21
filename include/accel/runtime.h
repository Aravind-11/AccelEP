#pragma once

#include "accel/device.h"

#include <cstdint>
#include <string>
#include <vector>

namespace accel {

struct AcceleratorStats {
  int64_t matmul_ops = 0;
  int64_t gemm_ops = 0;
  int64_t kernel_matmuls = 0;
  int64_t relu_epilogues = 0;
  int64_t gelu_epilogues = 0;
  int64_t add_epilogues = 0;
  int64_t device_transposes = 0;
  int64_t bytes_to_device = 0;
  int64_t bytes_to_host = 0;
};

// Valid only while no Compute is inside Execute* or a noting SimDevice method,
// and not mid-Compile (SaveConstant uploads). The race stays open: no mutex,
// no in-flight flag, no error return, no epoch, no generation counter.
void ResetAcceleratorStats();
AcceleratorStats GetAcceleratorStats();
void NoteMatmulOp();
void NoteGemmOp();
void NoteKernelMatmul();
void NoteReluEpilogue();
void NoteGeluEpilogue();
void NoteAddEpilogue();
void NoteDeviceTranspose();
void NoteBytesToDevice(size_t bytes);
void NoteBytesToHost(size_t bytes);

struct ProfileTensor {
  std::string name;
  std::vector<int64_t> shape;
};

struct ProfileRecord {
  std::string op;
  std::string activation;
  std::vector<ProfileTensor> inputs;
  ProfileTensor output;
  int64_t bytes_to_device = 0;
  int64_t bytes_to_host = 0;
};

void AppendProfileRecord(ProfileRecord record);
int WriteProfile(const char* path);
void ClearProfileForTest();

struct ConstantTouch {
  uint64_t id = 0;
  uint64_t address = 0;
  uint64_t bytes = 0;
};

void NoteConstantTouch(uint64_t id, const void* address, uint64_t bytes);
void ClearConstantTouches();
int64_t ConstantTouchCount();
uint64_t ConstantTouchId(int64_t index);
uint64_t ConstantTouchAddress(int64_t index);
uint64_t ConstantTouchBytes(int64_t index);

// Constant int8 weight kept packed. axis < 0 is one scale for the whole tensor.
// `q`, `scale`, and `zero` stay on the device. `zero` is null when the op has none.
struct QuantPlane {
  const int8_t* q = nullptr;
  const float* scale = nullptr;
  const int8_t* zero = nullptr;
  std::vector<int64_t> shape;
  int64_t axis = -1;
  int64_t offset = 0;
};

struct Operand {
  const float* host = nullptr;
  const OwnedBuffer* on_device = nullptr;
  std::vector<int64_t> shape;
  bool quant = false;
  QuantPlane quant_plane{};
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

bool InferMatMulShape(const std::vector<int64_t>& a, const std::vector<int64_t>& b, std::vector<int64_t>* out);

// Logical MatMul shape after transposing the last two dimensions of either input.
bool InferFusedMatMulShape(const std::vector<int64_t>& a, bool trans_a, const std::vector<int64_t>& b, bool trans_b,
                           std::vector<int64_t>* out);

// `bias` is null when Gemm has no C. On success `out` is [M, N].
bool InferGemmShape(const std::vector<int64_t>& a, bool trans_a, const std::vector<int64_t>& b, bool trans_b,
                    const std::vector<int64_t>* bias, std::vector<int64_t>* out);

OwnedBuffer Upload(Device& device, const float* data, const std::vector<int64_t>& shape);

// Swaps the last two dimensions of a constant weight into a second buffer.
// One Device::Transpose per leading batch. Does not fold alpha.
void PretransposeLastTwo(Device& device, const OwnedBuffer& src, OwnedBuffer* dst);

// Same product order as SimDevice::MatMul. A null plane reads the float pointer.
// A right-hand plane is dequantized into one 64x256 float panel at a time.
// Notes one kernel matmul and one packed-GEMM probe, like Device::MatMul.
void MatMulQuantized(const float* a, const QuantPlane* a_quant, const float* b, const QuantPlane* b_quant, float* c,
                     int64_t m, int64_t k, int64_t n);

// Writes the MatMul result into `out_host`, or moves it into `retain` and skips
// the host copy. `residual`, when set, is added elementwise after the activation.
// `activation` and `residual` are independent. A null residual does not note an add.
void ExecuteMatMul(Device& device, const Operand& a, const Operand& b, float* out_host,
                   Activation activation = Activation::None, const Operand* residual = nullptr,
                   OwnedBuffer* retain = nullptr);

// Writes the FusedMatMul result into `out_host`, or retains it on device.
void ExecuteFusedMatMul(Device& device, const Operand& a, const Operand& b, const FusedMatMulConfig& config,
                        Activation activation, float* out_host, OwnedBuffer* retain = nullptr);

// Writes the Gemm result into `out_host`. `bias` is null when the op has no C.
void ExecuteGemm(Device& device, const Operand& a, const Operand& b, const Operand* bias, const GemmConfig& config,
                 float* out_host, Activation activation = Activation::None, OwnedBuffer* retain = nullptr);

// Name-keyed device rows. Not an ONNX op and not a paged cache. Each instance
// has its own map. ResetAcceleratorStats does not clear slots.
class KvTableStore {
 public:
  explicit KvTableStore(Device& device);

  void Create(const std::string& name, int64_t rows, int64_t cols);
  void Append(const std::string& name, const float* data, int64_t n_rows);
  void Update(const std::string& name, int64_t row, const float* data, int64_t n_rows);
  bool Contains(const std::string& name) const;
  void Read(const std::string& name, float* dst, int64_t n_rows) const;
  const void* Storage(const std::string& name) const;
  int64_t Filled(const std::string& name) const;

 private:
  struct Slot {
    OwnedBuffer buffer;
    int64_t rows = 0;
    int64_t cols = 0;
    int64_t filled = 0;
  };

  Device* device_ = nullptr;
  std::vector<std::pair<std::string, Slot>> slots_;

  Slot* Find(const std::string& name);
  const Slot* Find(const std::string& name) const;
};

}  // namespace accel
