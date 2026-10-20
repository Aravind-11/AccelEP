#pragma once

#include "accel/device.h"

#include <cstdint>
#include <string>
#include <vector>

namespace accel {

struct SliceRef {
  uint64_t id = 0;
  uint64_t byte_offset = 0;
  uint64_t length = 0;
};

// One forwarded device call. Float payloads stay off this struct.
struct TraceOp {
  std::string op;
  int64_t rows = 0;
  int64_t cols = 0;
  int64_t m = 0;
  int64_t k = 0;
  int64_t n = 0;
  float alpha = 0.f;
  float beta = 0.f;
  bool has_bias = false;
  std::string activation;
  uint64_t id = 0;
  uint64_t bytes = 0;
  SliceRef src;
  SliceRef dst;
  SliceRef a;
  SliceRef b;
  SliceRef c;
  SliceRef bias;
  uint64_t hash = 0;
  uint64_t src_hash = 0;
  uint64_t dst_hash = 0;
  uint64_t a_hash = 0;
  uint64_t b_hash = 0;
  uint64_t c_hash = 0;
  uint64_t hash_before = 0;
  uint64_t hash_after = 0;
};

uint64_t Fnv1a64(const void* data, size_t bytes);

// Forwards to an inner device and records the calls. Allocate ids start at 1
// for this wrapper. A slice keeps id 0 on the device buffer and is logged by
// which live allocation covers the pointer. Replay runs the same calls again.
class RecordingDevice final : public Device {
 public:
  explicit RecordingDevice(Device& inner);

  const char* Name() const override;
  DeviceBuffer Allocate(size_t bytes) override;
  void Free(DeviceBuffer buffer) override;
  void CopyHostToDevice(DeviceBuffer dst, const void* src, size_t bytes) override;
  void CopyDeviceToHost(void* dst, DeviceBuffer src, size_t bytes) override;
  void MatMul(DeviceBuffer a, DeviceBuffer b, DeviceBuffer c, int64_t m, int64_t k, int64_t n) override;
  void Transpose(DeviceBuffer src, DeviceBuffer dst, int64_t rows, int64_t cols) override;
  void GemmEpilogue(DeviceBuffer c, int64_t m, int64_t n, float alpha, const DeviceBuffer* bias,
                    const std::vector<int64_t>& bias_shape, float beta, Activation activation) override;

  const std::vector<TraceOp>& Trace() const { return trace_; }
  bool Replay(Device& target) const;

 private:
  struct Live {
    void* data = nullptr;
    size_t bytes = 0;
    uint64_t id = 0;
    DeviceBuffer inner;
  };

  SliceRef Cover(DeviceBuffer buffer) const;
  uint64_t Peek(DeviceBuffer buffer) const;

  Device& inner_;
  uint64_t next_id_ = 1;
  std::vector<Live> live_;
  std::vector<TraceOp> trace_;
  std::vector<std::vector<unsigned char>> tapes_;
};

}  // namespace accel
