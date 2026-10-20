#include "accel/sim_api.h"

#include "accel/device.h"
#include "accel/runtime.h"

#include <cstring>
#include <exception>
#include <memory>
#include <mutex>
#include <string>
#include <utility>

struct AccelBuf {
  accel::OwnedBuffer owned;
};

struct AccelSim {
  std::unique_ptr<accel::Device> device;
  std::mutex mu;
};

namespace {

thread_local std::string g_error;

void SetError(const std::string& message) { g_error = message; }

void ClearError() { g_error.clear(); }

}  // namespace

AccelSim* accel_sim_create(void) {
  try {
    auto sim = std::unique_ptr<AccelSim>(new AccelSim());
    sim->device = accel::CreateSimDevice();
    ClearError();
    return sim.release();
  } catch (const std::exception& ex) {
    SetError(ex.what());
    return nullptr;
  }
}

void accel_sim_destroy(AccelSim* sim) { delete sim; }

AccelBuf* accel_sim_upload(AccelSim* sim, const float* data, int64_t count) {
  if (sim == nullptr || count < 0 || (count > 0 && data == nullptr)) {
    SetError("invalid upload");
    return nullptr;
  }
  try {
    std::lock_guard<std::mutex> lock(sim->mu);
    auto buf = std::unique_ptr<AccelBuf>(new AccelBuf());
    const size_t bytes = static_cast<size_t>(count) * sizeof(float);
    buf->owned.device = sim->device.get();
    buf->owned.buffer = sim->device->Allocate(bytes);
    buf->owned.shape = {count};
    if (bytes > 0) {
      sim->device->CopyHostToDevice(buf->owned.buffer, data, bytes);
    }
    ClearError();
    return buf.release();
  } catch (const std::exception& ex) {
    SetError(ex.what());
    return nullptr;
  }
}

void accel_sim_release(AccelSim* sim, AccelBuf* buf) {
  if (sim == nullptr || buf == nullptr) {
    return;
  }
  std::lock_guard<std::mutex> lock(sim->mu);
  delete buf;
}

int accel_sim_matmul(AccelSim* sim, const float* a, int64_t m, int64_t k, AccelBuf* b, int64_t n,
                     float* c) {
  if (sim == nullptr || b == nullptr || m < 0 || k < 0 || n < 0 || (m > 0 && k > 0 && a == nullptr) ||
      (m > 0 && n > 0 && c == nullptr)) {
    SetError("invalid matmul");
    return 1;
  }
  try {
    std::lock_guard<std::mutex> lock(sim->mu);
    accel::OwnedBuffer left;
    accel::OwnedBuffer out;
    left.device = sim->device.get();
    out.device = sim->device.get();
    const size_t a_bytes = static_cast<size_t>(m) * static_cast<size_t>(k) * sizeof(float);
    const size_t c_bytes = static_cast<size_t>(m) * static_cast<size_t>(n) * sizeof(float);
    left.buffer = sim->device->Allocate(a_bytes);
    out.buffer = sim->device->Allocate(c_bytes);
    if (a_bytes > 0) {
      sim->device->CopyHostToDevice(left.buffer, a, a_bytes);
    }
    sim->device->MatMul(left.buffer, b->owned.buffer, out.buffer, m, k, n);
    if (c_bytes > 0) {
      sim->device->CopyDeviceToHost(c, out.buffer, c_bytes);
    }
    ClearError();
    return 0;
  } catch (const std::exception& ex) {
    SetError(ex.what());
    return 1;
  }
}

void accel_sim_reset_stats(void) { accel::ResetAcceleratorStats(); }

int64_t accel_sim_matmuls(void) { return accel::GetAcceleratorStats().kernel_matmuls; }

int64_t accel_sim_bytes_to_device(void) { return accel::GetAcceleratorStats().bytes_to_device; }

int64_t accel_sim_bytes_to_host(void) { return accel::GetAcceleratorStats().bytes_to_host; }

const char* accel_sim_last_error(void) { return g_error.c_str(); }
