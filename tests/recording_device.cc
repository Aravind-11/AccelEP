#include "recording_device.h"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <unordered_map>

namespace accel {
namespace {

DeviceBuffer Placed(const DeviceBuffer& allocated, const SliceRef& ref) {
  auto* base = static_cast<unsigned char*>(allocated.data);
  return DeviceBuffer{base + ref.byte_offset, static_cast<size_t>(ref.length), 0};
}

}  // namespace

uint64_t Fnv1a64(const void* data, size_t bytes) {
  uint64_t hash = 14695981039346656037ull;
  const auto* bytes_ptr = static_cast<const unsigned char*>(data);
  for (size_t i = 0; i < bytes; ++i) {
    hash ^= bytes_ptr[i];
    hash *= 1099511628211ull;
  }
  return hash;
}

RecordingDevice::RecordingDevice(Device& inner) : inner_(inner) {}

const char* RecordingDevice::Name() const { return inner_.Name(); }

DeviceBuffer RecordingDevice::Allocate(size_t bytes) {
  DeviceBuffer inner = inner_.Allocate(bytes);
  Live live;
  live.data = inner.data;
  live.bytes = bytes;
  live.id = next_id_++;
  live.inner = inner;
  live_.push_back(live);
  TraceOp op;
  op.op = "Allocate";
  op.id = live.id;
  op.bytes = bytes;
  trace_.push_back(op);
  tapes_.emplace_back();
  return DeviceBuffer{inner.data, bytes, live.id};
}

void RecordingDevice::Free(DeviceBuffer buffer) {
  if (buffer.data == nullptr) {
    return;
  }
  auto it = live_.end();
  for (auto cursor = live_.begin(); cursor != live_.end(); ++cursor) {
    if (cursor->data == buffer.data && cursor->bytes == buffer.bytes && cursor->id == buffer.id) {
      it = cursor;
      break;
    }
  }
  if (it == live_.end()) {
    throw std::logic_error("free of unknown device buffer");
  }
  TraceOp op;
  op.op = "Free";
  op.id = buffer.id;
  op.bytes = buffer.bytes;
  trace_.push_back(op);
  tapes_.emplace_back();
  DeviceBuffer inner = it->inner;
  live_.erase(it);
  inner_.Free(inner);
}

void RecordingDevice::CopyHostToDevice(DeviceBuffer dst, const void* src, size_t bytes) {
  TraceOp op;
  op.op = "CopyHostToDevice";
  op.dst = Cover(dst);
  op.bytes = bytes;
  op.hash = Fnv1a64(bytes == 0 ? nullptr : src, bytes);
  std::vector<unsigned char> tape(bytes);
  if (bytes > 0) {
    std::memcpy(tape.data(), src, bytes);
  }
  trace_.push_back(op);
  tapes_.push_back(std::move(tape));
  inner_.CopyHostToDevice(dst, src, bytes);
}

void RecordingDevice::CopyDeviceToHost(void* dst, DeviceBuffer src, size_t bytes) {
  inner_.CopyDeviceToHost(dst, src, bytes);
  TraceOp op;
  op.op = "CopyDeviceToHost";
  op.src = Cover(src);
  op.bytes = bytes;
  trace_.push_back(op);
  tapes_.emplace_back();
}

void RecordingDevice::MatMul(DeviceBuffer a, DeviceBuffer b, DeviceBuffer c, int64_t m, int64_t k, int64_t n) {
  inner_.MatMul(a, b, c, m, k, n);
  TraceOp op;
  op.op = "MatMul";
  op.m = m;
  op.k = k;
  op.n = n;
  op.a = Cover(a);
  op.b = Cover(b);
  op.c = Cover(c);
  op.a_hash = Peek(a);
  op.b_hash = Peek(b);
  op.c_hash = Peek(c);
  trace_.push_back(op);
  tapes_.emplace_back();
}

void RecordingDevice::Transpose(DeviceBuffer src, DeviceBuffer dst, int64_t rows, int64_t cols) {
  inner_.Transpose(src, dst, rows, cols);
  TraceOp op;
  op.op = "Transpose";
  op.rows = rows;
  op.cols = cols;
  op.src = Cover(src);
  op.dst = Cover(dst);
  op.src_hash = Peek(src);
  op.dst_hash = Peek(dst);
  trace_.push_back(op);
  tapes_.emplace_back();
}

void RecordingDevice::GemmEpilogue(DeviceBuffer c, int64_t m, int64_t n, float alpha, const DeviceBuffer* bias,
                                   const std::vector<int64_t>& bias_shape, float beta, Activation activation) {
  const uint64_t before = Peek(c);
  inner_.GemmEpilogue(c, m, n, alpha, bias, bias_shape, beta, activation);
  TraceOp op;
  op.op = "GemmEpilogue";
  op.m = m;
  op.n = n;
  op.alpha = alpha;
  op.beta = beta;
  op.has_bias = bias != nullptr;
  op.activation = ActivationName(activation);
  op.c = Cover(c);
  if (bias != nullptr) {
    op.bias = Cover(*bias);
  }
  op.hash_before = before;
  op.hash_after = Peek(c);
  trace_.push_back(op);
  tapes_.emplace_back();
}

void RecordingDevice::Embedding(DeviceBuffer ids, int64_t id_offset, DeviceBuffer table, DeviceBuffer out, int64_t seq,
                                int64_t hidden, int64_t vocab) {
  inner_.Embedding(ids, id_offset, table, out, seq, hidden, vocab);
  TraceOp op;
  op.op = "Embedding";
  op.a = Cover(ids);
  op.b = Cover(table);
  op.c = Cover(out);
  op.rows = id_offset;
  op.m = seq;
  op.k = hidden;
  op.n = vocab;
  op.c_hash = Peek(out);
  trace_.push_back(op);
  tapes_.emplace_back();
}

void RecordingDevice::RmsNorm(DeviceBuffer x, DeviceBuffer weight, DeviceBuffer y, int64_t rows, int64_t dim, float eps,
                              int64_t row_offset) {
  inner_.RmsNorm(x, weight, y, rows, dim, eps, row_offset);
  TraceOp op;
  op.op = "RmsNorm";
  op.a = Cover(x);
  op.b = Cover(weight);
  op.c = Cover(y);
  op.rows = rows;
  op.cols = dim;
  op.m = row_offset;
  op.alpha = eps;
  op.c_hash = Peek(y);
  trace_.push_back(op);
  tapes_.emplace_back();
}

void RecordingDevice::Rope(DeviceBuffer states, DeviceBuffer cos_table, DeviceBuffer sin_table, int64_t seq,
                           int64_t n_heads, int64_t head_dim, int64_t position) {
  inner_.Rope(states, cos_table, sin_table, seq, n_heads, head_dim, position);
  TraceOp op;
  op.op = "Rope";
  op.a = Cover(states);
  op.b = Cover(cos_table);
  op.c = Cover(sin_table);
  op.m = seq;
  op.k = n_heads;
  op.n = head_dim;
  op.rows = position;
  op.a_hash = Peek(states);
  trace_.push_back(op);
  tapes_.emplace_back();
}

void RecordingDevice::AppendCache(DeviceBuffer src, DeviceBuffer cache, int64_t seq, int64_t position, int64_t n_heads,
                                  int64_t head_dim) {
  inner_.AppendCache(src, cache, seq, position, n_heads, head_dim);
  TraceOp op;
  op.op = "AppendCache";
  op.src = Cover(src);
  op.dst = Cover(cache);
  op.m = seq;
  op.k = position;
  op.n = n_heads;
  op.rows = head_dim;
  op.dst_hash = Peek(cache);
  trace_.push_back(op);
  tapes_.emplace_back();
}

void RecordingDevice::Attention(DeviceBuffer q, DeviceBuffer k, DeviceBuffer v, DeviceBuffer scores, DeviceBuffer out,
                                int64_t seq, int64_t position, int64_t heads, int64_t kv_heads, int64_t head_dim) {
  inner_.Attention(q, k, v, scores, out, seq, position, heads, kv_heads, head_dim);
  TraceOp op;
  op.op = "Attention";
  op.a = Cover(q);
  op.b = Cover(k);
  op.c = Cover(v);
  op.src = Cover(scores);
  op.dst = Cover(out);
  op.m = seq;
  op.k = position;
  op.n = heads;
  op.rows = kv_heads;
  op.cols = head_dim;
  op.dst_hash = Peek(out);
  trace_.push_back(op);
  tapes_.emplace_back();
}

void RecordingDevice::SiluMul(DeviceBuffer gate, DeviceBuffer up, int64_t count) {
  inner_.SiluMul(gate, up, count);
  TraceOp op;
  op.op = "SiluMul";
  op.a = Cover(gate);
  op.b = Cover(up);
  op.m = count;
  op.a_hash = Peek(gate);
  trace_.push_back(op);
  tapes_.emplace_back();
}

void RecordingDevice::ResidualAdd(DeviceBuffer dest, DeviceBuffer extra, int64_t count) {
  inner_.ResidualAdd(dest, extra, count);
  TraceOp op;
  op.op = "ResidualAdd";
  op.a = Cover(dest);
  op.b = Cover(extra);
  op.m = count;
  op.a_hash = Peek(dest);
  trace_.push_back(op);
  tapes_.emplace_back();
}

void RecordingDevice::SampleToken(DeviceBuffer logits, DeviceBuffer chosen_ids, int64_t chosen_index, int64_t vocab,
                                  float temperature, uint32_t* rng_state, int top_k) {
  const uint32_t rng_before = rng_state == nullptr ? 0u : *rng_state;
  inner_.SampleToken(logits, chosen_ids, chosen_index, vocab, temperature, rng_state, top_k);
  TraceOp op;
  op.op = "SampleToken";
  op.a = Cover(logits);
  op.dst = Cover(chosen_ids);
  op.m = chosen_index;
  op.n = vocab;
  op.k = top_k;
  op.alpha = temperature;
  op.hash = rng_before;
  op.dst_hash = Peek(chosen_ids);
  trace_.push_back(op);
  tapes_.emplace_back();
}

SliceRef RecordingDevice::Cover(DeviceBuffer buffer) const {
  const auto ptr = reinterpret_cast<uintptr_t>(buffer.data);
  for (const Live& live : live_) {
    const auto base = reinterpret_cast<uintptr_t>(live.data);
    if (ptr >= base && buffer.bytes <= live.bytes && ptr - base <= live.bytes - buffer.bytes) {
      return SliceRef{live.id, static_cast<uint64_t>(ptr - base), buffer.bytes};
    }
  }
  return SliceRef{buffer.id, 0, buffer.bytes};
}

uint64_t RecordingDevice::Peek(DeviceBuffer buffer) const {
  if (buffer.bytes == 0) {
    return Fnv1a64(nullptr, 0);
  }
  std::vector<unsigned char> tmp(buffer.bytes);
  inner_.CopyDeviceToHost(tmp.data(), buffer, buffer.bytes);
  return Fnv1a64(tmp.data(), tmp.size());
}

bool RecordingDevice::Replay(Device& target) const {
  try {
    std::unordered_map<uint64_t, DeviceBuffer> allocs;
    auto placed = [&](const SliceRef& ref, DeviceBuffer* out) -> bool {
      auto found = allocs.find(ref.id);
      if (found == allocs.end()) {
        return false;
      }
      *out = Placed(found->second, ref);
      return true;
    };
    auto same = [&](DeviceBuffer buf, uint64_t expect) -> bool {
      std::vector<unsigned char> tmp(buf.bytes);
      if (!tmp.empty()) {
        target.CopyDeviceToHost(tmp.data(), buf, tmp.size());
      }
      return Fnv1a64(tmp.empty() ? nullptr : tmp.data(), tmp.size()) == expect;
    };
    for (size_t index = 0; index < trace_.size(); ++index) {
      const TraceOp& op = trace_[index];
      if (op.op == "Allocate") {
        allocs.emplace(op.id, target.Allocate(static_cast<size_t>(op.bytes)));
      } else if (op.op == "Free") {
        auto it = allocs.find(op.id);
        if (it == allocs.end()) {
          return false;
        }
        target.Free(it->second);
        allocs.erase(it);
      } else if (op.op == "CopyHostToDevice") {
        auto it = allocs.find(op.dst.id);
        if (it == allocs.end() || tapes_[index].size() != op.bytes) {
          return false;
        }
        const void* src = op.bytes == 0 ? nullptr : tapes_[index].data();
        target.CopyHostToDevice(Placed(it->second, op.dst), src, static_cast<size_t>(op.bytes));
      } else if (op.op == "CopyDeviceToHost") {
        auto it = allocs.find(op.src.id);
        if (it == allocs.end()) {
          return false;
        }
        std::vector<unsigned char> tmp(static_cast<size_t>(op.bytes));
        void* dst = op.bytes == 0 ? nullptr : tmp.data();
        target.CopyDeviceToHost(dst, Placed(it->second, op.src), static_cast<size_t>(op.bytes));
      } else if (op.op == "Transpose") {
        auto src = allocs.find(op.src.id);
        auto dst = allocs.find(op.dst.id);
        if (src == allocs.end() || dst == allocs.end()) {
          return false;
        }
        DeviceBuffer src_buf = Placed(src->second, op.src);
        DeviceBuffer dst_buf = Placed(dst->second, op.dst);
        target.Transpose(src_buf, dst_buf, op.rows, op.cols);
        std::vector<unsigned char> tmp(static_cast<size_t>(op.dst.length));
        if (!tmp.empty()) {
          target.CopyDeviceToHost(tmp.data(), dst_buf, tmp.size());
        }
        if (Fnv1a64(tmp.empty() ? nullptr : tmp.data(), tmp.size()) != op.dst_hash) {
          return false;
        }
      } else if (op.op == "MatMul") {
        auto a = allocs.find(op.a.id);
        auto b = allocs.find(op.b.id);
        auto c = allocs.find(op.c.id);
        if (a == allocs.end() || b == allocs.end() || c == allocs.end()) {
          return false;
        }
        DeviceBuffer a_buf = Placed(a->second, op.a);
        DeviceBuffer b_buf = Placed(b->second, op.b);
        DeviceBuffer c_buf = Placed(c->second, op.c);
        target.MatMul(a_buf, b_buf, c_buf, op.m, op.k, op.n);
        std::vector<unsigned char> tmp(static_cast<size_t>(op.c.length));
        if (!tmp.empty()) {
          target.CopyDeviceToHost(tmp.data(), c_buf, tmp.size());
        }
        if (Fnv1a64(tmp.empty() ? nullptr : tmp.data(), tmp.size()) != op.c_hash) {
          return false;
        }
      } else if (op.op == "GemmEpilogue") {
        auto c = allocs.find(op.c.id);
        if (c == allocs.end()) {
          return false;
        }
        DeviceBuffer c_buf = Placed(c->second, op.c);
        DeviceBuffer bias_buf{};
        const DeviceBuffer* bias = nullptr;
        std::vector<int64_t> bias_shape;
        if (op.has_bias) {
          auto found = allocs.find(op.bias.id);
          if (found == allocs.end()) {
            return false;
          }
          bias_buf = Placed(found->second, op.bias);
          bias = &bias_buf;
          bias_shape.push_back(static_cast<int64_t>(op.bias.length / sizeof(float)));
        }
        Activation activation = Activation::None;
        if (op.activation == "relu") {
          activation = Activation::Relu;
        } else if (op.activation == "gelu") {
          activation = Activation::Gelu;
        } else if (op.activation != "none") {
          return false;
        }
        target.GemmEpilogue(c_buf, op.m, op.n, op.alpha, bias, bias_shape, op.beta, activation);
        std::vector<unsigned char> tmp(static_cast<size_t>(op.c.length));
        if (!tmp.empty()) {
          target.CopyDeviceToHost(tmp.data(), c_buf, tmp.size());
        }
        if (Fnv1a64(tmp.empty() ? nullptr : tmp.data(), tmp.size()) != op.hash_after) {
          return false;
        }
      } else if (op.op == "Embedding") {
        DeviceBuffer ids;
        DeviceBuffer table;
        DeviceBuffer out;
        if (!placed(op.a, &ids) || !placed(op.b, &table) || !placed(op.c, &out)) {
          return false;
        }
        target.Embedding(ids, op.rows, table, out, op.m, op.k, op.n);
        if (!same(out, op.c_hash)) {
          return false;
        }
      } else if (op.op == "RmsNorm") {
        DeviceBuffer x;
        DeviceBuffer weight;
        DeviceBuffer y;
        if (!placed(op.a, &x) || !placed(op.b, &weight) || !placed(op.c, &y)) {
          return false;
        }
        target.RmsNorm(x, weight, y, op.rows, op.cols, op.alpha, op.m);
        if (!same(y, op.c_hash)) {
          return false;
        }
      } else if (op.op == "Rope") {
        DeviceBuffer states;
        DeviceBuffer cos_table;
        DeviceBuffer sin_table;
        if (!placed(op.a, &states) || !placed(op.b, &cos_table) || !placed(op.c, &sin_table)) {
          return false;
        }
        target.Rope(states, cos_table, sin_table, op.m, op.k, op.n, op.rows);
        if (!same(states, op.a_hash)) {
          return false;
        }
      } else if (op.op == "AppendCache") {
        DeviceBuffer src;
        DeviceBuffer cache;
        if (!placed(op.src, &src) || !placed(op.dst, &cache)) {
          return false;
        }
        target.AppendCache(src, cache, op.m, op.k, op.n, op.rows);
        if (!same(cache, op.dst_hash)) {
          return false;
        }
      } else if (op.op == "Attention") {
        DeviceBuffer q;
        DeviceBuffer k;
        DeviceBuffer v;
        DeviceBuffer scores;
        DeviceBuffer out;
        if (!placed(op.a, &q) || !placed(op.b, &k) || !placed(op.c, &v) || !placed(op.src, &scores) ||
            !placed(op.dst, &out)) {
          return false;
        }
        target.Attention(q, k, v, scores, out, op.m, op.k, op.n, op.rows, op.cols);
        if (!same(out, op.dst_hash)) {
          return false;
        }
      } else if (op.op == "SiluMul") {
        DeviceBuffer gate;
        DeviceBuffer up;
        if (!placed(op.a, &gate) || !placed(op.b, &up)) {
          return false;
        }
        target.SiluMul(gate, up, op.m);
        if (!same(gate, op.a_hash)) {
          return false;
        }
      } else if (op.op == "ResidualAdd") {
        DeviceBuffer dest;
        DeviceBuffer extra;
        if (!placed(op.a, &dest) || !placed(op.b, &extra)) {
          return false;
        }
        target.ResidualAdd(dest, extra, op.m);
        if (!same(dest, op.a_hash)) {
          return false;
        }
      } else if (op.op == "SampleToken") {
        DeviceBuffer logits;
        DeviceBuffer chosen;
        if (!placed(op.a, &logits) || !placed(op.dst, &chosen)) {
          return false;
        }
        uint32_t rng = static_cast<uint32_t>(op.hash);
        target.SampleToken(logits, chosen, op.m, op.n, op.alpha, &rng, static_cast<int>(op.k));
        if (!same(chosen, op.dst_hash)) {
          return false;
        }
      } else {
        return false;
      }
    }
    return true;
  } catch (const std::exception&) {
    return false;
  }
}

}  // namespace accel
