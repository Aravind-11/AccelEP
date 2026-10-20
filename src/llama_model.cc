#include "accel/sim_api.h"

#include "accel/bpe.h"
#include "accel/device.h"

#include <cmath>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

thread_local std::string g_llama_error;

void Fail(const std::string& message) { g_llama_error = message; }

void Clear() { g_llama_error.clear(); }

struct Layer {
  accel::OwnedBuffer attn_norm;
  accel::OwnedBuffer post_norm;
  accel::OwnedBuffer q;
  accel::OwnedBuffer k;
  accel::OwnedBuffer v;
  accel::OwnedBuffer o;
  accel::OwnedBuffer gate;
  accel::OwnedBuffer up;
  accel::OwnedBuffer down;
};

struct LlamaModel {
  AccelLlamaConfig config{};
  std::unique_ptr<accel::Device> device;
  std::mutex mu;
  int32_t head_dim = 0;
  int32_t filled = 0;
  accel::Bpe bpe;
  accel::OwnedBuffer embed;
  accel::OwnedBuffer final_norm;
  accel::OwnedBuffer lm_head;
  accel::OwnedBuffer rope_cos;
  accel::OwnedBuffer rope_sin;
  accel::OwnedBuffer hidden;
  accel::OwnedBuffer normed;
  accel::OwnedBuffer query;
  accel::OwnedBuffer key;
  accel::OwnedBuffer value;
  accel::OwnedBuffer mixed;
  accel::OwnedBuffer delta;
  accel::OwnedBuffer gate;
  accel::OwnedBuffer up;
  accel::OwnedBuffer logits;
  accel::OwnedBuffer score_buf;
  accel::OwnedBuffer chosen;
  accel::OwnedBuffer prompt_ids;
  accel::OwnedBuffer text_out;
  std::vector<Layer> layers;
  std::vector<accel::OwnedBuffer> k_cache;
  std::vector<accel::OwnedBuffer> v_cache;

  accel::OwnedBuffer Upload(const float* data, size_t count) {
    accel::OwnedBuffer buf;
    buf.device = device.get();
    buf.buffer = device->Allocate(count * sizeof(float));
    if (count > 0) {
      device->CopyHostToDevice(buf.buffer, data, count * sizeof(float));
    }
    return buf;
  }

  accel::OwnedBuffer UploadTransposed(const float* data, int64_t rows, int64_t cols) {
    std::vector<float> transposed(static_cast<size_t>(rows) * static_cast<size_t>(cols));
    for (int64_t out = 0; out < rows; ++out) {
      for (int64_t in = 0; in < cols; ++in) {
        transposed[static_cast<size_t>(in) * static_cast<size_t>(rows) + static_cast<size_t>(out)] =
            data[static_cast<size_t>(out) * static_cast<size_t>(cols) + static_cast<size_t>(in)];
      }
    }
    return Upload(transposed.data(), transposed.size());
  }

  bool Ready() const {
    if (embed.buffer.data == nullptr || final_norm.buffer.data == nullptr || lm_head.buffer.data == nullptr) {
      return false;
    }
    for (const Layer& layer : layers) {
      if (layer.q.buffer.data == nullptr || layer.k.buffer.data == nullptr || layer.v.buffer.data == nullptr ||
          layer.o.buffer.data == nullptr || layer.gate.buffer.data == nullptr || layer.up.buffer.data == nullptr ||
          layer.down.buffer.data == nullptr || layer.attn_norm.buffer.data == nullptr ||
          layer.post_norm.buffer.data == nullptr) {
        return false;
      }
    }
    return true;
  }

  void Generate(const int32_t* prompt, int32_t prompt_count, int32_t max_new, int32_t eos_id, float temperature,
                uint32_t seed, int32_t* out_ids, int32_t* out_count) {
    filled = 0;
    if (prompt == nullptr || prompt_count <= 0 || prompt_count > config.context) {
      throw std::invalid_argument("forward exceeds context");
    }
    device->CopyHostToDevice(prompt_ids.buffer, prompt, static_cast<size_t>(prompt_count) * sizeof(int32_t));
    Forward(prompt_ids.buffer, prompt_count, 0);
    filled = prompt_count;
    uint32_t rng = seed == 0 ? 0x6d2b79f5u : seed;
    int32_t produced = 0;
    for (int32_t n = 0; n < max_new; ++n) {
      if (produced >= config.context) {
        throw std::invalid_argument("generation exceeded the token buffer");
      }
      device->SampleToken(logits.buffer, chosen.buffer, produced, config.vocab, temperature, &rng);
      const int32_t id = static_cast<const int32_t*>(chosen.buffer.data)[produced];
      produced += 1;
      if (id == eos_id || filled >= config.context) {
        break;
      }
      Forward(chosen.buffer, 1, static_cast<int64_t>(produced) - 1);
      filled += 1;
    }
    device->CopyDeviceToHost(out_ids, chosen.buffer, static_cast<size_t>(produced) * sizeof(int32_t));
    *out_count = produced;
  }

  void CopyOutText(const std::string& text, char* out_text, int32_t cap, int32_t* out_len) {
    if (out_text == nullptr || out_len == nullptr || cap < 0 || static_cast<int64_t>(text.size()) > cap) {
      throw std::invalid_argument("completion text buffer is too small");
    }
    *out_len = static_cast<int32_t>(text.size());
    if (text.empty()) {
      return;
    }
    if (text_out.buffer.data == nullptr || text_out.buffer.bytes < text.size()) {
      text_out.Reset();
      text_out.device = device.get();
      text_out.buffer = device->Allocate(text.size());
    }
    std::memcpy(text_out.buffer.data, text.data(), text.size());
    device->CopyDeviceToHost(out_text, text_out.buffer, text.size());
  }

  void Complete(const char* prompt, int32_t prompt_len, int32_t max_new, int32_t eos_id, float temperature,
                uint32_t seed, char* out_text, int32_t out_text_cap, int32_t* out_text_len, int32_t* out_ids,
                int32_t out_ids_cap, int32_t* out_count, int32_t* prompt_tokens) {
    if (!bpe.loaded()) {
      throw std::invalid_argument("tokenizer is not loaded");
    }
    if (prompt_len < 0 || (prompt_len > 0 && prompt == nullptr) || out_ids == nullptr || out_count == nullptr ||
        prompt_tokens == nullptr || out_ids_cap < max_new) {
      throw std::invalid_argument("invalid complete");
    }
    const std::string bytes(prompt_len == 0 ? std::string() : std::string(prompt, static_cast<size_t>(prompt_len)));
    const std::vector<int32_t> ids = bpe.Encode(bytes);
    if (ids.empty()) {
      throw std::invalid_argument("prompt encoded to no tokens");
    }
    if (static_cast<int32_t>(ids.size()) > config.context) {
      throw std::invalid_argument("prompt exceeds context");
    }
    Generate(ids.data(), static_cast<int32_t>(ids.size()), max_new, eos_id, temperature, seed, out_ids, out_count);
    const std::string text = bpe.Decode(static_cast<const int32_t*>(chosen.buffer.data), *out_count, true);
    CopyOutText(text, out_text, out_text_cap, out_text_len);
    *prompt_tokens = static_cast<int32_t>(ids.size());
  }

  void Forward(accel::DeviceBuffer token_ids, int32_t seq, int64_t id_offset) {
    if (token_ids.data == nullptr || seq <= 0 || filled < 0 || seq > config.context || filled > config.context - seq) {
      throw std::invalid_argument("forward exceeds context");
    }
    const int32_t hidden_size = config.hidden;
    const int32_t position = filled;
    const int64_t n_hidden = static_cast<int64_t>(seq) * hidden_size;
    const int64_t n_inter = static_cast<int64_t>(seq) * config.intermediate;
    device->Embedding(token_ids, id_offset, embed.buffer, hidden.buffer, seq, hidden_size, config.vocab);
    for (int32_t layer_index = 0; layer_index < config.layers; ++layer_index) {
      Layer& layer = layers[static_cast<size_t>(layer_index)];
      const accel::DeviceBuffer k_buf = k_cache[static_cast<size_t>(layer_index)].buffer;
      const accel::DeviceBuffer v_buf = v_cache[static_cast<size_t>(layer_index)].buffer;
      device->RmsNorm(hidden.buffer, layer.attn_norm.buffer, normed.buffer, seq, hidden_size, config.eps, 0);
      device->MatMul(normed.buffer, layer.q.buffer, query.buffer, seq, hidden_size,
                     static_cast<int64_t>(config.heads) * head_dim);
      device->MatMul(normed.buffer, layer.k.buffer, key.buffer, seq, hidden_size,
                     static_cast<int64_t>(config.kv_heads) * head_dim);
      device->MatMul(normed.buffer, layer.v.buffer, value.buffer, seq, hidden_size,
                     static_cast<int64_t>(config.kv_heads) * head_dim);
      device->Rope(query.buffer, rope_cos.buffer, rope_sin.buffer, seq, config.heads, head_dim, position);
      device->Rope(key.buffer, rope_cos.buffer, rope_sin.buffer, seq, config.kv_heads, head_dim, position);
      device->AppendCache(key.buffer, k_buf, seq, position, config.kv_heads, head_dim);
      device->AppendCache(value.buffer, v_buf, seq, position, config.kv_heads, head_dim);
      device->Attention(query.buffer, k_buf, v_buf, score_buf.buffer, mixed.buffer, seq, position, config.heads,
                        config.kv_heads, head_dim);
      device->MatMul(mixed.buffer, layer.o.buffer, delta.buffer, seq, hidden_size, hidden_size);
      device->ResidualAdd(hidden.buffer, delta.buffer, n_hidden);
      device->RmsNorm(hidden.buffer, layer.post_norm.buffer, normed.buffer, seq, hidden_size, config.eps, 0);
      device->MatMul(normed.buffer, layer.gate.buffer, gate.buffer, seq, hidden_size, config.intermediate);
      device->MatMul(normed.buffer, layer.up.buffer, up.buffer, seq, hidden_size, config.intermediate);
      device->SiluMul(gate.buffer, up.buffer, n_inter);
      device->MatMul(gate.buffer, layer.down.buffer, delta.buffer, seq, config.intermediate, hidden_size);
      device->ResidualAdd(hidden.buffer, delta.buffer, n_hidden);
    }
    device->RmsNorm(hidden.buffer, final_norm.buffer, normed.buffer, 1, hidden_size, config.eps,
                    static_cast<int64_t>(seq) - 1);
    device->MatMul(normed.buffer, lm_head.buffer, logits.buffer, 1, hidden_size, config.vocab);
  }
};

accel::OwnedBuffer* WeightSlot(LlamaModel* model, int32_t kind, int32_t layer) {
  if (kind == ACCEL_W_EMBED) {
    return &model->embed;
  }
  if (kind == ACCEL_W_FINAL_NORM) {
    return &model->final_norm;
  }
  if (kind == ACCEL_W_LM_HEAD) {
    return &model->lm_head;
  }
  if (layer < 0 || layer >= model->config.layers) {
    return nullptr;
  }
  Layer& block = model->layers[static_cast<size_t>(layer)];
  switch (kind) {
    case ACCEL_W_ATTN_NORM:
      return &block.attn_norm;
    case ACCEL_W_POST_NORM:
      return &block.post_norm;
    case ACCEL_W_Q:
      return &block.q;
    case ACCEL_W_K:
      return &block.k;
    case ACCEL_W_V:
      return &block.v;
    case ACCEL_W_O:
      return &block.o;
    case ACCEL_W_GATE:
      return &block.gate;
    case ACCEL_W_UP:
      return &block.up;
    case ACCEL_W_DOWN:
      return &block.down;
    default:
      return nullptr;
  }
}

bool ShapeOk(const LlamaModel* model, int32_t kind, int64_t rows, int64_t cols) {
  const int32_t hidden = model->config.hidden;
  const int32_t q_out = model->config.heads * model->head_dim;
  const int32_t k_out = model->config.kv_heads * model->head_dim;
  switch (kind) {
    case ACCEL_W_EMBED:
    case ACCEL_W_LM_HEAD:
      return rows == model->config.vocab && cols == hidden;
    case ACCEL_W_FINAL_NORM:
    case ACCEL_W_ATTN_NORM:
    case ACCEL_W_POST_NORM:
      return rows == hidden && cols == 1;
    case ACCEL_W_Q:
      return rows == q_out && cols == hidden;
    case ACCEL_W_K:
    case ACCEL_W_V:
      return rows == k_out && cols == hidden;
    case ACCEL_W_O:
    case ACCEL_W_DOWN:
      return rows == hidden && cols == (kind == ACCEL_W_O ? q_out : model->config.intermediate);
    case ACCEL_W_GATE:
    case ACCEL_W_UP:
      return rows == model->config.intermediate && cols == hidden;
    default:
      return false;
  }
}

}  // namespace

AccelLlama* accel_llama_create(const AccelLlamaConfig* config) {
  if (config == nullptr || config->vocab <= 0 || config->hidden <= 0 || config->layers <= 0 || config->heads <= 0 ||
      config->kv_heads <= 0 || config->intermediate <= 0 || config->context <= 0 || config->hidden % config->heads != 0 ||
      config->heads % config->kv_heads != 0) {
    Fail("invalid llama config");
    return nullptr;
  }
  try {
    auto model = std::unique_ptr<LlamaModel>(new LlamaModel());
    model->config = *config;
    model->head_dim = config->hidden / config->heads;
    model->device = accel::CreateSimDevice();
    model->layers.resize(static_cast<size_t>(config->layers));
    const int32_t dim = model->head_dim;
    const size_t tokens = static_cast<size_t>(config->context);
    std::vector<float> cos(tokens * static_cast<size_t>(dim));
    std::vector<float> sin(tokens * static_cast<size_t>(dim));
    const int32_t half = dim / 2;
    for (int32_t pos = 0; pos < config->context; ++pos) {
      for (int32_t i = 0; i < half; ++i) {
        const double inv = 1.0 / std::pow(static_cast<double>(config->rope_theta), (2.0 * i) / static_cast<double>(dim));
        const double angle = static_cast<double>(pos) * inv;
        const float c = static_cast<float>(std::cos(angle));
        const float s = static_cast<float>(std::sin(angle));
        cos[static_cast<size_t>(pos) * dim + i] = c;
        cos[static_cast<size_t>(pos) * dim + half + i] = c;
        sin[static_cast<size_t>(pos) * dim + i] = s;
        sin[static_cast<size_t>(pos) * dim + half + i] = s;
      }
    }
    model->rope_cos = model->Upload(cos.data(), cos.size());
    model->rope_sin = model->Upload(sin.data(), sin.size());
    const size_t hidden_n = tokens * static_cast<size_t>(config->hidden);
    const size_t q_n = tokens * static_cast<size_t>(config->heads) * dim;
    const size_t k_n = tokens * static_cast<size_t>(config->kv_heads) * dim;
    const size_t inter_n = tokens * static_cast<size_t>(config->intermediate);
    auto alloc = [&](size_t count) {
      accel::OwnedBuffer buf;
      buf.device = model->device.get();
      buf.buffer = model->device->Allocate(count * sizeof(float));
      return buf;
    };
    model->hidden = alloc(hidden_n);
    model->normed = alloc(hidden_n);
    model->query = alloc(q_n);
    model->key = alloc(k_n);
    model->value = alloc(k_n);
    model->mixed = alloc(q_n);
    model->delta = alloc(hidden_n);
    model->gate = alloc(inter_n);
    model->up = alloc(inter_n);
    model->logits = alloc(static_cast<size_t>(config->vocab));
    model->score_buf = alloc(static_cast<size_t>(config->context));
    {
      accel::OwnedBuffer ids;
      ids.device = model->device.get();
      ids.buffer = model->device->Allocate(static_cast<size_t>(config->context) * sizeof(int32_t));
      model->chosen = std::move(ids);
    }
    {
      accel::OwnedBuffer ids;
      ids.device = model->device.get();
      ids.buffer = model->device->Allocate(static_cast<size_t>(config->context) * sizeof(int32_t));
      model->prompt_ids = std::move(ids);
    }
    model->k_cache.reserve(static_cast<size_t>(config->layers));
    model->v_cache.reserve(static_cast<size_t>(config->layers));
    for (int32_t layer = 0; layer < config->layers; ++layer) {
      model->k_cache.push_back(alloc(k_n));
      model->v_cache.push_back(alloc(k_n));
    }
    Clear();
    return reinterpret_cast<AccelLlama*>(model.release());
  } catch (const std::exception& ex) {
    Fail(ex.what());
    return nullptr;
  }
}

void accel_llama_destroy(AccelLlama* model) { delete reinterpret_cast<LlamaModel*>(model); }

int accel_llama_weight(AccelLlama* handle, int32_t kind, int32_t layer, const float* data, int64_t rows, int64_t cols) {
  LlamaModel* model = reinterpret_cast<LlamaModel*>(handle);
  if (model == nullptr || data == nullptr || rows <= 0 || cols <= 0 || !ShapeOk(model, kind, rows, cols)) {
    Fail("invalid weight");
    return 1;
  }
  try {
    std::lock_guard<std::mutex> lock(model->mu);
    accel::OwnedBuffer* slot = WeightSlot(model, kind, layer);
    if (slot == nullptr) {
      Fail("unknown weight");
      return 1;
    }
    const bool norm = kind == ACCEL_W_FINAL_NORM || kind == ACCEL_W_ATTN_NORM || kind == ACCEL_W_POST_NORM;
    const bool gather = kind == ACCEL_W_EMBED;
    *slot = (norm || gather) ? model->Upload(data, static_cast<size_t>(rows) * static_cast<size_t>(cols))
                             : model->UploadTransposed(data, rows, cols);
    Clear();
    return 0;
  } catch (const std::exception& ex) {
    Fail(ex.what());
    return 1;
  }
}

int accel_llama_generate(AccelLlama* handle, const int32_t* prompt, int32_t prompt_count, int32_t max_new,
                         int32_t eos_id, float temperature, uint32_t seed, int32_t* out_ids, int32_t* out_count) {
  LlamaModel* model = reinterpret_cast<LlamaModel*>(handle);
  if (model == nullptr || prompt == nullptr || out_ids == nullptr || out_count == nullptr || prompt_count <= 0 ||
      max_new < 0) {
    Fail("invalid generate");
    return 1;
  }
  try {
    std::lock_guard<std::mutex> lock(model->mu);
    if (!model->Ready()) {
      Fail("llama weights are incomplete");
      return 1;
    }
    model->Generate(prompt, prompt_count, max_new, eos_id, temperature, seed, out_ids, out_count);
    Clear();
    return 0;
  } catch (const std::exception& ex) {
    Fail(ex.what());
    return 1;
  }
}

int accel_llama_tokenizer(AccelLlama* handle, const char* path) {
  LlamaModel* model = reinterpret_cast<LlamaModel*>(handle);
  if (model == nullptr || path == nullptr || path[0] == '\0') {
    Fail("invalid tokenizer");
    return 1;
  }
  try {
    std::lock_guard<std::mutex> lock(model->mu);
    model->bpe.Load(path);
    Clear();
    return 0;
  } catch (const std::exception& ex) {
    Fail(ex.what());
    return 1;
  }
}

int accel_llama_encode(AccelLlama* handle, const char* text, int32_t text_len, int32_t* out_ids, int32_t out_cap,
                       int32_t* out_count) {
  LlamaModel* model = reinterpret_cast<LlamaModel*>(handle);
  if (model == nullptr || text_len < 0 || (text_len > 0 && text == nullptr) || out_ids == nullptr ||
      out_count == nullptr || out_cap < 0) {
    Fail("invalid encode");
    return 1;
  }
  try {
    std::lock_guard<std::mutex> lock(model->mu);
    const std::string bytes(text_len == 0 ? std::string() : std::string(text, static_cast<size_t>(text_len)));
    const std::vector<int32_t> ids = model->bpe.Encode(bytes);
    if (static_cast<int32_t>(ids.size()) > out_cap) {
      Fail("encode buffer is too small");
      return 1;
    }
    if (!ids.empty()) {
      std::memcpy(out_ids, ids.data(), ids.size() * sizeof(int32_t));
    }
    *out_count = static_cast<int32_t>(ids.size());
    Clear();
    return 0;
  } catch (const std::exception& ex) {
    Fail(ex.what());
    return 1;
  }
}

int accel_llama_decode(AccelLlama* handle, const int32_t* ids, int32_t count, int32_t skip_special, char* out_text,
                       int32_t out_cap, int32_t* out_len) {
  LlamaModel* model = reinterpret_cast<LlamaModel*>(handle);
  if (model == nullptr || out_text == nullptr || out_len == nullptr || count < 0 || (count > 0 && ids == nullptr) ||
      out_cap < 0) {
    Fail("invalid decode");
    return 1;
  }
  try {
    std::lock_guard<std::mutex> lock(model->mu);
    const std::string text = model->bpe.Decode(ids, count, skip_special != 0);
    if (static_cast<int64_t>(text.size()) > out_cap) {
      Fail("decode buffer is too small");
      return 1;
    }
    if (!text.empty()) {
      std::memcpy(out_text, text.data(), text.size());
    }
    *out_len = static_cast<int32_t>(text.size());
    Clear();
    return 0;
  } catch (const std::exception& ex) {
    Fail(ex.what());
    return 1;
  }
}

int accel_llama_complete(AccelLlama* handle, const char* prompt, int32_t prompt_len, int32_t max_new, int32_t eos_id,
                         float temperature, uint32_t seed, char* out_text, int32_t out_text_cap, int32_t* out_text_len,
                         int32_t* out_ids, int32_t out_ids_cap, int32_t* out_count, int32_t* prompt_tokens) {
  LlamaModel* model = reinterpret_cast<LlamaModel*>(handle);
  if (model == nullptr || prompt_len < 0 || (prompt_len > 0 && prompt == nullptr) || max_new < 0) {
    Fail("invalid complete");
    return 1;
  }
  try {
    std::lock_guard<std::mutex> lock(model->mu);
    if (!model->Ready()) {
      Fail("llama weights are incomplete");
      return 1;
    }
    model->Complete(prompt, prompt_len, max_new, eos_id, temperature, seed, out_text, out_text_cap, out_text_len,
                    out_ids, out_ids_cap, out_count, prompt_tokens);
    Clear();
    return 0;
  } catch (const std::exception& ex) {
    Fail(ex.what());
    return 1;
  }
}

void accel_llama_reset(AccelLlama* handle) {
  LlamaModel* model = reinterpret_cast<LlamaModel*>(handle);
  if (model == nullptr) {
    return;
  }
  std::lock_guard<std::mutex> lock(model->mu);
  model->filled = 0;
}

const char* accel_llama_last_error(void) { return g_llama_error.c_str(); }
