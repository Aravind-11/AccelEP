#pragma once

#include <stdint.h>

#if defined(_WIN32)
#define ACCEL_SIM_API __declspec(dllexport)
#else
#define ACCEL_SIM_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* One simulated accelerator. Weights uploaded here stay on the device. */
typedef struct AccelSim AccelSim;
typedef struct AccelBuf AccelBuf;

ACCEL_SIM_API AccelSim* accel_sim_create(void);
ACCEL_SIM_API void accel_sim_destroy(AccelSim* sim);

/* Copy `count` float32 values onto the device. The caller keeps `data`. */
ACCEL_SIM_API AccelBuf* accel_sim_upload(AccelSim* sim, const float* data, int64_t count);
ACCEL_SIM_API void accel_sim_release(AccelSim* sim, AccelBuf* buf);

/* Row-major c[m, n] = a[m, k] @ b[k, n]. `b` must already be on the device.
   Returns 0 on success. */
ACCEL_SIM_API int accel_sim_matmul(AccelSim* sim, const float* a, int64_t m, int64_t k, AccelBuf* b,
                                   int64_t n, float* c);

ACCEL_SIM_API void accel_sim_reset_stats(void);
ACCEL_SIM_API int64_t accel_sim_matmuls(void);
ACCEL_SIM_API int64_t accel_sim_bytes_to_device(void);
ACCEL_SIM_API int64_t accel_sim_bytes_to_host(void);

/* Last error from this thread's failed call. Empty when the last call worked. */
ACCEL_SIM_API const char* accel_sim_last_error(void);

/* A Llama held on the simulator. Weights, activations, the KV cache, the
   tokenizer, and the next-token choice stay on the device. The host sends
   prompt text and receives completion text. */
typedef struct AccelLlama AccelLlama;

typedef struct AccelLlamaConfig {
  int32_t vocab;
  int32_t hidden;
  int32_t layers;
  int32_t heads;
  int32_t kv_heads;
  int32_t intermediate;
  int32_t context;
  float eps;
  float rope_theta;
} AccelLlamaConfig;

enum {
  ACCEL_W_EMBED = 1,
  ACCEL_W_FINAL_NORM = 2,
  ACCEL_W_LM_HEAD = 3,
  ACCEL_W_ATTN_NORM = 4,
  ACCEL_W_POST_NORM = 5,
  ACCEL_W_Q = 6,
  ACCEL_W_K = 7,
  ACCEL_W_V = 8,
  ACCEL_W_O = 9,
  ACCEL_W_GATE = 10,
  ACCEL_W_UP = 11,
  ACCEL_W_DOWN = 12
};

ACCEL_SIM_API AccelLlama* accel_llama_create(const AccelLlamaConfig* config);
ACCEL_SIM_API void accel_llama_destroy(AccelLlama* model);
/* Hugging Face layout: matrices are [out, in], norms are [dim] with cols == 1. */
ACCEL_SIM_API int accel_llama_weight(AccelLlama* model, int32_t kind, int32_t layer, const float* data,
                                    int64_t rows, int64_t cols);
/* Load a Hugging Face tokenizer.json. Encode and decode then run inside the simulator. */
ACCEL_SIM_API int accel_llama_tokenizer(AccelLlama* model, const char* path);
/* temperature <= 0 is greedy argmax. Logits never leave the device.
   Copies back the chosen int32 token ids. *out_count is the number written. */
ACCEL_SIM_API int accel_llama_generate(AccelLlama* model, const int32_t* prompt, int32_t prompt_count,
                                       int32_t max_new, int32_t eos_id, float temperature, uint32_t seed,
                                       int32_t* out_ids, int32_t* out_count);
/* Encode `text_len` bytes on the simulator. The buffer may contain NUL.
   *out_count is the number of ids written. */
ACCEL_SIM_API int accel_llama_encode(AccelLlama* model, const char* text, int32_t text_len, int32_t* out_ids,
                                     int32_t out_cap, int32_t* out_count);
/* Decode token ids on the simulator. skip_special omits special tokens.
   *out_len is the utf-8 byte count, with no trailing NUL required. */
ACCEL_SIM_API int accel_llama_decode(AccelLlama* model, const int32_t* ids, int32_t count, int32_t skip_special,
                                     char* out_text, int32_t out_cap, int32_t* out_len);
/* Encode the prompt, sample, and decode on the simulator. `prompt_len` is the
   byte count and may include NUL. Copies back the completion text and the
   chosen token ids. Logits stay on the device. */
ACCEL_SIM_API int accel_llama_complete(AccelLlama* model, const char* prompt, int32_t prompt_len, int32_t max_new,
                                       int32_t eos_id, float temperature, uint32_t seed, char* out_text,
                                       int32_t out_text_cap, int32_t* out_text_len, int32_t* out_ids,
                                       int32_t out_ids_cap, int32_t* out_count, int32_t* prompt_tokens);
ACCEL_SIM_API void accel_llama_reset(AccelLlama* model);
ACCEL_SIM_API const char* accel_llama_last_error(void);

#ifdef __cplusplus
}
#endif
