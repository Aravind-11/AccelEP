# AccelEP

AccelEP is an ONNX Runtime 1.31 plugin for matrix multiplies, and a CPU simulator that serves a small Llama. `SimDevice` is a heap in this process. The model calls each kernel by name. The simulator runs on the CPU. A hardware backend can replace the kernels and keep the same calls.

The checked model is [SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct). You can try the chat at <https://accelep.vercel.app/>. After the site has been idle, the first reply can take a few seconds while the model loads.

The weight file is bfloat16, about 257 MB, and is left out of git. The loader converts every tensor to float32. A full-size Llama does not fit in 8 GB. The server keeps 512 tokens of context. The checkpoint allows 8192 positions.

## What a chat request does

The browser sends the message and shows the reply. Python writes the chat string and answers HTTP. The C++ library tokenizes, runs the forward pass, samples, and turns the ids back into text.

1. **Chat string.** When the first message is not a system turn, SmolLM2 gets this text, with a newline after `<|im_end|>`:

   ```text
   <|im_start|>system
   You are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>
   ```

   Each message is then `<|im_start|>`, the role, a newline, the text, `<|im_end|>`, and a newline. The string ends with `<|im_start|>assistant` and a newline. `POST /v1/completions` skips this and completes the raw prompt.

2. **Tokenize.** `src/bpe.cc` turns the string into token ids. The vocabulary is missing 21 GPT-2 byte symbols. Those bytes are dropped, and the rest of the piece is merged. Encode and decode match the `tokenizers` package.

3. **Copy the prompt.** The ids are copied onto the simulator. The weights are already there. This is the copy counted for the request.

4. **Forward, then sample.** One forward runs on the whole prompt. `SampleToken` writes the next id on that heap. The logit row stays there.

5. **Stop check.** The host reads that id through the buffer pointer. Id 2 is `<|im_end|>`. On that id, or when the context is full, the loop stops. Otherwise the next forward runs on that single id, and sampling repeats.

6. **Copy the reply.** The ids are copied back. Decode skips special tokens, so id 2 stays in the id list and drops out of the text. The text is copied back too.

Greedy SmolLM2 chat for "The capital of France is", 8 new tokens:

| | |
| --- | --- |
| text | The capital of France is Paris. |
| ids | `[504, 3575, 282, 4649, 314, 7042, 30, 2]` |
| copied back | 63 bytes: 31 of UTF-8, plus 32 of ids |
| copied in | 140 bytes: 35 prompt ids |
| forwards | 8 |
| matrix multiplies | 1,688 |

The 8th id is the end token, so the loop stops before another forward. Each forward does 211 matrix multiplies.

The hosted chat for "whats 2+2", also 8 tokens at temperature 0, returned "2 + 2 = 4". The ids were `[34, 1232, 216, 34, 446, 216, 36, 2]`. That call copied 144 bytes in (36 prompt ids) and 41 bytes back (9 of text, plus 32 of ids), and ran the same 1,688 matrix multiplies.

## One forward pass

SmolLM2 is hidden 576, 30 layers, 9 heads, 3 key/value heads, head size 64, intermediate 1536, vocab 49152. `rope_theta` is 100000. The MLP is SiLU. The embedding and the vocabulary matrix share one weight.

At startup the host builds the RoPE cosine and sine tables for all 512 positions and uploads them. Embedding and RMSNorm weights keep the file layout. Every matrix, including the vocabulary matrix, is transposed on the host and then uploaded.

`LlamaModel::Forward` only calls `SimDevice` methods.

1. Embedding of the token ids.
2. For each of the 30 layers, in order:
   - RMSNorm
   - Q, K, and V matrix multiplies
   - RoPE on Q and K. Each head is rotated as two halves.
   - Append the new K and V to the cache
   - Attention
   - Output matrix multiply
   - Residual add
   - RMSNorm
   - Gate and up matrix multiplies
   - SiLU on the gate, multiplied by the up projection
   - Down matrix multiply
   - Residual add
3. RMSNorm of the last position only.
4. Vocabulary matrix multiply. One row of 49,152 logits.

Attention is causal. Query head `h` reads key/value head `h / 3`. Scores are scaled by `1/sqrt(64)`. A token sees the cache only through its own position.

The 211 matrix multiplies are seven per layer, times 30, plus the vocabulary matrix.

`SampleToken` runs after the forward. Temperature 0, and any temperature that is not positive, keeps the largest logit. A tie keeps the earlier id. A positive temperature draws from the softmax with an xorshift generator. Seed 0 starts at `0x6d2b79f5`.

## Serve SmolLM2

Put these files in `models/`. They are left out of the git tree.

- `config.json`
- `tokenizer.json`
- `tokenizer_config.json`
- `model.safetensors`

```bash
make simlib
python3 -m venv .venv
.venv/bin/pip install numpy tokenizers
PYTHONPATH=. .venv/bin/python -m serve --model models --port 8000
```

The server listens on `127.0.0.1:8000`. One lock wraps each completion.

| Route | What it does |
| --- | --- |
| `GET /health` | Returns `ok`, plus the layer count, hidden size, and vocab size. |
| `POST /v1/completions` | Completes a raw prompt. |
| `POST /v1/chat/completions` | Builds the SmolLM2 chat string, then completes it. |

## Tests

On this Mac, run:

```bash
make sim-test
```

That needs the `models/` files above and a `.venv` with `numpy` and `tokenizers`. Three programs run, in order. A good run prints one line from each.

1. `build/test_sim_kernels` prints `kernel tests passed`.
   Eight kernels, each on a few hand values: embedding, RMSNorm, RoPE, the cache append, attention, SiLU, the residual add, and sampling. Logits `[1, 3, 3]` at temperature 0 return id 1.

2. `build/test_runtime` prints `runtime tests passed`.
   Twenty-four checks of the matrix runtime used by the ONNX plugin: MatMul and Gemm shapes, a fused Relu, a residual, transposes, allocation, a recorded call trace, and the copy counters.

3. `tests/test_serve.py` prints `serve tests passed`.
   Five checks:
   - Encode and decode match `tokenizers` on 24 strings, including bytes missing from the vocabulary.
   - One simulator matrix multiply matches NumPy.
   - The Python RMSNorm and RoPE helpers match their formulas. The serving path uses the C++ kernels from step 1.
   - Greedy chat for "The capital of France is" contains "Paris". Bytes copied back equal the UTF-8 length plus 4 per completion id. Bytes copied in equal 4 per prompt id. One token at temperature 0.8 follows the same rule.
   - `GET /health`, `POST /v1/chat/completions`, and `POST /v1/completions` return 200. The chat reply contains Paris.

This Mac printed all three pass lines. The France reply in the table above is the one that run produced. When `models/model.safetensors` is absent, step 3 skips the model and HTTP checks and still prints `serve tests passed`.

### A second checkpoint

[SmolLM-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM-135M-Instruct) also ran, from `models/SmolLM-135M-Instruct/`. The shape matches SmolLM2. Its `rope_theta` is 10000. Its own chat template has no system turn.

Greedy completion of "The capital of France is", with that template, returned "The capital of France is Paris, the largest city in France and the capital of France. It is located in the north". The prompt was 14 tokens (56 bytes copied in). The reply was 24 ids and 208 bytes copied back: 112 of text plus 96 of ids. The loop ran 25 forwards, which is 5,275 matrix multiplies. The last forward is the one after the 24th token, because the end token had not appeared.

Six more greedy replies, each stopped at 32 tokens:

| Question | Reply |
| --- | --- |
| What is 2 + 2? | The number 2 + 2! is a fascinating concept that has puzzled mathematicians and scientists for centuries. Here's a breakdown of the formula: |
| Who wrote Romeo and Juliet? | The answer to this question is a bit complex, as Romeo and Juliet is a play written by William Shakespeare, but it's not a single author. However, |
| What color is the sky on a clear day? | The sky on a clear day is indeed a beautiful and vibrant color, with hues of blue, green, and sometimes even purple. The sky is a reflection of |
| Name the largest planet in the solar system. | The largest planet in the solar system is Jupiter, with a diameter of approximately 142,948 kilometers (88,972 miles |
| What is the boiling point of water in Celsius? | The boiling point of water is a temperature at which the vapor pressure of a liquid is equal to the pressure of the surrounding gas. In other words, it's |
| Say hello in one word. | What a great question! The phrase "say hello" is a universal greeting that has been used in various contexts, including greetings, introductions, and farewell |

Jupiter is named directly. The other replies talk around the question.

### What these steps leave open

The kernel checks use one or two elements. The model uses head size 64 and grouped-query attention. The France check looks for one word in one reply. A logit match against an independent forward, and a full prefill compared with one token at a time, are still open.

The loader accepts a Llama with SiLU, no RoPE scaling, and a Hugging Face BPE tokenizer whose pretokenizer is `Digits` then `ByteLevel`, with a null normalizer, a null post-processor, and byte fallback off. `JackFram/llama-68m` has no `tokenizer.json`. `ModelCloud/tinyllama-15M-stories` sets byte fallback and an `unk_token`, so this loader rejects it.

## ONNX plugin

The plugin takes float32 MatMul, 2D Gemm, and `com.microsoft.FusedMatMul` when the shapes are static. One leading MatMul batch dimension may still be unknown when the graph is compiled. Softmax, layer norm, and every other op stay on the CPU execution provider.

A following Relu stays on the device. So does an opset-20 Gelu when `approximate` is absent or `"none"`. The next matrix in that group reads the activated values from the device. A lone MatMul can add one residual of the same shape. A constant-weight projection leaves that residual on the CPU when the Add feeds more than one node.

A constant int8 `DequantizeLinear` that feeds a float MatMul is uploaded as float32. Other quantizer shapes stay on the CPU.

| Op | Domain | What is accepted |
| --- | --- | --- |
| MatMul | `ai.onnx` | Rank 2 or higher. Batch dimensions broadcast like NumPy. |
| Gemm | `ai.onnx` | Strictly 2D. `transA` and `transB` are 0 or 1. An optional bias must broadcast onto `[M, N]`. |
| FusedMatMul | `com.microsoft` | Rank 2 or higher. `transA` and `transB` are 0 or 1. `transBatchA` and `transBatchB` stay 0. |

`MatMul` computes row-major `C = A @ B` in float32. Each product is a multiply followed by an add. Constant weights are uploaded once, when the node is compiled. Each run copies the activations that are still on the host, and copies that group's result back.

### Build the plugin

The plugin shared library uses `-Wl,-z,defs`, so build it with GNU ld. Ubuntu 24.04 with `g++`, `make`, and `python3-venv` is enough.

```bash
make venv
make test
```

`make venv` installs `onnxruntime==1.31.0` and `onnx==1.21.0` into `/tmp/accel-venv`. A good run prints `runtime tests passed` and `ep tests passed`. `make bench` is separate. `make clean` removes `build/`.

On this Mac the linker rejects `-z`, so `make build/libaccel_ep.so` stops there. `make simlib` still builds `build/libaccel_sim.dylib` for the Llama server. Run `make clean` before a Linux plugin build, so the Mac objects are not reused.

```python
import onnxruntime as ort

ort.register_execution_provider_library("AccelEP", "build/libaccel_ep.so")
devices = [device for device in ort.get_ep_devices() if device.ep_name == "AccelEP"]

options = ort.SessionOptions()
options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
options.add_provider_for_devices(devices, {})
session = ort.InferenceSession(model_bytes, options)
```

The ONNX decoder-block test has no paged KV cache, continuous batching, CUDA, tokens/s figure, causal mask, or RoPE. Its attention scores stay on the CPU. This simulator can be slower than ONNX Runtime's own CPU kernels.

## Layout

```text
include/accel/      Device interface and kernel headers
src/sim_device.cc   CPU simulator
src/sim_*.cc        Embedding, RMSNorm, RoPE, cache, attention, SiLU, add, sample
src/llama_model.cc  Llama forward. It only calls device kernels.
src/bpe.cc          Tokenizer encode and decode
src/ep.cc           ONNX Runtime plugin
serve/              HTTP server
tests/              C++ and Python tests
third_party/        ONNX Runtime 1.31.0 public headers
```

The headers under `third_party/onnxruntime` are the public C/C++ API from ONNX Runtime release v1.31.0. Microsoft owns that code and ships it under the MIT License. See `third_party/onnxruntime/NOTICE`.
