# AccelEP

AccelEP is an ONNX Runtime 1.31 execution provider for a pretend accelerator, and a CPU simulator that serves a small Llama. The primary objective is to check that software before the chip exists, or when the chip is not on this machine.

You can try the chat at <https://accelep.vercel.app/>. After the site has been idle, the first reply can take a few seconds while the model loads.

## Primary objective

A model is a recipe: a list of steps such as "multiply these matrices, add, apply ReLU." ONNX is the file format for that recipe. ONNX Runtime reads the file and carries out each step. An accelerator is hardware that takes certain steps, mostly large matrix multiplies. An execution provider is how Runtime hands work to one piece of hardware. Runtime asks each provider which steps it can take. Claimed steps go to that provider. The rest stay on the CPU. That split is the graph partition.

AccelEP is the provider for an accelerator that is not there. It plugs into Runtime the way a GPU provider would. It claims static float32 MatMul, 2D Gemm, and `com.microsoft.FusedMatMul`. A following Relu stays with the matrix, and so does an exact opset-20 Gelu. One same-shape residual can stay, and a single-consumer opset-23 RMSNormalization can stay. A constant int8 weight stays packed, and the multiply dequantizes one panel at a time. There is no chip. `SimDevice` is the stand-in: a heap and named kernels in C++, running on the CPU. A hardware backend can replace those kernels and keep the same calls.

The stand-in is for the questions you can answer before the hardware exists:

- Does the provider register, and does Runtime hand it the steps it claimed?
- Does fusion leave the activation on the device, and does the packed int8 path match the dequantized float32 product?
- Which calls ran, in what order, and with what sizes? The recording device logs that.
- Does the output match a trusted reference? Plugin tests compare with ONNX Runtime's CPU provider at `rtol=1e-5` and `atol=1e-5`. The SmolLM check compares the device logit row with an independent NumPy forward. On the checked 4-token prefill the max abs difference was `4.20e-5`. On the following decode it was `3.58e-5`.

Runtime's own CPU provider runs the same recipe on the CPU and stops there. AccelEP hands the claimed steps to the stand-in. Each product is a multiply followed by an add, in a fixed `k` order. The path is slower. What you get is a reproducible account of what an accelerator-shaped device was asked to do, and whether that account matches the reference.

The chat is a second program on the same device. The page does not load an ONNX file and does not call the plugin. The host builds the chat string. The simulator tokenizes it, runs the forward pass, picks the next token, and detokenizes. Temperature is applied in that choice, after the scores exist. It does not change the forward pass.

A match shows that the software and the reference math agree. It does not show that a real chip returns the same numbers. A chip can add in a different order and round differently. That comparison is what [mmaprobe](mmaprobe/README.md) is for. AccelEP does not apply an `mma.sync` rounding model. The mmaprobe tree in this repo records no silicon result.

The checked chat model is [SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct).

The weight file is bfloat16, about 257 MB, and is left out of git. The loader converts one tensor at a time with the same bfloat16 bit shift it has always used, uploads it, and drops the host copy before the next tensor. A full-size Llama does not fit in 8 GB. Context length is `min(max_position_embeddings, 1 GiB / bytes_per_token)`. This shape uses 71,948 bytes per token, so SmolLM2 keeps 8192 positions and SmolLM-135M-Instruct keeps 2048. A caller that passes `context` still overrides that default.

## What a chat request does

The browser sends the message and shows the reply. Python writes the chat string and answers HTTP. The C++ library tokenizes, runs the forward pass, samples, and turns the ids back into text.

1. **Chat string.** The server renders `chat_template` from `tokenizer_config.json`. It accepts only the skeleton both SmolLM files use: one pass over the messages, the system line only when that file's `{% if loop.first %}` block is present, and the assistant header when a reply is requested. Anything else raises. SmolLM2's template contains the system sentence, so a user message still begins with:

   ```text
   <|im_start|>system
   You are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>
   ```

   Each message is then `<|im_start|>`, the role, a newline, the text, `<|im_end|>`, and a newline. The string ends with `<|im_start|>assistant` and a newline. An empty message list does not insert the system line, because that `if` sits inside the `for`. SmolLM-135M's template has no system sentence, so none is added. `POST /v1/completions` skips this and completes the raw prompt.

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

At startup the host builds the RoPE cosine and sine tables for every position in the context and uploads them. Embedding and RMSNorm weights keep the file layout. Every matrix, including the vocabulary matrix, is transposed on the host and then uploaded. The transpose scratch is freed before the next tensor is read.

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

`SampleToken` runs after the forward. Temperature 0, and any temperature that is not positive, keeps the largest logit. A tie keeps the earlier id. `top_k` is ignored on that path. A positive temperature with `top_k` of 0, or with `top_k` at least the vocabulary size, draws from the full vocabulary with an xorshift generator. `top_k` from 1 to 64 keeps a 64-slot table filled in one pass. The earlier id wins a tie. The exp walk visits those slots in increasing token id. Seed 0 is replaced with `0x6d2b79f5` inside `Generate`, before sampling. `SampleToken` does not reseed a zero state on its own.

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
PYTHONPATH=. .venv/bin/python -m serve --port 8000
```

`python -m serve` loads [SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct) from `models/SmolLM2-360M-Instruct/`. That checkpoint is Llama-shaped, with the same chat template as SmolLM2-135M. It has 32 layers, hidden size 960, and 15 query heads. The float32 weights are about 1.56 GB. The 1 GiB buffer budget still leaves 8192 positions, so a resident model is about 2.5 GB. Pass `--model models` to serve the checked 135M checkpoint instead. Pass `--context` to shorten the cache. A 2 GB container should use `--context 1024`, which keeps the weights and the dense buffers under that limit.

The server listens on `127.0.0.1:8000`. The model loads on a background thread. `GET /health` returns `{"status":"loading"}` until that finishes, then `ok` with the model name, layer count, hidden size, vocab size, and context length. One lock wraps each completion. `max_tokens` above 48 is rejected with `max_tokens must be at most 48` before the forward. The server default is 16. The page caps the same field at 48.

`stream: true` on either POST returns newline-delimited JSON with `Transfer-Encoding: chunked`. Each intermediate line is `{"text": "..."}`, the stable UTF-8 prefix so far. A trailing U+FFFD is held back until a later token can complete it. The final line has the same fields as the JSON completion, including the ids and the copy counters. The page reads that body with `response.body.getReader()` and paints `#answer`. Detokenizing uses the host `tokenizers` package and does not call back into the model.

| Route | What it does |
| --- | --- |
| `GET /` | Serves the chat page. |
| `GET /health` | Returns `loading`, then `ok` with the model name, layer count, hidden size, vocab size, and context length. |
| `POST /v1/completions` | Completes a raw prompt. |
| `POST /v1/chat/completions` | Builds the chat string from the loaded template, then completes it. |

## Tests

On this Mac, run:

```bash
make sim-test
```

That needs the `models/` files above and a `.venv` with `numpy` and `tokenizers`. Three programs run, in order. A good run prints one line from each.

1. `build/test_sim_kernels` prints `kernel tests passed`.
   Eight kernels, each on a few hand values: embedding, RMSNorm, RoPE, the cache append, attention, SiLU, the residual add, and sampling. Logits `[1, 3, 3]` at temperature 0 return id 1.

2. `build/test_runtime` prints `runtime tests passed`.
   Twenty-six checks of the matrix runtime used by the ONNX plugin: MatMul and Gemm shapes, a fused Relu, a residual, transposes, allocation, a recorded call trace, and the copy counters. One check builds a second device, `CreateTiledDevice()`, whose matmul panels are 32 by 64 by 32. That scratch is 16,384 bytes, half of this machine's 32 KB L1 data cache. Each output keeps one accumulator and adds products for `k = 0, 1, …`, so the bits match `ReferenceMatMul`. `CreateSimDevice()` is still what the page and the plugin use. Another check multiplies a float activation by an int8 weight inside a 64 by 256 panel. The scale is applied while the panel is filled. The result matches `ReferenceMatMul` of the dequantized floats, including a per-row scale and a scalar scale. A transposed packed weight throws.

3. `tests/test_serve.py` prints `serve tests passed`.
   Five checks:
   - Encode and decode match `tokenizers` on 24 strings, including bytes missing from the vocabulary.
   - One simulator matrix multiply matches NumPy.
   - The Python RMSNorm and RoPE helpers match their formulas. The serving path uses the C++ kernels from step 1.
   - Context is 71,948 bytes per token, so SmolLM2 defaults to 8192 and SmolLM-135M-Instruct to 2048. `context=8` still overrides. The embed float32 CRC is unchanged after the upload. A 4-token prefill of ids `[10, 20, 30, 40]` and the following decode of id 50 match positions 3 and 4 of one independent 5-token NumPy forward, max abs at most `1e-3`. That comparison copies 20 bytes in and nothing back. Greedy chat for "The capital of France is" is `The capital of France is Paris.` with ids `[504, 3575, 282, 4649, 314, 7042, 30, 2]`, 63 bytes back, 140 bytes in, and 1,688 matmuls. The chat string comes from the loaded template. An empty message list is only the assistant header. `max_tokens` 49 raises before a forward. One token at temperature 0.8, with `top_k` left at 0, copies only its text and its id.
   - `GET /health` is polled until `ok`. `max_tokens` 49 returns 400 with `max_tokens must be at most 48`. A streaming Paris completion is chunked, has no `Content-Length`, and the final line carries the same ids and the same 63/140/1,688 counters. A non-streaming short completion still returns 200.

This Mac printed all three pass lines. The France reply in the table above is the one that run produced. The logit check printed max abs `4.20e-5` on the 4-token prefill and `3.58e-5` on the following decode, with argmax 32 and 20. When `models/model.safetensors` is absent, step 3 skips the model and HTTP checks and still prints `serve tests passed`.

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

The kernel checks use one or two elements. The model uses head size 64 and grouped-query attention. The serve test compares the device logit row with one causal NumPy forward: the 4-token prefill against position 3, and the next decode against position 4. Walking a whole prompt one token at a time, apart from that single decode step, is still open.

The loader accepts a Llama with SiLU, no RoPE scaling, and a Hugging Face BPE tokenizer whose pretokenizer is `Digits` then `ByteLevel`, with a null normalizer, a null post-processor, and byte fallback off. `JackFram/llama-68m` has no `tokenizer.json`. `ModelCloud/tinyllama-15M-stories` sets byte fallback and an `unk_token`, so this loader rejects it.

## ONNX plugin

The plugin takes float32 MatMul, 2D Gemm, and `com.microsoft.FusedMatMul` when the shapes are static. One leading MatMul batch dimension may still be unknown when the graph is compiled. Softmax stays on the CPU execution provider.

A following Relu stays on the device. So does an opset-20 Gelu when `approximate` is absent or `"none"`. The next matrix in that group reads the activated values from the device. A lone MatMul can add one residual of the same shape. A constant-weight projection leaves that residual on the CPU when the Add feeds more than one node.

A single-consumer opset-23 `RMSNormalization` can stay on the device when its scale is a constant last-axis vector, it has no bias, and its only attributes are `axis` and `epsilon`. Graph resolution also copies the schema default `stash_type` of 1, which is the float sum, and that default still fuses. Any other `stash_type` stays on the CPU. On the prenorm toy, the context MatMul, the residual add, that norm, the MLP, and the closing residual are one kernel. The pre-norm sum stays on the device for the closing add. That graph copies 576 bytes to the device and 192 back, with two residual adds. The attention norm still has two consumers, so it stays on the CPU. The toy decoder still uses `LayerNormalization`, and its counters stay 1,152 bytes to the device and 896 back, with no fused add.

A constant int8 `DequantizeLinear` that feeds one float MatMul stays packed. The multiply dequantizes into a 64 by 256 panel and applies the scale there. A scalar scale uses axis `-1`. A rank-1 scale whose axis is the schema default `1` stays on the CPU, and so do block scales and uint8. The panel is not a second copy onto the device. This is not Q4_K, and it does not shrink the SmolLM weights. A request to transpose a packed int8 weight throws.

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

On this Mac the linker rejects `-z`, so `make build/libaccel_ep.so` stops there. `src/ep.cc` does compile to `build/ep.o` with the same warning flags. The Python plugin tests were not run here. `make simlib` still builds `build/libaccel_sim.dylib` for the Llama server. Run `make clean` before a Linux plugin build, so the Mac objects are not reused.

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
