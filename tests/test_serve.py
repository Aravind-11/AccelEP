"""Simulator matmul, Llama math, and one HTTP completion when weights are present."""

import json
import math
import threading
import time
import urllib.error
import urllib.request
import zlib
from pathlib import Path

import http.client

import numpy as np

from serve.engine import Engine, parse_chat_template, render_chat, stable_text
from serve.llama import Llama, apply_rope, context_bytes_per_token, default_context, rmsnorm, rope_tables
from serve.server import serve
from serve.sim import Sim
from serve.weights import SafetensorsReader

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models"
PARIS_PROMPT = (
    "<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n"
    "<|im_start|>user\nThe capital of France is<|im_end|>\n"
    "<|im_start|>assistant\n"
)
PARIS_IDS = [504, 3575, 282, 4649, 314, 7042, 30, 2]
PARIS_TEXT = "The capital of France is Paris."


def _matmul(left, right):
    """Float32 products in k order. One multiply and one add per product, matching ReferenceMatMul."""
    left = np.ascontiguousarray(left, dtype=np.float32)
    right = np.ascontiguousarray(right, dtype=np.float32)
    rows, width = left.shape
    out = np.zeros((rows, right.shape[1]), dtype=np.float32)
    for kk in range(width):
        out += left[:, kk : kk + 1] * right[kk : kk + 1, :]
    return out


def _proj(states, weight):
    """Hugging Face stores [out, in]. The device multiplies by the transpose."""
    transposed = np.ascontiguousarray(np.transpose(weight), dtype=np.float32)
    return _matmul(states, transposed)


def _rms(states, weight, eps):
    """Double sum of squares, then float(x * inv) * weight. This is Device::RmsNorm."""
    values = np.ascontiguousarray(states, dtype=np.float32)
    scale = np.ascontiguousarray(weight, dtype=np.float32).reshape(-1)
    flat = values.reshape(-1, values.shape[-1])
    out = np.empty_like(flat)
    dim = flat.shape[1]
    eps64 = float(np.float32(eps))
    for row in range(flat.shape[0]):
        square = 0.0
        for i in range(dim):
            value = float(flat[row, i])
            square += value * value
        inv = 1.0 / math.sqrt(square / dim + eps64)
        for i in range(dim):
            out[row, i] = np.float32(float(flat[row, i]) * inv) * scale[i]
    return out.reshape(values.shape)


def _silu_mul(gate, up):
    gate = np.ascontiguousarray(gate, dtype=np.float32)
    up = np.ascontiguousarray(up, dtype=np.float32)
    return (gate / (np.float32(1.0) + np.exp(-gate))) * up


def _attend(query, key, value):
    """Causal grouped-query attention. Dots are sequential float32 adds."""
    seq, heads, dim = query.shape
    kv_heads = key.shape[1]
    repeats = heads // kv_heads
    scale = np.float32(np.float32(1.0) / np.sqrt(np.float32(dim)))
    mixed = np.zeros((seq, heads, dim), dtype=np.float32)
    for s in range(seq):
        limit = s
        for h in range(heads):
            kv = h // repeats
            qv = query[s, h]
            scores = np.empty(limit + 1, dtype=np.float32)
            max_score = np.float32(-np.inf)
            for t in range(limit + 1):
                kvv = key[t, kv]
                dot = np.float32(0.0)
                for d in range(dim):
                    dot = np.float32(dot + qv[d] * kvv[d])
                scores[t] = np.float32(dot * scale)
                if scores[t] > max_score:
                    max_score = scores[t]
            total = np.float32(0.0)
            for t in range(limit + 1):
                scores[t] = np.exp(np.float32(scores[t] - max_score))
                total = np.float32(total + scores[t])
            for d in range(dim):
                acc = np.float32(0.0)
                column = value[:, kv, d]
                for t in range(limit + 1):
                    acc = np.float32(acc + (scores[t] / total) * column[t])
                mixed[s, h, d] = acc
    return mixed


def _tied_logits(row, embed):
    """Last-position vocabulary multiply. b[k, j] is embed[j, k] because the head is transposed."""
    flat = np.ascontiguousarray(row, dtype=np.float32).reshape(-1)
    table = np.ascontiguousarray(embed, dtype=np.float32)
    out = np.zeros(table.shape[0], dtype=np.float32)
    for kk in range(flat.shape[0]):
        out += flat[kk] * table[:, kk]
    return out


def _causal_logit_rows(model_dir, token_ids, positions):
    """One independent causal forward. Position 3 matches a 4-token prefill; position 4 matches the decode."""
    reader = SafetensorsReader(model_dir / "model.safetensors")
    with open(model_dir / "config.json", encoding="utf-8") as handle:
        config = json.load(handle)
    hidden_size = int(config["hidden_size"])
    layers = int(config["num_hidden_layers"])
    heads = int(config["num_attention_heads"])
    kv_heads = int(config["num_key_value_heads"])
    head_dim = hidden_size // heads
    eps = float(config["rms_norm_eps"])
    theta = float(config["rope_theta"])
    embed = reader.get("model.embed_tokens.weight")
    try:
        hidden = np.stack([embed[int(token)] for token in token_ids]).astype(np.float32)
    finally:
        del embed
    seq = hidden.shape[0]
    cos, sin = rope_tables(head_dim, seq, theta)
    for index in range(layers):
        if index % 10 == 0:
            print(f"numpy layer {index}", flush=True)
        prefix = f"model.layers.{index}"
        normed = _rms(hidden, reader.get(f"{prefix}.input_layernorm.weight"), eps)
        query = _proj(normed, reader.get(f"{prefix}.self_attn.q_proj.weight")).reshape(seq, heads, head_dim)
        key = _proj(normed, reader.get(f"{prefix}.self_attn.k_proj.weight")).reshape(seq, kv_heads, head_dim)
        value = _proj(normed, reader.get(f"{prefix}.self_attn.v_proj.weight")).reshape(seq, kv_heads, head_dim)
        query = apply_rope(query, cos, sin)
        key = apply_rope(key, cos, sin)
        mixed = _attend(query, key, value).reshape(seq, heads * head_dim)
        np.add(hidden, _proj(mixed, reader.get(f"{prefix}.self_attn.o_proj.weight")), out=hidden)
        normed = _rms(hidden, reader.get(f"{prefix}.post_attention_layernorm.weight"), eps)
        gate = _proj(normed, reader.get(f"{prefix}.mlp.gate_proj.weight"))
        up = _proj(normed, reader.get(f"{prefix}.mlp.up_proj.weight"))
        down = _proj(_silu_mul(gate, up), reader.get(f"{prefix}.mlp.down_proj.weight"))
        np.add(hidden, down, out=hidden)
    final_weight = reader.get("model.norm.weight")
    embed = reader.get("model.embed_tokens.weight")
    try:
        rows = {}
        for pos in positions:
            rows[pos] = _tied_logits(_rms(hidden[pos], final_weight, eps), embed)
        return rows
    finally:
        del embed


def _embed_crc(path):
    reader = SafetensorsReader(path)
    values = reader.get("model.embed_tokens.weight")
    try:
        blob = np.ascontiguousarray(values, dtype=np.float32).tobytes()
        return zlib.crc32(blob) & 0xFFFFFFFF
    finally:
        del values


class _HoldReplacement:
    def decode(self, ids, skip_special_tokens=True):
        return "ab\ufffd"


def test_tokenizer_matches_reference():
    tok_path = MODEL / "tokenizer.json"
    if not tok_path.is_file():
        print("tokenizer absent, skipped tokenizer parity")
        return
    from tokenizers import Tokenizer

    reference = Tokenizer.from_file(str(tok_path))
    model = Llama(MODEL, context=8, load_weights=False)
    try:
        samples = [
            "",
            "Hello",
            "Hey 123 friend!",
            "dog's day\nnext",
            "café",
            "🎉",
            "a  b",
            "a\tb",
            "\x04",
            "a\x04b",
            "!\x04!",
            ".\x04.",
            "\n\x04\n",
            " \x04!",
            " \x04 ",
            "\x00",
            "\x1f ",
            "hello\x04",
            "\U00040000",
            "A\U00040000B",
            "!\U00040000@",
            "<|im_start|>hi<|im_end|>",
            "The capital of France is",
            (
                "<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n"
                "<|im_start|>user\nThe capital of France is<|im_end|>\n"
                "<|im_start|>assistant\n"
            ),
        ]
        for text in samples:
            got = model.encode(text)
            expect = reference.encode(text, add_special_tokens=False).ids
            if got != expect:
                raise SystemExit(f"encode mismatch for {text!r}\n got {got}\n expect {expect}")
            got_text = model.decode(got)
            expect_text = reference.decode(expect, skip_special_tokens=True)
            if got_text != expect_text:
                raise SystemExit(f"decode mismatch for {text!r}\n got {got_text!r}\n expect {expect_text!r}")
        paris = [504, 3575, 282, 4649, 314, 7042, 30, 2]
        if model.decode(paris) != reference.decode(paris, skip_special_tokens=True):
            raise SystemExit("paris token decode mismatch")
        print("tokenizer parity ok")
    finally:
        model.close()


def test_matmul_matches_numpy():
    left = np.arange(6, dtype=np.float32).reshape(2, 3)
    weight = np.arange(12, dtype=np.float32).reshape(3, 4)
    with Sim() as sim:
        sim.reset_stats()
        uploaded = sim.upload(weight)
        got = sim.matmul(left, uploaded, 4)
        sim.release(uploaded)
        assert sim.matmuls() == 1
        assert sim.bytes_to_device() > 0
    expected = left @ weight
    if not np.allclose(got, expected):
        raise SystemExit(f"matmul mismatch\n{got}\n{expected}")


def test_rmsnorm_and_rope():
    weight = np.array([1.0, 2.0, 0.5, 1.5], dtype=np.float32)
    row = np.array([1.0, -1.0, 2.0, 0.0], dtype=np.float32)
    scale = np.sqrt(np.mean(np.square(row)) + 1e-5)
    expected = row / scale * weight
    got = rmsnorm(row, weight, 1e-5)
    if not np.allclose(got, expected):
        raise SystemExit("rmsnorm mismatch")
    cos, sin = rope_tables(4, 3, 10000.0)
    states = np.ones((2, 1, 4), dtype=np.float32)
    rotated = apply_rope(states, cos[:2], sin[:2])
    if rotated.shape != (2, 1, 4):
        raise SystemExit(f"rope shape {rotated.shape}")
    # Position 0 has cos 1 and sin 0, so the vector is unchanged.
    if not np.allclose(rotated[0], states[0]):
        raise SystemExit("rope at position 0 moved the vector")


def _close_probe(model):
    model.close()


def test_forward_stays_on_device():
    weights = MODEL / "model.safetensors"
    if not weights.is_file():
        print("model weights absent, skipped on-device forward")
        return
    per_token = context_bytes_per_token(576, 9, 3, 1536, 30)
    if per_token != 71948:
        raise SystemExit(f"bytes per token {per_token}")
    if default_context(8192, per_token) != 8192 or default_context(2048, per_token) != 2048:
        raise SystemExit("default context missed the checkpoint cap")
    override = Llama(MODEL, context=8, load_weights=False)
    try:
        if override.context != 8:
            raise SystemExit(f"context override is {override.context}")
    finally:
        _close_probe(override)
    small_dir = MODEL / "SmolLM-135M-Instruct"
    if (small_dir / "config.json").is_file():
        small = Llama(small_dir, load_weights=False)
        try:
            if small.context != 2048:
                raise SystemExit(f"SmolLM-135M context is {small.context}")
        finally:
            _close_probe(small)
        small_template = json.loads((small_dir / "tokenizer_config.json").read_text(encoding="utf-8"))["chat_template"]
        small_text = render_chat(
            parse_chat_template(small_template),
            [{"role": "user", "content": "The capital of France is"}],
        )
        if "helpful AI assistant" in small_text:
            raise SystemExit(f"SmolLM-135M template injected a system line: {small_text!r}")
        if small_text != (
            "<|im_start|>user\nThe capital of France is<|im_end|>\n<|im_start|>assistant\n"
        ):
            raise SystemExit(f"SmolLM-135M render drifted: {small_text!r}")
    wide = Llama(MODEL, load_weights=False)
    try:
        if wide.context != 8192:
            raise SystemExit(f"SmolLM2 context is {wide.context}")
    finally:
        _close_probe(wide)
    try:
        parse_chat_template("{% for message in messages %}nope")
        raise SystemExit("garbage template was accepted")
    except ValueError:
        pass
    if stable_text(_HoldReplacement(), [1, 2]) != "ab":
        raise SystemExit("stable text kept a trailing replacement character")
    file_size = weights.stat().st_size
    embed_crc = _embed_crc(weights)
    engine = Engine(MODEL)
    try:
        if engine.model.context != 8192:
            raise SystemExit(f"loaded context is {engine.model.context}")
        if weights.stat().st_size != file_size or _embed_crc(weights) != embed_crc:
            raise SystemExit("embed bytes changed while the model was resident")
        prompt = engine.chat_prompt([{"role": "user", "content": "The capital of France is"}])
        if prompt != PARIS_PROMPT:
            raise SystemExit(f"chat prompt drifted: {prompt!r}")
        if engine.chat_prompt([]) != "<|im_start|>assistant\n":
            raise SystemExit("empty messages injected a system line")
        print("numpy causal forward", flush=True)
        rows = _causal_logit_rows(MODEL, [10, 20, 30, 40, 50], (3, 4))
        engine.model.reset_stats()
        engine.model.forward_ids([10, 20, 30, 40], True)
        prefill = engine.model.logit_max_abs(rows[3])
        engine.model.forward_ids([50], False)
        decode = engine.model.logit_max_abs(rows[4])
        host_now = engine.model.bytes_to_host()
        device_now = engine.model.bytes_to_device()
        print(
            f"logit max abs prefill {prefill:.8e} decode {decode:.8e} "
            f"argmax {int(np.argmax(rows[3]))} {int(np.argmax(rows[4]))} "
            f"to_device {device_now} to_host {host_now}",
            flush=True,
        )
        del rows
        if prefill > 1e-3 or decode > 1e-3:
            raise SystemExit(f"logit max abs {prefill} {decode} exceeds 1e-3")
        if host_now != 0 or device_now != 20:
            raise SystemExit(f"logit copies to_host {host_now} to_device {device_now}")
        engine.model.reset_stats()
        result = engine.complete(prompt, max_tokens=8, temperature=0)
        if result["text"] != PARIS_TEXT or result["token_ids"] != PARIS_IDS:
            raise SystemExit(f"on-device completion drifted: {result}")
        host_bytes = engine.model.bytes_to_host()
        device_bytes = engine.model.bytes_to_device()
        expected_host = len(result["text"].encode("utf-8")) + result["completion_tokens"] * 4
        expected_device = result["prompt_tokens"] * 4
        if host_bytes != 63 or device_bytes != 140 or engine.model.matmuls() != 1688:
            raise SystemExit(
                f"paris counters host {host_bytes} device {device_bytes} matmuls {engine.model.matmuls()}"
            )
        if host_bytes != expected_host or device_bytes != expected_device:
            raise SystemExit(
                f"generation crossed the host boundary: to_host {host_bytes} expected {expected_host}, "
                f"to_device {device_bytes} expected {expected_device}"
            )
        print(
            "on-device completion:",
            json.dumps(result["text"]),
            "tokens",
            result["token_ids"],
            "host_bytes",
            host_bytes,
        )
        engine.model.reset_stats()
        try:
            engine.complete("Hello", max_tokens=49, temperature=0)
            raise SystemExit("max_tokens 49 ran")
        except ValueError as exc:
            if str(exc) != "max_tokens must be at most 48":
                raise SystemExit(f"max_tokens error {exc}") from exc
        if engine.model.matmuls() != 0:
            raise SystemExit("max_tokens 49 ran a forward")
        engine.model.reset_stats()
        sampled = engine.complete("Hello", max_tokens=1, temperature=0.8, seed=1)
        if sampled["completion_tokens"] != 1:
            raise SystemExit(f"temperature sample did not return one token: {sampled}")
        sampled_host = len(sampled["text"].encode("utf-8")) + sampled["completion_tokens"] * 4
        sampled_device = sampled["prompt_tokens"] * 4
        if engine.model.bytes_to_host() != sampled_host or engine.model.bytes_to_device() != sampled_device:
            raise SystemExit(
                f"temperature sample copied model state: to_host {engine.model.bytes_to_host()} expected {sampled_host}, "
                f"to_device {engine.model.bytes_to_device()} expected {sampled_device}"
            )
    finally:
        engine.close()


def _wait_until_ready():
    health = None
    for _ in range(400):
        try:
            with urllib.request.urlopen("http://127.0.0.1:8765/health", timeout=2) as response:
                health = json.load(response)
        except Exception:
            time.sleep(0.5)
            continue
        state = health.get("status")
        if state == "ok":
            return health
        if state == "error":
            raise SystemExit(f"server failed: {health}")
        time.sleep(0.5)
    raise SystemExit(f"server did not become healthy: {health}")


def test_http_completion():
    weights = MODEL / "model.safetensors"
    if not weights.is_file() or not (MODEL / "config.json").is_file():
        print("model weights absent, skipped http completion")
        return
    page = (ROOT / "serve" / "static" / "index.html").read_text(encoding="utf-8")
    if 'id="max" type="number" min="1" max="48"' not in page or "stream: true" not in page:
        raise SystemExit("page does not stream with a 48 token cap")
    if "getReader" not in page:
        raise SystemExit("page still reads the body as one JSON value")
    thread = threading.Thread(target=serve, args=(MODEL, "127.0.0.1", 8765), daemon=True)
    thread.start()
    _wait_until_ready()
    rejected = json.dumps(
        {
            "messages": [{"role": "user", "content": "The capital of France is"}],
            "max_tokens": 49,
            "temperature": 0,
        }
    ).encode()
    request = urllib.request.Request(
        "http://127.0.0.1:8765/v1/chat/completions",
        data=rejected,
        headers={"Content-Type": "application/json"},
    )
    try:
        urllib.request.urlopen(request, timeout=30)
        raise SystemExit("max_tokens 49 returned 200")
    except urllib.error.HTTPError as exc:
        payload = json.load(exc)
        if exc.code != 400 or payload.get("error") != "max_tokens must be at most 48":
            raise SystemExit(f"max_tokens http {exc.code} {payload}") from exc
    connection = http.client.HTTPConnection("127.0.0.1", 8765, timeout=300)
    stream_body = json.dumps(
        {
            "messages": [{"role": "user", "content": "The capital of France is"}],
            "max_tokens": 8,
            "temperature": 0,
            "stream": True,
        }
    ).encode()
    connection.request(
        "POST",
        "/v1/chat/completions",
        body=stream_body,
        headers={"Content-Type": "application/json"},
    )
    streamed = connection.getresponse()
    if streamed.status != 200:
        raise SystemExit(f"stream status {streamed.status} {streamed.read()!r}")
    if streamed.getheader("Transfer-Encoding") != "chunked":
        raise SystemExit(f"stream transfer {streamed.getheader('Transfer-Encoding')}")
    if streamed.getheader("Content-Length") is not None:
        raise SystemExit(f"stream content-length {streamed.getheader('Content-Length')}")
    lines = [json.loads(line) for line in streamed.read().decode("utf-8").splitlines() if line.strip()]
    connection.close()
    if len(lines) < 2:
        raise SystemExit(f"stream lines {lines}")
    final = lines[-1]
    if final.get("text") != PARIS_TEXT or final.get("token_ids") != PARIS_IDS:
        raise SystemExit(f"stream completion drifted: {final}")
    if final.get("bytes_to_host") != 63 or final.get("bytes_to_device") != 140 or final.get("matmuls") != 1688:
        raise SystemExit(
            f"stream counters host {final.get('bytes_to_host')} device {final.get('bytes_to_device')} "
            f"matmuls {final.get('matmuls')}"
        )
    for line in lines[:-1]:
        if "text" not in line or "token_ids" in line or not PARIS_TEXT.startswith(line["text"]):
            raise SystemExit(f"stream chunk {line}")
    short = json.dumps({"prompt": "Hello", "max_tokens": 2, "temperature": 0}).encode()
    request = urllib.request.Request(
        "http://127.0.0.1:8765/v1/completions",
        data=short,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        raw = json.load(response)
    if response.status != 200 or raw["completion_tokens"] < 1 or not raw["text"].strip():
        raise SystemExit(f"empty raw completion {raw}")
    print("completion:", json.dumps(final["text"]))


def main():
    test_tokenizer_matches_reference()
    test_matmul_matches_numpy()
    test_rmsnorm_and_rope()
    test_forward_stays_on_device()
    test_http_completion()
    print("serve tests passed")


if __name__ == "__main__":
    main()
