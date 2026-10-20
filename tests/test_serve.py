"""Simulator matmul, Llama math, and one HTTP completion when weights are present."""

import json
import threading
import urllib.request
from pathlib import Path

import numpy as np

from serve.engine import Engine
from serve.llama import Llama, apply_rope, rmsnorm, rope_tables
from serve.server import serve
from serve.sim import Sim

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models"


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


def test_forward_stays_on_device():
    weights = MODEL / "model.safetensors"
    if not weights.is_file():
        print("model weights absent, skipped on-device forward")
        return
    engine = Engine(MODEL)
    try:
        engine.model.reset_stats()
        result = engine.complete(
            engine.chat_prompt([{"role": "user", "content": "The capital of France is"}]),
            max_tokens=8,
            temperature=0,
        )
        if "Paris" not in result["text"]:
            raise SystemExit(f"on-device completion did not answer Paris: {result}")
        host_bytes = engine.model.bytes_to_host()
        device_bytes = engine.model.bytes_to_device()
        expected_host = len(result["text"].encode("utf-8")) + result["completion_tokens"] * 4
        expected_device = result["prompt_tokens"] * 4
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


def test_http_completion():
    weights = MODEL / "model.safetensors"
    if not weights.is_file() or not (MODEL / "config.json").is_file():
        print("model weights absent, skipped http completion")
        return
    thread = threading.Thread(target=serve, args=(MODEL, "127.0.0.1", 8765), daemon=True)
    thread.start()
    health = None
    for _ in range(240):
        try:
            with urllib.request.urlopen("http://127.0.0.1:8765/health", timeout=2) as response:
                health = json.load(response)
                break
        except Exception:
            import time

            time.sleep(0.5)
    if health is None or health.get("status") != "ok":
        raise SystemExit(f"server did not become healthy: {health}")
    body = json.dumps(
        {
            "messages": [{"role": "user", "content": "The capital of France is"}],
            "max_tokens": 8,
            "temperature": 0,
        }
    ).encode()
    request = urllib.request.Request(
        "http://127.0.0.1:8765/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        payload = json.load(response)
    text = payload.get("text", "")
    if "Paris" not in text:
        raise SystemExit(f"chat completion did not answer Paris: {payload}")
    short = json.dumps({"prompt": "Hello", "max_tokens": 2, "temperature": 0}).encode()
    request = urllib.request.Request(
        "http://127.0.0.1:8765/v1/completions",
        data=short,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        raw = json.load(response)
    if raw["completion_tokens"] < 1 or not raw["text"].strip():
        raise SystemExit(f"empty raw completion {raw}")
    print("completion:", json.dumps(text))


def main():
    test_tokenizer_matches_reference()
    test_matmul_matches_numpy()
    test_rmsnorm_and_rope()
    test_forward_stays_on_device()
    test_http_completion()
    print("serve tests passed")


if __name__ == "__main__":
    main()
