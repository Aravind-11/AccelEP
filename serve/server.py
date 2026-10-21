"""One-model HTTP server. The simulator holds the weights for the life of the process."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from serve.engine import MAX_NEW_TOKENS, Engine

PAGE = Path(__file__).resolve().parent / "static" / "index.html"


def make_handler(holder):
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            print(f"{self.address_string()} {fmt % args}")

        def _send(self, status, payload):
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _engine(self):
            engine = holder["engine"]
            if holder["error"]:
                self._send(500, {"error": holder["error"]})
                return None
            if engine is None:
                self._send(503, {"error": "simulator is loading"})
                return None
            return engine

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                body = PAGE.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path != "/health":
                self._send(404, {"error": "not found"})
                return
            if holder["error"]:
                self._send(500, {"status": "error", "error": holder["error"]})
                return
            engine = holder["engine"]
            if engine is None:
                self._send(200, {"status": "loading", "model": holder["model"]})
                return
            self._send(
                200,
                {
                    "status": "ok",
                    "model": holder["model"],
                    "layers": engine.model.layers,
                    "hidden": engine.model.hidden,
                    "vocab": engine.model.vocab,
                    "context": engine.model.context,
                },
            )

        def _write_chunk(self, payload):
            self.wfile.write(f"{len(payload):X}\r\n".encode("ascii"))
            self.wfile.write(payload)
            self.wfile.write(b"\r\n")
            self.wfile.flush()

        def _finish_chunks(self):
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()

        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            try:
                request = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                self._send(400, {"error": "body is not json"})
                return
            engine = self._engine()
            if engine is None:
                return
            try:
                if self.path == "/v1/completions":
                    prompt = request["prompt"]
                elif self.path == "/v1/chat/completions":
                    prompt = engine.chat_prompt(request["messages"])
                else:
                    self._send(404, {"error": "not found"})
                    return
                max_tokens = int(request.get("max_tokens", 16))
                if max_tokens > MAX_NEW_TOKENS:
                    raise ValueError("max_tokens must be at most 48")
                temperature = float(request.get("temperature", 0))
                seed = int(request.get("seed", 0))
                top_k = int(request.get("top_k", 0))
                stream = bool(request.get("stream", False))
            except (KeyError, TypeError, ValueError) as exc:
                self._send(400, {"error": str(exc)})
                return

            if not stream:
                try:
                    with lock:
                        engine.model.reset_stats()
                        result = engine.complete(
                            prompt,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            seed=seed,
                            top_k=top_k,
                        )
                        result["matmuls"] = engine.model.matmuls()
                        result["bytes_to_device"] = engine.model.bytes_to_device()
                        result["bytes_to_host"] = engine.model.bytes_to_host()
                except (KeyError, TypeError, ValueError) as exc:
                    self._send(400, {"error": str(exc)})
                    return
                self._send(200, result)
                return

            # Headers wait for the first line so a failure before any token can still be a 500.
            started = {"ok": False}

            def emit(payload):
                if not started["ok"]:
                    self.send_response(200)
                    self.send_header("Content-Type", "application/x-ndjson")
                    self.send_header("Transfer-Encoding", "chunked")
                    self.end_headers()
                    started["ok"] = True
                self._write_chunk(json.dumps(payload).encode() + b"\n")

            seen = {"text": ""}

            def on_token(token_id):
                text = engine.take_token(token_id)
                if text != seen["text"]:
                    seen["text"] = text
                    emit({"text": text})

            try:
                with lock:
                    engine.model.reset_stats()
                    engine.start_stream()
                    result = engine.complete(
                        prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        seed=seed,
                        top_k=top_k,
                        on_token=on_token,
                    )
                    result["matmuls"] = engine.model.matmuls()
                    result["bytes_to_device"] = engine.model.bytes_to_device()
                    result["bytes_to_host"] = engine.model.bytes_to_host()
                emit(result)
                self._finish_chunks()
            except (KeyError, TypeError, ValueError) as exc:
                if started["ok"]:
                    emit({"error": str(exc)})
                    self._finish_chunks()
                    return
                self._send(400, {"error": str(exc)})
            except Exception as exc:
                if not started["ok"]:
                    raise
                emit({"error": str(exc)})
                self._finish_chunks()

    return Handler


def serve(model_dir, host, port, context=None):
    holder = {"engine": None, "error": None, "model": Path(model_dir).name}

    def load():
        try:
            holder["engine"] = Engine(model_dir, context=context)
            print("simulator ready", "model", holder["model"], "context", holder["engine"].model.context)
        except Exception as exc:
            holder["error"] = str(exc)
            print(f"simulator failed: {exc}")

    threading.Thread(target=load, daemon=True).start()
    server = ThreadingHTTPServer((host, port), make_handler(holder))
    print(f"listening on http://{host}:{port}")
    try:
        server.serve_forever()
    finally:
        server.server_close()
        engine = holder["engine"]
        if engine is not None:
            engine.close()
