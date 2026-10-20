"""One-model HTTP server. The simulator holds the weights for the life of the process."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from serve.engine import Engine


def make_handler(engine):
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

        def do_GET(self):
            if self.path != "/health":
                self._send(404, {"error": "not found"})
                return
            self._send(
                200,
                {
                    "status": "ok",
                    "layers": engine.model.layers,
                    "hidden": engine.model.hidden,
                    "vocab": engine.model.vocab,
                },
            )

        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            try:
                request = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                self._send(400, {"error": "body is not json"})
                return
            try:
                if self.path == "/v1/completions":
                    prompt = request["prompt"]
                elif self.path == "/v1/chat/completions":
                    prompt = engine.chat_prompt(request["messages"])
                else:
                    self._send(404, {"error": "not found"})
                    return
                with lock:
                    result = engine.complete(
                        prompt,
                        max_tokens=int(request.get("max_tokens", 16)),
                        temperature=float(request.get("temperature", 0)),
                        seed=int(request.get("seed", 0)),
                    )
            except (KeyError, TypeError, ValueError) as exc:
                self._send(400, {"error": str(exc)})
                return
            self._send(200, result)

    return Handler


def serve(model_dir, host, port):
    engine = Engine(model_dir)
    server = ThreadingHTTPServer((host, port), make_handler(engine))
    print(f"listening on http://{host}:{port}")
    try:
        server.serve_forever()
    finally:
        server.server_close()
        engine.close()
