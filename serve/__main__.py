import argparse
from pathlib import Path

from serve.server import serve


def main():
    parser = argparse.ArgumentParser(description="Serve a Llama model on the CPU simulator.")
    root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--model",
        default=str(root / "models" / "SmolLM2-360M-Instruct"),
        help="Checkpoint directory. The checked tests still use models/ (SmolLM2-135M).",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--context",
        type=int,
        default=None,
        help="Override the context length. The default fits the dense buffers in 1 GiB.",
    )
    args = parser.parse_args()
    serve(args.model, args.host, args.port, context=args.context)


if __name__ == "__main__":
    main()
