import argparse
from pathlib import Path

from serve.server import serve


def main():
    parser = argparse.ArgumentParser(description="Serve a Llama model on the CPU simulator.")
    parser.add_argument("--model", default=str(Path(__file__).resolve().parents[1] / "models"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    serve(args.model, args.host, args.port)


if __name__ == "__main__":
    main()
