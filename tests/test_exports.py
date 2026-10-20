"""The plugin's exported names match the Python tests and ORT_API_VERSION."""

import ast
import ctypes
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
EP = ROOT / "src" / "ep.cc"
HEADER = ROOT / "third_party" / "onnxruntime" / "core" / "session" / "onnxruntime_c_api.h"
LIB = ROOT / "build" / "libaccel_ep.so"


def fail(message):
    raise SystemExit(message)


def walked_files():
    files = []
    for path in sorted((ROOT / "tests").glob("*.py")):
        if path.name == "test_exports.py":
            continue
        files.append(path)
    if not files:
        fail("no Python tests to walk")
    return files


def accel_names(path):
    found = set()
    tree = ast.parse(path.read_text(), filename=str(path))

    class Visit(ast.NodeVisitor):
        def visit_Name(self, node):
            if node.id.startswith("AccelEp"):
                found.add(node.id)
            self.generic_visit(node)

        def visit_Attribute(self, node):
            if node.attr.startswith("AccelEp"):
                found.add(node.attr)
            self.generic_visit(node)

        def visit_Constant(self, node):
            if isinstance(node.value, str) and node.value.startswith("AccelEp"):
                found.add(node.value)
            self.generic_visit(node)

    Visit().visit(tree)
    return found


def export_names(text):
    start = text.find('extern "C"')
    if start < 0:
        fail("src/ep.cc has no extern \"C\" block")
    names = []
    for line in text[start:].splitlines():
        if "ACCEL_EXPORT" not in line or "(" not in line:
            continue
        head = line.split("(", 1)[0].split()
        if not head:
            continue
        names.append(head[-1].lstrip("*&"))
    if "CreateEpFactories" not in names or "ReleaseEpFactory" not in names:
        fail(f"entry points missing from exports: {names}")
    return names


def main():
    ep_text = EP.read_text()
    header = HEADER.read_text()
    if not re.search(r"(?m)^#define ORT_API_VERSION 31$", header):
        fail("ORT_API_VERSION in the vendored header is not 31")
    if re.search(r"GetApi\(\s*\d+", ep_text):
        fail("GetApi must be passed ORT_API_VERSION, not an integer literal")
    for rhs in re.findall(r"ort_version_supported\s*=\s*([^;]+);", ep_text):
        token = rhs.strip()
        if re.fullmatch(r"\d+", token) and token != "31":
            fail(f"ort_version_supported uses {token}")

    exports = export_names(ep_text)
    walked = set()
    for path in walked_files():
        walked |= accel_names(path)
    missing = [name for name in exports if name not in ("CreateEpFactories", "ReleaseEpFactory") and name not in walked]
    if missing:
        fail("exports not named by a walked test: " + ", ".join(missing))

    lib = ctypes.CDLL(str(LIB), mode=ctypes.RTLD_LOCAL)
    for name in sorted(set(exports) | walked | {"CreateEpFactories", "ReleaseEpFactory"}):
        try:
            getattr(lib, name)
        except AttributeError:
            fail(f"dlsym missed {name}")
    print("export tests passed")


if __name__ == "__main__":
    sys.exit(main())
