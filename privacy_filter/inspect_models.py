from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import onnx


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _shape(value_info) -> list[int | str]:
    result: list[int | str] = []
    for dimension in value_info.type.tensor_type.shape.dim:
        if dimension.HasField("dim_value"):
            result.append(dimension.dim_value)
        elif dimension.HasField("dim_param"):
            result.append(dimension.dim_param)
        else:
            result.append("?")
    return result


def inspect_onnx(path: Path) -> None:
    model = onnx.load_model(path, load_external_data=False)
    opsets = {item.domain or "ai.onnx": item.version for item in model.opset_import}
    print("  container: ONNX protobuf")
    print(f"  producer: {model.producer_name or 'unknown'} {model.producer_version}".rstrip())
    print(f"  IR/opsets: {model.ir_version} / {opsets}")
    print(f"  nodes: {len(model.graph.node)}")
    print("  inputs:")
    for item in model.graph.input:
        print(f"    - {item.name}: {_shape(item)}")
    print("  outputs:")
    for item in model.graph.output:
        print(f"    - {item.name}: {_shape(item)}")


def inspect_path(path: Path) -> None:
    print(f"{path}:")
    if not path.is_file():
        print("  status: MISSING")
        return
    print(f"  size: {path.stat().st_size:,} bytes")
    print(f"  sha256: {_sha256(path)}")
    if path.suffix.lower() == ".onnx":
        inspect_onnx(path)
    elif path.suffix.lower() in {".pth", ".pt"}:
        with path.open("rb") as checkpoint:
            magic = checkpoint.read(2)
        protocol = magic[1] if len(magic) == 2 and magic[0] == 0x80 else "unknown"
        print(f"  container: PyTorch/Pickle training checkpoint (pickle protocol {protocol})")
        print("  usable by this app: no — convert or use a detector ONNX export")
        print("  safety: not unpickled; torch.load can execute code from an untrusted file")
    else:
        print("  container: unknown")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect local ONNX and checkpoint model metadata safely."
    )
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.paths:
        inspect_path(path.expanduser().resolve())


if __name__ == "__main__":
    main()
