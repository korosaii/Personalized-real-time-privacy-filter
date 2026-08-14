from __future__ import annotations

import argparse
from pathlib import Path

import onnx


def _set_dimension(dimension, value: int) -> None:
    dimension.ClearField("dim_param")
    dimension.dim_value = value


def make_static_model(source: Path, output: Path, size: int) -> None:
    model = onnx.load_model(source)
    input_shape = model.graph.input[0].type.tensor_type.shape.dim
    if len(input_shape) != 4:
        raise ValueError(f"Expected a four-dimensional input, got {len(input_shape)} dimensions")
    for dimension, value in zip(input_shape, (1, 3, size, size), strict=True):
        _set_dimension(dimension, value)

    outputs = list(model.graph.output)
    if len(outputs) != 9:
        raise ValueError(f"Expected a 9-output SCRFD_KPS graph, got {len(outputs)} outputs")
    counts = [2 * (size // stride) ** 2 for stride in (8, 16, 32)]
    for output_index, value_info in enumerate(outputs):
        shape = value_info.type.tensor_type.shape.dim
        _set_dimension(shape[0], counts[output_index % 3])

    onnx.checker.check_model(model)
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, output)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a fixed-resolution SCRFD graph for CoreML."
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--size", type=int, default=512)
    args = parser.parse_args()
    if args.size <= 0 or args.size % 32:
        raise SystemExit("--size must be a positive multiple of 32")
    make_static_model(args.source, args.output, args.size)
    print(f"Created {args.output} ({args.size}x{args.size})")


if __name__ == "__main__":
    main()
