from __future__ import annotations

import argparse
from pathlib import Path

import onnx


def _set_dimension(dimension, value: int) -> None:
    dimension.ClearField("dim_param")
    dimension.dim_value = value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fix the recognition ONNX batch dimension to one for CoreML."
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    model = onnx.load_model(args.source)
    input_shape = model.graph.input[0].type.tensor_type.shape.dim
    if len(input_shape) != 4:
        raise SystemExit("Recognition input must be four-dimensional")
    for dimension, value in zip(input_shape, (1, 3, 112, 112), strict=True):
        _set_dimension(dimension, value)
    output_shape = model.graph.output[0].type.tensor_type.shape.dim
    if len(output_shape) == 2:
        _set_dimension(output_shape[0], 1)
        _set_dimension(output_shape[1], 512)
    onnx.checker.check_model(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, args.output)
    print(f"Created {args.output} with static batch=1")


if __name__ == "__main__":
    main()
