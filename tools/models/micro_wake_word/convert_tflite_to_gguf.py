#!/usr/bin/env python3
"""Convert a trained microWakeWord TFLite model to audio.cpp GGUF.

The converter accepts the stateful MixedNet TFLite graph emitted by the
microWakeWord training project. It preserves tensor quantization metadata so
the native runtime can reproduce activation quantize/dequantize boundaries.
Weights are stored as F32; GGUF weight quantization is intentionally disabled
for this inference-only port.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import tflite
from safetensors.numpy import save_file


REPO = Path(__file__).resolve().parents[3]
SUPPORTED = {
    "CALL_ONCE", "VAR_HANDLE", "READ_VARIABLE", "ASSIGN_VARIABLE",
    "RESHAPE", "EXPAND_DIMS", "CONCATENATION", "STRIDED_SLICE", "SPLIT_V",
    "CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED", "LOGISTIC", "QUANTIZE",
}
TYPE_NAMES = {
    tflite.TensorType.FLOAT32: "f32",
    tflite.TensorType.INT8: "i8",
    tflite.TensorType.UINT8: "u8",
    tflite.TensorType.INT32: "i32",
    tflite.TensorType.RESOURCE: "resource",
}


def vector(obj, name: str) -> list[int]:
    length = getattr(obj, name + "Length")()
    getter = getattr(obj, name)
    return [int(getter(i)) for i in range(length)]


def tensor_array(model, tensor) -> np.ndarray:
    buffer = model.Buffers(tensor.Buffer())
    data = buffer.DataAsNumpy()
    if not isinstance(data, np.ndarray) or data.size == 0:
        raise ValueError(f"tensor has no constant buffer: {tensor.Name().decode()}")
    dtype = {
        tflite.TensorType.FLOAT32: np.dtype("<f4"),
        tflite.TensorType.INT8: np.dtype("i1"),
        tflite.TensorType.UINT8: np.dtype("u1"),
        tflite.TensorType.INT32: np.dtype("<i4"),
    }.get(tensor.Type())
    if dtype is None:
        raise ValueError(f"unsupported constant type {tensor.Type()}: {tensor.Name().decode()}")
    shape = vector(tensor, "Shape")
    return np.frombuffer(data.tobytes(), dtype=dtype).reshape(shape or ())


def quantization(tensor) -> dict[str, object] | None:
    q = tensor.Quantization()
    if q is None or q.ScaleLength() == 0:
        return None
    return {
        "scales": [float(q.Scale(i)) for i in range(q.ScaleLength())],
        "zero_points": [int(q.ZeroPoint(i)) for i in range(q.ZeroPointLength())],
        "dimension": int(q.QuantizedDimension()),
    }


def dequantize(value: np.ndarray, tensor) -> np.ndarray:
    if tensor.Type() == tflite.TensorType.FLOAT32:
        return np.asarray(value, dtype=np.float32)
    q = quantization(tensor)
    if q is None:
        raise ValueError(f"integer tensor lacks quantization: {tensor.Name().decode()}")
    scales = np.asarray(q["scales"], dtype=np.float32)
    zeros = np.asarray(q["zero_points"], dtype=np.float32)
    if scales.size == 1:
        return (value.astype(np.float32) - zeros[0]) * scales[0]
    shape = [1] * value.ndim
    shape[int(q["dimension"])] = scales.size
    return (value.astype(np.float32) - zeros.reshape(shape)) * scales.reshape(shape)


def builtin_options(op, cls):
    result = cls()
    table = op.BuiltinOptions()
    result.Init(table.Bytes, table.Pos)
    return result


def activation_name(value: int) -> str:
    names = {
        tflite.ActivationFunctionType.NONE: "none",
        tflite.ActivationFunctionType.RELU: "relu",
        tflite.ActivationFunctionType.RELU6: "relu6",
    }
    if value not in names:
        raise ValueError(f"unsupported fused activation enum: {value}")
    return names[value]


def read_small_constant(model, graph, index: int) -> list[int]:
    return [int(v) for v in tensor_array(model, graph.Tensors(index)).reshape(-1)]


def convert_graph(path: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    raw = path.read_bytes()
    model = tflite.Model.GetRootAsModel(raw, 0)
    if model.SubgraphsLength() < 1:
        raise ValueError("TFLite model has no inference subgraph")
    graph = model.Subgraphs(0)
    inputs = vector(graph, "Inputs")
    outputs = vector(graph, "Outputs")
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError("microWakeWord requires one feature input and one probability output")

    tensors: dict[str, np.ndarray] = {}
    tensor_meta: list[dict[str, object]] = []
    for index in range(graph.TensorsLength()):
        tensor = graph.Tensors(index)
        type_name = TYPE_NAMES.get(tensor.Type())
        if type_name is None:
            raise ValueError(f"unsupported tensor type {tensor.Type()} at index {index}")
        tensor_meta.append({
            "name": tensor.Name().decode("utf-8", errors="strict"),
            "shape": vector(tensor, "Shape"),
            "type": type_name,
            "quantization": quantization(tensor),
        })

    ops: list[dict[str, object]] = []
    variable_names: dict[int, str] = {}
    states: dict[int, dict[str, int | str]] = {}
    for op_index in range(graph.OperatorsLength()):
        op = graph.Operators(op_index)
        opcode = model.OperatorCodes(op.OpcodeIndex())
        name = tflite.opcode2name(opcode.BuiltinCode())
        if name not in SUPPORTED:
            raise ValueError(f"unsupported microWakeWord operator {name} at index {op_index}")
        ins, outs = vector(op, "Inputs"), vector(op, "Outputs")
        if name == "CALL_ONCE":
            continue
        if name == "VAR_HANDLE":
            variable_names[outs[0]] = tensor_meta[outs[0]]["name"]
            continue
        if name == "READ_VARIABLE":
            handle = ins[0]
            if handle not in variable_names:
                raise ValueError("READ_VARIABLE references an unknown handle")
            states[handle] = {"name": variable_names[handle], "read": outs[0], "assign": -1}
            continue
        if name == "ASSIGN_VARIABLE":
            handle = ins[0]
            if handle not in states:
                raise ValueError("ASSIGN_VARIABLE precedes READ_VARIABLE")
            states[handle]["assign"] = ins[1]
            continue

        row: dict[str, object] = {"type": name.lower(), "inputs": ins, "outputs": outs}
        if name in {"RESHAPE", "EXPAND_DIMS"}:
            row["shape"] = tensor_meta[outs[0]]["shape"]
        elif name == "CONCATENATION":
            options = builtin_options(op, tflite.ConcatenationOptions)
            row.update(axis=int(options.Axis()), activation=activation_name(options.FusedActivationFunction()))
        elif name == "STRIDED_SLICE":
            options = builtin_options(op, tflite.StridedSliceOptions)
            if any((options.EllipsisMask(), options.NewAxisMask(), options.ShrinkAxisMask())):
                raise ValueError("ellipsis/new-axis/shrink-axis STRIDED_SLICE is not supported")
            begin = read_small_constant(model, graph, ins[1])
            end = read_small_constant(model, graph, ins[2])
            strides = read_small_constant(model, graph, ins[3])
            input_shape = tensor_meta[ins[0]]["shape"]
            if len(begin) != len(input_shape) or any(value != 1 for value in strides):
                raise ValueError("microWakeWord STRIDED_SLICE requires rank-sized unit strides")
            for axis, size in enumerate(input_shape):
                if options.BeginMask() & (1 << axis):
                    begin[axis] = 0
                elif begin[axis] < 0:
                    begin[axis] += int(size)
                if options.EndMask() & (1 << axis):
                    end[axis] = int(size)
                elif end[axis] < 0:
                    end[axis] += int(size)
            row.update(begin=begin, end=end, strides=strides)
        elif name == "SPLIT_V":
            row.update(sizes=read_small_constant(model, graph, ins[1]),
                       axis=read_small_constant(model, graph, ins[2])[0])
        elif name in {"CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED"}:
            weight_id, bias_id = ins[1], ins[2]
            weight_tensor, bias_tensor = graph.Tensors(weight_id), graph.Tensors(bias_id)
            weight = dequantize(tensor_array(model, weight_tensor), weight_tensor)
            if name == "CONV_2D":
                weight = np.transpose(weight, (0, 3, 1, 2))
                options = builtin_options(op, tflite.Conv2DOptions)
                row.update(stride=[int(options.StrideH()), int(options.StrideW())],
                           dilation=[int(options.DilationHFactor()), int(options.DilationWFactor())],
                           padding=int(options.Padding()),
                           activation=activation_name(options.FusedActivationFunction()))
            elif name == "DEPTHWISE_CONV_2D":
                weight = np.transpose(weight, (3, 0, 1, 2))
                options = builtin_options(op, tflite.DepthwiseConv2DOptions)
                if options.DepthMultiplier() != 1:
                    raise ValueError("depthwise multiplier other than 1 is unsupported")
                row.update(stride=[int(options.StrideH()), int(options.StrideW())],
                           dilation=[int(options.DilationHFactor()), int(options.DilationWFactor())],
                           padding=int(options.Padding()),
                           activation=activation_name(options.FusedActivationFunction()))
            else:
                options = builtin_options(op, tflite.FullyConnectedOptions)
                row["activation"] = activation_name(options.FusedActivationFunction())
            weight_name, bias_name = f"op.{op_index}.weight", f"op.{op_index}.bias"
            tensors[weight_name] = np.ascontiguousarray(weight, dtype=np.float32)
            tensors[bias_name] = np.ascontiguousarray(
                dequantize(tensor_array(model, bias_tensor), bias_tensor), dtype=np.float32)
            row.update(weight=weight_name, bias=bias_name)
        ops.append(row)

    state_rows = [states[key] for key in sorted(states)]
    if not state_rows or any(int(row["assign"]) < 0 for row in state_rows):
        raise ValueError("microWakeWord graph has incomplete streaming state assignments")
    input_shape = tensor_meta[inputs[0]]["shape"]
    if len(input_shape) != 3 or input_shape[0] != 1 or input_shape[2] != 40:
        raise ValueError(f"expected feature input [1, frames, 40], got {input_shape}")
    config = {
        "format": "micro_wake_word_tflite_v1",
        "sample_rate": 16000,
        "feature_dim": 40,
        "feature_step_samples": 160,
        "input_frames": input_shape[1],
        "input_tensor": inputs[0],
        "output_tensor": outputs[0],
        "tensors": tensor_meta,
        "states": state_rows,
        "operators": ops,
    }
    return tensors, config


def manifest_values(path: Path | None) -> dict[str, object]:
    if path is None:
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    micro = value.get("micro", {})
    if not isinstance(micro, dict):
        raise ValueError("microWakeWord manifest field 'micro' must be an object")
    return {
        "wake_phrase": value.get("wake_word", value.get("name", "")),
        "probability_cutoff": micro.get("probability_cutoff", value.get("probability_cutoff", 0.5)),
        "sliding_window_size": micro.get("sliding_window_size", value.get("sliding_window_size", 5)),
        "feature_step_size": micro.get("feature_step_size", value.get("feature_step_size", 10)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="trained microWakeWord .tflite model")
    parser.add_argument("output", type=Path, help="output .gguf path")
    parser.add_argument("--manifest", type=Path, help="optional microWakeWord model JSON manifest")
    parser.add_argument("--wake-phrase", help="wake phrase label (overrides manifest)")
    parser.add_argument("--probability-cutoff", type=float, help="detection cutoff (overrides manifest)")
    parser.add_argument("--sliding-window-size", type=int, help="prediction smoothing window (overrides manifest)")
    parser.add_argument("--audiocpp-gguf", type=Path,
                        default=REPO / "build/debug/bin/audiocpp_gguf")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    model = args.model.resolve()
    output = args.output.resolve()
    if not model.is_file():
        raise SystemExit(f"TFLite model not found: {model}")
    if output.exists() and not args.overwrite:
        raise SystemExit(f"output exists (pass --overwrite): {output}")
    converter = args.audiocpp_gguf.resolve()
    if not converter.is_file():
        raise SystemExit(f"audiocpp_gguf not found: {converter}")

    weights, config = convert_graph(model)
    metadata = manifest_values(args.manifest.resolve() if args.manifest else None)
    if args.wake_phrase is not None:
        metadata["wake_phrase"] = args.wake_phrase
    if args.probability_cutoff is not None:
        metadata["probability_cutoff"] = args.probability_cutoff
    if args.sliding_window_size is not None:
        metadata["sliding_window_size"] = args.sliding_window_size
    phrase = str(metadata.get("wake_phrase", "")).strip()
    if not phrase:
        raise SystemExit("wake phrase is required through --wake-phrase or --manifest")
    cutoff = float(metadata.get("probability_cutoff", 0.5))
    window = int(metadata.get("sliding_window_size", 5))
    if not 0.0 <= cutoff <= 1.0 or window < 1:
        raise SystemExit("probability cutoff must be in [0, 1] and sliding window must be positive")
    if int(metadata.get("feature_step_size", 10)) != 10:
        raise SystemExit("only the standard 10 ms microWakeWord frontend step is supported")
    config.update(wake_phrase=phrase, probability_cutoff=cutoff, sliding_window_size=window,
                  source_model=model.name)

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="audiocpp-micro-wake-word-") as temporary:
        staging = Path(temporary)
        weights_path = staging / "model.safetensors"
        config_path = staging / "config.json"
        save_file(weights, str(weights_path), metadata={
            "format": "pt", "source": "microWakeWord TFLite", "audiocpp_family": "micro_wake_word"})
        config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        command = [str(converter), "--input", f"weights={weights_path}",
                   "--output", str(output), "--type", "orig", "--family", "micro_wake_word",
                   "--model-spec", str(REPO / "model_specs/micro_wake_word.json"),
                   "--root", str(staging)]
        if args.overwrite:
            command.append("--overwrite")
        print("+", " ".join(command))
        subprocess.run(command, check=True)
        shutil.copyfile(config_path, output.parent / "config.json")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
