"""Convert LatentSync's InsightFace 106-landmark CNN, folding inference batch norms."""

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper
from safetensors.numpy import save_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    graph = onnx.load(args.checkpoint).graph
    source = {value.name: numpy_helper.to_array(value) for value in graph.initializer}
    nodes = list(graph.node)
    if [node.op_type for node in nodes[:2]] != ["Sub", "Mul"]:
        raise ValueError("Unexpected landmark input normalization")
    config = {"input_offset": float(source[nodes[0].input[1]].item()),
              "input_scale": float(source[nodes[1].input[1]].item()), "layers": []}
    weights = {}
    index = 2
    while nodes[index].op_type == "Conv":
        conv, norm, activation = nodes[index:index + 3]
        if norm.op_type != "BatchNormalization" or activation.op_type != "PRelu":
            raise ValueError("Unexpected landmark convolution block")
        attributes = {item.name: helper.get_attribute_value(item) for item in conv.attribute}
        norm_attributes = {item.name: helper.get_attribute_value(item) for item in norm.attribute}
        kernel = source[conv.input[1]].astype(np.float32)
        bias = source[conv.input[2]].astype(np.float32) if len(conv.input) == 3 else np.zeros(kernel.shape[0], np.float32)
        gamma, beta, mean, variance = (source[key].astype(np.float32) for key in norm.input[1:])
        scale = gamma / np.sqrt(variance + norm_attributes["epsilon"])
        prefix = f"face.landmarks.layers.{len(config['layers'])}"
        weights[prefix + ".conv.weight"] = np.ascontiguousarray(kernel * scale[:, None, None, None])
        weights[prefix + ".conv.bias"] = np.ascontiguousarray((bias - mean) * scale + beta)
        slope = source[activation.input[1]].astype(np.float32).reshape(1, -1, 1, 1)
        weights[prefix + ".slope"] = slope
        weights[prefix + ".positive_scale"] = 1 - slope
        groups = attributes.get("group", 1)
        if groups not in (1, kernel.shape[0]) or attributes["pads"][:2] != attributes["pads"][2:]:
            raise ValueError("Unsupported landmark convolution grouping or asymmetric padding")
        config["layers"].append({"in_channels": int(kernel.shape[1] * groups),
            "out_channels": int(kernel.shape[0]), "kernel": list(kernel.shape[2:]),
            "stride": attributes["strides"], "padding": attributes["pads"][:2],
            "dilation": attributes["dilations"], "groups": groups})
        index += 3
    if [node.op_type for node in nodes[index:]] != ["Flatten", "Flatten", "Gemm"]:
        raise ValueError("Unexpected landmark output projection")
    fc = nodes[-1]
    attributes = {item.name: helper.get_attribute_value(item) for item in fc.attribute}
    if attributes != {"alpha": 1.0, "beta": 1.0, "transA": 0, "transB": 1}:
        raise ValueError("Unexpected landmark linear projection layout")
    weights["face.landmarks.output.weight"] = np.ascontiguousarray(source[fc.input[1]])
    weights["face.landmarks.output.bias"] = np.ascontiguousarray(source[fc.input[2]])
    args.output.mkdir(parents=True, exist_ok=True)
    save_file(weights, args.output / "weights.safetensors")
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"Converted {len(config['layers'])} landmark blocks, {len(weights)} tensors")


if __name__ == "__main__":
    main()
