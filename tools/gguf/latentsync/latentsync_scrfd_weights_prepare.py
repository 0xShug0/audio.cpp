"""Convert the fixed 512px SCRFD face detector used by LatentSync."""

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
    constants = {item.name: numpy_helper.to_array(item) for item in graph.initializer}
    shapes = {graph.input[0].name: [1, 3, 512, 512]}
    nodes, weights = [], {}
    for node in graph.node:
        op, inputs, output = node.op_type, list(node.input), node.output[0]
        attrs = {item.name: helper.get_attribute_value(item) for item in node.attribute}
        if op == "Shape":
            constants[output] = np.array(shapes[inputs[0]], dtype=np.int64)
            continue
        if op == "Gather":
            constants[output] = np.take(constants[inputs[0]], constants[inputs[1]], axis=attrs["axis"])
            continue
        if op == "Unsqueeze":
            constants[output] = np.expand_dims(constants[inputs[0]], tuple(attrs["axes"]))
            continue
        if op == "Slice":
            if constants[inputs[3]].tolist() != [0]:
                raise ValueError("Unexpected SCRFD shape slice")
            constants[output] = constants[inputs[0]][int(constants[inputs[1]][0]):int(constants[inputs[2]][0])]
            continue
        if op == "Concat":
            constants[output] = np.concatenate([np.atleast_1d(constants[key]) for key in inputs], axis=attrs["axis"])
            continue
        shape = list(shapes[inputs[0]])
        item = {"op": op, "inputs": inputs[:1], "output": output}
        prefix = f"face.scrfd.{output}"
        if op == "Conv":
            kernel = constants[inputs[1]].astype(np.float32)
            if attrs.get("group", 1) != 1 or attrs["pads"][:2] != attrs["pads"][2:]:
                raise ValueError("Unsupported SCRFD convolution")
            weights[prefix + ".weight"] = np.ascontiguousarray(kernel)
            weights[prefix + ".bias"] = np.ascontiguousarray(constants[inputs[2]], dtype=np.float32)
            item.update({"kernel": list(kernel.shape), "stride": attrs["strides"],
                         "padding": attrs["pads"][:2], "dilation": attrs["dilations"]})
            shape[1] = kernel.shape[0]
            for i in range(2):
                shape[i + 2] = (shape[i + 2] + 2 * attrs["pads"][i] - attrs["dilations"][i] * (kernel.shape[i + 2] - 1) - 1) // attrs["strides"][i] + 1
        elif op in ("MaxPool", "AveragePool"):
            if attrs["kernel_shape"] != [2, 2] or attrs["strides"] != [2, 2] or attrs["pads"] != [0, 0, 0, 0] or shape[2] % 2 or shape[3] % 2:
                raise ValueError("Unsupported SCRFD pooling layout")
            shape[2] //= 2
            shape[3] //= 2
        elif op == "Add":
            if shape != shapes[inputs[1]]:
                raise ValueError("SCRFD residual shape mismatch")
            item["inputs"] = inputs
        elif op == "Mul":
            if constants[inputs[1]].size != 1:
                raise ValueError("Expected SCRFD scalar scaling")
            weights[prefix + ".scale"] = constants[inputs[1]].astype(np.float32).reshape(1, 1, 1, 1)
        elif op == "Resize":
            if attrs["mode"] != b"nearest" or attrs["coordinate_transformation_mode"] != b"asymmetric" or attrs["nearest_mode"] != b"floor":
                raise ValueError("Unsupported SCRFD resize")
            shape = constants[inputs[3]].tolist()
        elif op == "Transpose":
            item["axes"] = attrs["perm"]
            shape = [shape[axis] for axis in attrs["perm"]]
        elif op == "Reshape":
            target = constants[inputs[1]].tolist()
            elements = int(np.prod(shape))
            if target.count(-1) == 1:
                target[target.index(-1)] = elements // -int(np.prod(target))
            if int(np.prod(target)) != elements:
                raise ValueError("Invalid SCRFD reshape")
            shape = target
        elif op not in ("Relu", "Sigmoid"):
            raise ValueError(f"Unsupported SCRFD operation: {op}")
        shapes[output] = shape
        item["shape"] = shape
        nodes.append(item)
    outputs = [value.name for value in graph.output]
    config = {"input": graph.input[0].name, "input_shape": [1, 3, 512, 512],
              "nodes": nodes, "outputs": outputs, "output_shapes": [shapes[key] for key in outputs]}
    args.output.mkdir(parents=True, exist_ok=True)
    save_file(weights, args.output / "weights.safetensors")
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"Converted SCRFD: {len(nodes)} nodes, {len(weights)} tensors; heads {config['output_shapes']}")


if __name__ == "__main__":
    main()
