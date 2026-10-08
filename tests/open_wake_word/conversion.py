"""Shared production and parity-trace GGUF writer for the official ONNX checkpoints."""

import json
from pathlib import Path

import gguf
import numpy as np
import onnx
from onnx import numpy_helper


def write_model(models, output, initial_features, traces=None):
    writer = gguf.GGUFWriter(str(output), "open_wake_word")
    tensors = {}
    frontend = onnx.load(models / "melspectrogram.onnx")
    weights = {tensor.name: numpy_helper.to_array(tensor) for tensor in frontend.graph.initializer}
    tensors["frontend.window"] = weights["0.stft.conv_real.weight"][0, 0].copy()
    tensors["frontend.mel_filter"] = weights["1.melW"].T.copy()
    tensors["runtime.initial_features"] = np.asarray(initial_features[-16:], dtype=np.float32)
    config = {"wake_phrase":"Alexa", "initial_suppression_frames":5}
    for name, filename in (("embedding", "embedding_model.onnx"), ("classifier", "alexa_v0.1.onnx")):
        graph = onnx.load(models / filename)
        for tensor in graph.graph.initializer:
            array = numpy_helper.to_array(tensor)
            if array.dtype == np.float32:
                tensors[f"{name}.{tensor.name}"] = array
        nodes = []
        for node in graph.graph.node:
            attributes = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
            attributes = {key: value.decode() if isinstance(value, bytes) else value for key, value in attributes.items()}
            nodes.append({"op":node.op_type, "input":list(node.input), "output":list(node.output), "attributes":attributes})
        config[f"{name}_graph"] = nodes
        writer.add_string(f"{name}.graph", json.dumps(nodes))
    writer.add_string("open_wake_word.config", json.dumps(config))
    if traces:
        for name, samples in traces.items():
            for index, (input_value, output_value) in enumerate(samples):
                tensors[f"reference.{name}.{index}.input"] = input_value.astype(np.float32)
                tensors[f"reference.{name}.{index}.output"] = output_value.astype(np.float32)
            writer.add_uint32(f"reference.{name}.count", len(samples))
    names, ranks, shapes = [], [], []
    for name, array in tensors.items():
        array = np.ascontiguousarray(array)
        writer.add_tensor(name, array)
        names.append(name); ranks.append(array.ndim); shapes.extend(array.shape)
    spec = (Path(__file__).resolve().parents[2] / "model_specs/open_wake_word.json").read_text()
    writer.add_uint32("audiocpp.model_spec.version", 1)
    writer.add_string("audiocpp.model_spec.family", "open_wake_word")
    writer.add_string("audiocpp.model_spec.json", spec)
    for key, values, kind in (("tensor_names", names, gguf.GGUFValueType.STRING),
                              ("tensor_ranks", ranks, gguf.GGUFValueType.INT32),
                              ("tensor_shapes", shapes, gguf.GGUFValueType.INT64)):
        writer.add_key_value("audiocpp." + key, values, gguf.GGUFValueType.ARRAY, kind)
    payload = json.dumps(config).encode()
    writer.add_key_value("audiocpp.embedded_files.names", ["config.json"], gguf.GGUFValueType.ARRAY, gguf.GGUFValueType.STRING)
    writer.add_key_value("audiocpp.embedded_files.offsets", [0, len(payload)], gguf.GGUFValueType.ARRAY, gguf.GGUFValueType.UINT64)
    writer.add_key_value("audiocpp.embedded_files.data", payload, gguf.GGUFValueType.ARRAY, gguf.GGUFValueType.UINT8)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
