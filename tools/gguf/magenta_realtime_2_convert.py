#!/usr/bin/env python3
"""Convert the official Magenta RealTime 2 checkpoints and MusicCoCa resources."""

import argparse
import hashlib
from fnmatch import fnmatchcase
import json
from pathlib import Path
import re
import subprocess
import tempfile

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file


def checkpoint_tensors(path):
    """Use channel-first convolutions and output-first linear projections."""
    result = {}
    with safe_open(path, framework="numpy") as source:
        for name in source.keys():
            value = source.get_tensor(name)
            name = name.removeprefix("params/")
            if name.startswith("depthformer/decoder/decoder_embedding/"):
                target = "token_embedding.weight"
            elif name.startswith("depthformer/decoder/depth_body/transformer/"):
                target = "depth." + name.split("/transformer/", 1)[1]
            elif name.startswith("depthformer/decoder/temporal_body/transformer/"):
                target = "temporal." + name.split("/transformer/", 1)[1]
            elif name.startswith("depthformer/decoder/depth_body/"):
                target = "depth." + name.split("/depth_body/", 1)[1]
            elif name.startswith("depthformer/encoder/body/"):
                target = "condition." + name.split("/body/", 1)[1]
                target = target.replace("layers_1/branched_mulan_embedder/mulan_embedder/", "style/")
                target = target.replace("layers_1/branched_regular_embedder/regular_embedder/", "controls/")
            elif name.startswith("soundstream/"):
                target = "codec." + name.removeprefix("soundstream/")
                if name == "soundstream/quantizer/embedding":
                    value = value[:12]
            else:
                raise ValueError(f"Unrecognized checkpoint tensor: {name}")
            target = re.sub(r"x_layers_(\d+)/", r"\1/", target)
            for before, after in (("self_attention/", "self/"), ("cross_attention/", "cross/"),
                                  ("attention/", ""), ("query_projection/", "q/"),
                                  ("key_projection/", "k/"), ("value_projection/", "v/"),
                                  ("output_projection/", "o/"), ("ffn_layer1/", "up/"),
                                  ("ffn_layer2/", "down/"), ("conv/", ""),
                                  ("shortcut_layer/", "shortcut/")):
                target = target.replace(before, after)
            if name.endswith("/kernel"):
                if value.ndim == 4:
                    value = value.transpose(3, 2, 0, 1)
                elif value.ndim == 3:
                    if "/output_projection/" in name:
                        value = value.reshape(value.shape[0], -1)
                    else:
                        value = value.reshape(value.shape[0], -1).T
                elif value.ndim == 2:
                    value = value.T
                else:
                    raise ValueError(f"Unexpected kernel shape: {name}: {value.shape}")
                target = target.removesuffix("kernel") + "weight"
            elif name.endswith("/embedding") and not target.endswith(".weight"):
                target = target.removesuffix("embedding") + "weight"
            elif name.endswith("/scale"):
                target = target.removesuffix("scale") + "weight"
            target = target.replace("/", ".")
            if target in result:
                raise ValueError(f"Duplicate tensor: {target}")
            result[target] = np.ascontiguousarray(value)
    return result


class MusicCoCaExport:
    """Resolve the checkpoint's static control flow before building GGML graphs."""

    def __init__(self, path):
        from ai_edge_litert import schema_py_generated as schema
        self.schema = schema
        self.data = path.read_bytes()
        self.model = schema.Model.GetRootAsModel(self.data, 0)
        self.names = {v: k for k, v in vars(schema.BuiltinOperator).items() if isinstance(v, int)}
        self.values = []
        self.constants = {}
        self.nodes = []
        self.types = {schema.TensorType.FLOAT32: np.float32, schema.TensorType.INT32: np.int32,
                      schema.TensorType.INT64: np.int64, schema.TensorType.BOOL: np.bool_}

    def value(self, shape, data=None, integer=False):
        index = len(self.values)
        self.values.append({"shape": list(shape), "integer": integer})
        if data is not None:
            self.constants[index] = np.asarray(data).reshape(shape)
        return index

    def options(self, op, name):
        cls = getattr(self.schema, name + "Options")
        result = cls()
        table = op.BuiltinOptions()
        result.Init(table.Bytes, table.Pos)
        return result

    def graph(self, graph_index=0, inputs=None):
        graph = self.model.Subgraphs(graph_index)
        mapping = dict(zip(graph.InputsAsNumpy().tolist(), inputs or []))
        for index in range(graph.TensorsLength()):
            if index in mapping:
                continue
            tensor = graph.Tensors(index)
            buffer = self.model.Buffers(tensor.Buffer())
            shape = tuple(int(x) for x in tensor.ShapeAsNumpy())
            data = None
            if buffer.DataLength():
                data = np.frombuffer(buffer.DataAsNumpy(), dtype=self.types[tensor.Type()]).reshape(shape)
            mapping[index] = self.value(shape, data, tensor.Type() in (self.schema.TensorType.INT32, self.schema.TensorType.INT64))
        for index in range(graph.OperatorsLength()):
            op = graph.Operators(index)
            name = self.names[self.model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()]
            source = [mapping[int(x)] if x >= 0 else -1 for x in op.InputsAsNumpy()]
            destinations = op.OutputsAsNumpy().tolist()
            if name == "WHILE":
                opts = self.options(op, "While")
                while True:
                    condition = self.graph(opts.CondSubgraphIndex(), source)[1][0]
                    if condition not in self.constants:
                        raise ValueError("MusicCoCa requires static loop bounds")
                    if not bool(self.constants[condition]):
                        break
                    source = self.graph(opts.BodySubgraphIndex(), source)[1]
                mapping.update(zip(destinations, source))
                continue
            attrs = {}
            values = [self.constants.get(x) for x in source]
            if name == "RESHAPE":
                shape = values[1].astype(int).tolist() if len(source) > 1 else self.options(op, "Reshape").NewShapeAsNumpy().tolist()
                attrs["shape"] = shape
            elif name == "TRANSPOSE":
                attrs["perm"] = values[1].astype(int).tolist()
            elif name == "CONCATENATION":
                opts = self.options(op, "Concatenation")
                attrs = {"axis": opts.Axis(), "activation": opts.FusedActivationFunction()}
            elif name == "PACK":
                attrs["axis"] = self.options(op, "Pack").Axis()
            elif name == "STRIDED_SLICE":
                opts = self.options(op, "StridedSlice")
                attrs = {"begin_mask": opts.BeginMask(), "end_mask": opts.EndMask(),
                         "shrink_axis_mask": opts.ShrinkAxisMask()}
            elif name == "GATHER":
                opts = self.options(op, "Gather")
                attrs = {"axis": opts.Axis(), "batch_dims": opts.BatchDims()}
            elif name == "BATCH_MATMUL":
                opts = self.options(op, "BatchMatMul")
                attrs = {"adj_x": opts.AdjX(), "adj_y": opts.AdjY()}
            elif name in ("SUM", "MEAN", "REDUCE_MAX"):
                attrs["keep_dims"] = self.options(op, "Reducer").KeepDims()
            elif name == "SOFTMAX":
                attrs["beta"] = self.options(op, "Softmax").Beta()
            elif name == "GELU":
                attrs["approximate"] = self.options(op, "Gelu").Approximate()
            elif name == "CUMSUM":
                opts = self.options(op, "Cumsum")
                attrs = {"exclusive": opts.Exclusive(), "reverse": opts.Reverse()}
            elif name in ("ADD", "SUB", "MUL", "DIV", "FULLY_CONNECTED"):
                opts = self.options(op, {"FULLY_CONNECTED": "FullyConnected"}.get(name, name.title()))
                attrs["activation"] = opts.FusedActivationFunction()
            result = self.fold(name, values, attrs) if all(x is not None or s < 0 for x, s in zip(values, source)) else None
            if result is not None:
                self.constants[mapping[destinations[0]]] = np.asarray(result).reshape(self.values[mapping[destinations[0]]]["shape"])
            else:
                self.nodes.append({"op": name, "inputs": source,
                                   "output": mapping[destinations[0]], **attrs})
        return [mapping[int(x)] for x in graph.InputsAsNumpy()], [mapping[int(x)] for x in graph.OutputsAsNumpy()]

    def fold(self, name, xs, attrs):
        binary = {"ADD": np.add, "SUB": np.subtract, "MUL": np.multiply, "DIV": np.divide,
                  "MAXIMUM": np.maximum, "MINIMUM": np.minimum, "LESS": np.less,
                  "GREATER_EQUAL": np.greater_equal}
        unary = {"SIN": np.sin, "COS": np.cos, "TANH": np.tanh, "RSQRT": lambda x: 1 / np.sqrt(x),
                 "SQRT": np.sqrt, "EXP": np.exp, "LOG": np.log, "SQUARE": np.square}
        if name in binary:
            return binary[name](*xs)
        if name in unary:
            return unary[name](xs[0])
        if name == "RESHAPE":
            return xs[0].reshape(attrs["shape"])
        if name == "TRANSPOSE":
            return xs[0].transpose(attrs["perm"])
        if name == "PACK":
            return np.stack(xs, axis=attrs["axis"])
        if name == "CONCATENATION":
            return np.concatenate(xs, axis=attrs["axis"])
        if name == "SLICE":
            return xs[0][tuple(slice(int(b), None if n == -1 else int(b + n)) for b, n in zip(xs[1], xs[2]))]
        if name == "GATHER":
            if attrs["batch_dims"]:
                raise ValueError("Unsupported constant batched gather")
            return np.take(xs[0], xs[1], axis=attrs["axis"])
        if name == "CAST":
            return xs[0].astype(np.float32)
        if name == "CUMSUM":
            value = np.flip(xs[0], int(xs[1])) if attrs["reverse"] else xs[0]
            result = np.cumsum(value, axis=int(xs[1]))
            if attrs["exclusive"]:
                result -= value
            return np.flip(result, int(xs[1])) if attrs["reverse"] else result
        return None

    def export(self, prefix, weights):
        inputs, outputs = self.graph()
        needed = set(outputs)
        nodes = []
        for node in reversed(self.nodes):
            if node["output"] in needed:
                nodes.append(node)
                needed.update(x for x in node["inputs"] if x >= 0)
        constants = {}
        hashes = {}
        for index in sorted(needed & self.constants.keys()):
            value = self.constants[index]
            if value.size <= 128:
                constants[str(index)] = {"values": value.reshape(-1).tolist(), "integer": value.dtype.kind in "ib"}
            else:
                value = np.ascontiguousarray(value, dtype=np.float32)
                digest = hashlib.sha256(value.tobytes()).hexdigest()
                name = hashes.get(digest)
                if name is None:
                    name = f"{prefix}.t{index}"
                    hashes[digest] = name
                    weights[name] = value.reshape((-1,) + value.shape[-3:]) if value.ndim > 4 else value
                constants[str(index)] = {"tensor": name}
        return {"inputs": inputs, "outputs": outputs, "values": self.values,
                "constants": constants, "nodes": list(reversed(nodes))}


def musiccoca_tensors(source, root):
    weights = {}
    graphs = {}
    for name in ("text_encoder", "mapper", "music_encoder"):
        graphs[name] = MusicCoCaExport(source / "resources/musiccoca" / f"{name}.tflite").export(name, weights)
    frontend = MusicCoCaExport(source / "resources/musiccoca/audio_preprocessor.tflite")
    graph = frontend.model.Subgraphs(0)
    for i in range(graph.TensorsLength()):
        tensor = graph.Tensors(i)
        name = tensor.Name().decode()
        if name.endswith("hann_window/sub_2") or name.endswith("linear_to_mel_weight_matrix/Maximum"):
            data = np.frombuffer(frontend.model.Buffers(tensor.Buffer()).DataAsNumpy(), dtype=np.float32).reshape(tensor.ShapeAsNumpy())
            if data.shape == (400,):
                weights["frontend.window"] = data
            elif data.shape == (1024, 128):
                weights["frontend.filters"] = np.ascontiguousarray(np.pad(data, ((1, 0), (0, 0))).T)
            else:
                raise ValueError(f"Unexpected MusicCoCa frontend tensor: {name} {data.shape}")
    if not all(name in weights for name in ("frontend.window", "frontend.filters")):
        raise ValueError("MusicCoCa frontend constants are missing")
    # RVQ codebooks are -0.5 times the fully-connected distance kernels.
    from ai_edge_litert.interpreter import Interpreter
    quantizer = Interpreter(model_path=str(source / "resources/musiccoca/pretrained_vector_quantizer.tflite"))
    quantizer.allocate_tensors()
    stages = [op for op in quantizer._get_ops_details() if op["op_name"] == "FULLY_CONNECTED"]
    if len(stages) != 12:
        raise ValueError("Expected 12 MusicCoCa quantizer stages")
    for i, op in enumerate(stages):
        weights[f"quantizer.{i}.weight"] = quantizer.get_tensor(op["inputs"][1]) * -0.5
        weights[f"quantizer.{i}.bias"] = quantizer.get_tensor(op["inputs"][2])
    (root / "musiccoca.json").write_text(json.dumps(graphs, separators=(",", ":")))
    (root / "spm.model").write_bytes((source / "resources/musiccoca/spm.model").read_bytes())
    return weights


def convert(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    spec = Path(__file__).resolve().parents[2] / "model_specs/magenta_realtime_2.json"
    with tempfile.TemporaryDirectory(prefix="magenta-convert-", dir=args.source) as temporary:
        root = Path(temporary)
        weights = checkpoint_tensors(args.source / "checkpoints" / f"mrt2_{args.component}.safetensors")
        config = {"variant": args.component, "sample_rate": 48000, "frame_rate": 25,
                  "temporal_layers": 12 if args.component == "small" else 20,
                  "temporal_dim": 1024 if args.component == "small" else 3072,
                  "temporal_heads": 8 if args.component == "small" else 24,
                  "temporal_ffn_dim": 4096 if args.component == "small" else 8192,
                  "condition_dim": 256 if args.component == "small" else 1024,
                  "window": 41 if args.component == "small" else 25,
                  "depth_dim": 768 if args.component == "small" else 1024,
                  "depth_heads": 6 if args.component == "small" else 8,
                  "depth_layers": 2 if args.component == "small" else 6,
                  "depth_ffn_dim": 3072 if args.component == "small" else 4096}
        (root / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        target = output / f"magenta-rt2-{args.component}-{args.type}.gguf"
        dtype = args.type
        sensitive_patterns = ("*.per_dim_scale", "*.sink*", "*.bias", "*norm.weight",
                              "*.encoder_ln.weight", "*.final_ln.weight")
        sensitive = sorted(name for name in weights
                           if any(fnmatchcase(name, pattern) for pattern in sensitive_patterns))
        save_file(weights, root / "generator.safetensors")
        del weights
        save_file(musiccoca_tensors(args.source, root), root / "musiccoca.safetensors")
        command = [str(args.converter.resolve()), "--input", f"generator={root / 'generator.safetensors'}",
                   "--input", f"musiccoca={root / 'musiccoca.safetensors'}",
                   "--root", str(root), "--output", str(target), "--type", dtype, "--log"]
        command += ["--model-spec", str(spec), "--keep-type", "generator/codec.*=f32",
                    "--keep-type", "musiccoca/*=f32"]
        for name in sensitive:
            command += ["--keep-type", f"generator/{name}=f32"]
        if args.overwrite:
            command.append("--overwrite")
        subprocess.run(command, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Official HF snapshot directory")
    parser.add_argument("--output", type=Path, required=True, help="Standalone GGUF package directory")
    parser.add_argument("--component", choices=("small", "base"), required=True)
    parser.add_argument("--type", choices=("orig", "f16", "bf16", "q8_0"), default="f16")
    parser.add_argument("--converter", type=Path, default=Path("build/debug/bin/audiocpp_gguf"))
    parser.add_argument("--overwrite", action="store_true")
    convert(parser.parse_args())
