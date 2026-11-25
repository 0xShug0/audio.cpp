# /// script
# requires-python = ">=3.10"
# dependencies = ["gguf>=0.19", "numpy>=1.26", "safetensors>=0.5", "torch>=2.6"]
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# ///
"""Convert Drama 3's scaled FP8 experts into a native audio.cpp GGUF package.

Uses CPU memory one expert bank at a time. The optional native converter stage
folds codec weight norms, embeds tokenizer/config files, and applies mixed
quantization. It never loads weights onto a GPU.
"""

import argparse
import json
import math
import struct
import subprocess
from pathlib import Path

import gguf
import numpy as np
import torch
from safetensors import safe_open

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--output", type=Path, help="Intermediate FP16 GGUF (about 68 GiB)")
parser.add_argument("--input-gguf", type=Path, help="Reuse a prepared FP16 GGUF instead of exporting again")
parser.add_argument("--quantized-output", type=Path, help="Standalone mixed-quantization GGUF")
parser.add_argument("--gguf-tool", type=Path, help="Path to the native audiocpp_gguf executable")
parser.add_argument("--expert-type", choices=("q2_k", "q3_k", "q4_k", "q5_k", "q6_k", "q8_0"), default="q4_k")
parser.add_argument("--backbone-type", choices=("q2_k", "q3_k", "q4_k", "q5_k", "q6_k", "q8_0"), default="q8_0",
                    help="Dense backbone matrix quantization; audio tower and codec retain their preparation types")
parser.add_argument("--embedding-type", choices=("f16", "q4_0", "q4_1", "q5_0", "q5_1", "q8_0"), default="q8_0",
                    help="Main text embedding storage; defaults to Q8_0 and exposes CUDA-supported row lookup types")
parser.add_argument("--threads", type=int, default=8)
arguments = parser.parse_args()
if arguments.threads <= 0:
    parser.error("--threads must be positive")
if arguments.input_gguf and arguments.output:
    parser.error("choose --output or --input-gguf")
if not arguments.input_gguf and not arguments.output:
    parser.error("provide --output or --input-gguf")
if arguments.input_gguf and not arguments.quantized_output:
    parser.error("--input-gguf requires --quantized-output")
if arguments.quantized_output and (not arguments.gguf_tool or not arguments.gguf_tool.is_file()):
    parser.error("--quantized-output requires an existing --gguf-tool executable")
if arguments.output and arguments.quantized_output and arguments.output.resolve() == arguments.quantized_output.resolve():
    parser.error("intermediate and quantized outputs must be different files")
for destination in (arguments.output, arguments.quantized_output):
    if destination and destination.exists():
        parser.error(f"output already exists: {destination}")
torch.set_num_threads(arguments.threads)
configuration = json.loads((arguments.checkpoint / "config.json").read_text())
if configuration.get("model_type") != "fish_omni":
    parser.error("checkpoint config must describe Fish Drama 3 (fish_omni)")


def quantize_archive(archive):
    if not arguments.quantized_output:
        return
    repository = Path(__file__).resolve().parents[2]
    command = [str(arguments.gguf_tool.resolve()), "--input", str(archive.resolve()),
               "--output", str(arguments.quantized_output.resolve()), "--type", arguments.backbone_type,
               "--family", "fish_drama3", "--model-spec", str(repository / "model_specs/fish_drama3.json"),
               "--root", str(arguments.checkpoint.resolve()),
               "--fold-weight-norm", "codec_weights*", "--keep-type", "codec_weights*=f16",
               "--keep-type", "model_weights/audio_decoder*=orig",
               "--keep-type", "model_weights/model.language_model.embed_tokens.weight=" + arguments.embedding_type]
    layers = configuration["text_config"]["num_hidden_layers"]
    for layer in range(layers):
        command += ["--keep-type", f"model_weights/model.language_model.layers.{layer}.mlp.experts.*={arguments.expert_type}"]
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "LICENSE", "chat_template.jinja"):
        path = arguments.checkpoint / name
        if path.is_file():
            command += ["--sidecar", f"{path.resolve()}={name}"]
    subprocess.run(command, check=True)


if arguments.input_gguf:
    if not arguments.input_gguf.is_file():
        parser.error(f"input GGUF does not exist: {arguments.input_gguf}")
    quantize_archive(arguments.input_gguf)
    raise SystemExit(0)

weight_map = json.loads((arguments.checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
layer_count = configuration["text_config"]["num_hidden_layers"]
expert_count = configuration["text_config"]["num_experts"]
expert_width = configuration["text_config"]["moe_intermediate_size"]
hidden_width = configuration["text_config"]["hidden_size"]
shards = {}
headers = {}
for shard_name in dict.fromkeys(weight_map.values()):
    shards[shard_name] = safe_open(arguments.checkpoint / shard_name, framework="pt", device="cpu")
    with (arguments.checkpoint / shard_name).open("rb") as shard_file:
        header_length = struct.unpack("<Q", shard_file.read(8))[0]
        headers.update(json.loads(shard_file.read(header_length)))


def load_tensor(tensor_name):
    tensor = shards[weight_map[tensor_name]].get_tensor(tensor_name)
    if tensor.dtype == torch.float8_e4m3fn:
        scale = shards[weight_map[tensor_name + "_scale_inv"]].get_tensor(tensor_name + "_scale_inv")
        tensor = tensor.float() * scale.repeat_interleave(128, 0).repeat_interleave(128, 1)[:tensor.shape[0], :tensor.shape[1]]
    return tensor.to(torch.float32 if tensor.ndim <= 1 or tensor.dtype == torch.float32 else torch.float16).numpy()


tensor_plan = []
for tensor_name in weight_map:
    if ".experts." in tensor_name or tensor_name.endswith("_scale_inv"):
        continue
    tensor_header = headers[tensor_name]
    tensor_shape = tensor_header["shape"]
    tensor_dtype = np.float32 if len(tensor_shape) <= 1 or tensor_header["dtype"] == "F32" else np.float16
    tensor_plan.append(("model_weights/" + tensor_name, tensor_shape, tensor_dtype, tensor_name))
for layer_index in range(layer_count):
    for projection_name in ("gate_proj", "up_proj", "down_proj"):
        matrix_shape = (expert_width, hidden_width) if projection_name != "down_proj" else (hidden_width, expert_width)
        tensor_name = f"model.language_model.layers.{layer_index}.mlp.experts.{projection_name}.weight"
        tensor_plan.append(("model_weights/" + tensor_name, (expert_count, *matrix_shape), np.float16, (layer_index, projection_name)))

codec = torch.load(arguments.checkpoint / "codec.pth", map_location="cpu", weights_only=True)
codec = codec.get("state_dict", codec)
codec = {name.removeprefix("generator."): value for name, value in codec.items()}
folded_codec = {}
for tensor_name, tensor in codec.items():
    if tensor_name.endswith("weight_g"):
        continue
    if tensor_name.endswith("weight_v"):
        scale = codec[tensor_name.removesuffix("weight_v") + "weight_g"]
        tensor = tensor * (scale / torch.linalg.vector_norm(tensor, dim=tuple(range(1, tensor.ndim)), keepdim=True))
        tensor_name = tensor_name.removesuffix("weight_v") + "weight"
    if not isinstance(tensor, torch.Tensor):
        continue
    tensor = tensor.to(torch.float32 if tensor.ndim <= 1 else torch.float16).contiguous()
    folded_codec[tensor_name] = tensor.numpy()
    tensor_plan.append(("codec_weights/" + tensor_name, tuple(tensor.shape), folded_codec[tensor_name].dtype, None))

arguments.output.parent.mkdir(parents=True, exist_ok=True)
writer = gguf.GGUFWriter(arguments.output, "fish_drama3")
writer.add_name("Fish Audio Drama 3")
writer.add_string("audiocpp.family", "fish_drama3")
writer.add_string("audiocpp.source_model", "fishaudio/drama-3")
writer.add_array("audiocpp.tensor_names", [tensor_name for tensor_name, _, _, _ in tensor_plan])
writer.add_key_value("audiocpp.tensor_ranks", [len(shape) for _, shape, _, _ in tensor_plan], gguf.GGUFValueType.ARRAY, gguf.GGUFValueType.INT32)
writer.add_key_value("audiocpp.tensor_shapes", [dimension for _, shape, _, _ in tensor_plan for dimension in shape], gguf.GGUFValueType.ARRAY, gguf.GGUFValueType.INT64)
for tensor_index, (tensor_name, tensor_shape, tensor_dtype, _) in enumerate(tensor_plan):
    writer.add_tensor_info(f"_audiocpp.{tensor_index}", tensor_shape, np.dtype(tensor_dtype), math.prod(tensor_shape) * np.dtype(tensor_dtype).itemsize)
writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_ti_data_to_file()
for tensor_index, (tensor_name, tensor_shape, tensor_dtype, source_name) in enumerate(tensor_plan):
    if tensor_name.startswith("codec_weights/"):
        tensor_values = folded_codec[tensor_name.removeprefix("codec_weights/")]
    elif isinstance(source_name, tuple):
        layer_index, projection_name = source_name
        tensor_values = np.empty(tensor_shape, dtype=tensor_dtype)
        for expert_index in range(expert_count):
            source_tensor_name = f"model.language_model.layers.{layer_index}.mlp.experts.{expert_index}.{projection_name}.weight"
            tensor_values[expert_index] = load_tensor(source_tensor_name)
    else:
        tensor_values = load_tensor(source_name)
    if not np.isfinite(tensor_values).all():
        raise RuntimeError(f"Non-finite values in {tensor_name}")
    writer.write_tensor_data(tensor_values)
    if tensor_index % 20 == 0 or isinstance(source_name, tuple):
        print(f"{tensor_index + 1}/{len(tensor_plan)} {tensor_name}", flush=True)
writer.close()
print(f"Wrote {arguments.output} with {len(tensor_plan)} tensors", flush=True)
quantize_archive(arguments.output)
