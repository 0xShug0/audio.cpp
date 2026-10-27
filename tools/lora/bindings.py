"""Model-local adapter names bound to complete audio.cpp GGUF tensor names."""

from dataclasses import dataclass
import json
from pathlib import Path
import re
import shutil

import gguf
import numpy as np
import torch
from safetensors import safe_open

from checkpoint import EXTENSIONS


FAMILIES = (
    "qwen3_tts", "qwen3_asr", "voxcpm2", "voxcpm1", "fish_audio",
    "cosyvoice3", "f5_tts", "outetts", "yue2", "ace_step", "vibevoice", "audio_flamingo",
)

MERGE_MODES = {"yue2": "rounded_bf16_delta", "vibevoice": "rankwise_f32"}

COMPONENTS = {
    "qwen3_tts": "model_weights", "voxcpm2": "weights",
    "fish_audio": "model_weights", "cosyvoice3": "llm",
    "f5_tts": "transformer", "outetts": "model_weights", "audio_flamingo": "weights",
}


@dataclass(frozen=True)
class Target:
    name: str
    first_row: int = 0
    rows: int = 0


class Bindings:
    def __init__(self, family, shapes, component=None):
        self.targets = {}
        component = component or COMPONENTS.get(family)
        names = list(shapes)
        if component:
            names = [name for name in names if name.startswith(component + "/")]
            if not names:
                raise ValueError(f"Base GGUF has no {component!r} component")
        for name in names:
            logical = name.split("/", 1)[-1]
            target = (Target(name),)
            self.add(name, target)
            self.add(logical, target)
            if family == "qwen3_tts" and logical.startswith("talker."):
                self.add(logical.removeprefix("talker."), target)
            elif family == "qwen3_asr":
                if logical.startswith("thinker."):
                    self.add(logical.removeprefix("thinker."), target)
                elif logical.startswith("model.language_model."):
                    self.add("model." + logical.removeprefix("model.language_model."), target)
                elif logical.startswith("model.audio_tower."):
                    self.add(logical.removeprefix("model."), target)
            elif family == "cosyvoice3" and logical.startswith("llm.model."):
                self.add(logical.removeprefix("llm.model."), target)
            elif family == "vibevoice" and logical.startswith("model.language_model."):
                self.add(logical.removeprefix("model.language_model."), target)
            elif family == "yue2" and logical.startswith("model."):
                self.add(logical.removeprefix("model."), target)
            elif family == "ace_step" and logical.startswith("decoder."):
                self.add(logical.removeprefix("decoder."), target)
            elif family == "fish_audio":
                self.add("model." + logical, target)
            elif family == "f5_tts":
                stripped = logical.removeprefix("ema_model.transformer.")
                self.add(stripped, target)
                self.add("transformer." + stripped, target)
            elif family == "voxcpm1":
                self.add_voxcpm1(logical, target)

        if family in {"fish_audio", "yue2"}:
            for name in names:
                if name.endswith(".q_proj.weight"):
                    prefix = name.removesuffix(".q_proj.weight")
                    projections = [prefix + "." + part + ".weight" for part in ("q_proj", "k_proj", "v_proj")]
                    fused = ".wqkv" if family == "fish_audio" else ".qkv_proj"
                elif family == "yue2" and name.endswith(".gate_proj.weight"):
                    prefix = name.removesuffix(".gate_proj.weight")
                    projections = [prefix + "." + part + ".weight" for part in ("gate_proj", "up_proj")]
                    fused = ".gate_up_proj"
                else:
                    continue
                offset = 0
                targets = []
                for projection in projections:
                    rows = shapes[projection][0]
                    targets.append(Target(projection, offset, rows))
                    offset += rows
                logical = (prefix + fused).split("/", 1)[-1]
                self.add(logical, tuple(targets))
                if family == "fish_audio":
                    self.add("model." + logical, tuple(targets))
                else:
                    self.add(logical.removeprefix("model."), tuple(targets))

    def add(self, name, targets):
        name = name.removesuffix(".weight")
        for alias in {name, name.replace(".", "_")}:
            if alias in self.targets and self.targets[alias] != targets:
                raise ValueError(f"Ambiguous adapter binding {alias!r}; select --component")
            self.targets[alias] = targets

    def resolve(self, name):
        for prefix in ("base_model.model.", "text_encoders.", "lora_unet_", "lora_transformer_", "lycoris_"):
            if name.startswith(prefix):
                name = name.removeprefix(prefix)
                break
        name = name.replace(".modules_to_save.default", "")
        key = name.removesuffix(".weight")
        if key not in self.targets:
            raise ValueError(f"Adapter target has no model binding: {name}")
        return self.targets[key]

    def add_voxcpm1(self, name, target):
        match = re.fullmatch(
            r"(?:(residual_lm|locenc|locdit)\.)?blk\.(\d+)\."
            r"(attn_q|attn_k|attn_v|attn_output|attn_norm|ffn_norm|ffn_gate|ffn_up|ffn_down)\.weight", name)
        if match:
            prefix = {None: "base_lm", "residual_lm": "residual_lm", "locenc": "feat_encoder.encoder",
                      "locdit": "feat_decoder.estimator.decoder"}[match[1]]
            projection = {
                "attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj", "attn_v": "self_attn.v_proj",
                "attn_output": "self_attn.o_proj", "attn_norm": "input_layernorm",
                "ffn_norm": "post_attention_layernorm", "ffn_gate": "mlp.gate_proj",
                "ffn_up": "mlp.up_proj", "ffn_down": "mlp.down_proj",
            }[match[3]]
            self.add(f"{prefix}.layers.{match[2]}.{projection}", target)
            return
        replacements = {
            "token_embd": "base_lm.embed_tokens", "output_norm": "base_lm.norm",
            "residual_lm.output_norm": "residual_lm.norm", "locenc.output_norm": "feat_encoder.encoder.norm",
            "locdit.output_norm": "feat_decoder.estimator.decoder.norm",
            "proj.enc_to_lm": "enc_to_lm_proj", "proj.lm_to_dit": "lm_to_dit_proj",
            "proj.res_to_dit": "res_to_dit_proj", "stop.stop_proj": "stop_proj", "stop.stop_head": "stop_head",
        }
        for suffix in (".weight", ".bias"):
            if not name.endswith(suffix):
                continue
            module = name.removesuffix(suffix)
            if module in replacements:
                self.add(replacements[module] + suffix, target)
            for internal, external in (("locenc.", "feat_encoder."), ("locdit.", "feat_decoder.estimator."),
                                       ("fsq.", "fsq_layer.")):
                if module.startswith(internal):
                    self.add(external + module.removeprefix(internal) + suffix, target)


class GGUFReader(gguf.GGUFReader):
    def _get_field_parts(self, offset, raw_type):
        # Embedded sidecars are byte arrays; one NumPy object per byte is costly.
        if raw_type == gguf.GGUFValueType.ARRAY:
            item_type = self._get(offset, np.uint32)
            if item_type[0] == gguf.GGUFValueType.UINT8:
                count = self._get(offset + 4, np.uint64)
                values = self._get(offset + 12, np.uint8, int(count[0]))
                return 12 + values.nbytes, [item_type, count, values], [2], [
                    gguf.GGUFValueType.ARRAY, gguf.GGUFValueType.UINT8]
        return super()._get_field_parts(offset, raw_type)


class BaseGGUF:
    def __init__(self, path):
        self.reader = GGUFReader(path)
        fields = self.reader.fields
        if "audiocpp.tensor_names" in fields:
            names = fields["audiocpp.tensor_names"].contents()
            ranks = fields["audiocpp.tensor_ranks"].contents()
            dimensions = fields["audiocpp.tensor_shapes"].contents()
            self.shapes = {}
            offset = 0
            for name, rank in zip(names, ranks, strict=True):
                self.shapes[name] = tuple(dimensions[offset:offset + rank])
                offset += rank
            if offset != len(dimensions):
                raise ValueError("Base GGUF logical shape metadata is inconsistent")
        else:
            names = [tensor.name for tensor in self.reader.tensors]
            self.shapes = {tensor.name: tuple(reversed(tensor.shape.tolist())) for tensor in self.reader.tensors}
        self.tensors = dict(zip(names, self.reader.tensors, strict=True))

    def read_f32(self, name):
        tensor = self.tensors[name]
        values = gguf.quants.dequantize(tensor.data, tensor.tensor_type)
        return torch.from_numpy(np.asarray(values, dtype=np.float32).reshape(self.shapes[name]).copy())

    def extract_sidecars(self, destination):
        fields = self.reader.fields
        names = fields["audiocpp.embedded_files.names"].contents()
        if "audiocpp.embedded_files.data" in fields:
            field = fields["audiocpp.embedded_files.data"]
            first = field.parts[field.data[0]]
            offset = first.__array_interface__["data"][0] - self.reader.data.__array_interface__["data"][0]
            data = self.reader.data[offset:offset + int(field.parts[4][0])]
            offsets = fields["audiocpp.embedded_files.offsets"].contents()
            contents = (memoryview(data)[offsets[i]:offsets[i + 1]] for i in range(len(names)))
        else:
            contents = (value.encode("utf-8") for value in fields["audiocpp.embedded_files.contents"].contents())
        for name, content in zip(names, contents, strict=True):
            target = destination / name
            if not target.resolve().is_relative_to(destination.resolve()):
                raise ValueError(f"Unsafe embedded file name: {name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)


class BaseSources:
    def __init__(self, inputs, root):
        self.root = root
        self.shapes = {}
        self.tensors = {}
        for value in inputs:
            prefix, separator, filename = value.partition("=")
            path = Path(filename if separator else prefix)
            prefix = prefix if separator else ""
            if path.name.endswith(".safetensors.index.json"):
                mapping = json.loads(path.read_text())["weight_map"]
                shards = {name: path.parent / shard for name, shard in mapping.items()}
            elif path.suffix == ".safetensors":
                with safe_open(path, framework="pt", device="cpu") as source:
                    shards = {name: path for name in source.keys()}
            else:
                raise ValueError(f"Tensor inputs must be safetensors or a shard index: {path}")
            for shard in sorted(set(shards.values())):
                with safe_open(shard, framework="pt", device="cpu") as source:
                    for name, file in shards.items():
                        if file != shard:
                            continue
                        logical = prefix + "/" + name if prefix else name
                        if logical in self.shapes:
                            raise ValueError(f"Duplicate input tensor: {logical}")
                        self.shapes[logical] = tuple(source.get_slice(name).get_shape())
                        self.tensors[logical] = (shard, name)

    def read_f32(self, name):
        path, key = self.tensors[name]
        with safe_open(path, framework="pt", device="cpu") as source:
            return source.get_tensor(key).float().clone()

    def extract_sidecars(self, destination):
        for path in self.root.rglob("*"):
            relative = path.relative_to(self.root)
            if any(part in {".git", ".cache"} for part in relative.parts):
                continue
            if not path.is_file() or path.suffix.lower() in (*EXTENSIONS, ".gguf"):
                continue
            if path.stat().st_size > 64 * 1024 * 1024:
                continue
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
