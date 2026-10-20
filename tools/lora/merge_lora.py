#!/usr/bin/env python3
"""Bind a model's LoRA to its base GGUF and merge with audiocpp_gguf."""

import argparse
import json
from pathlib import Path
import re
import subprocess
import tempfile

import torch
from safetensors.torch import save_file

from bindings import BaseGGUF, BaseSources, Bindings, FAMILIES, MERGE_MODES
from checkpoint import find_checkpoint, load_checkpoint
from prepare_fish_audio_adapter import prepare as prepare_fish


SUFFIXES = (
    (".lora_A.default.weight", "a"), (".lora_B.default.weight", "b"),
    (".lora_A.weight", "a"), (".lora_B.weight", "b"),
    (".lora_A", "a"), (".lora_B", "b"),
    (".lora_down.weight", "a"), (".lora_up.weight", "b"),
    (".lokr_w1", "lokr_w1"), (".lokr_w2", "lokr_w2"),
    (".lokr_w1_a", "lokr_w1_a"), (".lokr_w1_b", "lokr_w1_b"),
    (".lokr_w2_a", "lokr_w2_a"), (".lokr_w2_b", "lokr_w2_b"),
    (".alpha", "alpha"),
)


def read_adapter(path, family, work, alpha):
    directory = path if path.is_dir() else path.parent
    native_config = None
    native_cosy = False
    if family == "cosyvoice3":
        native_package = (
            find_checkpoint(path, "lora_weights", required=False) is not None
            if path.is_dir() else path.stem == "lora_weights")
        native_cosy = (directory / "config.json").is_file() and (
            native_package or not (directory / "adapter_config.json").is_file())
    if path.is_dir():
        peft_file = find_checkpoint(path, "adapter_model", required=False) if not native_cosy else None
        if native_cosy:
            path = find_checkpoint(path, "lora_weights")
        elif peft_file is not None:
            path = peft_file
        else:
            stem = "lora_weights" if family in {"voxcpm1", "voxcpm2"} else "adapter_model"
            path = find_checkpoint(path, stem)
    if family == "fish_audio" and alpha is not None:
        prepared = work / "fish.safetensors"
        prepare_fish(path, prepared, alpha)
        path = prepared
    if family in {"voxcpm1", "voxcpm2"}:
        native_config = json.loads((directory / "lora_config.json").read_text())["lora_config"]
        config = {"r": native_config["r"], "lora_alpha": native_config["alpha"]}
    elif native_cosy:
        native_config = json.loads((directory / "config.json").read_text())["config"]
        config = {"r": native_config["lora_r"], "lora_alpha": native_config["lora_alpha"]}
    else:
        config_file = directory / "adapter_config.json"
        config = json.loads(config_file.read_text()) if config_file.is_file() else {}
    state = load_checkpoint(path)
    if not state:
        raise ValueError(f"Adapter checkpoint is empty: {path}")
    peft_suffixes = (".lora_A.weight", ".lora_B.weight", ".lora_A.default.weight", ".lora_B.default.weight")
    has_peft = any(name.endswith(peft_suffixes) for name in state)
    if has_peft and native_config is None and not (directory / "adapter_config.json").is_file():
        for name in state:
            for suffix in peft_suffixes:
                if name.endswith(suffix) and name.removesuffix(suffix) + ".alpha" not in state:
                    raise ValueError("PEFT adapters require adapter_config.json or per-target alpha tensors")
    if not has_peft and native_config is None:
        config = {}
    return state, config, directory, native_config


def pattern_value(config, field, name, default):
    matches = [value for pattern, value in config.get(field, {}).items()
               if re.fullmatch(r"(.*\.)?(" + pattern + ")", name)]
    if matches and any(value != matches[0] for value in matches):
        raise ValueError(f"Ambiguous {field} for {name}")
    return matches[0] if matches else default


def normalize(state, config, bindings, output, scale, rs_config, native_scaling=False, require_scaling=True):
    groups = {}
    peft_targets = set()
    replacements = {}
    for name, tensor in state.items():
        for suffix, part in SUFFIXES:
            if name.endswith(suffix):
                target = name.removesuffix(suffix)
                group = groups.setdefault(target, {})
                if part in group:
                    raise ValueError(f"Duplicate adapter factor: {name}")
                group[part] = tensor
                if suffix in {".lora_A.weight", ".lora_B.weight", ".lora_A.default.weight", ".lora_B.default.weight"}:
                    peft_targets.add(target)
                break
        else:
            if not name.endswith((".weight", ".bias")):
                raise ValueError(f"Unsupported adapter tensor: {name}")
            targets = bindings.resolve(name)
            if len(targets) != 1 or targets[0].rows:
                raise ValueError(f"Cannot slice a full replacement: {name}")
            if targets[0].name in replacements:
                raise ValueError(f"Multiple replacements target {targets[0].name}")
            replacements[targets[0].name] = tensor.contiguous()
    for name, factors in groups.items():
        targets = bindings.resolve(name)
        group_config = config if name in peft_targets or native_scaling else {}
        if "a" in factors or "b" in factors:
            if factors.keys() - {"a", "b", "alpha"}:
                raise ValueError(f"Mixed adapter algorithms: {name}")
            a, b = factors["a"], factors["b"]
            if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[1]:
                raise ValueError(f"Invalid A/B shapes: {name}")
            rank = a.shape[0]
            if require_scaling and not group_config and "alpha" not in factors:
                raise ValueError(f"Adapter has no scaling metadata: {name}")
            if pattern_value(group_config, "rank_pattern", name.removeprefix("base_model.model."), group_config.get("r", rank)) != rank:
                raise ValueError(f"Configured rank differs from factors: {name}")
            coefficient = pattern_value(group_config, "alpha_pattern", name.removeprefix("base_model.model."),
                                        group_config.get("lora_alpha", rank))
            coefficient = float(factors.get("alpha", torch.tensor(coefficient)).item())
            next_row = 0
            for target in sorted(targets, key=lambda target: target.first_row):
                rows = target.rows or b.shape[0]
                if target.first_row != next_row or rows <= 0 or target.first_row + rows > b.shape[0]:
                    raise ValueError(f"Invalid adapter output row slices: {name}")
                next_row += rows
            if next_row != b.shape[0]:
                raise ValueError(f"Adapter output rows were not fully bound: {name}")
            for target in targets:
                rows = target.rows or b.shape[0]
                prefix = target.name
                if any(key == prefix or key.startswith(prefix + ".") for key in output):
                    raise ValueError(f"Multiple updates target {prefix}")
                output[prefix + ".lora_A"] = a.contiguous().clone()
                output[prefix + ".lora_B"] = b[target.first_row:target.first_row + rows].contiguous().clone()
                effective_alpha = coefficient * scale
                if group_config.get("use_rslora", False):
                    # Keep rsLoRA scaling in the framework importer, not in the factors.
                    output[prefix + ".lora_A.weight"] = output.pop(prefix + ".lora_A")
                    output[prefix + ".lora_B.weight"] = output.pop(prefix + ".lora_B")
                    rs_config["rank_pattern"][re.escape(prefix)] = rank
                    rs_config["alpha_pattern"][re.escape(prefix)] = effective_alpha
                else:
                    output[prefix + ".alpha"] = torch.tensor(effective_alpha, dtype=torch.float32)
        else:
            if len(targets) != 1 or targets[0].rows:
                raise ValueError(f"Cannot slice LoKr factors: {name}")
            prefix = targets[0].name
            if any(key == prefix or key.startswith(prefix + ".") for key in output):
                raise ValueError(f"Multiple updates target {prefix}")
            for part, tensor in factors.items():
                if part == "alpha":
                    output[prefix + ".alpha"] = tensor.float() * scale
                else:
                    output[prefix + "." + part] = tensor.contiguous()
            if scale != 1:
                raise ValueError("Non-unit LoKr export strength is not supported yet")
    for name, tensor in replacements.items():
        if name in output:
            raise ValueError(f"Duplicate replacement: {name}")
        output[name] = tensor


def sidecars(family, directory, native_config, base, bindings, output, work):
    root = None
    speaker_file = find_checkpoint(directory, "speaker_embedding", required=False) if family == "qwen3_tts" else None
    if speaker_file is not None:
        config_file = directory / "tts_config.json"
        config = json.loads(config_file.read_text())
        talker = config["talker_config"]
        if config["tts_model_type"] != "custom_voice" or len(talker["spk_id"]) != 1:
            raise ValueError("Speaker sidecar requires a single custom_voice speaker")
        target = bindings.resolve("talker.model.codec_embedding.weight")[0].name
        if any(key == target or key.startswith(target + ".") for key in output):
            raise ValueError("Speaker sidecar conflicts with codec embedding updates")
        shape = base.shapes[target]
        if shape != (talker["vocab_size"], talker["hidden_size"]):
            raise ValueError("Speaker sidecar configuration differs from the base checkpoint")
        speaker_id = next(iter(talker["spk_id"].values()))
        if not 0 <= speaker_id < shape[0]:
            raise ValueError("Speaker ID is outside the codec vocabulary")
        speaker = load_checkpoint(speaker_file)["speaker_embedding"]
        if tuple(speaker.shape) != (shape[1],):
            raise ValueError("Speaker embedding width differs from the base checkpoint")
        embedding = base.read_f32(target)
        embedding[speaker_id] = speaker.float()
        output[target] = embedding
        root = work / "model"
        root.mkdir()
        base.extract_sidecars(root)
        (root / "config.json").write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
    if family == "cosyvoice3" and native_config is not None:
        for module in ("speech_embedding", "llm_decoder"):
            state = load_checkpoint(find_checkpoint(directory, module))
            for name, tensor in state.items():
                target = bindings.resolve(module + "." + name)[0].name
                if target in output:
                    raise ValueError(f"Duplicate sidecar replacement: {target}")
                output[target] = tensor.contiguous()
    if family == "vibevoice":
        candidates = (
            ("diffusion_head/model", "model.prediction_head."),
            ("diffusion_head_full", "model.prediction_head."),
            ("diffusion_head/diffusion_head_full", "model.prediction_head."),
            ("acoustic_connector/pytorch_model", "model.acoustic_connector."),
            ("semantic_connector/pytorch_model", "model.semantic_connector."),
        )
        loaded = set()
        for filename, prefix in candidates:
            if prefix in loaded:
                continue
            logical = directory / filename
            path = find_checkpoint(logical.parent, logical.name, required=False)
            if path is None:
                continue
            for name, tensor in load_checkpoint(path).items():
                target = bindings.resolve(prefix + name)[0].name
                output[target] = tensor.contiguous()
            loaded.add(prefix)
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=FAMILIES, required=True)
    base_options = parser.add_mutually_exclusive_group(required=True)
    base_options.add_argument("--base", type=Path, help="Unmerged base GGUF, preferably original dtype or 16-bit")
    base_options.add_argument("--input", action="append", help="Original converter-compatible [namespace=]safetensors or shard index; repeat for components")
    parser.add_argument("--root", type=Path, help="Original checkpoint directory containing config/tokenizer files")
    parser.add_argument("--adapter", type=Path, help="Adapter file or directory; AR adapter for YuE2")
    parser.add_argument("--nar-adapter", type=Path, help="Additional YuE2 NAR adapter")
    parser.add_argument("--scale", type=float, default=1.0, help="Adapter strength multiplier")
    parser.add_argument("--nar-scale", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, help="Training alpha for a Fish Lightning checkpoint")
    parser.add_argument("--component", help="GGUF component namespace, e.g. dit_turbo_weights for ACE-Step")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--type", default="orig", choices=("orig", "f32", "f16", "bf16", "q8_0", "q4_0", "q2_k", "q3_k", "q4_k", "q5_k", "q6_k"))
    parser.add_argument("--keep-type", action="append", default=[], help="Converter tensor-pattern dtype override")
    parser.add_argument("--converter", type=Path, default=Path(__file__).resolve().parents[2] / "build/debug/bin/audiocpp_gguf")
    parser.add_argument("--model-spec", type=Path, help="Override the repository's current model spec")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.base and args.base.resolve() == args.output.resolve():
        parser.error("--output must not replace the base GGUF")
    for source in args.input or []:
        path = Path(source.partition("=")[2] if "=" in source else source)
        if path.resolve() == args.output.resolve():
            parser.error("--output must not replace an original weight source")
    if args.output.exists() and not args.overwrite:
        parser.error("Output exists; pass --overwrite to replace it")
    if args.nar_adapter and args.family != "yue2":
        parser.error("--nar-adapter is only valid for YuE2")
    if args.adapter is None and args.nar_adapter is None:
        parser.error("Provide --adapter, or --nar-adapter for a NAR-only YuE2 merge")
    if args.alpha is not None and args.family != "fish_audio":
        parser.error("--alpha is only used to prepare Fish Speech Lightning checkpoints")
    if not torch.isfinite(torch.tensor([args.scale, args.nar_scale])).all():
        parser.error("Adapter strengths must be finite")
    if args.base:
        base = BaseGGUF(args.base)
        inputs = [str(args.base)]
    else:
        inputs = args.input
        first = Path(inputs[0].partition("=")[2] if "=" in inputs[0] else inputs[0])
        base = BaseSources(inputs, args.root or first.parent)
        if any(path.resolve() == args.output.resolve() for path, _ in base.tensors.values()):
            parser.error("--output must not replace an original weight shard")
    bindings = Bindings(args.family, base.shapes, args.component)
    with tempfile.TemporaryDirectory(prefix="audiocpp-lora-") as temporary:
        work = Path(temporary)
        output = {}
        rs_config = {"r": 1, "lora_alpha": 1, "use_rslora": True, "rank_pattern": {}, "alpha_pattern": {}}
        root = args.root
        for adapter, strength in ((args.adapter, args.scale), (args.nar_adapter, args.nar_scale)):
            if adapter is None or (strength == 0 and args.family in {"yue2", "ace_step"}):
                continue
            state, config, directory, native = read_adapter(adapter, args.family, work, args.alpha)
            for option in ("use_dora", "use_qalora", "fan_in_fan_out", "lora_bias"):
                if config.get(option, False):
                    raise ValueError(f"Unsupported adapter option: {option}")
            normalize(state, config, bindings, output, strength if args.family == "yue2" else 1.0,
                      rs_config, native_scaling=native is not None, require_scaling=args.family not in {"yue2", "ace_step"})
            model_root = sidecars(args.family, directory, native, base, bindings, output, work)
            if model_root is not None:
                root = model_root
        command = [str(args.converter), "--output", str(args.output),
                   "--type", args.type, "--family", args.family, "--model-spec",
                   str(args.model_spec or Path(__file__).resolve().parents[2] / "model_specs" / (args.family + ".json")), "--log"]
        for value in inputs:
            command.extend(("--input", value))
        if output:
            normalized = work / "adapter.safetensors"
            save_file(output, normalized)
            if rs_config["rank_pattern"]:
                (work / "adapter_config.json").write_text(json.dumps(rs_config))
            command.extend(("--weight-adapter", str(normalized), "--adapter-merge-mode",
                            MERGE_MODES.get(args.family, "delta_f32")))
            if args.family != "yue2":
                command.extend(("--adapter-strength", str(args.scale)))
        if root is not None:
            command.extend(("--root", str(root)))
        for pattern in args.keep_type:
            command.extend(("--keep-type", pattern))
        if args.overwrite:
            command.append("--overwrite")
        subprocess.run(command, check=True)
    print(f"adapter_tensors={len(output)}")


if __name__ == "__main__":
    main()
