#!/usr/bin/env python3
"""Embed an audio.cpp v1 package in a native NVIDIA SALM decoder GGUF.

Preserves tensor names, shapes, quantization and payload bytes. The native
encoder remains a sibling and its checkpoint ID must match the decoder.
Dependencies: uv run --with gguf==0.19.0 --with numpy==1.26.4 ...
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import shutil
import struct
from contextlib import contextmanager
from pathlib import Path

import gguf
import numpy as np

FAMILY = "nemotron_transcribe"
ROOT = Path(__file__).resolve().parents[2]
SIDECARS = (
    "llm_backbone/config.json", "generation_config.json", "processor_config.json",
    "tokenizer.json", "tokenizer_config.json", "config.json", "chat_template.jinja",
    "LICENSE", "THIRD_PARTY_NOTICES.md", "licenses/Qwen3-ASR-LICENSE.txt", "licenses/vLLM-LICENSE.txt",
)
BACKBONE_FIELDS = {
    "hidden_size": "embedding_length", "vocab_size": "vocab_size",
    "num_attention_heads": "attention.head_count", "head_dim": "attention.key_length",
    "moe_intermediate_size": "expert_feed_forward_length",
    "moe_shared_expert_intermediate_size": "expert_shared_feed_forward_length",
    "n_shared_experts": "expert_shared_count", "n_routed_experts": "expert_count",
    "num_experts_per_tok": "expert_used_count", "n_group": "expert_group_count",
    "topk_group": "expert_group_used_count", "conv_kernel": "ssm.conv_kernel",
    "ssm_state_size": "ssm.state_size", "n_groups": "ssm.group_count",
    "mamba_num_heads": "ssm.time_step_rank",
}
TEKKEN_PATTERN = (
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|"
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|"
    r"\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)
PRE_TOKENIZER = {"type": "Sequence", "pretokenizers": [
    {"type": "Split", "pattern": {"Regex": TEKKEN_PATTERN}, "behavior": "Isolated", "invert": False},
    {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": False}]}
WINDOW_KEY = "perception.preprocessor.featurizer.window"
DIAR_WINDOW_KEY = "perception.encoder.diarization_model.preprocessor.featurizer.window"


def field(reader: gguf.GGUFReader, name: str):
    if name not in reader.fields:
        raise ValueError(f"missing native GGUF field: {name}")
    return reader.fields[name].contents()


def source_checkpoint_identity(source: Path) -> str:
    """Match NVIDIA's pair ID using headers and first/last 4 KiB per tensor.

    This binds the tokenizer/config source to the weights without hashing the
    entire 67 GB checkpoint. Algorithm from the pinned native conversion/salm.py.
    """
    digest = hashlib.sha256()
    files = sorted(source.glob("*.safetensors"))
    if not files:
        raise ValueError("source contains no safetensors checkpoint")
    for path in files:
        length = path.stat().st_size
        with path.open("rb") as stream:
            size = struct.unpack("<Q", stream.read(8))[0]
            if size > min(64 * 1024 * 1024, length - 8):
                raise ValueError("invalid source safetensors header")
            header = json.loads(stream.read(size))
            for name in sorted(key for key in header if key != "__metadata__"):
                info = header[name]
                begin, end = info["data_offsets"]
                if begin < 0 or end < begin or end > length - 8 - size:
                    raise ValueError("invalid source tensor offsets")
                digest.update(f"{name}|{info['dtype']}|{info['shape']}|".encode())
                stream.seek(8 + size + begin)
                digest.update(stream.read(min(4096, end - begin)))
                stream.seek(8 + size + max(begin, end - 4096))
                digest.update(stream.read(min(4096, end - begin)))
    return digest.hexdigest()[:32]


def source_frontend_window(source: Path) -> list[float]:
    """Read the trained STFT window without loading the full checkpoint."""
    windows = {}
    for path in sorted(source.glob("*.safetensors")):
        with path.open("rb") as stream:
            size = struct.unpack("<Q", stream.read(8))[0]
            if size > min(64 * 1024 * 1024, path.stat().st_size - 8):
                raise ValueError("invalid source safetensors header")
            header = json.loads(stream.read(size))
            for name in (WINDOW_KEY, DIAR_WINDOW_KEY):
                if name not in header:
                    continue
                item = header[name]
                begin, end = item["data_offsets"]
                if item["shape"] != [400] or item["dtype"] not in ("BF16", "F32"):
                    raise ValueError("unsupported trained STFT window shape/type")
                stream.seek(8 + size + begin)
                data = stream.read(end - begin)
                if item["dtype"] == "BF16":
                    bits = np.frombuffer(data, dtype="<u2").astype(np.uint32) << 16
                    values = bits.view(np.float32)
                else:
                    values = np.frombuffer(data, dtype="<f4")
                if len(values) != 400 or not np.isfinite(values).all():
                    raise ValueError("invalid trained STFT window values")
                windows[name] = values.copy()
    if WINDOW_KEY not in windows:
        raise ValueError("source checkpoint lacks the trained STFT window")
    if DIAR_WINDOW_KEY in windows and not np.array_equal(windows[WINDOW_KEY], windows[DIAR_WINDOW_KEY]):
        raise ValueError("speech and speaker STFT windows differ")
    return windows[WINDOW_KEY].tolist()


def validate_tokenizer(primary: gguf.GGUFReader, tokenizer: dict, configuration: dict) -> None:
    expected = {"type": "BPE", "dropout": None, "unk_token": None, "continuing_subword_prefix": None,
                "end_of_word_suffix": None, "fuse_unk": False, "byte_fallback": False, "ignore_merges": True}
    if any(tokenizer["model"].get(key) != value for key, value in expected.items()) or \
            tokenizer.get("normalizer") is not None or tokenizer.get("pre_tokenizer") != PRE_TOKENIZER:
        raise ValueError("unsupported source tokenizer normalization/pretokenization semantics")
    if configuration.get("eos_token") != "<|im_end|>" or configuration.get("clean_up_tokenization_spaces") is not False:
        raise ValueError("unsupported source tokenizer output/EOS semantics")
    tokens = field(primary, "tokenizer.ggml.tokens")
    token_types = field(primary, "tokenizer.ggml.token_type")
    vocabulary = tokenizer["model"]["vocab"]
    by_id = {int(index): token for token, index in vocabulary.items()}
    added_ids = set()
    locator = False
    for token in tokenizer.get("added_tokens", []):
        index = int(token["id"])
        if any(token.get(key, False) for key in ("single_word", "lstrip", "rstrip", "normalized")):
            raise ValueError("unsupported source added-token normalization")
        if token["content"] == "<|audio|>":
            if index != 131072 or not token.get("special"):
                raise ValueError("invalid source audio locator")
            locator = True
            continue
        if index < 0 or index >= len(token_types) or token_types[index] != (3 if token.get("special") else 4):
            raise ValueError("source and GGUF added-token behavior differs")
        added_ids.add(index)
        by_id[index] = token["content"]
    if not locator or added_ids != {i for i, kind in enumerate(token_types) if kind != 1}:
        raise ValueError("source and GGUF added-token coverage differs")
    if len(tokens) != 131072 or set(by_id) != set(range(len(tokens))):
        raise ValueError("source and native GGUF token vocabulary coverage differs")
    if any(by_id[index] != token for index, token in enumerate(tokens)):
        raise ValueError("source and native GGUF token vocabulary differs")
    merges = tokenizer["model"]["merges"]
    merges = [" ".join(pair) if isinstance(pair, list) else pair for pair in merges]
    native_merges = field(primary, "tokenizer.ggml.merges") if "tokenizer.ggml.merges" in primary.fields else []
    if merges != native_merges:
        raise ValueError("source and native GGUF BPE merge order differs")


def validate_backbone(primary: gguf.GGUFReader, config: dict) -> None:
    prefix = "nemotron_h_moe."
    for source_key, native_key in BACKBONE_FIELDS.items():
        if config[source_key] != field(primary, prefix + native_key):
            raise ValueError(f"source and GGUF backbone configuration differ: {source_key}")
    for source_key, native_key in (("norm_eps", "attention.layer_norm_rms_epsilon"),
                                  ("layer_norm_epsilon", "attention.layer_norm_epsilon"),
                                  ("routed_scaling_factor", "expert_weights_scale")):
        if np.float32(config[source_key]) != np.float32(field(primary, prefix + native_key)):
            raise ValueError(f"source and GGUF backbone configuration differ: {source_key}")
    if config["head_dim"] != field(primary, prefix + "attention.value_length") or \
            config["mamba_num_heads"] * config["mamba_head_dim"] != field(primary, prefix + "ssm.inner_size") or \
            config["norm_topk_prob"] != field(primary, prefix + "expert_weights_norm"):
        raise ValueError("source and GGUF attention/Mamba/router configuration differs")
    pattern = config["layers_block_type"]
    count = int(field(primary, prefix + "block_count"))
    kv = field(primary, prefix + "attention.head_count_kv")
    ff = field(primary, prefix + "feed_forward_length")
    if count not in (len(pattern), len(pattern) + 1) or len(kv) != count or len(ff) != count:
        raise ValueError("source and GGUF trunk/MTP block counts differ")
    for i, kind in enumerate(pattern):
        if kind not in ("attention", "moe", "mamba") or \
                kv[i] != (config["num_key_value_heads"] if kind == "attention" else 0) or \
                ff[i] != (config["moe_intermediate_size"] if kind == "moe" else 0):
            raise ValueError(f"source and GGUF block pattern differs at block {i}")
    if count > len(pattern) and (kv[-1] != config["num_key_value_heads"] or ff[-1] != config["moe_intermediate_size"]):
        raise ValueError("unsupported folded MTP metadata")
    # The native no-RoPE converter publishes generic context/rotary metadata;
    # these are intentionally not compared with HF positions/head dimensions.


@contextmanager
def mapped_reader(path: Path):
    reader = gguf.GGUFReader(str(path))
    try:
        yield reader
    finally:
        reader.data._mmap.close()


def prepare(decoder: Path, encoder: Path, source: Path, output: Path, spec_path: Path) -> dict:
    if output.exists():
        raise ValueError(f"output already exists: {output}")
    if output.resolve() in (decoder.resolve(), encoder.resolve()):
        raise ValueError("output must be separate from the native reference files")
    with mapped_reader(decoder) as primary, mapped_reader(encoder) as speech:
        return _prepare(decoder, encoder, source, output, spec_path, primary, speech)


def _prepare(decoder: Path, encoder: Path, source: Path, output: Path, spec_path: Path,
             primary: gguf.GGUFReader, speech: gguf.GGUFReader) -> dict:
    if field(primary, "general.architecture") != "nemotron_h_moe":
        raise ValueError("expected a native Nemotron-H MoE decoder")
    identity = field(primary, "nemo_speech.checkpoint_id")
    if not identity or identity != field(speech, "nemo_speech.checkpoint_id"):
        raise ValueError("encoder and decoder checkpoint IDs differ")
    if source_checkpoint_identity(source) != identity:
        raise ValueError("sidecar source belongs to a different checkpoint")
    sibling_name = field(primary, "nemo_speech.speech_model")
    if sibling_name != encoder.name or Path(sibling_name).name != sibling_name:
        raise ValueError("encoder filename does not match the native decoder pairing")
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if spec.get("schema_version") != 1 or spec.get("family") != FAMILY:
        raise ValueError("expected the Nemotron Transcribe v1 model spec")
    files: dict[str, bytes] = {}
    for relative in SIDECARS:
        path = source / relative
        if not path.is_file():
            raise ValueError(f"missing checkpoint sidecar: {path}")
        files[relative] = path.read_bytes()
    files["frontend_window.json"] = json.dumps(source_frontend_window(source), separators=(",", ":")).encode("utf-8")
    if not base64.b64decode(field(speech, "ctc.tokenizer.spm_model"), validate=True):
        raise ValueError("the encoder contains no CTC tokenizer")
    backbone = json.loads(files["llm_backbone/config.json"])
    if backbone.get("vocab_size") != 131072 or backbone.get("hidden_size") != 2688:
        raise ValueError("checkpoint sidecars do not describe the supported speech backbone")
    validate_backbone(primary, backbone)
    if json.loads(files["generation_config.json"]).get("eos_token_id") != [11]:
        raise ValueError("checkpoint generation_config does not use end-of-turn token 11")
    validate_tokenizer(primary, json.loads(files["tokenizer.json"]), json.loads(files["tokenizer_config.json"]))
    names = sorted(files)
    # All sidecars are UTF-8. The CTC tokenizer is already embedded in the
    # paired encoder's native metadata, so no binary sidecar is duplicated.
    # String arrays also avoid GGUFReader's per-byte scalar views of byte blobs.
    contents = [files[name].decode("utf-8") for name in names]
    output.parent.mkdir(parents=True, exist_ok=True)
    sibling = output.parent / sibling_name
    if sibling.exists():
        if sibling.stat().st_size != encoder.stat().st_size:
            raise ValueError("a different encoder already occupies the output sibling path")
        with mapped_reader(sibling) as existing:
            if field(existing, "nemo_speech.checkpoint_id") != identity:
                raise ValueError("the output encoder belongs to another checkpoint")
    else:
        # A hardlink avoids another large local copy on the same filesystem.
        # A copied sibling makes the same package on other filesystems.
        try:
            sibling.hardlink_to(encoder)
        except OSError:
            shutil.copyfile(encoder, sibling)
    temporary = output.with_name(output.name + ".packaging")
    if temporary.exists():
        raise ValueError(f"uninspected temporary output exists: {temporary}")
    writer = gguf.GGUFWriter(str(temporary), field(primary, "general.architecture"), endianess=primary.endianess)
    try:
        if "general.alignment" in primary.fields:
            writer.add_custom_alignment(int(field(primary, "general.alignment")))
        for name, value in primary.fields.items():
            if name.startswith("GGUF.") or name == "general.architecture" or name.startswith("audiocpp."):
                continue
            subtype = value.types[-1] if value.types[0] == gguf.GGUFValueType.ARRAY else None
            writer.add_key_value(name, value.contents(), value.types[0], sub_type=subtype)
        writer.add_uint32("audiocpp.model_spec.version", 1)
        writer.add_string("audiocpp.model_spec.family", FAMILY)
        writer.add_string("audiocpp.model_spec.json", json.dumps(spec, ensure_ascii=False, separators=(",", ":")))
        writer.add_array("audiocpp.embedded_files.names", names)
        writer.add_array("audiocpp.embedded_files.contents", contents)
        for tensor in primary.tensors:
            writer.add_tensor(tensor.name, tensor.data, raw_dtype=tensor.tensor_type)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file(progress=True)
    finally:
        writer.close()
    with mapped_reader(temporary) as checked:
        if len(checked.tensors) != len(primary.tensors):
            raise RuntimeError("packaged tensor count differs from native decoder")
        for original, packed in zip(primary.tensors, checked.tensors):
            if original.name != packed.name or original.tensor_type != packed.tensor_type or \
                    tuple(original.shape) != tuple(packed.shape) or original.n_bytes != packed.n_bytes:
                raise RuntimeError(f"packaging changed tensor metadata: {original.name}")
    temporary.replace(output)
    return {"family": FAMILY, "checkpoint_id": identity, "tensor_count": len(primary.tensors),
            "decoder": str(output), "encoder": str(sibling), "decoder_bytes": output.stat().st_size,
            "encoder_bytes": sibling.stat().st_size, "embedded_files": names,
            "tensor_payload_policy": "copied without requantization"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decoder", required=True, type=Path)
    parser.add_argument("--encoder", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path, help="Original full BF16 checkpoint directory")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-spec", type=Path, default=ROOT / "model_specs/nemotron_transcribe.json")
    args = parser.parse_args()
    result = prepare(args.decoder, args.encoder, args.source, args.output, args.model_spec)
    receipt = args.output.with_suffix(".package.json")
    receipt.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
