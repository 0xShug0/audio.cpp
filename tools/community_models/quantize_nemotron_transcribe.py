#!/usr/bin/env python3
"""Create audited Q4_K/Q3_K/Q2_K audio.cpp Nemotron Transcribe decoders.

Input is the unquantized BF16 decoder from the pinned NVIDIA SALM conversion.
One streamed, input-row-padded cache is reused across profiles. It adds zeros
only to routed expert input columns, keeps logical model metadata unchanged,
and requires audio.cpp's opt-in expert-row-alignment support. Native llama/NeMo
inference does not consume these padded layouts.

By default, run prepare_nemotron_transcribe.py afterwards to wrap the decoder.
Supplying --package-source and --encoder embeds the validated spec, sidecars and
trained frontend window in the shared cache instead. Quantizer outputs are then
self-contained without a second weights copy. Pre-existing audiocpp metadata is
regenerated, so stale logical tensor-shape overrides cannot survive padding.

Use a pristine llama-quantize built at PINNED_LLAMA_REVISION. The script checks
its reported commit in a dry run and audits actual output tensor types, shapes
and unchanged floating-point payloads before publishing a new output path.
Existing outputs, partial files and unverified caches are never overwritten or
deleted. No model download, inference, quantizer patching or upload is performed.

Dependencies: uv run --with gguf==0.19.0 --with numpy==1.26.4 ...
"""
from __future__ import annotations

import argparse
import base64
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Iterator
import uuid

import gguf
import numpy as np

PINNED_LLAMA_REVISION = "bd4f514db14d87fded667787a7a963bfbaa98e89"
ARCH = "nemotron_h_moe"
ALIGNMENT_KEY = "nemotron_transcribe.expert_row_alignment"
ROW_ALIGNMENT = 256
BUFFER_BYTES = 16 * 1024 * 1024
FLOAT_TYPES = {gguf.GGMLQuantizationType.BF16, gguf.GGMLQuantizationType.F32}
PROFILES = {name.lower(): getattr(gguf.GGMLQuantizationType, name)
            for name in ("Q4_K", "Q3_K", "Q2_K")}
EXPERT = re.compile(r"^blk\.(\d+)\.ffn_(up|down)_exps\.weight$")
DENSE = re.compile(r"^blk\.\d+\.(?:ssm_(?:in|out)|ffn_(?:up|down)_shexp|attn_(?:q(?:kv)?|k|v|output))\.weight$")


@dataclass(frozen=True)
class TensorPlan:
    name: str
    shape: tuple[int, ...]  # GGML order: input columns first
    padded_shape: tuple[int, ...]
    qtype: gguf.GGMLQuantizationType
    source_offset: int
    source_bytes: int
    padded_bytes: int
    category: str


@contextmanager
def mapped_reader(path: Path) -> Iterator[gguf.GGUFReader]:
    reader = gguf.GGUFReader(str(path))
    try:
        yield reader
    finally:
        reader.data._mmap.close()


def field(reader: gguf.GGUFReader, name: str):
    if name not in reader.fields:
        raise ValueError(f"missing source GGUF field: {name}")
    return reader.fields[name].contents()


def positive_u32(reader: gguf.GGUFReader, suffix: str) -> int:
    name = ARCH + "." + suffix
    value = field(reader, name)
    if reader.fields[name].types != [gguf.GGUFValueType.UINT32] or type(value) is not int or value <= 0:
        raise ValueError(f"invalid positive UINT32 model dimension: {name}")
    return value


def stat_identity(path: Path) -> dict:
    value = path.stat()
    return {"device": value.st_dev, "inode": value.st_ino,
            "bytes": value.st_size, "modified_ns": value.st_mtime_ns}


def source_plan(reader: gguf.GGUFReader, source: Path) -> list[TensorPlan]:
    if field(reader, "general.architecture") != ARCH:
        raise ValueError("expected an unquantized Nemotron-H MoE decoder GGUF")
    identity = field(reader, "nemo_speech.checkpoint_id")
    sibling = field(reader, "nemo_speech.speech_model")
    if not isinstance(identity, str) or not identity or not isinstance(sibling, str) or not sibling or \
            Path(sibling).name != sibling or "/" in sibling or "\\" in sibling:
        raise ValueError("invalid native encoder/checkpoint pairing metadata")
    if ALIGNMENT_KEY in reader.fields and field(reader, ALIGNMENT_KEY) != 1:
        raise ValueError("source is already padded; supply the original BF16 GGUF")
    if "split.count" in reader.fields and field(reader, "split.count") != 1:
        raise ValueError("split source GGUFs are unsupported; supply one complete BF16 decoder")
    hidden = positive_u32(reader, "embedding_length")
    width = positive_u32(reader, "expert_feed_forward_length")
    experts = positive_u32(reader, "expert_count")
    layers = positive_u32(reader, "block_count")
    names: set[str] = set()
    pairs: dict[int, set[str]] = {}
    plans = []
    size = source.stat().st_size
    for tensor in reader.tensors:
        name, qtype = tensor.name, tensor.tensor_type
        if name in names or re.fullmatch(r"[A-Za-z0-9_.]+", name) is None:
            raise ValueError(f"duplicate or unsupported tensor name: {name!r}")
        names.add(name)
        if qtype not in FLOAT_TYPES:
            raise ValueError(f"source must be BF16/F32, never a requantization input: {name} is {qtype.name}")
        shape = tuple(int(x) for x in tensor.shape)
        if not 1 <= len(shape) <= 3 or any(x <= 0 for x in shape):
            raise ValueError(f"invalid tensor shape: {name}")
        element_bytes = 2 if qtype == gguf.GGMLQuantizationType.BF16 else 4
        if tensor.n_bytes != math.prod(shape) * element_bytes or tensor.data_offset + tensor.n_bytes > size:
            raise ValueError(f"invalid source tensor byte extent: {name}")
        padded, category = shape, "preserved"
        match = EXPERT.fullmatch(name)
        if match:
            layer, direction = int(match[1]), match[2]
            expected = (hidden, width, experts) if direction == "up" else (width, hidden, experts)
            if layer >= layers or shape != expected or qtype != gguf.GGMLQuantizationType.BF16:
                raise ValueError(f"routed expert logical shape/BF16 metadata mismatch: {name}: {shape}, expected {expected}")
            pairs.setdefault(layer, set()).add(direction)
            padded = ((shape[0] + ROW_ALIGNMENT - 1) // ROW_ALIGNMENT * ROW_ALIGNMENT, *shape[1:])
            category = "routed"
        elif "_exps." in name:
            raise ValueError(f"unsupported routed expert tensor: {name}")
        elif DENSE.fullmatch(name):
            if len(shape) != 2 or shape[0] % 32:
                raise ValueError(f"dense Q8_0 tensor has an incompatible shape: {name}: {shape}")
            category = "dense"
        elif name in {"token_embd.weight", "output.weight"}:
            if len(shape) != 2 or shape[0] != hidden:
                raise ValueError(f"logical embedding/output shape mismatch: {name}")
        plans.append(TensorPlan(name, shape, padded, qtype, int(tensor.data_offset), int(tensor.n_bytes),
                                math.prod(padded) * element_bytes, category))
    if not pairs or any(pair != {"up", "down"} for pair in pairs.values()):
        raise ValueError("every routed expert layer must have one matching up/down pair")
    if "token_embd.weight" not in names or "output.weight" not in names:
        raise ValueError("source must retain token embeddings and output projection")
    return plans


def package_helpers():
    specification = importlib.util.spec_from_file_location("_nemotron_prepare_helpers",
        Path(__file__).with_name("prepare_nemotron_transcribe.py"))
    if not specification or not specification.loader:
        raise ImportError("prepare_nemotron_transcribe.py must accompany this converter")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def package_payload(source: Path, encoder: Path, spec_path: Path, reader: gguf.GGUFReader) -> dict:
    helpers = package_helpers()
    identity = field(reader, "nemo_speech.checkpoint_id")
    with mapped_reader(encoder) as speech:
        if helpers.field(speech, "nemo_speech.checkpoint_id") != identity:
            raise ValueError("encoder and decoder checkpoint IDs differ")
        if field(reader, "nemo_speech.speech_model") != encoder.name:
            raise ValueError("encoder filename does not match the native decoder pairing")
        if not base64.b64decode(helpers.field(speech, "ctc.tokenizer.spm_model"), validate=True):
            raise ValueError("the paired encoder contains no CTC tokenizer")
    if helpers.source_checkpoint_identity(source) != identity:
        raise ValueError("package sidecars belong to another checkpoint")
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if spec.get("schema_version") != 1 or spec.get("family") != helpers.FAMILY:
        raise ValueError("expected the Nemotron Transcribe v1 model spec")
    files = {}
    for name in helpers.SIDECARS:
        path = source / name
        if not path.is_file():
            raise ValueError(f"missing checkpoint sidecar: {name}")
        files[name] = path.read_text(encoding="utf-8")
    files["frontend_window.json"] = json.dumps(helpers.source_frontend_window(source), separators=(",", ":"))
    helpers.validate_backbone(reader, json.loads(files["llm_backbone/config.json"]))
    if json.loads(files["generation_config.json"]).get("eos_token_id") != [11]:
        raise ValueError("checkpoint generation_config does not use end-of-turn token 11")
    helpers.validate_tokenizer(reader, json.loads(files["tokenizer.json"]), json.loads(files["tokenizer_config.json"]))
    names = sorted(files)
    payload = {"family": helpers.FAMILY,
        "spec": json.dumps(spec, ensure_ascii=False, separators=(",", ":")),
        "names": names, "contents": [files[name] for name in names], "checkpoint_id": identity,
        "encoder_name": encoder.name, "encoder_bytes": encoder.stat().st_size}
    # Only the small package inputs are fingerprinted. The multi-gigabyte weight
    # lineage uses the native checkpoint identity and source/cache file stats.
    payload["fingerprint"] = hashlib.blake2s(json.dumps(payload, ensure_ascii=False,
        sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return payload


def ensure_encoder_sibling(encoder: Path, directory: Path, package: dict) -> Path:
    sibling = directory / package["encoder_name"]
    if sibling.exists():
        if sibling.stat().st_size != package["encoder_bytes"]:
            raise ValueError("a different encoder occupies the output sibling path")
        with mapped_reader(sibling) as reader:
            if field(reader, "nemo_speech.checkpoint_id") != package["checkpoint_id"]:
                raise ValueError("output encoder sibling belongs to another checkpoint")
        return sibling
    temporary = sibling.with_name(sibling.name + ".copying")
    if temporary.exists():
        raise FileExistsError("uninspected encoder copy exists")
    try:
        os.link(encoder, sibling)
    except FileExistsError:
        raise
    except OSError:
        # Same-volume files normally hardlink. Cross-volume packages copy the
        # one encoder once; subsequent profiles validate and reuse that sibling.
        with encoder.open("rb") as source, temporary.open("xb") as destination:
            shutil.copyfileobj(source, destination, BUFFER_BYTES)
            destination.flush()
            os.fsync(destination.fileno())
        with mapped_reader(temporary) as reader:
            if temporary.stat().st_size != package["encoder_bytes"] or \
                    field(reader, "nemo_speech.checkpoint_id") != package["checkpoint_id"]:
                raise ValueError("copied encoder failed pairing validation; partial retained")
        publish_no_overwrite(temporary, sibling)
    return sibling


def copy_metadata(writer: gguf.GGUFWriter, reader: gguf.GGUFReader, package: dict | None = None) -> None:
    if "general.alignment" in reader.fields:
        writer.add_custom_alignment(int(field(reader, "general.alignment")))
    for name, value in reader.fields.items():
        if name.startswith(("GGUF.", "audiocpp.")) or name in {"general.architecture", "general.alignment", ALIGNMENT_KEY}:
            continue
        subtype = value.types[-1] if value.types[0] == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(name, value.contents(), value.types[0], sub_type=subtype)
    writer.add_uint32(ALIGNMENT_KEY, ROW_ALIGNMENT)
    if package is not None:
        writer.add_uint32("audiocpp.model_spec.version", 1)
        writer.add_string("audiocpp.model_spec.family", package["family"])
        writer.add_string("audiocpp.model_spec.json", package["spec"])
        writer.add_array("audiocpp.embedded_files.names", package["names"])
        writer.add_array("audiocpp.embedded_files.contents", package["contents"])


@contextmanager
def claim(path: Path) -> Iterator[None]:
    token = uuid.uuid4().hex
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(descriptor, token.encode("ascii"))
        os.close(descriptor)
        descriptor = -1
        yield
    finally:
        if descriptor != -1:
            os.close(descriptor)
        # Only our small ownership marker is removed. Payload/partial files are retained.
        if path.is_file() and path.read_text(encoding="ascii") == token:
            path.unlink()


def publish_no_overwrite(temporary: Path, destination: Path) -> None:
    if os.name == "nt":
        os.rename(temporary, destination)  # Windows rename refuses an existing destination.
    else:
        os.link(temporary, destination)  # Atomic, same-filesystem exclusive publication.
        temporary.unlink()  # The verified payload is now reachable at destination.


def copy_exact(source, destination, count: int) -> None:
    while count:
        data = source.read(min(BUFFER_BYTES, count))
        if not data:
            raise ValueError("source was truncated during tensor streaming")
        destination.write(data)
        count -= len(data)


def write_padded(source: Path, temporary: Path, reader: gguf.GGUFReader, plans: list[TensorPlan], package: dict | None = None) -> None:
    if temporary.exists():
        raise FileExistsError(f"uninspected partial padded cache exists: {temporary}")
    # GGUFWriter handles typed metadata/tensor descriptors. Payloads are streamed
    # directly, so no expert tensor or second full checkpoint enters Python RAM.
    with temporary.open("xb"):
        pass
    writer = gguf.GGUFWriter(str(temporary), ARCH, endianess=reader.endianess)
    try:
        copy_metadata(writer, reader, package)
        for plan in plans:
            writer.add_tensor_info(plan.name, plan.padded_shape[::-1], np.dtype(np.float32),
                plan.padded_bytes, raw_dtype=plan.qtype)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_ti_data_to_file()
        alignment = writer.data_alignment
    finally:
        writer.close()
    with source.open("rb") as original, temporary.open("ab") as output:
        output.write(bytes(-output.tell() % alignment))
        for index, plan in enumerate(plans):
            original.seek(plan.source_offset)
            if plan.padded_shape == plan.shape:
                copy_exact(original, output, plan.source_bytes)
            else:
                item_bytes = 2  # Routed inputs were required to be original BF16.
                old_row, new_row = plan.shape[0] * item_bytes, plan.padded_shape[0] * item_bytes
                rows_left = math.prod(plan.shape[1:])
                chunk_rows = max(1, BUFFER_BYTES // new_row)
                while rows_left:
                    rows = min(rows_left, chunk_rows)
                    raw = original.read(rows * old_row)
                    if len(raw) != rows * old_row:
                        raise ValueError(f"truncated routed expert: {plan.name}")
                    padded = np.zeros((rows, new_row), dtype=np.uint8)
                    padded[:, :old_row] = np.frombuffer(raw, dtype=np.uint8).reshape(rows, old_row)
                    padded.tofile(output)
                    rows_left -= rows
            output.write(bytes(-plan.padded_bytes % alignment))
            print(f"padding {index+1}/{len(plans)} {plan.name}: {plan.shape} -> {plan.padded_shape}", flush=True)
        output.flush()
        os.fsync(output.fileno())


def check_cache(path: Path, plans: list[TensorPlan], checkpoint_id: str, package: dict | None = None) -> None:
    with mapped_reader(path) as reader:
        if field(reader, ALIGNMENT_KEY) != ROW_ALIGNMENT or \
                reader.fields[ALIGNMENT_KEY].types != [gguf.GGUFValueType.UINT32] or \
                field(reader, "nemo_speech.checkpoint_id") != checkpoint_id:
            raise ValueError("padded cache alignment/checkpoint metadata differs")
        if any(name in reader.fields for name in ("audiocpp.tensor_shapes", "audiocpp.tensor_ranks")):
            raise ValueError("padded cache contains stale logical tensor-shape overrides")
        if package is not None:
            expected = {"audiocpp.model_spec.version": 1, "audiocpp.model_spec.family": package["family"],
                "audiocpp.model_spec.json": package["spec"], "audiocpp.embedded_files.names": package["names"],
                "audiocpp.embedded_files.contents": package["contents"]}
            if any(field(reader, name) != value for name, value in expected.items()):
                raise ValueError("padded cache embedded package inputs differ")
        elif any(name.startswith("audiocpp.") for name in reader.fields):
            raise ValueError("native padded cache unexpectedly contains package metadata")
        if len(reader.tensors) != len(plans):
            raise ValueError("padded cache tensor count differs")
        for plan, tensor in zip(plans, reader.tensors):
            if tensor.name != plan.name or tuple(tensor.shape) != plan.padded_shape or \
                    tensor.tensor_type != plan.qtype or tensor.n_bytes != plan.padded_bytes:
                raise ValueError(f"padded cache tensor differs: {plan.name}")
        expected_end = reader.data_offset + sum((p.padded_bytes + reader.alignment - 1) // reader.alignment * reader.alignment for p in plans)
        if path.stat().st_size != expected_end:
            raise ValueError("padded cache has an unexpected file extent")


def padded_cache(source: Path, cache: Path, package: dict | None = None) -> tuple[list[TensorPlan], dict]:
    source, cache = source.resolve(), cache.resolve()
    if source == cache:
        raise ValueError("padded cache must not overwrite its BF16 source")
    cache.parent.mkdir(parents=True, exist_ok=True)
    receipt = cache.with_name(cache.name + ".cache.json")
    with claim(cache.with_name(cache.name + ".lock")), mapped_reader(source) as reader:
        plans = source_plan(reader, source)
        identity = {"source": stat_identity(source), "checkpoint_id": field(reader, "nemo_speech.checkpoint_id"),
            "package_fingerprint": None if package is None else package["fingerprint"],
            "layout": [{"name": p.name, "shape": p.shape, "padded_shape": p.padded_shape, "type": p.qtype.name} for p in plans]}
        # JSON normalizes tuple/list types for stable cross-process comparison.
        identity = json.loads(json.dumps(identity))
        if cache.exists():
            if not receipt.is_file():
                raise FileExistsError("padded cache exists without a verified completion receipt")
            recorded = json.loads(receipt.read_text(encoding="utf-8"))
            if recorded.get("identity") != identity or recorded.get("cache") != stat_identity(cache):
                raise ValueError("padded cache/source changed since its completed streaming write")
            check_cache(cache, plans, identity["checkpoint_id"], package)
            return plans, {"reused": True, "cache_bytes": cache.stat().st_size}
        if receipt.exists():
            raise FileExistsError("orphan cache receipt exists; inspect it before retrying")
        temporary = cache.with_name(cache.name + ".padding")
        before = stat_identity(source)
        write_padded(source, temporary, reader, plans, package)
        if stat_identity(source) != before:
            raise ValueError("BF16 source changed during padding; partial cache retained")
        check_cache(temporary, plans, identity["checkpoint_id"], package)
        publish_no_overwrite(temporary, cache)
        with receipt.open("x", encoding="utf-8") as stream:
            json.dump({"identity": identity, "cache": stat_identity(cache),
                "validation": "typed header, byte extent, streamed original rows with zero-only input padding"}, stream, indent=2)
        return plans, {"reused": False, "cache_bytes": cache.stat().st_size}


def expected_type(plan: TensorPlan, profile: str) -> gguf.GGMLQuantizationType:
    if plan.category == "routed":
        return PROFILES[profile]
    if plan.category == "dense":
        return gguf.GGMLQuantizationType.Q8_0
    return plan.qtype


def quantizer_command(binary: Path, source: Path, output: Path, rules: Path, profile: str,
                      threads: int, embedding_type: gguf.GGMLQuantizationType, dry_run=False) -> list[str]:
    command = [str(binary), "--leave-output-tensor", "--token-embedding-type", embedding_type.name.lower(),
        "--tensor-type-file", str(rules), "--pure"]
    if dry_run:
        command.append("--dry-run")
    return command + [str(source), str(output), profile.upper(), str(threads)]


def check_quantizer(binary: Path, source: Path, output: Path, rules: Path, profile: str,
                    threads: int, embedding_type: gguf.GGMLQuantizationType, log: Path) -> str:
    result = subprocess.run(quantizer_command(binary, source, output, rules, profile, threads, embedding_type, True),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    with log.open("x", encoding="utf-8") as stream:
        stream.write(result.stdout)
    match = re.search(r"(?m)^version:.*\bcommit ([0-9a-f]{7,40})([^)]*)\)", result.stdout)
    if result.returncode or not match or not PINNED_LLAMA_REVISION.startswith(match[1]) or match[2].strip():
        raise ValueError("quantizer dry run failed or its reported commit is not the pristine pinned llama revision")
    return match[1]


def audit_output(source: Path, output: Path, plans: list[TensorPlan], profile: str) -> dict:
    with mapped_reader(source) as original, mapped_reader(output) as converted:
        if field(converted, ALIGNMENT_KEY) != ROW_ALIGNMENT or \
                converted.fields[ALIGNMENT_KEY].types != [gguf.GGUFValueType.UINT32]:
            raise RuntimeError("quantizer dropped the opt-in row alignment metadata")
        # All model/tokenizer/pairing metadata must remain intact. Quantizer-owned
        # file type/version fields are expected to change.
        for name, value in original.fields.items():
            if name.startswith("GGUF.") or name in {"general.file_type", "general.quantization_version"}:
                continue
            if name not in converted.fields or converted.fields[name].types != value.types or \
                    converted.fields[name].contents() != value.contents():
                raise RuntimeError(f"quantizer changed model metadata: {name}")
        tensors = {tensor.name: tensor for tensor in converted.tensors}
        if len(tensors) != len(plans) or len(converted.tensors) != len(plans):
            raise RuntimeError("quantizer changed tensor names/count")
        counts, audit = {}, []
        with source.open("rb") as input_data, output.open("rb") as output_data:
            old_tensors = {tensor.name: tensor for tensor in original.tensors}
            for plan in plans:
                tensor = tensors.get(plan.name)
                wanted = expected_type(plan, profile)
                if tensor is None or tuple(tensor.shape) != plan.padded_shape or tensor.tensor_type != wanted:
                    actual = "missing" if tensor is None else tensor.tensor_type.name
                    raise RuntimeError(f"quantization fallback/shape change for {plan.name}: expected {wanted.name}, actual {actual}")
                if plan.category == "preserved":
                    previous = old_tensors[plan.name]
                    input_data.seek(previous.data_offset)
                    output_data.seek(tensor.data_offset)
                    left = previous.n_bytes
                    while left:
                        amount = min(BUFFER_BYTES, left)
                        if input_data.read(amount) != output_data.read(amount):
                            raise RuntimeError(f"quantizer changed a preserved floating-point payload: {plan.name}")
                        left -= amount
                counts[wanted.name] = counts.get(wanted.name, 0) + 1
                audit.append({"name": plan.name, "category": plan.category, "logical_shape": list(plan.shape),
                    "physical_shape": tensor.shape.tolist(), "actual_type": tensor.tensor_type.name, "bytes": tensor.n_bytes})
        return {"profile": profile, "alignment": ROW_ALIGNMENT, "counts": counts, "tensors": audit,
                "validation": "all actual tensor types/shapes, pairing/model metadata, unchanged non-quantized payloads"}


def convert(source: Path, quantizer: Path, output: Path, profile: str, threads: int = 4,
            cache: Path | None = None, package_source: Path | None = None,
            encoder: Path | None = None, model_spec: Path | None = None) -> dict:
    source, quantizer, output = source.resolve(), quantizer.resolve(), output.resolve()
    if profile not in PROFILES or threads < 1 or not source.is_file() or not quantizer.is_file():
        raise ValueError("valid profile, positive threads, existing BF16 GGUF and explicit quantizer are required")
    cache = (cache or source.with_name(source.stem + ".expert-row256-cache.gguf")).resolve()
    if output in {source, cache}:
        raise ValueError("quantized output must be separate from the original and padded BF16 cache")
    if (package_source is None) != (encoder is None) or (package_source is None and model_spec is not None):
        raise ValueError("--package-source and --encoder must be supplied together")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".quantizing")
    receipt = output.with_name(output.name + ".quantization.json")
    rules = output.with_name(output.name + ".tensor-types.txt")
    dry_log = output.with_name(output.name + ".dry-run.log")
    log = output.with_name(output.name + ".quantizer.log")
    with claim(output.with_name(output.name + ".lock")):
        if any(path.exists() for path in (output, temporary, receipt, rules, dry_log, log)):
            raise FileExistsError("output or uninspected conversion artifacts already exist")
        before = stat_identity(source)
        with mapped_reader(source) as reader:
            plans = source_plan(reader, source)
            package = None if package_source is None else package_payload(package_source.resolve(), encoder.resolve(),
                model_spec or Path(__file__).resolve().parents[2] / "model_specs/nemotron_transcribe.json", reader)
        with rules.open("x", encoding="utf-8") as stream:
            for plan in plans:
                stream.write(f"^{re.escape(plan.name)}$={expected_type(plan, profile).name.lower()}\n")
        embedding_type = next(p.qtype for p in plans if p.name == "token_embd.weight")
        # A dry run reads headers only. Reject a wrong/unavailable quantizer
        # before streaming a full-size padded intermediate.
        commit = check_quantizer(quantizer, source, temporary, rules, profile, threads, embedding_type, dry_log)
        if stat_identity(source) != before:
            raise ValueError("BF16 source changed during quantizer validation")
        cached_plans, cache_result = padded_cache(source, cache, package)
        if plans != cached_plans or stat_identity(source) != before:
            raise ValueError("BF16 source changed during cache preparation")
        command = quantizer_command(quantizer, cache, temporary, rules, profile, threads, embedding_type)
        print(f"quantizing {profile.upper()}; progress log: {log.name}", flush=True)
        with log.open("xb") as stream:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
        if not temporary.is_file() or not temporary.stat().st_size:
            raise RuntimeError("quantizer produced no output; artifacts retained")
        result = audit_output(cache, temporary, plans, profile)
        if package is not None:
            sibling = ensure_encoder_sibling(encoder.resolve(), output.parent, package)
            result["embedded_package"] = {"family": package["family"], "encoder": sibling.name,
                "files": package["names"], "fingerprint": package["fingerprint"]}
        result.update({"source": source.name, "output": output.name, "padded_cache": cache.name,
            "cache": cache_result, "reported_quantizer_commit": commit, "pinned_llama_revision": PINNED_LLAMA_REVISION,
            "threads": threads, "output_bytes": temporary.stat().st_size})
        publish_no_overwrite(temporary, output)
        with receipt.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2)
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bf16-gguf", type=Path, required=True)
    parser.add_argument("--quantizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=tuple(PROFILES), required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--padded-cache", type=Path, help="Shared profile-independent cache; defaults beside the BF16 source")
    parser.add_argument("--package-source", type=Path, help="Optionally embed validated checkpoint sidecars while writing the cache")
    parser.add_argument("--encoder", type=Path, help="Paired native encoder for the optional self-contained package")
    parser.add_argument("--model-spec", type=Path, help="Frozen v1 spec for the optional self-contained package")
    args = parser.parse_args()
    result = convert(args.source_bf16_gguf, args.quantizer, args.output, args.profile, args.threads, args.padded_cache,
        args.package_source, args.encoder, args.model_spec)
    print(json.dumps({key: value for key, value in result.items() if key != "tensors"}, indent=2))


if __name__ == "__main__":
    main()
