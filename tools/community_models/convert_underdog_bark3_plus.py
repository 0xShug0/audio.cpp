#!/usr/bin/env python3
"""Convert pinned Bark 3.0 Plus weights for the native Parakeet-TDT runtime.

Decode the official bark-levels-v3 data without executing downloaded Python.
The entropy/level reconstruction follows Conway Research's CC-BY-4.0
bark_loader.py; see the retained NOTICE. Requires numpy, constriction==0.5.0
and safetensors. Runtime inference has no Python/ANS dependency.
"""
import argparse
import hashlib
import json
import lzma
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from dataclasses import dataclass

import numpy as np
from convert_phonon2 import (compensate_batch_norm, compensate_projection,
                            lossless_f16_tensors, mixed_q8_tensors, read_exact, sha256)
from convert_underdog_bark3 import package_spec as pro_package_spec, validate_config

SOURCE_REPO = "ConwayResearch/Underdog-Bark-3.0-Plus"
SOURCE_REVISION = "d4fab2f8b95681e5ad3cfb4e8c5f4e0c009b3deb"
SOURCE_SHA256 = "fdd9f1052e4fdead07d71e8cd2138813eb0c209fd6ff8b1b6f47737e1e501857"
BASE_REPO = "nvidia/parakeet-tdt-0.6b-v3"
BASE_REVISION = "541d1f99c6b0c3cd0b11a95167540bb8edefd82b"
VARIANT = "underdog_bark3_plus"
SIDECARS = ("processor_config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json")
BASE_SHA256 = {
    "config.json": "e747b85e1bdfd300c8b8ac63bac8dd5221f8fe9bc275b48d06c735fcd6971b6e",
    "processor_config.json": "8346a93a3b987fa1dec57a78f045cd0817d21786589a5a096b41a57a446fd1d7",
    "tokenizer.json": "bd321b096832a3f270bd3b2a88823957920f1a5c5ada71114a26ea729d0cbe91",
    "tokenizer_config.json": "0b2fe0037599ee335f0b972fa682bf0ece74e4ccfec755cb7daa3405d3d3e874",
    "generation_config.json": "b141de6ec6d7f982ece13f98f604e3fe1807ea9c0e839185d0ab7064604209d0",
}


@dataclass(frozen=True)
class BarkPackedProfile:
    source_repo: str
    source_revision: str
    source_sha256: str
    variant: str
    display_name: str
    docs: str
    bf16_f32_prefixes: tuple[str, ...] = ()


PLUS_PROFILE = BarkPackedProfile(SOURCE_REPO, SOURCE_REVISION, SOURCE_SHA256,
                                VARIANT, "Underdog Bark 3.0 Plus", "docs/models/underdog_bark3.md")


def unpack_blob(data, compressed, size):
    if compressed:
        decoder = lzma.LZMADecompressor()
        data = decoder.decompress(data, max_length=size + 1)
        if not decoder.eof or decoder.unused_data:
            raise ValueError("invalid or oversized LZMA side blob")
    if len(data) != size:
        raise ValueError("incorrect side blob size")
    return data


def decode_record(entry, blobs):
    """Return the same values/dtypes as the official NumPy unpacker."""
    import constriction

    shape = entry["shape"]
    if not isinstance(shape, list) or len(shape) > 4 or any(type(n) is not int or n <= 0 for n in shape):
        raise ValueError("invalid tensor shape")
    count = math.prod(shape)
    if count > 64 * 1024 * 1024:
        raise ValueError("tensor exceeds the supported architecture")
    kind = entry["k"]
    if kind == "raw":
        dtype = {"torch.int64": "<i8", "torch.int32": "<i4", "torch.float32": "<f2"}.get(entry["dtype"])
        if dtype is None:
            raise ValueError("unsupported raw dtype")
        data = unpack_blob(blobs["b"], entry.get("zr", False), count * np.dtype(dtype).itemsize)
        return np.frombuffer(data, dtype=dtype).copy().reshape(shape)
    if len(shape) < 2 or len(blobs["b"]) % 4:
        raise ValueError("invalid ANS tensor")
    rows, cols = shape[0], count // shape[0]
    decoder = constriction.stream.stack.AnsCoder(np.frombuffer(blobs["b"], dtype="<u4").copy())
    if kind == "levels2":
        levels_count = entry["K"]
        if type(levels_count) is not int or not 1 <= levels_count <= 127:
            raise ValueError("invalid level count")
        levels = np.frombuffer(unpack_blob(blobs["lev"], entry.get("zl", False), rows * levels_count * 2),
                               dtype="<f2").reshape(rows, levels_count).astype(np.float32)
        packed_size = (cols * 4 + 7) // 8
        if len(blobs["meta"]) != packed_size + 16 * (2 * levels_count + 1) * 4:
            raise ValueError("invalid level metadata")
        bits = np.unpackbits(np.frombuffer(blobs["meta"][:packed_size], np.uint8))[:cols * 4].reshape(cols, 4)
        classes = np.packbits(np.concatenate([np.zeros((cols, 4), np.uint8), bits], 1), axis=1)[:, 0]
        counts = np.frombuffer(blobs["meta"][packed_size:], dtype="<u4").reshape(16, 2 * levels_count + 1).astype(np.float64)
        codes = np.empty((rows, cols), np.uint8)
        for group in range(16):
            if counts[group].sum() <= 0:
                raise ValueError("empty ANS symbol model")
            columns = np.nonzero(classes == group)[0]
            model = constriction.stream.model.Categorical(counts[group] / counts[group].sum(), perfect=False)
            codes[:, columns] = decoder.decode(model, rows * len(columns)).reshape(len(columns), rows).T
        magnitude = np.where(codes == 0, 0, np.where(codes <= levels_count, codes, codes - levels_count)).astype(np.int64)
        values = np.take_along_axis(np.concatenate([np.zeros((rows, 1), np.float32), levels], 1), magnitude, 1)
        return np.where(codes > levels_count, -values, values).reshape(shape)
    if kind not in ("ansint5", "ansint6"):
        raise ValueError("unsupported ANS record kind")
    bits = int(kind[-1])
    scales = np.frombuffer(unpack_blob(blobs["s"], entry.get("zs", False), rows * 2), dtype="<f2").astype(np.float32)
    if len(blobs["meta"]) != (2 ** bits - 1) * 4:
        raise ValueError("invalid integer symbol metadata")
    counts = np.frombuffer(blobs["meta"], dtype="<u4").astype(np.float64)
    if counts.sum() <= 0:
        raise ValueError("empty ANS symbol model")
    model = constriction.stream.model.Categorical(counts / counts.sum(), perfect=False)
    codes = decoder.decode(model, count)
    signed = (codes - (2 ** (bits - 1) - 1)).astype(np.float32)
    return (signed.reshape(rows, cols) * scales[:, None]).reshape(shape)


def read_container(path):
    tensors = {}
    with path.open("rb") as stream:
        size = int.from_bytes(read_exact(stream, 8), "little")
        if not 0 < size <= min(4 * 1024 * 1024, path.stat().st_size - 8):
            raise ValueError("invalid Bark header size")
        header = json.loads(read_exact(stream, size))
        if header.get("format") != "bark-levels-v3" or not isinstance(header.get("index"), list):
            raise ValueError("expected bark-levels-v3")
        for entry in header["index"]:
            name = entry["n"]
            if not isinstance(name, str) or name in tensors:
                raise ValueError("invalid or duplicate tensor name")
            fields = ("lev", "meta", "b") if entry["k"] == "levels2" else ("b",) if entry["k"] == "raw" else ("s", "meta", "b")
            blobs = {}
            for key in fields:
                length = entry[key]
                if type(length) is not int or not 0 <= length <= 256 * 1024 * 1024:
                    raise ValueError("invalid record size")
                blobs[key] = read_exact(stream, length)
            tensors[name] = decode_record(entry, blobs)
        if stream.read(1):
            raise ValueError("trailing Bark container bytes")
    return tensors, json.loads(header["config"])


def package_spec(repository, profile=PLUS_PROFILE):
    spec = pro_package_spec(repository)
    spec.update(display_name=profile.display_name,
                description=f"{profile.display_name}: English Parakeet-TDT ASR; native expansion of the pinned compressed checkpoint, with the reference 250 ms prefix.")
    spec["ui"]["docs"] = [profile.docs]
    return spec


def prepare(source, staging, profile=PLUS_PROFILE):
    from safetensors.numpy import save_file

    if staging.exists():
        raise FileExistsError("use a new staging directory or --pack-only")
    if sha256(source / "model.bark") != profile.source_sha256:
        raise ValueError("Bark container differs from the pinned revision")
    for name, expected in BASE_SHA256.items():
        if sha256(source / name) != expected:
            raise ValueError("base sidecar differs from the pinned revision: " + name)
    tensors, config = read_container(source / "model.bark")
    validate_config(config)
    if len(tensors) != 723:
        raise ValueError("unexpected pinned tensor inventory")
    # Verify the processor/tokenizer against the source's NeMo vocabulary.
    from convert_phonon2 import validate_sidecars
    validate_sidecars(source, json.loads((source / "mlx/config.json").read_text(encoding="utf-8")))
    # The runtime config comes from the container, not from the base checkpoint.
    base = json.loads((source / "config.json").read_text(encoding="utf-8"))
    for key in ("vocab_size", "blank_token_id", "decoder_hidden_size", "num_decoder_layers", "durations"):
        if config[key] != base[key]:
            raise ValueError("base sidecars and Bark container disagree: " + key)
    manifest = []
    counters = 0
    for name in list(tensors):
        if name.endswith(".num_batches_tracked"):
            del tensors[name]
            counters += 1
            continue
        values = np.ascontiguousarray(tensors[name], dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError("non-finite tensor: " + name)
        source_hash = hashlib.sha256(values.tobytes()).hexdigest()
        values = compensate_projection(name, values)
        values = compensate_batch_norm(name, values, tensors)
        tensors[name] = np.ascontiguousarray(values)
        manifest.append(dict(name=name, shape=list(values.shape), source_f32_sha256=source_hash,
                             staged_f32_sha256=hashlib.sha256(values.tobytes()).hexdigest()))
    if len(tensors) != 699 or counters != 24:
        raise ValueError("unexpected inference/training tensor inventory")
    staging.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="bark-packed-", dir=staging.parent) as temporary:
        target = Path(temporary) / "package"
        target.mkdir()
        config["variant"] = profile.variant
        (target / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        for name in (*SIDECARS, "NOTICE", "README.md"):
            shutil.copyfile(source / name, target / name)
        shutil.copyfile(source / "coreml/LICENSE", target / "LICENSE")
        provenance = dict(source=profile.source_repo, revision=profile.source_revision, model_sha256=profile.source_sha256,
                          variant=profile.variant, base_repo=BASE_REPO, base_revision=BASE_REVISION,
                          source_bytes=(source / "model.bark").stat().st_size,
                          input_prefix_samples=4000, sample_rate=16000,
                          subsampling_projection_scale=1 / 32,
                          batch_norm_gamma_correction="sqrt((max(var,1e-5)+1e-5)/(var+1e-5))",
                          dropped_training_counters=counters,
                          sidecar_sha256={name: sha256(source / name) for name in SIDECARS})
        (target / "conversion.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
        (target / "tensor_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        repository = Path(__file__).resolve().parents[2]
        (target / "model_spec.json").write_text(json.dumps(package_spec(repository, profile), indent=2), encoding="utf-8")
        save_file(tensors, target / "model.safetensors")
        target.rename(staging)


def pack(staging, tool, output, storage, overwrite=False, profile=PLUS_PROFILE):
    provenance = json.loads((staging / "conversion.json").read_text(encoding="utf-8"))
    if (provenance.get("source"), provenance.get("revision"), provenance.get("model_sha256"), provenance.get("variant")) != (
            profile.source_repo, profile.source_revision, profile.source_sha256, profile.variant):
        raise ValueError("staging belongs to a different source/variant")
    preserved, half = [], []
    if storage == "mixed-f16":
        preserved, _ = lossless_f16_tensors(staging)
    elif storage in ("q8_0", "mixed-q8"):
        preserved, half = mixed_q8_tensors(staging)
    elif storage == "bf16" and profile.bf16_f32_prefixes:
        preserved = [row["name"] for row in json.loads((staging / "tensor_manifest.json").read_text(encoding="utf-8"))
                     if row["name"].startswith(profile.bf16_f32_prefixes)]
    provenance.update(storage=storage, exact_staged_values=storage in ("f32", "mixed-f16"),
                      preserved_f32_tensors=preserved, preserved_f16_tensors=half)
    with tempfile.TemporaryDirectory(prefix="bark-packed-pack-", dir=staging.parent) as temporary:
        root = Path(temporary)
        for name in ("config.json", *SIDECARS, "NOTICE", "LICENSE", "README.md", "model_spec.json", "tensor_manifest.json"):
            shutil.copyfile(staging / name, root / name)
        (root / "conversion.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
        target_type = "f16" if storage == "mixed-f16" else "q8_0" if storage == "mixed-q8" else storage
        output.parent.mkdir(parents=True, exist_ok=True)
        command = [str(tool.resolve()), "--input", str((staging / "model.safetensors").resolve()),
                   "--root", str(root.resolve()), "--family", "parakeet_tdt", "--model-spec", str(root / "model_spec.json"),
                   "--type", target_type, "--keep-type", "*conv.norm.weight=f32", "--output", str(output.resolve())]
        for name in preserved:
            command += ["--keep-type", name + "=orig"]
        for name in half:
            command += ["--keep-type", name + "=f16"]
        if storage == "bf16":
            command += ["--keep-type", "encoder.subsampling.linear.*=orig"]
        if overwrite:
            command.append("--overwrite")
        for name in ("config.json", *SIDECARS, "NOTICE", "LICENSE", "README.md", "conversion.json", "tensor_manifest.json"):
            command += ["--sidecar", f"{root / name}={name}"]
        subprocess.run(command, check=True, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)


def main(profile=PLUS_PROFILE, description=__doc__):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--staging", required=True, type=Path)
    parser.add_argument("--gguf-tool", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--type", choices=["f32", "mixed-f16", "bf16", "q8_0", "mixed-q8"], default="mixed-f16")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--pack-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.prepare_only and args.pack_only:
        parser.error("prepare-only and pack-only are mutually exclusive")
    if not args.pack_only:
        prepare(args.source.resolve(), args.staging.resolve(), profile)
    if args.prepare_only:
        return
    if not args.gguf_tool or not args.output:
        parser.error("packing requires --gguf-tool and --output")
    pack(args.staging.resolve(), args.gguf_tool, args.output, args.type, args.overwrite, profile)


if __name__ == "__main__":
    main()
